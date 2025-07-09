
import asyncio, json, logging, time, fractions, collections, warnings
from concurrent.futures import ThreadPoolExecutor
import numpy as np, librosa, torch, av
from aiohttp import web, WSMsgType
from aiortc import (
    RTCPeerConnection, RTCSessionDescription, RTCIceCandidate,
    MediaStreamTrack, RTCConfiguration, RTCIceServer
)
from transformers import pipeline
from chatterbox.tts import ChatterboxTTS

# ------------------ Environment ------------------
try:
    import uvloop; uvloop.install()
except ImportError:
    pass

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)
log.getChild("aioice.ice").setLevel(logging.WARNING)
log.getChild("aiortc").setLevel(logging.WARNING)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
EXEC     = ThreadPoolExecutor(max_workers=8)
PCS      = set()

# ---------------- HTML Client --------------------
HTML = open(__file__).read().split("HTML_CLIENT =")[1].split('"""',2)[1]

# -------------------- VAD ------------------------
class SileroVAD:
    def __init__(self):
        self.model, util = torch.hub.load(
            "snakers4/silero-vad", "silero_vad", onnx=False
        )
        (self.get_ts, *_ ) = util

    def is_speech(self, wav, sr=16_000):
        if isinstance(wav, np.ndarray):
            wav = torch.from_numpy(wav.copy())
        if wav.abs().max() < 0.01:          # silence floor
            return False
        ts = self.get_ts(wav, self.model, sampling_rate=sr,
                         min_speech_duration_ms=300,
                         min_silence_duration_ms=100)
        return bool(ts)

# ----------------- Audio Buffer ------------------
class RingBuffer:
    def __init__(self, secs=6, sr=16_000):
        self.max = int(secs * sr)
        self.buf = collections.deque(maxlen=self.max)

    def push(self, x: np.ndarray):
        self.buf.extend(x.astype(np.float32))

    def dump(self) -> np.ndarray:
        return np.frombuffer(
            memoryview(np.array(self.buf, dtype=np.float32)), dtype=np.float32
        )

    def clear(self): self.buf.clear()
    def __len__(self): return len(self.buf)

# ------------- WebRTC Outgoing Track -------------
class OutAudio(MediaStreamTrack):
    kind = "audio"

    def __init__(self):
        super().__init__()
        self.q = asyncio.Queue(maxsize=10)  # 10×20 ms = 200 ms
        self.ts  = 0
        self.tb  = fractions.Fraction(1, 48_000)

    async def queue_pcm(self, pcm: np.ndarray):
        """
        Split PCM (float32 −1…1, 48 kHz mono) into 20 ms chunks and enqueue.
        Drops oldest chunk if queue full.
        """
        pcm16 = np.clip(pcm * 32767, -32768, 32767).astype(np.int16)
        step  = 960  # 20 ms
        for i in range(0, len(pcm16), step):
            chunk = pcm16[i:i + step]
            if self.q.full():
                try: self.q.get_nowait()
                except asyncio.QueueEmpty: pass
            await self.q.put(chunk)
            await asyncio.sleep(0)         # yield control

    async def recv(self):
        chunk = await self.q.get()
        f = av.AudioFrame.from_ndarray(
            np.expand_dims(chunk, 0), format="s16", layout="mono"
        )
        f.pts, f.sample_rate, f.time_base = self.ts, 48_000, self.tb
        self.ts += len(chunk)
        return f

# -------------- Audio Processor ------------------
class Processor:
    def __init__(self, track_in: MediaStreamTrack, track_out: OutAudio,
                 vad: SileroVAD):
        self.in_track  = track_in
        self.out_track = track_out
        self.vad       = vad
        self.buf       = RingBuffer()
        self._task     = None
        self.speaking  = asyncio.Event()   # set while TTS audio being sent

    async def start(self): self._task = asyncio.create_task(self._loop())
    async def stop(self):  self._task.cancel()

    async def _loop(self):
        while True:
            frame = await self.in_track.recv()
            pcm   = frame.to_ndarray().flatten().astype(np.float32) / 32768.0
            if frame.sample_rate != 16_000:
                pcm = librosa.resample(pcm, frame.sample_rate, 16_000)
            self.buf.push(pcm)

            # Do not ASR while bot is speaking
            if self.speaking.is_set(): continue
            if len(self.buf) < 16_000:     # need ≥1 s
                continue
            audio_chunk = self.buf.dump(); self.buf.clear()
            if not self.vad.is_speech(audio_chunk): continue
            asyncio.create_task(self._asr_tts(audio_chunk))

    async def _asr_tts(self, pcm16k: np.ndarray):
        text = await asyncio.get_running_loop().run_in_executor(
            EXEC, lambda: Ultravox(pcm16k)
        )
        if not text: return
        log.info("🗣 Ultravox → %s", text)
        # TTS generation in thread
        pcm48 = await asyncio.get_running_loop().run_in_executor(
            EXEC, lambda: TTS(text)
        )
        if pcm48.size == 0: return
        self.speaking.set()
        await self.out_track.queue_pcm(pcm48)
        self.speaking.clear()

# ------------------ Model Wrappers --------------
def Ultravox(pcm16k: np.ndarray) -> str:
    out = uv_pipe({"audio": pcm16k, "turns": [], "sampling_rate": 16_000},
                  max_new_tokens=300, do_sample=True,
                  temperature=0.7, top_p=0.9)
    return parse_ultravox_response(out)

def TTS(text: str) -> np.ndarray:
    wav24 = tts_model.generate(text).cpu().numpy().flatten()
    return librosa.resample(wav24.astype(np.float32), 24_000, 48_000)

# ------------------ Signalling ------------------
async def websocket(request: web.Request):
    ws = web.WebSocketResponse(heartbeat=20)
    await ws.prepare(request)

    pc = RTCPeerConnection(RTCConfiguration([
        RTCIceServer(urls="stun:stun.l.google.com:19302")
    ]))
    PCS.add(pc)

    # Heart-beat task
    async def _ping():
        while True:
            await asyncio.sleep(5)
            if ws.closed: break
            try: await ws.send_str("ping")
            except Exception: break
    asyncio.create_task(_ping())

    @pc.on("track")
    def on_track(tr: MediaStreamTrack):
        out = OutAudio(); pc.addTrack(out)
        proc = Processor(tr, out, vad_model)
        asyncio.create_task(proc.start())

    try:
        async for msg in ws:
            if msg.type is WSMsgType.TEXT:
                data = json.loads(msg.data)
                if data["type"] == "offer":
                    await pc.setRemoteDescription(
                        RTCSessionDescription(**data)
                    )
                    answer = await pc.createAnswer()
                    await pc.setLocalDescription(answer)
                    await ws.send_json({
                        "type": "answer", "sdp": pc.localDescription.sdp
                    })
                elif data["type"] == "ice-candidate" and data["candidate"]:
                    c = data["candidate"]
                    cand = RTCIceCandidate(
                        sdpMid=c["sdpMid"], sdpMLineIndex=c["sdpMLineIndex"],
                        **candidate_from_sdp(c["candidate"])
                    )
                    await pc.addIceCandidate(cand)
            elif msg.type in (WSMsgType.CLOSE, WSMsgType.ERROR):
                break
    finally:
        await pc.close(); PCS.discard(pc)
    return ws

async def index(_): return web.Response(text=HTML, content_type="text/html")

# -------------------- Main ----------------------
async def main():
    global uv_pipe, tts_model, vad_model
    vad_model = SileroVAD()
    uv_pipe   = pipeline("fixie-ai/ultravox-v0_4", trust_remote_code=True,
                         device_map="auto", torch_dtype=torch.float16,
                         attn_implementation="eager")
    tts_model = ChatterboxTTS.from_pretrained(device=DEVICE)

    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/ws", websocket)
    app.on_shutdown.append(lambda _: [pc.close() for pc in list(PCS)])

    runner = web.AppRunner(app); await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 7860); await site.start()
    log.info("Server live on :7860"); await asyncio.Event().wait()

if __name__ == "__main__":
    try: asyncio.run(main())
    except KeyboardInterrupt: pass
