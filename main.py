import torch
import asyncio
import json
import logging
import numpy as np
import fractions
import warnings
import collections
import time
import librosa
from concurrent.futures import ThreadPoolExecutor
import threading

from aiohttp import web, WSMsgType
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack, RTCIceCandidate, RTCConfiguration, RTCIceServer, mediastreams
import av

from transformers import pipeline
from chatterbox.tts import ChatterboxTTS
import torch.hub

# --- Enhanced Error Handling Setup ---
try:
    import uvloop
    uvloop.install()
    print("🚀 Using uvloop for asyncio event loop.")
except ImportError:
    print("⚠️ uvloop not found, using default asyncio event loop.")

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger('aioice.ice').setLevel(logging.WARNING)
logging.getLogger('aiortc').setLevel(logging.WARNING)

# --- Global Variables ---
uv_pipe, tts_model, vad_model = None, None, None
executor = ThreadPoolExecutor(max_workers=4)
pcs = set()

# --- HTML Client (Same as before) ---
HTML_CLIENT = """
<!DOCTYPE html>
<html>
<head>
    <title>UltraChat Voice Assistant</title>
    <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #2c3e50; color: #ecf0f1; display: flex; align-items: center; justify-content: center; min-height: 100vh; }
        .container { background: #34495e; padding: 40px; border-radius: 10px; box-shadow: 0 10px 20px rgba(0,0,0,0.2); text-align: center; max-width: 600px; width: 100%; }
        h1 { margin-bottom: 30px; font-weight: 300; }
        button { background: #2ecc71; color: white; border: none; padding: 15px 30px; font-size: 18px; border-radius: 5px; cursor: pointer; margin: 10px; transition: all 0.3s; }
        button:hover { transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.2); }
        button:disabled { background: #95a5a6; cursor: not-allowed; transform: none; box-shadow: none; }
        .stop-btn { background: #e74c3c; } .stop-btn:hover { background: #c0392b; }
        .status { margin: 20px 0; padding: 15px; border-radius: 5px; font-weight: 500; transition: background-color 0.5s; }
        .status.connected { background: #27ae60; } .status.disconnected { background: #c0392b; } .status.connecting { background: #f39c12; }
        .status.speaking { background: #3498db; animation: pulse 2s infinite; }
        @keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(52, 152, 219, 0.7); } 70% { box-shadow: 0 0 0 10px rgba(52, 152, 219, 0); } 100% { box-shadow: 0 0 0 0 rgba(52, 152, 219, 0); } }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎙️ UltraChat Voice Assistant</h1>
        <div class="controls">
            <button id="startBtn" onclick="start()">START</button>
            <button id="stopBtn" onclick="stop()" class="stop-btn" disabled>STOP</button>
        </div>
        <div id="status" class="status disconnected">🔌 Disconnected</div>
        <audio id="remoteAudio" autoplay playsinline></audio>
    </div>
<script>
    let pc, ws, localStream;
    const remoteAudio = document.getElementById('remoteAudio');
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    const statusDiv = document.getElementById('status');

    function updateStatus(message, className) { statusDiv.textContent = message; statusDiv.className = `status ${className}`; }

    async function start() {
        startBtn.disabled = true;
        updateStatus('🔄 Connecting...', 'connecting');
        
        try {
            // Enhanced audio context setup
            const audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: 48000,
                latencyHint: 'interactive'
            });
            if (audioContext.state === 'suspended') { await audioContext.resume(); }

            // Enhanced getUserMedia constraints
            localStream = await navigator.mediaDevices.getUserMedia({ 
                audio: { 
                    echoCancellation: true, 
                    noiseSuppression: true,
                    autoGainControl: false,
                    sampleRate: 48000,
                    sampleSize: 16,
                    channelCount: 1
                } 
            });

            pc = new RTCPeerConnection({ 
                iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
                rtcpMuxPolicy: 'require',
                bundlePolicy: 'max-bundle'
            });
            
            localStream.getTracks().forEach(track => pc.addTrack(track, localStream));

            pc.ontrack = e => {
                console.log('Remote track received!');
                if (remoteAudio.srcObject !== e.streams[0]) {
                    remoteAudio.srcObject = e.streams[0];
                    remoteAudio.play().catch(err => console.error("Autoplay failed:", err));
                    
                    remoteAudio.onplaying = () => {
                        updateStatus('🤖 AI Speaking...', 'speaking');
                    };
                    remoteAudio.onended = () => {
                         if(pc.connectionState === 'connected') updateStatus('✅ Listening...', 'connected');
                    };
                }
            };

            pc.onicecandidate = e => {
                if (e.candidate && ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({ type: 'ice-candidate', candidate: e.candidate.toJSON() }));
                }
            };
            
            pc.onconnectionstatechange = () => {
                const state = pc.connectionState;
                console.log(`Connection state: ${state}`);
                if (state === 'connecting') updateStatus('🤝 Establishing secure connection...', 'connecting');
                else if (state === 'connected') { updateStatus('✅ Listening...', 'connected'); stopBtn.disabled = false; }
                else if (state === 'failed' || state === 'closed' || state === 'disconnected') stop();
            };

            const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${location.host}/ws`);

            ws.onopen = async () => {
                const offer = await pc.createOffer();
                await pc.setLocalDescription(offer);
                ws.send(JSON.stringify(offer));
            };

            ws.onmessage = async e => {
                const data = JSON.parse(e.data);
                if (data.type === 'answer' && !pc.currentRemoteDescription) {
                    await pc.setRemoteDescription(new RTCSessionDescription(data));
                }
            };

            const closeHandler = () => { if (pc && pc.connectionState !== 'closed') stop(); };
            ws.onclose = closeHandler;
            ws.onerror = closeHandler;

        } catch (err) { 
            console.error('Error during start:', err); 
            updateStatus(`❌ Error: ${err.message}`, 'disconnected'); 
            stop(); 
        }
    }

    function stop() {
        if (ws) { ws.onclose = null; ws.onerror = null; ws.close(); ws = null; }
        if (pc) { pc.onconnectionstatechange = null; pc.onicecandidate = null; pc.ontrack = null; pc.close(); pc = null; }
        if (localStream) { localStream.getTracks().forEach(track => track.stop()); localStream = null; }
        updateStatus('🔌 Disconnected', 'disconnected');
        startBtn.disabled = false;
        stopBtn.disabled = true;
    }
</script>
</body>
</html>
"""

# --- Enhanced Utility Functions ---
def candidate_from_sdp(candidate_string: str) -> dict:
    if candidate_string.startswith("candidate:"): candidate_string = candidate_string[10:]
    bits = candidate_string.split()
    if len(bits) < 8: raise ValueError(f"Invalid candidate string: {candidate_string}")
    params = {'component': int(bits[1]), 'foundation': bits[0], 'ip': bits[4], 'port': int(bits[5]), 'priority': int(bits[3]), 'protocol': bits[2], 'type': bits[7]}
    for i in range(8, len(bits) - 1, 2):
        if bits[i] == "raddr": params['relatedAddress'] = bits[i + 1]
        elif bits[i] == "rport": params['relatedPort'] = int(bits[i + 1])
    return params

def parse_ultravox_response(result):
    """Enhanced Ultravox response parsing"""
    try:
        if isinstance(result, str): 
            return result.strip()
        elif isinstance(result, list) and len(result) > 0:
            item = result[0]
            if isinstance(item, str): 
                return item.strip()
            elif isinstance(item, dict):
                if 'generated_text' in item: 
                    return item['generated_text'].strip()
                elif 'text' in item:
                    return item['text'].strip()
        return ""
    except Exception as e:
        logger.error(f"Error parsing Ultravox response: {e}")
        return ""

# --- Enhanced VAD Implementation ---
class SileroVAD:
    def __init__(self):
        try:
            logger.info("🎤 Loading Silero VAD model...")
            # Enhanced VAD model loading with proper configuration
            self.model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad', 
                model='silero_vad', 
                force_reload=False, 
                onnx=False,
                trust_repo=True
            )
            (self.get_speech_timestamps, _, _, _, _) = utils
            
            # Set model to evaluation mode and optimize for inference
            self.model.eval()
            torch.set_num_threads(1)  # Optimize for single-threaded inference
            
            logger.info("✅ Silero VAD loaded successfully")
        except Exception as e:
            logger.error(f"❌ Error loading Silero VAD: {e}", exc_info=True)
            self.model = None

    def detect_speech(self, audio_tensor, sample_rate=16000):
        """Enhanced speech detection with better thresholds"""
        if self.model is None: 
            return True
            
        try:
            # Convert to proper tensor format
            if isinstance(audio_tensor, np.ndarray): 
                audio_tensor = torch.from_numpy(audio_tensor).float()
            
            # Ensure audio is in the right format
            if audio_tensor.dim() > 1:
                audio_tensor = audio_tensor.squeeze()
            
            # Check for minimum audio energy
            if audio_tensor.abs().max() < 0.01: 
                return False
            
            # Enhanced VAD detection with proper parameters
            speech_timestamps = self.get_speech_timestamps(
                audio_tensor, 
                self.model, 
                sampling_rate=sample_rate,
                min_speech_duration_ms=100,  # Reduced from 250ms
                min_silence_duration_ms=300, # Reduced from default
                speech_pad_ms=50
            )
            
            return len(speech_timestamps) > 0
            
        except Exception as e:
            logger.error(f"VAD detection error: {e}")
            return True  # Default to processing if VAD fails

# --- Enhanced Model Initialization ---
def initialize_models():
    global uv_pipe, tts_model, vad_model
    
    try:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"🚀 Initializing models on device: {device}")
        
        # Initialize VAD first
        vad_model = SileroVAD()
        if not vad_model.model: 
            logger.error("❌ VAD model failed to load")
            return False
        
        # Initialize Ultravox with enhanced configuration
        logger.info("📥 Loading Ultravox pipeline...")
        uv_pipe = pipeline(
            model="fixie-ai/ultravox-v0_4", 
            trust_remote_code=True, 
            device_map="auto" if device == "cuda:0" else None,
            torch_dtype=torch.float16 if device == "cuda:0" else torch.float32,
            model_kwargs={"attn_implementation": "flash_attention_2"} if device == "cuda:0" else {}
        )
        logger.info("✅ Ultravox pipeline loaded successfully")

        # Initialize TTS with enhanced configuration
        logger.info("📥 Loading Chatterbox TTS...")
        tts_model = ChatterboxTTS.from_pretrained(device=device)
        # Set TTS to evaluation mode for consistent output
        tts_model.eval()
        logger.info("✅ Chatterbox TTS loaded successfully")
        
        logger.info("🎉 All models loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Critical model loading error: {e}", exc_info=True)
        return False

# --- Enhanced Audio Buffer ---
class AudioBuffer:
    def __init__(self, max_duration=4.0, sample_rate=16000):  # Increased buffer size
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)
        self.buffer = collections.deque(maxlen=self.max_samples)
        self.last_process_time = time.time()
        self.min_speech_samples = int(0.3 * sample_rate)  # Reduced minimum
        self.process_interval = 0.3  # Reduced interval for faster response
        self.silence_threshold = 0.008  # Adjusted threshold
    
    def add_audio(self, audio_data):
        """Enhanced audio addition with proper normalization"""
        try:
            # Convert to float32 if needed
            if audio_data.dtype != np.float32:
                if audio_data.dtype == np.int16:
                    audio_data = audio_data.astype(np.float32) / 32768.0
                else:
                    audio_data = audio_data.astype(np.float32)
            
            # Proper normalization
            if np.abs(audio_data).max() > 1.0:
                audio_data = audio_data / np.abs(audio_data).max()
            
            # Add to buffer
            self.buffer.extend(audio_data.flatten())
            
        except Exception as e:
            logger.error(f"Error adding audio to buffer: {e}")

    def get_audio_array(self):
        """Get audio array with proper format"""
        return np.array(list(self.buffer), dtype=np.float32)
    
    def should_process(self):
        """Enhanced processing decision logic"""
        current_time = time.time()
        
        # Check timing and buffer size
        if (len(self.buffer) > self.min_speech_samples and 
            (current_time - self.last_process_time) > self.process_interval):
            
            self.last_process_time = current_time
            audio_array = self.get_audio_array()
            
            # Check for sufficient audio energy
            if np.abs(audio_array).max() < self.silence_threshold:
                return False
                
            # Use VAD for speech detection
            return vad_model.detect_speech(audio_array, self.sample_rate)
        
        return False

    def reset(self):
        """Reset buffer"""
        self.buffer.clear()

# --- Enhanced Response Audio Track ---
class ResponseAudioTrack(MediaStreamTrack):
    kind = "audio"
    
    def __init__(self):
        super().__init__()
        self._queue = asyncio.Queue(maxsize=50)  # Increased queue size
        self._current_chunk = None
        self._chunk_pos = 0
        self._timestamp = 0
        self._time_base = fractions.Fraction(1, 48000)
        self._frame_duration = 960  # 20ms at 48kHz
        self._lock = asyncio.Lock()

    async def recv(self):
        """Enhanced frame generation with better timing"""
        async with self._lock:
            frame_samples = self._frame_duration
            frame = np.zeros(frame_samples, dtype=np.int16)
            
            # Fill frame with audio data
            filled_samples = 0
            while filled_samples < frame_samples:
                if self._current_chunk is None or self._chunk_pos >= len(self._current_chunk):
                    try:
                        self._current_chunk = await asyncio.wait_for(self._queue.get(), timeout=0.02)
                        self._chunk_pos = 0
                    except asyncio.TimeoutError:
                        break
                
                if self._current_chunk is not None:
                    remaining_frame = frame_samples - filled_samples
                    remaining_chunk = len(self._current_chunk) - self._chunk_pos
                    
                    copy_samples = min(remaining_frame, remaining_chunk)
                    frame[filled_samples:filled_samples + copy_samples] = \
                        self._current_chunk[self._chunk_pos:self._chunk_pos + copy_samples]
                    
                    self._chunk_pos += copy_samples
                    filled_samples += copy_samples
            
            # Create audio frame with proper timing
            audio_frame = av.AudioFrame.from_ndarray(
                np.array([frame]), 
                format="s16", 
                layout="mono"
            )
            audio_frame.pts = self._timestamp
            audio_frame.sample_rate = 48000
            audio_frame.time_base = self._time_base
            
            self._timestamp += frame_samples
            return audio_frame

    async def queue_audio(self, audio_float32):
        """Enhanced audio queueing with overflow protection"""
        if audio_float32.size > 0:
            # Convert and clamp audio
            audio_int16 = np.clip(audio_float32 * 32767, -32768, 32767).astype(np.int16)
            
            # Add to queue with overflow protection
            if self._queue.qsize() < 45:  # Leave some headroom
                await self._queue.put(audio_int16)
            else:
                logger.warning("Audio queue full, dropping frame")

# --- Enhanced Audio Processor ---
class AudioProcessor:
    def __init__(self, output_track: ResponseAudioTrack, executor: ThreadPoolExecutor):
        self.track = None
        self.buffer = AudioBuffer()
        self.output_track = output_track
        self.task = None
        self.executor = executor
        self.is_speaking = False
        self.processing_lock = asyncio.Lock()
        self.speech_timeout = 5.0  # Timeout for speech processing

    def add_track(self, track): 
        self.track = track
        
    async def start(self): 
        self.task = asyncio.create_task(self._run())
        
    async def stop(self):
        if self.task: 
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass

    async def _run(self):
        """Enhanced audio processing loop"""
        try:
            while True:
                # Skip processing if AI is speaking
                if self.is_speaking:
                    try:
                        # Drain frames during speaking
                        await asyncio.wait_for(self.track.recv(), timeout=0.01)
                    except asyncio.TimeoutError:
                        await asyncio.sleep(0.01)
                    continue

                try:
                    # Receive audio frame with timeout
                    frame = await asyncio.wait_for(self.track.recv(), timeout=1.0)
                    
                    # Process audio frame
                    audio_float32 = frame.to_ndarray().flatten().astype(np.float32) / 32768.0
                    
                    # Resample to 16kHz for processing
                    if frame.sample_rate != 16000:
                        resampled_audio = librosa.resample(
                            audio_float32, 
                            orig_sr=frame.sample_rate, 
                            target_sr=16000
                        )
                    else:
                        resampled_audio = audio_float32
                    
                    # Add to buffer
                    self.buffer.add_audio(resampled_audio)
                    
                    # Check if we should process
                    if self.buffer.should_process():
                        audio_to_process = self.buffer.get_audio_array()
                        self.buffer.reset()
                        
                        logger.info(f"🧠 Processing {len(audio_to_process)} samples...")
                        
                        # Process speech in background
                        asyncio.create_task(self.process_speech(audio_to_process))
                        
                except asyncio.TimeoutError:
                    # No audio received, continue listening
                    continue
                except mediastreams.MediaStreamError:
                    logger.warning("Client media stream ended.")
                    break
                    
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Audio processor error: {e}", exc_info=True)
        finally:
            logger.info("Audio processor stopped.")

    def _blocking_tts(self, text: str) -> np.ndarray:
        """Enhanced TTS generation with better error handling"""
        try:
            # Ensure text is not empty and within limits
            if not text or len(text.strip()) == 0:
                return np.array([], dtype=np.float32)
            
            # Limit text length to prevent truncation
            if len(text) > 500:
                text = text[:500] + "..."
            
            logger.info(f"🔊 Generating TTS for: '{text[:50]}...'")
            
            with torch.inference_mode():
                # Generate TTS with enhanced parameters
                wav = tts_model.generate(
                    text, 
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                ).cpu().numpy().flatten()
                
                # Ensure audio is not empty
                if wav.size == 0:
                    logger.warning("TTS generated empty audio")
                    return np.array([], dtype=np.float32)
                
                # Resample from 24kHz to 48kHz
                resampled_wav = librosa.resample(
                    wav.astype(np.float32), 
                    orig_sr=24000, 
                    target_sr=48000
                )
                
                # Normalize audio
                if np.abs(resampled_wav).max() > 0:
                    resampled_wav = resampled_wav / np.abs(resampled_wav).max() * 0.8
                
                logger.info(f"✅ TTS generated {len(resampled_wav)} samples")
                return resampled_wav
                
        except Exception as e:
            logger.error(f"TTS generation failed: {e}", exc_info=True)
            return np.array([], dtype=np.float32)

    async def process_speech(self, audio_array):
        """Enhanced speech processing with better error handling"""
        if self.is_speaking:
            return
            
        async with self.processing_lock:
            try:
                # Ensure audio array is in correct format
                if audio_array.size == 0:
                    return
                
                # Prepare input for Ultravox
                turns = [
                    {
                        "role": "system",
                        "content": "You are a helpful AI assistant. Provide concise, natural responses."
                    }
                ]
                
                # Process with Ultravox
                logger.info("🤖 Processing with Ultravox...")
                
                with torch.inference_mode():
                    result = uv_pipe({
                        'audio': audio_array,
                        'turns': turns,
                        'sampling_rate': 16000
                    }, max_new_tokens=100, temperature=0.7)
                
                # Parse response
                response_text = parse_ultravox_response(result)
                
                if not response_text:
                    logger.warning("No response from Ultravox")
                    return
                
                logger.info(f"🎯 AI Response: '{response_text}'")
                
                # Generate TTS
                loop = asyncio.get_running_loop()
                resampled_wav = await loop.run_in_executor(
                    self.executor, 
                    self._blocking_tts, 
                    response_text
                )

                if resampled_wav.size > 0:
                    # Set speaking state
                    self.is_speaking = True
                    logger.info("🤖 AI started speaking...")
                    
                    # Queue audio for playback
                    await self.output_track.queue_audio(resampled_wav)
                    
                    # Calculate playback duration and wait
                    playback_duration = resampled_wav.size / 48000
                    await asyncio.sleep(playback_duration + 0.2)  # Add small buffer
                    
                    # Reset speaking state
                    self.is_speaking = False
                    logger.info("✅ AI finished speaking")
                    
            except Exception as e:
                logger.error(f"Speech processing error: {e}", exc_info=True)
                self.is_speaking = False

# --- Enhanced WebSocket Handler ---
async def websocket_handler(request):
    ws = web.WebSocketResponse(heartbeat=30)
    await ws.prepare(request)
    
    pc = RTCPeerConnection(RTCConfiguration([
        RTCIceServer(urls="stun:stun.l.google.com:19302")
    ]))
    pcs.add(pc)
    processor = None

    @pc.on("track")
    def on_track(track):
        nonlocal processor
        logger.info(f"🎧 Track {track.kind} received")
        if track.kind == "audio":
            output_audio_track = ResponseAudioTrack()
            pc.addTrack(output_audio_track)
            processor = AudioProcessor(output_audio_track, executor)
            processor.add_track(track)
            asyncio.create_task(processor.start())

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"ICE Connection State is {pc.connectionState}")
        if pc.connectionState in ["failed", "closed", "disconnected"]:
            if pc in pcs: 
                pcs.remove(pc)
            await pc.close()

    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                data = json.loads(msg.data)
                
                if data["type"] == "offer":
                    await pc.setRemoteDescription(RTCSessionDescription(
                        sdp=data["sdp"], 
                        type=data["type"]
                    ))
                    answer = await pc.createAnswer()
                    await pc.setLocalDescription(answer)
                    await ws.send_json({
                        "type": "answer", 
                        "sdp": pc.localDescription.sdp
                    })
                    
                elif data["type"] == "ice-candidate" and data.get("candidate"):
                    try:
                        candidate_data = data["candidate"]
                        candidate_string = candidate_data.get("candidate")
                        if candidate_string:
                            params = candidate_from_sdp(candidate_string)
                            candidate = RTCIceCandidate(
                                sdpMid=candidate_data.get("sdpMid"),
                                sdpMLineIndex=candidate_data.get("sdpMLineIndex"),
                                **params
                            )
                            await pc.addIceCandidate(candidate)
                    except Exception as e:
                        logger.error(f"Error adding ICE candidate: {e}")
                        
    except Exception as e:
        logger.error(f"WebSocket handler error: {e}", exc_info=True)
    finally:
        logger.info("WebSocket connection closed.")
        if processor: 
            await processor.stop()
        if pc in pcs: 
            pcs.remove(pc)
        if pc.connectionState != "closed": 
            await pc.close()
    
    return ws

# --- Enhanced Main Application ---
async def index_handler(request):
    return web.Response(text=HTML_CLIENT, content_type='text/html')

async def on_shutdown(app):
    logger.info("Shutting down server...")
    shutdown_tasks = []
    for pc_conn in list(pcs):
        shutdown_tasks.append(pc_conn.close())
    
    if shutdown_tasks:
        await asyncio.gather(*shutdown_tasks, return_exceptions=True)
    
    pcs.clear()
    executor.shutdown(wait=True)
    logger.info("Shutdown complete.")

async def main():
    # Enhanced model initialization with retry
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if initialize_models():
                break
            else:
                if attempt < max_retries - 1:
                    logger.warning(f"Model initialization failed, retrying ({attempt + 1}/{max_retries})...")
                    await asyncio.sleep(2)
                else:
                    logger.error("Failed to initialize models after all retries.")
                    return
        except Exception as e:
            logger.error(f"Model initialization error (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                return
            await asyncio.sleep(2)

    # Start web server
    app = web.Application()
    app.router.add_get('/', index_handler)
    app.router.add_get('/ws', websocket_handler)
    app.on_shutdown.append(on_shutdown)
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 7860)
    await site.start()
    
    print("✅ Server started successfully on http://0.0.0.0:7860")
    print("🚀 Your enhanced speech-to-speech agent is live!")
    print("   Press Ctrl+C to stop the server.")
    
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        logger.info("Server shutting down...")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Server shutting down by user request...")
