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
import weakref

from aiohttp import web, WSMsgType
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack, RTCIceCandidate, RTCConfiguration, RTCIceServer, mediastreams
import av

from transformers import pipeline
from chatterbox.tts import ChatterboxTTS
import torch.hub

# --- Basic Setup ---
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
executor = ThreadPoolExecutor(max_workers=8)
pcs = weakref.WeakSet()

# Enhanced HTML with connection stability improvements
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
    let pc, ws, localStream, keepAliveInterval;
    const remoteAudio = document.getElementById('remoteAudio');
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    const statusDiv = document.getElementById('status');

    function updateStatus(message, className) { 
        statusDiv.textContent = message; 
        statusDiv.className = `status ${className}`; 
    }

    async function start() {
        startBtn.disabled = true;
        updateStatus('🔄 Connecting...', 'connecting');
        try {
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            if (audioContext.state === 'suspended') { 
                await audioContext.resume(); 
            }

            localStream = await navigator.mediaDevices.getUserMedia({ 
                audio: { 
                    echoCancellation: false,
                    noiseSuppression: false,
                    autoGainControl: false,
                    sampleRate: 48000,
                    channelCount: 1
                } 
            });
            
            pc = new RTCPeerConnection({ 
                iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
                iceCandidatePoolSize: 10
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
                        if(pc && pc.connectionState === 'connected') {
                            setTimeout(() => updateStatus('✅ Listening...', 'connected'), 1000);
                        }
                    };
                    
                    remoteAudio.onpause = () => {
                        if(pc && pc.connectionState === 'connected') {
                            setTimeout(() => updateStatus('✅ Listening...', 'connected'), 1000);
                        }
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
                if (state === 'connecting') {
                    updateStatus('🤝 Establishing secure connection...', 'connecting');
                } else if (state === 'connected') { 
                    updateStatus('✅ Listening...', 'connected'); 
                    stopBtn.disabled = false;
                } else if (state === 'failed' || state === 'closed' || state === 'disconnected') {
                    stop();
                }
            };

            const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${location.host}/ws`);

            ws.onopen = async () => {
                console.log('WebSocket connected');
                const offer = await pc.createOffer();
                await pc.setLocalDescription(offer);
                ws.send(JSON.stringify(offer));
                
                // Keep connection alive
                keepAliveInterval = setInterval(() => {
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({ type: 'ping' }));
                    }
                }, 25000);
            };

            ws.onmessage = async e => {
                try {
                    const data = JSON.parse(e.data);
                    if (data.type === 'answer' && !pc.currentRemoteDescription) {
                        await pc.setRemoteDescription(new RTCSessionDescription(data));
                    } else if (data.type === 'pong') {
                        // Keep-alive response
                        console.log('Keep-alive pong received');
                    }
                } catch (err) {
                    console.error('WebSocket message error:', err);
                }
            };

            ws.onclose = e => {
                console.log('WebSocket closed:', e.code, e.reason);
                if (keepAliveInterval) {
                    clearInterval(keepAliveInterval);
                    keepAliveInterval = null;
                }
                if (pc && pc.connectionState !== 'closed') {
                    stop();
                }
            };
            
            ws.onerror = e => {
                console.error('WebSocket error:', e);
                stop();
            };

        } catch (err) { 
            console.error(err); 
            updateStatus(`❌ Error: ${err.message}`, 'disconnected'); 
            stop(); 
        }
    }

    function stop() {
        if (keepAliveInterval) {
            clearInterval(keepAliveInterval);
            keepAliveInterval = null;
        }
        if (ws) { 
            ws.onclose = null; 
            ws.onerror = null; 
            ws.close(); 
            ws = null; 
        }
        if (pc) { 
            pc.onconnectionstatechange = null; 
            pc.onicecandidate = null; 
            pc.ontrack = null; 
            pc.close(); 
            pc = null; 
        }
        if (localStream) { 
            localStream.getTracks().forEach(track => track.stop()); 
            localStream = null; 
        }
        if (remoteAudio.srcObject) {
            remoteAudio.srcObject = null;
        }
        updateStatus('🔌 Disconnected', 'disconnected');
        startBtn.disabled = false;
        stopBtn.disabled = true;
    }

    window.addEventListener('beforeunload', stop);
</script>
</body>
</html>
"""

# --- Utility Functions ---
def candidate_from_sdp(candidate_string: str) -> dict:
    if candidate_string.startswith("candidate:"): 
        candidate_string = candidate_string[10:]
    bits = candidate_string.split()
    if len(bits) < 8: 
        raise ValueError(f"Invalid candidate string: {candidate_string}")
    params = {
        'component': int(bits[1]), 
        'foundation': bits[0], 
        'ip': bits[4], 
        'port': int(bits[5]), 
        'priority': int(bits[3]), 
        'protocol': bits[2], 
        'type': bits[7]
    }
    for i in range(8, len(bits) - 1, 2):
        if bits[i] == "raddr": 
            params['relatedAddress'] = bits[i + 1]
        elif bits[i] == "rport": 
            params['relatedPort'] = int(bits[i + 1])
    return params

def parse_ultravox_response(result):
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
        elif isinstance(result, dict):
            if 'generated_text' in result:
                return result['generated_text'].strip()
            elif 'text' in result:
                return result['text'].strip()
        return ""
    except Exception as e:
        logger.error(f"Error parsing Ultravox response: {e}")
        return ""

# --- Model Loading and VAD ---
class SileroVAD:
    def __init__(self):
        try:
            logger.info("🎤 Loading Silero VAD model...")
            self.model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad', 
                model='silero_vad', 
                force_reload=False, 
                onnx=False
            )
            (self.get_speech_timestamps, _, _, _, _) = utils
            logger.info("✅ Silero VAD loaded successfully")
        except Exception as e:
            logger.error(f"❌ Error loading Silero VAD: {e}", exc_info=True)
            self.model = None

    def detect_speech(self, audio_tensor, sample_rate=16000):
        if self.model is None: 
            return True
        try:
            if isinstance(audio_tensor, np.ndarray): 
                audio_tensor = torch.from_numpy(audio_tensor.copy()).float()
            
            if audio_tensor.dtype != torch.float32:
                audio_tensor = audio_tensor.float()
            
            if audio_tensor.abs().max() > 1.0:
                audio_tensor = audio_tensor / audio_tensor.abs().max()
            
            if audio_tensor.abs().max() < 0.015: 
                return False
            
            speech_timestamps = self.get_speech_timestamps(
                audio_tensor, 
                self.model, 
                sampling_rate=sample_rate, 
                min_speech_duration_ms=400,
                min_silence_duration_ms=200
            )
            return len(speech_timestamps) > 0
        except Exception as e:
            logger.error(f"VAD detection error: {e}")
            return True

def initialize_models():
    global uv_pipe, tts_model, vad_model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"🚀 Initializing models on device: {device}")
    
    try:
        vad_model = SileroVAD()
        if not vad_model.model: 
            return False
            
        logger.info("📥 Loading Ultravox pipeline...")
        uv_pipe = pipeline(
            model="fixie-ai/ultravox-v0_4", 
            trust_remote_code=True, 
            device_map="auto", 
            torch_dtype=torch.float16,
            attn_implementation="eager"
        )
        logger.info("✅ Ultravox pipeline loaded successfully")

        logger.info("📥 Loading Chatterbox TTS...")
        tts_model = ChatterboxTTS.from_pretrained(device=device)
        logger.info("✅ Chatterbox TTS loaded successfully")
        
        logger.info("🎉 All models loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"❌ Critical model loading error: {e}", exc_info=True)
        return False

# --- Audio Processing Classes ---
class AudioBuffer:
    def __init__(self, max_duration=6.0, sample_rate=16000):
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)
        self.buffer = collections.deque(maxlen=self.max_samples)
        self.last_process_time = time.time()
        self.min_speech_samples = int(1.2 * sample_rate)  # 1.2 seconds minimum
        self.process_interval = 1.0  # Process every 1 second
        self.silence_threshold = 0.015
    
    def add_audio(self, audio_data):
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
            
        # Enhanced normalization for better Ultravox recognition
        max_val = np.abs(audio_data).max()
        if max_val > 1.0:
            audio_data = audio_data / max_val
        elif max_val > 0 and max_val < 0.4:
            # More aggressive amplification for quiet speech
            audio_data = audio_data * min(1.2, 0.6 / max_val)
            
        self.buffer.extend(audio_data.flatten())

    def get_audio_array(self): 
        return np.array(list(self.buffer), dtype=np.float32)
    
    def should_process(self):
        current_time = time.time()
        if (len(self.buffer) >= self.min_speech_samples and 
            (current_time - self.last_process_time) >= self.process_interval):
            
            self.last_process_time = current_time
            audio_array = self.get_audio_array()
            
            if np.abs(audio_array).max() < self.silence_threshold:
                return False
                
            return vad_model.detect_speech(audio_array, self.sample_rate)
        return False

    def reset(self): 
        self.buffer.clear()

class ResponseAudioTrack(MediaStreamTrack):
    """Completely rewritten audio track with stable streaming"""
    kind = "audio"
    
    def __init__(self):
        super().__init__()
        self._audio_buffer = []
        self._buffer_pos = 0
        self._timestamp = 0
        self._time_base = fractions.Fraction(1, 48000)
        self._frame_samples = 960  # 20ms at 48kHz
        self._is_streaming = False
        self._stream_lock = asyncio.Lock()

    async def recv(self):
        """Generate audio frames with stable streaming"""
        frame = np.zeros(self._frame_samples, dtype=np.int16)
        
        async with self._stream_lock:
            if self._buffer_pos < len(self._audio_buffer):
                # Calculate how many samples we can copy
                remaining_buffer = len(self._audio_buffer) - self._buffer_pos
                samples_to_copy = min(self._frame_samples, remaining_buffer)
                
                # Copy audio data to frame
                frame[:samples_to_copy] = self._audio_buffer[self._buffer_pos:self._buffer_pos + samples_to_copy]
                self._buffer_pos += samples_to_copy
                
                # Check if we've finished streaming
                if self._buffer_pos >= len(self._audio_buffer):
                    self._is_streaming = False
        
        # Create audio frame
        audio_frame = av.AudioFrame.from_ndarray(
            np.array([frame]), 
            format="s16", 
            layout="mono"
        )
        audio_frame.pts = self._timestamp
        audio_frame.sample_rate = 48000
        audio_frame.time_base = self._time_base
        
        self._timestamp += self._frame_samples
        return audio_frame

    async def stream_audio(self, audio_float32):
        """Stream audio with stable buffering"""
        if audio_float32.size == 0:
            return
            
        async with self._stream_lock:
            # Convert to int16 with proper clipping
            audio_int16 = np.clip(audio_float32 * 32767, -32767, 32767).astype(np.int16)
            
            # Replace current buffer
            self._audio_buffer = audio_int16.tolist()
            self._buffer_pos = 0
            self._is_streaming = True
            
        logger.info(f"🎵 Streaming {len(self._audio_buffer)} audio samples")

    def is_streaming(self):
        """Check if audio is currently being streamed"""
        return self._is_streaming

    async def wait_for_completion(self):
        """Wait for audio streaming to complete"""
        while self._is_streaming:
            await asyncio.sleep(0.1)

class AudioProcessor:
    def __init__(self, output_track: ResponseAudioTrack, executor: ThreadPoolExecutor):
        self.track = None
        self.buffer = AudioBuffer()
        self.output_track = output_track
        self.task = None
        self.executor = executor
        self.is_speaking = False
        self.processing_lock = asyncio.Lock()
        self.speech_lock = threading.Lock()

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
        try:
            while True:
                # Enhanced echo cancellation
                if self.is_speaking or self.output_track.is_streaming():
                    try:
                        # Drain incoming audio while speaking
                        await asyncio.wait_for(self.track.recv(), timeout=0.001)
                    except asyncio.TimeoutError:
                        await asyncio.sleep(0.001)
                    continue

                try: 
                    frame = await self.track.recv()
                except mediastreams.MediaStreamError: 
                    logger.warning("Client media stream ended.")
                    break
                except Exception as e:
                    logger.error(f"Frame receive error: {e}")
                    continue
                
                # Process audio frame
                audio_data = frame.to_ndarray().flatten()
                
                if audio_data.dtype == np.int16:
                    audio_float32 = audio_data.astype(np.float32) / 32768.0
                else:
                    audio_float32 = audio_data.astype(np.float32)
                
                # High-quality resampling to 16kHz for Ultravox
                if frame.sample_rate != 16000:
                    resampled_audio = librosa.resample(
                        audio_float32, 
                        orig_sr=frame.sample_rate, 
                        target_sr=16000,
                        res_type='kaiser_best'
                    )
                else:
                    resampled_audio = audio_float32
                
                self.buffer.add_audio(resampled_audio)
                
                # Process when ready
                if self.buffer.should_process() and not self.processing_lock.locked():
                    audio_to_process = self.buffer.get_audio_array()
                    self.buffer.reset()
                    logger.info(f"🧠 Processing {len(audio_to_process)} samples (max: {np.abs(audio_to_process).max():.4f})")
                    asyncio.create_task(self.process_speech(audio_to_process))
                        
        except asyncio.CancelledError: 
            pass
        except Exception as e: 
            logger.error(f"Audio processor error: {e}", exc_info=True)
        finally: 
            logger.info("Audio processor stopped.")

    def _blocking_tts(self, text: str) -> np.ndarray:
        """Generate TTS with length control for stability"""
        try:
            # Limit text length for connection stability
            if len(text) > 400:
                # Find last complete sentence within limit
                truncated = text[:400]
                last_period = truncated.rfind('.')
                last_exclamation = truncated.rfind('!')
                last_question = truncated.rfind('?')
                last_sentence_end = max(last_period, last_exclamation, last_question)
                
                if last_sentence_end > 200:  # If we found a reasonable sentence end
                    text = text[:last_sentence_end + 1]
                else:
                    text = text[:400] + "."
                
                logger.info(f"Text truncated for stability: '{text[:100]}...'")
            
            logger.info(f"🗣️ Generating TTS for: '{text[:100]}...'")
            
            with torch.inference_mode():
                wav = tts_model.generate(text).cpu().numpy().flatten()
                
                if wav.size == 0:
                    logger.warning("TTS generated empty audio")
                    return np.array([], dtype=np.float32)
                
                # Limit audio length for connection stability
                max_samples = 24000 * 20  # 20 seconds max at 24kHz
                if wav.size > max_samples:
                    wav = wav[:max_samples]
                    logger.info(f"Audio truncated to 20 seconds for stability")
                
                # High-quality resampling to 48kHz
                resampled_wav = librosa.resample(
                    wav.astype(np.float32), 
                    orig_sr=24000, 
                    target_sr=48000,
                    res_type='kaiser_best'
                )
                
                logger.info(f"✅ TTS generated {resampled_wav.size} samples ({resampled_wav.size/48000:.2f}s)")
                return resampled_wav
                
        except Exception as e:
            logger.error(f"TTS generation failed: {e}", exc_info=True)
            return np.array([], dtype=np.float32)

    async def process_speech(self, audio_array):
        """Enhanced speech processing with connection stability"""
        async with self.processing_lock:
            try:
                # Enhanced audio preprocessing for Ultravox
                audio_array = audio_array.astype(np.float32)
                
                # Better normalization for Ultravox
                max_val = np.abs(audio_array).max()
                if max_val > 1.0:
                    audio_array = audio_array / max_val
                elif max_val > 0 and max_val < 0.25:
                    # Amplify quiet speech more aggressively
                    audio_array = audio_array * (0.7 / max_val)
                
                if audio_array.ndim > 1:
                    audio_array = audio_array.flatten()
                
                # Ensure sufficient length for Ultravox
                if len(audio_array) < 19200:  # 1.2 seconds at 16kHz
                    logger.info("Audio too short for processing")
                    return
                
                # Pad to ensure consistent processing
                target_length = max(19200, len(audio_array))
                if len(audio_array) < target_length:
                    audio_array = np.pad(audio_array, (0, target_length - len(audio_array)), 'constant')
                
                logger.info(f"🎤 Processing audio: length={len(audio_array)}, max={np.abs(audio_array).max():.4f}")
                
                # Call Ultravox with conservative parameters for stability
                with torch.inference_mode():
                    result = uv_pipe(
                        {
                            'audio': audio_array, 
                            'turns': [], 
                            'sampling_rate': 16000
                        }, 
                        max_new_tokens=150,  # Reduced for stability
                        do_sample=True,
                        temperature=0.6,
                        top_p=0.85,
                        repetition_penalty=1.05
                    )

                response_text = parse_ultravox_response(result)
                if not response_text:
                    logger.info("No response generated by Ultravox.")
                    return

                logger.info(f"🤖 AI Response: '{response_text}'")

                # Set speaking state
                with self.speech_lock:
                    self.is_speaking = True
                    
                try:
                    # Generate TTS
                    loop = asyncio.get_running_loop()
                    resampled_wav = await loop.run_in_executor(self.executor, self._blocking_tts, response_text)

                    if resampled_wav.size > 0:
                        logger.info("🤖 AI is speaking...")
                        
                        # Stream audio
                        await self.output_track.stream_audio(resampled_wav)
                        
                        # Wait for streaming completion
                        await self.output_track.wait_for_completion()

                        logger.info("✅ AI finished speaking, now listening.")
                    else:
                        logger.warning("TTS generated empty audio.")
                        
                finally:
                    # Always reset speaking state
                    with self.speech_lock:
                        self.is_speaking = False

            except Exception as e:
                logger.error(f"Speech processing error: {e}", exc_info=True)
                with self.speech_lock:
                    self.is_speaking = False

# --- WebRTC and WebSocket Handling ---
async def websocket_handler(request):
    ws = web.WebSocketResponse(heartbeat=30, timeout=120)  # Increased timeout
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
            if processor:
                await processor.stop()
            if pc in pcs: 
                pcs.discard(pc)

    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    if data["type"] == "offer":
                        await pc.setRemoteDescription(RTCSessionDescription(sdp=data["sdp"], type=data["type"]))
                        answer = await pc.createAnswer()
                        await pc.setLocalDescription(answer)
                        await ws.send_json({"type": "answer", "sdp": pc.localDescription.sdp})
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
                    elif data["type"] == "ping":
                        # Keep-alive ping
                        await ws.send_json({"type": "pong"})
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error: {e}")
            elif msg.type == WSMsgType.ERROR:
                logger.error(f"WebSocket error: {ws.exception()}")
                break
    except Exception as e:
        logger.error(f"WebSocket handler error: {e}", exc_info=True)
    finally:
        logger.info("WebSocket connection closed.")
        if processor: 
            await processor.stop()
        if pc in pcs: 
            pcs.discard(pc)
        if pc.connectionState != "closed": 
            await pc.close()
    return ws

async def index_handler(request):
    return web.Response(text=HTML_CLIENT, content_type='text/html')

# --- Main Application Logic ---
async def on_shutdown(app):
    logger.info("Shutting down server...")
    pc_list = list(pcs)
    for pc_conn in pc_list: 
        try:
            await pc_conn.close()
        except Exception as e:
            logger.error(f"Error closing peer connection: {e}")
    pcs.clear()
    executor.shutdown(wait=True)
    logger.info("Shutdown complete.")

async def main():
    if not initialize_models():
        logger.error("Failed to initialize models. The application cannot start.")
        return

    app = web.Application()
    app.router.add_get('/', index_handler)
    app.router.add_get('/ws', websocket_handler)
    app.on_shutdown.append(on_shutdown)
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 7860)
    await site.start()
    
    print("✅ Server started successfully on http://0.0.0.0:7860")
    print("🚀 Your speech-to-speech agent is live!")
    print("   Press Ctrl+C to stop the server.")
    
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        print("\n🛑 Server shutting down by user request...")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Server shutting down by user request...")
