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

# Enhanced HTML with robust connection handling
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
    let pc, ws, localStream, keepAliveInterval, reconnectAttempts = 0;
    const maxReconnectAttempts = 5;
    const remoteAudio = document.getElementById('remoteAudio');
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    const statusDiv = document.getElementById('status');

    function updateStatus(message, className) { 
        statusDiv.textContent = message; 
        statusDiv.className = `status ${className}`; 
        console.log('Status update:', message);
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
                    
                    // Enhanced audio event handling
                    remoteAudio.onplaying = () => {
                        updateStatus('🤖 AI Speaking...', 'speaking');
                    };
                    
                    remoteAudio.onended = () => {
                        console.log('Audio playback ended');
                        if(pc && pc.connectionState === 'connected') {
                            setTimeout(() => updateStatus('✅ Listening...', 'connected'), 1500);
                        }
                    };
                    
                    remoteAudio.onpause = () => {
                        console.log('Audio playback paused');
                        if(pc && pc.connectionState === 'connected') {
                            setTimeout(() => updateStatus('✅ Listening...', 'connected'), 1500);
                        }
                    };
                    
                    remoteAudio.onerror = (e) => {
                        console.error('Audio playback error:', e);
                        updateStatus('⚠️ Audio Error', 'connecting');
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
                    reconnectAttempts = 0;
                } else if (state === 'failed') {
                    updateStatus('❌ Connection Failed', 'disconnected');
                    if (reconnectAttempts < maxReconnectAttempts) {
                        setTimeout(start, 2000);
                        reconnectAttempts++;
                    } else {
                        stop();
                    }
                } else if (state === 'closed' || state === 'disconnected') {
                    updateStatus('🔌 Disconnected', 'disconnected');
                    stop();
                }
            };

            await connectWebSocket();

        } catch (err) { 
            console.error('Start error:', err); 
            updateStatus(`❌ Error: ${err.message}`, 'disconnected'); 
            stop(); 
        }
    }

    async function connectWebSocket() {
        const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        ws = new WebSocket(`${protocol}//${location.host}/ws`);

        ws.onopen = async () => {
            console.log('WebSocket connected');
            try {
                const offer = await pc.createOffer();
                await pc.setLocalDescription(offer);
                ws.send(JSON.stringify(offer));
                
                // Keep connection alive with longer intervals
                keepAliveInterval = setInterval(() => {
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({ type: 'ping' }));
                    }
                }, 30000); // Every 30 seconds
                
            } catch (err) {
                console.error('Offer creation error:', err);
                updateStatus('❌ Connection Error', 'disconnected');
            }
        };

        ws.onmessage = async e => {
            try {
                const data = JSON.parse(e.data);
                if (data.type === 'answer' && !pc.currentRemoteDescription) {
                    await pc.setRemoteDescription(new RTCSessionDescription(data));
                } else if (data.type === 'pong') {
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
            
            // Don't auto-reconnect if user manually stopped
            if (e.code !== 1000 && pc && pc.connectionState !== 'closed') {
                updateStatus('🔄 Reconnecting...', 'connecting');
                if (reconnectAttempts < maxReconnectAttempts) {
                    setTimeout(connectWebSocket, 2000);
                    reconnectAttempts++;
                } else {
                    stop();
                }
            }
        };
        
        ws.onerror = e => {
            console.error('WebSocket error:', e);
            updateStatus('❌ Connection Error', 'disconnected');
        };
    }

    function stop() {
        if (keepAliveInterval) {
            clearInterval(keepAliveInterval);
            keepAliveInterval = null;
        }
        if (ws) { 
            ws.onclose = null; 
            ws.onerror = null; 
            ws.close(1000); // Normal closure
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
        reconnectAttempts = 0;
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
            
            if audio_tensor.abs().max() < 0.02: 
                return False
            
            speech_timestamps = self.get_speech_timestamps(
                audio_tensor, 
                self.model, 
                sampling_rate=sample_rate, 
                min_speech_duration_ms=500,
                min_silence_duration_ms=300
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
    def __init__(self, max_duration=8.0, sample_rate=16000):
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)
        self.buffer = collections.deque(maxlen=self.max_samples)
        self.last_process_time = time.time()
        self.min_speech_samples = int(1.5 * sample_rate)  # 1.5 seconds minimum
        self.process_interval = 1.2  # Process every 1.2 seconds
        self.silence_threshold = 0.02
    
    def add_audio(self, audio_data):
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
            
        # Enhanced normalization for Ultravox
        max_val = np.abs(audio_data).max()
        if max_val > 1.0:
            audio_data = audio_data / max_val
        elif max_val > 0 and max_val < 0.5:
            audio_data = audio_data * min(1.0, 0.7 / max_val)
            
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

class StableAudioTrack(MediaStreamTrack):
    """Completely rewritten stable audio track"""
    kind = "audio"
    
    def __init__(self):
        super().__init__()
        self._audio_samples = []
        self._sample_index = 0
        self._timestamp = 0
        self._time_base = fractions.Fraction(1, 48000)
        self._frame_samples = 960  # 20ms at 48kHz
        self._lock = threading.Lock()
        self._active = False

    async def recv(self):
        """Safe recv implementation that never crashes the connection"""
        frame = np.zeros(self._frame_samples, dtype=np.int16)
        
        with self._lock:
            if self._sample_index < len(self._audio_samples):
                # Calculate available samples
                available = len(self._audio_samples) - self._sample_index
                samples_to_copy = min(self._frame_samples, available)
                
                # Copy samples to frame
                frame[:samples_to_copy] = self._audio_samples[self._sample_index:self._sample_index + samples_to_copy]
                self._sample_index += samples_to_copy
                
                # Mark as inactive when done
                if self._sample_index >= len(self._audio_samples):
                    self._active = False
        
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

    def set_audio(self, audio_float32):
        """Set new audio data for streaming"""
        if audio_float32.size == 0:
            return
            
        with self._lock:
            # Convert to int16 with proper clipping
            audio_int16 = np.clip(audio_float32 * 32767, -32767, 32767).astype(np.int16)
            
            # Set new audio samples
            self._audio_samples = audio_int16.tolist()
            self._sample_index = 0
            self._active = True
            
        logger.info(f"🎵 Set {len(self._audio_samples)} audio samples for streaming")

    def is_active(self):
        """Check if audio is currently being streamed"""
        with self._lock:
            return self._active

    def get_duration(self):
        """Get expected playback duration"""
        with self._lock:
            return len(self._audio_samples) / 48000 if self._audio_samples else 0

class AudioProcessor:
    def __init__(self, output_track: StableAudioTrack, executor: ThreadPoolExecutor):
        self.track = None
        self.buffer = AudioBuffer()
        self.output_track = output_track
        self.task = None
        self.executor = executor
        self.is_speaking = False
        self.processing_lock = asyncio.Lock()
        self.speech_lock = threading.Lock()
        self.running = True

    def add_track(self, track): 
        self.track = track
        
    async def start(self): 
        self.running = True
        self.task = asyncio.create_task(self._run())
        
    async def stop(self):
        self.running = False
        if self.task: 
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass

    async def _run(self):
        """Main audio processing loop with robust error handling"""
        try:
            while self.running:
                # Enhanced echo cancellation
                if self.is_speaking or self.output_track.is_active():
                    try:
                        # Safely drain incoming audio while speaking
                        await asyncio.wait_for(self.track.recv(), timeout=0.01)
                    except (asyncio.TimeoutError, mediastreams.MediaStreamError):
                        await asyncio.sleep(0.01)
                    except Exception as e:
                        logger.error(f"Error during audio draining: {e}")
                        await asyncio.sleep(0.01)
                    continue

                try: 
                    frame = await asyncio.wait_for(self.track.recv(), timeout=0.1)
                except asyncio.TimeoutError:
                    await asyncio.sleep(0.01)
                    continue
                except mediastreams.MediaStreamError: 
                    logger.warning("Client media stream ended.")
                    break
                except Exception as e:
                    logger.error(f"Frame receive error: {e}")
                    await asyncio.sleep(0.1)
                    continue
                
                # Process audio frame safely
                try:
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
                    
                    # Process when ready and not already processing
                    if self.buffer.should_process() and not self.processing_lock.locked():
                        audio_to_process = self.buffer.get_audio_array()
                        self.buffer.reset()
                        logger.info(f"🧠 Processing {len(audio_to_process)} samples (max: {np.abs(audio_to_process).max():.4f})")
                        asyncio.create_task(self.process_speech(audio_to_process))
                        
                except Exception as e:
                    logger.error(f"Audio processing error: {e}")
                    await asyncio.sleep(0.1)
                        
        except asyncio.CancelledError: 
            pass
        except Exception as e: 
            logger.error(f"Audio processor critical error: {e}", exc_info=True)
        finally: 
            logger.info("Audio processor stopped.")

    def _blocking_tts(self, text: str) -> np.ndarray:
        """Generate TTS with strict limits for connection stability"""
        try:
            # Strict text length control for stability
            if len(text) > 300:
                # Find sentence boundary
                sentences = text.split('.')
                truncated = ""
                for sentence in sentences:
                    if len(truncated + sentence + ".") <= 300:
                        truncated += sentence + "."
                    else:
                        break
                text = truncated if truncated else text[:300] + "."
                logger.info(f"Text truncated for stability")
            
            logger.info(f"🗣️ Generating TTS for: '{text[:50]}...'")
            
            with torch.inference_mode():
                wav = tts_model.generate(text).cpu().numpy().flatten()
                
                if wav.size == 0:
                    logger.warning("TTS generated empty audio")
                    return np.array([], dtype=np.float32)
                
                # Strict audio length limit for connection stability
                max_samples = 24000 * 15  # 15 seconds max at 24kHz
                if wav.size > max_samples:
                    wav = wav[:max_samples]
                    logger.info(f"Audio truncated to 15 seconds for stability")
                
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
                elif max_val > 0 and max_val < 0.3:
                    audio_array = audio_array * (0.8 / max_val)
                
                if audio_array.ndim > 1:
                    audio_array = audio_array.flatten()
                
                # Ensure sufficient length for Ultravox
                if len(audio_array) < 24000:  # 1.5 seconds at 16kHz
                    logger.info("Audio too short for processing")
                    return
                
                # Pad to ensure consistent processing
                target_length = max(24000, len(audio_array))
                if len(audio_array) < target_length:
                    audio_array = np.pad(audio_array, (0, target_length - len(audio_array)), 'constant')
                
                logger.info(f"🎤 Processing audio: length={len(audio_array)}, max={np.abs(audio_array).max():.4f}")
                
                # Call Ultravox with stability-focused parameters
                with torch.inference_mode():
                    result = uv_pipe(
                        {
                            'audio': audio_array, 
                            'turns': [], 
                            'sampling_rate': 16000
                        }, 
                        max_new_tokens=120,  # Conservative for stability
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
                        
                        # Set audio for stable streaming
                        self.output_track.set_audio(resampled_wav)
                        
                        # Wait for expected playback duration
                        expected_duration = self.output_track.get_duration()
                        await asyncio.sleep(expected_duration + 1.0)  # Buffer time

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
    """Enhanced WebSocket handler with robust connection management"""
    ws = web.WebSocketResponse(heartbeat=45, timeout=180)  # Longer timeouts
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
            output_audio_track = StableAudioTrack()
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
                        # Respond to keep-alive ping
                        await ws.send_json({"type": "pong"})
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error: {e}")
                except Exception as e:
                    logger.error(f"Message handling error: {e}")
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
