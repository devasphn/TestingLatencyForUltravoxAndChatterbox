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
logging.getLogger('websockets').setLevel(logging.WARNING)
logging.getLogger('huggingface_hub').setLevel(logging.WARNING)
logging.getLogger('transformers').setLevel(logging.WARNING)
logging.getLogger('torch').setLevel(logging.WARNING)

# --- Global Variables ---
uv_pipe, tts_model, vad_model = None, None, None
executor = ThreadPoolExecutor(max_workers=4)
pcs = weakref.WeakSet()

# --- Constants ---
SAMPLE_RATE_VAD = 16000
SAMPLE_RATE_TTS_INPUT = 24000
SAMPLE_RATE_OUTPUT = 48000
VAD_MIN_SPEECH_DURATION_MS = 250
VAD_MIN_SILENCE_DURATION_MS = 150
AUDIO_PROCESS_INTERVAL_MS = 400
AUDIO_MAX_INPUT_DURATION_SEC = 5.0
TTS_MAX_DURATION_SEC = 10.0
ULTRAVOX_MAX_NEW_TOKENS = 80
ULTRAVOX_AUDIO_CHUNK_SIZE_SAMPLES = int(SAMPLE_RATE_VAD * 0.5)

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
    let pc, ws, localStream, reconnectAttempts = 0;
    const remoteAudio = document.getElementById('remoteAudio');
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    const statusDiv = document.getElementById('status');
    let isSpeaking = false;

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
                    channelCount: 1,
                    latency: { ideal: 0.025, min: 0.010, max: 0.050 }
                } 
            });
            
            pc = new RTCPeerConnection({ 
                iceServers: [{urls: 'stun:stun.l.google.com:19302'}],
                iceCandidatePoolSize: 10
            });
            
            localStream.getTracks().forEach(track => pc.addTrack(track, localStream));

            pc.ontrack = e => {
                console.log('Remote track received!');
                if (remoteAudio.srcObject !== e.streams[0]) {
                    remoteAudio.srcObject = e.streams[0];
                    remoteAudio.play().catch(err => console.error("Autoplay failed:", err));
                    
                    remoteAudio.onplaying = () => {
                        isSpeaking = true;
                        updateStatus('🤖 AI Speaking...', 'speaking');
                    };
                    
                    remoteAudio.onended = () => {
                        if(isSpeaking) {
                            isSpeaking = false;
                            setTimeout(() => {
                                if(!isSpeaking && pc && pc.connectionState === 'connected') {
                                    updateStatus('✅ Listening...', 'connected');
                                }
                            }, 500);
                        }
                    };
                    
                    remoteAudio.onpause = () => {
                        if(isSpeaking) {
                            isSpeaking = false;
                            setTimeout(() => {
                                if(!isSpeaking && pc && pc.connectionState === 'connected') {
                                    updateStatus('✅ Listening...', 'connected');
                                }
                            }, 500);
                        }
                    };
                }
            };

            pc.onicecandidate = e => {
                if (e.candidate && ws && ws.readyState === WebSocket.OPEN) {
                    // Correctly structure ICE candidate data for sending
                    const candidate_data = {
                        type: 'ice-candidate',
                        // The 'candidate' key here holds the actual ICE string from e.candidate.candidate
                        // and other metadata like sdpMid and sdpMLineIndex.
                        candidate: {
                            candidate: e.candidate.candidate,
                            sdpMid: e.candidate.sdpMid,
                            sdpMLineIndex: e.candidate.sdpMLineIndex,
                            usernameFragment: e.candidate.usernameFragment
                        }
                    };
                    ws.send(JSON.stringify(candidate_data));
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
                } else if (state === 'failed' || state === 'closed' || state === 'disconnected') {
                    stop();
                }
            };

            const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${location.host}/ws`);
            ws.binaryType = 'arraybuffer';

            ws.onopen = async () => {
                console.log('WebSocket connected');
                const offer = await pc.createOffer();
                await pc.setLocalDescription(offer);
                ws.send(JSON.stringify({type: offer.type, sdp: offer.sdp}));
            };

            ws.onmessage = async e => {
                try {
                    const data = JSON.parse(e.data);
                    const msg_type = data.type;

                    if (msg_type === 'answer') {
                        console.log('Received answer from server.');
                        await pc.setRemoteDescription(new RTCSessionDescription(data));
                    } else if (msg_type === 'ice-candidate') {
                        console.log('Received ICE candidate from server.');
                        // Get the inner object containing candidate details
                        const candidate_data = data.candidate; 
                        
                        // Check if the 'candidate' key (which is the ICE string) exists
                        if (candidate_data && candidate_data.candidate) {
                            try {
                                // Construct RTCIceCandidate using the correct properties
                                const candidate = new RTCIceCandidate({
                                    candidate: candidate_data.candidate, // This is the ICE string itself
                                    sdpMid: candidate_data.sdpMid,
                                    sdpMLineIndex: candidate_data.sdpMLineIndex,
                                    usernameFragment: candidate_data.usernameFragment
                                });
                                await pc.addIceCandidate(candidate);
                            } catch (e) {
                                logger.error(f"Failed to create and add RTCIceCandidate: {e}. Data: {candidate_data}", exc_info=True);
                            }
                        } else {
                            console.error('Received ICE candidate message without valid candidate data.');
                        }
                    } else {
                        console.log(`Received unknown message type: ${msg_type}`);
                    }
                } catch (err) {
                    console.error('WebSocket message error:', err);
                    if (err instanceof SyntaxError) {
                        console.error("Received non-JSON message:", e.data);
                    }
                }
            };

            ws.onclose = e => {
                console.log('WebSocket closed:', e.code, e.reason);
                if (pc && pc.connectionState !== 'closed') {
                    stop();
                }
            };
            
            ws.onerror = e => {
                console.error('WebSocket error:', e);
                stop();
            };

        } catch (err) { 
            console.error("Error during start():", err); 
            updateStatus(`❌ Error: ${err.message}`, 'disconnected'); 
            stop(); 
        }
    }

    function stop() {
        console.log("Stopping all connections and streams...");
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
        isSpeaking = false;
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
# Removed candidate_from_sdp as it's handled directly in JS now for RTCIceCandidate construction.

# --- Model Loading and VAD ---
class SileroVAD:
    def __init__(self):
        self.model = None
        self.utils = None
        try:
            logger.info("🎤 Loading Silero VAD model...")
            self.model, self.utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            if torch.cuda.is_available():
                self.model.to("cuda:0")
            logger.info("✅ Silero VAD loaded successfully")
        except Exception as e:
            logger.error(f"❌ Error loading Silero VAD: {e}", exc_info=True)

    def detect_speech(self, audio_tensor: torch.Tensor, sample_rate: int) -> bool:
        """Detects speech in an audio tensor using Silero VAD."""
        if self.model is None:
            return True

        try:
            if audio_tensor.dtype != torch.float32:
                audio_tensor = audio_tensor.float()
            
            max_val = audio_tensor.abs().max()
            if max_val > 1.0:
                audio_tensor = audio_tensor / max_val
            
            if max_val < 0.005:
                return False

            if audio_tensor.ndim == 1:
                audio_tensor = audio_tensor.unsqueeze(0)

            speech_timestamps = self.utils[0](
                audio_tensor,
                self.model,
                sampling_rate=sample_rate,
                min_speech_duration_ms=VAD_MIN_SPEECH_DURATION_MS,
                min_silence_duration_ms=VAD_MIN_SILENCE_DURATION_MS,
                return_seconds=True
            )
            
            return len(speech_timestamps) > 0
        except Exception as e:
            logger.error(f"VAD detection error: {e}", exc_info=True)
            return True

def initialize_models():
    """Initializes all necessary models."""
    global uv_pipe, tts_model, vad_model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"🚀 Initializing models on device: {device}")
    
    try:
        vad_model = SileroVAD()
        if vad_model.model is None:
            logger.error("Silero VAD model failed to load. Aborting.")
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
    """Manages a rolling buffer of audio data, processing it when speech is detected."""
    def __init__(self, sample_rate: int = SAMPLE_RATE_VAD):
        self.sample_rate = sample_rate
        self.max_samples = int(AUDIO_MAX_INPUT_DURATION_SEC * sample_rate)
        self.buffer = collections.deque(maxlen=self.max_samples)
        self.last_process_check_time = time.time()
        self.min_speech_samples_for_vad = int(0.8 * sample_rate)
        self.process_interval_sec = AUDIO_PROCESS_INTERVAL_MS / 1000.0
        self.silence_threshold_norm = 0.005

    def add_audio(self, audio_data: np.ndarray):
        """Adds raw audio data (expected to be normalized float32)."""
        if audio_data.size == 0:
            return

        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        max_val = np.abs(audio_data).max()
        if max_val > 1.0:
            audio_data = audio_data / max_val
        elif max_val > 0 and max_val < 0.2:
            audio_data = audio_data * min(2.0, 0.4 / max_val)
        elif max_val == 0:
            return

        self.buffer.extend(audio_data.flatten())

    def get_audio_array(self) -> np.ndarray:
        """Returns the current audio buffer as a numpy array."""
        return np.array(list(self.buffer), dtype=np.float32)

    def get_current_length_samples(self) -> int:
        """Returns the number of samples currently in the buffer."""
        return len(self.buffer)

    def reset(self):
        """Clears the audio buffer."""
        self.buffer.clear()

    def should_process_now(self) -> bool:
        """Determines if the buffer should be processed for speech."""
        current_time = time.time()
        
        if (current_time - self.last_process_check_time) < self.process_interval_sec:
            return False
        
        self.last_process_check_time = current_time

        buffer_len = len(self.buffer)

        if buffer_len < self.min_speech_samples_for_vad:
            return False

        audio_array = self.get_audio_array()
        
        if np.abs(audio_array).max() < self.silence_threshold_norm:
            return False

        return vad_model.detect_speech(torch.from_numpy(audio_array), self.sample_rate)

class ResponseAudioTrack(MediaStreamTrack):
    """
    MediaStreamTrack that sends audio frames to the client.
    Handles buffering and sending of TTS output.
    """
    kind = "audio"
    
    def __init__(self):
        super().__init__()
        self._audio_queue = asyncio.Queue()
        self._current_chunk = None
        self._chunk_pos = 0
        self._timestamp = 0
        self._time_base = fractions.Fraction(1, SAMPLE_RATE_OUTPUT)
        self._frame_samples = 960
        self._total_samples_to_send = 0
        self._samples_sent_in_chunk = 0
        self._is_active = False

    async def recv(self):
        """Receives an audio frame to send to the client."""
        frame_data = np.zeros(self._frame_samples, dtype=np.int16)
        
        samples_copied = 0
        while samples_copied < self._frame_samples:
            if self._current_chunk is None or self._chunk_pos >= len(self._current_chunk):
                try:
                    self._current_chunk = await asyncio.wait_for(self._audio_queue.get(), timeout=0.01)
                    self._chunk_pos = 0
                    if not self._is_active and len(self._current_chunk) > 0:
                        self._is_active = True
                except asyncio.TimeoutError:
                    break
            
            if self._current_chunk is not None:
                samples_available_in_chunk = len(self._current_chunk) - self._chunk_pos
                samples_needed_for_frame = self._frame_samples - samples_copied
                
                copy_count = min(samples_available_in_chunk, samples_needed_for_frame)
                
                frame_data[samples_copied : samples_copied + copy_count] = \
                    self._current_chunk[self._chunk_pos : self._chunk_pos + copy_count]
                
                samples_copied += copy_count
                self._chunk_pos += copy_count
                self._samples_sent_in_chunk += copy_count
        
        if samples_copied == 0:
            self._is_active = False
            return None

        audio_frame = av.AudioFrame.from_ndarray(
            frame_data.reshape(1, -1),
            format="s16", 
            layout="mono"
        )
        audio_frame.pts = self._timestamp
        audio_frame.sample_rate = SAMPLE_RATE_OUTPUT
        audio_frame.time_base = self._time_base
        
        self._timestamp += self._frame_samples
        return audio_frame

    async def queue_audio(self, audio_float32: np.ndarray):
        """Queues audio data (float32) for streaming to the client."""
        if audio_float32.size == 0:
            logger.warning("Attempted to queue empty audio data.")
            return
            
        audio_int16 = np.clip(audio_float32 * 32767, -32767, 32767).astype(np.int16)
        
        self._total_samples_to_send = len(audio_int16)
        self._samples_sent_in_chunk = 0
        self._is_active = False
        
        chunk_size = 1920
        for i in range(0, len(audio_int16), chunk_size):
            chunk = audio_int16[i:i + chunk_size]
            await self._audio_queue.put(chunk)
        
        await self._audio_queue.put(np.array([], dtype=np.int16)) # End marker

    def is_playing(self) -> bool:
        """Checks if there is still audio data to be sent."""
        return self._is_active

    async def wait_for_completion(self, timeout: float = 15.0):
        """Waits until all queued audio has been sent."""
        start_time = time.time()
        while self.is_playing():
            await asyncio.sleep(0.05)
            if time.time() - start_time > timeout:
                logger.warning(f"Timeout waiting for audio completion after {timeout}s.")
                break
        await asyncio.sleep(0.1)

class AudioProcessor:
    """
    Processes incoming audio frames, detects speech, sends to Ultravox,
    gets response, generates TTS, and streams it back via ResponseAudioTrack.
    """
    def __init__(self, output_track: ResponseAudioTrack, executor: ThreadPoolExecutor):
        self.input_track = None
        self.output_track = output_track
        self.audio_buffer = AudioBuffer(SAMPLE_RATE_VAD)
        self._processor_task = None
        self.executor = executor
        self._is_processing_speech = False
        self._ai_is_speaking = False
        self._processing_lock = asyncio.Lock()
        self._speaking_lock = asyncio.Lock()
        self._stop_event = asyncio.Event()

    def add_input_track(self, track: MediaStreamTrack):
        """Sets the incoming audio track."""
        self.input_track = track
        
    async def start(self): 
        """Starts the audio processing loop."""
        if self.input_track is None:
            logger.error("Cannot start AudioProcessor: Input track not set.")
            return
        if self.output_track is None:
            logger.error("Cannot start AudioProcessor: Output track not set.")
            return
            
        logger.info("Starting audio processor...")
        self._processor_task = asyncio.create_task(self._run_processing_loop())
        
    async def stop(self):
        """Stops the audio processing loop and cleans up."""
        logger.info("Stopping audio processor...")
        self._stop_event.set()
        
        if self._processor_task:
            try:
                await asyncio.wait_for(self._processor_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Audio processor task did not stop gracefully, cancelling.")
                self._processor_task.cancel()
            except asyncio.CancelledError:
                pass
        self.audio_buffer.reset()
        logger.info("Audio processor stopped.")

    async def _run_processing_loop(self):
        """The main loop for receiving, buffering, and processing audio."""
        try:
            while not self._stop_event.is_set():
                if self._ai_is_speaking or self.output_track.is_playing():
                    try:
                        await asyncio.wait_for(self.input_track.recv(), timeout=0.005)
                        await asyncio.sleep(0.005)
                    except asyncio.TimeoutError:
                        pass
                    except mediastreams.MediaStreamError:
                        logger.warning("Client audio stream ended while AI was speaking.")
                        break
                    except Exception as e:
                        logger.error(f"Error receiving audio while AI speaking: {e}")
                    continue

                try: 
                    frame = await asyncio.wait_for(self.input_track.recv(), timeout=0.1)
                except asyncio.TimeoutError:
                    await asyncio.sleep(0.01)
                    continue
                except mediastreams.MediaStreamError: 
                    logger.warning("Client audio stream ended.")
                    break
                except Exception as e:
                    logger.error(f"Error receiving audio frame: {e}", exc_info=True)
                    await asyncio.sleep(0.01)
                    continue

                audio_data_raw = frame.to_ndarray()
                if audio_data_raw.size == 0:
                    continue

                if frame.sample_rate != SAMPLE_RATE_VAD:
                    try:
                        if audio_data_raw.dtype != np.float32:
                            audio_data_raw = audio_data_raw.astype(np.float32)
                        
                        max_val_raw = np.abs(audio_data_raw).max()
                        if max_val_raw > 1.0:
                            audio_data_raw = audio_data_raw / max_val_raw
                        elif max_val_raw > 0 and max_val_raw < 0.2:
                            audio_data_raw = audio_data_raw * min(2.0, 0.4 / max_val_raw)

                        resampled_audio = librosa.resample(
                            audio_data_raw.flatten(), 
                            orig_sr=frame.sample_rate, 
                            target_sr=SAMPLE_RATE_VAD,
                            res_type='kaiser_fast'
                        )
                    except Exception as e:
                        logger.error(f"Resampling error: {e}. Dropping frame.", exc_info=True)
                        continue
                else:
                    resampled_audio = audio_data_raw.flatten()
                
                self.audio_buffer.add_audio(resampled_audio)
                
                if self.audio_buffer.should_process_now() and not self._processing_lock.locked():
                    audio_to_process = self.audio_buffer.get_audio_array()
                    self.audio_buffer.reset()
                    
                    logger.info(f"🧠 Processing {len(audio_to_process)} samples (max: {np.abs(audio_to_process).max():.4f})")
                    asyncio.create_task(self._process_speech_segment(audio_to_process))
                        
        except asyncio.CancelledError: 
            logger.info("Audio processing loop cancelled.")
        except Exception as e: 
            logger.error(f"Unhandled exception in audio processing loop: {e}", exc_info=True)
        finally: 
            logger.info("Audio processor loop finished.")

    def _blocking_tts_generation(self, text: str) -> np.ndarray:
        """
        Generates TTS audio for a given text. This is a blocking call and should be
        run in an executor thread.
        """
        try:
            if len(text) > 200:
                text = text[:200] + "..."
                logger.info(f"AI Response text truncated to 200 characters.")
            
            logger.info(f"🗣️ Generating TTS for: '{text[:50]}...'")
            
            with torch.inference_mode():
                wav = tts_model.generate(text).cpu().numpy().flatten()
                
                if wav.size == 0:
                    logger.warning("TTS generation returned empty audio.")
                    return np.array([], dtype=np.float32)
                
                max_samples_tts_input = int(TTS_MAX_DURATION_SEC * SAMPLE_RATE_TTS_INPUT)
                if wav.size > max_samples_tts_input:
                    wav = wav[:max_samples_tts_input]
                    logger.info(f"TTS audio truncated to {TTS_MAX_DURATION_SEC} seconds.")
                
                resampled_wav = librosa.resample(
                    wav.astype(np.float32), 
                    orig_sr=SAMPLE_RATE_TTS_INPUT, 
                    target_sr=SAMPLE_RATE_OUTPUT,
                    res_type='kaiser_best'
                )
                
                logger.info(f"✅ TTS generated {resampled_wav.size} samples ({resampled_wav.size/SAMPLE_RATE_OUTPUT:.2f}s).")
                return resampled_wav
                
        except Exception as e:
            logger.error(f"TTS generation failed: {e}", exc_info=True)
            return np.array([], dtype=np.float32)

    async def _process_speech_segment(self, audio_array: np.ndarray):
        """
        Processes a segment of audio: sends to Ultravox, gets response,
        generates TTS, and queues it for playback.
        """
        async with self._processing_lock:
            if self._is_processing_speech:
                return
            self._is_processing_speech = True
            
            try:
                # 1. Prepare Audio for Ultravox
                if audio_array.dtype != np.float32:
                    audio_array = audio_array.astype(np.float32)
                
                max_val = np.abs(audio_array).max()
                if max_val > 1.0:
                    audio_array = audio_array / max_val
                elif max_val > 0 and max_val < 0.15:
                    audio_array = audio_array * (0.5 / max_val)
                
                if audio_array.ndim > 1:
                    audio_array = audio_array.flatten()
                
                min_samples_for_ultravox = int(0.8 * SAMPLE_RATE_VAD)
                if len(audio_array) < min_samples_for_ultravox:
                    logger.info(f"Audio segment too short ({len(audio_array)} samples), padding to {min_samples_for_ultravox}.")
                    padding_needed = min_samples_for_ultravox - len(audio_array)
                    audio_array = np.pad(audio_array, (0, padding_needed), 'constant')

                logger.info(f"🎤 Sending audio to Ultravox: length={len(audio_array)} samples, max_norm={np.abs(audio_array).max():.4f}")
                
                # 2. Call Ultravox for Text Generation
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    self.executor, 
                    uv_pipe, 
                    {
                        'audio': audio_array, 
                        'turns': [],
                        'sampling_rate': SAMPLE_RATE_VAD
                    },
                    max_new_tokens=ULTRAVOX_MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=0.5,
                    top_p=0.8
                )

                response_text = parse_ultravox_response(result)
                if not response_text:
                    logger.warning("Ultravox returned empty response.")
                    return

                logger.info(f"🤖 AI Response: '{response_text}'")

                # 3. Generate TTS Audio
                tts_audio_data = await loop.run_in_executor(
                    self.executor, 
                    self._blocking_tts_generation, 
                    response_text
                )

                # 4. Queue TTS Audio for Playback
                if tts_audio_data.size > 0:
                    async with self._speaking_lock:
                        self._ai_is_speaking = True
                    
                    logger.info("🤖 AI is speaking...")
                    await self.output_track.queue_audio(tts_audio_data)
                    
                    await self.output_track.wait_for_completion(timeout=TTS_MAX_DURATION_SEC + 2.0)

                    logger.info("✅ AI finished speaking, now listening.")
                else:
                    logger.warning("No TTS audio data to play.")
            
            except Exception as e:
                logger.error(f"Error during speech processing segment: {e}", exc_info=True)
            finally:
                async with self._speaking_lock:
                    self._ai_is_speaking = False
                self._is_processing_speech = False

# --- WebRTC and WebSocket Handling ---
async def websocket_handler(request):
    """Handles WebSocket connections for WebRTC signaling."""
    ws = web.WebSocketResponse(heartbeat=30, timeout=60)
    await ws.prepare(request)
    
    pc = RTCPeerConnection(RTCConfiguration([
        RTCIceServer(urls="stun:stun.l.google.com:19302")
    ]))
    pcs.add(pc)
    
    processor = None
    audio_track_to_send = None

    @pc.on("track")
    def on_track(track):
        """Callback when a media track is received from the client."""
        nonlocal processor, audio_track_to_send
        logger.info(f"🎧 Track {track.kind} received from client.")
        
        if track.kind == "audio":
            audio_track_to_send = ResponseAudioTrack()
            pc.addTrack(audio_track_to_send)
            
            processor = AudioProcessor(audio_track_to_send, executor)
            processor.add_input_track(track)
            asyncio.create_task(processor.start())

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        """Callback when the WebRTC connection state changes."""
        logger.info(f"ICE Connection State is {pc.connectionState}")
        if pc.connectionState in ["failed", "closed", "disconnected"]:
            logger.warning(f"WebRTC connection state is {pc.connectionState}. Cleaning up.")
            if processor:
                await processor.stop()
            if pc in pcs: 
                pcs.discard(pc)
            
            if not ws.closed:
                await ws.close()

    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    msg_type = data.get("type")

                    if msg_type == "offer":
                        logger.info("Received offer from client.")
                        await pc.setRemoteDescription(RTCSessionDescription(sdp=data["sdp"], type=msg_type))
                        answer = await pc.createAnswer()
                        await pc.setLocalDescription(answer)
                        await ws.send_json({"type": "answer", "sdp": pc.localDescription.sdp})
                    
                    elif msg_type == "ice-candidate":
                        logger.info("Received ICE candidate from client.")
                        # `data` is the parsed JSON. `data.candidate` should be the object
                        # containing the actual ICE string and metadata.
                        candidate_data = data.get("candidate") 
                        
                        # IMPORTANT FIX: Check if `candidate_data` is a dictionary AND if it contains the 'candidate' key (the ICE string itself)
                        if isinstance(candidate_data, dict) and 'candidate' in candidate_data:
                            try:
                                # Construct RTCIceCandidate using the correct properties from the dictionary
                                candidate = RTCIceCandidate(
                                    candidate_data.get("candidate"), # This is the actual ICE candidate string
                                    sdpMid=candidate_data.get("sdpMid"),
                                    sdpMLineIndex=candidate_data.get("sdpMLineIndex"),
                                    usernameFragment=candidate_data.get("usernameFragment")
                                )
                                await pc.addIceCandidate(candidate)
                            except Exception as e:
                                logger.error(f"Failed to create and add RTCIceCandidate: {e}. Data: {candidate_data}", exc_info=True)
                        else:
                            logger.error(f"Received ICE candidate message with invalid or missing 'candidate' data. Data: {candidate_data}")

                except json.JSONDecodeError:
                    logger.error(f"Received malformed JSON from WebSocket: {msg.data}")
                except Exception as e:
                    logger.error(f"Error processing WebSocket message: {e}", exc_info=True)

            elif msg.type == WSMsgType.ERROR:
                logger.error(f"WebSocket error: {ws.exception()}")
                break
            elif msg.type == WSMsgType.CLOSE:
                logger.info("WebSocket close frame received.")
                break

    except Exception as e:
        logger.error(f"Error in WebSocket handler: {e}", exc_info=True)
    finally:
        logger.info("WebSocket handler finished. Cleaning up.")
        if processor: 
            await processor.stop()
        if pc in pcs: 
            pcs.discard(pc)
        if pc.connectionState != "closed": 
            await pc.close()
        if not ws.closed:
            await ws.close()

    return ws

async def index_handler(request):
    """Serves the HTML client interface."""
    return web.Response(text=HTML_CLIENT, content_type='text/html')

async def on_shutdown(app):
    """Gracefully shuts down the server and all associated resources."""
    logger.info("Shutting down server and cleaning up resources...")
    
    pcs_list = list(pcs)
    for pc_conn in pcs_list: 
        try:
            await pc_conn.close()
        except Exception as e:
            logger.error(f"Error closing peer connection during shutdown: {e}")
    pcs.clear()
    
    logger.info("Shutting down thread pool executor...")
    executor.shutdown(wait=True)
    logger.info("Executor shut down complete.")
    logger.info("Shutdown complete.")

async def main():
    """Main function to initialize models and start the web server."""
    if not initialize_models():
        logger.critical("Failed to initialize models. Application cannot start.")
        return

    app = web.Application()
    app.router.add_get('/', index_handler)
    app.router.add_get('/ws', websocket_handler)
    app.on_shutdown.append(on_shutdown)
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 7860)
    
    await site.start()
    
    print("\n" + "*"*50)
    print("✅ Server started successfully on http://0.0.0.0:7860")
    print("🚀 Your speech-to-speech agent is live!")
    print("   Press Ctrl+C to stop the server.")
    print("*"*50 + "\n")
    
    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        pass
    except KeyboardInterrupt:
        print("\n🛑 Keyboard interrupt detected. Initiating server shutdown...")
    finally:
        await runner.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Server shutting down by user request...")
