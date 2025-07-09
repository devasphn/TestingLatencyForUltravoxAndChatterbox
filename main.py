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
executor = ThreadPoolExecutor(max_workers=4) # Reduced workers to prevent overload if needed
pcs = weakref.WeakSet() # To keep track of active peer connections

# --- Constants ---
SAMPLE_RATE_VAD = 16000
SAMPLE_RATE_TTS_INPUT = 24000 # Chatterbox input sample rate
SAMPLE_RATE_OUTPUT = 48000 # WebRTC audio output sample rate
VAD_MIN_SPEECH_DURATION_MS = 250
VAD_MIN_SILENCE_DURATION_MS = 150
AUDIO_PROCESS_INTERVAL_MS = 400 # How often to check buffer for processing (in ms)
AUDIO_MAX_INPUT_DURATION_SEC = 5.0 # Max duration of audio to send to Ultravox at once
TTS_MAX_DURATION_SEC = 10.0 # Max duration of TTS output
ULTRAVOX_MAX_NEW_TOKENS = 80 # Conservative token limit for Ultravox
ULTRAVOX_AUDIO_CHUNK_SIZE_SAMPLES = int(SAMPLE_RATE_VAD * 0.5) # Process audio chunks of 0.5s

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
    let isSpeaking = false; // Track if AI is speaking for UI feedback

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

            // Request higher quality audio if available and necessary
            localStream = await navigator.mediaDevices.getUserMedia({ 
                audio: { 
                    echoCancellation: false,
                    noiseSuppression: false,
                    autoGainControl: false,
                    sampleRate: 48000, // Request 48kHz for better input quality
                    channelCount: 1,
                    latency: { ideal: 0.025, min: 0.010, max: 0.050 } // Lower latency
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
                    
                    // More robustly detect audio end
                    remoteAudio.onended = () => {
                        if(isSpeaking) { // Only change status if it was speaking
                            isSpeaking = false;
                            // Wait a moment for potential continuation or silence
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
                    reconnectAttempts = 0; // Reset reconnect attempts on successful connection
                } else if (state === 'failed' || state === 'closed' || state === 'disconnected') {
                    stop(); // Stop everything on connection loss
                }
            };

            const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${location.host}/ws`);
            ws.binaryType = 'arraybuffer'; // Expect binary data for audio frames

            ws.onopen = async () => {
                console.log('WebSocket connected');
                const offer = await pc.createOffer();
                await pc.setLocalDescription(offer);
                ws.send(JSON.stringify({type: offer.type, sdp: offer.sdp}));
            };

            ws.onmessage = async e => {
                try {
                    const data = JSON.parse(e.data);
                    if (data.type === 'answer' && !pc.currentRemoteDescription) {
                        await pc.setRemoteDescription(new RTCSessionDescription(data));
                    }
                } catch (err) {
                    console.error('WebSocket message error:', err);
                    if (err instanceof SyntaxError) { // Handle malformed JSON
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

    // Prevent page unload issues
    window.addEventListener('beforeunload', stop);
</script>
</body>
</html>
"""

# --- Utility Functions ---
def candidate_from_sdp(candidate_data: dict) -> RTCIceCandidate:
    """Parses ICE candidate from JSON data to an RTCIceCandidate object."""
    try:
        candidate = RTCIceCandidate(
            component=candidate_data.get("component"),
            foundation=candidate_data.get("foundation"),
            ip=candidate_data.get("ip"),
            port=candidate_data.get("port"),
            priority=candidate_data.get("priority"),
            protocol=candidate_data.get("protocol"),
            type=candidate_data.get("type"),
            relatedAddress=candidate_data.get("relatedAddress"),
            relatedPort=candidate_data.get("relatedPort"),
            sdpMid=candidate_data.get("sdpMid"),
            sdpMLineIndex=candidate_data.get("sdpMLineIndex")
        )
        return candidate
    except Exception as e:
        logger.error(f"Error parsing ICE candidate: {e} - Data: {candidate_data}")
        return None

def parse_ultravox_response(result):
    """Safely parse Ultravox pipeline output."""
    try:
        if isinstance(result, str):
            return result.strip()
        elif isinstance(result, list) and len(result) > 0:
            item = result[0]
            if isinstance(item, str):
                return item.strip()
            elif isinstance(item, dict):
                return item.get('generated_text', item.get('text', '')).strip()
        elif isinstance(result, dict):
            return result.get('generated_text', result.get('text', '')).strip()
        return ""
    except Exception as e:
        logger.error(f"Error parsing Ultravox response: {e}")
        return ""

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
                force_reload=False, # Set to True if you suspect cache issues
                onnx=False
            )
            # Ensure model is on the correct device if GPU is available
            if torch.cuda.is_available():
                self.model.to("cuda:0")
            logger.info("✅ Silero VAD loaded successfully")
        except Exception as e:
            logger.error(f"❌ Error loading Silero VAD: {e}", exc_info=True)

    def detect_speech(self, audio_tensor: torch.Tensor, sample_rate: int) -> bool:
        """Detects speech in an audio tensor using Silero VAD."""
        if self.model is None:
            return True # Assume speech if VAD failed to load

        try:
            # Ensure input is float32 and normalized to [-1, 1]
            if audio_tensor.dtype != torch.float32:
                audio_tensor = audio_tensor.float()
            
            max_val = audio_tensor.abs().max()
            if max_val > 1.0:
                audio_tensor = audio_tensor / max_val
            
            # If audio is very quiet, consider it silence
            if max_val < 0.005: # Lowered threshold for very quiet signals
                return False

            # Silero VAD expects a tensor of shape (batch_size, num_samples)
            if audio_tensor.ndim == 1:
                audio_tensor = audio_tensor.unsqueeze(0)

            speech_timestamps = self.utils[0]( # get_speech_timestamps
                audio_tensor,
                self.model,
                sampling_rate=sample_rate,
                min_speech_duration_ms=VAD_MIN_SPEECH_DURATION_MS,
                min_silence_duration_ms=VAD_MIN_SILENCE_DURATION_MS,
                return_seconds=True # Returning seconds can be easier for some debugging
            )
            
            # If any speech segment is detected, return True
            return len(speech_timestamps) > 0
        except Exception as e:
            logger.error(f"VAD detection error: {e}", exc_info=True)
            return True # Be lenient on errors and assume speech if processing fails

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
        # Using attn_implementation="eager" to avoid potential issues with flash attention if not fully compatible
        uv_pipe = pipeline(
            model="fixie-ai/ultravox-v0_4",
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16, # Use float16 for memory efficiency
            attn_implementation="eager" # Explicitly use eager attention
        )
        logger.info("✅ Ultravox pipeline loaded successfully")

        logger.info("📥 Loading Chatterbox TTS...")
        # Ensure TTS model is loaded on the same device as other models if possible
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
        self.min_speech_samples_for_vad = int(0.8 * sample_rate) # Minimum audio length to run VAD
        self.process_interval_sec = AUDIO_PROCESS_INTERVAL_MS / 1000.0 # Interval to check buffer
        self.silence_threshold_norm = 0.005 # Normalized silence threshold

    def add_audio(self, audio_data: np.ndarray):
        """Adds raw audio data (expected to be normalized float32)."""
        if audio_data.size == 0:
            return

        # Ensure input is float32
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        # Normalize audio to [-1, 1] range
        max_val = np.abs(audio_data).max()
        if max_val > 1.0:
            audio_data = audio_data / max_val
        elif max_val > 0 and max_val < 0.2: # Amplify quiet signals slightly
            audio_data = audio_data * min(2.0, 0.4 / max_val)
        elif max_val == 0: # If all zeros, return early
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
        
        # Check if enough time has passed since the last check
        if (current_time - self.last_process_check_time) < self.process_interval_sec:
            return False
        
        self.last_process_check_time = current_time

        buffer_len = len(self.buffer)

        # Only run VAD if there's a minimum amount of audio
        if buffer_len < self.min_speech_samples_for_vad:
            return False

        audio_array = self.get_audio_array()
        
        # Check for silence before running VAD
        if np.abs(audio_array).max() < self.silence_threshold_norm:
            return False

        # Finally, run VAD
        return vad_model.detect_speech(torch.from_numpy(audio_array), self.sample_rate)

class ResponseAudioTrack(MediaStreamTrack):
    """
    MediaStreamTrack that sends audio frames to the client.
    Handles buffering and sending of TTS output.
    """
    kind = "audio"
    
    def __init__(self):
        super().__init__()
        self._audio_queue = asyncio.Queue() # Queue for incoming audio chunks
        self._current_chunk = None
        self._chunk_pos = 0
        self._timestamp = 0
        self._time_base = fractions.Fraction(1, SAMPLE_RATE_OUTPUT)
        self._frame_samples = 960  # 20ms at 48kHz
        self._total_samples_to_send = 0
        self._samples_sent_in_chunk = 0
        self._is_active = False # Flag indicating if there's audio to send

    async def recv(self):
        """Receives an audio frame to send to the client."""
        # Prepare a mono frame of int16 audio data
        frame_data = np.zeros(self._frame_samples, dtype=np.int16)
        
        samples_copied = 0
        while samples_copied < self._frame_samples:
            if self._current_chunk is None or self._chunk_pos >= len(self._current_chunk):
                try:
                    # Get the next chunk of audio from the queue
                    self._current_chunk = await asyncio.wait_for(self._audio_queue.get(), timeout=0.01)
                    self._chunk_pos = 0
                    if not self._is_active and len(self._current_chunk) > 0:
                        self._is_active = True # Mark as active when we get the first non-empty chunk
                except asyncio.TimeoutError:
                    break # No more audio data available for now
            
            if self._current_chunk is not None:
                # Determine how many samples to copy from the current chunk
                samples_available_in_chunk = len(self._current_chunk) - self._chunk_pos
                samples_needed_for_frame = self._frame_samples - samples_copied
                
                copy_count = min(samples_available_in_chunk, samples_needed_for_frame)
                
                frame_data[samples_copied : samples_copied + copy_count] = \
                    self._current_chunk[self._chunk_pos : self._chunk_pos + copy_count]
                
                samples_copied += copy_count
                self._chunk_pos += copy_count
                self._samples_sent_in_chunk += copy_count
        
        # If no samples were copied, it means the queue is empty and we should stop.
        if samples_copied == 0:
            self._is_active = False # Mark as inactive if we couldn't get any data
            return None # Signal end of stream for this track

        # Create an av.AudioFrame
        audio_frame = av.AudioFrame.from_ndarray(
            frame_data.reshape(1, -1), # Reshape for av (channels, samples)
            format="s16", 
            layout="mono"
        )
        audio_frame.pts = self._timestamp
        audio_frame.sample_rate = SAMPLE_RATE_OUTPUT
        audio_frame.time_base = self._time_base
        
        self._timestamp += self._frame_samples # Increment timestamp for the next frame
        return audio_frame

    async def queue_audio(self, audio_float32: np.ndarray):
        """Queues audio data (float32) for streaming to the client."""
        if audio_float32.size == 0:
            logger.warning("Attempted to queue empty audio data.")
            return
            
        # Clip and convert to int16 for WebRTC
        # Scale to the range of int16 (-32768 to 32767)
        audio_int16 = np.clip(audio_float32 * 32767, -32767, 32767).astype(np.int16)
        
        # Reset state for new audio
        self._total_samples_to_send = len(audio_int16)
        self._samples_sent_in_chunk = 0
        self._is_active = False # Will be set to True when the first chunk is retrieved
        
        # Split into chunks for smoother streaming and manage queue size
        chunk_size = 1920  # ~40ms chunks at 48kHz
        for i in range(0, len(audio_int16), chunk_size):
            chunk = audio_int16[i:i + chunk_size]
            await self._audio_queue.put(chunk)
        
        # Add an empty chunk to signal the end of the audio stream
        await self._audio_queue.put(np.array([], dtype=np.int16))

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
        # Give a small buffer for the last frame to be processed
        await asyncio.sleep(0.1)

class AudioProcessor:
    """
    Processes incoming audio frames, detects speech, sends to Ultravox,
    gets response, generates TTS, and streams it back via ResponseAudioTrack.
    """
    def __init__(self, output_track: ResponseAudioTrack, executor: ThreadPoolExecutor):
        self.input_track = None # The incoming audio stream track
        self.output_track = output_track # The track to send TTS audio
        self.audio_buffer = AudioBuffer(SAMPLE_RATE_VAD)
        self._processor_task = None
        self.executor = executor # For blocking model calls
        self._is_processing_speech = False # Flag to prevent concurrent speech processing
        self._ai_is_speaking = False # Flag to indicate if TTS is playing
        self._processing_lock = asyncio.Lock() # Protects _is_processing_speech
        self._speaking_lock = asyncio.Lock() # Protects _ai_is_speaking
        self._stop_event = asyncio.Event() # Event to signal stopping the processor

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
        self._stop_event.set() # Signal the loop to stop
        
        if self._processor_task:
            try:
                # Wait for the task to finish gracefully
                await asyncio.wait_for(self._processor_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Audio processor task did not stop gracefully, cancelling.")
                self._processor_task.cancel() # Force cancel if it hangs
            except asyncio.CancelledError:
                pass # Task was already cancelled
        self.audio_buffer.reset()
        logger.info("Audio processor stopped.")

    async def _run_processing_loop(self):
        """The main loop for receiving, buffering, and processing audio."""
        try:
            while not self._stop_event.is_set():
                # If the AI is speaking or the output track is actively playing,
                # we should prioritize draining the input without processing.
                if self._ai_is_speaking or self.output_track.is_playing():
                    try:
                        # Attempt to receive a frame, but with a very small timeout
                        # to avoid blocking indefinitely if the client stops sending.
                        await asyncio.wait_for(self.input_track.recv(), timeout=0.005)
                        await asyncio.sleep(0.005) # Small sleep to yield control
                    except asyncio.TimeoutError:
                        pass # No new frame, just continue
                    except mediastreams.MediaStreamError:
                        logger.warning("Client audio stream ended unexpectedly while AI was speaking.")
                        break # Client disconnected
                    except Exception as e:
                        logger.error(f"Error receiving audio while AI speaking: {e}")
                    continue # Skip processing, wait for AI to finish

                # Attempt to receive an audio frame from the client
                try: 
                    frame = await asyncio.wait_for(self.input_track.recv(), timeout=0.1) # Shorter timeout when not speaking
                except asyncio.TimeoutError:
                    # No frame received within timeout, continue loop
                    await asyncio.sleep(0.01) # Yield control
                    continue
                except mediastreams.MediaStreamError: 
                    logger.warning("Client audio stream ended.")
                    break # Client disconnected
                except Exception as e:
                    logger.error(f"Error receiving audio frame: {e}", exc_info=True)
                    await asyncio.sleep(0.01) # Yield control
                    continue

                # Process the received audio frame
                audio_data_raw = frame.to_ndarray()
                if audio_data_raw.size == 0:
                    continue # Skip empty frames

                # Resample if necessary to VAD's expected sample rate (16kHz)
                if frame.sample_rate != SAMPLE_RATE_VAD:
                    try:
                        # Ensure raw data is float32 before resampling
                        if audio_data_raw.dtype != np.float32:
                            audio_data_raw = audio_data_raw.astype(np.float32)
                        
                        # Normalize if needed before resampling (resampling can amplify)
                        max_val_raw = np.abs(audio_data_raw).max()
                        if max_val_raw > 1.0:
                            audio_data_raw = audio_data_raw / max_val_raw
                        elif max_val_raw > 0 and max_val_raw < 0.2:
                            audio_data_raw = audio_data_raw * min(2.0, 0.4 / max_val_raw)

                        resampled_audio = librosa.resample(
                            audio_data_raw.flatten(), 
                            orig_sr=frame.sample_rate, 
                            target_sr=SAMPLE_RATE_VAD,
                            res_type='kaiser_fast' # Efficient resampling
                        )
                    except Exception as e:
                        logger.error(f"Resampling error: {e}. Dropping frame.", exc_info=True)
                        continue
                else:
                    resampled_audio = audio_data_raw.flatten()
                
                # Add the processed audio to the buffer
                self.audio_buffer.add_audio(resampled_audio)
                
                # Check if the buffer is ready for speech processing
                if self.audio_buffer.should_process_now() and not self._processing_lock.locked():
                    # Get the audio chunk to process
                    audio_to_process = self.audio_buffer.get_audio_array()
                    self.audio_buffer.reset() # Clear buffer for new input
                    
                    logger.info(f"🧠 Processing {len(audio_to_process)} samples (max: {np.abs(audio_to_process).max():.4f})")
                    # Start speech processing as a separate task to avoid blocking the loop
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
            # Strict text length limit to avoid overly long responses
            if len(text) > 200:
                text = text[:200] + "..." # Truncate text
                logger.info(f"AI Response text truncated to 200 characters.")
            
            logger.info(f"🗣️ Generating TTS for: '{text[:50]}...'")
            
            with torch.inference_mode():
                # Chatterbox TTS generates at 24kHz
                wav = tts_model.generate(text).cpu().numpy().flatten()
                
                if wav.size == 0:
                    logger.warning("TTS generation returned empty audio.")
                    return np.array([], dtype=np.float32)
                
                # Strict audio length limit for TTS output
                max_samples_tts_input = int(TTS_MAX_DURATION_SEC * SAMPLE_RATE_TTS_INPUT)
                if wav.size > max_samples_tts_input:
                    wav = wav[:max_samples_tts_input]
                    logger.info(f"TTS audio truncated to {TTS_MAX_DURATION_SEC} seconds.")
                
                # Resample TTS output to 48kHz for WebRTC streaming
                resampled_wav = librosa.resample(
                    wav.astype(np.float32), 
                    orig_sr=SAMPLE_RATE_TTS_INPUT, 
                    target_sr=SAMPLE_RATE_OUTPUT,
                    res_type='kaiser_best' # High-quality resampling
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
        This function is designed to be called concurrently but is protected
        by a lock to ensure only one segment is processed at a time.
        """
        async with self._processing_lock: # Acquire lock to prevent concurrent processing
            if self._is_processing_speech: # Double check if another task started
                return
            self._is_processing_speech = True
            
            try:
                # --- 1. Prepare Audio for Ultravox ---
                # Ensure audio array is float32 and normalized to [-1, 1]
                if audio_array.dtype != np.float32:
                    audio_array = audio_array.astype(np.float32)
                
                max_val = np.abs(audio_array).max()
                if max_val > 1.0:
                    audio_array = audio_array / max_val
                elif max_val > 0 and max_val < 0.15: # Amplify slightly quiet signals
                    audio_array = audio_array * (0.5 / max_val)
                
                if audio_array.ndim > 1:
                    audio_array = audio_array.flatten()
                
                # Pad audio to meet Ultravox minimum input requirements if too short
                # Ultravox might expect a certain number of samples for its internal processing.
                # Let's ensure at least 0.8s of audio (12800 samples at 16kHz)
                min_samples_for_ultravox = int(0.8 * SAMPLE_RATE_VAD)
                if len(audio_array) < min_samples_for_ultravox:
                    logger.info(f"Audio segment too short ({len(audio_array)} samples), padding to {min_samples_for_ultravox}.")
                    padding_needed = min_samples_for_ultravox - len(audio_array)
                    audio_array = np.pad(audio_array, (0, padding_needed), 'constant')

                # Log input details for debugging
                logger.info(f"🎤 Sending audio to Ultravox: length={len(audio_array)} samples, max_norm={np.abs(audio_array).max():.4f}")
                
                # --- 2. Call Ultravox for Text Generation ---
                # Use the executor for potentially blocking pipeline calls
                # Pass audio as numpy array
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    self.executor, 
                    uv_pipe, 
                    {
                        'audio': audio_array, 
                        'turns': [], # No history for this example
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
                    return # Exit if no text was generated

                logger.info(f"🤖 AI Response: '{response_text}'")

                # --- 3. Generate TTS Audio ---
                # Use the executor for the blocking TTS generation
                tts_audio_data = await loop.run_in_executor(
                    self.executor, 
                    self._blocking_tts_generation, 
                    response_text
                )

                # --- 4. Queue TTS Audio for Playback ---
                if tts_audio_data.size > 0:
                    # Set the AI speaking flag before queueing audio
                    async with self._speaking_lock:
                        self._ai_is_speaking = True
                    
                    logger.info("🤖 AI is speaking...")
                    await self.output_track.queue_audio(tts_audio_data)
                    
                    # Wait for the output track to finish playing the queued audio
                    await self.output_track.wait_for_completion(timeout=TTS_MAX_DURATION_SEC + 2.0) # Give it extra time

                    logger.info("✅ AI finished speaking, now listening.")
                else:
                    logger.warning("No TTS audio data to play.")
            
            except Exception as e:
                logger.error(f"Error during speech processing segment: {e}", exc_info=True)
            finally:
                # Reset flags and release lock
                async with self._speaking_lock:
                    self._ai_is_speaking = False
                self._is_processing_speech = False # Release processing lock

# --- WebRTC and WebSocket Handling ---
async def websocket_handler(request):
    """Handles WebSocket connections for WebRTC signaling."""
    ws = web.WebSocketResponse(heartbeat=30, timeout=60) # Keepalive and timeout
    await ws.prepare(request)
    
    # Create a new Peer Connection for each client
    pc = RTCPeerConnection(RTCConfiguration([
        RTCIceServer(urls="stun:stun.l.google.com:19302") # Public STUN server
    ]))
    pcs.add(pc) # Add to the weak set to track active connections
    
    processor = None # Audio processor instance for this client
    audio_track_to_send = None # The track to send audio back to client

    @pc.on("track")
    def on_track(track):
        """Callback when a media track is received from the client."""
        nonlocal processor, audio_track_to_send
        logger.info(f"🎧 Track {track.kind} received from client.")
        
        if track.kind == "audio":
            # Create an audio track to send synthesized speech back to the client
            audio_track_to_send = ResponseAudioTrack()
            pc.addTrack(audio_track_to_send) # Add this track to the outgoing stream
            
            # Initialize the audio processor with the incoming and outgoing tracks
            processor = AudioProcessor(audio_track_to_send, executor)
            processor.add_input_track(track)
            asyncio.create_task(processor.start()) # Start the processing loop

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        """Callback when the WebRTC connection state changes."""
        logger.info(f"ICE Connection State is {pc.connectionState}")
        if pc.connectionState in ["failed", "closed", "disconnected"]:
            logger.warning(f"WebRTC connection state is {pc.connectionState}. Cleaning up.")
            if processor:
                await processor.stop() # Stop the audio processor
            if pc in pcs: 
                pcs.discard(pc) # Remove from tracking set
            
            # Ensure the WebSocket is closed if the PC closes
            if not ws.closed:
                await ws.close()

    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    msg_type = data.get("type")

                    if msg_type == "offer":
                        # Set the remote description (client's offer)
                        await pc.setRemoteDescription(RTCSessionDescription(sdp=data["sdp"], type=msg_type))
                        # Create and set the local description (our answer)
                        answer = await pc.createAnswer()
                        await pc.setLocalDescription(answer)
                        # Send the answer back to the client
                        await ws.send_json({"type": "answer", "sdp": pc.localDescription.sdp})
                    
                    elif msg_type == "ice-candidate" and data.get("candidate"):
                        # Add the ICE candidate received from the client
                        candidate_data = data["candidate"]
                        candidate = candidate_from_sdp(candidate_data)
                        if candidate:
                            await pc.addIceCandidate(candidate)
                        else:
                            logger.warning(f"Failed to parse ICE candidate: {candidate_data}")

                except json.JSONDecodeError:
                    logger.error(f"Received malformed JSON from WebSocket: {msg.data}")
                except Exception as e:
                    logger.error(f"Error processing WebSocket message: {e}", exc_info=True)

            elif msg.type == WSMsgType.ERROR:
                logger.error(f"WebSocket error: {ws.exception()}")
                break # Exit loop on error
            elif msg.type == WSMsgType.CLOSE:
                logger.info("WebSocket close frame received.")
                break # Exit loop on close frame

    except Exception as e:
        logger.error(f"Error in WebSocket handler: {e}", exc_info=True)
    finally:
        logger.info("WebSocket handler finished. Cleaning up.")
        # Ensure all resources are cleaned up when the handler exits
        if processor: 
            await processor.stop()
        if pc in pcs: 
            pcs.discard(pc)
        if pc.connectionState != "closed": 
            await pc.close() # Close the peer connection
        if not ws.closed:
            await ws.close() # Ensure WebSocket is closed

    return ws

async def index_handler(request):
    """Serves the HTML client interface."""
    return web.Response(text=HTML_CLIENT, content_type='text/html')

async def on_shutdown(app):
    """Gracefully shuts down the server and all associated resources."""
    logger.info("Shutting down server and cleaning up resources...")
    
    # Close all active peer connections
    pcs_list = list(pcs) # Create a copy to avoid modifying while iterating
    for pc_conn in pcs_list: 
        try:
            await pc_conn.close()
        except Exception as e:
            logger.error(f"Error closing peer connection during shutdown: {e}")
    pcs.clear() # Clear the set
    
    # Shutdown the thread pool executor
    logger.info("Shutting down thread pool executor...")
    executor.shutdown(wait=True)
    logger.info("Executor shut down complete.")
    logger.info("Shutdown complete.")

async def main():
    """Main function to initialize models and start the web server."""
    if not initialize_models():
        logger.critical("Failed to initialize models. Application cannot start.")
        return

    # Create the web application
    app = web.Application()
    app.router.add_get('/', index_handler) # Route for the main HTML page
    app.router.add_get('/ws', websocket_handler) # Route for WebSocket signaling
    app.on_shutdown.append(on_shutdown) # Register shutdown handler
    
    # Setup the web server runner
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 7860) # Listen on all interfaces, port 7860
    
    # Start the server
    await site.start()
    
    print("\n" + "*"*50)
    print("✅ Server started successfully on http://0.0.0.0:7860")
    print("🚀 Your speech-to-speech agent is live!")
    print("   Press Ctrl+C to stop the server.")
    print("*"*50 + "\n")
    
    try:
        # Keep the server running indefinitely until interrupted
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        # This can happen if the shutdown signal is caught
        pass
    except KeyboardInterrupt:
        print("\n🛑 Keyboard interrupt detected. Initiating server shutdown...")
    finally:
        # Ensure runner is cleaned up
        await runner.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # This catch is useful if the signal is received before asyncio.run starts
        print("\n🛑 Server shutting down by user request...")
