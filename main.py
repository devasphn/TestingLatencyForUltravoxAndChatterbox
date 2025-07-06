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

from aiohttp import web, WSMsgType
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack, RTCIceCandidate, RTCConfiguration, RTCIceServer, mediastreams
import av

# --- Ensure these libraries are installed ---
# pip install torch torchvision torchaudio numpy librosa aiohttp aiortc av transformers chatterbox-tts
# For uvloop (optional but recommended): pip install uvloop

# If you plan to use ONNX Runtime for optimized inference:
# pip install onnxruntime onnxruntime-gpu # (if using GPU)

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

# Reduce verbosity for common libraries to focus on application logs
logging.getLogger('aioice.ice').setLevel(logging.WARNING)
logging.getLogger('aiortc').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('websockets').setLevel(logging.WARNING)
logging.getLogger('asyncio').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('charset_normalizer').setLevel(logging.WARNING)
logging.getLogger('safetensors.safetensors_ப்பட்டன').setLevel(logging.WARNING)


# --- Global Variables ---
uv_pipe = None
tts_model = None
vad_model = None
executor = ThreadPoolExecutor(max_workers=2) # Reduced workers, inference is often GPU bound
pcs = set() # Active RTCPeerConnections
ws_clients = set() # Active WebSocket connections

# --- HTML Client ---
HTML_CLIENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>UltraChat Voice Assistant</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        body {
            font-family: 'Inter', sans-serif;
            margin: 0;
            padding: 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #fff;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            box-sizing: border-box;
            overflow-y: auto; /* Allow scrolling if content exceeds viewport */
        }
        .container {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            padding: 40px;
            text-align: center;
            max-width: 650px;
            width: 100%;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        h1 {
            font-size: 2.5em;
            margin-bottom: 20px;
            font-weight: 600;
            letter-spacing: -0.5px;
            background: linear-gradient(90deg, #fbc2eb 0%, #a6c1ee 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        p.subtitle {
            font-size: 1.1em;
            margin-bottom: 30px;
            font-weight: 400;
            opacity: 0.9;
        }
        .controls {
            margin-bottom: 30px;
        }
        button {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(45deg, #84fab0 0%, #8fd3f4 100%);
            color: #2c3e50;
            border: none;
            padding: 15px 35px;
            font-size: 1.1em;
            border-radius: 50px;
            cursor: pointer;
            margin: 0 10px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            font-weight: 500;
        }
        button:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.2);
        }
        button:disabled {
            background: #bdc3c7;
            color: #7f8c8d;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        .stop-btn {
            background: linear-gradient(45deg, #ff9a8b 0%, #ff6a88 99%);
            color: white;
        }
        .stop-btn:hover {
            background: linear-gradient(45deg, #ff6a88 0%, #ff9a8b 99%);
        }
        .status-display {
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 25px;
            font-size: 1.1em;
            font-weight: 500;
        }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 10px;
            background-color: #ccc; /* Default */
            animation: pulse 1.5s infinite ease-in-out;
            flex-shrink: 0; /* Prevent shrinking */
        }
        .status-indicator.connected { background-color: #84fab0; animation-play-state: paused;}
        .status-indicator.connecting { background-color: #fbc2eb; animation: pulse 1.5s infinite ease-in-out;}
        .status-indicator.disconnected { background-color: #e74c3c; animation: pulse 1.5s infinite ease-in-out; animation-play-state: paused;}
        .status-indicator.speaking { background-color: #8fd3f4; animation: pulse 1.5s infinite ease-in-out;}

        @keyframes pulse {
            0% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.2); opacity: 0.7; }
            100% { transform: scale(1); opacity: 1; }
        }
        
        .status-area {
            margin-top: 25px;
            text-align: left;
            padding: 20px;
            background: rgba(0, 0, 0, 0.15);
            border-radius: 10px;
            max-height: 300px; /* Limit height */
            overflow-y: auto; /* Add scroll */
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .status-area h3 {
            margin-top: 0;
            font-size: 1.2em;
            border-bottom: 1px solid rgba(255, 255, 255, 0.2);
            padding-bottom: 10px;
            margin-bottom: 15px;
            font-weight: 500;
        }
        .status-log div {
            margin-bottom: 10px;
            font-size: 0.95em;
            opacity: 0.85;
            word-wrap: break-word;
            line-height: 1.4;
            padding-right: 5px; /* Space for potential scrollbar */
        }
        .status-log .log-user { color: #a6c1ee; font-weight: 500; }
        .status-log .log-ai { color: #fbc2eb; font-weight: 500; }
        .status-log .log-system { color: #84fab0; font-weight: 500; }
        .status-log .log-ai-thinking { color: #fbc2eb; font-weight: 500; opacity: 0.7;}

        @media (max-width: 600px) {
            .container { padding: 30px; }
            h1 { font-size: 2em; }
            button { padding: 12px 25px; font-size: 1em; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎙️ UltraChat Assistant</h1>
        <p class="subtitle">Your AI-powered voice companion.</p>
        <div class="controls">
            <button id="startBtn" onclick="start()">Connect</button>
            <button id="stopBtn" onclick="stop()" class="stop-btn" disabled>Disconnect</button>
        </div>
        <div class="status-display">
             <span id="statusIndicator" class="status-indicator disconnected"></span>
             <span id="statusText">🔌 Disconnected</span>
        </div>
        
        <div class="status-area">
            <h3>Conversation Log</h3>
            <div id="statusLog" class="status-log">
                <div>System: Assistant initialized. Waiting for connection.</div>
            </div>
        </div>
        <audio id="remoteAudio" autoplay playsinline></audio>
    </div>

<script>
    let pc, ws, localStream;
    const remoteAudio = document.getElementById('remoteAudio');
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    const statusText = document.getElementById('statusText');
    const statusIndicator = document.getElementById('statusIndicator');
    const statusLog = document.getElementById('statusLog');

    function updateStatus(text, indicatorClass) {
        statusText.textContent = text;
        statusIndicator.className = `status-indicator ${indicatorClass}`;
    }

    function logMessage(message, type = 'system') {
        const logEntry = document.createElement('div');
        logEntry.className = `log-${type}`;
        logEntry.textContent = message;
        statusLog.appendChild(logEntry);
        statusLog.scrollTop = statusLog.scrollHeight; // Auto-scroll to bottom
    }

    async function start() {
        logMessage('Attempting to start connection...', 'system');
        startBtn.disabled = true;
        updateStatus('Connecting...', 'connecting');
        
        try {
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            if (audioContext.state === 'suspended') {
                await audioContext.resume();
                logMessage('AudioContext resumed.', 'system');
            }

            localStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    channelCount: 1, 
                    sampleRate: 16000 
                }
            });
            logMessage('Microphone access granted.', 'system');

            pc = new RTCPeerConnection({
                iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
            });
            logMessage('RTCPeerConnection created.', 'system');

            localStream.getTracks().forEach(track => pc.addTrack(track, localStream));
            logMessage('Local audio track added.', 'system');

            pc.ontrack = (event) => {
                logMessage('Remote audio track received.', 'system');
                if (event.streams && event.streams[0]) {
                    remoteAudio.srcObject = event.streams[0];
                    remoteAudio.play().then(() => {
                        logMessage('AI is speaking...', 'ai');
                        updateStatus('AI is speaking...', 'speaking');
                    }).catch(error => {
                        logMessage(`Autoplay failed: ${error.message}`, 'system');
                        updateStatus('Playback error', 'disconnected');
                    });
                    
                    remoteAudio.onended = () => {
                        if (pc.connectionState === 'connected') {
                            logMessage('AI finished speaking.', 'system');
                            updateStatus('Listening...', 'connected');
                        }
                    };
                } else {
                    logMessage('Received track without stream.', 'system');
                }
            };

            pc.onicecandidate = (event) => {
                if (event.candidate && ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({ type: 'ice-candidate', candidate: event.candidate.toJSON() }));
                }
            };
            
            pc.onconnectionstatechange = () => {
                const state = pc.connectionState;
                logMessage(`Connection state changed: ${state}`, 'system');
                if (state === 'connecting') {
                    updateStatus('Establishing connection...', 'connecting');
                } else if (state === 'connected') {
                    updateStatus('Connected. Listening...', 'connected');
                    stopBtn.disabled = false;
                    logMessage('Connection established successfully!', 'system');
                } else if (state === 'failed' || state === 'closed' || state === 'disconnected') {
                    logMessage(`Connection lost: ${state}`, 'system');
                    stop(); 
                }
            };

            const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${location.host}/ws`);
            logMessage('WebSocket connecting...', 'system');

            ws.onopen = async () => {
                logMessage('WebSocket connected.', 'system');
                updateStatus('Connected. Ready.', 'connected');
                try {
                    const offer = await pc.createOffer();
                    await pc.setLocalDescription(offer);
                    ws.send(JSON.stringify(offer));
                    logMessage('Sent WebRTC offer.', 'system');
                } catch (error) {
                    logMessage(`Error creating offer: ${error.message}`, 'system');
                    stop();
                }
            };

            ws.onmessage = async (event) => {
                const data = JSON.parse(event.data);
                if (data.type === 'answer' && pc.currentRemoteDescription === null) {
                    try {
                        await pc.setRemoteDescription(new RTCSessionDescription(data));
                        logMessage('Received WebRTC answer.', 'system');
                    } catch (error) {
                        logMessage(`Error setting remote description: ${error.message}`, 'system');
                        stop();
                    }
                } else if (data.type === 'transcription') {
                    const { text, subtype } = data; // Use subtype for logging
                    logMessage(text, subtype); // Log with appropriate class (user, ai, system, ai_thinking)
                    if (subtype === 'user' || subtype === 'ai_thinking') {
                         updateStatus('Listening...', 'connected'); // Reset status if user spoke or AI is thinking
                    }
                }
            };

            ws.onclose = () => {
                logMessage('WebSocket disconnected.', 'system');
                stop();
            };

            ws.onerror = (error) => {
                logMessage(`WebSocket error: ${error.message || 'Unknown error'}`, 'system');
                stop();
            };

        } catch (error) {
            logMessage(`Failed to start: ${error.message}`, 'system');
            updateStatus('Connection Failed', 'disconnected');
            stop();
        }
    }

    function stop() {
        logMessage('Disconnecting...', 'system');
        if (ws) {
            ws.onclose = null; 
            ws.onerror = null;
            ws.close(); 
            ws = null;
        }
        if (pc) {
            pc.ontrack = null;
            pc.onicecandidate = null;
            pc.onconnectionstatechange = null;
            if (pc.connectionState !== 'closed') {
                pc.close();
            }
            pc = null;
        }
        if (localStream) {
            localStream.getTracks().forEach(track => track.stop());
            localStream = null;
        }
        updateStatus('Disconnected', 'disconnected');
        startBtn.disabled = false;
        stopBtn.disabled = true;
        logMessage('Connection closed.', 'system');
    }
</script>
</body>
</html>
"""

# --- Utility Functions ---

def candidate_from_sdp(candidate_string: str) -> dict:
    """Parses an ICE candidate string into a dictionary format for aiortc."""
    if candidate_string.startswith("candidate:"): candidate_string = candidate_string[10:]
    bits = candidate_string.split()
    if len(bits) < 8: raise ValueError(f"Invalid candidate string: {candidate_string}")
    params = {'component': int(bits[1]), 'foundation': bits[0], 'ip': bits[4], 'port': int(bits[5]), 'priority': int(bits[3]), 'protocol': bits[2], 'type': bits[7]}
    for i in range(8, len(bits) - 1, 2):
        if bits[i] == "raddr": params['relatedAddress'] = bits[i + 1]
        elif bits[i] == "rport": params['relatedPort'] = int(bits[i + 1])
    return params

def parse_ultravox_response(result):
    """Extracts text from Ultravox pipeline output."""
    try:
        if isinstance(result, str): return result
        elif isinstance(result, list) and len(result) > 0:
            item = result[0]
            if isinstance(item, str): return item
            elif isinstance(item, dict) and 'generated_text' in item: return item['generated_text']
        return ""
    except Exception as e:
        logger.error(f"Error parsing Ultravox response: {e}", exc_info=True)
        return ""

# --- Model Loading ---

class SileroVAD:
    """Voice Activity Detection using Silero VAD model."""
    def __init__(self):
        self.model = None
        self.get_speech_timestamps = None
        try:
            logger.info("🎤 Loading Silero VAD model...")
            # Using PyTorch version. ONNX might be faster but requires export.
            self.model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, onnx=False)
            self.get_speech_timestamps, _, _, _, _ = utils
            logger.info("✅ Silero VAD loaded successfully")
        except Exception as e:
            logger.error(f"❌ Error loading Silero VAD: {e}", exc_info=True)
            # Model remains None if loading fails

    def detect_speech(self, audio_tensor, sample_rate=16000) -> bool:
        """Detects speech in an audio tensor."""
        if self.model is None: 
            # Fallback: If VAD failed to load, assume speech is present to avoid blocking.
            # This will increase latency as silence is processed.
            return True 
        try:
            if isinstance(audio_tensor, np.ndarray):
                audio_tensor = torch.from_numpy(audio_tensor).float()
            
            # Basic check for silence to avoid unnecessary model calls
            if audio_tensor.abs().max() < 0.005: 
                return False

            # Run VAD. `get_speech_timestamps` expects shape (num_samples,)
            # Reduced min_speech_duration_ms for lower latency detection
            speech_timestamps = self.get_speech_timestamps(audio_tensor, self.model, sampling_rate=sample_rate, min_speech_duration_ms=150) 
            return len(speech_timestamps) > 0
        except Exception as e:
            logger.error(f"VAD detection error: {e}", exc_info=True)
            return True # Assume speech on error

def initialize_models():
    """Loads all AI models. Returns False if critical models fail."""
    global uv_pipe, tts_model, vad_model
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"🚀 Initializing models on device: {device}")

    # --- Initialize VAD ---
    vad_model = SileroVAD()
    if vad_model.model is None:
        logger.error("Silero VAD model failed to load. The system may not function correctly without VAD.")
        # Critical failure - return False to prevent server startup
        return False

    # --- Initialize Ultravox (STT + NLU/Generation) ---
    try:
        logger.info("📥 Loading Ultravox pipeline...")
        uv_pipe = pipeline(
            model="fixie-ai/ultravox-v0_4", 
            trust_remote_code=True, 
            device_map="auto", 
            torch_dtype=torch.float16 # Already using FP16, good for latency
        )
        logger.info("✅ Ultravox pipeline loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load Ultravox pipeline: {e}", exc_info=True)
        return False # Critical failure

    # --- Initialize Chatterbox TTS ---
    try:
        logger.info("📥 Loading Chatterbox TTS...")
        # Check ChatterboxTTS documentation for performance tuning options or alternative models.
        tts_model = ChatterboxTTS.from_pretrained(device=device)
        logger.info("✅ Chatterbox TTS loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load Chatterbox TTS: {e}", exc_info=True)
        return False # Critical failure

    logger.info("🎉 All models loaded successfully!")
    return True

# --- Audio Processing Classes ---

class AudioBuffer:
    """Manages audio buffering and triggers processing based on VAD and time."""
    def __init__(self, max_duration=1.5, sample_rate=16000): # Reduced max_duration for lower latency
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)
        self.buffer = collections.deque(maxlen=self.max_samples)
        
        # Parameters for processing trigger
        self.min_processing_samples = int(0.3 * sample_rate) # Minimum samples before considering VAD
        self.process_interval = 0.2  # Minimum time between VAD checks (seconds)
        self.last_process_check_time = time.time()

    def add_audio(self, audio_data: np.ndarray):
        """Adds audio data (float32, expected range -1.0 to 1.0) to the buffer."""
        if audio_data.dtype != np.float32:
            logger.warning(f"Audio data dtype is {audio_data.dtype}, converting to float32.")
            audio_data = audio_data.astype(np.float32)
            # Attempt normalization if it looks like PCM int data
            if np.abs(audio_data).max() > 1.1: # Heuristic for PCM data
                 audio_data /= 32768.0
        
        audio_data = np.clip(audio_data, -1.0, 1.0) # Clamp values
        self.buffer.extend(audio_data.flatten())

    def get_audio_chunk_for_processing(self) -> np.ndarray:
        """Returns the current audio buffer as a NumPy array."""
        return np.array(list(self.buffer), dtype=np.float32)
    
    def should_process(self) -> bool:
        """
        Determines if the buffered audio should be sent for AI processing.
        Triggers if:
        1. Enough time has passed since the last check.
        2. The buffer contains at least `min_processing_samples`.
        3. VAD detects speech (if available).
        """
        current_time = time.time()
        
        if (current_time - self.last_process_check_time) < self.process_interval:
            return False
        
        if len(self.buffer) < self.min_processing_samples:
            return False
            
        audio_chunk = self.get_audio_chunk_for_processing()
        if np.abs(audio_chunk).max() < 0.005: # Threshold for silence
            self.last_process_check_time = current_time # Reset timer even if silent
            return False

        # Check VAD ONLY if the model is loaded
        if vad_model and vad_model.model:
            if vad_model.detect_speech(audio_chunk, self.sample_rate):
                self.last_process_check_time = current_time # Update timer if speech detected
                return True
            else:
                # If VAD detected silence, reset timer but don't process
                # This helps avoid issues if VAD is overly sensitive
                self.last_process_check_time = current_time 
                return False
        else:
            # Fallback: If VAD is not available, process based on buffer size and time
            logger.warning("VAD model not available, processing audio based on buffer size and time.")
            self.last_process_check_time = current_time # Update timer as we are processing
            return True

    def reset(self):
        """Clears the audio buffer."""
        self.buffer.clear()

class ResponseAudioTrack(MediaStreamTrack):
    """Custom MediaStreamTrack to send synthesized audio back to the client."""
    kind = "audio"

    def __init__(self):
        super().__init__()
        self._queue = asyncio.Queue(maxsize=100) # Limit queue size
        self._current_chunk = None
        self._chunk_pos = 0
        self._timestamp = 0
        self._time_base = fractions.Fraction(1, 48000) # WebRTC typically uses 48kHz
        self._frame_samples = 960 # 20ms frames at 48kHz

    async def queue_audio(self, audio_float32: np.ndarray):
        """Adds synthesized audio (float32) to the queue, converting to int16."""
        if audio_float32.size > 0:
            int16_audio = (np.clip(audio_float32, -1.0, 1.0) * 32767).astype(np.int16)
            await self._queue.put(int16_audio)

    async def recv(self) -> av.AudioFrame:
        """Provides the next audio frame for WebRTC."""
        frame = np.zeros(self._frame_samples, dtype=np.int16)
        
        if self._current_chunk is None or self._chunk_pos >= len(self._current_chunk):
            try:
                self._current_chunk = await asyncio.wait_for(self._queue.get(), timeout=0.05)
                self._chunk_pos = 0
            except asyncio.TimeoutError:
                pass # Return silence if queue is empty
        
        if self._current_chunk is not None:
            end_pos = min(self._chunk_pos + self._frame_samples, len(self._current_chunk))
            data_to_copy = self._current_chunk[self._chunk_pos:end_pos]
            frame[:len(data_to_copy)] = data_to_copy
            self._chunk_pos += len(data_to_copy)
        
        audio_frame = av.AudioFrame.from_ndarray(frame.reshape(1, -1), format="s16", layout="mono")
        audio_frame.pts = self._timestamp
        audio_frame.sample_rate = 48000
        
        self._timestamp += self._frame_samples
        return audio_frame

class AudioProcessor:
    """
    Handles the core audio processing pipeline:
    Receives audio -> Buffers -> VAD -> STT/NLU (Ultravox) -> TTS (Chatterbox) -> Sends audio back.
    Includes logic to prevent echo and manage speaking turns.
    """
    def __init__(self, output_track: ResponseAudioTrack, executor: ThreadPoolExecutor):
        self.input_track = None
        self.audio_buffer = AudioBuffer()
        self.output_track = output_track
        self.executor = executor
        self.processing_task = None
        self.is_ai_speaking = False # State: True when AI is generating/playing audio
        self.last_user_audio_time = time.time() # Track when user audio was last processed

        # Parameters for processing
        self.silence_timeout = 1.5 # Seconds of silence after user stops speaking before AI responds
        self.min_response_delay = 0.2 # Minimum delay before AI starts responding, even if user stops abruptly

    def add_input_track(self, track: MediaStreamTrack):
        """Assigns the incoming audio track."""
        self.input_track = track
        logger.info("Input audio track assigned.")

    async def start(self):
        """Starts the main processing loop."""
        if self.input_track is None:
            logger.error("Cannot start AudioProcessor: Input track not assigned.")
            return
        self.processing_task = asyncio.create_task(self._run())
        logger.info("Audio processor started.")

    async def stop(self):
        """Stops the processing loop."""
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                logger.info("Audio processor task cancelled.")
        logger.info("Audio processor stopped.")

    async def _run(self):
        """Main loop for receiving, buffering, and processing audio."""
        try:
            while True:
                # --- Handle AI Speaking State ---
                if self.is_ai_speaking:
                    try:
                        # Consume incoming audio frames without processing to prevent echo
                        await asyncio.wait_for(self.input_track.recv(), timeout=0.01)
                        await asyncio.sleep(0.005) # Yield control
                    except asyncio.TimeoutError: pass # No audio frame received
                    except mediastreams.MediaStreamError:
                        logger.warning("Input stream ended while AI was speaking.")
                        break # Exit loop if stream closes
                    continue # Skip rest of loop iteration

                # --- Process Incoming Audio Frame ---
                try:
                    frame = await self.input_track.recv()
                    audio_data = frame.to_ndarray().flatten().astype(np.float32) / 32768.0
                    
                    resampled_audio = audio_data
                    if frame.sample_rate != 16000:
                        resampled_audio = librosa.resample(audio_data, orig_sr=frame.sample_rate, target_sr=16000)
                    
                    self.audio_buffer.add_audio(resampled_audio)
                    self.last_user_audio_time = time.time() # Update last processed time

                    # --- Check if Buffer is Ready for Processing ---
                    if self.audio_buffer.should_process():
                        audio_to_process = self.audio_buffer.get_audio_chunk_for_processing()
                        self.audio_buffer.reset() # Clear buffer after getting data
                        
                        # Run inference and send transcription
                        # Use a separate async task for inference to not block the main loop
                        # This task will handle sending transcription messages
                        inference_task = asyncio.create_task(self._handle_inference_and_response(audio_to_process))
                        # Optionally wait for the task if you need to sequence things tightly, 
                        # but for responsiveness, letting it run concurrently is better.
                        # await inference_task 

                except mediastreams.MediaStreamError:
                    logger.warning("Input stream ended.")
                    break # Exit loop if client disconnects
                except Exception as e:
                    logger.error(f"Error processing audio frame: {e}", exc_info=True)
                    self.audio_buffer.reset() # Clear buffer on error

                # --- Detect End of User Utterance and Trigger AI Response ---
                # If AI is not speaking, and there's a gap in user audio, trigger response.
                if not self.is_ai_speaking and (time.time() - self.last_user_audio_time > self.silence_timeout):
                    if (time.time() - self.last_user_audio_time > self.min_response_delay):
                        logger.info("Detected potential end of user utterance (silence detected).")
                        # Re-check buffer one last time
                        if self.audio_buffer.get_audio_chunk_for_processing().size > 0:
                            audio_to_process = self.audio_buffer.get_audio_chunk_for_processing()
                            self.audio_buffer.reset()
                            # Trigger inference and response
                            asyncio.create_task(self._handle_inference_and_response(audio_to_process))
                        else:
                            # If buffer is empty after silence, reset timer to avoid rapid re-triggering
                            self.last_user_audio_time = time.time() 
                        
        except asyncio.CancelledError:
            logger.info("Audio processing loop cancelled.")
        except Exception as e:
            logger.error(f"Unexpected error in AudioProcessor run loop: {e}", exc_info=True)
        finally:
            logger.info("Audio processor loop finished.")

    async def _handle_inference_and_response(self, audio_chunk: np.ndarray):
        """ Orchestrates inference, transcription sending, and TTS response. """
        if not audio_chunk.size > 0:
            logger.warning("Received empty audio chunk for inference.")
            return

        # Send transcription message for the processed audio chunk
        # This is the STT output
        # For now, we are sending the raw audio chunk to the pipeline directly.
        # A real implementation might require passing the actual transcribed text here.
        # For demonstration, we'll assume the pipeline returns text we can log.
        # TODO: Extract actual transcription from pipeline output if possible for better logging.
        
        # Run inference (STT + NLU/Generation)
        response_text = await self._run_ultravox_inference(audio_chunk)
        
        if response_text:
            # Send the transcribed text (or acknowledge processing) to the client
            # This tells the client what was understood
            await send_transcription_to_client(f"You: {response_text}", 'user') # Mark as user input for logging clarity
            
            # Generate TTS response
            await self._synthesize_and_send_response(response_text)
        else:
            logger.warning("Ultravox inference returned no response text.")
            await send_transcription_to_client("AI could not understand.", 'ai') # Inform user

    async def _run_ultravox_inference(self, audio_chunk: np.ndarray) -> str:
        """Runs the Ultravox pipeline (STT + NLU/Generation)."""
        logger.info(f"Running inference on {len(audio_chunk)} samples...")
        # Inform client that AI is processing
        await send_transcription_to_client("AI is thinking...", 'ai_thinking')
        
        # --- Run Inference ---
        # FOR ONNX/TENSORRT: Replace this section with calls to your exported model sessions.
        # Example:
        # try:
        #     # Preprocess audio_chunk (e.g., Mel Spectrogram)
        #     # preprocessed_input = preprocess_audio(audio_chunk, ...) 
        #     # Run ONNX session
        #     # output = uv_session.run(None, {input_name: preprocessed_input})
        #     # response_text = postprocess_onnx_output(output)
        # except Exception as e:
        #     logger.error(f"Error during ONNX inference: {e}", exc_info=True)
        #     return ""
        
        # Default pipeline inference:
        try:
            with torch.inference_mode(): # Crucial for performance
                # The 'pipeline' expects a dictionary. Check model card for exact input names.
                # Assuming it accepts raw audio directly. Needs verification!
                result = uv_pipe(audio_chunk, sampling_rate=16000, max_new_tokens=50) 
                
            response_text = parse_ultravox_response(result)
            logger.info(f"Ultravox generated: '{response_text}'")
            return response_text

        except Exception as e:
            logger.error(f"Error during Ultravox inference: {e}", exc_info=True)
            return "" 

    def _run_tts_sync(self, text: str) -> np.ndarray:
        """Synchronous TTS generation, intended for executor."""
        if not text: return np.array([], dtype=np.float32)
        
        logger.info(f"Generating TTS for: '{text[:50]}...'")
        try:
            # FOR ONNX/TENSORRT: Replace with your TTS model session inference.
            # Example:
            # tokenized_text = preprocess_text(text, ...)
            # audio_output_np = tts_session.run(None, {'input_ids': tokenized_text})[0]
            # audio_output_np = postprocess_tts_output(audio_output_np) # Resample, etc.
            # return audio_output_np

            # Default ChatterboxTTS inference
            with torch.inference_mode():
                wav = tts_model.generate(text).cpu().numpy().flatten()
            
            target_sr = 48000 # WebRTC standard
            if tts_model.sampling_rate != target_sr:
                logger.info(f"Resampling TTS output from {tts_model.sampling_rate}Hz to {target_sr}Hz.")
                wav_resampled = librosa.resample(wav.astype(np.float32), orig_sr=tts_model.sampling_rate, target_sr=target_sr)
            else:
                wav_resampled = wav.astype(np.float32)
                
            return wav_resampled

        except Exception as e:
            logger.error(f"Error during TTS generation: {e}", exc_info=True)
            return np.array([], dtype=np.float32)

    async def _synthesize_and_send_response(self, response_text: str):
        """Handles TTS generation and sending audio back."""
        if not response_text:
            logger.warning("No response text received for TTS.")
            return

        # Run TTS in the thread pool executor
        loop = asyncio.get_running_loop()
        synthesized_audio = await loop.run_in_executor(self.executor, self._run_tts_sync, response_text)

        if synthesized_audio.size > 0:
            # --- State Management for Speaking ---
            self.is_ai_speaking = True
            # Client UI status update handled by JS on 'speaking' event or explicit message
            
            await self.output_track.queue_audio(synthesized_audio)
            
            playback_duration = synthesized_audio.size / 48000.0 # 48kHz
            await asyncio.sleep(playback_duration + 0.1) # Add a small buffer
            
            self.is_ai_speaking = False
            # Client UI status update will be handled by JS remoteAudio.onended
            # Or we can send an explicit message:
            await send_transcription_to_client("AI finished speaking.", 'system')
            logger.info("AI finished speaking.")
        else:
            logger.warning("TTS generation resulted in empty audio.")
            self.is_ai_speaking = False # Reset state on TTS failure
            await send_transcription_to_client("AI could not generate speech.", 'system')


# --- WebSocket Communication Handler ---

async def send_transcription_to_client(text: str, subtype: str):
    """Sends transcription/status messages to all connected WebSocket clients."""
    if not text: return
    message = json.dumps({"type": "transcription", "text": text, "subtype": subtype})
    
    disconnected_sockets = set()
    for ws in ws_clients:
        if ws.closed:
            disconnected_sockets.add(ws)
            continue
        try:
            await ws.send_str(message)
        except Exception as e:
            logger.error(f"Error sending message to WebSocket client: {e}", exc_info=True)
            disconnected_sockets.add(ws)
            
    for ws in disconnected_sockets:
        ws_clients.discard(ws)

async def websocket_handler(request):
    """Handles WebSocket connections for signaling and transcription updates."""
    ws = web.WebSocketResponse(heartbeat=30)
    await ws.prepare(request)
    ws_clients.add(ws)
    logger.info(f"WebSocket client connected: {ws}")

    # Initialize PeerConnection for this connection
    pc = RTCPeerConnection(RTCConfiguration([RTCIceServer(urls="stun:stun.l.google.com:19302")]))
    pcs.add(pc)
    audio_processor = None
    logger.info("RTCPeerConnection created for WebSocket handler.")

    @pc.on("track")
    async def on_track(track):
        nonlocal audio_processor
        logger.info(f"Track {track.kind} received.")
        if track.kind == "audio":
            output_track = ResponseAudioTrack()
            pc.addTrack(output_track)
            logger.info("Added outgoing audio track.")
            
            audio_processor = AudioProcessor(output_track, executor)
            audio_processor.add_input_track(track)
            await audio_processor.start()
            logger.info("Audio processor started.")
            
    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        logger.info(f"ICE connection state is {pc.iceConnectionState}")
        if pc.iceConnectionState in ["failed", "closed", "disconnected"]:
            logger.warning(f"ICE connection lost: {pc.iceConnectionState}")
            await cleanup_connection(pc, audio_processor, ws)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"PeerConnection state is {pc.connectionState}")
        if pc.connectionState in ["failed", "closed", "disconnected"]:
            logger.warning(f"PeerConnection lost: {pc.connectionState}")
            await cleanup_connection(pc, audio_processor, ws)

    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                data = json.loads(msg.data)
                logger.debug(f"Received WebSocket message: {data.get('type', 'unknown')}")
                
                if data["type"] == "offer":
                    await pc.setRemoteDescription(RTCSessionDescription(sdp=data["sdp"], type=data["type"]))
                    answer = await pc.createAnswer()
                    await pc.setLocalDescription(answer)
                    await ws.send_json({"type": "answer", "sdp": pc.localDescription.sdp})
                    logger.info("Sent WebRTC answer.")
                    
                elif data["type"] == "ice-candidate" and data.get("candidate"):
                    try:
                        candidate_data = data["candidate"]
                        candidate_string = candidate_data.get("candidate")
                        if candidate_string:
                            params = candidate_from_sdp(candidate_string)
                            candidate = RTCIceCandidate(sdpMid=candidate_data.get("sdpMid"), sdpMLineIndex=candidate_data.get("sdpMLineIndex"), **params)
                            await pc.addIceCandidate(candidate)
                            logger.debug(f"Added remote ICE candidate.")
                    except Exception as e:
                        logger.error(f"Error adding remote ICE candidate: {e}", exc_info=True)
            elif msg.type == WSMsgType.ERROR:
                logger.error(f"WebSocket error: {ws.exception()}")
                await cleanup_connection(pc, audio_processor, ws)
            elif msg.type == WSMsgType.CLOSE:
                logger.info("WebSocket closed by client.")
                await cleanup_connection(pc, audio_processor, ws)

    except Exception as e:
        logger.error(f"Error in WebSocket handler: {e}", exc_info=True)
        await cleanup_connection(pc, audio_processor, ws)
    finally:
        if ws in ws_clients:
            ws_clients.discard(ws)
        logger.info("WebSocket connection finished.")
    return ws

async def cleanup_connection(pc, audio_processor, ws):
    """Helper function to clean up resources when a connection is lost."""
    logger.info("Cleaning up connection resources...")
    if audio_processor:
        await audio_processor.stop()
        audio_processor = None # Clear reference
        
    if pc and pc.connectionState != "closed":
        await pc.close()
    if pc in pcs:
        pcs.remove(pc)
        
    if ws and ws in ws_clients:
        ws_clients.discard(ws)
        if not ws.closed:
             await ws.close()
    logger.info("Connection resources cleaned up.")


# --- Web Server Setup ---
async def index_handler(request):
    """Serves the HTML client interface."""
    return web.Response(text=HTML_CLIENT, content_type='text/html')

async def on_shutdown(app):
    """Gracefully shuts down all connections and resources."""
    logger.info("Shutting down server and closing connections...")
    for pc_conn in list(pcs):
        if pc_conn.connectionState != "closed":
            await pc_conn.close()
    pcs.clear()
    
    for ws_conn in list(ws_clients):
        if not ws_conn.closed:
            await ws_conn.close(code=1001, message="Server shutting down")
    ws_clients.clear()

    executor.shutdown(wait=True)
    logger.info("Executor shut down. Server stopped.")

async def main():
    """Main function to initialize models and start the web server."""
    if not initialize_models():
        logger.error("Model initialization failed. Exiting.")
        return

    app = web.Application()
    app.router.add_get('/', index_handler)
    app.router.add_get('/ws', websocket_handler)
    app.on_shutdown.append(on_shutdown)
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 7860)
    
    try:
        await site.start()
        logger.info("✅ Server started successfully on http://0.0.0.0:7860")
        logger.info("🚀 Your voice assistant is live!")
        
        # Keep the server running
        await asyncio.Event().wait()
    except Exception as e:
        logger.error(f"Failed to start server: {e}", exc_info=True)
    finally:
        # Ensure shutdown is called even if there's an unexpected exit
        await on_shutdown(app)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Server stopping due to user interruption...")
