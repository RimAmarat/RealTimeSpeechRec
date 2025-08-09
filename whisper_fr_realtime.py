import pyaudio
import numpy as np
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa

# Load Whisper model and processor
model_name = "openai/whisper-small"  # Use "tiny", "base", "small", "medium", or "large"
model = WhisperForConditionalGeneration.from_pretrained(model_name)
processor = WhisperProcessor.from_pretrained(model_name)

# Real-time Audio Streaming and Buffering Configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Whisper expects 16kHz
CHUNK = 1024  # 1024 samples (~64ms)
FRAME_BUFFER_SIZE = 10  # Buffer 10 chunks (~640ms total)

# Initialize PyAudio
p = pyaudio.PyAudio()


# Callback to continuously capture and buffer audio
def callback(in_data, frame_count, time_info, status):
    global audio_buffer
    data = np.frombuffer(in_data, dtype=np.int16)
    audio_buffer = np.append(audio_buffer, data)
    return (in_data, pyaudio.paContinue)


# Initialize audio buffer
audio_buffer = np.array([])

# Open a PyAudio stream for real-time audio capture
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                stream_callback=callback)

print("Listening for real-time transcription...")

# Start stream
stream.start_stream()


def process_audio_buffer(buffer, rate):
    """ Transcribe the current audio buffer using Whisper """
    # Ensure buffer is in float32 and normalize
    audio = buffer / np.max(np.abs(buffer))

    # Get input features
    input_features = processor(audio, sampling_rate=rate, return_tensors="pt").input_features

    # Generate transcription
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    return transcription


# Real-time transcription loop
try:
    while stream.is_active():
        if len(audio_buffer) >= FRAME_BUFFER_SIZE * CHUNK:
            # Process audio in chunks
            chunk = audio_buffer[:FRAME_BUFFER_SIZE * CHUNK]
            transcription = process_audio_buffer(chunk, RATE)
            print(f"Transcription: {transcription}")

            # Remove the processed chunk from the buffer
            audio_buffer = audio_buffer[FRAME_BUFFER_SIZE * CHUNK:]
except KeyboardInterrupt:
    print("Terminating transcription...")

# Stop stream and terminate PyAudio
stream.stop_stream()
stream.close()
p.terminate()

import time

latency_start_time = time.time()
# After processing a chunk of audio
latency_end_time = time.time()
latency = latency_end_time - latency_start_time
print(f"Latency for this transcription: {latency:.3f} seconds")

import jiwer

# Reference transcription (ground truth)
reference = "Bonjour, je m'appelle Whisper."

# Hypothesis (predicted transcription)
hypothesis = transcription

# Compute Word Error Rate (WER)
wer = jiwer.wer(reference, hypothesis)
print(f"Word Error Rate (WER): {wer:.3f}")

process_start_time = time.time()
# Process audio (e.g., Whisper transcription)
process_end_time = time.time()

runtime = process_end_time - process_start_time
print(f"Processing time for this chunk: {runtime:.3f} seconds")


process_start_time = time.time()
# Process audio (e.g., Whisper transcription)
process_end_time = time.time()

runtime = process_end_time - process_start_time
print(f"Processing time for this chunk: {runtime:.3f} seconds")

def evaluate_audio_file(file_path, ground_truth):
    # Load audio file
    audio, sr = librosa.load(file_path, sr=RATE)

    # Process the entire audio
    transcription = process_audio_buffer(audio, sr)

    # Calculate WER
    wer = jiwer.wer(ground_truth, transcription)

    return transcription, wer


# Example audio evaluation
audio_file_path = "french_audio_sample.wav"
ground_truth_transcription = "Bonjour, je suis un exemple d'audio français."
transcription, wer = evaluate_audio_file(audio_file_path, ground_truth_transcription)

print(f"Transcription: {transcription}")
print(f"WER: {wer:.3f}")
