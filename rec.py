'''
RECORDS AUDIO CONTINUOUSLY . TERMINATES MANUALLY
'''


import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import os
from collections import deque

# Configuration
SAMPLE_RATE = 44100  # Sampling rate (44.1 kHz)
DURATION = 20  # Duration of each recording in seconds
CHANNELS = 1  # Mono audio
BUFFER_SIZE = 5  # Number of recordings in the buffer

print(f"Recording {DURATION}-second chunks continuously. Oldest files will be discarded when buffer size exceeds {BUFFER_SIZE}...")

# Circular buffer to hold audio recordings
audio_buffer = deque(maxlen=BUFFER_SIZE)
recording_counter = 0

try:
    while True:
        recording_counter += 1
        print(f"Recording chunk {recording_counter}...")
        audio_data = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16')
        sd.wait()  # Wait for the recording to finish

        # Generate filename for the current chunk
        chunk_filename = f"chunk_{recording_counter}.wav"

        # Save the recording to a file
        write(chunk_filename, SAMPLE_RATE, audio_data)
        print(f"Chunk {recording_counter} saved as '{chunk_filename}'")

        # Add the new file to the buffer
        if len(audio_buffer) == BUFFER_SIZE:
            # Remove the oldest file from the buffer and delete it from disk
            oldest_file = audio_buffer.popleft()
            if os.path.exists(oldest_file):
                os.remove(oldest_file)
                print(f"Removed oldest file: '{oldest_file}'")

        audio_buffer.append(chunk_filename)

except KeyboardInterrupt:
    print("Recording terminated by user.")
    print("Remaining files in buffer:", list(audio_buffer))
