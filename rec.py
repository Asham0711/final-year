import sounddevice as sd
import soundfile as sf
import os
import time

class AudioRecorder:
    def __init__(self, directory, sample_rate=16000, chunk_duration=3):
        self.directory = directory
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration  # seconds
        self.max_chunks = 5
        self.current_index = 0
        os.makedirs(self.directory, exist_ok=True)

    def safe_delete_file(self, filepath, retries=3, delay=0.5):
        for attempt in range(retries):
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
                    print(f"Deleted: {os.path.basename(filepath)}")
                return
            except PermissionError:
                print(f"Permission denied while deleting {os.path.basename(filepath)}. Retrying {attempt+1}/{retries}...")
                time.sleep(delay)

    def record_chunk(self):
        filename = os.path.join(self.directory, f"chunk{self.current_index}.wav")

        # Delete the oldest file if more than 5 already exist
        if self.current_index >= self.max_chunks:
            delete_index = self.current_index - self.max_chunks
            old_filename = os.path.join(self.directory, f"chunk{delete_index}.wav")
            self.safe_delete_file(old_filename)

        print(f"Recording chunk{self.current_index}.wav for {self.chunk_duration} seconds...")
        audio_data = sd.rec(int(self.sample_rate * self.chunk_duration), samplerate=self.sample_rate, channels=1)
        sd.wait()
        sf.write(filename, audio_data, self.sample_rate)
        print(f"Saved: {os.path.basename(filename)}")

        self.current_index += 1

    def record_forever(self):
        print("Recording started... Press Ctrl+C to stop.")
        try:
            while True:
                self.record_chunk()
        except KeyboardInterrupt:
            print("Recording stopped.")

if __name__ == "__main__":
    save_dir = r"C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/chunks"
    recorder = AudioRecorder(save_dir)
    recorder.record_forever()
