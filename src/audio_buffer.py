import sounddevice as sd
import numpy as np
import queue
import yaml

# Load our rulebook
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

class AudioStreamer:
    def __init__(self):
        self.rate = config['audio']['sampling_rate']
        self.chunk_size = int(self.rate * config['audio']['chunk_duration'])
        self.queue = queue.Queue()

    def _callback(self, indata, frames, time, status):
        """Standard PortAudio callback for real-time streams."""
        if status:
            print(f"⚠️ Audio Status: {status}")
        # Add the chunk to the queue
        self.queue.put(indata.copy())

    def start(self):
        print("🎤 Mic is LIVE. Listening for toxicity...")
        with sd.InputStream(samplerate=self.rate, 
                            channels=1, 
                            callback=self._callback, 
                            blocksize=self.chunk_size):
            while True:
                # This pulls 1 second of audio data at a time
                audio_chunk = self.queue.get()
                yield audio_chunk

    def process_chunk(self, data):
        # 1. Convert to float32 if not already
        audio = data.flatten().astype(np.float32)
        
        # 2. Normalize: Make the loudest part of your voice exactly 1.0
        max_vol = np.max(np.abs(audio))
        if max_vol > 0.01:  # Only normalize if there's actual sound
            audio = audio / max_vol
        
        return audio

    def start_listening(self, duration_seconds=10):
        print(f"🚀 Listening for {duration_seconds} seconds...")
        
        chunks_to_record = int(duration_seconds / (self.chunk_size / self.rate))
        
        with sd.InputStream(samplerate=self.rate, 
                            channels=1, 
                            callback=self._callback, 
                            blocksize=self.chunk_size):
            for _ in range(chunks_to_record):
                audio_data = self.queue.get() 
                self.process_chunk(audio_data)
        
        print("✅ Test complete. Mic closed.")

if __name__ == "__main__":
    # Test run
    streamer = AudioStreamer()
    streamer.start_listening(duration_seconds=5)