from faster_whisper import WhisperModel
import os

class WhisperTranslator:
    def __init__(self, model_size="tiny.en"):
        print(f"🧠 Loading Whisper ({model_size})...")
        # We use CPU with int8 for speed
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def translate_buffer(self, audio_array):
        """
        Processes raw numpy audio directly (no need to save to disk!)
        audio_array: a 1D numpy array of 16kHz audio
        """
        segments, info = self.model.transcribe(
            audio_array,
            beam_size=1,              # Fastest decoding
            language="en",            # Skip language detection
            without_timestamps=True,  # Skip timestamp computation
            vad_filter=True,          # Skip silent parts automatically
        )
        
        full_text = ""
        for segment in segments:
            full_text += segment.text
            
        return full_text.strip()

if __name__ == "__main__":
    # Quick Test
    import numpy as np
    dummy_audio = np.zeros(16000, dtype=np.float32)
    translator = WhisperTranslator()
    print("✅ Whisper is online and ready!")