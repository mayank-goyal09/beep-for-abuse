import numpy as np
import tensorflow as tf
from src.audio_buffer import AudioStreamer
from src.translator import WhisperTranslator
from src.gate_controller import AudioGate
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import scipy.io.wavfile as wav
import time

# --- SETTINGS ---
TOXIC_THRESHOLD = 0.7  # 0.0 to 1.0 (Adjust based on how strict you want the bouncer)
MAX_LEN = 50           # Must match what you used in classifier_trainer.py
SILENCE_THRESHOLD = 0.01  # Audio energy below this = silence, skip it

class ToxicInterceptor:
    def __init__(self):
        print("⚙️ Loading the real Brain...")
        self.translator = WhisperTranslator()
        self.model = tf.keras.models.load_model("assets/models/toxic_cnn.h5")
        
        # Load the Tokenizer from Phase 2
        with open('assets/models/tokenizer.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)
            
        self.gate = AudioGate() # Initializes our censor beeper and speaker routing
        print("✅ Real Brain & Audio Gate are Online!")

    def predict_toxicity(self, text):
        # 1. Turn text into numbers using the tokenizer
        sequences = self.tokenizer.texts_to_sequences([text])
        # 2. Pad it so it's the right length for the CNN
        padded = pad_sequences(sequences, maxlen=50, padding='post')
        # 3. Get the REAL score from the model
        prediction = self.model.predict(padded, verbose=0)
        return prediction[0][0] # Returns a float between 0.0 and 1.0

    def run(self):
        streamer = AudioStreamer()
        debug_saved = False  # Only save the debug file once
        
        for audio_chunk in streamer.start():
            # 0. NORMALIZE the audio (the "Clear Voice" fix)
            audio_chunk = streamer.process_chunk(audio_chunk)
            
            # --- SKIP SILENCE: Don't waste time transcribing dead air ---
            energy = np.sqrt(np.mean(audio_chunk ** 2))  # RMS energy
            if energy < SILENCE_THRESHOLD:
                continue  # Skip this chunk, it's just silence/noise
            
            print(f"🔊 Sound detected (energy: {energy:.4f})")
            
            # --- DEBUG: Save one chunk so you can hear what the AI hears ---
            if not debug_saved:
                wav.write("debug_mic.wav", 16000, audio_chunk)
                print("👉 Listen to 'debug_mic.wav'. Is it clear?")
                debug_saved = True
            
            # 1. TRANSLATE (Audio -> Text)
            t_start = time.time()
            text = self.translator.translate_buffer(audio_chunk)
            t_whisper = time.time() - t_start
            
            if not text:
                print(f"⏱️ Whisper took {t_whisper:.2f}s (no speech found)")
                continue
                
            print(f"🎤 You said: \"{text}\" (Whisper: {t_whisper:.2f}s)")

            # 2. CLASSIFY (Text -> Toxicity Score)
            score = self.predict_toxicity(text)
            
            # 3. ACTION (The Audio Gate)
            if score > TOXIC_THRESHOLD:
                print(f"🚫 [MUTED] Toxicity Detected: {score:.2f} | Word: {text}")
                self.gate.play_censored() # Play a BEEP sound through the speakers!
            else:
                print(f"🟢 [CLEAN] Score: {score:.2f}")
                self.gate.play_clean(audio_chunk) # Pass out the actual voice!

if __name__ == "__main__":
    app = ToxicInterceptor()
    app.run()