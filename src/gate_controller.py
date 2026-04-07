import numpy as np
import sounddevice as sd

class AudioGate:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        # Pre-generate a 1-second "Censor Beep" (1000Hz sine wave)
        t = np.linspace(0, 1, self.sample_rate, False)
        self.beep_sound = 0.2 * np.sin(1000 * 2 * np.pi * t)
        self.beep_sound = self.beep_sound.astype(np.float32)

    def play_clean(self, audio_data):
        """Passes the original audio through to the speakers."""
        sd.play(audio_data, self.sample_rate)
        sd.wait()

    def play_censored(self):
        """Plays the BEEP instead of the toxic audio."""
        print("🤫 [GATE] Playing Censor Beep!")
        sd.play(self.beep_sound, self.sample_rate)
        sd.wait()

if __name__ == "__main__":
    # Test the beep!
    gate = AudioGate()
    gate.play_censored()