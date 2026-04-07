import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout

# 1. Load the Data
# Ensure your CSV has columns named 'message' and 'target' (0 for clean, 1 for toxic)
DATA_PATH = "assets/samples/toxic_chat_data.csv"
df = pd.read_csv(DATA_PATH)

# 2. Preprocessing Parameters
MAX_WORDS = 10000     # Top 10,000 most common words
MAX_LEN = 50          # Max words per 1-second chunk
EMBEDDING_DIM = 100   # Complexity of word relationships

# 3. Clean and Tokenize
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(df['message'].values.astype(str))
sequences = tokenizer.texts_to_sequences(df['message'].values.astype(str))
padded = pad_sequences(sequences, maxlen=MAX_LEN, padding='post')

labels = df['target'].values

# 4. Build the "Lightweight Brain" (1D-CNN)
model = Sequential([
    Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=MAX_LEN),
    Conv1D(128, 5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(24, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid') # Output 0.0 (Clean) to 1.0 (Toxic)
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 5. Train! 🚀
print("🔥 Training the Bouncer's Brain...")
model.fit(padded, labels, epochs=5, batch_size=32, validation_split=0.2)

# 6. Save for Production
model.save("assets/models/toxic_cnn.h5")
print("✅ Model saved to assets/models/toxic_cnn.h5")

# 💡 Bonus: Convert to TFLite for extreme speed
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("assets/models/toxic_cnn.tflite", "wb") as f:
    f.write(tflite_model)
print("⚡ TFLite version created for bare-metal performance!")

import pickle
with open('assets/models/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)