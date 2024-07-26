import os
import numpy as np
import joblib
from keras.models import load_model
from keras.utils import pad_sequences
from src.components.data_cleaning import text_cleaner, text_preprocessor
from src.components.data_tokenization import tokenize_text

# Load the pre-trained model and tokenizer
model = load_model('models/model.h5')
tokenizer = joblib.load('models/tokenizer.pkl')

max_length = 300  # Adjust based on your needs

def run_pipeline(text):
    text = text_cleaner(text)
    text = text_preprocessor(text)
    
    seq = tokenizer.texts_to_sequences([text])
    seq = pad_sequences(seq, maxlen=max_length, padding='post', truncating='post')
    
    prediction = model.predict(seq)[0][0]
    return 'Positive' if prediction > 0.5 else 'Negative'
