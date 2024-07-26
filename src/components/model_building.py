from keras import Sequential
from keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout

def build_model(vocab_size, embedding_dim, max_length, embedding_matrix):
    model = Sequential([
        Input(shape=(max_length,)),
        Embedding(vocab_size, embedding_dim, input_length=max_length, trainable=False),
        Bidirectional(LSTM(128, return_sequences=True)),
        Bidirectional(LSTM(64)),
        Dropout(0.2),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.layers[0].set_weights([embedding_matrix])
    
    return model
