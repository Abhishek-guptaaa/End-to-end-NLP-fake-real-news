from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

def tokenize_text(train_texts, test_texts, max_length):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_texts)
    
    train_seq = tokenizer.texts_to_sequences(train_texts)
    test_seq = tokenizer.texts_to_sequences(test_texts)
    
    train_seq = pad_sequences(train_seq, maxlen=max_length, padding='post', truncating='post')
    test_seq = pad_sequences(test_seq, maxlen=max_length, padding='post', truncating='post')
    
    return tokenizer, train_seq, test_seq

def load_embedding_matrix(glove_file, tokenizer, embedding_dim):
    embeddings_index = {}
    with open(glove_file) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    vocab_size = len(tokenizer.word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix
