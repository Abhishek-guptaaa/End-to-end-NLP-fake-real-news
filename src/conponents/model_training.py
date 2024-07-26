from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

def train_model(model, train_seq, y_train, epochs, lr):
    model.compile(optimizer=Adam(learning_rate=lr), loss=BinaryCrossentropy(), metrics=['accuracy'])
    
    model_es = EarlyStopping(monitor='val_loss', mode='min', patience=2, restore_best_weights=True)
    model_rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, mode='min')
    
    history = model.fit(train_seq, y_train, epochs=epochs, validation_split=0.2, callbacks=[model_es, model_rlr])
    
    return history
