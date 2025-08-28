import tensorflow as tf
from keras import layers, Sequential

def build_cnn_lstm_model(input_shape=(128, 128, 1)):

    model = Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
    model.add(layers.MaxPool2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPool2D(2, 2))
    
    model.add(layers.Conv2D(128, (3, 3), activation="relu"))
    model.add(layers.MaxPool2D((2, 2)))

    model.add(layers.Reshape((x.shape[1], x.shape[2] * x.shape[3])))
    model.add(layers.LSTM(64))

    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model