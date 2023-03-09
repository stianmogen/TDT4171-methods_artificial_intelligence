import pickle
from typing import Dict, List, Any, Union
import numpy as np
import pandas as pd
# Keras
from keras import Input
from keras.layers import Bidirectional, LSTM, Embedding, Dense, Activation, Lambda, RepeatVector, Permute, Concatenate
from tensorflow import keras
import tensorflow as tf
from keras.utils import pad_sequences
from keras.models import Model
from keras import backend



def load_data() -> Dict[str, Union[List[Any], int]]:
    path = "keras-data.pickle"
    with open(file=path, mode="rb") as file:
        data = pickle.load(file)
    return data


def preprocess_data(data: Dict[str, Union[List[Any], int]]) -> Dict[str, Union[List[Any], np.ndarray, int]]:
    """
    Preprocesses the data dictionary. Both the training-data and the test-data must be padded
    to the same length; play around with the maxlen parameter to trade off speed and accuracy.
    """
    maxlen = data["max_length"]//16
    data["x_train"] = pad_sequences(data['x_train'], maxlen=maxlen)
    data["y_train"] = np.asarray(data['y_train'])
    data["x_test"] = pad_sequences(data['x_test'], maxlen=maxlen)
    data["y_test"] = np.asarray(data['y_test'])

    return data

def ff_model(vocab_size=1000):
    input_tensor = Input(shape=(None,))
    x = Embedding(input_dim=vocab_size, output_dim=128)(input_tensor)
    y = Dense(1, activation='sigmoid')(x)
    model = Model(input_tensor, y)
    return model


def lstm_model(vocab_size=1000):
    hidden_size = 64
    input_tensor = Input(shape=(None,))
    x = Embedding(input_dim=vocab_size, output_dim=128)(input_tensor)
    x = Bidirectional(LSTM(units=hidden_size))(x)

    y = Dense(1, activation='sigmoid')(x)
    model = Model(input_tensor, y)
    return model

def train_model(data: Dict[str, Union[List[Any], np.ndarray, int]], model_type="feedforward", batch_size=1024, epochs=10) -> float:
    """
    Build a neural network of type model_type and train the model on the data.
    Evaluate the accuracy of the model on test data.

    :param data: The dataset dictionary to train neural network on
    :param model_type: The model to be trained, either "feedforward" for feedforward network
                        or "recurrent" for recurrent network
    :return: The accuracy of the model on test data
    """

    # TODO build the model given model_type, train it on (data["x_train"], data["y_train"])
    #  and evaluate its accuracy on (data["x_test"], data["y_test"]). Return the accuracy
    x_train = data["x_train"]
    y_train = data["y_train"]
    vocab_size = data["vocab_size"]
    model = lstm_model(vocab_size)
    if (model_type=="reccurent"):
        pass
    else:
        pass
    print("Training..")

    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=['accuracy'])
    model_checkpoints = keras.callbacks.ModelCheckpoint(f"model.h5", save_best_only=True)
    print("x_train", x_train.shape)
    print("y_train", y_train.shape)
    print(batch_size)
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
    model.save()
    model_history = pd.DataFrame(history.history)
    model_history['epoch'] = history.epoch
    return model_history

def main() -> None:
    print("1. Loading data...")
    keras_data = load_data()
    print("2. Preprocessing data...")
    keras_data = preprocess_data(keras_data)
    print("3. Training feedforward neural network...")
    #fnn_test_accuracy = train_model(keras_data, model_type="feedforward")
    #print('Model: Feedforward NN.\n'
    #      f'Test accuracy: {fnn_test_accuracy:.3f}')
    print("4. Training recurrent neural network...")
    rnn_test_accuracy = train_model(keras_data, model_type="recurrent")
    print('Model: Recurrent NN.\n'
          f'Test accuracy: {rnn_test_accuracy:.3f}')



if __name__ == '__main__':
    main()

