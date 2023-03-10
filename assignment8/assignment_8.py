import pickle
from typing import Dict, List, Any, Union
import numpy as np
import pandas as pd
from keras import Input, Sequential
from keras.layers import Bidirectional, LSTM, Embedding, Dense, Flatten
# from keras.utils import pad_sequences
from keras_preprocessing.sequence import pad_sequences
from keras.models import Model
import tensorflow as tf
import os


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

def ff_model(vocab_size=1000, num_features=66):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=128, input_shape=(num_features,)),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Flatten(),
        Dense(128, activation='sigmoid'),
        Dense(1, activation='sigmoid')
    ])
    return model


def lstm_model(vocab_size=1000, num_features=66):
    # high value of hidden size just for fun, 64 is sufficient
    hidden_size = 256
    input_tensor = Input(shape=(num_features,))
    x = Embedding(input_dim=vocab_size, output_dim=128)(input_tensor)
    x = Bidirectional(LSTM(units=hidden_size))(x)

    y = Dense(1, activation='sigmoid')(x)
    model = Model(input_tensor, y)
    return model



def train_model(data: Dict[str, Union[List[Any], np.ndarray, int]], model_type="feedforward", batch_size=1024, epochs=20) -> pd:
    """
    Build a neural network of type model_type and train the model on the data.
    Evaluate the accuracy of the model on test data.

    :param data: The dataset dictionary to train neural network on
    :param model_type: The model to be trained, either "feedforward" for feedforward network
                        or "recurrent" for recurrent network
    :return: The model and accuracy of the model on test data
    """
    x_train = data["x_train"]
    y_train = data["y_train"]
    x_test = data["x_test"]
    y_test = data["y_test"]
    vocab_size = data["vocab_size"]
    seq_length, num_features = x_train.shape[0], x_train.shape[1]

    if model_type== "recurrent":
        print("Creating LSTM model:")
        model = lstm_model(vocab_size, num_features)
    else:
        print("Creating FF model")
        model = ff_model(vocab_size, num_features)

    # There might be an argument for using a different optimizer like adam for the FF network, however for the
    # sake of comparison both use rmsprop in this case
    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=['accuracy'])

    print("Training", model_type, "...")
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)

    #model.save()
    model_history = pd.DataFrame(history.history)
    model_history['epoch'] = history.epoch
    eval = model.evaluate(x_test, y_test)
    print(eval)
    return model_history, eval[1]

def main() -> None:
    print("1. Loading data...")
    keras_data = load_data()
    print("2. Preprocessing data...")
    keras_data = preprocess_data(keras_data)
    print("3. Training feedforward neural network...")
    fnn_model_history, fnn_test_accuracy = train_model(keras_data, model_type="feedforward", epochs=10)
    print('Model: Feedforward NN.\n'
          f'Test accuracy: {fnn_test_accuracy:.3f}')
    #print("4. Training recurrent neural network...")
    #rnn_model_history, rnn_test_accuracy = train_model(keras_data, model_type="recurrent")
    #print('Model: Recurrent NN.\n'
    #      f'Test accuracy: {rnn_test_accuracy:.3f}')



if __name__ == '__main__':
    # Running application with cuda if gpu is available
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    built = tf.test.is_built_with_cuda()
    print("tf is built with CUDA? ", built)
    sys_details = tf.sysconfig.get_build_info()
    # Only show actual errors, not warning
    # Note that the warnings are quite nice for debugging
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    gpus = tf.config.list_physical_devices('GPU')

    print("Num GPUs Available: ", len(gpus))

    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)
    main()

