import yaml

import numpy as np
import tensorflow as tf

from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


if __name__ == "__main__":
    # Paths
    datapath = "data/dataset04_alpha"
    cfg_path = "cfg/training_cfg.yaml"
    output_path = "models"

    # Load the configuration
    with open(cfg_path) as cfg_file:
        cfg = yaml.safe_load(cfg_file)

    # Load the training data
    X = np.load(datapath+"/X_samples.npy")
    Y = np.load(datapath+"/Y_samples.npy")

    # Adapt the format
    # scaler_in = StandardScaler()
    # scaler_out = StandardScaler()
    scaler_in = MinMaxScaler()
    scaler_out = MinMaxScaler()
    X_norm = scaler_in.fit_transform(X=X.reshape(-1, X.shape[-1]))
    Y_norm = scaler_out.fit_transform(X=Y.reshape(-1, Y.shape[-1]))
    X_norm = X_norm.reshape(X.shape)
    Y_norm = Y_norm.reshape(Y.shape)
    # Split the data in train and test
    X_test, X_train, Y_test, Y_train = \
        train_test_split(X_norm, Y_norm, test_size=0.3)
    # Exclude the output data that are not to eb predicted
    Y_test = np.transpose(np.transpose(Y_test)[0])
    Y_train = np.transpose(np.transpose(Y_train)[0])

    # Define the model
    if cfg["model"] == "lstm":
        ann_model = tf.keras.Sequential(
            [
                tf.keras.layers.Input( shape=X_train[0].shape[-3:] ),
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.GRU(30, activation="relu",
                                        recurrent_activation='leaky_relu',
                                        recurrent_dropout=0.05,
                                        return_sequences=True),
                ),
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.GRU(30, activation="relu",
                                        recurrent_dropout=0.05,
                                        recurrent_activation='leaky_relu')),
                tf.keras.layers.Dense(20, activation="tanh"),
                tf.keras.layers.Dropout(0.01),
                tf.keras.layers.Dense(1)
            ]
        )
    elif cfg["model"] == "cnn":
        X_test = X_test.reshape(*X_test.shape, 1)
        X_train = X_train.reshape(*X_train.shape, 1)
        print(np.shape(X_test))
        # Y_test = Y_test.reshape(*Y_test.shape, 1)
        # Y_train = Y_train.reshape(*Y_train.shape, 1)

        ann_model = tf.keras.Sequential(
            [
                tf.keras.layers.Input( shape=X_train[0].shape[-3:] ),
                tf.keras.layers.Reshape((40, 25, 1)),
                tf.keras.layers.Conv2D(32, (3,3), activation="leaky_relu",
                                       input_shape=(500, 2)),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3,3), activation="leaky_relu"),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(20, activation="leaky_relu"),
                tf.keras.layers.Dropout(0.01),
                tf.keras.layers.Dense(1)
            ]
        )
    print(ann_model.summary())

    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg["lr"])
    ann_model.compile(optimizer=optimizer, loss=cfg["loss_fn"])

    print("Starting training..")
    ann_model.fit(X_train, Y_train, epochs=cfg["epochs"], batch_size=cfg["batch_size"], 
                  validation_data=(X_test,Y_test))
    print("Done.")

    today = datetime.now()
    modelpath = output_path + today.strftime("/model_%Y_%m_%d_%H_%M")
    ann_model.export(modelpath + '.model')