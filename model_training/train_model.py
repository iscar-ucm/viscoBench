import yaml

import numpy as np
import tensorflow as tf

from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


if __name__ == "__main__":
    # Paths
    datapath = "data/dataset01"
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
    ann_model = tf.keras.Sequential(
        [
            tf.keras.layers.Input( shape=X_train[0].shape[-3:] ),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.GRU(20, activation="relu",
                                    recurrent_activation='leaky_relu',
                                    recurrent_dropout=0.1,
                                    return_sequences=True),
            ),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.GRU(20, activation="relu",
                                    recurrent_activation='leaky_relu',
                                    recurrent_dropout=0.1)),
            tf.keras.layers.Dense(20, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.01),
            tf.keras.layers.Dense(1)
        ]
    )
    print(ann_model.summary())

    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=cfg["lr"])
    ann_model.compile(optimizer=optimizer, loss=cfg["loss_fn"])

    print("Starting training..")
    ann_model.fit(X_train, Y_train, epochs=cfg["epochs"], batch_size=16, 
                  validation_data=(X_test,Y_test))
    print("Done.")

    today = datetime.now()
    modelpath = output_path + today.strftime("/model_%Y_%m_%d_%H_%M")
    ann_model.save(modelpath + '.model')