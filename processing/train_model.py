import yaml

import numpy as np
import tensorflow as tf
from tensorflow import initializers

from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


if __name__ == "__main__":
    # Paths
    cfg_path = "cfg/training_cfg.yaml"
    
    # Load the configuration
    with open(cfg_path) as cfg_file:
        cfg = yaml.safe_load(cfg_file)
    datapath = cfg["datapath"]
    output_path = cfg["output_path"]

    # Load the training data
    X = np.load(datapath+"/X_samples.npy")
    Y = np.load(datapath+"/Y_samples.npy")

    print("Dimensions X: {}".format(np.shape(X)))
    print("Dimensions Y: {}\n".format(np.shape(Y)))

    # Augment the data
    if cfg["augment_data"]:
        X_aug1 = X
        X_aug2 = X
        for idx, x in enumerate(X):
            # Robust against variations in ratio R1/R2 by offseting P1 or P2
            max_i = np.amax(X_aug1[idx,:,0])
            if (max_i - 1.0) < cfg["offset_val"]:
                # Max value allowed for offseting: 1.0 + cfg["offset_val"]
                X_aug1[idx,:,0] = x[:,0] + \
                    np.random.uniform(0.0, cfg["offset_val"])
            else:
                # Otherwise, subtract to P2
                X_aug1[idx,:,1] = x[:,1] - \
                    np.random.uniform(0.0, cfg["offset_val"])
            # Robust against smaller variations of P1 and P2
            X_aug2[idx] = x[:] * \
                np.random.uniform(1.0-cfg["scale_val_down"], 
                                  1.0+cfg["scale_val_up"])

        X_aug = np.concatenate([X, X_aug1, X_aug2])
        Y_aug = np.concatenate([Y, Y, Y])
    else: 
        X_aug = X
        Y_aug = Y

    # Normalize from configuration
    # X_norm = (X_aug / (cfg["max_X"] - cfg["min_X"])) \
    #     * cfg["norm_X_range"] + cfg["norm_X_offset"]
    X_norm = np.multiply(np.divide(X, 
                                   np.subtract(cfg["max_X"], cfg["min_X"])) ,
                         cfg["norm_X_range"]) + cfg["norm_X_offset"]
    Y_norm = (Y_aug / (cfg["max_Y"] - cfg["min_Y"])) \
        * cfg["norm_Y_range"] + cfg["norm_Y_offset"]

    print("Min\max from X: {}, {}".format(np.min(X), np.max(X)))
    print("Min\max from Y: {}, {}".format(np.min(np.transpose(np.transpose(Y)[cfg["out_col"]])), 
                                          np.max(np.transpose(np.transpose(Y)[cfg["out_col"]]))))
    print("Min\max from X norm: {}, {}".format(np.min(X_norm), np.max(X_norm)))
    print("Min\max from Y norm: {}, {}\n".format(np.min(np.transpose(np.transpose(Y_norm)[cfg["out_col"]])), 
                                                 np.max(np.transpose(np.transpose(Y_norm)[cfg["out_col"]]))))

    # Split the data in train and test
    X_test, X_train, Y_test, Y_train = \
        train_test_split(X_norm, Y_norm, test_size=0.3)
    # Exclude the output data that are not to eb predicted
    Y_test = np.transpose(np.transpose(Y_test)[cfg["out_col"]])
    Y_train = np.transpose(np.transpose(Y_train)[cfg["out_col"]])

    # Define the model
    if cfg["model"] == "lstm":
        optimizer = tf.keras.optimizers.Adam(learning_rate=cfg["lr"], clipnorm=1.0)
        initializer = initializers.GlorotUniform()

        ann_model = tf.keras.Sequential(
            [
                tf.keras.layers.Input( shape=X_train[0].shape[-3:] ),
                tf.keras.layers.LSTM(128, 
                                    # activation="leaky_relu",
                                    # recurrent_activation='sigmoid',
                                    recurrent_dropout=0.2,
                                    kernel_initializer=initializer,
                                    return_sequences=True),
                tf.keras.layers.LSTM(64, 
                    # activation="leaky_relu",
                    recurrent_dropout=0.2,
                    # recurrent_activation='sigmoid'
                    kernel_initializer=initializer,
                    ),
                tf.keras.layers.Dense(15, activation="leaky_relu", 
                                      kernel_initializer=initializer,),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(1)
            ]
        )
    if cfg["model"] == "cnn-lstm":
        optimizer = tf.keras.optimizers.Adam(learning_rate=cfg["lr"], clipnorm=1.0)
        initializer = initializers.GlorotUniform()

        ann_model = tf.keras.Sequential(
            [
                tf.keras.layers.Conv1D(16, (4,), strides=3, activation="relu",
                                       input_shape=X_train[0].shape[-2:]),
                tf.keras.layers.MaxPooling1D(2),
                tf.keras.layers.LSTM(16, 
                    # activation="leaky_relu",
                    # recurrent_dropout=0.2,
                    # recurrent_activation='sigmoid'
                    kernel_initializer=initializer,
                    ),
                tf.keras.layers.Dense(15, activation="leaky_relu", 
                                      kernel_initializer=initializer,),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(1)
            ]
        )
    elif cfg["model"] == "cnn":
        optimizer = tf.keras.optimizers.Adam(learning_rate=cfg["lr"])

        X_test = X_test.reshape(*X_test.shape, 1)
        X_train = X_train.reshape(*X_train.shape, 1)

        if int(cfg["in_points"]) == 500:
            resh_dims = (40, 25, 1) # For 500 points
        elif int(cfg["in_points"]) == 1000:
            resh_dims = (40, 50, 1) # For 1000 points
        else: 
            print("Input reshape dims must be specified for {} points.".format(
               cfg["in_points"] 
            ))
            ax1 = input("Axis 1: ")
            print("For ax2: {} points".format(cfg["in_points"] / int(ax1)))
            resh_dims = (ax1, cfg["in_points"] / int(ax1), 1)

        ann_model = tf.keras.Sequential(
            [
                tf.keras.layers.Input( shape=X_train[0].shape[-3:] ),
                tf.keras.layers.Reshape(resh_dims), 
                tf.keras.layers.Conv2D(32, (3,3), activation="relu",
                                       input_shape=resh_dims),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(16, (3,3), activation="relu"),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.MaxPooling2D((3, 3)),
                tf.keras.layers.Conv2D(8, (3,3), activation="relu"),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Dense(20, activation="leaky_relu"),
                tf.keras.layers.Dense(1)
            ]
        )
    elif cfg["model"] == "1d-cnn":
        optimizer = tf.keras.optimizers.Adam(learning_rate=cfg["lr"])

        ann_model = tf.keras.Sequential(
            [
                # tf.keras.layers.Input( shape=X_train[0].shape[-3:] ),
                tf.keras.layers.Conv1D(32, (4,), strides=3, activation="relu",
                                       input_shape=X_train[0].shape[-2:]),
                tf.keras.layers.MaxPooling1D(3),
                tf.keras.layers.Conv1D(16, (3,), strides=2, activation="relu"),
                # tf.keras.layers.Dropout(0.1),
                tf.keras.layers.MaxPooling1D(3),
                tf.keras.layers.Conv1D(8, (3,), strides=1, activation="relu"), 
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(15, activation="leaky_relu"),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(1)
            ]
        )

    print(ann_model.summary())

    ann_model.compile(optimizer=optimizer, loss=cfg["loss_fn"])

    print("Starting training..")
    ann_model.fit(X_train, Y_train, epochs=cfg["epochs"], batch_size=cfg["batch_size"], 
                  validation_data=(X_test,Y_test))
    print("Done.")

    today = datetime.now()
    modelpath = output_path + today.strftime("/model_%Y_%m_%d_%H_%M")
    ann_model.save(modelpath + '.model')
    # Valid for Keras 3.5 and TensorFlow 2.17
    # ann_model.export(modelpath + '.model')

    # Save the associated training configuration
    with open(modelpath+'.model'+"/model_cfg.yaml", 'w') as yaml_file:
        yaml.dump(cfg, stream=yaml_file, default_flow_style=None)