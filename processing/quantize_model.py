import os
import yaml

import numpy as np
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler


if __name__ == "__main__":
    """
    Converts a TF model into a quantized model by using some representative data for the
    model operation range.
    """
    cfg_path = "cfg/quantization_cfg.yaml"
    with open(cfg_path) as cfg_f:
        cfg = yaml.safe_load(cfg_f)

    X_train = np.load(cfg["X_train_path"])

    # Normalization
    scaler_in = MinMaxScaler()
    X_norm = scaler_in.fit_transform(X=X_train.reshape(-1, X_train.shape[-1]))
    X_norm = X_norm.reshape(X_train.shape)

    # Get some representative data for the conversion
    def representative_data_gen():
        for input_value in \
            tf.data.Dataset.from_tensor_slices(X_norm).batch(10).take(cfg["samples"]):
            yield [np.float32(input_value)]
    
    converter = tf.lite.TFLiteConverter.from_saved_model( cfg["model_path"] )
    # Get the name of the model without the extension
    # E.g. "models/mymodel.h5" -> "mymodel"
    model_name = cfg["model_path"].split("/")[-1].split(".")[0]

    # Configure the conversion
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    # converter.experimental_new_converter=True
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                           tf.lite.OpsSet.SELECT_TF_OPS]
    
    # If automatic input/output quantization
    if cfg["auto_in_quant"]:
        converter.inference_input_type = tf.float32
    else:
        converter.inference_input_type = getattr(tf, cfg["input_type"])
    if cfg["auto_out_quant"]:
        converter.inference_output_type = tf.float32
    else:
        converter.inference_output_type = getattr(tf, cfg["output_type"])

    # Voila!
    tflite_model = converter.convert()
    if not os.path.exists( cfg["output_folder"] ):
        os.mkdir( cfg["output_folder"] )
    if not os.path.exists( cfg["output_folder"] + "/" + model_name ):
        os.mkdir( cfg["output_folder"] + "/" + model_name )
    
    out_model_filepath = cfg["output_folder"] + "/" + model_name \
        + "/" + model_name + '.tflite'
    
    tflite_model_size = open(out_model_filepath, "wb").write(tflite_model)
    print("Quantized model is %d bytes" % tflite_model_size)
    print("Saved at: {}".format(out_model_filepath))
