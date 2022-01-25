import os
import logging

import tensorflow as tf
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants

MODELS_DIR = "frozen_models"
OUTPUT_DIR = "saved_models"
INPUT_TENSOR_NAME = "input:0"
OUTPUT_TENSOR_NAME = {"v1": "MobilenetV1/Predictions/Reshape:0", "v2": "MobilenetV2/Predictions/Reshape:0",
                      "v3": "MobilenetV3/Predictions/Reshape:0"}

logging.basicConfig(level=logging.DEBUG,
                    filename='restore_frozen.log',
                    filemode='w')
console = logging.StreamHandler()
logger = logging.getLogger("frozen")
logger.addHandler(console)


def convert_pb_to_saved_model(pb_model_path, export_dir, input_name='input:0', output_name='output:0'):
    graph_def = read_pb_model(pb_model_path)
    convert_pb_saved_model(graph_def, export_dir, input_name, output_name)


def read_pb_model(pb_model_path):
    with tf.compat.v2.io.gfile.GFile(pb_model_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        return graph_def


def convert_pb_saved_model(graph_def, export_dir, input_name='input:0', output_name='output:0'):
    builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_dir)

    sigs = {}
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        tf.import_graph_def(graph_def, name="")
        g = tf.compat.v1.get_default_graph()
        inp = g.get_tensor_by_name(input_name)
        out = g.get_tensor_by_name(output_name)

        sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
            tf.compat.v1.saved_model.signature_def_utils.predict_signature_def(
                {"input": inp}, {"output": out})

        builder.add_meta_graph_and_variables(sess,
                                             [tag_constants.SERVING],
                                             signature_def_map=sigs)
        builder.save()


if __name__ == "__main__":
    for ver in os.listdir(MODELS_DIR):
        if ver not in OUTPUT_TENSOR_NAME:
            logger.error(f"{ver} is not a valid mobilenet version")
            break
        for model in os.listdir(os.path.join(MODELS_DIR, ver)):
            try:
                convert_pb_to_saved_model(os.path.join(MODELS_DIR, ver, model), os.path.join(OUTPUT_DIR, ver, model.replace(".pb", "")),
                                          input_name=INPUT_TENSOR_NAME, output_name=OUTPUT_TENSOR_NAME[ver])
                logger.info(f"Successfully saved model {model} in saved model format")
            except Exception as e:
                logger.error(f"{model} couldn't be saved in saved model format with exception: {str(e)}")


