import os
import logging
import re
import multiprocessing
import numpy as np

from tensorflow.python.compiler.tensorrt import trt_convert as trt

MODELS_DIR = "saved_models"
OUTPUT_DIR = "converted_models"


logging.basicConfig(level=logging.DEBUG,
                    filename='convert.log',
                    filemode='w')
console = logging.StreamHandler()
logger = logging.getLogger("convert")
logger.addHandler(console)


def convert(model_dir, save_dir):

    def input_fn():
        dim = int(re.search(r"mobilenet_v[1,2,3]_[0,1]\.[0-9]+_([0-9]+)_.+", model_dir).group(1))
        yield [np.zeros((1, dim, dim, 3)).astype(np.float32)]

    if "quant" in model_dir:
        conversion_params = trt.TrtConversionParams(rewriter_config_template=None,
                max_workspace_size_bytes=(1<<25),
                precision_mode=trt.TrtPrecisionMode.INT8, minimum_segment_size=3,
                is_dynamic_op=True, maximum_cached_engines=1, use_calibration=True,
                max_batch_size=1)
    else:
        conversion_params = trt.TrtConversionParams(rewriter_config_template=None,
                max_workspace_size_bytes=(1<<25),
                precision_mode=trt.TrtPrecisionMode.FP16, minimum_segment_size=3,
                is_dynamic_op=True, maximum_cached_engines=1, use_calibration=True,
                max_batch_size=1)

    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=model_dir,
        conversion_params=conversion_params)

    try:
        if "quant" in model_dir:
            converter.convert(calibration_input_fn=input_fn)
        else:
            converter.convert()

        converter.build(input_fn=input_fn)

        converter.save(save_dir)
        logger.info("Model converted successfully")
    except Exception as e:
        logger.error("Error :" + str(e) + " while converting model")


if __name__ == "__main__":
    for ver in os.listdir(MODELS_DIR):
        logger.info(f"Currently converting version {ver}")
        for model in os.listdir(os.path.join(MODELS_DIR, ver)):
            if os.path.exists(os.path.join(OUTPUT_DIR, ver, model.replace("_frozen", ""))):
                logger.info(f"{model} already in converted folder, skipping")
                continue
            logger.info(f"Converting {model} to TF-TRT")
            # ugly way of forcing tensorflow to free memory
            proc_convert = multiprocessing.Process(target=convert, args=(os.path.join(MODELS_DIR, ver, model), os.path.join(OUTPUT_DIR, ver, model.replace("_frozen", "")),))
            proc_convert.start()
            proc_convert.join()
