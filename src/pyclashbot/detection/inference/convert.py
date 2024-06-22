from onnxconverter_common.float16 import convert_float_to_float16
import onnxmltools
import os

current_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(current_path, "model.onnx")
model_out_path = os.path.join(current_path, "model_fp16.onnx")

mdl_in = onnxmltools.utils.load_model(model_path)
mdl = convert_float_to_float16(mdl_in)
onnxmltools.utils.save_model(mdl, model_out_path)
