import numpy as np
import onnxruntime as ort
import cv2


class OnnxDetector:
    def __init__(self, model_path):
        self.model_path = model_path

        providers = list(
            set(ort.get_available_providers())
            & {"CUDAExecutionProvider", "CPUExecutionProvider"}
        )
        self.sess = ort.InferenceSession(
            self.model_path,
            providers=providers,
        )
        self.output_name = self.sess.get_outputs()[0].name

        input_ = self.sess.get_inputs()[0]
        self.input_name = input_.name
        self.model_height, self.model_width = input_.shape[2:]

    def preprocess(self, x: np.ndarray):
        x = cv2.resize(x, (self.model_width, self.model_height))
        return x

    def fix_bboxes(self, x, width, height, padding):
        x[:, [0, 2]] -= padding[0]
        x[:, [1, 3]] -= padding[2]
        x[..., [0, 2]] *= width / (self.model_width - padding[0] - padding[1])
        x[..., [1, 3]] *= height / (self.model_height - padding[2] - padding[3])
        return x

    def _infer(self, x: np.ndarray):
        if x.dtype == np.uint8:
            x = x.astype(np.float16) / 255.0
        else:
            x = x.astype(np.float16)
        x = np.expand_dims(x.transpose(2, 0, 1), axis=0)
        return self.sess.run([self.output_name], {self.input_name: x})[0]

    def run(self, image):
        raise NotImplementedError
