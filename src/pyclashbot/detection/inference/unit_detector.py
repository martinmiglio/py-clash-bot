import numpy as np

from pyclashbot.detection.inference.detector import OnnxDetector


class UnitDetector(OnnxDetector):
    MIN_CONF = 0.6
    UNIT_Y_START = 0.05
    UNIT_Y_END = 0.80

    def run(self, image):
        pred = self._infer(image).astype(np.float32)[0]
        pred = pred[pred[:, 4] > self.MIN_CONF]
        return pred
