import logging
import os

from pyclashbot.detection.inference.draw import draw_bboxes
from pyclashbot.detection.inference.unit_detector import UnitDetector
from pyclashbot.memu.client import screenshot
from pyclashbot.memu.pmc import pmc
from matplotlib import pyplot as plt

logging.basicConfig(level=logging.INFO)

current_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(current_path, "model_fp16.onnx")

logging.info(f"Loading model from {model_path}")
detector = UnitDetector(model_path)
logging.info("Model loaded")


logging.info("Starting VM")
vm_index = 3
pmc.start_vm(vm_index)
logging.info("VM started")

logging.info("Starting detection loop")
while True:
    try:
        ss = screenshot(vm_index)
        image = detector.preprocess(ss)
        pred = detector.run(image)
        if len(pred) == 0:
            continue
        image = draw_bboxes(ss, pred, pred_dims=(640, 640))
        plt.imshow(image)
        plt.show()
    except Exception as e:
        print(e)
    except KeyboardInterrupt:
        break
