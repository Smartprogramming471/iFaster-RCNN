import os
import torch as t
from config import opt
from faster_rcnn_vgg16 import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from util import  read_image
from vis_tool import vis_bbox
import array_tool as at
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

img = read_image('/home/up202003072/Documents/GRAZPEDWRI-DX/pascalvoc/images/2241_1074115078_02_WRI-R2_M012.png')
img = t.from_numpy(img)[None]

faster_rcnn = FasterRCNNVGG16()
trainer = FasterRCNNTrainer(faster_rcnn).cuda()

# caffe

start_time = time.time()
loaded_model = trainer.load('/home/up202003072/Documents/prep_tese/checkpoints/fasterrcnn_06190954_0.7047410797760758', load_optimizer=False, parse_opt=True)
print("loaded model= ", loaded_model)

# Perform inference and measure the time
start_time = time.time()
_bboxes, _labels, _scores = trainer.faster_rcnn.predict(img,visualize=True)
end_time = time.time()
inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
print("Inference Time:", inference_time, "milliseconds")

vis_img = vis_bbox(at.tonumpy(img[0]),
         at.tonumpy(_bboxes[0]),
         at.tonumpy(_labels[0]).reshape(-1),
         at.tonumpy(_scores[0]).reshape(-1))


plt.imshow(vis_img.get_images()[0].get_array())
plt.savefig("/home/up202003072/Documents/2241_1074115078_02_WRI-R2_M012_predicted_faster.png", format="png")



