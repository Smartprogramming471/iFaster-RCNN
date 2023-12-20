import os
import torch as t
from config import opt
from faster_rcnn_vgg16 import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from util import read_image
from vis_tool import vis_bbox
import array_tool as at
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

# Read the image
img = read_image('/home/up202003072/Documents/GRAZPEDWRI-DX/pascalvoc/images/2241_1074115078_02_WRI-R2_M012.png') 
img = t.from_numpy(img)[None]

#/home/up202003072/Documents/GRAZPEDWRI-DX/pascalvoc/images/3841_1074609671_04_WRI-L2_M013.png -  frature
#/home/up202003072/Documents/GRAZPEDWRI-DX/pascalvoc/images/3050_0923627159_01_WRI-R2_M015.png - metal 
#'/home/up202003072/Documents/GRAZPEDWRI-DX/pascalvoc/images/4903_0814003975_05_WRI-L2_M015.png' - soft
#/home/up202003072/Documents/preds_metal/1524_0639700139_01_WRI-R2_F017_predicted.png (metal boa predict)

# Create Faster RCNN model and trainer
faster_rcnn = FasterRCNNVGG16()
trainer = FasterRCNNTrainer(faster_rcnn).cuda()

# Load the model
loaded_model = trainer.load('/home/up202003072/Documents/faster_protoPnet/checkpoints/fasterrcnn_08121839_0.6605683802020664', load_optimizer=True, parse_opt=True)
print("Loaded model:", loaded_model)

# Perform inference and measure the time
start_time = time.time()
pred_bboxes_, pred_labels_, pred_scores_, logits, min_distances, conv_output, distances = faster_rcnn.predict(img, visualize=True)
end_time = time.time()
inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
print("Inference Time:", inference_time, "milliseconds")

# Visualize the results
vis_img = vis_bbox(at.tonumpy(img[0]),
                   at.tonumpy(pred_bboxes_[0]),
                   at.tonumpy(pred_labels_[0]).reshape(-1),
                   at.tonumpy(pred_scores_[0]).reshape(-1))

plt.imshow(vis_img.get_images()[0].get_array())
plt.savefig("/home/up202003072/Documents/2241_1074115078_02_WRI-R2_M012_predicted_fasterppnet.png", format="png")


