import os
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from skimage.io import imread
import ast
from util import read_image
import cv2 
import xml.etree.ElementTree as ET
import csv

""" 
format of images: I (32-bit signed integer pixels) 3 channel
shapes das imagens sao todas:
img = (H, W) format nao tem todas o mesmo formato 

labels:
boneanomaly (276 boxes), -----tirar
bonelesion (45 boxes), -----tirar
foreignbody (8 boxes),-----tirar
fracture (18090 boxes), - 0
metal (818 boxes), - 1
periostealreaction (3453 boxes), - 2
pronatorsign (567 boxes), - 3
softtissue (464 boxes), - 4
text (23722 boxes) -----tirar

bbox_list = lista de bounding boxes 

"""

class CovidDataset(Dataset):
  """Args:
        data_dir (string): Path to the root of the training data. 
            i.e. '/home/up202003072/Documents/GRAZPEDWRI-DX/'
        split ({'train', 'val', 'trainval'}): Select a split of the
            dataset.
  """
  def __init__(self, data_dir, split = 'train_3L'): 

    id_list_file = os.path.join(data_dir, '{0}.txt'.format(split))

    self.ids = [id_.strip() for id_ in open(id_list_file)]
    self.data_dir = '/home/up202003072/Documents/GRAZPEDWRI-DX/pascalvoc/'
    self.label_names = VOC_BBOX_LABEL_NAMES

  def __len__(self):
      return len(self.ids) 
  
  def __getitem__(self, idx):
    """Returns the i-th example.

    Returns a color image and bounding boxes. The image is in CHW format. The returned image is RGB.
    Returns: tuple of an image and bounding boxes
      """
    id_ = self.ids[idx]
    anno = ET.parse(os.path.join(self.data_dir, id_ + '.xml'))
    bbox = list()
    label = list()
    # difficult = list()
    for obj in anno.findall('object'):
      name = obj.find('name').text.lower().strip()
      if name in ['boneanomaly', 'bonelesion', 'foreignbody', 'periostealreaction', 'pronatorsign', 'text']:
          continue
      bndbox_anno = obj.find('bndbox')
      # subtract 1 to make pixel indexes 0-based
      bbox.append([
          int(float(bndbox_anno.find(tag).text)) - 1 #antes estava int sem o float
          for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
      #name = obj.find('name').text.lower().strip()
      # print(type(bbox))
      label.append(VOC_BBOX_LABEL_NAMES.index(name))
    bbox_list = np.stack(bbox).astype(np.float32)
    # print(type(bbox_list))
    lab = np.stack(label).astype(np.int32)

    # Load a image
    img_file = os.path.join(self.data_dir, 'images', id_ + '.png') 
    image = read_image(img_file, color=True)

    return image, bbox_list, lab


VOC_BBOX_LABEL_NAMES = (
  # 'boneanomaly', 
  # 'bonelesion',
  # 'foreignbody',
  'fracture',
  'metal',
  #'periostealreaction',
  #'pronatorsign',
  'softtissue'
  #'text' 
)

# dataset = CovidDataset(data_dir= '/home/up202003072/Documents/GRAZPEDWRI-DX/pascalvoc/')
# print(dataset.__getitem__(0))
# # # print(dataset.__len__())

# #help function to get all the data
# data_dir = '/home/up202003072/Documents/GRAZPEDWRI-DX/pascalvoc/'
# split = 'new_train'
# dataset = CovidDataset(data_dir, split)

# data = []
# for i in range(len(dataset)):
#     image_id = dataset.ids[i]
#     labels = dataset[i][2]#.tolist()
#     bbox_list = dataset[i][1]#.tolist()

#     if len(bbox_list) == 0 or len(labels) == 0 :
#         continue

#     # Append the data to the list
#     data.append([image_id, labels, bbox_list])

# with open('/home/up202003072/Documents/GRAZPEDWRI-DX/pascalvoc/boneset_5l.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(["image_id", "labels", "bbox_list"])
#     for d in data:
#         writer.writerow(d)

##______________________________________________________________________

