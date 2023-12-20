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
from boneset import CovidDataset
import torch

#help function to get all the data ###################################################

data_dir = '/home/up202003072/Documents/GRAZPEDWRI-DX/pascalvoc/'
split = 'trainval'
dataset = CovidDataset(data_dir, split)

data = []
for i in range(len(dataset)):
    image_id = dataset.ids[i]
    labels = dataset[i][1]#.tolist()
    bbox_list = dataset[i][2]#.tolist()

    if len(bbox_list) == 0 or len(labels) == 0 :
        continue
    data.append([image_id, labels, bbox_list])

with open('/home/up202003072/Documents/GRAZPEDWRI-DX/pascalvoc/boneset_5l.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["image_id", "labels", "bbox_list"])
    for d in data:
        writer.writerow(d)



#------------------------------------------- finde periostealreaction -----------

import os
import xml.etree.ElementTree as ET

directory_path = "/home/up202003072/Documents/GRAZPEDWRI-DX/pascalvoc/" 
def __getcena__(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith(".xml"):
            full_path = os.path.join(directory_path, filename)
            tree = ET.parse(full_path)
            root = tree.getroot()
            for obj in root.findall('object'):
                name = obj.find('name').text.lower().strip()
                if name == 'periostealreaction':
                    print(filename)
                    exit()  # exit the script after finding the first example
            
__getcena__(directory_path)



#------------------------------------------- SPLIT TRAIN/VAL/TEST____________________________________

import pandas as pd
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)

df = pd.read_csv('/home/up202003072/Documents/GRAZPEDWRI-DX/pascalvoc/boneset_5l.csv')

from sklearn.model_selection import train_test_split

train_size=0.8

X = df['image_id']
y = df[['labels', 'bbox_list']]

X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=train_size, random_state=random_seed)

test_size = 0.5
X_valid, X_test, y_valid, y_test = train_test_split(X_rem,y_rem, test_size=0.5)


X_valid.to_csv('/home/up202003072/Documents/GRAZPEDWRI-DX/pascalvoc/val5.csv', index= False)

with open('/home/up202003072/Documents/GRAZPEDWRI-DX/pascalvoc/val5.csv', 'r') as f_in, open('/home/up202003072/Documents/GRAZPEDWRI-DX/pascalvoc/val5.txt', 'w') as f_out:

    content = f_in.read()
    f_out.write(content)
X_test.to_csv('/home/up202003072/Documents/GRAZPEDWRI-DX/pascalvoc/test5.csv', index= False)
with open('/home/up202003072/Documents/GRAZPEDWRI-DX/pascalvoc/test5.csv', 'r') as f_in, open('/home/up202003072/Documents/GRAZPEDWRI-DX/pascalvoc/test5.txt', 'w') as f_out:
    content = f_in.read()
    f_out.write(content)

X_train.to_csv('/home/up202003072/Documents/GRAZPEDWRI-DX/pascalvoc/train5.csv', index= False)
with open('/home/up202003072/Documents/GRAZPEDWRI-DX/pascalvoc/train5.csv', 'r') as f_in, open('/home/up202003072/Documents/GRAZPEDWRI-DX/pascalvoc/train5.txt', 'w') as f_out:
    content = f_in.read()
    f_out.write(content)


#_______________________________________ depois de split

train_txt_path = '/home/up202003072/Documents/GRAZPEDWRI-DX/pascalvoc/train5.txt'
val_txt_path = '/home/up202003072/Documents/GRAZPEDWRI-DX/pascalvoc/val5.txt'
trainval_txt_path = '/home/up202003072/Documents/GRAZPEDWRI-DX/pascalvoc/trainval5.txt'

with open(train_txt_path, 'r') as f_train, open(val_txt_path, 'r') as f_val:
    train_content = f_train.read()
    val_content = f_val.read()

trainval_content = train_content + val_content

with open(trainval_txt_path, 'w') as f_trainval:
    f_trainval.write(trainval_content)