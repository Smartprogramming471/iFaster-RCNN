from torch.utils.data import DataLoader
from boneset import CovidDataset
import matplotlib.pyplot as plt
from matplotlib import patches, patheffects
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor, Lambda
from torchvision import transforms
import torchvision
import torch
from dataset_main import Dataset
from config import opt


#Reading
dataset = CovidDataset(data_dir = '/home/up202003072/Documents/GRAZPEDWRI-DX/pascalvoc/')

dataset2 = Dataset(opt)

test_dataset = CovidDataset(data_dir = '/home/up202003072/Documents/GRAZPEDWRI-DX/pascalvoc/', split = 'test_3L')

#print(dataset.__len__())
#print("Dataset1:", dataset.__getitem__(3556)) #>>tensor (img), label (0,1,2,3)), bbox_list
print("TestSet:", test_dataset.__getitem__(2509))


# VISUALIZE THE DATA
def draw_outline(obj):
    obj.set_path_effects([patheffects.Stroke(linewidth=4, foreground='black'), patheffects.Normal()])

def visualize_data(idx_image=0):
    if idx_image > len(test_dataset) or idx_image < 0:
        print("Image request out of dataset range")
    else:
        image = test_dataset[idx_image][0]
        boxes = test_dataset[idx_image][1] 
        label_class = test_dataset[idx_image][2]

        image = image / np.max(image)

        if image.ndim == 3 and image.shape[0] < image.shape[-1]:
            image = image.transpose((1, 2, 0))

        plt.imshow(image, cmap='gray')
        print(image.shape)

        for b in boxes: 
            for i in range(len(boxes)): 
                y = b[0] #ymin
                x = b[1] #xmin
                h = b[2] - b[0]  #ymax - ymin
                w = b[3] - b[1]  #xmax - xmin
                plt.gca().add_patch(patches.Rectangle((x, y), w, h, fill=False, edgecolor='r', facecolor='none', lw=1))
                # x = b[0] #xmin
                # y = b[1] #ymin
                # w = b[2] - b[0]  #xmax - xmin
                # h = b[3] - b[1]  #ymax - ymin
                # plt.gca().add_patch(patches.Rectangle((x, y), w, h, fill=False, edgecolor='r', facecolor='none', lw=1))
                txt = str(label_class)
                text_class = plt.gca().text(x, (y-10), txt, verticalalignment='top',
                                            color='white', fontsize=9, weight='bold')
                draw_outline(text_class)

        plt.show()

#Read
visualize_data(idx_image=2509)
