#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 10:23:01 2022

@author: up202103249
"""

from __future__ import  absolute_import
import os

import ipdb
import matplotlib
from tqdm import tqdm

from utils.config import opt
from data.dataset import Dataset, TestDataset, inverse_normalize
from model import FasterRCNNVGG16
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc

# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
import resource
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from utils_kitty.datasets import*
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

#matplotlib.use('agg')


def eval(dataloader, faster_rcnn,trainer, test_num=100):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (img, gt_bboxes_, gt_labels_, scale) in enumerate(dataloader):
        #sizes = [sizes[0][0].item(), sizes[1][0].item()]
        if len(gt_labels_)==0:
            continue
        o_img=img
        img = inverse_normalize(at.tonumpy(img[0]))
        
        pred_bboxes_, pred_labels_, pred_scores_ ,logits,min_d,conv_output,distances= faster_rcnn.predict([img],visualize=True)
        print("predicted")
        print(torch.argmax(torch.Tensor(logits[0]), dim=1))
        print("true labels")
        print(gt_labels_+1)
        min_d=torch.Tensor(min_d[0][0]).unsqueeze(0)
        prototype_activations = faster_rcnn.head.ppnet.distance_2_similarity(min_d)        
        print(torch.sort(prototype_activations[0]))
        #input_img = np.transpose(o_img[0], (2, 1, 0))
        # import matplotlib
        # matplotlib.use('TkAgg')
        # import matplotlib.pyplot as plt
        # plt.imshow(input_img, interpolation='nearest')
        # plt.show()
        # plt.savefig('image.png')
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        
        if ii == test_num: break
    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults=None,
        use_07_metric=True)
   
    return result


def train(**kwargs):
    
    opt._parse(kwargs)

   
    faster_rcnn = FasterRCNNVGG16().cuda()
    repetitions=100
    total_time = 0

    
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)
    #trainer.vis.text(dataset.db.label_names, win='labels')
    
    train_path="/home/up202103249/KITTI/val20.txt"
    path="/home/up202103249/KITTI/training/image_2/"
    labels_path="/home/up202103249/KITTI/training/label_2/"
    test_dataloader=torch.utils.data.DataLoader(ListDataset(train_path,path,labels_path),batch_size=1,shuffle=False,num_workers=0)
    
    #best so far 
    best_path="checkpoints/fasterrcnn_08131119_0.691027372890261"
    
    #best_path="checkpoints/fasterrcnn_08140237_0.6738859340614863"
    trainer.load(best_path)
    faster_rcnn_t=trainer.faster_rcnn
    eval_result = eval(test_dataloader, faster_rcnn_t, trainer,test_num=1000)
    # print("t-mAP")
    #print(eval_result["map"])
    # specify the test image to be analyzed
    test_image_dir = "/home/up202103249/KITTI/training/image_2/"#'./local_analysis/Painted_Bunting_Class15_0081/'
    #
    test_image_name = "006004.png" #'Painted_Bunting_0081_15230.jpg'
    test_image_label = 3 #15
    
    test_image_path = os.path.join(test_image_dir, test_image_name)
 
    
    
    
    ##### HELPER FUNCTIONS FOR PLOTTING
    import matplotlib.pyplot as plt
    import cv2
    from PIL import Image
    load_model_dir="/home/up202103249/Faster-RCNN_ProtoPNet_Kitty_v2/simple-faster-rcnn-pytorch/proto"
    load_img_dir = load_model_dir
     
    
    def save_prototype(fname, epoch, index):
         p_img = plt.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch), 'prototype-img'+str(index)+'.png'))
         #plt.axis('off')
         plt.imsave(fname, p_img)
        
    def save_prototype_self_activation(fname, epoch, index):
         p_img = plt.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch),
                                         'prototype-img-original_with_self_act'+str(index)+'.png'))
         #plt.axis('off')
         plt.imsave(fname, p_img)
    
    def save_prototype_original_img_with_bbox(fname, epoch, index,
                                               bbox_height_start, bbox_height_end,
                                               bbox_width_start, bbox_width_end, color=(0, 255, 255)):
         p_img_bgr = cv2.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch), 'prototype-img-original'+str(index)+'.png'))
         cv2.rectangle(p_img_bgr, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1),
                       color, thickness=2)
         p_img_rgb = p_img_bgr[...,::-1]
         p_img_rgb = np.float32(p_img_rgb) / 255
         #plt.imshow(p_img_rgb)
         #plt.axis('off')
         plt.imsave(fname, p_img_rgb)
    
    def imsave_with_bbox(fname, img_rgb, bbox_height_start, bbox_height_end,
                          bbox_width_start, bbox_width_end, color=(0, 255, 255)):
         img_bgr_uint8 = cv2.cvtColor(np.uint8(255*img_rgb), cv2.COLOR_RGB2BGR)
         cv2.rectangle(img_bgr_uint8, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1),
                       color, thickness=2)
         img_rgb_uint8 = img_bgr_uint8[...,::-1]
         img_rgb_float = np.float32(img_rgb_uint8) / 255
         #plt.imshow(img_rgb_float)
         #plt.axis('off')
         plt.imsave(fname, img_rgb_float)
    def normalize_function(x):
         """
         Normalize a list of sample image data in the range of 0 to 1
         : x: List of image data.  The image shape is (32, 32, 3)
         : return: Numpy array of normalized data
         """
         return np.array((x - np.min(x)) / (np.max(x) - np.min(x)))
    
     # load the test image and forward it through the network
    from model.preprocess import mean, std, preprocess_input_function, undo_preprocess_input_function
    from model.log import create_logger
    save_analysis_path="/home/up202103249/Faster-RCNN_ProtoPNet_Kitty_v2/simple-faster-rcnn-pytorch/explained/"
    log, logclose = create_logger(log_filename=os.path.join(save_analysis_path, 'local_analysis.log'))
    normalize = transforms.Normalize(mean=mean,
                                      std=std)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        normalize
     ])
    
    img_pil = Image.open(test_image_path)
    img_tensor = preprocess(img_pil).permute(0,2,1)
    img_variable = Variable(img_tensor.unsqueeze(0))
    
    img=np.array(Image.open(test_image_path))
    img=normalize_function(img)
    img=inverse_normalize(img)
    img = np.transpose(img, (2, 1, 0))
    img=torch.from_numpy(img).float()
    
    pred_bboxes_, pred_labels_, pred_scores_ ,logits,min_distances,conv_output,distances= faster_rcnn_t.predict([img],visualize=True)
    cropped_image=img_variable[:,:,int(pred_bboxes_[0][0][0]):int(pred_bboxes_[0][0][2]),int(pred_bboxes_[0][0][1]):int(pred_bboxes_[0][0][3])]
    images_test = img_variable.cuda()
    labels_test = torch.tensor([test_image_label])
    
    image = cv2.imread("/home/up202103249/KITTI/training/image_2/"+test_image_name)
    start_point = (int(pred_bboxes_[0][0][0]), int(pred_bboxes_[0][0][1]))
    end_point = (int(pred_bboxes_[0][0][2]), int(pred_bboxes_[0][0][3]))
    image = cv2.rectangle(image, start_point, end_point, (255, 0, 0), 2)
    image = image[...,::-1]
    plt.imsave("/home/up202103249/Faster-RCNN_ProtoPNet_Kitty_v2/simple-faster-rcnn-pytorch/explained/most_activated_prototypes/original_image.png", image)
    
    o_img = img_variable.numpy()
    o_img = np.transpose(o_img[0], (2, 1, 0))
    o_img=normalize_function(o_img)
    imsave_with_bbox(fname=os.path.join(save_analysis_path, 'most_activated_prototypes',
                            'original_image.png'),
                     img_rgb=o_img,
                     bbox_height_start=int(pred_bboxes_[0][0][1]),
                     bbox_height_end=int(pred_bboxes_[0][0][3]),
                     bbox_width_start=int(pred_bboxes_[0][0][0]),
                     bbox_width_end=int(pred_bboxes_[0][0][2]), color=(0, 255, 255))
    
    epoch_number_str=11
    prototype_info = np.load(os.path.join(load_img_dir, 'epoch-'+str(epoch_number_str), 'bb'+str(epoch_number_str)+'.npy'))
    prototype_img_identity = prototype_info[:, -1]

    log('Prototypes are chosen from ' + str(len(set(prototype_img_identity))) + ' number of classes.')
    log('Their class identities are: ' + str(prototype_img_identity))
    
    #sanity check 
    # confirm prototype connects most strongly to its own class
    prototype_max_connection = torch.argmax(faster_rcnn_t.head.ppnet.last_layer.weight, dim=0)
    prototype_max_connection = prototype_max_connection.cpu().numpy()
    if np.sum(prototype_max_connection == prototype_img_identity) == faster_rcnn_t.head.ppnet.num_prototypes:
        log('All prototypes connect most strongly to their respective classes.')
    else:
        log('WARNING: Not all prototypes connect most strongly to their respective classes.')
    
    # logits, min_distances = faster_rcnn.head.ppnet_multi(images_test)
    scale=1
    a=faster_rcnn_t.extractor(images_test)
    rpn_locs, rpn_scores, rois, roi_indices, anchor=faster_rcnn_t.rpn(a, torch.Tensor([1224,370]),scale)
    
    
    roi_indices = at.totensor(roi_indices).float()
    rois = at.totensor(rois).float()
    indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1)
     # NOTE: important: yx->xy
    xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
    indices_and_rois =  xy_indices_and_rois.contiguous()
    pool=faster_rcnn_t.head.roi(a,indices_and_rois)
    
    
    #conv_output, distances = faster_rcnn_t.head.ppnet.push_forward(pool)
    min_distances=torch.Tensor(min_distances[0][0]).unsqueeze(0)
    distances=torch.Tensor(distances[0]).unsqueeze(0)
    conv_output=torch.Tensor(conv_output[0]).unsqueeze(0)
     #min_distances = min_distances.view(-1, n_prototypes)
    prototype_activations = faster_rcnn_t.head.ppnet.distance_2_similarity(min_distances)
    prototype_activation_patterns = faster_rcnn_t.head.ppnet.distance_2_similarity(distances)
    
    prototype_shape = faster_rcnn_t.head.ppnet.prototype_shape
    max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]
    if faster_rcnn_t.head.ppnet.prototype_activation_function == 'linear':
         prototype_activations = prototype_activations + max_dist
         prototype_activation_patterns = prototype_activation_patterns + max_dist
    
    
    logits=torch.Tensor(logits)
    tables = []
    for i in range(logits.size(0)):
         tables.append((torch.argmax(logits[0], dim=1)[i].item(), labels_test[i].item()))
         log(str(i) + ' ' + str(tables[-1]))
    
    idx = 0
    predicted_cls = tables[idx][0]
    correct_cls = tables[idx][1]
    log('Predicted: ' + str(predicted_cls))
    log('Actual: ' + str(correct_cls))
    original_img = cropped_image
    size=original_img.size()
    original_img = original_img.numpy()
    original_img = np.transpose(original_img[0], (2, 1, 0))
    original_img=normalize_function(original_img)
    
     ##### MOST ACTIVATED (NEAREST) 10 PROTOTYPES OF THIS IMAGE
    from helpers import makedir, find_high_activation_crop
    makedir(os.path.join(save_analysis_path, 'most_activated_prototypes'))
    
    start_epoch_number=11
    epoch_number_str=11
    prototype_info = np.load(os.path.join(load_img_dir, 'epoch-'+str(epoch_number_str), 'bb'+str(epoch_number_str)+'.npy'))
    prototype_img_identity = prototype_info[:, -1]
    prototype_max_connection = torch.argmax(faster_rcnn_t.head.ppnet.last_layer.weight, dim=0)
    log('Most activated 10 prototypes of this image:')
    array_act, sorted_indices_act = torch.sort(prototype_activations[idx])
    # sorted_indices_act=sorted_indices_act[0]
    # array_act=array_act[0]
    for i in range(1,11):
        if prototype_img_identity[sorted_indices_act[-i].item()]==predicted_cls and array_act[-i]>=2.9999:
          log('top {0} activated prototype for this image:'.format(i))
          save_prototype(os.path.join(save_analysis_path, 'most_activated_prototypes',
                                      'top-%d_activated_prototype.png' % i),
                        start_epoch_number, sorted_indices_act[-i].item())
          save_prototype_original_img_with_bbox(fname=os.path.join(save_analysis_path, 'most_activated_prototypes',
                                                                  'top-%d_activated_prototype_in_original_pimg.png' % i),
                                                epoch=start_epoch_number,
                                                index=sorted_indices_act[-i].item(),
                                                bbox_height_start=prototype_info[sorted_indices_act[-i].item()][1],
                                                bbox_height_end=prototype_info[sorted_indices_act[-i].item()][2],
                                                bbox_width_start=prototype_info[sorted_indices_act[-i].item()][3],
                                                bbox_width_end=prototype_info[sorted_indices_act[-i].item()][4],
                                                color=(0, 255, 255))
          save_prototype_self_activation(os.path.join(save_analysis_path, 'most_activated_prototypes',
                                                      'top-%d_activated_prototype_self_act.png' % i),
                                        start_epoch_number, sorted_indices_act[-i].item())
          log('prototype index: {0}'.format(sorted_indices_act[-i].item()))
          log('prototype class identity: {0}'.format(prototype_img_identity[sorted_indices_act[-i].item()]))
          if prototype_max_connection[sorted_indices_act[-i].item()] != prototype_img_identity[sorted_indices_act[-i].item()]:
              log('prototype connection identity: {0}'.format(prototype_max_connection[sorted_indices_act[-i].item()]))
          log('activation value (similarity score): {0}'.format(array_act[-i]))
          log('last layer connection with predicted class: {0}'.format( faster_rcnn_t.head.ppnet.last_layer.weight[predicted_cls][sorted_indices_act[-i].item()]))
        
          activation_pattern = prototype_activation_patterns[idx][sorted_indices_act[-i].item()].detach().cpu().numpy()
          upsampled_activation_pattern = cv2.resize(activation_pattern, dsize=(size[2], size[3]),
                                                    interpolation=cv2.INTER_CUBIC)
        
          # show the most highly activated patch of the image by this prototype
          high_act_patch_indices = find_high_activation_crop(upsampled_activation_pattern)
          high_act_patch = original_img[high_act_patch_indices[0]:high_act_patch_indices[1],
                                        high_act_patch_indices[2]:high_act_patch_indices[3], :]
          log('most highly activated patch of the chosen image by this prototype:')
          #plt.axis('off')
          plt.imsave(os.path.join(save_analysis_path, 'most_activated_prototypes',
                                  'most_highly_activated_patch_by_top-%d_prototype.png' % i),
                    high_act_patch)
          log('most highly activated patch by this prototype shown in the original image:')
          imsave_with_bbox(fname=os.path.join(save_analysis_path, 'most_activated_prototypes',
                                  'most_highly_activated_patch_in_original_img_by_top-%d_prototype.png' % i),
                          img_rgb=original_img,
                          bbox_height_start=high_act_patch_indices[0],
                          bbox_height_end=high_act_patch_indices[1],
                          bbox_width_start=high_act_patch_indices[2],
                          bbox_width_end=high_act_patch_indices[3], color=(0, 255, 255))
        
          # show the image overlayed with prototype activation map
          rescaled_activation_pattern = upsampled_activation_pattern - np.amin(upsampled_activation_pattern)
          rescaled_activation_pattern = rescaled_activation_pattern / np.amax(rescaled_activation_pattern)
          heatmap = cv2.applyColorMap(np.uint8(255*rescaled_activation_pattern), cv2.COLORMAP_JET)
          heatmap = np.float32(heatmap) / 255
          heatmap = heatmap[...,::-1]
          overlayed_img = 0.5 * original_img + 0.3 * heatmap
          log('prototype activation map of the chosen image:')
          #plt.axis('off')
          plt.imsave(os.path.join(save_analysis_path, 'most_activated_prototypes',
                                  'prototype_activation_map_by_top-%d_prototype.png' % i),
                    overlayed_img)
          log('--------------------------------------------------------------')
    



if __name__ == '__main__':
    train(env='fasterrcnn',plot_every='100')

