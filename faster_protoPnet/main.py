# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 22:38:26 2022

@author: Acee
"""

import cv2
from faster_rcnn_vgg16 import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
from dataset_main import Dataset, TestDataset, inverse_normalize
import torch
import array_tool as at


best_path="/home/up202003072/Documents/faster_protoPnet/checkpoints/fasterrcnn_04200958_0.6716047224979176"
faster_rcnn = FasterRCNNVGG16().cuda()
trainer = FasterRCNNTrainer(faster_rcnn).cuda()
trainer.load(best_path)
faster_rcnn_t=trainer.faster_rcnn
for i in range(112):
    i_str=str(i)
    full_i=i_str.zfill(10)
    cap=cv2.imread("/home/up202103249/Faster-RCNN_ProtoPNet_Kitty_v2/simple-faster-rcnn-pytorch/data/"+full_i+".png")
    bb= faster_rcnn_t.predict([np.transpose(cap, (2, 1, 0))],visualize=True)
    if len(bb[5])!=0:
        for a in range(0,len(bb[0])):
            start_point = (int(bb[0][0][a][0]), int(bb[0][0][a][1]))
            end_point = (int(bb[0][0][a][2]), int(bb[0][0][a][3]))
            cap = cv2.rectangle(cap, start_point, end_point, (255, 0, 0), 2)
            if bb[1][a][0]==0:
                label="Car"
            elif bb[1][a][0]==1:
                label="Pedestrian"
            elif bb[1][a][0]==2:
                label="Cyclist"
            cap=cv2.putText(cap, label, (int(bb[0][0][a][0]), int(bb[0][0][a][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
   
    load_model_dir="/home/up202103249/Faster-RCNN_ProtoPNet_Kitty_v2/simple-faster-rcnn-pytorch/proto"
    load_img_dir = load_model_dir
    test_image_name = "000000.png" #'Painted_Bunting_0081_15230.jpg'
    test_image_label = 2 #15
    test_image_dir = "/home/up202103249/KITTI/training/image_2/"
    #test_image_path = os.path.join(test_image_dir, test_image_name)
    test_image_path="/home/up202103249/Faster-RCNN_ProtoPNet_Kitty_v2/simple-faster-rcnn-pytorch/data/"+full_i+".png"
    
    def save_prototype(fname, epoch, index):
          p_img = cv2.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch), 'prototype-img'+str(index)+'.png'))
          return p_img
          #plt.axis('off')
          #plt.imsave(fname, p_img)
        
    def save_prototype_self_activation(fname, epoch, index,bbox_height_start, bbox_height_end,
                                       bbox_width_start, bbox_width_end, color=(0, 255, 255)):
          p_img = plt.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch),
                                          'prototype-img-original_with_self_act'+str(index)+'.png'))
          cv2.rectangle(p_img, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1),
                         color, thickness=2)
          #plt.imsave("/home/up202103249/Desktop/theia_prototype/self_act.png", p_img)
          return p_img
          #plt.axis('off')
          
    
    def save_prototype_original_img_with_bbox(fname, epoch, index,
                                                bbox_height_start, bbox_height_end,
                                                bbox_width_start, bbox_width_end, color=(0, 255, 255)):
          p_img_bgr = cv2.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch), 'prototype-img-original'+str(index)+'.png'))
          #return p_img_bgr
          cv2.rectangle(p_img_bgr, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1),
                         color, thickness=2)
          
          p_img_rgb = p_img_bgr[...,::-1]
          p_img_rgb = np.float32(p_img_rgb) / 255
          return p_img_bgr
          #plt.imshow(p_img_rgb)
          #plt.axis('off')
          #plt.imsave(fname, p_img_rgb)
    
    def imsave_with_bbox(fname, img_rgb, bbox_height_start, bbox_height_end,
                          bbox_width_start, bbox_width_end, color=(0, 255, 255)):
          img_bgr_uint8 = cv2.cvtColor(np.uint8(255*img_rgb), cv2.COLOR_RGB2BGR)
          cv2.rectangle(img_bgr_uint8, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1),
                        color, thickness=2)
          img_rgb_uint8 = img_bgr_uint8[...,::-1]
          img_rgb_float = np.float32(img_rgb_uint8) / 255
          #plt.imshow(img_rgb_float)
          #plt.axis('off')
          #plt.imsave(fname, img_rgb_float)
    def normalize_function(x):
          """
          Normalize a list of sample image data in the range of 0 to 1
          : x: List of image data.  The image shape is (32, 32, 3)
          : return: Numpy array of normalized data
          """
          return np.array((x - np.min(x)) / (np.max(x) - np.min(x)))
    
      # load the test image and forward it through the network
    from preprocess import mean, std, preprocess_input_function, undo_preprocess_input_function
    from log import create_logger
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
    
    pred_bboxes_, pred_labels_, pred_scores_ ,logits,min_distances,_,_= faster_rcnn_t.predict([img],visualize=True)
    prototype=np.array([1,2,3], ndmin=3,dtype=np.uint8)
    prototype1=np.array([1,2,3], ndmin=3,dtype=np.uint8)
    prototype_self=np.array([1,2,3], ndmin=3,dtype=np.uint8)
    overlayed_img=np.array([1,2,3], ndmin=3,dtype=np.uint8)
    if pred_bboxes_[0].shape[0]!=0:
        cropped_image=img_variable[:,:,int(pred_bboxes_[0][0][0]):int(pred_bboxes_[0][0][2]),int(pred_bboxes_[0][0][1]):int(pred_bboxes_[0][0][3])]
        cutted=cap[int(pred_bboxes_[0][0][1]):int(pred_bboxes_[0][0][3]),int(pred_bboxes_[0][0][0]):int(pred_bboxes_[0][0][2]),:]
        images_test = img_variable.cuda()
        labels_test = torch.tensor([test_image_label])
        
        image = cv2.imread("/home/up202103249/Faster-RCNN_ProtoPNet_Kitty_v2/simple-faster-rcnn-pytorch/data/"+full_i+".png")
        start_point = (int(pred_bboxes_[0][0][0]), int(pred_bboxes_[0][0][1]))
        end_point = (int(pred_bboxes_[0][0][2]), int(pred_bboxes_[0][0][3]))
        image_save = cv2.rectangle(image, start_point, end_point, (255, 0, 0), 2)
        image_save = image_save[...,::-1]
        #plt.imsave("/home/up202103249/Faster-RCNN_ProtoPNet_Kitty_v2/simple-faster-rcnn-pytorch/explained/most_activated_prototypes/original_image.png", image)
        
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
        
        
        conv_output, distances = faster_rcnn_t.head.ppnet.push_forward(pool)
        min_distances=torch.Tensor(min_distances)
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
        #original_img = torchvision.transforms.Resize((112,112))(cropped_image)
        original_img=cropped_image
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
        sorted_indices_act=sorted_indices_act[0]
        array_act=array_act[0]
        for i in range(1,2):
            if prototype_img_identity[sorted_indices_act[-i].item()]==predicted_cls and prototype_activations[0][0][sorted_indices_act[-i].item()]>=2.8:
              log('top {0} activated prototype for this image:'.format(i))
              prototype1=save_prototype(os.path.join(save_analysis_path, 'most_activated_prototypes',
                                          'top-%d_activated_prototype.png' % i),
                            start_epoch_number, sorted_indices_act[-i].item())
              prototype=save_prototype_original_img_with_bbox(fname=os.path.join(save_analysis_path, 'most_activated_prototypes',
                                                                       'top-%d_activated_prototype_in_original_pimg.png' % i),
                                                     epoch=start_epoch_number,
                                                     index=sorted_indices_act[-i].item(),
                                                     bbox_height_start=prototype_info[sorted_indices_act[-i].item()][1],
                                                     bbox_height_end=prototype_info[sorted_indices_act[-i].item()][2],
                                                     bbox_width_start=prototype_info[sorted_indices_act[-i].item()][3],
                                                     bbox_width_end=prototype_info[sorted_indices_act[-i].item()][4],
                                                     color=(0, 255, 255))
              prototype_self=save_prototype_self_activation(os.path.join(save_analysis_path, 'most_activated_prototypes',
                                                           'top-%d_activated_prototype_self_act.png' % i),
                                             start_epoch_number, sorted_indices_act[-i].item(),                                                      bbox_height_start=prototype_info[sorted_indices_act[-i].item()][1],
                                             bbox_height_end=prototype_info[sorted_indices_act[-i].item()][2],
                                             bbox_width_start=prototype_info[sorted_indices_act[-i].item()][3],
                                             bbox_width_end=prototype_info[sorted_indices_act[-i].item()][4],
                                             color=(0, 255, 255))
              # log('prototype index: {0}'.format(sorted_indices_act[-i].item()))
              # log('prototype class identity: {0}'.format(prototype_img_identity[sorted_indices_act[-i].item()]))
              # if prototype_max_connection[sorted_indices_act[-i].item()] != prototype_img_identity[sorted_indices_act[-i].item()]:
              #     log('prototype connection identity: {0}'.format(prototype_max_connection[sorted_indices_act[-i].item()]))
              # log('activation value (similarity score): {0}'.format(array_act[-i]))
              # log('last layer connection with predicted class: {0}'.format( faster_rcnn_t.head.ppnet.last_layer.weight[predicted_cls][sorted_indices_act[-i].item()]))
            
              activation_pattern = prototype_activation_patterns[idx][sorted_indices_act[-i].item()].detach().cpu().numpy()
              upsampled_activation_pattern = cv2.resize(activation_pattern, dsize=(size[2], size[3]),
                                                          interpolation=cv2.INTER_CUBIC)
            
               # show the most highly activated patch of the image by this prototype
              # high_act_patch_indices = find_high_activation_crop(upsampled_activation_pattern)
              # high_act_patch = original_img[high_act_patch_indices[0]:high_act_patch_indices[1],
              #                                high_act_patch_indices[2]:high_act_patch_indices[3], :]
              # log('most highly activated patch of the chosen image by this prototype:')
              # #plt.axis('off')
              # # plt.imsave(os.path.join(save_analysis_path, 'most_activated_prototypes',
              # #                         'most_highly_activated_patch_by_top-%d_prototype.png' % i),
              # #           high_act_patch)
              # log('most highly activated patch by this prototype shown in the original image:')
              # imsave_with_bbox(fname=os.path.join(save_analysis_path, 'most_activated_prototypes',
              #                         'most_highly_activated_patch_in_original_img_by_top-%d_prototype.png' % i),
              #                 img_rgb=original_img,
              #                 bbox_height_start=high_act_patch_indices[0],
              #                 bbox_height_end=high_act_patch_indices[1],
              #                 bbox_width_start=high_act_patch_indices[2],
              #                 bbox_width_end=high_act_patch_indices[3], color=(0, 255, 255))
            
              # # show the image overlayed with prototype activation map
              rescaled_activation_pattern = upsampled_activation_pattern - np.amin(upsampled_activation_pattern)
              rescaled_activation_pattern = rescaled_activation_pattern / np.amax(rescaled_activation_pattern)
              heatmap = cv2.applyColorMap(np.uint8(255*rescaled_activation_pattern), cv2.COLORMAP_JET)
              heatmap = np.float32(heatmap) / 255
              heatmap = heatmap[...,::-1]
              overlayed_img = 0.5 * original_img + 0.3 * heatmap
              # log('prototype activation map of the chosen image:')
              # #plt.axis('off')
              # # plt.imsave(os.path.join(save_analysis_path, 'most_activated_prototypes',
              # #                         'prototype_activation_map_by_top-%d_prototype.png' % i),
              # #           overlayed_img)
              # log('--------------------------------------------------------------')
 
    img2=np.zeros((cap.shape[0],cap.shape[1],cap.shape[2]),dtype=np.uint8)
    add_100=lambda i:i+255
    vectorize_white=np.vectorize(add_100,otypes=[np.uint8])
    exp=vectorize_white(img2)
    prototype=prototype[:,:,0:3]
    #prototype=prototype*255
    prototype=prototype.astype(np.uint8)
    prototype= cv2.cvtColor(prototype_self, cv2.COLOR_RGB2BGR)
    nb = exp.shape[0]
    na = prototype.shape[0]
    we,he,_=exp.shape
    wp,hp,_=prototype.shape
    lower_w=(we) // 2 - (wp // 2)
    upper_w= (we) // 2 + (wp // 2)
    
    lower_h=(he) // 2 - (hp // 2)
    upper_h= (he) // 2 + (hp // 2)
    
    try:
        exp[lower_w:upper_w, lower_h-348:upper_h-348] = (prototype[:,:,0:3])*255
    except:
        print("error")
    
    overlayed_img= cv2.cvtColor(overlayed_img, cv2.COLOR_RGB2BGR)
    prototype1=overlayed_img*255
    #prototype1=prototype_self*255
    #prototype1=cutted
    na = prototype1.shape[0]
    we,he,_=exp.shape
    wp,hp,_=prototype1.shape
    
    lower_w=(we) // 2 - (wp // 2)
    upper_w= (we) // 2 + (wp // 2)
    
    lower_h=(he) // 2 - (hp // 2)
    upper_h= (he) // 2 + (hp // 2)
    

    if (hp % 2) != 0 and (wp % 2) == 0:
         exp[lower_w:upper_w, lower_h+348:upper_h+348+1] = prototype1 
    if (wp % 2) != 0 and (hp % 2) != 0:
         exp[lower_w:upper_w+1, lower_h+348:upper_h+348+1] = prototype1
    if (wp % 2)!= 0 and (hp % 2) == 0:
          exp[lower_w:upper_w+1, lower_h+348:upper_h+348] = prototype1
    if (hp % 2) ==0 and (wp % 2) == 0:
        exp[lower_w:upper_w, lower_h+348:upper_h+348] = prototype1
        
    Verti = np.concatenate((cap, exp), axis=0, dtype=np.uint8)
    cv2.imshow('VERTICAL', Verti)
    key=cv2.waitKey(1)
    # if key==27:
    #     break
cv2.destroyAllWindows()
