from __future__ import  absolute_import
import torch as t
from torch import nn
from torchvision.models import vgg16
from torchvision.ops import RoIPool
from torchvision.ops import RoIAlign

from region_proposal_network import RegionProposalNetwork
from faster_rcnn import FasterRCNN
import array_tool as at
from config import opt

from m_ppnet import *
import argparse
import re

from settings import base_architecture, img_size, prototype_shape, num_classes,prototype_activation_function, add_on_layers_type, experiment_run, joint_optimizer_lrs, joint_lr_step_size,warm_optimizer_lrs,last_layer_optimizer_lr,coefs,num_train_epochs, num_warm_epochs, push_start, push_epochs
from helpers import list_of_distances, make_one_hot
import push as push
from preprocess import mean, std, preprocess_input_function
import torchvision

import time
import torch
import numpy as np
from torchvision.ops import nms
from torch.nn import functional as F
import os


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    # xA = max(boxA[0], boxB[0])
    # yA = max(boxA[1], boxB[1])
    # xB = min(boxA[2], boxB[2])
    # yB = min(boxA[3], boxB[3])
    xA = max(boxA[1], boxB[1])
    #print(xA)
    yA = max(boxA[0], boxB[0])
    #print(yA)
    xB = min(boxA[3], boxB[3])
    yB = min(boxA[2], boxB[2])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


# def bbox_iou(bbox_a, bbox_b):
#     """Calculate the Intersection of Unions (IoUs) between bounding boxes.

#     IoU is calculated as a ratio of area of the intersection
#     and area of the union.

#     This function accepts both :obj:`numpy.ndarray` and :obj:`cupy.ndarray` as
#     inputs. Please note that both :obj:`bbox_a` and :obj:`bbox_b` need to be
#     same type.
#     The output is same type as the type of the inputs.

#     Args:
#         bbox_a (array): An array whose shape is :math:`(N, 4)`.
#             :math:`N` is the number of bounding boxes.
#             The dtype should be :obj:`numpy.float32`.
#         bbox_b (array): An array similar to :obj:`bbox_a`,
#             whose shape is :math:`(K, 4)`.
#             The dtype should be :obj:`numpy.float32`.

#     Returns:
#         array:
#         An array whose shape is :math:`(N, K)`. \
#         An element at index :math:`(n, k)` contains IoUs between \
#         :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
#         box in :obj:`bbox_b`.

#     """
#     if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
#         raise IndexError

#     # top left
#     tl = xp.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
#     # bottom right
#     br = xp.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])

#     area_i = xp.prod(br - tl, axis=2) * (tl < br).all(axis=2)
#     area_a = xp.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
#     area_b = xp.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
#     return area_i / (area_a[:, None] + area_b - area_i)

def loc2bbox(src_bbox, loc):
    import numpy as xp

    if src_bbox.shape[0] == 0:
        return xp.zeros((0, 4), dtype=loc.dtype)

    src_bbox = src_bbox.astype(src_bbox.dtype, copy=False)

    src_height = src_bbox[:, 2] - src_bbox[:, 0]
    src_width = src_bbox[:, 3] - src_bbox[:, 1]
    src_ctr_y = src_bbox[:, 0] + 0.5 * src_height
    src_ctr_x = src_bbox[:, 1] + 0.5 * src_width

    dy = loc[:, 0::4]
    dx = loc[:, 1::4]
    dh = loc[:, 2::4]
    dw = loc[:, 3::4]

    ctr_y = dy * src_height[:, xp.newaxis] + src_ctr_y[:, xp.newaxis]
    ctr_x = dx * src_width[:, xp.newaxis] + src_ctr_x[:, xp.newaxis]

    # import warnings
    # warnings.filterwarnings("error")
    # try:
    #     h = xp.exp(dh) * src_height[xp.newaxis]
    #     w = xp.exp(dw) * src_width[xp.newaxis]
    # except RuntimeWarning:
    #     print(h)
    #     print(dh)
    h = xp.exp(dh) * src_height[:, xp.newaxis]
    w = xp.exp(dw) * src_width[:, xp.newaxis]
    dst_bbox = xp.zeros(loc.shape, dtype=loc.dtype)
    dst_bbox[:, 0::4] = ctr_y - 0.5 * h
    dst_bbox[:, 1::4] = ctr_x - 0.5 * w
    dst_bbox[:, 2::4] = ctr_y + 0.5 * h
    dst_bbox[:, 3::4] = ctr_x + 0.5 * w

    return dst_bbox

def decom_vgg16():
    # the 30th layer of features is relu of conv5_3
    if opt.caffe_pretrain:
        model = vgg16(pretrained=False)
        if not opt.load_path:
            model.load_state_dict(t.load(opt.caffe_pretrain_path))
    else:
        model = vgg16(not opt.load_path)

    features = list(model.features)[:30]
    classifier = model.classifier

    classifier = list(classifier)
    del classifier[6]
    if not opt.use_drop:
        del classifier[5]
        del classifier[2]
    classifier = nn.Sequential(*classifier)

    # freeze top4 conv
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False

    return nn.Sequential(*features), classifier

def train_or_test(model, dataloader, target, optimizer=None, class_specific=True, use_l1_mask=True,
                   coefs=None, log=print):
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''
    is_train = optimizer is not None
    start = time.time()
    n_examples = 0
    n_correct = 0
    n_batches = 0
    total_cross_entropy = 0
    total_cluster_cost = 0
    # separation cost is meaningful only for class_specific
    total_separation_cost = 0
    total_avg_separation_cost = 0

    for i in range(1):
        input = dataloader.cuda()
        label=target.cpu()
        #label=target
        target = target.cuda()
        
        #torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            # nn.Module has implemented __call__() function
            # so no need to call .forward
            output, min_distances = model(input)

            # compute loss
            # import sklearn
            # import warnings
            # warnings.simplefilter(action='ignore', category=FutureWarning)
            #class_weights=sklearn.utils.class_weight.compute_class_weight('balanced',np.unique(label.cpu()),label.cpu().numpy())
            # class_weights=torch.tensor([0.15,0.15,0.15,0.35],dtype=torch.float)
            # criterion_weighted = nn.CrossEntropyLoss(weight=class_weights.cuda(),reduction='mean')
            
            cross_entropy = torch.nn.functional.cross_entropy(output, target)
            #cross_entropy=criterion_weighted(output, target)

            if class_specific:
                max_dist = (model.module.prototype_shape[1]
                            * model.module.prototype_shape[2]
                            * model.module.prototype_shape[3])

                # prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
                # calculate cluster cost
                #print(label)
                try:
                    prototypes_of_correct_class = torch.t(model.module.prototype_class_identity[:,label]).cuda()
                except:
                    print(model.module.prototype_class_identity[:,label])
                try:
                    inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)
                except:
                    print()
                inverted_distances_no_negative_roi=list()
                for i in range(len(label)):
                    try:
                        if label[i]!=0:
                            inverted_distances_no_negative_roi.append(inverted_distances[i])
                    except:
                        print("try1")
                inverted_distances_no_negative_roi=torch.Tensor(inverted_distances_no_negative_roi).cpu()
                cluster_cost = torch.mean(max_dist - inverted_distances_no_negative_roi)
                

                # calculate separation cost
                prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                inverted_distances_to_nontarget_prototypes, _ = \
                    torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
                inverted_distances_no_negative_roi=list()
                for i in range(len(label)):
                    try:
                        if label[i]!=0:
                            inverted_distances_no_negative_roi.append(inverted_distances_to_nontarget_prototypes[i])
                    except:
                        print("try1")
                    
                inverted_distances_no_negative_roi=torch.Tensor(inverted_distances_no_negative_roi).cpu()
                separation_cost = torch.mean(max_dist - inverted_distances_no_negative_roi)
                print('separation_cost: ', separation_cost)

                # calculate avg cluster cost
                avg_separation_cost = \
                    torch.sum(min_distances * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class, dim=1)
                avg_separation_cost = torch.mean(avg_separation_cost)
                
                if use_l1_mask:
                    l1_mask = 1 - torch.t(model.module.prototype_class_identity).cuda()
                    l1 = (model.module.last_layer.weight * l1_mask).norm(p=1)
                else:
                    l1 = model.module.last_layer.weight.norm(p=1) 

            else:
                min_distance, _ = torch.min(min_distances, dim=1)
                cluster_cost = torch.mean(min_distance)
                l1 = model.module.last_layer.weight.norm(p=1)

            # evaluation statistics
            _, predicted = torch.max(output.data, 1)
            n_examples += target.size(0)
            n_correct += (predicted== target).sum().item()
            

            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            total_cluster_cost += cluster_cost.item()
            total_separation_cost += separation_cost.item()
            total_avg_separation_cost += avg_separation_cost.item()

        # compute gradient and do SGD step
        if is_train:
            if class_specific:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy
                          + coefs['clst'] * cluster_cost
                          + coefs['sep'] * separation_cost
                          + coefs['l1'] * l1)
                else:
                    loss = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost + 1e-4 * l1
            else:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy
                          + coefs['clst'] * cluster_cost
                          + coefs['l1'] * l1)
                else:
                    loss = cross_entropy + 0.8 * cluster_cost + 1e-4 * l1
            # optimizer.zero_grad()
            # loss.backward(retain_graph=True)
            # optimizer.step()
        out=output
        del input
        del target
        del output
        del predicted
        del min_distances

    end = time.time()


    p = model.module.prototype_vectors.view(model.module.num_prototypes, -1).cpu()
    # the two lines of code below are very slow
    # with torch.no_grad():
    #     p_avg_pair_dist = torch.mean(list_of_distances(p, p))
    
    # accuracy=(n_correct / n_examples)*100
    # print("Accuracy of PPNet:"+str(accuracy))
    
    # print("Cross Entropy")
    # print(cross_entropy)
    # print("cluster_cost")
    # print(cluster_cost)
    # print("separation_cost")
    # print(separation_cost)
    #print(label)
    #print(loss)
    #loss=cluster_cost
    return out,loss
    



class FasterRCNNVGG16(FasterRCNN):
    """Faster R-CNN based on VGG-16.
    For descriptions on the interface of this model, please refer to
    :class:`model.faster_rcnn.FasterRCNN`.

    Args:
        n_fg_class (int): The number of classes excluding the background.
        ratios (list of floats): This is ratios of width to height of
            the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference
            window.

    """

    feat_stride = 16  # downsample 16x for output of conv5 in vgg16

    def __init__(self,
                 n_fg_class=3, #3
                 ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32]
                 ):
                 
        extractor, classifier = decom_vgg16()

        rpn = RegionProposalNetwork(
            512, 512,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride,
        )

        head = VGG16RoIHead(
            n_class=n_fg_class+1,
            roi_size=7,
            spatial_scale=(1. / self.feat_stride),
            classifier=classifier
        )

        super(FasterRCNNVGG16, self).__init__(
            extractor,
            rpn,
            head,
        )


class VGG16RoIHead(nn.Module):
    """Faster R-CNN Head for VGG-16 based implementation.
    This class is used as a head for Faster R-CNN.
    This outputs class-wise localizations and classification based on feature
    maps in the given RoIs.
    
    Args:
        n_class (int): The number of classes possibly including the background.
        roi_size (int): Height and width of the feature maps after RoI-pooling.
        spatial_scale (float): Scale of the roi is resized.
        classifier (nn.Module): Two layer Linear ported from vgg16

    """

    def __init__(self, n_class, roi_size, spatial_scale,
                 classifier):
        # n_class includes the background
        super(VGG16RoIHead, self).__init__()

        self.classifier = classifier
        self.cls_loc = nn.Linear(4096, n_class * 4)
        self.score = nn.Linear(4096, n_class)

        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        #self.roi = RoIPool( (self.roi_size, self.roi_size),self.spatial_scale)
        self.roi = RoIAlign(  (self.roi_size, self.roi_size),self.spatial_scale,sampling_ratio=-1)
        #self.roi_ppnet = RoIAlign( (50,50),self.spatial_scale,sampling_ratio=-1)
        from settings import base_architecture, img_size, prototype_shape, num_classes,prototype_activation_function, add_on_layers_type, experiment_run, joint_optimizer_lrs, joint_lr_step_size,warm_optimizer_lrs,last_layer_optimizer_lr,coefs,num_train_epochs, num_warm_epochs, push_start, push_epochs

        base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)

        ppnet = construct_PPNet(classifier,base_architecture=base_architecture,
                                      pretrained=True, img_size=img_size,
                                      prototype_shape=prototype_shape,
                                      num_classes=num_classes,
                                      prototype_activation_function=prototype_activation_function,
                                      add_on_layers_type=add_on_layers_type)
        #if prototype_activation_function == 'linear':
        #    ppnet.set_last_layer_incorrect_connection(incorrect_strength=0)
        ppnet = ppnet.cuda()
        ppnet_multi = torch.nn.DataParallel(ppnet)
        class_specific = True

        # define optimizer
        #from model.settings import joint_optimizer_lrs, joint_lr_step_size
        joint_optimizer_specs = \
        [{'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3}, # bias are now also being regularized
         {'params': ppnet.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
         {'params': ppnet.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
        ]
        joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
        joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=joint_lr_step_size, gamma=0.1)

        #from settings import warm_optimizer_lrs
        warm_optimizer_specs = \
        [{'params': ppnet.add_on_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
         {'params': ppnet.prototype_vectors, 'lr': warm_optimizer_lrs['prototype_vectors']},
        ]
        warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

        #from settings import last_layer_optimizer_lr
        last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
        last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

        # weighting of different training losses
        #from settings import coefs

        # number of training epochs, number of warm epochs, push start epoch, push epochs
        #from settings import num_train_epochs, num_warm_epochs, push_start, push_epochs


        self.ppnet=ppnet
        self.ppnet_multi=ppnet_multi
        
        
    def forward(self, x, rois, roi_indices,o_bbox=None,o_label=None,epoch=None,iteration=None,gt_roi_scores=None,imgs=None,optimizer_ppnet=None,global_min_proto_dist=None,global_min_fmap_patches=None,push_iteration=None,evaluate=True):
        """Forward the chain.

        We assume that there are :math:`N` batches.

        Args:
            x (Variable): 4D image variable.
            rois (Tensor): A bounding box array containing coordinates of
                proposal boxes.  This is a concatenation of bounding box
                arrays from multiple images in the batch.
                Its shape is :math:`(R', 4)`. Given :math:`R_i` proposed
                RoIs from the :math:`i` th image,
                :math:`R' = \\sum _{i=1} ^ N R_i`.
            roi_indices (Tensor): An array containing indices of images to
                which bounding boxes correspond to. Its shape is :math:`(R',)`.

        """
        from settings import base_architecture, img_size, prototype_shape, num_classes,prototype_activation_function, add_on_layers_type, experiment_run, joint_optimizer_lrs, joint_lr_step_size,warm_optimizer_lrs,last_layer_optimizer_lr,coefs,num_train_epochs, num_warm_epochs, push_start, push_epochs

        
        class_specific = True
                # define optimizer
        #from model.settings import joint_optimizer_lrs, joint_lr_step_size
        joint_optimizer_specs = \
        [{'params': self.ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-4}, # bias are now also being regularized
         {'params': self.ppnet.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-4},
         {'params': self.ppnet.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
        ]
        joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
        joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=joint_lr_step_size, gamma=0.1)

        # in case roi_indices is  ndarray
        roi_indices = at.totensor(roi_indices).float()
        rois = at.totensor(rois).float()
        indices_and_rois = t.cat([roi_indices[:, None], rois], dim=1)
        # NOTE: important: yx->xy
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois =  xy_indices_and_rois.contiguous()
        
        # pool_v2=np.zeros((len(xy_indices_and_rois[:,0]),512,len(x[0,0,:,:]),len(x[0,0,0,:])), dtype=float)
        # pool_v2=t.from_numpy(pool_v2).cuda()
        # b = np.zeros([len(x[0,0,:,:]), len(x[0,0,0,:])]).astype(float)
        # b=t.from_numpy(b).cuda()
        
        # for i in range(len(xy_indices_and_rois[:,0])):
            
        #     x_min=int(xy_indices_and_rois[i][1]*1/16)
        #     y_min=int(xy_indices_and_rois[i][2]*1/16)
        #     x_max=int(xy_indices_and_rois[i][3]*1/16)
        #     y_max=int(xy_indices_and_rois[i][4]*1/16)
        #     b[y_min: y_max,x_min:x_max]=1
        #     #pool_v2[i,:,:,:]=x[0,i,:,:]*b
        #     for a in range(0,512):
        #         pool_v2[i,a,:,:]=t.from_numpy(x[0,a,:,:].cpu().detach().numpy()*b.cpu().numpy())
        
        # # #pool_v2=torch.from_numpy(pool_v2)
        # pool_v2=pool_v2.type(torch.FloatTensor).cuda()
        pool = self.roi(x, indices_and_rois)
        # with torch.no_grad():
        #    x=x.reshape(1,512,46,38)
          
        # xy_indices_and_rois_alternative=torch.ones([128,5])
       
        # for i in range(len(xy_indices_and_rois)):
        #     x_scale =  46 / (xy_indices_and_rois[i,1:5][2]-xy_indices_and_rois[i,1:5][0])
        #     y_scale = 38 / (xy_indices_and_rois[i,1:5][3]-xy_indices_and_rois[i,1:5][1])
            
        #     x1 = xy_indices_and_rois[i,1:5][0].cpu() * x_scale.cpu()
        #     y1 = xy_indices_and_rois[i,1:5][1].cpu() * y_scale.cpu()
        #     xmax = xy_indices_and_rois[i,1:5][2].cpu() * x_scale.cpu()
        #     ymax = xy_indices_and_rois[i,1:5][3].cpu() * y_scale.cpu()
        #     xy_indices_and_rois_alternative[i,1:5]=torch.Tensor([x1,y1,xmax,ymax]).type(torch.FloatTensor).cuda()
        # with torch.no_grad():
        #    x=x.reshape(1,512,46,38)
        # pool_for_ppnet= self.roi(x, xy_indices_and_rois_alternative.cuda())
        #pool_for_ppnet=self.roi_ppnet(x, indices_and_rois)
       
        #pool_ppnet=self.ppnet(pool)
        if evaluate==True:
            
            #pool_ppnet=self.ppnet(pool)
            pool_ppnet, min_distances=self.ppnet(pool)
            #_,predicted=torch.max(pool_ppnet.data, 1)
            conv_output, distances = self.ppnet.push_forward(pool)
            pool = pool.view(pool.size(0), -1)
            fc7 = self.classifier(pool)
            roi_cls_locs = self.cls_loc(fc7)
            #roi_scores = self.score(fc7)
            roi_scores=pool_ppnet
            
            return roi_cls_locs, roi_scores,pool_ppnet, min_distances,conv_output, distances
            
        if evaluate==False:
            
            #output que sao os rois e loss
            pool_ppnet,o_ppnet_loss=train_or_test(self.ppnet_multi, pool,gt_roi_scores, optimizer=optimizer_ppnet, class_specific=True, use_l1_mask=True,coefs=None, log=print)
            print('loss ppnet:  ', o_ppnet_loss)
            #pool_ppnet=self.ppnet(pool)
            #_,predicted=torch.max(pool_ppnet.data, 1)
            pool_push=pool
            pool = pool.view(pool.size(0), -1)
            fc7 = self.classifier(pool)
            o_roi_cls_locs = self.cls_loc(fc7)
            roi_scores = self.score(fc7)
            o_roi_scores=pool_ppnet
            

            #if epoch==11 and iteration % 2==0: #neste bloco: bounding box manipulation, non-maximum suppression
            if epoch==42 or epoch==82:
                img_dir=None
                prototype_img_filename_prefix = 'prototype-img'
                prototype_self_act_filename_prefix = 'prototype-self-act'
                proto_bound_boxes_filename_prefix = 'bb'
                log=None
                scale=1
                fc7 = self.classifier(pool)
                roi_cls_locs = self.cls_loc(fc7)
                roi_scores=pool_ppnet
                #roi_cls_loc, roi_scores, rois, _ = self(imgs, scale=1)
                # We are assuming that batch size is 1.
                roi_score = roi_scores.data
                roi_cls_loc = roi_cls_locs.data
                roi = at.totensor(rois) / scale  

                # Convert predictions to bounding boxes in image coordinates.
                # Bounding boxes are scaled to the scale of the input images.
                loc_normalize_mean = (0., 0., 0., 0.)
                loc_normalize_std = (0.1, 0.1, 0.2, 0.2)
                n_class=4 #4
                mean = t.Tensor(loc_normalize_mean).cuda(). \
                    repeat(n_class)[None]
                std = t.Tensor(loc_normalize_std).cuda(). \
                    repeat(n_class)[None]

                roi_cls_loc = (roi_cls_loc * std + mean)
                roi_cls_loc = roi_cls_loc.view(-1, n_class, 4)
                roi = roi.view(-1, 1, 4).expand_as(roi_cls_loc)
                cls_bbox = loc2bbox(at.tonumpy(roi).reshape((-1, 4)),
                                    at.tonumpy(roi_cls_loc).reshape((-1, 4)))
                cls_bbox = at.totensor(cls_bbox)
                cls_bbox = cls_bbox.view(-1, n_class * 4)
                # clip bounding box
                cls_bbox[:, 0::2] = (cls_bbox[:, 0::2]).clamp(min=0, max=1224)
                cls_bbox[:, 1::2] = (cls_bbox[:, 1::2]).clamp(min=0, max=370)

                prob = (F.softmax(at.totensor(roi_score), dim=1))
                
                bbox = list()
                label = list()
                score = list()
                # skip cls_id = 0 because it is the background class
                raw_cls_bbox=cls_bbox
                raw_prob=prob
                background=0
                
                for l in range(0, n_class):
                    cls_bbox_l = raw_cls_bbox.reshape((-1, n_class, 4))[:, l, :]
                    prob_l = raw_prob[:, l]
                    mask = prob_l > 0.7
                    cls_bbox_l = cls_bbox_l[mask]
                    prob_l = prob_l[mask]
                    keep = nms(cls_bbox_l, prob_l,0.3)
                    single_label=l
                    # keep=[0,1]
                    # cls_bbox_l=raw_cls_bbox.reshape((-1, n_class, 4))[:, l, :]
                   
                    if len(keep)!=0:
                        for i in range(len(keep)):
                           
                                bbox_1=cls_bbox_l[keep[i],:]
                                
                                image=imgs[:,:,int(bbox_1[0]):int(bbox_1[2]),int(bbox_1[1]):int(bbox_1[3])]
                                if image.size()[2]!=0 and image.size()[3]!=0:
                                    image=torchvision.transforms.Resize((112,112))(image)
                                else:
                                    continue
                                # for a in range(len(o_bbox)):
                                #     iou=bb_intersection_over_union(o_bbox[a],bbox_1)
                                #     w=xy_indices_and_rois[keep[i],1:5][2]-xy_indices_and_rois[keep[i],1:5][0]
                                #     h=xy_indices_and_rois[keep[i],1:5][3]-xy_indices_and_rois[keep[i],1:5][1]
                                #     sq=RoIAlign((h,w),self.spatial_scale,sampling_ratio=-1)
                                #     f_pool=sq(x,indices_and_rois[0].unsqueeze(0))
                                ratio=0
                                   
                                for a in range(len(o_bbox)):
                                    iou=bb_intersection_over_union(o_bbox[a],bbox_1)   
                                   
                                    if iou>=0.70 and o_label[a]+1==l or l==0 and  background==0:
                                        if background==0:
                                            background=1
                                       
                                        push.push_prototypes(
                                        ratio,
                                        iteration,
                                        global_min_proto_dist,
                                        global_min_fmap_patches,
                                        image,
                                        pool_push[keep[i]].unsqueeze(0),
                                        single_label,# pytorch dataloader (must be unnormalized in [0,1])
                                        prototype_network_parallel=self.ppnet_multi, # pytorch network with prototype_vectors
                                        class_specific=class_specific,
                                        preprocess_input_function=preprocess_input_function, # normalize if needed
                                        prototype_layer_stride=1,
                                        root_dir_for_saving_prototypes=img_dir, # if not None, prototypes will be saved here
                                        epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
                                        prototype_img_filename_prefix=prototype_img_filename_prefix,
                                        prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                                        proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
                                        save_prototype_class_identity=True,
                                        log=log,
                                        )
                
                    #     prototype_update = np.reshape(global_min_fmap_patches,
                    #                                   tuple(prototype_shape))
                    #     self.ppnet_multi.module.prototype_vectors.data.copy_(torch.tensor(prototype_update, dtype=torch.float32).cuda())
                    #     print("vectors are updated")
                     # prototype_network_parallel.cuda()
                            
                    # import ipdb;ipdb.set_trace()
                    # keep = cp.asnumpy(keep)
                    bbox.append(cls_bbox_l[keep].cpu().numpy())
                    # The labels are in [0, self.n_class - 2].
                    label.append((l - 1) * np.ones((len(keep),)))
                    score.append(prob_l[keep].cpu().numpy())
                bbox = np.concatenate(bbox, axis=0).astype(np.float32)
                label = np.concatenate(label, axis=0).astype(np.int32)
                score = np.concatenate(score, axis=0).astype(np.float32)
                if any(global_min_fmap_patches[0]!=0) and any(global_min_fmap_patches[1]!=0) and  any(global_min_fmap_patches[2]!=0) and  any(global_min_fmap_patches[3]!=0) and iteration==5984:
                    if epoch==82:
                        proto_epoch_dir="/home/up202003072/Documents/faster_protoPnet/proto/epoch-82"
                    if epoch==42:
                        proto_epoch_dir="/home/up202003072/Documents/faster_protoPnet/proto/epoch-42"
                    if proto_epoch_dir != None and proto_bound_boxes_filename_prefix != None:
                        np.save(os.path.join(proto_epoch_dir, proto_bound_boxes_filename_prefix + '-receptive_field' + str(epoch) + '.npy'),
                                self.ppnet_multi.module.proto_rf_boxes)
                        np.save(os.path.join(proto_epoch_dir, proto_bound_boxes_filename_prefix + str(epoch) + '.npy'),
                                self.ppnet_multi.module.proto_bound_boxes)
                    prototype_update = np.reshape(global_min_fmap_patches,
                                                  tuple(prototype_shape))
                   
                    self.ppnet_multi.module.prototype_vectors.data.copy_(torch.tensor(prototype_update, dtype=torch.float32).cuda())
                    print("vectors are updated")
                      
                    

            return o_roi_cls_locs, o_roi_scores,o_ppnet_loss
            

        
            # else:
            #     pool_ppnet=self.ppnet(pool)
            #     pool = pool.view(pool.size(0), -1)
            #     fc7 = self.classifier(pool)
            #     roi_cls_locs = self.cls_loc(fc7)
            #     roi_scores = self.score(fc7)
            #     roi_scores=pool_ppnet[0]
            #     return roi_cls_locs, roi_scores
                
                

            
            
                    
        
      


def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
