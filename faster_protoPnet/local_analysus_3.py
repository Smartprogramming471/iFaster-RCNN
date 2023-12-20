from __future__ import  absolute_import
import os

import ipdb
import matplotlib
from tqdm import tqdm

from config import opt
from dataset_main import Dataset, TestDataset, inverse_normalize
from faster_rcnn_vgg16 import FasterRCNNVGG16
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
import array_tool as at
from vis_tool import visdom_bbox, vis_bbox
from eval_tool import eval_detection_voc
from settings import base_architecture, img_size, prototype_shape, num_classes,prototype_activation_function, add_on_layers_type, experiment_run, joint_optimizer_lrs, joint_lr_step_size,warm_optimizer_lrs,last_layer_optimizer_lr,coefs,num_train_epochs, num_warm_epochs, push_start, push_epochs
from util import  read_image
from torch.utils.data import WeightedRandomSampler, DataLoader

# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
import resource
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
#from utils_kitty.datasets import*
import torchvision
import cv2

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

#matplotlib.use('agg')


def eval(dataloader, faster_rcnn, trainer, test_num=2757):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels = list(), list()
    for ii, (img, sizes, gt_bboxes_, gt_labels_) in tqdm(enumerate(dataloader)):
        #sizes = [sizes[0][0].item(), sizes[1][0].item()] ##comment sizes
        if len(gt_labels_)==0:
            continue
        o_img=img
        img = inverse_normalize(at.tonumpy(img[0]))
        # print(type(img)) #numpy array
        img = torch.from_numpy(img)[None] #experimentar isto e tirar o [img] brakets 
        # print(type(img)) #torch.tensor #torch tensor
        
        pred_bboxes_, pred_labels_, pred_scores_ ,logits,min_distances,conv_output,distances= faster_rcnn.predict(img, visualize = True)#antes (img, [sizes]) e descom os sizes visualiza = false
        
        print("predicted")
        try:
            print(torch.argmax(torch.Tensor(logits)[0], dim=1))
        except:
            print("empty")
        print("scores")
        print(pred_scores_)
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        
        if ii == test_num: break
    #cac calculate the evaluation metrics based on the predicted bounding boxes, labels, scores, and the ground truth boxes and labels.
    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels,
        use_07_metric=True)
    return result


def train(**kwargs):
    opt._parse(kwargs)

    dataset = Dataset(opt)
    # #WeightedRandomSampler

    # bone_class_sample_count = [14485, 640, 368] #0,1,2 # sem dataaug [14451, 648, 2721, 455, 378] 
    # bone_class_weights = 1 / torch.tensor(bone_class_sample_count)

    # weights_per_sample = [bone_class_weights[x[2]] for x in dataset]
    # # Convert the list of tensors to a list of scalar values
    # weights_per_sample = [weight[0].item() for weight in weights_per_sample]

    # # Convert the list to a tensor
    # weights_per_sample = torch.tensor(weights_per_sample)

    # bone_sampler = WeightedRandomSampler(weights_per_sample, num_samples = len(weights_per_sample), replacement=True)

    print('load data')
    #dataloader = data_.DataLoader(dataset,batch_size=1, sampler=bone_sampler, num_workers=opt.num_workers)
    dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=True, \
                                  # pin_memory=True,
                                  num_workers=opt.num_workers)
    testset = TestDataset(opt)
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False, \
                                       pin_memory=True
                                       )
    faster_rcnn = FasterRCNNVGG16().cuda()
    repetitions=100
    total_tome=0
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)
    #trainer.vis.text(dataset.db.label_names, win='labels')

    # best_map = 0
    
    # joint_optimizer_specs = \
    # [{'params': faster_rcnn.head.ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3}, # bias are now also being regularized
    #  {'params': faster_rcnn.head.ppnet.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
    #  {'params': faster_rcnn.head.ppnet.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
    # ]
    # joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
    # joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=joint_lr_step_size, gamma=0.1)
    # #optimizer_ppnet=joint_optimizer 
    # last_layer_optimizer_specs = [{'params': faster_rcnn.head.ppnet.last_layer.parameters(), 'lr':0.001}]
    # last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)
    
    # lr_ = opt.lr
    # maps=list()
    
    # prototype_shape =faster_rcnn.head.ppnet_multi.module.prototype_shape
    # n_prototypes =faster_rcnn.head.ppnet_multi.module.num_prototypes
    # global_min_proto_dist = np.full(n_prototypes, np.inf)
    # # saves the patch representation that gives the current smallest distance
    # push_iteration=np.zeros([1])
    # global_min_fmap_patches = np.zeros(
    #     [n_prototypes,
    #      prototype_shape[1],
    #      prototype_shape[2],
    #      prototype_shape[3]])
    

    #load para o local analysys 
    #best_path="/home/up202003072/Documents/faster_protoPnet/checkpoints/fasterrcnn_06282136_0.5586111118938318" 
    best_path="/home/up202003072/Documents/faster_protoPnet/checkpoints/fasterrcnn_08121839_0.6605683802020664" 
    
    # for epoch in range(opt.epoch):
    #     trainer.reset_meters()
    #     print("Epoch:")
    #     print(epoch)
    #     iteration=0
    #     ep=epoch
    #     for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
    #         if len(label_)==0:
    #             continue
    #         scale = at.scalar(scale)
    #         img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
    #         if ep==7 or epoch==12:
    #             global_min_proto_dist = np.full(n_prototypes, np.inf)
    #             global_min_fmap_patches = np.zeros(
    #                 [n_prototypes,
    #                   prototype_shape[1],
    #                   prototype_shape[2],
    #                   prototype_shape[3]])
    #             optimizer=last_layer_optimizer
    #         else :
    #             optimizer=joint_optimizer
            
    #         iteration=ii
    #         trainer.train_step(img, bbox, label, scale,ep,iteration,optimizer,global_min_proto_dist,global_min_fmap_patches)
    #         iteration=iteration+1

    #         # if (ii + 1) % opt.plot_every == 0:
    #         #     print(" Training Loss:")
    #         #     print(trainer.get_meter_data())
    #         #     # eval_result = eval(test_dataloader, faster_rcnn, trainer, test_num=opt.test_num)
    #         #     # print("mAP")
    #         #     #print(eval_result["map"])
    #         #     if os.path.exists(opt.debug_file):
    #         #         ipdb.set_trace()

    #         #     # plot loss
    #         #     trainer.vis.plot_many(trainer.get_meter_data())

    #         #     # plot groud truth bboxes
    #         #     ori_img_ = inverse_normalize(at.tonumpy(img[0]))
    #         #     gt_img = visdom_bbox(ori_img_,
    #         #                          at.tonumpy(bbox_[0]),
    #         #                          at.tonumpy(label_[0]))
    #         #     trainer.vis.img('gt_img', gt_img)

    #         #     # plot predicti bboxes
    #         #     _bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_], visualize=True)
    #         #     pred_img = visdom_bbox(ori_img_,
    #         #                            at.tonumpy(_bboxes[0]),
    #         #                            at.tonumpy(_labels[0]).reshape(-1),
    #         #                            at.tonumpy(_scores[0]))
    #         #     trainer.vis.img('pred_img', pred_img)

    #         #     # rpn confusion matrix(meter)
    #         #     trainer.vis.text(str(trainer.rpn_cm.value().tolist()), win='rpn_cm')
    #         #     # roi confusion matrix
    #         #     trainer.vis.img('roi_cm', at.totensor(trainer.roi_cm.conf, False).float())
                
    #     print(" Training Loss:")
    #     print(trainer.get_meter_data()["total_loss"])
    #     print("testing")
    #     eval_result = eval(test_dataloader, faster_rcnn, trainer,test_num=2757)
    #     print("mAP")
    #     print(eval_result["map"])
    #     print("ap")
    #     print(eval_result["ap"])
    #     maps.append(eval_result["map"])
        
    #     print("Loss:")
    #     print(trainer.get_meter_data())

    #     if eval_result['map'] > best_map:
    #         best_map = eval_result['map']
    #         best_path = trainer.save(best_map=best_map)
    #         print("Saved")
    #     if epoch == 70:
    #         trainer.load(best_path)
    #         trainer.faster_rcnn.scale_lr(opt.lr_decay)
    #         lr_ = lr_ * opt.lr_decay
    
    #LOCAL ANALYSIS PRODUÇÃO DE EXPLICABILIDADE----------------------------------------------

    faster_rcnn_t= FasterRCNNVGG16()
    
    trainer.load(best_path)
    faster_rcnn_t=trainer.faster_rcnn

    # #specify the test image to be analyzed
    test_image_dir = "/home/up202003072/Documents/GRAZPEDWRI-DX/pascalvoc/images/"#'DIRETORIO ONDE TEM A IMAGEM QE PRETENDEMOS ANALISAR

    test_image_name = "2241_1074115078_02_WRI-R2_M012.png"

    test_image_label = 1 #label: 0,1,2,3 
    test_image_path = os.path.join(test_image_dir, test_image_name)

    ##### HELPER FUNCTIONS FOR PLOTTING
    import matplotlib.pyplot as plt
    from matplotlib import pyplot as plot
    import cv2
    from PIL import Image
    load_model_dir="/home/up202003072/Documents/faster_protoPnet/proto"
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
    
      # load the test image and forward it through the network_______________________________________________--
    from preprocess import mean, std, preprocess_input_function, undo_preprocess_input_function
    from log import create_logger
    save_analysis_path="/home/up202003072/Documents/faster_protoPnet/explained/"
    log, logclose = create_logger(log_filename=os.path.join(save_analysis_path, 'local_analysis.log'))
    normalize = transforms.Normalize(mean=mean,
                                      std=std)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        normalize
     ])
    
 
    img_cv = cv2.imread(test_image_path) 
    img_pil = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_tensor = preprocess(img_pil).permute(0, 1, 2)
    img_variable = Variable(img_tensor.unsqueeze(0)) 
    
    
    img = read_image(test_image_path)
    img = torch.from_numpy(img)[None] #C, H, W

    #idd=0
    pred_bboxes_, pred_labels_, pred_scores_ ,logits,min_distances,_,_= faster_rcnn_t.predict(img, visualize = True)
    print(pred_bboxes_,pred_labels_, pred_scores_, logits)
    cropped_image=img_variable[:,:,int(pred_bboxes_[0][0][0]):int(pred_bboxes_[0][0][2]),int(pred_bboxes_[0][0][1]):int(pred_bboxes_[0][0][3])] 
    images_test = img_variable.cuda()
    labels_test = torch.tensor([test_image_label])


    image = cv2.imread("/home/up202003072/Documents/GRAZPEDWRI-DX/pascalvoc/images/"+test_image_name)
    start_point = (int(pred_bboxes_[0][0][1]), int(pred_bboxes_[0][0][0]))
    end_point = (int(pred_bboxes_[0][0][3]), int(pred_bboxes_[0][0][2]))
    image = cv2.rectangle(image, start_point, end_point, (255, 0, 0), 2)
    image = image[...,::-1] 
    plt.imsave("/home/up202003072/Documents/faster_protoPnet/explained/most_activated_prototypes/original_image.png", image)
    
    epoch_number_str=82
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
    # print(logits[0][1])
    #print(type(logits[0][1]))
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
    original_img=cropped_image #torch.Size([1, 3, 96, 150])
    size=original_img.size()
    original_img = original_img.numpy()
    original_img = np.transpose(original_img[0], (2, 1, 0)) #(150, 96, 3)
    # print(original_img.shape)
    original_img=normalize_function(original_img)
    
     ##### MOST ACTIVATED (NEAREST) 10 PROTOTYPES OF THIS IMAGE
    from helpers import makedir, find_high_activation_crop
    makedir(os.path.join(save_analysis_path, 'most_activated_prototypes'))
    
    start_epoch_number=82
    epoch_number_str=82
    prototype_info = np.load(os.path.join(load_img_dir, 'epoch-'+str(epoch_number_str), 'bb'+str(epoch_number_str)+'.npy'))
    prototype_img_identity = prototype_info[:, -1]
    prototype_max_connection = torch.argmax(faster_rcnn_t.head.ppnet.last_layer.weight, dim=0)
    log('Most activated 10 prototypes of this image:')
    array_act, sorted_indices_act = torch.sort(prototype_activations[idx])
    sorted_indices_act=sorted_indices_act[0]
    array_act=array_act[0]
    for i in range(1,11):
        if prototype_img_identity[sorted_indices_act[-i].item()]==predicted_cls:
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
         high_act_patch_indices = find_high_activation_crop(upsampled_activation_pattern) #(0, 109, 0, 79)
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
        #  print(original_img.shape)
        #  print(heatmap.shape)
         overlayed_img = 0.5 * original_img + 0.3 * heatmap
         log('prototype activation map of the chosen image:')
         #plt.axis('off')
         plt.imsave(os.path.join(save_analysis_path, 'most_activated_prototypes',
                                 'prototype_activation_map_by_top-%d_prototype.png' % i),
                    overlayed_img)
         log('--------------------------------------------------------------')
    



if __name__ == '__main__':
    train(env='fasterrcnn',plot_every='100')


