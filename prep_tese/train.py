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
from vis_tool import visdom_bbox
from eval_tool import eval_detection_voc
import torch
from torch.utils.data import WeightedRandomSampler, DataLoader

# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')


def eval(dataloader, faster_rcnn, trainer, test_num=2757):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels = list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_) in tqdm(enumerate(dataloader)):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        # pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        # if len(gt_labels_)==0:
        #     continue
        # o_img=img
        # img = inverse_normalize(at.tonumpy(img[0]))
        
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        #gt_difficults += list(gt_difficults_.numpy())
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

    bone_class_sample_count = [14485, 640, 368] #0,1,2 # sem dataaug [14451, 648, 2721, 455, 378] 
    bone_class_weights = 1 / torch.tensor(bone_class_sample_count)

    weights_per_sample = [bone_class_weights[x[2]] for x in dataset]

    weights_per_sample = [weight[0].item() for weight in weights_per_sample]

    weights_per_sample = torch.tensor(weights_per_sample)

    bone_sampler = WeightedRandomSampler(weights_per_sample, num_samples = len(weights_per_sample), replacement=True)

    print('load data')
    # dataloader = data_.DataLoader(dataset, \
    #                               batch_size=1, \
    #                               shuffle=True, \
    #                               # pin_memory=True,
    #                               num_workers=opt.num_workers)
    dataloader = data_.DataLoader(dataset,batch_size=1, sampler=bone_sampler, num_workers=opt.num_workers)  #shuffle=True, nao posso usar ao mesmo tempo que o sampler por si ja faz shuffle
    testset = TestDataset(opt)
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False, \
                                       pin_memory=True
                                       )
    faster_rcnn = FasterRCNNVGG16()
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    #trainer = FasterRCNNTrainer(faster_rcnn)
    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)
    trainer.vis.text(dataset.db.label_names, win='labels')
    best_map = 0
    lr_ = opt.lr
    maps=list()
    for epoch in range(opt.epoch):
        trainer.reset_meters()
        print("Epoch:")
        print(epoch)
        for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
            scale = at.scalar(scale)
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            #img, bbox, label = img.float(), bbox_, label_
            trainer.train_step(img, bbox, label, scale)
            #print(trainer.get_meter_data())
            if (ii + 1) % opt.plot_every == 0:
                print(" Training Loss:")
                print(trainer.get_meter_data())
                # eval_result = eval(test_dataloader, faster_rcnn, trainer, test_num=opt.test_num)
                # print("mAP_train")
                # print(eval_result["map"])
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()

                # plot loss
                trainer.vis.plot_many(trainer.get_meter_data())

                # plot groud truth bboxes
                ori_img_ = inverse_normalize(at.tonumpy(img[0]))
                gt_img = visdom_bbox(ori_img_,
                                     at.tonumpy(bbox_[0]),
                                     at.tonumpy(label_[0]))
                trainer.vis.img('gt_img', gt_img)

                # plot predicti bboxes
                _bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_], visualize=True)
                pred_img = visdom_bbox(ori_img_,
                                       at.tonumpy(_bboxes[0]),
                                       at.tonumpy(_labels[0]).reshape(-1),
                                       at.tonumpy(_scores[0]))
                trainer.vis.img('pred_img', pred_img)

                # rpn confusion matrix(meter)
                trainer.vis.text(str(trainer.rpn_cm.value().tolist()), win='rpn_cm')
                # roi confusion matrix
                trainer.vis.img('roi_cm', at.totensor(trainer.roi_cm.conf, False).float())
                

        print("testing")        
        eval_result = eval(test_dataloader, faster_rcnn, trainer, test_num=opt.test_num)
        trainer.vis.plot('test_map', eval_result['map'])
        lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']
        log_info = 'lr:{}, map:{},ap:{},loss:{}'.format(str(lr_),
                                                  str(eval_result['map']),
                                                  str(eval_result['ap']),#
                                                  str(trainer.get_meter_data()))
        print("mAP_t")
        print(eval_result["map"])
        print("ap")
        print(eval_result["ap"])
        maps.append(eval_result["map"])

        print("Loss:")
        print(trainer.get_meter_data())
        
        trainer.vis.log(log_info)

        if eval_result['map'] > best_map:
            best_map = eval_result['map']
            best_path = trainer.save(best_map=best_map)
        if epoch == 70: #80
            trainer.load(best_path)
            trainer.faster_rcnn.scale_lr(opt.lr_decay)
            lr_ = lr_ * opt.lr_decay

        #if epoch == 13: 
            #break


if __name__ == '__main__':
    import fire

    fire.Fire()
