# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb, sys
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pickle
sys.path.append(os.getcwd()+"/lib")
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.roi_layers.nms import nms
from nms import nms as nms2
from model.rpn.bbox_transform import clip_boxes
from model.rpn.bbox_transform import bbox_transform_inv
from model.rpn.bbox_transform import center_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
from model.utils.blob import im_list_to_blob

import pdb

try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dset', choices=['gta', 'kitti'], default='kitti',
                        help='Generate GTA or KITTI dataset')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='kitti', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfgs/vgg16.yml', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='res101', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models', default="models",
                        type=str)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large imag scale',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument('--parallel_type', dest='parallel_type',
                        help='which part of model to parallel, 0: all, '
                             '1: model before roi pooling',
                        default=0, type=int)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=300, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load network',
                        default=100, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load network',
                        default=175, type=int)
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        default=True)
    parser.add_argument('--anno', dest='anno',
                        choices=['val', 'test', 'train'],
                        default='val', type=str)
    parser.add_argument('--web', dest='webcam',
                        help="use webcam as input",
                        default=True)
    parser.add_argument('--class', dest='num_classes',
                        default=2, type=int)
    args = parser.parse_args()
    return args


lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

class Frame_Iter:
    def __init__(self, path):
        self.frames = []
        for i in os.listdir(os.getcwd()+path):
            if ".png" in i or ".jpg" in i:
                #print(os.getcwd()+path+i)
                self.frames.append(cv2.imread(os.getcwd()+path+i))
        self.count = 0
        self.size = len(self.frames)
    def read(self):
        if self.count < self.size:
            self.count += 1
            return True, self.frames[self.count-1]
        return False, None

def test_net(webcam):
    args = parse_args()
    args.cuda = True
    if torch.cuda.is_available() and not args.cuda:
        print(
            "WARNING: You have a CUDA device, so you should probably run with "
            "--cuda")

    np.random.seed(cfg.RNG_SEED)
    if args.dataset == "pascal_voc":
        args.imdb_name = "voc_2007_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS',
                         '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        cfg.DATASET = 'pascal_voc'
    elif args.dataset == "pascal_voc_0712":
        args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS',
                         '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        cfg.DATASET = 'pascal_voc'
    elif args.dataset == "coco":
        args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
        args.imdbval_name = "coco_2014_minival"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS',
                         '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
        cfg.DATASET = 'coco'
    elif args.dataset == "imagenet":
        args.imdb_name = "imagenet_train"
        args.imdbval_name = "imagenet_val"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS',
                         '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
        cfg.DATASET = 'imagenet'
    elif args.dataset == "kitti":
        # train sizes: train, smalltrain, minitrain
        # train scale: ['150-50-20', '150-50-50', '500-150-80', 
        # '750-250-150', '1750-700-450', '1600-400-20']
        args.imdb_name = "kitti_training"
        args.imdbval_name = "kitti_testing"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS',
                         '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
    elif args.dataset == "vg":
        # train sizes: train, smalltrain, minitrain
        # train scale: ['150-50-20', '150-50-50', '500-150-80', 
        # '750-250-150', '1750-700-450', '1600-400-20']
        args.imdb_name = "vg_150-50-50_minitrain"
        args.imdbval_name = "vg_150-50-50_minival"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS',
                         '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
    elif args.dataset == 'gta_det':
        args.imdb_name = 'gta_det_train'
        args.imdbval_name = 'gta_det_' + args.anno
        args.set_cfgs = ['ANCHOR_SCALES', '[2, 4, 8, 16]',
                         'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES',
                         '100']
        cfg.BINARY_CLASS = True
        cfg.USE_DEBUG_SET = False
        cfg.ANNO_PATH = args.anno
        cfg.TRAIN.USE_FLIPPED = False  # not ready
        cfg.USE_GPU_NMS = True
        cfg.RUN_POSE = True
        cfg.FOCAL = 935.3074360871937
        cfg.DATASET = 'gta'

    args.cfg_file = "cfgs/{}_ls.yml".format(
        args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    print('Called with args:')
    print(args)

    cfg.TRAIN.USE_FLIPPED = False

    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdbval_name,
                                                          False)
    imdb.competition_mode(on=True)
    print('{:d} roidb entries'.format(len(roidb)))

    input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(input_dir):
        raise Exception(
            'There is no input directory for loading network from ' + input_dir)
    load_name = os.path.join(input_dir,
                             'faster_rcnn_{}_{}_{}.pth'.format(
                                 args.checksession, args.checkepoch,
                                 args.checkpoint))
    output_dir = 'vis' + \
                 '/faster_rcnn_{}_{}_{}/'.format(args.checksession,
                                                 args.checkepoch,
                                                 args.checkpoint)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # initilize the network here.
    if args.net == 'vgg16':
        fasterRCNN = vgg16(imdb.classes, pretrained=False,
                           class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        fasterRCNN = resnet(imdb.classes, 101, pretrained=False,
                            class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        fasterRCNN = resnet(imdb.classes, 50, pretrained=False,
                            class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
        fasterRCNN = resnet(imdb.classes, 152, pretrained=False,
                            class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()

    print("load checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']

    print('load model successfully!')
    # initialize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    fixed_center = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()
        fixed_center = fixed_center.cuda()
    
    image_vars = [im_data, im_info, num_boxes, gt_boxes, fixed_center]
    
    if args.cuda:
        cfg.CUDA = True

    if args.cuda:
        fasterRCNN.cuda()

    start = time.time()

    vis = args.vis

    if vis:
        thresh = 0.6
    else:
        thresh = 0.6

    # save_name = 'faster_rcnn_10'
    if webcam:
        data_iter = cv2.VideoCapture(0) #webcam
    else:
        data_iter = Frame_Iter("/data/gta5_tracking/val/image/rec_10090911_clouds_21h53m_x-968y-1487tox2523y214/")
    num_images = 1

    _t = {'im_detect': time.time(), 'misc': time.time()}
    det_file = os.path.join(output_dir, 'detections_{}.pkl'.format(args.anno))
 
    fasterRCNN.eval()
        
        #print(imdb.classes, "class number")
        #for i in range(num_images):
        #print(i, num_images)
    #with open(det_file, 'wb') as f:
        #pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    #print('Evaluating detections')
    # imdb.evaluate_detections(all_boxes, output_dir)

    #end = time.time()
    #print("test time: %0.4fs" % (end - start))
    return data_iter, fasterRCNN, args, imdb, [im_data, im_info, gt_boxes, num_boxes, fixed_center], start

def run_single(fasterRCNN, frame, class_agnostic, vis, webcam, imdb, image_vars, start):
    thresh = 0.6
    max_per_image = 300
    im_data, im_info, gt_boxes, num_boxes, fixed_center = image_vars
    i = 0
    all_boxes = [[] for _ in range(imdb.num_classes)]
    empty_array = np.transpose(np.array([[], [], [], [], [], [], []]), (1, 0))
    with torch.no_grad(): #Need to indent everything afterwards
        data_tic = time.time()
        if not webcam and not True:
            data = frame
            im_data.resize_(data[0].size()).copy_(data[0])
            im_info.resize_(data[1].size()).copy_(data[1])
            gt_boxes.resize_(data[2].size()).copy_(data[2])
            num_boxes.resize_(data[3].size()).copy_(data[3])
            fixed_center.resize_(data[4].size()).copy_(data[4])
        im_in = np.array(frame)
        if len(im_in.shape) == 2:
            im_in = im_in[:, :, np.newaxis]
            im_in = np.concatenate((im_in, im_in, im_in), axis=2)
        im = im_in[:, :, ::-1]
        blobs, im_scales = _get_image_blob(im)
        assert len(im_scales) == 1, "Only single-image batch implemented"
        im_blob = blobs
        im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)
        im_data_pt = torch.from_numpy(im_blob)
        im_data_pt = im_data_pt.permute(0, 3, 1, 2)
        im_info_pt = torch.from_numpy(im_info_np)
        im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
        #print(im_data, data_iter.get(3), data_iter.get(4))
        im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
        gt_boxes = torch.Tensor([])
        num_boxes = torch.Tensor([])
        fixed_center = torch.Tensor([])
    
        #print(type(gt_boxes))
    
        data_time = time.time() - data_tic
        det_tic = time.time()
        rois, cls_prob, bbox_pred, center_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        RCNN_loss_center, rois_label = fasterRCNN(im_data, im_info, gt_boxes,
                                              num_boxes, fixed_center)
        #print(bbox_pred.data.size(), type(bbox_pred.data))
        #print(bbox_pred[0,0,:])
        #print(bbox_pred.data)
        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            center_deltas = center_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                if class_agnostic:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
                        cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(
                        cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4)
                    center_deltas = center_deltas.view(1, -1, 2)
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
                        cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(
                        cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))
                    center_deltas = center_deltas.view(1, -1,
                                                   2 * len(imdb.classes))

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)

            pred_center = center_transform_inv(boxes, center_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        if webcam:
            pred_boxes /= im_info[0][2].item()
        else:
            pred_boxes /= data[1][0][2].item()

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        pred_center = pred_center.squeeze()
        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()
        if vis and not webcam:
            im = cv2.imread(imdb.image_path_at(i))
            im2show = np.copy(im)
        if webcam and vis:
            im2show = frame
        for j in xrange(1, imdb.num_classes):
            inds = torch.nonzero(scores[:, j] > thresh).view(-1)
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                    cls_centers = pred_center[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
                    cls_centers = pred_center[inds][:, j * 2:(j + 1) * 2]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                cls_dets_with_center = torch.cat(
                    (cls_boxes, cls_centers, cls_scores.unsqueeze(1)), 1)
                # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                cls_dets = cls_dets[order]
                cls_dets_with_center = cls_dets_with_center[order]
                keep = torch.arange(float(cls_dets.shape[0]))
                #print(keep, "one\n")
                #keep = nms(cls_dets[order,:], cls_scores[order], 0.3) #BM Did this to maybe make nms a thing
                #keep = nms2.rboxes(cls_dets[order,:].tolist(), cls_scores[order].tolist(), nms_threshold=0.5) Commented the working nms out because reasons?
                #print(keep,"two\n")
                keep = torch.Tensor(keep)
                #print(len(cls_dets))
                cls_dets = cls_dets[keep.view(-1).long()]
                #print(len(cls_dets))
                cls_dets_with_center = cls_dets_with_center[
                    keep.view(-1).long()]
                print(cls_dets_with_center.shape, "shape me daddy")
                if vis:
                    print("yeahhh boii show me them boxes")
                    im2show = vis_detections(im2show, imdb.classes[j], cls_dets_with_center.cpu().numpy(), gt_boxes.cpu().numpy(), fixed_center.cpu().numpy(), num_boxes.cpu().numpy(), thresh)
                if not webcam:
                    all_boxes[j][i] = cls_dets_with_center.cpu().numpy()
                else:
                    print("pack them boxes where they belong")
                    all_boxes[j].append(cls_dets_with_center.cpu().numpy())
            else:
                print("you had one job you useless cunt")
                all_boxes[j].append(empty_array)

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                  for j in xrange(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        misc_toc = time.time()
        nms_time = misc_toc - misc_tic
  
        sys.stdout.write('im_detect: #{:d} {:.3f}s {:.3f}s {:.3f}s   \r' \
                     .format(i + 1, data_time, detect_time, nms_time))
        sys.stdout.flush()

        if vis:
            #cv2.imwrite(os.path.join(output_dir, 'result%d.png' % (i + 1)), im2show)
            if webcam:
                im2showRGB = cv2.cvtColor(im2show, cv2.COLOR_BGR2RGB)
                cv2.imshow("frame", im2showRGB)
                total_toc = time.time()
                total_time = total_toc - start
                frame_rate = 1 / total_time
                print('Frame rate:', frame_rate)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    sys.exit()
        # pdb.set_trace()
        # cv2.imshow('test', im2show)
        # cv2.waitKey(0)
    return all_boxes, len(all_boxes)

def _get_image_blob(im):
    """Converts an image into a network input.
    Arguments:
      im (ndarray): a color image in BGR order
    Returns:
      blob (ndarray): a data blob holding an image pyramid
      im_scale_factors (list): list of image scales (relative to im) used
        in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        dim = (int(im_orig.shape[1]*im_scale), int(im_orig.shape[0]*im_scale))
        #print(im_orig.shape, dim, testy.shape)
        #try:
        im = cv2.resize(im_orig, dim, interpolation=cv2.INTER_LINEAR)
        #except Exception as e:
            #print("ugly: ",e)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)

if __name__ == '__main__':
    data_iter, fasterRCNN, args, imdb, im_vars, start = test_net()
    ret = True
    while ret:
        ret, frame = data_iter.read()
        run_single(fasterRCNN, frame, args.class_agnostic, args.vis, args.webcam, imdb, im_vars, start)
