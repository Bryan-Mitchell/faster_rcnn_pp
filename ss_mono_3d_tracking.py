import _init_paths

import argparse
import json
import os
import pickle
import time

from joblib import Parallel, delayed
from tqdm import tqdm

import utils.tracking_utils as tu
from modle.tracker_2d import Tracker2D
from modle.tracker_3d import Tracker3D
from tools.eval_mot_bdd import TrackEvaluation

import ss_gen_pred as gen_pred
import ss_mono_3d_estimation as estims
import ss_test_net as detector
import numpy as np
import utils.bdd_helper as bh 
import cv2, torch, imutils
from tools import plot_tracking


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Monocular 3D Tracking',
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dset', help='Which f for tracking',
                        choices=['gta', 'kitti'], type=str)
    parser.add_argument('-j', dest='n_jobs', help='How many jobs in parallel'
                        ' (will brake the tracking ID) if more than 1',
                        default=1, type=int)
    parser.add_argument('--path', help='Path of input info for tracking',
                        default='./output/623_100_kitti_train_set/*bdd_roipool_output.pkl', type=str)
    parser.add_argument('--out_path', help='Path of tracking sequence output',
                        default='./623_100_kitti_train_set/', type=str)
    parser.add_argument('--debug_log_file', help='Path of debug log')
    parser.add_argument('--gpu', help='Using which GPU(s)',
                        default=None)
    parser.add_argument('--verbose', help='Show more information',
                        default=False, action='store_true')
    parser.add_argument('--visualize', help='Show current prediction',
                        default=False, action='store_true')

    parser.add_argument('--max_age', help='Maximum lifespan of a track',
                        default=10, type=int)
    parser.add_argument('--min_hits', help='Minimum hits to set a track',
                        default=3, type=int)
    parser.add_argument('--affinity_thres', help='Affinity threshold',
                        default=0.3, type=float)
    parser.add_argument('--skip', help='Skip frame by n',
                        default=1, type=int)
    parser.add_argument('--min_seq_len',
                        help='skip a sequence if less than n frames',
                        default=10, type=int)
    parser.add_argument('--max_depth', help='tracking within max_depth meters',
                        default=150, type=int)

    parser.add_argument('--occ', dest='use_occ',
                        help='use occlusion and depth ordering to help '
                             'tracking',
                        default=False, action='store_true')
    parser.add_argument('--deep', dest='deep_sort',
                        help='feature similarity to associate',
                        default=False, action='store_true')

    method = parser.add_mutually_exclusive_group(required=False)
    method.add_argument('--kf2d', dest='kf2d',
                        help='2D Kalman filter to smooth',
                        default=False, action='store_true')
    method.add_argument('--kf3d', dest='kf3d',
                        help='3D Kalman filter to smooth',
                        default=False, action='store_true')
    method.add_argument('--lstm', dest='lstm3d',
                        help='Estimate motion using LSTM to help 3D prediction',
                        default=False, action='store_true')
    method.add_argument('--lstmkf', dest='lstmkf3d',
                        help='Estimate motion in LSTM Kalman Filter to help '
                             '3D prediction',
                        default=False, action='store_true')

    args = parser.parse_args()
    args.device = 'cpu' if args.gpu is None else 'cuda'
    return args

def copy_border_reflect(img, p_h, p_w):
    if p_h > 0:
        pad_down = img[-p_h:, :][::-1, :]
        img = np.vstack((img, pad_down))
    if p_w > 0:
        pad_right = img[:, -p_w:][:, ::-1]
        img = np.hstack((img, pad_right))
    return img

def setup_tracker(args):
    max_dep = 150
    max_age = 10
    min_hits = 3
    affinity_thres = 0.3
    deep_sort = False
    use_occ = False
    kf3d = False
    lstm3d = True
    lstmkf3d = False
    device = 'cuda'
    verbose = False
    visualize = False
    mot_tracker = Tracker3D(
                dataset=args.dataset,
                max_depth=max_dep,
                max_age=max_age,
                min_hits=min_hits,
                affinity_threshold=affinity_thres,
                deep_sort=deep_sort,
                use_occ=use_occ,
                kf3d=kf3d,
                lstm3d=lstm3d,
                lstmkf3d=lstmkf3d,
                device=device,
                verbose=verbose,
                visualize=visualize)
    
    return mot_tracker

def update_tracker(mot_tracker, data, piccy, frame):
    """
    Core function of tracking along a sequence using mot_tracker
    """
    batch_time = tu.AverageMeter()
    frames_hypo = []
    frames_anno = []
    end = time.time()
    trackers = mot_tracker.update(data, piccy)
    batch_time.update(time.time() - end)


    # save gt frame annotations
    gt_anno = mot_tracker.frame_annotation
    frame_gt = {'timestamp': frame,
                'num': frame,
                'im_path': data['im_path'],
                'class': 'frame',
                'annotations': gt_anno}
    frames_anno.append(frame_gt)

    # save detect results
    frame_hypo = {'timestamp': frame,
                  'num': frame,
                  'im_path': data['im_path'],
                  'class': 'frame',
                  'hypotheses': trackers}
    frames_hypo.append(frame_hypo)
            

    seq_gt = {'frames': frames_anno, 'class': 'video', 'filename': "Live_footage_gt.mp4"}
    seq_hypo = {'frames': frames_hypo, 'class': 'video', 'filename': "Live_footage.mp4"}

    return seq_gt, seq_hypo

class Mono3DTracker:

    def __init__(self, args):
        self.seq_hypo_list = []
        self.seq_gt_list = []
        self.args = args
        if args.device == 'cuda':
            assert args.gpu != '', 'No gpu specific'
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

        if os.path.isdir(args.path):
            print('Load lists of pkl files')
            input_names = sorted(
                [n for n in os.listdir(args.path)
                 if n.endswith('bdd_roipool_output.pkl')])
            self.label_paths = [os.path.join(args.path, n) for n in input_names]
        elif args.path.endswith('bdd_roipool_output.pkl'):
            print('Load single pkl file')
            self.label_paths = [args.path]
        elif args.path.endswith('gta_roipool_output.pkl'):
            print('Load bundled pkl file')
            self.label_paths = args.path
        else:
            self.label_paths = []

    def run_app(self):
        """
        Entry function of calling parallel tracker on sequences
        """
        self.seq_gt_name = os.path.join(os.path.dirname(self.args.path),
                                         'gt.json')
        self.seq_pd_name = self.args.out_path + '_pd.json'

        if isinstance(self.label_paths, str):
            label_paths = pickle.load(open(self.label_paths, 'rb'))
        else:
            label_paths = self.label_paths

        n_seq = len(label_paths)
        print('* Number of sequence: {}'.format(n_seq))
        assert n_seq > 0, "Number of sequence is 0!"

        print('=> Building gt & hypo...')
        result = Parallel(n_jobs=self.args.n_jobs)(
            delayed(self.run_parallel)(seq_path, i_s)
            for i_s, seq_path in enumerate(tqdm(
                label_paths,
                disable=not self.args.verbose))
        )

        self.seq_gt_list = [n[0] for n in result]
        self.seq_hypo_list = [n[1] for n in result]

        if not os.path.isfile(self.seq_gt_name):
            with open(self.seq_gt_name, 'w') as f:
                print("Writing to {}".format(self.seq_gt_name))
                json.dump(self.seq_gt_list, f)
        with open(self.seq_pd_name, 'w') as f:
            print("Writing to {}".format(self.seq_pd_name))
            json.dump(self.seq_hypo_list, f)

    def run_parallel(self, seq_path, i_s):
        """
        Major function inside parallel calling run_seq with tracker and seq
        """

        if isinstance(seq_path, str):
            with open(seq_path, 'rb') as f:
                seqs = pickle.load(f)
            seq = seqs[0]

            if self.args.verbose: print(
                "Seq {} has {} frames".format(seq_path, len(seq)))
        else:
            # Bundled file
            seq = seq_path
            if self.args.verbose: print(
                "Seq {} has {} frames".format(i_s, len(seq)))

        # NOTE: Not in our case but will exclude small sequences from computing
        if len(seq) < self.args.min_seq_len:
            print("Warning: Skip sequence due to short length {}".format(
                len(seq)))
            seq_gt = {'frames': None, 'class': 'video', 'filename': 'null'}
            seq_hypo = {'frames': None, 'class': 'video', 'filename': 'null'}
            return seq_gt, seq_hypo

        # create instance of the SORT tracker
        if self.args.kf3d \
                or self.args.lstm3d \
                or self.args.lstmkf3d:
            mot_tracker = Tracker3D(
                dataset=self.args.dataset,
                max_depth=self.args.max_depth,
                max_age=self.args.max_age,
                min_hits=self.args.min_hits,
                affinity_threshold=self.args.affinity_thres,
                deep_sort=self.args.deep_sort,
                use_occ=self.args.use_occ,
                kf3d=self.args.kf3d,
                lstm3d=self.args.lstm3d,
                lstmkf3d=self.args.lstmkf3d,
                device=self.args.device,
                verbose=self.args.verbose,
                visualize=self.args.visualize)
        else:
            mot_tracker = Tracker2D(
                dataset=self.args.dataset,
                max_depth=self.args.max_depth,
                max_age=self.args.max_age,
                min_hits=self.args.min_hits,
                affinity_threshold=self.args.affinity_thres,
                kf2d=self.args.kf2d,
                deep_sort=self.args.deep_sort,
                verbose=self.args.verbose,
                visualize=self.args.visualize)

        if self.args.verbose: print("Processing seq{}".format(i_s))
        frames_anno, frames_hypo = self.run_seq(mot_tracker, seq)

        seq_gt = {'frames': frames_anno, 'class': 'video', 'filename': seq_path}
        seq_hypo = {'frames': frames_hypo, 'class': 'video', 'filename': seq_path}

        return seq_gt, seq_hypo

    def run_seq(self, mot_tracker, seq):
        """
        Core function of tracking along a sequence using mot_tracker
        """
        frame = 0
        batch_time = tu.AverageMeter()
        frames_hypo = []
        frames_anno = []
        for i_f, data in tqdm(enumerate(seq), disable=not self.args.verbose):
            if not i_f % self.args.skip == 0:
                continue
            frame += 1  # detection and frame numbers begin at 1

            end = time.time()
            trackers = mot_tracker.update(data)
            batch_time.update(time.time() - end)


            # save gt frame annotations
            gt_anno = mot_tracker.frame_annotation
            frame_gt = {'timestamp': i_f,
                        'num': i_f,
                        'im_path': data['im_path'],
                        'class': 'frame',
                        'annotations': gt_anno}
            frames_anno.append(frame_gt)

            # save detect results
            frame_hypo = {'timestamp': i_f,
                          'num': i_f,
                          'im_path': data['im_path'],
                          'class': 'frame',
                          'hypotheses': trackers}
            frames_hypo.append(frame_hypo)
            
        if self.args.verbose:
            print(
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '.format(
            batch_time=batch_time))

        return frames_anno, frames_hypo

def turn_preds_to_data(frame_pd, img, idx, args):
    n_box_limit = 20
    is_normalizing = False
    #frame = json.load(open(self.label_name[idx], 'r'))
    #pd_name = self.label_name[idx].replace('data', 'output')
    #pd_name = pd_name.replace('label', 'pred')
    #if os.path.isfile(pd_name):
        #frame_pd = json.load(open(pd_name, 'r'))
    #else:
        # No prediction json file found
        #frame_pd = {'prediction': []}
    n_box = 0
    #if n_box > self.n_box_limit:
        # print("n_box ({}) exceed the limit {}, clip up to
        # limit.".format(n_box, self.n_box_limit))
        #n_box = self.n_box_limit
    # Frame level annotations
    im_path = ""
    endvid = 0#int(idx + 1 in self.seq_accum)
    cam_rot = np.array([0,0,0])#np.array(frame['extrinsics']['rotation'])
    cam_loc = np.array([0,0,0])#np.array(frame['extrinsics']['location'])
    cam_calib = np.array([[935.3074360871937, 0, 960, 0],[0, 935.3074360871937, 540, 0],[0, 0, 1, 0]])#np.array(frame['intrinsics']['cali'])
    cam_focal = np.array([0,0,0])#np.array(frame['intrinsics']['focal'])
    cam_near_clip = np.array(0)#np.array(frame['intrinsics']['nearClip'])
    cam_fov_h = np.array(60)#np.array(frame['intrinsics']['fov'])
    pose = tu.Pose(cam_loc, cam_rot, True)

    # Object level annotations
    labels = []
    predictions = frame_pd['prediction']


    rois_pd = bh.get_box2d_array(predictions).astype(float)
    rois_gt = bh.get_box2d_array(labels).astype(float)
    tid = bh.get_label_array(labels, ['id'], (0)).astype(int)
    # Dim: H, W, L
    dim = bh.get_label_array(labels, ['box3d', 'dimension'], (0, 3)).astype(
        float)
    # Alpha: -pi ~ pi
    alpha = bh.get_label_array(labels, ['box3d', 'alpha'], (0)).astype(float)
    # Location in cam coord: x-right, y-down, z-front
    location = bh.get_label_array(labels, ['box3d', 'location'],
                                  (0, 3)).astype(float)

    # Center
    # f_x,   s, cen_x, ext_x
    #   0, f_y, cen_y, ext_y
    #   0,   0,     1, ext_z
    ext_loc = np.hstack([location, np.ones([len(location), 1])])  # (B, 4)
    proj_loc = ext_loc.dot(cam_calib.T)  # (B, 4) dot (3, 4).T => (B, 3)
    center_gt = proj_loc[:, :2] / proj_loc[:, 2:3]  # normalize

    center_pd = bh.get_cen_array(predictions)

    cam_calib = np.expand_dims(cam_calib,0)   


    # Depth
    depth = np.maximum(0, location[:, 2])

    ignore = bh.get_label_array(labels, ['attributes', 'ignore'], (0)).astype(
            int)
    # Get n_box_limit batch
    rois_gt = np.vstack([rois_gt, np.zeros([n_box_limit, 5])])[
         :n_box_limit]
    rois_pd = np.vstack([rois_pd, np.zeros([n_box_limit, 5])])[
              :n_box_limit]
    tid = np.hstack([tid, np.zeros(n_box_limit)])[:n_box_limit]
    alpha = np.hstack([alpha, np.zeros(n_box_limit)])[
            :n_box_limit]
    depth = np.hstack([depth, np.zeros(n_box_limit)])[
            :n_box_limit]
    center_pd = np.vstack([center_pd, np.zeros([n_box_limit, 2])])[
                :n_box_limit]
    center_gt = np.vstack([center_gt, np.zeros([n_box_limit, 2])])[
                :n_box_limit]
    dim = np.vstack([dim, np.zeros([n_box_limit, 3])])[
          :n_box_limit]
    ignore = np.hstack([ignore, np.zeros(n_box_limit)])[
             :n_box_limit]
    
    rois_pd = np.expand_dims(rois_pd,0)
    # objects center in the world coordinates
    loc_gt = np.array([0,0,0])#tu.point3dcoord(center_gt, depth, cam_calib, pose)

    # Load images
    assert img is not None, "Cannot read {}".format(im_path)

    h, w, _ = img.shape
    p_h = 1088 - h #1088 for gta 384 for kitti
    p_w = 1920 - w #1920 for gta 1248 for kitti
    assert p_h >= 0, "target hight - image hight = {}".format(p_h)
    assert p_w >= 0, "target width - image width = {}".format(p_w)
    img = copy_border_reflect(img, p_h, p_w)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_patch = np.rollaxis(img, 2, 0)
    img_patch = img_patch.astype(float) / 255.0

    # Normalize
    if args.dataset == 'kitti':
        mean = [0.28679871, 0.30261545, 0.32524435]
        std = [0.27106311, 0.27234113, 0.27918578]
    elif args.dataset == 'gta':
        mean = [0.34088846, 0.34000116, 0.35496006]
        std = [0.21032437, 0.19707282, 0.18238117]
    else:
        print("Not normalized!!")
        mean = [0.0, 0.0, 0.0]
        std = [1.0, 1.0, 1.0]
    if is_normalizing:
        img_patch = (img_patch - mean) / std

    bin_cls = np.zeros((n_box_limit, 2))
    bin_res = np.zeros((n_box_limit, 2))

    for i in range(n_box):
        if alpha[i] < np.pi / 6. or alpha[i] > 5 * np.pi / 6.:
            bin_cls[i, 0] = 1
            bin_res[i, 0] = alpha[i] - (-0.5 * np.pi)
        if alpha[i] > -np.pi / 6. or alpha[i] < -5 * np.pi / 6.:
            bin_cls[i, 1] = 1
            bin_res[i, 1] = alpha[i] - (0.5 * np.pi)

    box_info = {
        'im_path': im_path,
        'rois_pd': torch.from_numpy(rois_pd).float(),
        'rois_gt': torch.from_numpy(rois_gt).float(),
        'dim_gt': torch.from_numpy(dim).float(),
        'bin_cls_gt': torch.from_numpy(bin_cls).long(),
        'bin_res_gt': torch.from_numpy(bin_res).float(),
        'alpha_gt': torch.from_numpy(alpha).float(),
        'depth_gt': torch.from_numpy(depth).float(),
        'cen_pd': torch.from_numpy(center_pd).float(),
        'cen_gt': torch.from_numpy(center_gt).float(),
        'loc_gt': torch.from_numpy(loc_gt).float(),
        'tid_gt': torch.from_numpy(tid).int(),
        'ignore': torch.from_numpy(ignore).int(),
        'n_box': torch.Tensor([n_box]),
        'endvid': torch.Tensor(endvid),
        'cam_calib': torch.from_numpy(cam_calib).float(),
        'cam_rot': torch.from_numpy(np.expand_dims(pose.rotation,0)).float(),
        'cam_loc': torch.from_numpy(np.expand_dims(pose.position,0)).float(),
        }
    print("rot ",box_info['cam_rot'], " end me")
    return torch.from_numpy(img_patch).float(), box_info

def copy_border_reflect(img, p_h, p_w):
    if p_h > 0:
        pad_down = img[-p_h:, :][::-1, :]
        img = np.vstack((img, pad_down))
    if p_w > 0:
        pad_right = img[:, -p_w:][:, ::-1]
        img = np.hstack((img, pad_right))
    return img

def gt_me(frame, count, gts):
    temp = json.load(open(gts[count], 'r'))
    print(temp)
    rois_gt = bh.get_box2d_array([temp]).astype(float)
    print(rois_gt)
    print("roughie ", frame[0])


def main():
    if not os.path.exists('output'):
        os.makedirs('output')

    args = parse_args()
    assert not args.use_occ or (args.lstmkf3d or args.lstm3d or args.kf3d), \
        "Occlusion only be used in 3D method"
    
    webcam = False
    data_iter, fasterRCNN, args, imdb, im_vars, start = detector.test_net(webcam)
    ret = True
    draw3d = False
    draw2d = True
    birds_eye = True
    count = 0
    gts = None
    json_path = os.getcwd()+"/data/gta5_tracking/val/label" #GTA Val jsons
    if not webcam:
        gts = gen_pred.load_gts(json_path, 'gta')
    while ret:
        start = time.time()
        ret, frame = data_iter.read()
        if not webcam:
            frame = copy_border_reflect(frame, 8, 0)
            print(frame.shape)
  
        frame_old = frame
        all_boxes, num_boxes = detector.run_single(fasterRCNN, frame, args.class_agnostic, True, True, imdb, im_vars, start)
        frame_data = gen_pred.ss_pred_gen(all_boxes,0)
        something, box_info = turn_preds_to_data(frame_data, frame, 0, args)
        tense_frame = torch.Tensor(np.moveaxis(np.expand_dims(frame,0),3,1))
        print('check: ',tense_frame.shape)
        model, tracking_list = estims.setup()
        model.eval()
        tracking_list = estims.ss_box_me(model, tense_frame, box_info, tracking_list)
        if not webcam:
            tracking_list = gt_me(tracking_list, count, gts)
        mot_tracker = setup_tracker(args)
        #print(type(tracking_list[0]), len(tracking_list))
        frames_anno, frames_hypo = update_tracker(mot_tracker, tracking_list[0], frame, count)
        plot_tracking.plot_shit(frames_anno, frames_hypo, birds_eye, draw2d, draw3d, "Webcam_boii", frame_old)
        #print(tracking_list)
        #print("number: ",count)
        count+=1
        end = time.time()
        print("spf:",end-start," seconds")
    #tracker = Mono3DTracker(args)
    #tracker.run_app()

    #te = TrackEvaluation(tracker.seq_gt_list, tracker.seq_hypo_list, args.out_path, _debug = args.debug_log_file, verbose = args.verbose)
    #te.eval_app()

if __name__ == '__main__':
    main()
