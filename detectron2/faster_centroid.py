# USAGE
# python object_tracker.py --prototxt deploy.prototxt --model res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
import sys, os, imp, tqdm
sys.path.append('/home/mitch/Documents/3d-vehicle-tracking-master/3d-tracking/lib/')
#sys.path.append('./lib/')
from demo.predictor import VisualizationDemo

from pyimagesearch.centroidtracker import CentroidTracker
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
from detectron2.config import get_cfg


def setup_cfg():
	# load config from file and command-line arguments
	cfg = get_cfg()
	# To use demo for Panoptic-DeepLab, please uncomment the following two lines.
	# from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
	# add_panoptic_deeplab_config(cfg)
	cfg.merge_from_file("configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
	cfg.merge_from_list(['MODEL.WEIGHTS','model_final_68b088.pkl'])
	# Set score_threshold for builtin models
	cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
	cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
	cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5
	cfg.freeze()
	return cfg




# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=False,
	help="(w)ebcam or (v)ideo")
ap.add_argument("-m", "--model", required=False,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize our centroid tracker and frame dimensions
ct = CentroidTracker()
(H, W) = (None, None)

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe("./deploy.prototxt", "./res10_300x300_ssd_iter_140000.caffemodel")

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")
#vs = VideoStream(src=0).start()

#Not required for faster RCNN
#if args["video"] == None:
	#vs = cv2.VideoCapture(0) #webcam
#else:
	#vs = cv2.VideoCapture(args['video']) 

time.sleep(2.0)

#stream = os.popen('./darknet detect cfg/yolov3.cfg yolov3.weights', mode='w')

# loop over the frames from the video stream
cam = cv2.VideoCapture(0)
dataset = 'kitti'
count = 0
draw3d = False
draw2d = True
birds_eye = True
cfg = setup_cfg()
demo = VisualizationDemo(cfg)
start = time.time()
for vis, preds in tqdm.tqdm(demo.run_on_video(cam)):
	frame = vis
	scores = preds['instances'].scores.cpu()
	all_boxes = np.asarray(preds['instances'].pred_boxes)
	dets = []
	for i in range(all_boxes.size):
		all_boxes[i] = np.asarray(all_boxes[i].cpu())
		dets.append(all_boxes[i])
		#all_boxes[i] = np.append(all_boxes[i],scores[i])
		#print(all_boxes[i])

	rects = []
	count = 0
	# loop over the detections
	for i in range(1, len(dets)):
		# filter out weak detections by ensuring the predicted
		# probability is greater than a minimum threshold
		#if detections[0, 0, i, 2] > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the object, then update the bounding box rectangles list
			#box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
			#rects.append(box.astype("int"))

			# draw a bounding box surrounding the object so we can
			# visualize it
			#(startX, startY, endX, endY) = box.astype("int")
		#for j in range(0, len(dets[i])):
		cv2.rectangle(frame, (int(dets[i][0]), int(dets[i][1])), (int(dets[i][2]), int(dets[i][3])), (0, 255, 0), 2)
		rects.append(dets[i][0:4])
		rects[count][2] += rects[count][0]
		rects[count][3] += rects[count][1]
		count += 1

	# update our centroid tracker using the computed set of bounding
	# box rectangles
	objects = ct.update(rects)

	# loop over the tracked objects
	for (objectID, centroid) in objects.items():
		# draw both the ID of the object and the centroid of the
		# object on the output frame
		text = "ID "+str(objectID)
		cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
	count += 1
	end = time.time()
	print("spf:", end-start," seconds")
	start = time.time()
cam.release()
cv2.destroyAllWindows()
while True:
	# read the next frame from the video stream and resize it
	start = time.time()
	ret, frame = vs.read()
	frame = imutils.resize(frame, width=600)
	cv2.imwrite(fname, frame)
	#stream.write(fname)
	p0, num_dets = faster.run_single(fasterRCNN, frame, args.class_agnostic, args.vis, args.webcam, imdb, im_vars, start)
	# if the frame dimensions are None, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	
	end = time.time()
	print("spf:",end-start," seconds")

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
