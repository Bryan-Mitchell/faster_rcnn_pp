# USAGE
# python object_tracker.py --prototxt deploy.prototxt --model res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
from pyimagesearch.centroidtracker import CentroidTracker
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import sys, os
sys.path.append(os.path.join(os.getcwd(),'python/'))
import darknet as dn

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
if args["video"] == None:
	vs = cv2.VideoCapture(0) #webcam
else:
	vs = cv2.VideoCapture(args['video']) 

time.sleep(2.0)

#stream = os.popen('./darknet detect cfg/yolov3.cfg yolov3.weights', mode='w')
fname = "./im.jpg"
net = dn.load_net(b"./cfg/yolov3.cfg",b"./yolov3.weights",0)
names = dn.get_names(b"./cfg/coco.data")
# loop over the frames from the video stream
while True:
	# read the next frame from the video stream and resize it
	start = time.time()
	ret, frame = vs.read()
	frame = imutils.resize(frame, width=400)
	cv2.imwrite(fname, frame)
	#stream.write(fname)
	dn.ace(names, net,bytes(fname, encoding='utf8'), 0.5, 0.5)
	# if the frame dimensions are None, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# construct a blob from the frame, pass it through the network,
	# obtain our output predictions, and initialize the list of
	# bounding box rectangles
	#blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H),
		#(104.0, 177.0, 123.0))
	#net.setInput(blob)
	#detections = net.forward()
	boxes = open("./box_centres.txt", 'r')
	blob = boxes.readlines()
	p0 = []
	boxnames = []
	a = [1,2,3,4]
	for i in blob:
		i = i.split(",")
		a[0] = float(i[0])
		a[1] = float(i[1])
		a[2] = float(i[2])
		a[3] = float(i[3])
		boxnames.append(i[4].split("\n")[0])
		p0.append(tuple(a))
	#p0 = np.asarray(p0)
	
	print(p0, boxnames)
	boxes.close()
	detections = np.asarray(p0)
	detections = detections.reshape(1,1,-1,2)

	rects = p0
	
	# loop over the detections
	for i in range(0, len(p0)):
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
		cv2.rectangle(frame, (int(p0[i][0]), int(p0[i][1])), (int(p0[i][2]), int(p0[i][3])), (0, 255, 0), 2)

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
	end = time.time()
	print("spf:",end-start," seconds")
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
