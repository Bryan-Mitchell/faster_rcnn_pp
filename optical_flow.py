import numpy as np
import cv2 as cv
import argparse
import os

parser = argparse.ArgumentParser(description='This sample demonstrates Lucas-Kanade Optical Flow calculation. \
                                              The example file can be downloaded from: \
                                              https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4')
parser.add_argument('image', type=str, help='path to image file')
args = parser.parse_args()

cap = cv.VideoCapture(args.image)

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
fname = './im.jpg'
cv.imwrite(fname, old_frame)
stream = os.popen('./darknet detector test cfg/coco.data cfg/yolov3.cfg yolov3.weights ./im.jpg', mode='r')
print(stream.read())
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
boxes = open("./box_centres.txt", 'r')
blob = boxes.readlines()
p0 = []
for i in blob:
    a = i.split(",")
    a[0] = np.float32(i[0])
    a[1] = np.float32(a[1].split("\n")[0])
    p0.append(a)
p0 = np.asarray(p0)
p0 = p0.reshape(-1,1,2)
print(p0)
boxes.close()
print(type(p0[0][0]))

#p0 will hold centres of bounding boxes
#p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
#print(p0)
#print(type(p0[0][0]))
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

while(1):
    ret,frame = cap.read()
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]

    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new, good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv.circle(frame,(a,b),5,color[i].tolist(),-1)
    img = cv.add(frame,mask)

    cv.imshow('frame',img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)