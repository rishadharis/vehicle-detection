from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
import imutils

parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
args = parser.parse_args()

#create Background Subtractor objects
if args.algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2(detectShadows=True,
                                                history=200,varThreshold=90)
else:
    backSub = cv.createBackgroundSubtractorKNN()
#end create background
    
## [capture]
capture = cv.VideoCapture('video1.avi')
if not capture.isOpened:
    print('Unable to open: ' + args.input)
    exit(0)
## [end capture]

while True:
    ret, frame = capture.read()
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (1, 1))
    # Fill any small holes
    closing = cv.morphologyEx(frame, cv.MORPH_CLOSE, kernel)
    # Remove noise
    opening = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel)
    
    #rets, trss = cv.threshold(opening, 128, 0, cv.THRESH_BINARY)

    # Dilate to merge adjacent blobs
    dilation = cv.erode(opening, None, iterations=2)

    # threshold
    th = dilation[dilation < 240] = 0
    if frame is None:
        break

    ## [apply]
    #update the background model
    fgMask = backSub.apply(opening)
    cnts = cv.findContours(fgMask.copy(), cv.RETR_EXTERNAL,
	cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    for c in cnts:
        (x,y,w,h) = cv.boundingRect(c)
        if w>=30 and y>=30:
            cv.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)
        
        #cv.drawContours(frame, [c], -1, (240, 0, 159), 1)
	# draw each contour on the output image with a 3px thick purple
	# outline, then display the output contours one at a time
    
    #cv.line(frame, (80, 40), (230, 60), (255, 0, 255), 2)
    ## [apply]

    ## [display_frame_number]
    #get the frame number and write it on the current frame
    cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    ## [display_frame_number]

    ## [show]
    #show the current frame and the fg masks
    cv.imshow('Frame', frame)
    cv.imshow('FG Mask', fgMask)
    ## [show]

    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
