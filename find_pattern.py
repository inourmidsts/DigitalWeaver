#!/usr/bin/env python3
# coding: Latin

# Load library functions we want
import time
import os
import sys
# import ThunderBorg
import io
import threading
import numpy
import picamera
import picamera.array
import cv2
from pyimagesearch.shapedetector import ShapeDetector
from pyimagesearch.colorlabeler import ColorLabeler
#import argparse
import imutils
from collections import OrderedDict


print('Libraries loaded')

# Global values
global running
# global TB
global camera
global processor
global debug
global colour

running = True
debug = True
colour = 'green'

# Camera settings
imageWidth = 320  # Camera image width
imageHeight = 240  # Camera image height
frameRate = 3  # Camera image capture frame rate

# Auto drive settings
autoMaxPower = 1.0  # Maximum output in automatic mode
autoMinPower = 0.2  # Minimum output in automatic mode
autoMinArea = 10  # Smallest target to move towards
autoMaxArea = 10000  # Largest target to move towards
autoFullSpeedArea = 300  # Target size at which we use the maximum allowed output

# Image stream processing thread
class StreamProcessor(threading.Thread):
    def __init__(self):
        super(StreamProcessor, self).__init__()
        self.stream = picamera.array.PiRGBArray(camera)
        self.event = threading.Event()
        self.terminated = False
        self.start()
        self.begin = 0

    def run(self):
        # This method runs in a separate thread
        while not self.terminated:
            # Wait for an image to be written to the stream
            if self.event.wait(1):
                try:
                    # Read the image and do some processing on it
                    self.stream.seek(0)
                    self.ProcessImage(self.stream.array, colour)
                finally:
                    # Reset the stream and event
                    self.stream.seek(0)
                    self.stream.truncate()
                    self.event.clear()

    # Image processing function
    def ProcessImage(self, image, colour):
        # View the original image seen by the camera.
        if debug:
            cv2.imshow('original', image)
            time.sleep(0.3)
            #cv2.waitKey(0)

        # convert the resized image to grayscale, blur it slightly,
        # and threshold it
        blurred_img = cv2.GaussianBlur(image, (7, 7), 0)
        #if debug:
        #    cv2.imshow('GaussianBlur', blurred_img)
        #    cv2.waitKey(0)
        grey_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2GRAY)
        #if debug:
        #    cv2.imshow('cvtColor GRAY', grey_img)
        #    cv2.waitKey(0)
        lab_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2LAB)
        #if debug:
        #    cv2.imshow('cvtColor LAB', lab_img)
        #    cv2.waitKey(0)
        #thresh_img = cv2.threshold(grey_img, 60, 255, cv2.THRESH_BINARY)[1]
        thresh_img = cv2.adaptiveThreshold(grey_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
        if debug:
            cv2.imshow('threshold', thresh_img)
            time.sleep(0.3)
            #cv2.waitKey(0)

        ## Blur the image
        #image = cv2.medianBlur(image, 5)
        #if debug:
        #    cv2.imshow('blur', image)
        #    cv2.waitKey(0)

        ## Convert the image from 'BGR' to HSV colour space
        #image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        #if debug:
        #    cv2.imshow('cvtColour', image)
        #    cv2.waitKey(0)

        ## We want to extract the 'Hue', or colour, from the image. The 'inRange'
        ## method will extract the colour we are interested in (between 0 and 180)
        ## In testing, the Hue value for red is between 95 and 125
        ## Green is between 50 and 75
        ## Blue is between 20 and 35
        ## Yellow is... to be found!
        #if colour == "red":
        #    imrange = cv2.inRange(image, numpy.array((95, 127, 64)), numpy.array((125, 255, 255)))
        #elif colour == "green":
        #    imrange = cv2.inRange(image, numpy.array((40, 127, 64)), numpy.array((75, 255, 255)))
        #elif colour == 'blue':
        #    imrange = cv2.inRange(image, numpy.array((20, 64, 64)), numpy.array((35, 255, 255)))
        # View the filtered image found by 'imrange'
        #if debug:
        #    cv2.imshow('imrange', imrange)
        #    cv2.waitKey(0)

        # I used the following code to find the approximate 'hue' of the ball in
        # front of the camera
        #        for crange in range(0,170,10):
        #            imrange = cv2.inRange(image, numpy.array((crange, 64, 64)), numpy.array((crange+10, 255, 255)))
        #            print(crange)
        #            cv2.imshow('range',imrange)
        #            cv2.waitKey(0)
        

        # Find the contours
        #contour_img, contours, hierarchy = cv2.findContours(thresh_img.copy(), cv2.RETR_LIST,
        #                                                     cv2.CHAIN_APPROX_SIMPLE)
        #if debug:
        #    cv2.imshow('contour', contour_img)
        #    cv2.waitKey(0)
        # cv2.RETR_EXTERNAL - just outermost contour
        # cv2.RETR_LIST - all contours, no heirarchy
        contours = cv2.findContours(thresh_img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if imutils.is_cv2() else contours[1]
        print('count of contours={}'.format(len(contours)))

        sd = ShapeDetector()
        cl = ColorLabeler()

        # Go through each contour
        #foundArea = -1
        #foundX = -1
        #foundY = -1

        squares = []
        sc = []
        for idx, contour in enumerate(contours):
            #print('contour={}'.f5ormat(contour))
            x, y, w, h = cv2.boundingRect(contour)
            if w <= 15 or h <= 15 or w > 50 or h > 50:
                continue
            sc.append(contour)
            print('x={} y={} w={} h={}'.format(x, y, h, w))
            cx = x + (w / 2)
            cy = y + (h / 2)
            area = w * h
            print('  cx={} cy={} area={}'.format(cx, cy, area))
            shape = sd.detect(contour)
            colour = cl.label(lab_img, contour)
            print('  shape={} color={}'.format(shape, colour))
            squares.append([idx,x,y,h,w,cx,cy,area,colour])
            #if foundArea < area:
            #    foundArea = area
            #    foundX = cx
            #    foundY = cy
        #if foundArea > 0:
        #    ball = [foundX, foundY, foundArea]
        #else:
        #    ball = None
        ## Set drives or report ball status
        #self.SetSpeedFromBall(ball)
        print('Found {} shapes.'.format(len(squares)))
        print('Shapes={}'.format(squares))
        #print([i[0] for i in squares])
        for sq in squares:
            cv2.rectangle(image, (sq[1], sq[2]), (sq[1]+sq[3], sq[2]+sq[4]),
                          (0, 0, 255) if sq[8]=='red' else (255,0,0), cv2.FILLED)
        cv2.drawContours(image, sc, -1, (255, 0, 0), cv2.FILLED) 
        cv2.imshow('filled', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        #sort by cy
        ysorted_sqs = OrderedDict()
        min_x = squares[0][0]
        min_y = squares[0][1]
        for sq in squares:
            # add cx, cy, h, v, colour
            ysorted_sqs[sq[6]] = (sq[5], sq[6], sq[3], sq[4], sq[8])
            if sq[0] < min_x: 
                min_x = sq[0]
            if sq[1] < min_y:
                min_y = sq[1]

        #group by rows, centres differ by less than a 3rd.
        rows = []
        for idx, ssq in enumerate(ysorted_sqs):
            if idx != 0:
                # 0=cx, 1=cy, 2=h, 3=v, 4=colour
                if abs(ssq[1] - ysorted_sqs[idx-1][1]) > ysorted_sqs[idx-1][3]/3:
                    # not same row, add to new row
                    rows.append([ssq])
            # add square to current row
            rows.extend([ssq])
               
        #sort by cx
        for row in rows:
            xsorted_sqs = OrderedDict()
            for ssq in row:
                xsorted_sqs[ssq[0]] = ssq
                
        

##    # Set the motor speed from the ball position
##    def SetSpeedFromBall(self, ball):
##        global TB
##        driveLeft = 0.0
##        driveRight = 0.0
##        if ball:
##            x = ball[0]
##            area = ball[2]
##            if area < autoMinArea:
##                print('Too small / far')
##            elif area > autoMaxArea:
##                print('Close enough')
##            else:
##                if area < autoFullSpeedArea:
##                    speed = 1.0
##                else:
##                    speed = 1.0 / (area / autoFullSpeedArea)
##                speed *= autoMaxPower - autoMinPower
##                speed += autoMinPower
##                direction = (x - imageCentreX) / imageCentreX
##                if direction < 0.0:
##                    # Turn right
##                    print('Turn Right')
##                    driveLeft = speed
##                    driveRight = speed * (1.0 + direction)
##                else:
##                    # Turn left
##                    print('Turn Left')
##                    driveLeft = speed * (1.0 - direction)
##                    driveRight = speed
##                print('%.2f, %.2f' % (driveLeft, driveRight))
##        else:
##            print('No ball')


# Image capture thread
class ImageCapture(threading.Thread):
    def __init__(self):
        super(ImageCapture, self).__init__()
        self.start()

    def run(self):
        global camera
        global processor
        print('Start the stream using the video port')
        camera.capture_sequence(self.TriggerStream(), format='bgr', use_video_port=True)
        print('Terminating camera processing...')
        processor.terminated = True
        processor.join()
        print('Processing terminated.')

    # Stream delegation loop
    def TriggerStream(self):
        global running
        while running:
            if processor.event.is_set():
                time.sleep(0.01)
            else:
                yield processor.stream
                processor.event.set()


# Startup sequence
print('Setup camera')
camera = picamera.PiCamera()
camera.resolution = (imageWidth, imageHeight)
camera.framerate = frameRate
imageCentreX = imageWidth / 2.0
imageCentreY = imageHeight / 2.0

print('Setup the stream processing thread')
processor = StreamProcessor()

print('Wait ...')
time.sleep(2)
captureThread = ImageCapture()

try:
    print('Press CTRL+C to quit')
    # Loop indefinitely until we are no longer running
    while running:
        # Wait for the interval period

        # You could have the code do other work in here 🙂
        time.sleep(1.0)
except KeyboardInterrupt:
    # CTRL+C exit, disable all drives
    print('\nUser shutdown')
except:
    # Unexpected error, shut down!
    e = sys.exc_info()[0]
    print
    print(e)
    print('\nUnexpected error, shutting down!')
cv2.destroyAllWindows()
# Tell each thread to stop, and wait for them to end
running = False
captureThread.join()
processor.terminated = True
processor.join()
del camera
print('Program terminated.')
