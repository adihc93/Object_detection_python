# -*- coding: utf-8 -*-
"""
Object detection using Computer vision

Created on Thu Apr  5 10:29:03 2018

@author: HP
"""

from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils
#import time
import cv2
import tkinter 
from tkinter.filedialog import askopenfilename
from tkinter import messagebox
from PIL import ImageTk, Image
import urllib
import urllib.request
import sys

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

print("[INFO] loading model...")

net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")

def using_image():
    tkinter.Tk().withdraw()
    filename = askopenfilename()
    imag = cv2.imread(filename)
    (h, w) = imag.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(imag, (300, 300)), 0.007843, (300, 300), 127.5)
    print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            print("[INFO] {}".format(label))
            cv2.rectangle(imag, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(imag, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
    cv2.imshow("Output", imag)
    cv2.waitKey(0)
    
def using_webcam():
    print("[INFO] starting video stream...")
    vs = VideoStream(0).start()
    #time.sleep(2.0)
    fps = FPS().start()
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=900)
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        # pass the blob through the network and obtain the detections and predictions
        net.setInput(blob)
        detections = net.forward()
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the prediction
            confidence = detections[0, 0, i, 2]
            # filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
            if confidence > 0.2:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                print("[INFO] {}".format(label))
                cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
        # show the output frame
        cv2.imshow("press q to exit", frame)
        
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q") or key == ord("Q"):
            break
        # update the FPS counter
        fps.update()
    
    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()
    vs.stream.release()
    
def using_ipwebcam():
    host=str(E1.get())
    url='http://'+host+'/shot.jpg'
    while True:
            try:
                # Use urllib to get the image from the IP camera
                imgResp = urllib.request.urlopen(url)
                # Numpy to convert into a array
                imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)
    
                # Finally decode the array to OpenCV usable format ;) 
                img = cv2.imdecode(imgNp,-1)
            except:
                print("\n\n[ERROR] Camera not found or disconnected")
                tkinter.messagebox.showinfo("ERROR",'Camera not found or disconnected')
                break
            
            (h, w) = img.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 0.007843, (300, 300), 127.5)
            print("[INFO] computing object detections...")
            net.setInput(blob)
            detections = net.forward()
            for i in np.arange(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with the prediction
                confidence = detections[0, 0, i, 2]
                # filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
                if confidence > 0.2:
                    idx = int(detections[0, 0, i, 1])
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                    print("[INFO] {}".format(label))
                    cv2.rectangle(img, (startX, startY), (endX, endY), COLORS[idx], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(img, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
            
            # put the image on screen
            cv2.imshow('press q to exit',img)

            #To give the processor some less stress
            #time.sleep(0.1) 

            # Quit if q is pressed
            key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
            if key == ord("q") or key == ord("Q"):
                break
    
    cv2.destroyAllWindows()
                    

def ask_quit():
    mainWindow.destroy()
    sys.exit(0)
    
mainWindow=tkinter.Tk()
mainWindow.configure(background="white")
img = ImageTk.PhotoImage(Image.open("cv.png"))
mainWindow.title("Object Detection")
mainWindow.geometry("650x650")
mainWindow.resizable(0, 0)
l2=tkinter.Label(master=mainWindow, text='REALTIME OBJECT DETECTION', font=("Helvetica", 20))
l2.pack(side='top')
l2.configure(background="white") 
panel = tkinter.Label(mainWindow, image = img)     
panel.pack(side = "top", expand = "yes")
panel.configure(background="white")
l5=tkinter.Label(master=mainWindow, text='Host', font=("Helvetica", 16))
l5.pack(side='top')
l5.configure(background="white") 
E1=tkinter.Entry(master=mainWindow)
E1.insert(0, '192.168.2.2:8080')
E1.pack(side='top')
l1=tkinter.Label(master=mainWindow, text='Select option', font=("Helvetica", 16))
l1.pack(side='top')
l1.configure(background="white")  
B1=tkinter.Button(master=mainWindow, text='Get Image', command=using_image, relief='raised')
B1.pack(side='top')
B1.configure(background="white")
B2=tkinter.Button(master=mainWindow, text='Open webcam', command=using_webcam, relief='raised')
B2.pack(side='top')
B2.configure(background="white")
B3=tkinter.Button(master=mainWindow, text='Connect IP Webcam', command=using_ipwebcam, relief='raised')
B3.pack(side='top')
B3.configure(background="white")
dev_tag="\nDeveloped by:\nAditya Chaturvedi\nMCA SEM 4 Shift 1"
l3=tkinter.Label(master=mainWindow, text=dev_tag, font=("Helvetica", 10))
l3.pack(side='bottom')
l3.configure(background="white") 
mainWindow.protocol("WM_DELETE_WINDOW", ask_quit)
mainWindow.mainloop()