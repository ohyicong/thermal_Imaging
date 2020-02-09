# -*- coding: utf-8 -*-q
"""
Created on Fri Feb  7 09:27:41 2020

@author: OHyic
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os
import numpy as np

def get_red_spots(input_img):
    #this function identify the red spots from the thermal image
    lower_red = np.array([0,0,0])
    upper_red = np.array([35,255,255])
    mask_red = cv2.inRange(input_img, lower_red, upper_red)
    red_spots_img = cv2.bitwise_or(input_img, input_img, mask = mask_red)

    #using AI to detect relevant spot
    temp_img= red_spots_img.copy()
    gray = cv2.cvtColor(input_img,cv2.COLOR_BGR2GRAY)
    faces = model.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)
    for x,y,w,h in faces:
        temp_img[y:y+w,x:x+h]=np.zeros(3)
    
    output_img=red_spots_img-temp_img
    
    for x,y,w,h in faces:
        output_img=cv2.rectangle(output_img,(x,y),(x+w,y+h),(255,255,255),1)
        output_img = cv2.putText(output_img, 'human', (x,y-10),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),1)
    
    return output_img

def overlay_images(background,overlay):
    #this function overlay the background image with the thermal imaging
    return cv2.addWeighted(background,0.5,overlay,1,0)

def transform_image(input_img):
    #this function resizes the webcam output to meet thermal imaging standards 
    cropped_img = input_img[120:420,150:(150+380)]
    '''
    scale_percent = 60 # percent of original size
    height = int(input_img.shape[0] * scale_percent / 100)
    width = int(input_img.shape[1] * scale_percent / 100)
    dim = (width, height)
    resized_img = cv2.resize(cropped_img, dim, interpolation = cv2.INTER_AREA)
    '''
    return cropped_img

def get_suspected(input_img,model):
    #this function identify possible suspect
    #[todo] now the model is a vanilla model used to identify faces.. not high risk personnel
    gray = cv2.cvtColor(input_img,cv2.COLOR_BGR2GRAY)
    faces = model.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5) 
    for x,y,w,h in faces:
        input_img=cv2.rectangle(input_img,(x,y),(x+w,y+h),(255,255,255),1)
        input_img = cv2.putText(input_img, 'human', (x,y-10),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),1)
    return input_img


    
cap = cv2.VideoCapture(0)
model = cv2.CascadeClassifier(os.getcwd()+"\\haarcascades\\haarcascade_frontalface_default.xml")
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    #Resize the image according to your thermal imager
    frame= transform_image(frame)
    # Display the resulting frame
    red_frame = get_red_spots(frame)
    #print(red_frame.shape)
    #Get Suspect 
    #frame=get_suspected(frame,model)
    overlay_frame= overlay_images(frame,red_frame)
    cv2.imshow('red_frame',red_frame)
    cv2.imshow('overlay_frame',overlay_frame)    
    cv2.imshow('frame',frame)    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
