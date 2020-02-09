# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:04:00 2020

@author: canno
"""
import os
import cv2
import numpy as np

path_dir_input=os.getcwd()+"\\data_input"
path_dir_output=os.getcwd()+"\\data_output"
files = os.listdir(path_dir_input)
def get_red_spots(input_img):
    lower_red = np.array([0,0,0])
    upper_red = np.array([30,255,255])
    mask_red = cv2.inRange(input_img, lower_red, upper_red)
    output_img = cv2.bitwise_or(input_img, input_img, mask = mask_red)
    return output_img

def overlay_images(background,overlay):
    return cv2.addWeighted(background,0.5,overlay,1,0)
    
for file in files:
    input_img = cv2.imread(path_dir_input+"\\"+file)
    #identify red spots
    output_img = get_red_spots(input_img)
    #overlay 
    output_img = overlay_images(input_img,output_img)
    cv2.imwrite(path_dir_output+"\\"+file,output_img)


