import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
import copy

import harris_croner_max_supress as HC

 



# need to process a video file.
# apply processing on it.
# an then save output in video format.
# we can make it genenric like , running an image processing function
#on any video  or any video processed by any function.

 
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
 
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    #print(frame.shape)
    color_img=copy.deepcopy(frame)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray_blur=cv.GaussianBlur(gray,(21,21),0)
    #print(gray.shape)
    # find edges
    Ver_edge=HC.Ix(gray_blur)
    Hor_edge=HC.Iy(gray_blur)
    
    #find high intensity region
    corners=HC.higheigen_region(Ver_edge,Hor_edge)
    
    #supress corners
    corners_copy=copy.deepcopy(corners)
    max_supressed_img,dilute = HC.non_max_suppresion(corners_copy,21)
    
    #further processing to remove redundant corners and visualise the final result
    max_supressed_img_mask = (max_supressed_img >= 0.9*np.max(max_supressed_img))
    max_supressed_img_mask=max_supressed_img_mask.astype(int)
    max_supressed_img_revamped=np.multiply(max_supressed_img,max_supressed_img_mask)# the corners

    max_supressed_img_revamped=cv.dilate(max_supressed_img_revamped,None)#dilating to better visualise corners location in image

    conv=copy.deepcopy(color_img)
    conv[max_supressed_img_revamped>0]=[255.0,0,0]
    #find edgesß
    # Display the resulting frame
    #cv.imshow('frame', gray)
    cv.imshow("corners",max_supressed_img_revamped)
    cv.imshow("corners_onimage",conv)
    if cv.waitKey(5) == ord('q'):
        break
 
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()