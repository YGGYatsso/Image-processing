import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
import copy


# find edges -> sobelx, sobely
# initial stage filters are 

def Ix(image):  # change in Func / Change in horizontal direction.
    kernel=np.array([[1,0,-1],[1,0,-1],[1,0,-1]])/6.0
    vertical_edge= cv.filter2D(image,-1,kernel)
    return vertical_edge

def Iy(image):  # change in Func / Change in vertical direction.
    kernel=np.array([[1,1,1],[0,0,0],[-1,-1,-1]])/6.0
    horizontal_edge= cv.filter2D(image,-1,kernel)
    return horizontal_edge


def degree0 (i,j,mag):
    if mag[i][j-1]>= mag[i][j] or mag[i][j+1]>=mag[i][j]:
        return 0
    else:
        return 1

def degree90 (i,j,mag):
    if mag[i-1][j]>=mag[i][j] or mag[i+1][j] >=mag[i][j] :
        return 0
    else :
        return 1

def degree45 (i,j,mag):
    if mag[i-1][j+1]>=mag[i][j] or mag[i+1][j-1] >=mag[i][j] :
        return 0
    else :
        return 1

def degree135 (i,j,mag):
    if mag[i-1][j-1]>=mag[i][j] or mag[i+1][j+1] >=mag[i][j] :
        return 0
    else :
        return 1

def supress_non_max(Ix,Iy,mag,img):
    
    h,w=Ix.shape # unpacked the tuple.
    mask=np.zeros((h,w),dtype=np.uint8)
    for i in range(2,h-1):
        for j in range(2,w-1):
            
            # pixel location (i,j)
            
            if Ix[i][j]==0 and Iy[i][j]==0 : 
                # all along smooth. 
                continue
            elif Ix[i][j]==0 :# then horizontal edge
                # go up and down
                mask[i][j]=degree90(i,j,mag)
            elif Iy[i][j]==0: # then vertical edge
                # go left and right
                mask[i][j]=degree0(i,j,mag)        
            else: 
                degree=(np.arctan(Iy[i][j]/Ix[i][j])*180)/np.pi
                # i got the degrees.abs
                # tanh is an odd function 
                # output value will be between 
                degree=degree+90 # now scale will be 0-180 only
                #print(degree)
                if degree <22.5 or degree >157.5:                        
                    #go up and down
                    mask[i][j]=degree90(i,j,mag) 
                elif degree <=67.5 :
                    #go slant left-right
                    mask[i][j]=degree135(i,j,mag)
                elif degree <=122.5 :
                    #go left-right
                    mask[i][j]=degree0(i,j,mag)
                elif degree <=157.5 :
                    mask[i][j]=degree45(i,j,mag)
                else :
                    continue
    return mask*mag


def supress_weak_edges(mag,img,min_thresh,max_thresh):
    h,w=img.shape # unpacked the tuple.
    mask=np.zeros((h,w),dtype=np.uint8)
    mask[mag>min_thresh]=1
    mask[mag>max_thresh]=2
    # 1 refers to weak edge 
    # 2 refers to strong edge
    mask_copy=copy.deepcopy(mask)
    kernel=np.ones((3,3))
    dilation=cv.dilate(mask_copy,kernel,iterations=1) 
    
    #we need to check how many 1's are covered by 2, after 2 layer filtering.
    
    for i in range(2,h):
        for j in range(2,w):
            if mask[i][j]== 1 :
                if dilation[i][j]==2:
                    mask[i][j]=2
                else :
                    mask[i][j]=0
            else :
                continue
            
    
    # we got the final mask,
    return (mask//2)*mag 
        
    
            
            

if __name__ == '__main__':
    
    #reading image and conversion
    dir_path="/Users/ygyatso/Documents/some pics/"
    images_list=os.listdir(dir_path)
    get_image_name=images_list[7]
    image_path=os.path.join(dir_path,get_image_name)
    
    #print(image_path)
    
    #image_path="/Users/ygyatso/Documents/Trying_to_replicate/pexels-pixabay-280471.jpg"
    
    color_img=cv.imread(image_path)
    img=cv.imread(image_path,cv.IMREAD_GRAYSCALE)
    img_canny=copy.deepcopy(img)
    #img=img.astype(np.float32)
    
    
    #name="Image"
    smooth_img=cv.GaussianBlur(img,(5,5),0)
    # #processing image for vertical and horizontal edge
    ver_edge=Ix(smooth_img)
    hor_edge=Iy(smooth_img)
    mag=np.sqrt(ver_edge**2 +hor_edge**2)
    
    mag=np.multiply(mag,255.0/np.max(mag)).astype(np.uint8)
    #mag_copy=copy.deepcopy(mag).astype(np.uint8)
    #angle=np.arctan2(hor_edge,ver_edge)
    thin_edge=supress_non_max(ver_edge,hor_edge,mag,img)
    cv.imshow("img",img)
    
    minimum_threshold=0.2*np.max(mag)
    maximum_threshold=0.6*np.max(mag)
    final_edge=supress_weak_edges(mag,img,minimum_threshold,maximum_threshold)
    #cv.imshow("smooth",smooth_img)
    #cv.imshow("diff",img-smooth_img)
    #cv.imshow("hor_edge",hor_edge)
    #cv.imshow("ver_edge",ver_edge)
    
    #trying to figure what extra info is causing image to appear so rusty
    #but not able to found anything or the noise is very less.
    # delta=thin_edge-thin_edge.astype(np.uint8)
    # cv.imshow("delta",delta*200000)
    
    cv.imshow("mag",mag)
    cv.imshow("thin",thin_edge)
    cv.imshow("final",final_edge)
    
    cv.waitKey(0) 
    # cv.destroyAllWindows()
    # this float64 to uint8 data format has fucked my code alot
    # like even the minimalist level
    # Its better to decide what we want.
    #print(t)
    # print(one)