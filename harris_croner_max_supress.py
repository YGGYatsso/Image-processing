import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
import copy


def Ix(image):  # change in Func / Change in horizontal direction.
    filter=np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
    vertical_edge= cv.filter2D(image,-1,filter)
    return vertical_edge

def Iy(image):  # change in Func / Change in vertical direction.
    filter=np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    horizontal_edge= cv.filter2D(image,-1,filter)
    return horizontal_edge


def intensity_diff(ix,iy):
    """
    Ix^2+Iy^2 +Ix*Iy which approximates change in intensity in local neighborhood of a pixel.
    """
    
    Ixx=np.multiply(ix,ix)
    Iyy=np.multiply(iy,iy)
    Ixy=np.multiply(ix,iy)
    
    return (Ixx+Iyy+2*Ixy)  # 

def higheigen_region(ix,iy,k=0.04):
    """
    provides region in image where image intenisty is high.
    """
    
    Ixx=np.multiply(ix,ix)
    Iyy=np.multiply(iy,iy)
    Ixy=np.multiply(ix,iy)
    
    kernel=np.ones(3,dtype=np.float32)
    
    A=cv.filter2D(Ixx,-1,kernel)
    B=cv.filter2D(Iyy,-1,kernel)
    C=cv.filter2D(Ixy,-1,kernel)
    
    high_val = ( np.multiply(A,B)-np.multiply(C,C)) - k*(A+B)*(A+B)   
    return high_val
    


# need to find the positions where changes are very steep and fine.
# det(A)-lambda*()

def non_max_suppresion(intensity_img,supression_size):
    """
    wherever it is not max, make it 0
    else it remains same. 
    we can use max filter, if element at that position matches max element then its ok,else gg
    """
    copy_intensity = intensity_img
    h,w = intensity_img.shape
    filter_size = supression_size//2 + 1 
    kernel=np.ones((filter_size,filter_size),np.uint8)
    dilation=cv.dilate(intensity_img,kernel,iterations=1)
    
    for i in range(0,h):
        for j in range(0,w):
            if copy_intensity[i][j]< dilation[i][j]:
                copy_intensity[i][j]=0.0
                #print(0)
    
    
    return copy_intensity,dilation



if __name__ == '__main__':
    dir_path="/Users/ygyatso/Documents/some pics/"
    images_list=os.listdir(dir_path)
    #print(images_list)
    get_image_name=images_list[8]
    image_path=os.path.join(dir_path,get_image_name)
    print(image_path)

    color_img=cv.imread(image_path)
    img=cv.imread(image_path,cv.IMREAD_GRAYSCALE)
    name="Image"
    
    gray_blur=cv.GaussianBlur(img,(21,21),0)
    ver_edge=Ix(gray_blur)#/255.0   # vertical edges
    hor_edge=Iy(gray_blur)#/255.0
    

    image_energy= intensity_diff(ver_edge,hor_edge)
    intensity_level=copy.deepcopy(image_energy) # creates an independent copy of the array

    corners=higheigen_region(ver_edge,hor_edge,0.04)



    corners_copy=copy.deepcopy(corners)
    max_supressed_img,dilute = non_max_suppresion(corners_copy ,21)

    #lets also filter some pixels, whose values are less than k*max_value
    max_supressed_img_mask = (max_supressed_img >= 0.9*np.max(max_supressed_img))
    max_supressed_img_mask=max_supressed_img_mask.astype(int)
    max_supressed_img_revamped=np.multiply(max_supressed_img,max_supressed_img_mask)# the corners

    max_supressed_img_revamped=cv.dilate(max_supressed_img_revamped,None)#dilating to better visualise corners location in image

    conv=copy.deepcopy(color_img)
    conv[max_supressed_img_revamped>0]=[255.0,0,0]
    # corners placed on the image

    #print(np.min(hori))
    # cv.imshow("hor_edge",hor_edge)
    # cv.imshow("ver_edge",ver_edge)
    cv.imshow("color_img",color_img)
    cv.imshow("intensity_diff", intensity_level)
    cv.imshow("non_max_supressed_output",max_supressed_img_revamped)
    cv.imshow("dilute",dilute)
    cv.imshow("corners",corners)
    cv.imshow("corners+img",conv)
    cv.waitKey(0) 
    cv.destroyAllWindows()
        
        