#!/usr/bin/env python
# coding: utf-8

# 
# 
#


#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
import cv2
#get_ipython().run_line_magic('matplotlib', 'inline')


# ## Read in an Image

# In[2]:


#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')




import math

def grayscale(img):
    """Applies the Grayscale transform
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=5):
 
    left_group_x=[]
    right_group_x=[]
    left_group_y=[]
    right_group_y=[]
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            m = (y2-y1)/(x2-x1)
       
            
                
            if m <= 0:
                left_group_x.extend([x1,x2])
                left_group_y.extend([y1,y2])
            else:
                right_group_x.extend([x1,x2])
                right_group_y.extend([y1,y2])
                
                
    if not all([left_group_x] and [left_group_y] and [right_group_x] and [right_group_y]):
        return

        
    max_y = int(320)
    min_y = int(img.shape[0]) 

    line_fit = np.poly1d(np.polyfit(left_group_y,left_group_x,deg=1))

    max_x1 = int(line_fit(max_y))
    min_x1 = int(line_fit(min_y))
    
    line_fit2 = np.poly1d(np.polyfit(right_group_y,right_group_x,deg=1))

    max_x2 = int(line_fit2(max_y))
    min_x2 = int(line_fit2(min_y))



    cv2.line(img, (max_x1, max_y), (min_x1, min_y), color, thickness)
    cv2.line(img, (max_x2, max_y), (min_x2, min_y), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img



def weighted_img(img, initial_img, a=0.8, b=1., gama=0.):
 
    
    return cv2.addWeighted(initial_img, a, img, b, gama)





import os
os.listdir("test_images/")


# ## Build a Lane Finding Pipeline




def pipline_lane(image):
    imshape = image.shape
    im1 = grayscale(image) # convert grayscle to meet the canny edge requirements
    kernel_size = 1  # kernel window size
    im1 = gaussian_blur(im1,kernel_size)    # blur image to remove noise
    im1 = canny(im1, 50,150 )  # get edge deetections by using gradients

    vertices = np.array([[(0,imshape[0]),(450, 310), (490, 310), (imshape[1],imshape[0])]], dtype=np.int32)  #vectices of a trapazium like polygon

    masked_edges = region_of_interest(im1, vertices) # Edges detected within the region of interest

    # parameters for hough transform function
    rho = 6 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 160     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 40 #minimum number of pixels making up a line
    max_line_gap = 25    # maximum gap in pixels between connectable line segments

    lines = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)


    final = weighted_img(lines, image, a=0.8, b=1., gama=0.)
    

    #plt.imshow(final)
# then save them to the test_images_output directory.
    cv2.imwrite('test_images/test.jpg',final)
    return final




# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


# In[31

def process_image(image):
  
    result = pipline_lane(image)
    return result




white_output = 'test_videos_output/solidWhiteRight.mp4'

clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) 
get_ipython().run_line_magic('time', 'white_clip.write_videofile(white_output, audio=False)')





yellow_output = 'test_videos_output/solidYellowLeft.mp4'

clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
get_ipython().run_line_magic('time', 'yellow_clip.write_videofile(yellow_output, audio=False)')




