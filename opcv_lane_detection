# -*- coding: cp949 -*-
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np

# img 불러오기
img = mpimg.imread('solidWhiteCurve.jpg')

# 함수 정의

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img,(kernel_size, kernel_size),0)

def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img,low_threshold, high_threshold)

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)

    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,)*channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask,vertices,ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)

    return masked_image

def draw_lines(img,lines,color=[255],thickness = 5):
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img,(x1,y1),(x2,y2),color,thickness)

def hough_lines(img,rho,theta,threshold,min_line_len,max_line_gap):
    lines = cv2.HoughLinesP(img,rho,theta,threshold,np.array([]),minLineLength = min_line_len,maxLineGap = max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1],3),dtype = np.uint8)
    
    draw_lines(line_img, lines)
    return line_img

def weighted_img(img, initial_img, alpha = 0.8, beta = 1., gamma = 0.):
    return cv2.addWeighted(initial_img,alpha,img,beta,gamma)

# 회색화면 만들기
gray = grayscale(img)
kernel_size = 5
blur_gray = gaussian_blur(gray,kernel_size)

# Canny Edge Detection

low_threshold = 50
high_threshold = 200
edges = canny(blur_gray, low_threshold,high_threshold)




# # edge 를 딴 figure
# plt.figure(figsize=(10,8))
# plt.imshow(edges, cmap = 'gray')
# plt.show()
# 
# # 까만 화면 만들기 (ROI 재설정)
# mask = np.zeros_like(img)
# plt.figure(figsize=(10,8))
# plt.imshow(mask,cmap='gray')
# plt.show()


# ROI 내의 edge 만 plot 하기
imshape = img.shape

vertices = np.array([[(100,imshape[0]),(450,320),(550,320),(imshape[1]-20,imshape[0])]],dtype = np.int32)

mask = region_of_interest(edges, vertices)

# edge 따라 선 그리기 (HoughLinesP)

rho = 2
theta = np.pi/180
threshold = 90
min_line_len = 120
max_line_gap = 150

lines = hough_lines(mask,rho,theta,threshold,min_line_len,max_line_gap)

lines_edges = weighted_img(lines,img,alpha = 0.8,beta = 1., gamma = 0.)


plt.figure(figsize=(10,8))
plt.imshow(lines_edges)
# plt.imshow(mask, cmap = 'gray')
# plt.imshow(lines, cmap = 'gray')
plt.show()
