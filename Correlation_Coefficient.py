"""
There are two useful functions:
1.    correlationCoef will tell you the coreelation coefficient of two patches of same size
      the greater this coefficient is, the similar this two patches are.
2.    matchTemplate will automatically go through the whole input 'img' with a sliding window
      and implement correlationCoef function on every window comparing it to template.

"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
def correlationCoef(g1,g2):
    """
    Parameters:
        g1: graph one, grayscale(0-255)
        g2: graph two, grayscale(0-255)

    Return:
        Correlation coefficient(float).
    """
    #1. make sure I read the correct patches
    if(g1.shape!=g2.shape):
        print('Invalid patch. Patch should be in same size')
        print('Size of graph 1:',(g1.shape))
        print('Size of graph 2:',(g2.shape))
        return 0
    #2. Calculate Statistic Infomation
    std_g1=np.std(g1)
    std_g2=np.std(g2)
    array1=g1.ravel()
    array2=g2.ravel()
    cov_matrix=np.cov(array1,array2)
    cov=cov_matrix[1,0]
    #3. Calculate coefficient(float)
    coef=cov/(std_g1*std_g2)
    return coef


def matchTemplate(img,template):
    """
    Parameters:
        img: image, such as a cat, grayscale(0-255)
        template: your target, such as a cat's paw, grayscale(0-255)

    Return:
        a float image consisted of correlation coefficient of each pixel.
    """
    
    win_w,win_h=template.shape[::-1]
    w,h=img.shape[::-1]
    result=np.zeros(img.shape)
    for row in range(h-win_h):
        for col in range(w-win_w):    
            t_patch=img[row:row+win_h,col:col+win_w]
            result[row,col]=correlationCoef(template,t_patch)
    return result


