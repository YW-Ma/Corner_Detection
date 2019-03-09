import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image

def cornerFinder_Harris(img,win_size=5,sigmaXY=0.4,k=0.04):
    """
    Parameters:
        img: input gray image, grayscale(0-255)
        win_size: sliding window size
        sigmaXY: parameter of GaussianBlur model, set 0 to turn the model off; Both sigmaX and sigmaY are specified by sigmaXY.
        k: a constant in Harris corner detection. OpenCV tutorial set it to 0.04
    Return:
        A grayscale image(float32) with value from 0 to 255 which marks all the corners detected.
    """
    #1. Get the gradiant of the whole image
    [dx,dy] = np.gradient(img)
    #2. GaussianBlur [optional]
    if(sigmaXY!=0):
        dx=cv2.GaussianBlur(dx,(win_size,win_size),sigmaXY,sigmaXY)
        dy=cv2.GaussianBlur(dy,(win_size,win_size),sigmaXY,sigmaXY) 
    #3. Calculate Intensit. I = det(M)-k*tr(M)

    
    cv2.imshow('dx',dx)
    cv2.imshow('dy',dy)
    corners = img
    return corners

pathName = r'Image\chess.bmp'
img = cv2.imread(pathName) #dtype = uint8
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #dtype = uint8
gray = np.float32(gray) #dtype uint8 --> float32 

corners = cornerFinder_Harris(gray,5,0.4,0.04)

img[corners>0.01*corners.max()]=[0,0,255] #mark the result
cv2.imshow('corners',img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()