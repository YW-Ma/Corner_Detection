import cv2
import numpy as np
from matplotlib import pyplot as plt
from Correlation_Coefficient import matchTemplate

# read the image

img = cv2.imread('Image\kittens.png',0) #dtype = uint8
img2=img.copy()
template=cv2.imread('Image\paw.png',0)
w,h=template.shape[::-1]

res=matchTemplate(img2,template)
#res=cv2.matchTemplate(img2,template,cv2.TM_CCOEFF_NORMED)

# Visualization
min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(res)
top_left = max_loc
bottom_right = (top_left[0]+w,top_left[1]+h)
cv2.rectangle(img,top_left,bottom_right,255,2)

plt.subplot(121),plt.imshow(template,cmap='gray')
plt.title('Template'),plt.xticks([]),plt.yticks([]) #remove axies labels
plt.subplot(122),plt.imshow(img,cmap='gray')
plt.title('Result'),plt.xticks([]),plt.yticks([])
plt.show()