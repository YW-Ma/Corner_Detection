import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image

def cornerFinder_Moravec(img,win_grad=5,win_sup=7):
    """
    win_supeters:
        img: input gray image, grayscale(0-255)
        win_grad: window size for the window calculating IV(Interest Value)
                  IV: the sum of squares of d-value between the middle pixel and neigbour pixels along four directions
        win_sup: non-maximum suppression window size
    Return:
        A grayscale image(float32) which marks all the corners detected with 1.0f and the rest with 0.0f.
    """
    #1. Data preperation: calculating the d-value of four directions.
    #   This will save runtime of 'pow' function
    #1.1 Data preperation:
    #img_U: 90(270) degree - everypixel move upward for one pixel distence
    temp_1 = img[0,:]
    temp_1 = temp_1[np.newaxis,:]
    temp_2 = img[1:img.shape[0],:]
    img_U  = np.vstack((temp_2,temp_1))
    #img_R: 0(180) degree - everypixel move to right for one pixel distence
    temp_1 = img[:,img.shape[1]-1]
    temp_1 = temp_1[:,np.newaxis]
    temp_2 = img[:,0:img.shape[1]-1]
    img_R  = np.hstack((temp_1,temp_2))
    #img_UR: 45(225) degree
    temp_1 = img_U[:,img.shape[1]-1]
    temp_1 = temp_1[:,np.newaxis]
    temp_2 = img_U[:,0:img.shape[1]-1]
    img_UR  = np.hstack((temp_1,temp_2))
    #img_UL: 135(315) degree
    temp_1 = img_U[:,0]
    temp_1 = temp_1[:,np.newaxis]
    temp_2 = img_U[:,1:img.shape[1]]
    img_UL  = np.hstack((temp_2,temp_1))
    #1.2 Calculating the D-Value
    diff_R = img_R-img
    diff_R = diff_R**2

    diff_UR = img_UR-img
    diff_UR = diff_UR**2

    diff_U = img_U-img
    diff_U = diff_U**2

    diff_UL = img_UL-img
    diff_UL = diff_UL**2

    #2. Get IV(Interest Value) of each pixel and implement Non-maximum suppression on them.
    shape_X,shape_Y = img.shape
    group_X=shape_X//win_sup
    group_Y=shape_Y//win_sup
    #for the main part
    for step_X in range(group_X-1):
        for step_Y in range(group_Y-1):
            
            # pad=img[(step_X)*win_sup:(step_X+1)*win_sup,(step_Y)*win_sup:(step_Y+1)*win_sup]
            # for i in range(win_sup):
            #     for j in range(win_sup):
            #         IV=np.zeros(4)
            #         IV[0]=
        
    #for the last line ((step_Y+1)*win_sup will exceed the boundary)
    for step_X in range(group_X-1):
        # pad=img[(step_X)*win_sup:(step_X+1)*win_sup,(step_Y)*win_sup:shape_Y]
        # mask[(step_X)*win_sup:(step_X+1)*win_sup,(step_Y)*win_sup:shape_Y]=(pad>0.001)*(pad==pad.max())
            
        
    #for the last column((step_X+1)*win_sup will exceed the boundary)
    for step_Y in range(group_Y-1):
        # pad=img[(step_X)*win_sup:shape_X, (step_Y)*win_sup:(step_Y+1)*win_sup]
        # mask[(step_X)*win_sup:shape_X, (step_Y)*win_sup:(step_Y+1)*win_sup]=(pad>0.001)*(pad==pad.max())


    return img

# read the image
pathName = r'Image\chess.bmp'
img = cv2.imread(pathName) #dtype = uint8
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #dtype = uint8
gray = np.float32(gray) #dtype uint8 --> float32 

corners = cornerFinder_Moravec(gray,5,7)

#mark the result
#1. get the location of corners
shape=corners.shape
row=shape[0]
col=shape[1]
corners=corners.reshape(row*col)
corners=np.squeeze(corners)
index=np.where(corners==1)
index=np.array(index)
index=np.squeeze(index)

index_y=index//col
index_x=index-index_y*col

#2. mark corners with red circle, users can personalize the radius of them.
for item in range(index.size):
    cv2.circle(img,(index_x[item],index_y[item]),1,(0,0,255),-1) #mark the result

cv2.imshow('corners',img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()