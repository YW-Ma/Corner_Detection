import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image

def cornerFinder_Harris(img,win_size=5,sigmaXY=0.5,k=0.04,mode=2,param=5):
    """
    Parameters:
        img: input gray image, grayscale(0-255)
        win_size: sliding window size
        sigmaXY: parameter of GaussianBlur model, set it to 0 to turn the model off; Both sigmaX and sigmaY are specified by sigmaXY.
        k: a constant in Harris corner detection. OpenCV tutorial set it to 0.04
        mode & param: corner extraction mode selection
            mode 1: sort all corners by Intensity, select the best points using threshold 'param';
            mode 2: choose the best one of every 'param' x 'param' local regions
    Return:
        A grayscale image(float32) which marks all the corners detected with 1.0f and the rest with 0.0f.
    """
    #1. Get the gradiant of the whole image
    [gx,gy] = np.gradient(img)
    #2. GaussianBlur [optional]
    if(sigmaXY!=0):
        gx=cv2.GaussianBlur(gx,(win_size,win_size),sigmaXY,sigmaXY)
        gy=cv2.GaussianBlur(gy,(win_size,win_size),sigmaXY,sigmaXY) 
    #3. Calculate Intensity pixelwisely. I = det(M)-k*tr(M); M is consisted of gx and gy
    det_M=(gx*gy)-(gx*gy*gx*gy)
    tr_M=gx+gy
    Intensity=det_M-k*tr_M*tr_M
    #4. 
    # mode 1: Sort all coners by Intensity, select the best points using threshold 'param';
    # mode 2: Choose the best one of every 'param' x 'param' local regions
    Intensity=np.abs(Intensity)
    if(mode!=1 and mode!=2):
        print('Invalid mode\n')
        return img
    
    elif(mode==1): #mode 1
        #normalize the Intensity
        Intensity=(Intensity-Intensity.min())/Intensity.max()
        mask=Intensity>param
        Intensity=np.zeros(Intensity.shape)
        Intensity[mask]=1
        return Intensity
    
    elif(mode==2): #mode 2
        Intensity=(Intensity-Intensity.min())/Intensity.max()
        shape_X,shape_Y = Intensity.shape
        mask=Intensity>0
        group_X=shape_X//param
        group_Y=shape_Y//param
        #for the main part
        for step_X in range(group_X-1):
            for step_Y in range(group_Y-1):
                pad=Intensity[(step_X)*param:(step_X+1)*param,(step_Y)*param:(step_Y+1)*param]
                mask[(step_X)*param:(step_X+1)*param,(step_Y)*param:(step_Y+1)*param]=(pad>0.001)*(pad==pad.max())
                
        #for the last line ((step_Y+1)*param will exceed the boundary)
        for step_X in range(group_X-1):
            pad=Intensity[(step_X)*param:(step_X+1)*param,(step_Y)*param:shape_Y]
            mask[(step_X)*param:(step_X+1)*param,(step_Y)*param:shape_Y]=(pad>0.001)*(pad==pad.max())
            
        #for the last column((step_X+1)*param will exceed the boundary)
        for step_Y in range(group_Y-1):
            pad=Intensity[(step_X)*param:shape_X, (step_Y)*param:(step_Y+1)*param]
            mask[(step_X)*param:shape_X, (step_Y)*param:(step_Y+1)*param]=(pad>0.001)*(pad==pad.max())
            
        #to the last block
        pad=Intensity[step_X*param:shape_X,step_Y*param:shape_Y]
        mask[step_X*param:shape_X,step_Y*param:shape_Y]=(pad>0.001)*(pad==pad.max())
        
        Intensity=np.zeros(Intensity.shape)
        Intensity[mask]=1
        
        return Intensity

# read the image
pathName = r'Image\chess.bmp'
img = cv2.imread(pathName) #dtype = uint8
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #dtype = uint8
gray = np.float32(gray) #dtype uint8 --> float32 


# corners = cornerFinder_Harris(gray,5,0.5,0.04,1,0.15) #--> uncomment it to use mode 1
corners = cornerFinder_Harris(gray,5,0.5,0.04,2,5) # mode 2

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