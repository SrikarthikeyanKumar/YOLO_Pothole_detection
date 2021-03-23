import cv2
import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import os

anchors=np.array(pd.read_csv(r'AnchorFinal.txt',header=None,sep=" "))
anchors=np.reshape(anchors,(int(len(anchors)/2),2))

def batch_gen(path,batch_size=8, resize=416):
   YOLO_FORMAT_path=os.path.join(path,'Pothole_YOLO_FORMAT\\*.txt')
   IMG_PATH=os.path.join(path,'Pothole_Images_labelled\\*.jpg')
   
   filename=glob.glob(YOLO_FORMAT_path)
   images=glob.glob(IMG_PATH)
   
   
   #create anchors
   while True:
       for offset in range(0,len(filename),batch_size):
           X_train=[]
           Y_train=[]
           batch_filename_labels=filename[offset:offset+batch_size]
           batch_filename_images=images[offset:offset+batch_size]
           for batch_sample_label,batch_sample_image in zip(batch_filename_labels,batch_filename_images): 
               data=pd.read_csv(batch_sample_label,header=None,sep=" ")
               ytrue_img=np.zeros((13,13,5,6))
               def anchor_IOU(box):
                   gridx=np.floor(box[0]*13)+0.5
                   gridy=np.floor(box[1]*13)+0.5
                   IOU_max=0
                   chosen_anchor=0
                   nanchor=0
               
                   for a in anchors:
                       x1_anchor=gridx-a[0]*13/2
                       y1_anchor=gridy-a[1]*13/2
                       x2_anchor=gridx+a[0]*13/2
                       y2_anchor=gridy+a[1]*13/2
                
                       xa=max(x1_anchor,box[0]*13-box[2]*13/2)
                       xb=min(x2_anchor,box[0]*13+box[2]*13/2)
                       ya=max(y1_anchor,box[1]*13-box[3]*13/2)
                       yb=min(y2_anchor,box[1]*13+box[3]*13/2)
                
                       intersect_area=max(0.0,xb-xa)*max(0.0,yb-ya)
                       box_area=box[2]*box[3]*4*13*13
                       anchor_area=(x2_anchor-x1_anchor)*(y2_anchor-y1_anchor)
                
                       union_area=box_area+anchor_area-intersect_area
                       IOU_anchor=intersect_area/union_area
                
                       if(IOU_anchor>IOU_max):
                           chosen_anchor=nanchor 
                           IOU_max=IOU_anchor
                    
                       nanchor+=1
            
                   return chosen_anchor
               for i in range(0, len(data)):
                   gridx=np.floor(data[1][i]*13)
                   gridy=np.floor(data[2][i]*13)
                   x=data[1][i]
                   y=data[2][i]
                   w=data[3][i]
                   h=data[4][i]
                   box=np.array([x,y,w,h])
                   c_anchor=anchor_IOU(box)
                   box[0]=x
                   box[1]=y
                   box[2]=math.log(box[2]/anchors[c_anchor][0])
                   box[3]=math.log(box[3]/anchors[c_anchor][1])
                   ytrue_img[int(gridx),int(gridy),c_anchor,0:4]=box
                   ytrue_img[int(gridx),int(gridy),c_anchor,4]=1
                   ytrue_img[int(gridx),int(gridy),c_anchor,5]=1
               ytrue_img=ytrue_img.reshape(13,13,30)
               Y_train.append(ytrue_img)
        
               img=cv2.imread(batch_sample_image)
               img=cv2.resize(img, (resize,resize))
               X_train.append(np.float32(img)/255)
           X_train=np.array(X_train)
           Y_train=np.array(Y_train)
           yield X_train, Y_train