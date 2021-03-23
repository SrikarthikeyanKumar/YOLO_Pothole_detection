import cv2
import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
anchors=np.array(pd.read_csv(r'AnchorFinal.txt',header=None,sep=" "))
anchors=np.reshape(anchors,(5,2))

def npsigmoid(x):
    z=1/(1+np.exp(-x)) 
    return z

def find_IOUS(max_index,box1,ub1):
    IOU_scores=np.zeros(len(ub1))
    for i in range(len(ub1)):
        if(i==max_index):
            continue
        iou=0
        xa=max(box1[0]-box1[2],ub1[i][0]-ub1[i][2])
        xb=min(box1[0]+box1[2],ub1[i][0]+ub1[i][2])
        ya=max(box1[1]-box1[3],ub1[i][1]-ub1[i][3])
        yb=min(box1[1]+box1[3],ub1[i][1]+ub1[i][3])
                
        intersect_area=max(0.0,xb-xa)*max(0.0,yb-ya)
        box1_area=box1[2]*box1[3]*4
        box2_area=ub1[i][2]*ub1[i][3]*4
        union_area=box1_area+box2_area-intersect_area
        iou=intersect_area/(union_area+1e-8)
        IOU_scores[i]=iou
    
    return IOU_scores

def boundingbox(X,y_true,count,object_thresh=0.5,class_thresh=0.5):
    X=X*255
    img=X.astype(np.uint8)
    y_true=y_true.reshape(13,13,5,6)
    box_list=[]
    for i in range(0,13):
        for j in range(0,13):
            for u in range(0,5): 
                if (npsigmoid(y_true[i,j,u,4])>object_thresh and npsigmoid(y_true[i,j,u,5])>class_thresh):
                    box=y_true[i,j,u,:]
                    bx=int((i+npsigmoid(box[0]))/13*X.shape[0])
                    by=int((j+npsigmoid(box[1]))/13*X.shape[1])
                    bw=int((math.exp(box[2])*anchors[u][0])*X.shape[0]/2)
                    bh=int(math.exp(box[3])*anchors[u][1]*X.shape[1]/2)
                    objectness_score=npsigmoid(y_true[i,j,u,4])
                    object_class=npsigmoid(y_true[i,j,u,5])
                    box_list.append([bx,by,bw,bh,objectness_score,object_class,0,1])
                    print(i,j,u)
                    print(bx,by,bw,bh,objectness_score,object_class)
    box_list=np.array(box_list)  
    #ub- unprocessed_boxes    
    ub=np.array(box_list)  
    n_unprocessed=len(ub)      
    print(n_unprocessed)
    while(n_unprocessed>0):
        mask1=(ub[:,6]==0)
        mask2=(ub[:,7]==1)
        #print(mask2.shape)
        max_index=np.argmax(ub[:,4]*mask1*mask2)
        ub[max_index,6]=1
        IOUS=find_IOUS(max_index,ub[max_index,0:4],ub[:,0:4])
        for u in range(0,len(IOUS)):
            if(IOUS[u]>0.5):
               ub[u,7]=0 
               ub[u,6]=1
        del mask1, mask2 
        n_unprocessed=len(ub)-np.sum(ub[:,6])
        
    pbox=ub
    print(len(pbox))
    for i in range(0, len(pbox)):
        if(pbox[i,7]==1):
            bx=int(pbox[i][0])
            by=int(pbox[i][1])
            bw=int(pbox[i][2])
            bh=int(pbox[i][3])
            score=pbox[i][4]*pbox[i][5]
            cv2.rectangle(img, (bx-bw,by-bh),(bx+bw,by+bh),(0,255,0),2)
            #cv2.putText(img,'Pothole : %f'%score,(bx-bw,by-bh-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,0,0),2)
        
    
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis('off')
    plt.savefig('Results/%d.jpg'%count,dpi=300)
    plt.close()
