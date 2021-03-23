import os
import glob
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

filename=glob.glob(r'Pothole_YOLO_FORMAT\*.txt')
anchor=[]

for file in filename:
    data=pd.read_csv(file,header=None,sep=" ")
    for i in range(0, len(data)):
        anchor.append([data[3][i],data[4][i]])
        
        
anchor_data=np.array(anchor)

#wcss within clsuter sum of squares

wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=600,n_init=10,random_state=0)
    
kmeans.fit(anchor_data)
wcss.append(kmeans.inertia_)


kmeans= KMeans(n_clusters=5, init ='k-means++', max_iter=600, n_init=10,random_state=0)
y_kmeans = kmeans.fit_predict(anchor_data)

anchors_final=(kmeans.cluster_centers_).flatten()

np.savetxt('AnchorFinal.txt',anchors_final,delimiter=" ")

