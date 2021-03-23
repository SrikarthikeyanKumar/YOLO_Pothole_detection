import tensorflow as tf
import keras
import numpy as np
import pandas as pd
anchors=np.array(pd.read_csv(r'AnchorFinal.txt',header=None,sep=" "))
anchors=np.reshape(anchors,(int(len(anchors)/2),2))
anchors_tensor=tf.convert_to_tensor(anchors,dtype=tf.float32)
lambda_localisation=5.0
lambda_object=5.0
lambda_noobject=0.5
 
grids=13

def cellgrid(shape):
    hindex=tf.reshape(tf.range(0,limit=shape[0]),(shape[0],1))
    hindex=tf.tile(hindex,[1,shape[1]])
    windex=tf.reshape(tf.range(0,limit=shape[1]),(1,shape[1]))
    windex=tf.tile(windex,[shape[0],1])
    idx=tf.stack([windex,hindex], axis=-1)
    idx=tf.reshape(idx,shape=(1,shape[0],shape[1],1,2))
    return idx
def customloss(y_true, y_pred):
    
    # Classsification loss
    output_shape=y_pred.get_shape().as_list()
    n_anchor=5
    n_class=1
    output=tf.reshape(y_pred, shape=(-1,output_shape[1],output_shape[2],n_anchor,n_class+5) )
    g_truth=tf.reshape(y_true, shape=(-1,output_shape[1],output_shape[2],n_anchor,n_class+5) )
    
 
    
    offset=tf.cast(cellgrid([output_shape[1],output_shape[2]]),output.dtype)
    boxXY_pred=tf.nn.sigmoid(output[...,:2])+offset
    boxWH_pred=tf.math.exp(output[...,2:4])*anchors_tensor
    
    
    boxXY_actual=g_truth[...,:2]*grids
    boxWH_actual=tf.math.exp(g_truth[...,2:4])*anchors_tensor
    
    localisation_loss=tf.square(boxXY_actual-boxXY_pred)+tf.square(tf.sqrt(boxWH_actual)-tf.sqrt(boxWH_pred))
    mask_objectness=g_truth[...,4:5]
    localisation_loss=localisation_loss*mask_objectness
    local_loss=lambda_localisation*tf.math.reduce_sum(localisation_loss,axis=[1,2,3,4])
    
    #classification_loss
    classification_loss=tf.square(g_truth[...,5:]-tf.nn.sigmoid(output[...,5:]))
    classification_loss*=mask_objectness
    class_loss=tf.math.reduce_sum(classification_loss,axis=[1,2,3,4])
    
    
    #confidence loss
    objectscore=tf.nn.sigmoid(output[...,4:5])
    true_objectscore=g_truth[...,4:5]
    #no_object confidence loss
    no_obj_mask=(1-mask_objectness)*lambda_noobject
    no_object_loss=tf.square(0-objectscore)
    no_object_loss*=no_obj_mask
    noobject_loss=tf.math.reduce_sum(no_object_loss,axis=[1,2,3,4])
    
    
    objectness_loss=(tf.square(1-objectscore))
    objectness_loss=objectness_loss*mask_objectness*lambda_object
    object_loss=tf.math.reduce_sum(objectness_loss,axis=[1,2,3,4])
    
    return tf.math.reduce_mean(local_loss+class_loss+noobject_loss+object_loss)