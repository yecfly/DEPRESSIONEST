import numpy as np
import tensorflow as tf
import cv2
import tflearn
from FaceProcessUtil import PreprocessImage as PPI

MAPPING = {0:'neutral', 1:'anger', 2:'surprise', 3:'disgust', 4:'fear', 5:'happy', 6:'sadness'}
MP = './models/'
DEFAULT_PADDING = 'SAME'
TypeThreshold=100
eye_p_shape=[None, 26, 64, 1]
midd_p_shape=[None, 49, 28, 1]
mou_p_shape=[None, 30, 54, 1]
            


###dependent modules for network definition
###
#4 network definition under tflearn
def FacePatches_NET_3Conv_IInception_tflear(eyep, middlep, mouthp):
    e_net=tflearn.conv_2d(eyep, 8, 3, activation='relu',name='eye_conv1_1_3x3')
    e_net=tflearn.conv_2d(e_net, 8, 3, activation='relu',name='eye_conv1_2_3x3')
    e_net=tflearn.max_pool_2d(e_net,2,2,name='eye_pool1')
    e_net=tflearn.conv_2d(e_net, 32, 3, activation='relu', name='eye_conv2_1_3x3')
    e_net=tflearn.conv_2d(e_net, 32, 3, activation='relu', name='eye_conv2_2_3x3')
    e_net=tflearn.max_pool_2d(e_net, 2, 2, name='eye_pool2')
    e_net=tflearn.conv_2d(e_net, 128, 3, activation='relu', name='eye_conv3_1_3x3')
    e_net=tflearn.conv_2d(e_net, 128, 3, activation='relu', name='eye_conv3_2_3x3')
    e_net=tflearn.max_pool_2d(e_net, 2, 2, name='eye_pool3')
    e_net=tflearn.fully_connected(e_net, 1024, activation='tanh', name='eye_fc1')

    mi_net=tflearn.conv_2d(middlep, 8, 3, activation='relu',name='middle_conv1_1_3x3')
    mi_net=tflearn.conv_2d(mi_net, 8, 3, activation='relu',name='middle_conv1_2_3x3')
    mi_net=tflearn.max_pool_2d(mi_net,2,2,name='middle_pool1')
    mi_net=tflearn.conv_2d(mi_net, 32, 3, activation='relu', name='middle_conv2_1_3x3')
    mi_net=tflearn.conv_2d(mi_net, 32, 3, activation='relu', name='middle_conv2_2_3x3')
    mi_net=tflearn.max_pool_2d(mi_net, 2, 2, name='middle_pool2')
    mi_net=tflearn.conv_2d(mi_net, 128, 3, activation='relu', name='middle_conv3_1_3x3')
    mi_net=tflearn.conv_2d(mi_net, 128, 3, activation='relu', name='middle_conv3_2_3x3')
    mi_net=tflearn.max_pool_2d(mi_net, 2, 2, name='middle_pool3')
    mi_net=tflearn.fully_connected(mi_net, 1024, activation='tanh', name='middle_fc1')

    mo_net=tflearn.conv_2d(mouthp, 8, 3, activation='relu',name='mouth_conv1_1_3x3')
    mo_net=tflearn.conv_2d(mo_net, 8, 3, activation='relu',name='mouth_conv1_2_3x3')
    mo_net=tflearn.max_pool_2d(mo_net,2,2,name='mouth_pool1')
    mo_net=tflearn.conv_2d(mo_net, 32, 3, activation='relu', name='mouth_conv2_1_3x3')
    mo_net=tflearn.conv_2d(mo_net, 32, 3, activation='relu', name='mouth_conv2_2_3x3')
    mo_net=tflearn.max_pool_2d(mo_net, 2, 2, name='mouth_pool2')
    mo_net=tflearn.conv_2d(mo_net, 128, 3, activation='relu', name='mouth_conv3_1_3x3')
    mo_net=tflearn.conv_2d(mo_net, 128, 3, activation='relu', name='mouth_conv3_2_3x3')
    mo_net=tflearn.max_pool_2d(mo_net, 2, 2, name='mouth_pool3')
    mo_net=tflearn.fully_connected(mo_net, 1024, activation='tanh', name='mouth_fc1')

    fc_net=tf.concat([e_net,mi_net,mo_net], 1, name='fusion_1')
    fc_net=tflearn.fully_connected(fc_net, 2048, activation='relu', name='fc1')
    fc_net=tflearn.dropout(fc_net, 0.8, name='drop1')
    fc_net=tflearn.fully_connected(fc_net, 2048, activation='relu', name='fc2')
    fc_net=tflearn.dropout(fc_net, 0.8, name='drop2')
    softmax=tflearn.fully_connected(fc_net, 7, activation='softmax', name='prob')
    return softmax
#5 network definition under tflearn
def FacePatches_NET_3Conv_2Inception_tflearn(eyep, middlep, mouthp):
    e_net=tflearn.conv_2d(eyep, 8, 3, activation='relu',name='eye_conv1_1_3x3')
    e_net=tflearn.conv_2d(e_net, 8, 3, activation='relu',name='eye_conv1_2_3x3')
    e_net=tflearn.max_pool_2d(e_net,2,2,name='eye_pool1')
    e_net=tflearn.conv_2d(e_net, 32, 3, activation='relu', name='eye_conv2_1_3x3')
    e_net=tflearn.conv_2d(e_net, 32, 3, activation='relu', name='eye_conv2_2_3x3')
    e_net=tflearn.max_pool_2d(e_net, 2, 2, name='eye_pool2')
    efc2 = tflearn.fully_connected(e_net, 1024, activation='tanh', name='eye_fc2')
    e_net=tflearn.conv_2d(e_net, 128, 3, activation='relu', name='eye_conv3_1_3x3')
    e_net=tflearn.conv_2d(e_net, 128, 3, activation='relu', name='eye_conv3_2_3x3')
    e_net=tflearn.max_pool_2d(e_net, 2, 2, name='eye_pool3')
    e_net=tflearn.fully_connected(e_net, 1024, activation='tanh', name='eye_fc1')
    e_net=tf.concat([e_net, efc2], 1, name='eye_fc')

    mi_net=tflearn.conv_2d(middlep, 8, 3, activation='relu',name='middle_conv1_1_3x3')
    mi_net=tflearn.conv_2d(mi_net, 8, 3, activation='relu',name='middle_conv1_2_3x3')
    mi_net=tflearn.max_pool_2d(mi_net,2,2,name='middle_pool1')
    mi_net=tflearn.conv_2d(mi_net, 32, 3, activation='relu', name='middle_conv2_1_3x3')
    mi_net=tflearn.conv_2d(mi_net, 32, 3, activation='relu', name='middle_conv2_2_3x3')
    mi_net=tflearn.max_pool_2d(mi_net, 2, 2, name='middle_pool2')
    mifc2 = tflearn.fully_connected(mi_net, 1024, activation='tanh', name='middle_fc2')
    mi_net=tflearn.conv_2d(mi_net, 128, 3, activation='relu', name='middle_conv3_1_3x3')
    mi_net=tflearn.conv_2d(mi_net, 128, 3, activation='relu', name='middle_conv3_2_3x3')
    mi_net=tflearn.max_pool_2d(mi_net, 2, 2, name='middle_pool3')
    mi_net=tflearn.fully_connected(mi_net, 1024, activation='tanh', name='middle_fc1')
    mi_net=tf.concat([mi_net, mifc2], 1, name='middle_fc')

    mo_net=tflearn.conv_2d(mouthp, 8, 3, activation='relu',name='mouth_conv1_1_3x3')
    mo_net=tflearn.conv_2d(mo_net, 8, 3, activation='relu',name='mouth_conv1_2_3x3')
    mo_net=tflearn.max_pool_2d(mo_net,2,2,name='mouth_pool1')
    mo_net=tflearn.conv_2d(mo_net, 32, 3, activation='relu', name='mouth_conv2_1_3x3')
    mo_net=tflearn.conv_2d(mo_net, 32, 3, activation='relu', name='mouth_conv2_2_3x3')
    mo_net=tflearn.max_pool_2d(mo_net, 2, 2, name='mouth_pool2')
    mfc2 = tflearn.fully_connected(mo_net, 1024, activation='tanh', name='mouth_fc2')
    mo_net=tflearn.conv_2d(mo_net, 128, 3, activation='relu', name='mouth_conv3_1_3x3')
    mo_net=tflearn.conv_2d(mo_net, 128, 3, activation='relu', name='mouth_conv3_2_3x3')
    mo_net=tflearn.max_pool_2d(mo_net, 2, 2, name='mouth_pool3')
    mo_net=tflearn.fully_connected(mo_net, 1024, activation='tanh', name='mouth_fc1')
    mo_net=tf.concat([mo_net, mfc2], 1, name='mouth_fc')

    fc_net=tf.concat([e_net,mi_net,mo_net], 1, name='fusion_1')
    fc_net=tflearn.fully_connected(fc_net, 2048, activation='relu', name='fc1')
    fc_net=tflearn.dropout(fc_net, 0.8, name='drop1')
    fc_net=tflearn.fully_connected(fc_net, 2048, activation='relu', name='fc2')
    fc_net=tflearn.dropout(fc_net, 0.8, name='drop2')
    softmax=tflearn.fully_connected(fc_net, 7, activation='softmax', name='prob')
    return softmax
#6 network definition under tflearn
def FacePatches_NET_3Conv_3Inception_tflearn(eyep, middlep, mouthp):
    e_net=tflearn.conv_2d(eyep, 8, 3, activation='relu',name='eye_conv1_1_3x3')
    e_net=tflearn.conv_2d(e_net, 8, 3, activation='relu',name='eye_conv1_2_3x3')
    e_net=tflearn.max_pool_2d(e_net,2,2,name='eye_pool1')
    efc3 = tflearn.fully_connected(e_net, 1024, activation='tanh', name='eye_fc3')
    e_net=tflearn.conv_2d(e_net, 32, 3, activation='relu', name='eye_conv2_1_3x3')
    e_net=tflearn.conv_2d(e_net, 32, 3, activation='relu', name='eye_conv2_2_3x3')
    e_net=tflearn.max_pool_2d(e_net, 2, 2, name='eye_pool2')
    efc2 = tflearn.fully_connected(e_net, 1024, activation='tanh', name='eye_fc2')
    e_net=tflearn.conv_2d(e_net, 128, 3, activation='relu', name='eye_conv3_1_3x3')
    e_net=tflearn.conv_2d(e_net, 128, 3, activation='relu', name='eye_conv3_2_3x3')
    e_net=tflearn.max_pool_2d(e_net, 2, 2, name='eye_pool3')
    e_net=tflearn.fully_connected(e_net, 1024, activation='tanh', name='eye_fc1')
    e_net=tf.concat([e_net, efc2, efc3], 1, name='eye_fc')

    mi_net=tflearn.conv_2d(middlep, 8, 3, activation='relu',name='middle_conv1_1_3x3')
    mi_net=tflearn.conv_2d(mi_net, 8, 3, activation='relu',name='middle_conv1_2_3x3')
    mi_net=tflearn.max_pool_2d(mi_net,2,2,name='middle_pool1')
    mifc3 = tflearn.fully_connected(mi_net, 1024, activation='tanh', name='middle_fc3')
    mi_net=tflearn.conv_2d(mi_net, 32, 3, activation='relu', name='middle_conv2_1_3x3')
    mi_net=tflearn.conv_2d(mi_net, 32, 3, activation='relu', name='middle_conv2_2_3x3')
    mi_net=tflearn.max_pool_2d(mi_net, 2, 2, name='middle_pool2')
    mifc2 = tflearn.fully_connected(mi_net, 1024, activation='tanh', name='middle_fc2')
    mi_net=tflearn.conv_2d(mi_net, 128, 3, activation='relu', name='middle_conv3_1_3x3')
    mi_net=tflearn.conv_2d(mi_net, 128, 3, activation='relu', name='middle_conv3_2_3x3')
    mi_net=tflearn.max_pool_2d(mi_net, 2, 2, name='middle_pool3')
    mi_net=tflearn.fully_connected(mi_net, 1024, activation='tanh', name='middle_fc1')
    mi_net=tf.concat([mi_net, mifc2, mifc3], 1, name='middle_fc')

    mo_net=tflearn.conv_2d(mouthp, 8, 3, activation='relu',name='mouth_conv1_1_3x3')
    mo_net=tflearn.conv_2d(mo_net, 8, 3, activation='relu',name='mouth_conv1_2_3x3')
    mo_net=tflearn.max_pool_2d(mo_net,2,2,name='mouth_pool1')
    mfc3 = tflearn.fully_connected(mo_net, 1024, activation='tanh', name='mouth_fc3')
    mo_net=tflearn.conv_2d(mo_net, 32, 3, activation='relu', name='mouth_conv2_1_3x3')
    mo_net=tflearn.conv_2d(mo_net, 32, 3, activation='relu', name='mouth_conv2_2_3x3')
    mo_net=tflearn.max_pool_2d(mo_net, 2, 2, name='mouth_pool2')
    mfc2 = tflearn.fully_connected(mo_net, 1024, activation='tanh', name='mouth_fc2')
    mo_net=tflearn.conv_2d(mo_net, 128, 3, activation='relu', name='mouth_conv3_1_3x3')
    mo_net=tflearn.conv_2d(mo_net, 128, 3, activation='relu', name='mouth_conv3_2_3x3')
    mo_net=tflearn.max_pool_2d(mo_net, 2, 2, name='mouth_pool3')
    mo_net=tflearn.fully_connected(mo_net, 1024, activation='tanh', name='mouth_fc1')
    mo_net=tf.concat([mo_net, mfc2, mfc3], 1, name='mouth_fc')

    fc_net=tf.concat([e_net,mi_net,mo_net], 1, name='fusion_1')
    fc_net=tflearn.fully_connected(fc_net, 2048, activation='relu', name='fc1')
    fc_net=tflearn.dropout(fc_net, 0.8, name='drop1')
    fc_net=tflearn.fully_connected(fc_net, 2048, activation='relu', name='fc2')
    fc_net=tflearn.dropout(fc_net, 0.8, name='drop2')
    softmax=tflearn.fully_connected(fc_net, 7, activation='softmax', name='prob')
    return softmax
###using net 24
def FacePatches_NET_3C_1I_2P(eyep, mouthp):
    ###using net 24
    e_net=tflearn.conv_2d(eyep, 8, 3, activation='relu',name='eye_conv1_1_3x3')
    e_net=tflearn.conv_2d(e_net, 8, 3, activation='relu',name='eye_conv1_2_3x3')
    e_net=tflearn.max_pool_2d(e_net,2,2,name='eye_pool1')
    e_net=tflearn.conv_2d(e_net, 32, 3, activation='relu', name='eye_conv2_1_3x3')
    e_net=tflearn.conv_2d(e_net, 32, 3, activation='relu', name='eye_conv2_2_3x3')
    e_net=tflearn.max_pool_2d(e_net, 2, 2, name='eye_pool2')
    e_net=tflearn.conv_2d(e_net, 128, 3, activation='relu', name='eye_conv3_1_3x3')
    e_net=tflearn.conv_2d(e_net, 128, 3, activation='relu', name='eye_conv3_2_3x3')
    e_net=tflearn.max_pool_2d(e_net, 2, 2, name='eye_pool3')
    e_net=tflearn.fully_connected(e_net, 1024, activation='tanh', name='eye_fc1')

    mo_net=tflearn.conv_2d(mouthp, 8, 3, activation='relu',name='mouth_conv1_1_3x3')
    mo_net=tflearn.conv_2d(mo_net, 8, 3, activation='relu',name='mouth_conv1_2_3x3')
    mo_net=tflearn.max_pool_2d(mo_net,2,2,name='mouth_pool1')
    mo_net=tflearn.conv_2d(mo_net, 32, 3, activation='relu', name='mouth_conv2_1_3x3')
    mo_net=tflearn.conv_2d(mo_net, 32, 3, activation='relu', name='mouth_conv2_2_3x3')
    mo_net=tflearn.max_pool_2d(mo_net, 2, 2, name='mouth_pool2')
    mo_net=tflearn.conv_2d(mo_net, 128, 3, activation='relu', name='mouth_conv3_1_3x3')
    mo_net=tflearn.conv_2d(mo_net, 128, 3, activation='relu', name='mouth_conv3_2_3x3')
    mo_net=tflearn.max_pool_2d(mo_net, 2, 2, name='mouth_pool3')
    mo_net=tflearn.fully_connected(mo_net, 1024, activation='tanh', name='mouth_fc1')

    fc_net=tf.concat([e_net, mo_net], 1, name='fusion_1')
    fc_net=tflearn.fully_connected(fc_net, 2048, activation='relu', name='fc1')
    fc_net=tflearn.dropout(fc_net, 0.8, name='drop1')
    fc_net=tflearn.fully_connected(fc_net, 2048, activation='relu', name='fc2')
    fc_net=tflearn.dropout(fc_net, 0.8, name='drop2')
    softmax=tflearn.fully_connected(fc_net, 7, activation='softmax', name='prob')
    return softmax
###using net 25 
def FacePatches_NET_3C_2I_2P(eyep, mouthp):
    ###using net 25 
    e_net=tflearn.conv_2d(eyep, 8, 3, activation='relu',name='eye_conv1_1_3x3')
    e_net=tflearn.conv_2d(e_net, 8, 3, activation='relu',name='eye_conv1_2_3x3')
    e_net=tflearn.max_pool_2d(e_net,2,2,name='eye_pool1')
    e_net=tflearn.conv_2d(e_net, 32, 3, activation='relu', name='eye_conv2_1_3x3')
    e_net=tflearn.conv_2d(e_net, 32, 3, activation='relu', name='eye_conv2_2_3x3')
    e_net=tflearn.max_pool_2d(e_net, 2, 2, name='eye_pool2')
    efc2 = tflearn.fully_connected(e_net, 1024, activation='tanh', name='eye_fc2')
    e_net=tflearn.conv_2d(e_net, 128, 3, activation='relu', name='eye_conv3_1_3x3')
    e_net=tflearn.conv_2d(e_net, 128, 3, activation='relu', name='eye_conv3_2_3x3')
    e_net=tflearn.max_pool_2d(e_net, 2, 2, name='eye_pool3')
    e_net=tflearn.fully_connected(e_net, 1024, activation='tanh', name='eye_fc1')
    e_net=tf.concat([e_net, efc2], 1, name='eye_fc')

    mo_net=tflearn.conv_2d(mouthp, 8, 3, activation='relu',name='mouth_conv1_1_3x3')
    mo_net=tflearn.conv_2d(mo_net, 8, 3, activation='relu',name='mouth_conv1_2_3x3')
    mo_net=tflearn.max_pool_2d(mo_net,2,2,name='mouth_pool1')
    mo_net=tflearn.conv_2d(mo_net, 32, 3, activation='relu', name='mouth_conv2_1_3x3')
    mo_net=tflearn.conv_2d(mo_net, 32, 3, activation='relu', name='mouth_conv2_2_3x3')
    mo_net=tflearn.max_pool_2d(mo_net, 2, 2, name='mouth_pool2')
    mfc2 = tflearn.fully_connected(mo_net, 1024, activation='tanh', name='mouth_fc2')
    mo_net=tflearn.conv_2d(mo_net, 128, 3, activation='relu', name='mouth_conv3_1_3x3')
    mo_net=tflearn.conv_2d(mo_net, 128, 3, activation='relu', name='mouth_conv3_2_3x3')
    mo_net=tflearn.max_pool_2d(mo_net, 2, 2, name='mouth_pool3')
    mo_net=tflearn.fully_connected(mo_net, 1024, activation='tanh', name='mouth_fc1')
    mo_net=tf.concat([mo_net, mfc2], 1, name='mouth_fc')

    fc_net=tf.concat([e_net, mo_net], 1, name='fusion_1')
    fc_net=tflearn.fully_connected(fc_net, 2048, activation='relu', name='fc1')
    fc_net=tflearn.dropout(fc_net, 0.8, name='drop1')
    fc_net=tflearn.fully_connected(fc_net, 2048, activation='relu', name='fc2')
    fc_net=tflearn.dropout(fc_net, 0.8, name='drop2')
    softmax=tflearn.fully_connected(fc_net, 7, activation='softmax', name='prob')
    return softmax
###using net 26
def FacePatches_NET_3C_3I_2P(eyep, mouthp):
    ###using net 26
    e_net=tflearn.conv_2d(eyep, 8, 3, activation='relu',name='eye_conv1_1_3x3')
    e_net=tflearn.conv_2d(e_net, 8, 3, activation='relu',name='eye_conv1_2_3x3')
    e_net=tflearn.max_pool_2d(e_net,2,2,name='eye_pool1')
    efc3 = tflearn.fully_connected(e_net, 1024, activation='tanh', name='eye_fc3')
    e_net=tflearn.conv_2d(e_net, 32, 3, activation='relu', name='eye_conv2_1_3x3')
    e_net=tflearn.conv_2d(e_net, 32, 3, activation='relu', name='eye_conv2_2_3x3')
    e_net=tflearn.max_pool_2d(e_net, 2, 2, name='eye_pool2')
    efc2 = tflearn.fully_connected(e_net, 1024, activation='tanh', name='eye_fc2')
    e_net=tflearn.conv_2d(e_net, 128, 3, activation='relu', name='eye_conv3_1_3x3')
    e_net=tflearn.conv_2d(e_net, 128, 3, activation='relu', name='eye_conv3_2_3x3')
    e_net=tflearn.max_pool_2d(e_net, 2, 2, name='eye_pool3')
    e_net=tflearn.fully_connected(e_net, 1024, activation='tanh', name='eye_fc1')
    e_net=tf.concat([e_net, efc2, efc3], 1, name='eye_fc')

    mo_net=tflearn.conv_2d(mouthp, 8, 3, activation='relu',name='mouth_conv1_1_3x3')
    mo_net=tflearn.conv_2d(mo_net, 8, 3, activation='relu',name='mouth_conv1_2_3x3')
    mo_net=tflearn.max_pool_2d(mo_net,2,2,name='mouth_pool1')
    mfc3 = tflearn.fully_connected(mo_net, 1024, activation='tanh', name='mouth_fc3')
    mo_net=tflearn.conv_2d(mo_net, 32, 3, activation='relu', name='mouth_conv2_1_3x3')
    mo_net=tflearn.conv_2d(mo_net, 32, 3, activation='relu', name='mouth_conv2_2_3x3')
    mo_net=tflearn.max_pool_2d(mo_net, 2, 2, name='mouth_pool2')
    mfc2 = tflearn.fully_connected(mo_net, 1024, activation='tanh', name='mouth_fc2')
    mo_net=tflearn.conv_2d(mo_net, 128, 3, activation='relu', name='mouth_conv3_1_3x3')
    mo_net=tflearn.conv_2d(mo_net, 128, 3, activation='relu', name='mouth_conv3_2_3x3')
    mo_net=tflearn.max_pool_2d(mo_net, 2, 2, name='mouth_pool3')
    mo_net=tflearn.fully_connected(mo_net, 1024, activation='tanh', name='mouth_fc1')
    mo_net=tf.concat([mo_net, mfc2, mfc3], 1, name='mouth_fc')

    fc_net=tf.concat([e_net,mo_net], 1, name='fusion_1')
    fc_net=tflearn.fully_connected(fc_net, 2048, activation='relu', name='fc1')
    fc_net=tflearn.dropout(fc_net, 0.8, name='drop1')
    fc_net=tflearn.fully_connected(fc_net, 2048, activation='relu', name='fc2')
    fc_net=tflearn.dropout(fc_net, 0.8, name='drop2')
    softmax=tflearn.fully_connected(fc_net, 7, activation='softmax', name='prob')
    return softmax

#
def getModelPathForPrediction(mid=0):
    #if mid==300:
    #    mp=MP+'D502_M3_N3_T0_V0_R4_20171009235521_1.1895357370_.ckpt-16197'#0.9587	

    #elif mid==301:
    #    mp=MP+'D502_M3_N3_T4_V4_R4_20171010084104_1.2033878565_.ckpt-18110'#0.9165

    #elif mid==303:
    #    mp=MP+'D502_M3_N3_T5_V5_R4_20171010103653_1.1808838844_.ckpt-19024'#0.9779
    if mid==400:
        mp=MP+'';
    elif mid==500:
        mp=MP+'';
    elif mid==600:
        mp=MP+'';
    else:
        print('Unexpected Model ID. TRY another one.')
        exit(-1)
    return mp

#model for prediction
class FacePatchesModel:
    def __init__(self, mid=300):
        ###define the graph
        self.networkGraph=tf.Graph()
        with self.networkGraph.as_default():
            self.eye_p = tf.placeholder(tf.float32, eye_p_shape)
            self.mou_p = tf.placeholder(tf.float32, mou_p_shape)
            #if (mid//TypeThreshold)==3:
            #    self.network = FacePatches_NET_3Conv_2Inception({'eyePatch_data':self.eye_p,
            #                                                    'middlePatch_data':self.midd_p,
            #                                                    'mouthPatch_data':self.mou_p})
            if (mid//TypeThreshold)<7 and (mid//TypeThreshold)>3:
                self.midd_p = tf.placeholder(tf.float32, midd_p_shape)
                self.prob = FacePatches_NET_3Conv_IInception_tflear(self.eye_p,
                                                                self.midd_p, self.mou_p)
            elif (mid//TypeThreshold) >23 and (mid//TypeThreshold) <27:
                self.prob = FacePatches_NET_3Conv_IInception_tflear(self.eye_p, self.mou_p)

            else:
                print('ERROR: Unexpected network type. Try another mid')
                exit(-1)
            self.saver=tf.train.Saver()

            ###load pretrained model
            self.sess=tf.InteractiveSession(graph=self.networkGraph)
            try:
                #must initialize the variables in the graph for compution or loading pretrained weights
                self.sess.run(tf.variables_initializer(var_list=self.networkGraph.get_collection(name='variables')))
                print('Network variables initialized.')
                #the saver must define in the graph of its owner session, or it will occur error in restoration or saving
                self.saver.restore(sess=self.sess, save_path=getModelPathForPrediction(mid))
                print('Network Model loaded\n')
            except:
                print('ERROR: Unable to load the pretrained network.')
                traceback.print_exc()
                exit(2)

    def predict(self, eye_p, midd_p, mou_p):#img must have the shape of [1, 128, 128, 1]
        probability = self.prob.eval(feed_dict={self.eye_p:eye_p, self.midd_p:midd_p, self.mou_p:mou_p})
        emotion = MAPPING[np.argmax(probability)]
        return emotion, probability
