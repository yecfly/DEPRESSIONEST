#Here starts the depression estimation processes
##from 2017.09.26, the stop criteria of training is changing to 
####patched on 20180915, fix the bug(implementation logic error for tlabels) in Valid_on_TestSet_3NI
##
import numpy as np
import tensorflow as tf
import os, pickle, time, sys, traceback, collections
import DataSetPrepare
import tflearn

from DataSetPrepare import Dataset_Dictionary
import win_unicode_console
win_unicode_console.enable()

continue_test=True #set continuesly test for M7
OverTimes=20

M3N4S1={'eye_conv1_1_3x3/W:0':0,'eye_conv1_1_3x3/b:0':0,
        'eye_conv1_2_3x3/W:0':0,'eye_conv1_2_3x3/b:0':0,
        'eye_conv2_1_3x3/W:0':0,'eye_conv2_1_3x3/b:0':0,
        'eye_conv2_2_3x3/W:0':0,'eye_conv2_2_3x3/b:0':0,
        'eye_fc2/W:0':0, 'eye_fc2/b:0':0,
        'eye_conv3_1_3x3/W:0':0,'eye_conv3_1_3x3/b:0':0,
        'eye_conv3_2_3x3/W:0':0,'eye_conv3_2_3x3/b:0':0,
        'eye_fc1/W:0':0,'eye_fc1/b:0':0}
M3N4S2={'middle_conv1_1_3x3/W:0':0,'middle_conv1_1_3x3/b:0':0,
        'middle_conv1_2_3x3/W:0':0,'middle_conv1_2_3x3/b:0':0,
        'middle_conv2_1_3x3/W:0':0,'middle_conv2_1_3x3/b:0':0,
        'middle_conv2_2_3x3/W:0':0,'middle_conv2_2_3x3/b:0':0,
        'middle_conv3_1_3x3/W:0':0,'middle_conv3_1_3x3/b:0':0,
        'middle_conv3_2_3x3/W:0':0,'middle_conv3_2_3x3/b:0':0,
        'middle_fc1/W:0':0,'middle_fc1/b:0':0}
M3N4S3={'mouth_conv1_1_3x3/W:0':0,'mouth_conv1_1_3x3/b:0':0,
        'mouth_conv1_2_3x3/W:0':0,'mouth_conv1_2_3x3/b:0':0,
        'mouth_conv2_1_3x3/W:0':0,'mouth_conv2_1_3x3/b:0':0,
        'mouth_conv2_2_3x3/W:0':0,'mouth_conv2_2_3x3/b:0':0,
        'mouth_conv3_1_3x3/W:0':0,'mouth_conv3_1_3x3/b:0':0,
        'mouth_conv3_2_3x3/W:0':0,'mouth_conv3_2_3x3/b:0':0,
        'mouth_fc1/W:0':0,'mouth_fc1/b:0':0}
M3N5S1={'eye_conv1_1_3x3/W:0':0,'eye_conv1_1_3x3/b:0':0,
        'eye_conv1_2_3x3/W:0':0,'eye_conv1_2_3x3/b:0':0,
        'eye_conv2_1_3x3/W:0':0,'eye_conv2_1_3x3/b:0':0,
        'eye_conv2_2_3x3/W:0':0,'eye_conv2_2_3x3/b:0':0,
        'eye_conv3_1_3x3/W:0':0,'eye_conv3_1_3x3/b:0':0,
        'eye_conv3_2_3x3/W:0':0,'eye_conv3_2_3x3/b:0':0,
        'eye_fc1/W:0':0,'eye_fc1/b:0':0}
M3N5S2={'middle_conv1_1_3x3/W:0':0,'middle_conv1_1_3x3/b:0':0,
        'middle_conv1_2_3x3/W:0':0,'middle_conv1_2_3x3/b:0':0,
        'middle_conv2_1_3x3/W:0':0,'middle_conv2_1_3x3/b:0':0,
        'middle_conv2_2_3x3/W:0':0,'middle_conv2_2_3x3/b:0':0,
        'middle_fc2/W:0':0, 'middle_fc2/b:0':0,
        'middle_conv3_1_3x3/W:0':0,'middle_conv3_1_3x3/b:0':0,
        'middle_conv3_2_3x3/W:0':0,'middle_conv3_2_3x3/b:0':0,
        'middle_fc1/W:0':0,'middle_fc1/b:0':0}
M3N5S3={'mouth_conv1_1_3x3/W:0':0,'mouth_conv1_1_3x3/b:0':0,
        'mouth_conv1_2_3x3/W:0':0,'mouth_conv1_2_3x3/b:0':0,
        'mouth_conv2_1_3x3/W:0':0,'mouth_conv2_1_3x3/b:0':0,
        'mouth_conv2_2_3x3/W:0':0,'mouth_conv2_2_3x3/b:0':0,
        'mouth_fc2/W:0':0, 'mouth_fc2/b:0':0,
        'mouth_conv3_1_3x3/W:0':0,'mouth_conv3_1_3x3/b:0':0,
        'mouth_conv3_2_3x3/W:0':0,'mouth_conv3_2_3x3/b:0':0,
        'mouth_fc1/W:0':0,'mouth_fc1/b:0':0}

lr_drate=0.8
batchsize_step=0
times=20 #which control the decay learning rate decays at every %times% epochs
test_bat=200
TestNumLimit = 200
Mini_Epochs = 140
show_threshold = 1.62
class SIMSTS():
    def __init__(self, NC):
        self.min=1.0
        self.max=0.0
        self.amout=0
        self.mean=0
        self.count=NC
    def addFigure(self, figure):
        if self.min>figure:
            self.min=figure
        if self.max<figure:
            self.max=figure
        self.amout=self.amout+figure

    def getSTS(self):
        self.mean=self.amout/self.count 
        return self.mean, self.max, self.min

    def logfile(self, Module, Dataset, Network, NE, MSS, MSL):
        filename='./logs/M%dtests/D%d_N%d.txt'%(Module, Dataset, Network)
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        filein=open(filename,'a')
        filein.write('MEAN:%.6f\tMAX:%.6f\tMIN:%.6f\tnum_estimators:%d\tmin_samples_split:%d\tmin_samples_leaf:%d\tD%d\tN%d\n'%(self.mean,
                                                                  self.max, self.min, NE, MSS, MSL,Dataset, Network))
        filein.close()

def initialize_dirs():
    if not os.path.exists('./logs/VL'):
        os.makedirs('./logs/VL')
    if not os.path.exists('./saves'):
        os.makedirs('./saves')
class LOSS_ANA:
    '''The LOSS_ANA class collects the training losses and analyzes them.
        The initial length should be divided by 50 with no remainder.'''
    def __init__(self):
        self.__Validation_Loss_List = []
        self.__Current_Length = 0#indicates whether the Validation_Loss_List has reach the maximum Length
        self.__Min_Loss = 10000.0
        self.__Min_Loss_Second = 10001.0

    @property
    def minimun_loss(self):
        return self.__Min_Loss
    @property
    def second_minimun_loss(self):
        return self.__Min_Loss_Second
    @property
    def loss_length(self):
        return self.__Current_Length

    def setMinimun_loss(self, m):
        self.__Min_Loss=m

    def analyzeLossVariation(self, loss):
        '''Analize the LastN*2 validation losses, where LastN is defined in __init__
        Inputs:
            loss: float type, the current loss of the validation set
            
        Outputs:
            boolean type: indicates whether the input is less than all others before it
        '''
        self.__Current_Length = self.__Current_Length + 1
        flag=False
        if loss < self.__Min_Loss:
            self.__Min_Loss_Second = self.__Min_Loss
            self.__Min_Loss = loss
            flag=True
        self.__Validation_Loss_List.append(loss)
        return flag

    def outputlosslist(self, logfilename):
        '''input the file name to log out all the validation losses in the current training'''
        fw=open(logfilename,'w')
        for v in self.__Validation_Loss_List:
            fw.write('%.16f\n'%(v))
        fw.close()
def calR(predict_labels_in, groundtruth_labels_in, cn=7):
    #print(len(predict_labels_in.shape))
    #print(len(predict_labels_in))
    #print(len(np.asarray(groundtruth_labels_in).shape))
    #print(len(groundtruth_labels_in))
    #exit()
    if len(np.asarray(predict_labels_in).shape)==1:
        predict_labels=DataSetPrepare.dense_to_one_hot(predict_labels_in, cn)
        #print(predict_labels.shape)
    else:
        predict_labels=predict_labels_in
    if len(np.asarray(groundtruth_labels_in).shape)==1:
        groundtruth_labels=DataSetPrepare.dense_to_one_hot(groundtruth_labels_in, cn)
        #print(groundtruth_labels.shape)
    else:
        groundtruth_labels=groundtruth_labels_in
    assert len(predict_labels)==len(groundtruth_labels), ('predict_labels length: %d groundtruth_labels length: %d' % (len(predict_labels), len(groundtruth_labels)))
    nc=len(groundtruth_labels)
    g_c=np.zeros([cn])
    #confusion_mat=[[0,0,0,0,0,0,0],
    #        [0,0,0,0,0,0,0],
    #        [0,0,0,0,0,0,0],
    #        [0,0,0,0,0,0,0],
    #        [0,0,0,0,0,0,0],
    #        [0,0,0,0,0,0,0],
    #        [0,0,0,0,0,0,0]]
    confusion_mat=list(np.zeros([cn,cn]))
    for i in range(nc):
        cmi=list(groundtruth_labels[i]).index(max(groundtruth_labels[i]))
        g_c[cmi]=g_c[cmi]+1
        pri=list(predict_labels[i]).index(max(predict_labels[i]))
        confusion_mat[cmi][pri]=confusion_mat[cmi][pri]+1
    for i in range(len(g_c)):
        if g_c[i]>0:
            confusion_mat[i]=list(np.asarray(confusion_mat[i])/g_c[i])
    return confusion_mat
def overAllAccuracy(conf_m, afc=None):
    accuracy_for_every_categary=[]
    r=len(conf_m)
    if r>0:
        c=len(conf_m[0])
    else:
        print('ERROR: Confusion Matrix is unexpected.')
        exit()
    assert r==c, ('ERROR: Confusion Matrix is unexpected for its unequal rows and cols: %d %d'%(r,c))
    ac=0.0
    for i in range(r):
        ac=ac+conf_m[i][i]
        accuracy_for_every_categary.append(conf_m[i][i])
    ac=ac/r
    if not afc is None:
        afc=afc.extend(accuracy_for_every_categary)
    del accuracy_for_every_categary
    return ac
def Valid_on_TestSet(cn, sess, accuracy, sum_test, loss, softmax,
                       placeholder1, placeholder1_input, 
                       placeholder_labels, placeholder_labels_input,afc=None):
    '''Evalute the data with 1 network input in the session input
Inputs:
    sess:
    accuracy:
    sum_test:
    loss:
    softmax:

    
Outputs:
    v_accuracy:
    valid_loss:
    oaa:
    confu_mat'''
    ncount=len(placeholder_labels_input)
    tlabels=[]
    if ncount>TestNumLimit:               
        test_iter=np.floor_divide(ncount,test_bat)
        v_accuracy=0
        valid_loss=0
        for ite in range(test_iter):
            start=test_bat*ite
            end=test_bat*(ite+1)
            st, v_loss, tlab=sess.run([sum_test, loss, softmax], feed_dict={placeholder1:placeholder1_input[start:end],
                                                            placeholder_labels:placeholder_labels_input[start:end]})
            v_accuracy=v_accuracy+st
            valid_loss=valid_loss+v_loss
            tlabels.extend(tlab)
        if ncount%test_bat>0:
            st, v_loss, tlab=sess.run([sum_test, loss, softmax], feed_dict={placeholder1:placeholder1_input[test_bat*test_iter:ncount], 
                                        placeholder_labels:placeholder_labels_input[test_bat*test_iter:ncount]})
            v_accuracy=v_accuracy+st
            valid_loss=valid_loss+v_loss
        v_accuracy=v_accuracy/ncount
        valid_loss=valid_loss/(test_iter+1)
        tlabels.extend(tlab)
    else:
        v_accuracy, valid_loss, tlab = sess.run([accuracy, loss, softmax], feed_dict={placeholder1:placeholder1_input, 
                                                                       placeholder_labels:placeholder_labels_input})
        tlabels.extend(tlab)
    confu_mat=calR(tlabels, placeholder_labels_input, cn)
    oaa=overAllAccuracy(confu_mat,afc=afc)
    return v_accuracy, valid_loss, oaa, confu_mat
def Valid_on_TestSet_3NI(cn, sess, accuracy, sum_test, loss, softmax,
                       placeholder1, placeholder1_input, 
                       placeholder2, placeholder2_input,
                       placeholder3, placeholder3_input,
                       placeholder_labels, placeholder_labels_input, afc=None):
    '''Evalute the data with 3 network inputs in the session input
Inputs:
    sess:
    accuracy:
    sum_test:
    loss:
    softmax:

    
Outputs:
    v_accuracy:
    valid_loss:
    oaa:
    confu_mat'''
    ncount=len(placeholder_labels_input)
    tlabels=[]
    if ncount>TestNumLimit:               
        test_iter=np.floor_divide(ncount,test_bat)
        v_accuracy=0
        valid_loss=0
        for ite in range(test_iter):
            start=test_bat*ite
            end=test_bat*(ite+1)
            st, v_loss, tlab=sess.run([sum_test, loss, softmax], feed_dict={placeholder1:placeholder1_input[start:end],
                                                                            placeholder2:placeholder2_input[start:end],
                                                                            placeholder3:placeholder3_input[start:end],
                                                                            placeholder_labels:placeholder_labels_input[start:end]})
            v_accuracy=v_accuracy+st
            valid_loss=valid_loss+v_loss
            tlabels.extend(tlab)
        if ncount%test_bat>0:
            st, v_loss, tlab=sess.run([sum_test, loss, softmax], feed_dict={placeholder1:placeholder1_input[test_bat*test_iter:ncount],
                                                                        placeholder2:placeholder2_input[test_bat*test_iter:ncount],
                                                                        placeholder3:placeholder3_input[test_bat*test_iter:ncount],
                                                                        placeholder_labels:placeholder_labels_input[test_bat*test_iter:ncount]})
            tlabels.extend(tlab)
        v_accuracy=v_accuracy+st
        valid_loss=valid_loss+v_loss
        v_accuracy=v_accuracy/ncount
        valid_loss=valid_loss/(test_iter+1)
        
    else:
        v_accuracy, valid_loss, tlab = sess.run([accuracy, loss, softmax], feed_dict={placeholder1:placeholder1_input, 
                                                                                         placeholder2:placeholder2_input,
                                                                                         placeholder3:placeholder3_input,
                                                                                         placeholder_labels:placeholder_labels_input})
        tlabels.extend(tlab)
    confu_mat=calR(tlabels, placeholder_labels_input, cn)
    oaa=overAllAccuracy(confu_mat, afc=afc)
    return v_accuracy, valid_loss, oaa, confu_mat
def logfile(file_record, runs, OAA, afc, valid_loss, valid_min_loss, final_train_loss, train_min_loss, TA, TC, ILR, FLR, LS, ites, Epo, cBS, iBS, input, CM, T, df):
    file_record="Run%02d\tOverAllACC:%0.8f\tTestAccuracy:%.8f\tACs: %s\tFinalLoss:%.10f\tMinimunLoss:%.10f\tFinaltrainloss:%.10f\tMinimumtrainloss:%.10f\tTimeComsumed:%08.6f\tInitialLearningRate:%.8f\tFinalLearningRate:%.8f\tLearningStepForDroppingMagnitude:%08d\tTotalIterations:%08d\tEpoches:%08d\tcurrentBatchSize:%05d\tinitialBatchSize:%05d\tInput:%s\t%s\tTime:%s\tDataFile:%s"%(runs, 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                              OAA, TA, str(afc), valid_loss, valid_min_loss, final_train_loss, train_min_loss, TC, ILR,FLR, LS,ites,Epo,cBS,iBS,str(input),str(CM),time.strftime('%Y%m%d%H%M%S',T),df)
    return file_record    
def logfileV2(file_record, runs, V_string, final_train_loss, train_min_loss, TC, ILR, FLR, LS, ites, Epo, cBS, iBS, input, CMstring, T, df):
    file_record="Run%02d\t%s\tFinaltrainloss:%.10f\tMinimumtrainloss:%.10f\tTimeComsumed:%08.6f\tInitialLearningRate:%.8f\tFinalLearningRate:%.8f\tLearningStepForDroppingMagnitude:%08d\tTotalIterations:%08d\tEpoches:%08d\tcurrentBatchSize:%05d\tinitialBatchSize:%05d\tInput:%s\t%s\tTime:%s\tDataFile:%s"%(runs, 
                                          V_string, final_train_loss, train_min_loss, TC, ILR,FLR, LS,ites,Epo,cBS,iBS,str(input),str(CMstring),time.strftime('%Y%m%d%H%M%S',T),df)
    return file_record    
def logfileForSklearnModel(file_record, runs, model, TA, OAA, CM, df, train_ac, toaa, tcm):
    modelstring=''
    for v in str(model).splitlines():
        modelstring=modelstring+v
    file_record='Run%02d\tOverAllACC:%.8f\tTestAccuracy:%.8f\tTrainOAA:%.8f\tTrainAC:%.8f\tinput:%s\tCM:%s\tTCM:%s\t%s\t%s'%(runs, OAA, TA, toaa, train_ac, (sys.argv), str(CM), str(tcm), df, modelstring)
    return file_record
def load(data_path, session, ignore_missing=False):
    '''Load network weights.
    data_path: The path to the numpy-serialized network weights
    session: The current TensorFlow session
    ignore_missing: If true, serialized weights for missing layers are ignored.
    '''
    data_dict = np.load(data_path).item()
    for op_name in data_dict:
        with tf.variable_scope(op_name, reuse=True):
            for param_name, data in data_dict[op_name].items():
                try:
                    var = tf.get_variable(param_name)
                    session.run(var.assign(data))
                except ValueError:
                    if not ignore_missing:
                        raise
def restorefacepatchModel(TrainID, sess, NetworkType, graph):
    vl=graph.get_collection(name='trainable_variables')
    saver1=None
    saver2=None
    saver3=None
    if NetworkType==4:
        for v in vl:
            if M3N4S1.get(v.name, -1)==0:
                #print(M3N4S1[v.name])
                M3N4S1[v.name]=v
                #print(M3N4S1[v.name])
                #exit(9)
            elif M3N4S2.get(v.name, -1)==0:
                M3N4S2[v.name]=v
            elif M3N4S3.get(v.name, -1)==0:
                M3N4S3[v.name]=v

        saver1=tf.train.Saver(M3N4S1)
        saver2=tf.train.Saver(M3N4S2)
        saver3=tf.train.Saver(M3N4S3)

        if TrainID%100>30:
            saver1.restore(sess, './FPPTM/EyePatch_TrainonD502_TestonD531_N4_R4_20171025123948_1.59218006134_.ckpt')#OverAllACC:0.56836735	TestAccuracy:0.56836735	FinalLoss:1.5921800613
            saver2.restore(sess, './FPPTM/MiddlePatch_TrainonD502_TestonD531_N4_R4_20171025113147_1.68774459362_.ckpt')#OverAllACC:0.46938776	TestAccuracy:0.46938776	FinalLoss:1.6877445936
            saver3.restore(sess, './FPPTM/MouthPatch_TrainonD502_TestonD531_N4_R8_20171025144404_1.57691563368_.ckpt')#OverAllACC:0.58367347	TestAccuracy:0.58367347	FinalLoss:1.5769156337
        elif TrainID%100<20:
            saver1.restore(sess, './FPPTM/EyePatch_TrainonD532_TestonD501_N4_R9_20171019103910_1.51784744629_only_trainable_variables.ckpt')#OverAllACC:0.57857271	TestAccuracy:0.65412330	FinalLoss:1.5178474463
            saver2.restore(sess, './FPPTM/MiddlePatch_TrainonD532_TestonD501_N4_R11_20171019080535_1.66863813767_only_trainable_variables.ckpt')#OverAllACC:0.44420250	TestAccuracy:0.49079263	FinalLoss:1.6686381377
            saver3.restore(sess, './FPPTM/MouthPatch_TrainonD532_TestonD501_N4_R1_20171018224312_1.42820624205_only_trainable_variables.ckpt')#OverAllACC:0.68346248	TestAccuracy:0.74299440	FinalLoss:1.4282062420
        else:
            print('Unexpected case occurred when loading pretrain model in restorefacepatchModel')
            exit(-1)

    elif NetworkType==5:#for discrimination, N3 under tflearn was replaced as N5
        for v in vl:
            if M3N5S1.get(v.name, -1)==0:
                M3N5S1[v.name]=v
            elif M3N5S2.get(v.name, -1)==0:
                M3N5S2[v.name]=v
            elif M3N5S3.get(v.name, -1)==0:
                M3N5S3[v.name]=v

        saver1=tf.train.Saver(M3N5S1)
        saver2=tf.train.Saver(M3N5S2)
        saver3=tf.train.Saver(M3N5S3)

        if TrainID%100>30:
            saver1.restore(sess, './FPPTM/EyePatch_TrainonD502_TestonD531_N3_R10_20171102144530_1.5524974227_.ckpt')#Run10	OverAllACC:0.61836735	TestAccuracy:0.61836735	FinalLoss:1.5524974227
            saver2.restore(sess, './FPPTM/MiddlePatch_TrainonD502_TestonD531_N3_R7_20171102190719_1.69338421822_.ckpt')#Run07	OverAllACC:0.46428571	TestAccuracy:0.46428571	FinalLoss:1.6933842182
            saver3.restore(sess, './FPPTM/MouthPatch_TrainonD502_TestonD531_N3_R14_20171103033147_1.55810719728_.ckpt')#Run14	OverAllACC:0.60612245	TestAccuracy:0.60612245	FinalLoss:1.5581071973
        elif TrainID%100<20:
            saver1.restore(sess, './FPPTM/EyePatch_TrainonD532_TestonD501_N3_R0_20171102203504_1.5470389036_.ckpt')#Run00	OverAllACC:0.58779029	TestAccuracy:0.61569255	FinalLoss:1.5470389036
            saver2.restore(sess, './FPPTM/MiddlePatch_TrainonD532_TestonD501_N3_R14_20171102201934_1.65476641288_.ckpt')#Run14	OverAllACC:0.46619803	TestAccuracy:0.51401121	FinalLoss:1.6547664129	MinimunLoss:1.6547664129
            saver3.restore(sess, './FPPTM/MouthPatch_TrainonD532_TestonD501_N3_R9_20171102141218_1.41499766937_.ckpt')#Run09	OverAllACC:0.69564812	TestAccuracy:0.76220977	FinalLoss:1.4149976694
        else:
            print('Unexpected case occurred when loading pretrain model in restorefacepatchModel')
            exit(-1)
    else:
        exit(3)
def restorevggModel(sess, NetworkType, graph):
    vl=graph.get_collection(name='trainable_variables')
    if NetworkType==10 or NetworkType==11 or NetworkType==12:
        data_dict=np.load('./networkmodel/VGGFACE.npy').item()
        #print(type(data_dict))
        #print(len(data_dict))
        ##print(data_dict)
        #for name in data_dict:
        #    print(name)
        for v in vl:
            #print(v.name)
            namescope=v.name.split('/')[0]
            var=v.name.split('/')[1]
            val=data_dict.get(namescope, None)
            #print(v.name, namescope, var, var.find('W:0'), var.find('b:0'), type(val))
            if val==None:
                continue
            elif var.find('W:0')>-1:
                shape=val['weights'].shape
                #print(shape)
                if shape[2]==3:
                    val['weights']=np.reshape(val['weights'][:,:,1,:],[shape[0], shape[1], 1, shape[3]])
                sess.run(v.assign(val['weights']))
                print('Variable %s restored'%(v.name))
            elif var.find('b:0')>-1:
                #shape=val['biases'].shape
                sess.run(v.assign(val['biases']))
                print('Variable %s restored'%(v.name))
            else:
                continue
    else:
        exit(3) 
def loadPretrainedModel(NetworkType, network, session, module):
    #if NetworkType==4 or NetworkType==0 or NetworkType==1 or NetworkType==2 or NetworkType==3:
    if module==1:
        if NetworkType==4 or NetworkType<10:
            try:
                print("Loading pretrained network model: VGGFACE.npy......")
                network.load('./networkmodel/VGGFACE.npy', session, ignore_missing=True)
                print('\nPreserved Model of VGGFACE was loaded.\n')
            except:
                print('ERROR: unable to load pretrain network weights')
                traceback.print_exc()
                exit(-1)
        else:
            print('No pretrain network weights are fit to the current network type. Please try another network type.')
            exit()
    elif module==4:
        if NetworkType==4 or NetworkType<9:
            try:
                print("Loading pretrained network model: VGGFACE.npy......")
                network.load('./networkmodel/VGGFACE.npy', session, ignore_missing=True)
                print('\nPreserved Model of VGGFACE was loaded.\n')
            except:
                print('ERROR: unable to load pretrained VGGFACE network weights')
                traceback.print_exc()
                exit(-1)
        elif NetworkType==30:
            try:
                print("Loading pretrained network model: ResNet50.npy......")
                network.load('./networkmodel/ResNet50.npy', session, ignore_missing=True)
                print('\nPreserved Model of ResNet50 was loaded.\n')
            except:
                print('ERROR: unable to load pretrained ResNet50 network weights')
                traceback.print_exc()
                exit(-1)
        elif NetworkType==33:
            try:
                print("Loading pretrained network model: AlexNetoxford102.npy......")
                network.load('./networkmodel/AlexNetoxford102.npy', session, ignore_missing=True)
                print('\nPreserved Model of AlexNetoxford102 was loaded.\n')
            except:
                print('ERROR: unable to load pretrained AlexNetoxford102 network weights')
                traceback.print_exc()
                exit(-1)
        else:
            print('No pretrain network weights are fit to the current network type. Please try another network type.')
            exit()
    else:
        print('Module %d has no pretrained model embedded. Please try another module or check the input again.'%(module))
        exit()
Datasets = collections.namedtuple('Datasets', ['train', 'test', 'validation'])
def groupdata(Apredata, ValidID, TestID):
    '''This function will delete the contents in Apredata. 
    Please be careful when you use it.'''
    nl=len(Apredata)
    train={'X':[], 'Y':[]}
    test={'X':[], 'Y':[]}
    valid={'X':[], 'Y':[]}
    for i in range(nl):
        if i==int(TestID):
            test['X'].extend(Apredata[i]['X'])
            del Apredata[i]['X']
            test['Y'].extend(Apredata[i]['Y'])
            del Apredata[i]['Y']
            if ValidID==TestID:
                valid=test
        elif i==int(ValidID):
            valid['X'].extend(Apredata[i]['X'])
            del Apredata[i]['X']
            valid['Y'].extend(Apredata[i]['Y'])
            del Apredata[i]['Y']
        else:
            train['X'].extend(Apredata[i]['X'])
            del Apredata[i]['X']
            train['Y'].extend(Apredata[i]['Y'])
            del Apredata[i]['Y']
    return Datasets(train=train, test=test, validation=valid)
def multiprocessingUnitForModule8tests(metrics, sst, model_save_path, runs, t1, test_run, 
                                       NetworkType, data,facepatchpreprocessdatafilename, log,
                                       n_estimators, min_samples_split, min_samples_leaf):
    ct=time.time()
    m8_model_save_path=model_save_path.replace('_R'+str(runs)+time.strftime('_%Y%m%d%H%M%S',time.localtime(t1)),
                                                '_R'+str(test_run)+time.strftime('_%Y%m%d%H%M%S',time.localtime(ct)))
                
    logpostfix='_E%d_MSS%d_MSL%d_'%(n_estimators, min_samples_split, min_samples_leaf)

    if NetworkType%10==0:
        from sklearn import tree
        optm = tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=min_samples_split, 
                                            min_samples_leaf=min_samples_leaf)
    elif NetworkType%10==1:
        from sklearn import tree
        optm = tree.DecisionTreeClassifier(criterion='gini', min_samples_split=min_samples_split, 
                                            min_samples_leaf=min_samples_leaf)
    elif NetworkType%10==2:
        from sklearn.ensemble import RandomForestClassifier
        optm = RandomForestClassifier(n_estimators=n_estimators, criterion='entropy', 
                                        min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
    elif NetworkType%10==3:
        from sklearn.ensemble import RandomForestClassifier
        optm = RandomForestClassifier(n_estimators=n_estimators, criterion='gini', 
                                        min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
    else:
        print('ERROR:::::$$$$: Unexpected networktype encount.')
        exit(-1)
    m8_model_save_path=m8_model_save_path.replace('.ckpt', '_%s.ckpt'%(type(optm).__name__))
    optm.fit(data.train['X'], data.train['Y'])

    tY=optm.predict(data.train['X'])
    train_acc=metrics.accuracy_score(np.asarray(data.train['Y']), tY)
    tcm=calR(tY, data.train['Y'],cn)
    toaa=overAllAccuracy(tcm)

    pY=optm.predict(data.test['X'])
    #print(pY.shape)
    #print((np.asarray(data.test['Y'])).shape)
    accuracy=metrics.accuracy_score(np.asarray(data.test['Y']), pY)
    cm=calR(pY, data.test['Y'],cn)
    oaa=overAllAccuracy(cm)
    tt=time.time()
    print('OT:%2d\tOAA:%.8f\tAcc:%.8f\tTOAA:%.8f\tTAc:%.8f\t%s\tT:%fs'%(test_run, oaa, accuracy, toaa, train_acc, str(type(optm).__name__),(tt-ct)))
    sst.addFigure(oaa)
    file_record=logfileForSklearnModel(file_record,test_run, optm, accuracy, oaa, cm, facepatchpreprocessdatafilename, train_acc, toaa, tcm)
    #loss_a.setMinimun_loss(oaa)
    modelname=m8_model_save_path.replace('.ckpt','_%s_.pkl'%(str(oaa)))
    with open(modelname, 'wb') as fin:
        pickle.dump(optm, fin, 4)

    tt=time.time()
    logf=log.replace('.txt',('_'+str(type(optm).__name__)+logpostfix+'.txt'))
    filelog=open(logf,'a')
    filelog.write('%s\t\t TotalTimeConsumed: %f\tOptimizer: %s\n'%(file_record, (tt-ct), str(type(optm).__name__)))
    filelog.close()

    return oaa

def savelistcontent(filename, list):
    fw=open(filename, 'w')
    for v in list:
        fw.write('%s\n'%(str(v)))
    fw.close()
def run(GPU_Device_ID, Module, 
        DataSet,ValidID,TestID, 
        NetworkType, runs
        ,cLR=0.0001,batchSize=15,loadONW=False,reshape=False):
    try:
        initialize_dirs()
        '''GPU Option---------------------------------------------------------------------------------------------
        Determine which GPU is going to be used
        ------------------------------------------------------------------------------------------------------------'''
        print('GPU Option: %s'%(GPU_Device_ID))
        if (0==GPU_Device_ID) or (1==GPU_Device_ID):
            os.environ["CUDA_VISIBLE_DEVICES"]=str(GPU_Device_ID)
            errorlog='./logs/errors_gpu'+str(GPU_Device_ID)+'.txt'
            templog='./logs/templogs_newSC_gpu'+str(GPU_Device_ID)+'_M'+str(Module)+'_D'+str(DataSet)+'.txt'
        else:
            print("Usage: python finetune.py <GPUID> <Module> <NetworkType>\nGPUID must be 0 or 1\nModule must be 1, 2, or 3\nNetworkType must be 0, 1, 2, 3")
            exit(-1)
        '''GPU Option ENDS---------------------------------------------------------------------------------------'''
        cn=7#category numbers
        if int(DataSet)>60000:
            cn=6
            if int(DataSet==66505):
                cn=7
        mini_loss=10000
        loss_a=LOSS_ANA()
        file_record=None
        t1=time.time()
        logprefix='./logs/'
        model_save_path=''

        labelshape=[None, cn]
        m1shape= [None, 128, 128, 1]
        global Mini_Epochs
        #
        #
        #
        '''Input Data-------------------------------------------------------------------------------------------------
        -------------------------------------------------------------------------------------------------------------'''
        #
        ##data set loading
        #
        D_f=False
        if Module==2 and NetworkType<3:
            D_f=True
        dfile=Dataset_Dictionary.get(DataSet, False)
        if dfile==False:
            print('\nERROR: Unexpected DatasetID %d encouted.\n\n'%(int(DataSet)))
            exit(-1)
        logprefix="./logs/D%d_gpu"%(DataSet)
        if Module==7:
            print('Module 7: Face patches and Geometry')
        elif Module==8:
            print('Module 8: Face pathces cnn outputs')
        else:
            if Module==2 and NetworkType>9:
                data = DataSetPrepare.loadCKplus10gdata_v4(dfile, ValidID, TestID, Module=Module, Df=False,reshape=False, one_hot=False, cn=cn)
            else:
                #data = DataSetPrepare.loadCKplus10gdata_v2(dfile, ValidID, TestID, Df=D_f,reshape=reshape, cn=cn)
                data = DataSetPrepare.loadCKplus10gdata_v4(dfile, ValidID, TestID, Module=Module, Df=D_f, reshape=reshape, cn=cn)
            
            if DataSet==2:
                print("Processing 8 groups>>>>>>>>\n%s"%(logprefix))
            elif DataSet==3:
                print("Processing 8 groups>>>>>>>>\n%s"%(logprefix))
            elif DataSet==4:
                print("Processing 8 groups>>>>>>>>\n%s"%(logprefix))
            elif DataSet==5:
                print("Processing 8 groups>>>>>>>>\n%s"%(logprefix))
            elif DataSet==6:
                m2d=258
                print("Processing 8 groups>>>>>>>>\n%s"%(logprefix))
            elif DataSet==7:
                print("Processing 8 groups>>>>>>>>\n%s"%(logprefix))
            elif DataSet==8:
                print("Processing 8 groups>>>>>>>>\n%s"%(logprefix))
            elif DataSet==9:
                m1shape= [None, 224, 224, 1]
                print("Processing 8 groups>>>>>>>>\n%s"%(logprefix))
            elif DataSet==10:
                print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
            elif DataSet==11:
                m1shape= [None, 224, 224, 1]
                print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
            elif DataSet==12:
                print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
            elif DataSet==13:
                m2d=258
                print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
            elif DataSet==15:
                dfilet=Dataset_Dictionary.get(10)
                #datatest = DataSetPrepare.loadCKplus10gdata_v2(dfilet, ValidID, TestID, Df=D_f,reshape=reshape, cn=cn)
                datatest = DataSetPrepare.loadCKplus10gdata_v4(dfilet, ValidID, TestID, Module=Module, Df=D_f,reshape=reshape, cn=cn)
                print('Before reset: %d'%data.test.num_examples)
                data.test.reset(datatest.test.res_images, datatest.test.geometry, 
                                datatest.test.eyep, datatest.test.middlep, datatest.test.mouthp, datatest.test.innerf,
                                datatest.test.labels)
                data.validation.reset(datatest.validation.res_images, datatest.validation.geometry, 
                                datatest.validation.eyep, datatest.validation.middlep, datatest.validation.mouthp, datatest.validation.innerf,
                                datatest.validation.labels)
                print('After reset: %d'%data.test.num_examples)
                del datatest
                batchSize=60
            
                print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
            elif DataSet==16:
                dfilet=Dataset_Dictionary.get(10)
                #datatest = DataSetPrepare.loadCKplus10gdata_v2(dfilet, ValidID, TestID, Df=D_f,reshape=reshape, cn=cn)
                datatest = DataSetPrepare.loadCKplus10gdata_v4(dfilet, ValidID, TestID, Module=Module, Df=D_f,reshape=reshape, cn=cn)
                print('Before reset: %d'%data.test.num_examples)
                data.test.reset(datatest.test.res_images, datatest.test.geometry, 
                                datatest.test.eyep, datatest.test.middlep, datatest.test.mouthp, datatest.test.innerf,
                                datatest.test.labels)
                data.validation.reset(datatest.validation.res_images, datatest.validation.geometry, 
                                datatest.validation.eyep, datatest.validation.middlep, datatest.validation.mouthp, datatest.validation.innerf,
                                datatest.validation.labels)
                print('After reset: %d'%data.test.num_examples)
                del datatest
                if runs%2==0:
                    batchSize=30
                else:
                    batchSize=15
            
                print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
            elif DataSet==17:
                print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
            elif DataSet==18:
                print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
            elif DataSet==19:
                print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
            elif DataSet==33:
                batchSize=35
                print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
            elif DataSet==32:
                dfilet=Dataset_Dictionary.get(33)
                #datatest = DataSetPrepare.loadCKplus10gdata_v2(dfilet, ValidID, TestID, Df=D_f,reshape=reshape, cn=cn)
                datatest = DataSetPrepare.loadCKplus10gdata_v4(dfilet, ValidID, TestID, Module=Module, Df=D_f,reshape=reshape, cn=cn)
                print('Before reset: %d'%data.test.num_examples)
                data.test.reset(datatest.test.res_images, datatest.test.geometry, 
                                datatest.test.eyep, datatest.test.middlep, datatest.test.mouthp, datatest.test.innerf,
                                datatest.test.labels)
                data.validation.reset(datatest.validation.res_images, datatest.validation.geometry, 
                                datatest.validation.eyep, datatest.validation.middlep, datatest.validation.mouthp, datatest.validation.innerf,
                                datatest.validation.labels)
                print('After reset: %d'%data.test.num_examples)
                del datatest
                batchSize=70
                print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
            elif DataSet==34:
                dfilet=Dataset_Dictionary.get(33)
                #datatest = DataSetPrepare.loadCKplus10gdata_v2(dfilet, ValidID, TestID, Df=D_f,reshape=reshape, cn=cn)
                datatest = DataSetPrepare.loadCKplus10gdata_v4(dfilet, ValidID, TestID, Module=Module, Df=D_f,reshape=reshape, cn=cn)
                print('Before reset: %d'%data.test.num_examples)
                data.test.reset(datatest.test.res_images, datatest.test.geometry, 
                                datatest.test.eyep, datatest.test.middlep, datatest.test.mouthp, datatest.test.innerf,
                                datatest.test.labels)
                data.validation.reset(datatest.validation.res_images, datatest.validation.geometry, 
                                datatest.validation.eyep, datatest.validation.middlep, datatest.validation.mouthp, datatest.validation.innerf,
                                datatest.validation.labels)
                print('After reset: %d'%data.test.num_examples)
                del datatest
                batchSize=70
                print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
            elif DataSet==42:
                dfilet=Dataset_Dictionary.get(40)
                #datatest = DataSetPrepare.loadCKplus10gdata_v2(dfilet, ValidID, TestID, Df=D_f,reshape=reshape, cn=cn)
                datatest = DataSetPrepare.loadCKplus10gdata_v4(dfilet, ValidID, TestID, Module=Module, Df=D_f,reshape=reshape, cn=cn)
                print('Before reset: %d'%data.test.num_examples)
                data.test.reset(datatest.test.res_images, datatest.test.geometry, 
                                datatest.test.eyep, datatest.test.middlep, datatest.test.mouthp, datatest.test.innerf,
                                datatest.test.labels)
                data.validation.reset(datatest.validation.res_images, datatest.validation.geometry, 
                                datatest.validation.eyep, datatest.validation.middlep, datatest.validation.mouthp, datatest.validation.innerf,
                                datatest.validation.labels)
                print('After reset: %d'%data.test.num_examples)
                del datatest
                batchSize=60
                print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
            elif DataSet==40:
                print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
            elif DataSet==43:
                dfilet=Dataset_Dictionary.get(40)
                #datatest = DataSetPrepare.loadCKplus10gdata_v2(dfilet, ValidID, TestID, Df=D_f,reshape=reshape, cn=cn)
                datatest = DataSetPrepare.loadCKplus10gdata_v4(dfilet, ValidID, TestID, Module=Module, Df=D_f,reshape=reshape, cn=cn)
                print('Before reset: %d'%data.test.num_examples)
                data.test.reset(datatest.test.res_images, datatest.test.geometry, 
                                datatest.test.eyep, datatest.test.middlep, datatest.test.mouthp, datatest.test.innerf,
                                datatest.test.labels)
                data.validation.reset(datatest.validation.res_images, datatest.validation.geometry, 
                                datatest.validation.eyep, datatest.validation.middlep, datatest.validation.mouthp, datatest.validation.innerf,
                                datatest.validation.labels)
                print('After reset: %d'%data.test.num_examples)
                del datatest
                batchSize=60
                print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
            elif DataSet==111:
                batchSize=30
                print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
            elif DataSet==222:
                dfilet=Dataset_Dictionary.get(111)
                #datatest = DataSetPrepare.loadCKplus10gdata_v2(dfilet, ValidID, TestID, Df=D_f,reshape=reshape, cn=cn)
                datatest = DataSetPrepare.loadCKplus10gdata_v4(dfilet, ValidID, TestID, Module=Module, Df=D_f,reshape=reshape, cn=cn)
                print('Before reset: %d'%data.test.num_examples)
                data.test.reset(datatest.test.res_images, datatest.test.geometry, 
                                datatest.test.eyep, datatest.test.middlep, datatest.test.mouthp, datatest.test.innerf,
                                datatest.test.labels)
                data.validation.reset(datatest.validation.res_images, datatest.validation.geometry, 
                                datatest.validation.eyep, datatest.validation.middlep, datatest.validation.mouthp, datatest.validation.innerf,
                                datatest.validation.labels)
                print('After reset: %d'%data.test.num_examples)
                del datatest
                batchSize=30
                print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
            elif DataSet==333:
                dfilet=Dataset_Dictionary.get(444)
                #datatest = DataSetPrepare.loadCKplus10gdata_v2(dfilet, ValidID, TestID, Df=D_f,reshape=reshape, cn=cn)
                datatest = DataSetPrepare.loadCKplus10gdata_v4(dfilet, ValidID, TestID, Module=Module, Df=D_f,reshape=reshape, cn=cn)
                print('Before reset: %d'%data.test.num_examples)
                data.test.reset(datatest.test.res_images, datatest.test.geometry, 
                                datatest.test.eyep, datatest.test.middlep, datatest.test.mouthp, datatest.test.innerf,
                                datatest.test.labels)
                data.validation.reset(datatest.validation.res_images, datatest.validation.geometry, 
                                datatest.validation.eyep, datatest.validation.middlep, datatest.validation.mouthp, datatest.validation.innerf,
                                datatest.validation.labels)
                print('After reset: %d'%data.test.num_examples)
                del datatest
                batchSize=30
                print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
            elif DataSet==444:
                batchSize=30
                print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
            elif DataSet==501:
                if runs%2==0:
                    batchSize=30
                else:
                    batchSize=15
                print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
            elif DataSet==502:
                dfilet=Dataset_Dictionary.get(501)
                #datatest = DataSetPrepare.loadCKplus10gdata_v2(dfilet, ValidID, TestID, Df=D_f,reshape=reshape, cn=cn)
                datatest = DataSetPrepare.loadCKplus10gdata_v4(dfilet, ValidID, TestID,Module=Module, Df=D_f,reshape=reshape, cn=cn)
                print('Before reset: %d'%data.test.num_examples)
                data.test.reset(datatest.test.res_images, datatest.test.geometry, 
                                datatest.test.eyep, datatest.test.middlep, datatest.test.mouthp, datatest.test.innerf,
                                datatest.test.labels)
                data.validation.reset(datatest.validation.res_images, datatest.validation.geometry, 
                                datatest.validation.eyep, datatest.validation.middlep, datatest.validation.mouthp, datatest.validation.innerf,
                                datatest.validation.labels)
                print('After reset: %d'%data.test.num_examples)
                del datatest
                if runs%2==0:
                    batchSize=30
                else:
                    batchSize=15
                print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
            elif DataSet==503:
                if runs%2==0:
                    batchSize=30
                else:
                    batchSize=15
                print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
            elif DataSet==531:
                if runs%2==0:
                    batchSize=15
                else:
                    batchSize=30
                print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
            elif DataSet==532:
                dfilet=Dataset_Dictionary.get(531)
                #datatest = DataSetPrepare.loadCKplus10gdata_v2(dfilet, ValidID, TestID, Df=D_f,reshape=reshape, cn=cn)
                datatest = DataSetPrepare.loadCKplus10gdata_v4(dfilet, ValidID, TestID, Module=Module, Df=D_f,reshape=reshape, cn=cn)
                print('Before reset: %d'%data.test.num_examples)
                data.test.reset(datatest.test.res_images, datatest.test.geometry, 
                                datatest.test.eyep, datatest.test.middlep, datatest.test.mouthp, datatest.test.innerf,
                                datatest.test.labels)
                data.validation.reset(datatest.validation.res_images, datatest.validation.geometry, 
                                datatest.validation.eyep, datatest.validation.middlep, datatest.validation.mouthp, datatest.validation.innerf,
                                datatest.validation.labels)
                print('After reset: %d'%data.test.num_examples)
                del datatest
                if runs%2==0:
                    batchSize=15
                else:
                    batchSize=30
                print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
            elif DataSet==551:
                if runs%2==0:
                    batchSize=21
                else:
                    batchSize=42
                print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
            elif DataSet==552:
                if runs%2==0:
                    batchSize=21
                else:
                    batchSize=42
                print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
            elif DataSet==553:
                if runs%2==0:
                    batchSize=21
                else:
                    batchSize=42
                print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
            elif DataSet==554:
                if runs%2==0:
                    batchSize=21
                else:
                    batchSize=42
                print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
            elif DataSet==610:
                if runs%3==0:
                    batchSize=35
                elif runs%3==1:
                    batchSize=70
                else:
                    batchSize=128
                    Mini_Epochs=Mini_Epochs*2
                cLR=0.00001
                print("Processing dataset>>>>>>>>\n%s"%(logprefix))
            elif DataSet==611:
                if runs%3==0:
                    batchSize=35
                elif runs%3==1:
                    batchSize=70
                else:
                    batchSize=128
                    Mini_Epochs=Mini_Epochs*2
                cLR=0.00001
                print("Processing dataset>>>>>>>>\n%s"%(logprefix))
            elif DataSet==620:
                if runs%3==0:
                    batchSize=35
                elif runs%3==1:
                    batchSize=70
                else:
                    batchSize=128
                    Mini_Epochs=Mini_Epochs*2
                cLR=0.00001
                print("Processing dataset>>>>>>>>\n%s"%(logprefix))
            elif DataSet==621:
                if runs%3==0:
                    batchSize=35
                elif runs%3==1:
                    batchSize=70
                else:
                    batchSize=128
                    Mini_Epochs=Mini_Epochs*2
                cLR=0.00001
                print("Processing dataset>>>>>>>>\n%s"%(logprefix))
            elif DataSet==1001:
                if runs%2==0:
                    batchSize=30
                else:
                    batchSize=15
                print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
            elif DataSet==1002:
                dfilet=Dataset_Dictionary.get(1001)
                datatest = DataSetPrepare.loadCKplus10gdata_v2(dfilet, ValidID, TestID, Df=D_f,reshape=reshape, cn=cn)
                print('Before reset: %d'%data.test.num_examples)
                data.test.reset(datatest.test.res_images, datatest.test.geometry, 
                                datatest.test.eyep, datatest.test.middlep, datatest.test.mouthp, datatest.test.innerf,
                                datatest.test.labels)
                data.validation.reset(datatest.validation.res_images, datatest.validation.geometry, 
                                datatest.validation.eyep, datatest.validation.middlep, datatest.validation.mouthp, datatest.validation.innerf,
                                datatest.validation.labels)
                print('After reset: %d'%data.test.num_examples)
                if runs%2==0:
                    batchSize=30
                else:
                    batchSize=30
                print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
            elif DataSet==66501:
                batchSize=30
                print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
            elif DataSet==66502:
                dfilet=Dataset_Dictionary.get(66501)
                #datatest = DataSetPrepare.loadCKplus10gdata_v2(dfilet, ValidID, TestID, Df=D_f,reshape=reshape, cn=cn)
                datatest = DataSetPrepare.loadCKplus10gdata_v4(dfilet, ValidID, TestID,Module=Module, Df=D_f,reshape=reshape, cn=cn)
                print('Before reset: %d'%data.test.num_examples)
                data.test.reset(datatest.test.res_images, datatest.test.geometry, 
                                datatest.test.eyep, datatest.test.middlep, datatest.test.mouthp, datatest.test.innerf,
                                datatest.test.labels)
                data.validation.reset(datatest.validation.res_images, datatest.validation.geometry, 
                                datatest.validation.eyep, datatest.validation.middlep, datatest.validation.mouthp, datatest.validation.innerf,
                                datatest.validation.labels)
                print('After reset: %d'%data.test.num_examples)
                del datatest
                batchSize=30
                print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
            elif DataSet==66503:
                cLR=0.001
                batchSize=30
                print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
            elif DataSet==66504:
                #cLR=0.001
                batchSize=30
                print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
            elif DataSet==66505:
                #cLR=0.001
                batchSize=30
                print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
            elif DataSet==66531:
                batchSize=30
                print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
            elif DataSet==66532:
                dfilet=Dataset_Dictionary.get(66531)
                #datatest = DataSetPrepare.loadCKplus10gdata_v2(dfilet, ValidID, TestID, Df=D_f,reshape=reshape, cn=cn)
                datatest = DataSetPrepare.loadCKplus10gdata_v4(dfilet, ValidID, TestID, Module=Module, Df=D_f,reshape=reshape, cn=cn)
                print('Before reset: %d'%data.test.num_examples)
                data.test.reset(datatest.test.res_images, datatest.test.geometry, 
                                datatest.test.eyep, datatest.test.middlep, datatest.test.mouthp, datatest.test.innerf,
                                datatest.test.labels)
                data.validation.reset(datatest.validation.res_images, datatest.validation.geometry, 
                                datatest.validation.eyep, datatest.validation.middlep, datatest.validation.mouthp, datatest.validation.innerf,
                                datatest.validation.labels)
                print('After reset: %d'%data.test.num_examples)
                del datatest
                batchSize=30
                print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
            elif DataSet==66554:
                batchSize=30
                print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
            elif DataSet==66555:
                batchSize=30
                print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
            elif DataSet==66610:
                if runs%2==0:
                    batchSize=30
                elif runs%2==1:
                    batchSize=60
                cLR=0.00001
                print("Processing dataset>>>>>>>>\n%s"%(logprefix))
            elif DataSet==66611:
                if runs%2==0:
                    batchSize=30
                elif runs%2==1:
                    batchSize=60
                cLR=0.00001
                print("Processing dataset>>>>>>>>\n%s"%(logprefix))
            elif DataSet==66620:
                if runs%2==0:
                    batchSize=30
                elif runs%2==1:
                    batchSize=60
                cLR=0.00001
                print("Processing dataset>>>>>>>>\n%s"%(logprefix))
            elif DataSet==66621:
                if runs%2==0:
                    batchSize=30
                elif runs%2==1:
                    batchSize=60
                cLR=0.00001
                print("Processing dataset>>>>>>>>\n%s"%(logprefix))
            else:
                print('ERROR: Unexpeted Dataset ID')
                exit()
        #
        lrstep=int(data.train.num_examples/batchSize*times)
        print('\nlearning rate decay steps: %d'%lrstep)
        #
        tt=time.time()
        if reshape:
            logprefix=logprefix+'_reshape64x64'
        if Module==6:
            log=logprefix+str(GPU_Device_ID)+"_M"+str(Module)+"_D"+str(DataSet)+"_N"+str(NetworkType)+"_newStopCriteriaV3.txt"
        elif loadONW:
            log=logprefix+str(GPU_Device_ID)+"_M"+str(Module)+"_D"+str(DataSet)+"_N"+str(NetworkType)+"_withPretrainModelWeight_newStopCriteriaV3.txt"
        else:
            log=logprefix+str(GPU_Device_ID)+"_M"+str(Module)+"_D"+str(DataSet)+"_N"+str(NetworkType)+"_noPretrain_newStopCriteriaV3.txt"
            #logfilename=time.strftime('%Y%m%d%H%M%S',time.localtime(tt))+str(sys.argv[2:4])
        print('Time used for loading data: %fs'%(tt-t1))

        if os.path.exists("J:/Models/saves/"):
            model_save_path=("J:/Models/saves/"+'M'+str(Module)+'/D'+str(DataSet)+'/N'+str(NetworkType)+'/')
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
            model_save_path=(model_save_path+'D'+str(DataSet)+'_M'+str(Module)+'_N'+str(NetworkType)+'_T'+str(TestID)+'_V'+str(ValidID)+'_R'
                                +str(runs)+time.strftime('_%Y%m%d%H%M%S',time.localtime(t1))+".ckpt")
        else:
            model_save_path=("./saves/"+'M'+str(Module)+'/D'+str(DataSet)+'/N'+str(NetworkType)+'/')
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
            model_save_path=(model_save_path+'D'+str(DataSet)+'_M'+str(Module)+'_N'+str(NetworkType)+'_T'+str(TestID)+'_V'+str(ValidID)+'_R'
                                +str(runs)+time.strftime('_%Y%m%d%H%M%S',time.localtime(t1))+".ckpt")

        '''Input Data Ends-----------------------------------------------------------------------------------------'''
        #
        #
        #
        if reshape:
            m1shape=[None, 64, 64, 1]
            print('Module 1 images input shape has been set to %s'%str(m1shape))
            model_save_path=model_save_path.replace(".ckpt", "_reshape.ckpt")
        #
        #
        #
        if Module==1 and NetworkType==10 or NetworkType==4:
            cLR=0.00002
            if loadONW==False:
                lrstep=14000
        global_step = tf.Variable(0, trainable=False)
        lr=tf.train.exponential_decay(cLR, global_step, lrstep, lr_drate, staircase=True)

        if Module==1:
            stcmwvlilttv=1.19#save_the_current_model_when_validation_loss_is_less_than_this_value
            if DataSet==554 or DataSet==551 or DataSet==552 or DataSet==553:
                stcmwvlilttv=1.7
            '''MODULE1---------------------------------------------------------------------------------------------------- 
            Options for the whole-face-network
            Only need to select one of the import options as the network for the whole face feature extraction.
            -------------------------------------------------------------------------------------------------------------'''
            print('Network Type: %s'%(NetworkType))
            if NetworkType==0:
                from VGG_NET import VGG_NET_20l_512o as WFN
            elif NetworkType==1:
                from VGG_NET import VGG_NET_20l_128o as WFN
            elif NetworkType==2:
                from VGG_NET import VGG_NET_16l_128o as WFN
            elif NetworkType==3:
                from VGG_NET import VGG_NET_16l_72o as WFN
            elif NetworkType==4:
                from VGG_NET import VGG_NET_o as WFN
            elif NetworkType==8:
                from VGG_NET import VGG_NET_Inception1 as WFN
            elif NetworkType==9:
                from VGG_NET import VGG_NET_Inception2 as WFN
            elif NetworkType==10:
                from VGG_NET import VGG_NET_O_tfl as WFN
            elif NetworkType==11:
                from VGG_NET import VGG_NET_I5 as WFN
            elif NetworkType==12:
                from VGG_NET import VGG_NET_I5_ELU as WFN
            else:
                print("Usage: python finetune.py <GPUID> <Module> <NetworkType>\nWith Module 1, NetworkType must be 0, 1, 2, 3")
                exit(-1)
            '''Here begins the implementation logic-------------------------------------------------------------------
            -------------------------------------------------------------------------------------------------------------'''
            #Holder for gray images with m1shape in a batch size of batch_size
            images = tf.placeholder(tf.float32, m1shape)
            #Holder for labels in a batch size of batch_size, number of labels are to be determined
            labels = tf.placeholder(tf.float32, labelshape)#the number of labels are to be determined

            if NetworkType==10 or NetworkType==11 or NetworkType==12:
                Mini_Epochs = 40
                softmax=WFN(images)
            else:
                whole_face_net = WFN({'data':images})
                softmax=whole_face_net.layers['prob']

            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=softmax),0)
            #optm=tf.train.RMSPropOptimizer(lr)
            optm=tf.train.AdamOptimizer(lr)
            train_op=optm.minimize(loss,global_step=global_step)#for train
            
            #for test
            correcta_prediction = tf.equal(tf.argmax(softmax,1),tf.argmax(labels,1))
            test_cast=tf.cast(correcta_prediction, "float")
            sum_test=tf.reduce_sum(test_cast)#for large test set
            accuracy = tf.reduce_mean(test_cast)#for small test set

            with tf.Session() as sess:

                sess.run(tf.global_variables_initializer())

                if loadONW:
                    if NetworkType==10 or NetworkType==11 or NetworkType==12:
                        restorevggModel(sess, NetworkType, tf.get_default_graph())
                    else:
                        loadPretrainedModel(NetworkType, whole_face_net, sess,Module)
                    print('Model has been restored.\n')
                    #exit(-1)

                saver = tf.train.Saver()
                iters=int((data.train.num_examples*Mini_Epochs)/batchSize)+1
                for i in range(iters):
                    afc=[]
                    batch=data.train.next_batch(batchSize, shuffle=False)
                    tloss, _=sess.run([loss, train_op], feed_dict={images:batch[0], labels:batch[5]})
                    if tloss<mini_loss:
                        mini_loss=tloss
                    v_accuracy, valid_loss, oaa, confu_mat = Valid_on_TestSet(cn, sess, accuracy, sum_test, loss, softmax,
                                                                                  images, data.test.res_images, labels, data.test.labels,afc=afc)
                    laflag = loss_a.analyzeLossVariation(valid_loss)
                    clr=cLR*(lr_drate)**(i//lrstep)
                    tt=time.time()
                    print("CLR:%.8f Ite:%06d Bs:%03d Epo:%04d Los:%.8f mLo:%08f\tVALID>> mVL: %.8f\tVL: %.8f\tVA: %f\tOAA: %f\tT: %fs"%
                          (clr,i,batchSize,data.train.epochs_completed, tloss, mini_loss, loss_a.minimun_loss, valid_loss, v_accuracy, oaa, (tt-t1)))
                    if laflag:
                        file_record = logfile(file_record, runs=runs, OAA=oaa, afc=afc, valid_loss=valid_loss, valid_min_loss=loss_a.minimun_loss, 
                            final_train_loss=tloss, train_min_loss=mini_loss, TA=v_accuracy, TC=(tt-t1),ILR=cLR, FLR=clr, LS=lrstep, ites=i,
                            Epo=data.train.epochs_completed, cBS=batchSize, iBS=batchSize,
                            input=sys.argv, CM=confu_mat, T=time.localtime(tt), df=dfile)
                        if loss_a.minimun_loss < stcmwvlilttv:
                            saver.save(sess=sess, save_path=model_save_path)
                '''MODULE1 ENDS---------------------------------------------------------------------------------------------'''
        #
        #
        #
        elif Module==2:
            #stcmwvlilttv=1.1854#value need to be determined. save_the_current_model_when_validation_loss_is_less_than_this_value
            #'''MODULE2---------------------------------------------------------------------------------------------------- 
            #Options for the Geometry-network
            #Only need to select one of the import options as the network for the geometry feature extraction.
            #-------------------------------------------------------------------------------------------------------------'''
            #print('Geometry Network Type: %s'%(NetworkType))
            #if NetworkType==0:
            #    from Geometric_NET import Geometric_NET_2c2l as GeN
            #elif NetworkType==1:
            #    from Geometric_NET import Geometric_NET_2c2lcc1 as GeN
            #elif NetworkType==2:
            #    from Geometric_NET import Geometric_NET_2c2lcc1l1 as GeN
            #elif NetworkType==3:
            #    from Geometric_NET import Geometric_NET_1h as GeN
            #elif NetworkType==4:
            #    from Geometric_NET import Geometric_NET_2h1I as GeN
            #elif NetworkType==5:
            #    from Geometric_NET import Geometric_NET_3h1I as GeN
            #    clr=0.00001
            #    learningRate=0.00001
            #elif NetworkType==6:
            #    from Geometric_NET import Geometric_NET_h1I as GeN
            #else:
            #    print("Usage: python finetune.py <GPUID> <Module> <NetworkType>\nWith Module 2, NetworkType must be 0, 1, 2")
            #    exit(-1)
            #'''Here begins the implementation logic-------------------------------------------------------------------
            #-------------------------------------------------------------------------------------------------------------'''
            ##Holder for geometry features with 122 in a batch size of batch_size
            #if D_f:
            #    geo_features = tf.placeholder(tf.float32, [None, m2d, 1])
            #else:
            #    geo_features = tf.placeholder(tf.float32, [None, m2d])

            ##Holder for labels in a batch size of batch_size, number of labels are to be determined
            #labels = tf.placeholder(tf.float32, labelshape)#the number of labels are to be determined
            
            #Geometry_net = GeN({'data':geo_features})
            #print(type(Geometry_net))
            #softmax=tf.nn.softmax('prob')

            #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=softmax),0)
            #optm=tf.train.RMSPropOptimizer(lr)
            ##optm=tf.train.RMSPropOptimizer(lr)
            #train_op=optm.minimize(loss)#for train

            ##for test
            #correcta_prediction = tf.equal(tf.argmax(softmax,1),tf.argmax(labels,1))
            #accuracy = tf.reduce_mean(tf.cast(correcta_prediction, "float"))
        
            #with tf.Session() as sess:

            #    sess.run(tf.global_variables_initializer())
            #    saver = tf.train.Saver()
                
                #'''MODULE2 ENDS---------------------------------------------------------------------------------------------'''
            from sklearn import metrics
            '''MODULE2---------------------------------------------------------------------------------------------------- 
            Options for the Geometry features
            -------------------------------------------------------------------------------------------------------------'''
            print('Network Type: %s'%(NetworkType))
            '''Here begins the implementation logic-------------------------------------------------------------------
            -------------------------------------------------------------------------------------------------------------'''
            overtimes=1
            if continue_test:
                overtimes=OverTimes

            nel=[7, 10, 14, 18, 21, 25, 28, 32]
            mssl=[4, 8, 10, 14, 18, 21, 27, 32]
            msll=[1, 2, 3, 5, 8, 10, 14, 18, 24, 27]
            loopflag=False
            log=log.replace('./logs','./logs/M%dtests'%(Module))#use for tuning
            for v_nel in nel:
                if loopflag:
                    break
                if NetworkType==10 or NetworkType==11:
                    loopflag=True
                for v_mss in mssl:
                    for v_msl in msll:
                        n_estimators=v_nel#10, estimators for random forest classifier
                        min_samples_split=v_mss#10
                        min_samples_leaf=v_msl#5

                        #n_estimators=14#10, estimators for random forest classifier
                        #min_samples_split=10#10
                        #min_samples_leaf=5#5
                        print('n_estimators(RFC):%d\tmin_samples_split:%d\tmin_samples_leaf:%d'%(n_estimators, 
                                                                                                               min_samples_split, min_samples_leaf))
                        sst=SIMSTS(overtimes)
                        for test_run in range(overtimes):
                            ct=time.time()
                            m7_model_save_path=model_save_path.replace('_R'+str(runs)+time.strftime('_%Y%m%d%H%M%S',time.localtime(t1)),
                                                                       '_R'+str(test_run)+time.strftime('_%Y%m%d%H%M%S',time.localtime(ct)))
                
                            if NetworkType==10:
                                from sklearn import tree
                                optm = tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=min_samples_split, 
                                                                   min_samples_leaf=min_samples_leaf)
                                logpostfix=''
                            elif NetworkType==11:
                                from sklearn import tree
                                optm = tree.DecisionTreeClassifier(criterion='gini', min_samples_split=min_samples_split, 
                                                                   min_samples_leaf=min_samples_leaf)
                                logpostfix=''
                            elif NetworkType==12:
                                from sklearn.ensemble import RandomForestClassifier
                                optm = RandomForestClassifier(n_estimators=n_estimators, criterion='entropy', 
                                                              min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
                                logpostfix='_E%d'%(n_estimators)
                            elif NetworkType==13:
                                from sklearn.ensemble import RandomForestClassifier
                                optm = RandomForestClassifier(n_estimators=n_estimators, criterion='gini', 
                                                              min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
                                logpostfix='_E%d'%(n_estimators)
                            else:
                                print('ERROR:::::$$$$: Unexpected networktype encount.')
                                exit(-1)
                            logpostfix=logpostfix+'_MSS%d_MSL%d_'%(min_samples_split, min_samples_leaf)

                            m7_model_save_path=m7_model_save_path.replace('.ckpt', '_%s.ckpt'%(type(optm).__name__))
                            #print(type(data.train.geometry))
                            #print(len(data.train.geometry))
                            #print(type(data.train.labels))
                            #print(data.train.labels.shape)
                            optm.fit(data.train.geometry, data.train.labels)

                            tY=optm.predict(data.train.geometry)
                            train_acc=metrics.accuracy_score(data.train.labels, tY)
                            tcm=calR(tY, data.train.labels)
                            toaa=overAllAccuracy(tcm)

                            pY=optm.predict(data.test.geometry)
                            accuracy=metrics.accuracy_score(data.test.labels, pY)
                            cm=calR(pY, data.test.labels)
                            oaa=overAllAccuracy(cm)
                            tt=time.time()
                            print('OT:%2d\tOAA:%.8f\tAcc:%.8f\tTOAA:%.8f\tTAc:%.8f\t%s\tT:%fs'%(test_run, oaa, accuracy, toaa, train_acc, str(type(optm).__name__),(tt-ct)))
                            sst.addFigure(oaa)
                            file_record=logfileForSklearnModel(file_record,test_run, optm, accuracy, oaa, cm, dfile, train_acc, toaa, tcm)
                            loss_a.setMinimun_loss(oaa)
                            modelname=m7_model_save_path.replace('.ckpt','_%s_.pkl'%(str(oaa)))
                            with open(modelname, 'wb') as fin:
                                pickle.dump(optm, fin, 4)

                            tt=time.time()
                            logf=log.replace('.txt',('_'+str(type(optm).__name__)+logpostfix+'.txt'))
                            filelog=open(logf,'a')
                            filelog.write('%s\t\t TotalTimeConsumed: %f\tOptimizer: %s\n'%(file_record, (tt-ct), str(type(optm).__name__)))
                            filelog.close()
                        state=sst.getSTS()
                        print('Mean:%f\tMax:%f\tMin:%f'%(state[0], state[1], state[2]))
                        sst.logfile(Module, DataSet, NetworkType, n_estimators, min_samples_split, min_samples_leaf)
                        '''MODULE2 ENDS---------------------------------------------------------------------------------------------'''
        #
        #
        #
        elif Module==3:
            stcmwvlilttv=1.2154#value need to be determined. save_the_current_model_when_validation_loss_is_less_than_this_value
            if DataSet==502 or DataSet==501:
                stcmwvlilttv=1.1854
            elif DataSet==532 or DataSet==531:
                stcmwvlilttv=1.1904
            elif DataSet==554 or DataSet==551 or DataSet==552 or DataSet==553:
                stcmwvlilttv=1.7
            elif DataSet>60000:
                stcmwvlilttv=1.045
            '''MODULE3---------------------------------------------------------------------------------------------------- 
            Options for the face_patches-network
    
            -------------------------------------------------------------------------------------------------------------'''
            print('FacePatch Network Type: %s'%(NetworkType))
            if NetworkType==0:
                from FacePatches_NET import FacePatches_NET_2Inceptions as PaN
            elif NetworkType==1:
                from FacePatches_NET import FacePatches_NET_2Inceptions_4lrn as PaN
            elif NetworkType==2:
                from FacePatches_NET import FacePatches_NET_2Inceptions_4lrn2 as PaN
            elif NetworkType==3:
                from FacePatches_NET import FacePatches_NET_3Conv_2Inception as PaN
            elif NetworkType==4:
                #from FacePatches_NET import FacePatches_NET_3Conv_1Inception as PaN
                from FacePatches_NET import FacePatches_NET_3Conv_IInception_tflear as PaN
            elif NetworkType==5:
                from FacePatches_NET import FacePatches_NET_3Conv_3Inception_tflearn_5 as PaN
                stcmwvlilttv=0.00022
            elif NetworkType==6:
                from FacePatches_NET import FacePatches_NET_3Conv_3Inception_tflearn as PaN
            elif NetworkType==7:
                from FacePatches_NET import FacePatches_NET_3Conv_3Inception_tflearn_ELU as PaN
            elif NetworkType==8:
                from FacePatches_NET import FacePatches_NET_3Conv_3Inception_tflearn_8 as PaN
            elif NetworkType==9:
                from FacePatches_NET import FacePatches_NET_3Conv_3Inception_tflearn_9 as PaN
            elif NetworkType==10:
                from FacePatches_NET import FacePatches_NET_3Conv_3Inception_tflearn_10 as PaN
            elif NetworkType==11:
                from FacePatches_NET import FacePatches_NET_3Conv_3Inception_tflearn_11 as PaN
            elif NetworkType==12:
                from FacePatches_NET import FacePatches_NET_3Conv_3Inception_tflearn_12 as PaN
            elif NetworkType==24:
                from FacePatches_NET import FacePatches_NET_3C_1I_2P as PaN
            elif NetworkType==25:
                from FacePatches_NET import FacePatches_NET_3C_2I_2P as PaN
            elif NetworkType==26:
                from FacePatches_NET import FacePatches_NET_3C_3I_2P as PaN
            else:
                print("Usage: python finetune.py <GPUID> <Module> <NetworkType>\nWith Module 2, NetworkType must be 0, 1")
                exit(-1)
            '''Here begins the implementation logic-------------------------------------------------------------------
            -------------------------------------------------------------------------------------------------------------'''
            #Holders for gray images
            eye_p_shape=[None, 26, 64, 1]
            midd_p_shape=[None, 49, 28, 1]
            mou_p_shape=[None, 30, 54, 1]

            eye_p = tf.placeholder(tf.float32, eye_p_shape)
            midd_p = tf.placeholder(tf.float32, midd_p_shape)
            mou_p = tf.placeholder(tf.float32, mou_p_shape)
            #Holder for labels in a batch size of batch_size, number of labels are to be determined
            labels = tf.placeholder(tf.float32, labelshape)#the number of labels are to be determined

            #FacePatch_net = PaN({'eyePatch_data':eye_p, 'middlePatch_data':midd_p, 'mouthPatch_data':mou_p})
            #print(type(FacePatch_net))
            #softmax=FacePatch_net.layers['prob']
            if NetworkType > 3 and NetworkType < 13:###current 4 5 6 7
                softmax=PaN(eye_p, midd_p, mou_p, classNo=cn)
            elif NetworkType >23 and NetworkType <27:###using only eye patch and mouth patch
                softmax=PaN(eye_p, mou_p, classNo=cn)
            else:
                FacePatch_net = PaN({'eyePatch_data':eye_p, 'middlePatch_data':midd_p, 'mouthPatch_data':mou_p})
                print(type(FacePatch_net))
                softmax=FacePatch_net.layers['prob']

            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=softmax),0)
            #optm=tf.train.RMSPropOptimizer(lr)
            optm=tf.train.AdamOptimizer(lr)
            train_op=optm.minimize(loss, global_step)#for train

            #for test
            correcta_prediction = tf.equal(tf.argmax(softmax,1),tf.argmax(labels,1))
            test_cast=tf.cast(correcta_prediction, "float")
            sum_test=tf.reduce_sum(test_cast)#for large test set
            accuracy = tf.reduce_mean(test_cast)#for small test set
    
            saver = tf.train.Saver()
            with tf.Session() as sess:
               
                sess.run(tf.global_variables_initializer())
                
                if loadONW:
                    #print('\n\n>>>>>>>>>>>>>>all collection keys')
                    #print(tf.get_default_graph().get_all_collection_keys())
                    #savelistcontent('./M3_all_collection_keys.txt',tf.get_default_graph().get_all_collection_keys())
                    #print('\n\n>>>>>>>>>>>>>>all variables')
                    #print(tf.get_default_graph().get_collection(name='variables'))
                    #savelistcontent('./M3_all_variables.txt',tf.get_default_graph().get_collection(name='variables'))
                    #print('\n\n>>>>>>>>>>>>>>all train_op')
                    #print(tf.get_default_graph().get_collection(name='train_op'))
                    #savelistcontent('./M3_train_op.txt',tf.get_default_graph().get_collection(name='train_op'))
                    #print('\n\n>>>>>>>>>>>>>>all trainable variables')
                    #print(tf.get_default_graph().get_collection(name='trainable_variables'))
                    #savelistcontent('./M3_trainable_variables_n5.txt', tf.get_default_graph().get_collection(name='trainable_variables'))
                    #exit(2)
                    restorefacepatchModel(DataSet, sess, NetworkType, tf.get_default_graph())
                    print('\nModels have been loaded.\n')
                iters=int((data.train.num_examples*Mini_Epochs)/batchSize)+1
                for i in range(iters):
                    afc=[]
                    batch=data.train.next_batch(batchSize, shuffle=False)
                    tloss, _=sess.run([loss, train_op], feed_dict={eye_p:batch[2], midd_p:batch[3], mou_p:batch[4], labels:batch[5]})
                    if tloss<mini_loss:
                        mini_loss=tloss
                    v_accuracy, valid_loss, oaa, confu_mat = Valid_on_TestSet_3NI(cn, sess, accuracy, sum_test, loss, softmax,
                                                                                  eye_p, data.test.eyep, midd_p, data.test.middlep,
                                                                                  mou_p, data.test.mouthp, labels, data.test.labels, afc=afc)
                    laflag = loss_a.analyzeLossVariation(valid_loss)
                    clr=cLR*(lr_drate)**(i//lrstep)
                    tt=time.time()
                    print("CLR:%.8f Ite:%06d Bs:%03d Epo:%04d Los:%.8f mLo:%08f\tVALID>> mVL: %.8f\tVL: %.8f\tVA: %f\tOAA: %f\tT: %fs"%
                          (clr,i,batchSize,data.train.epochs_completed, tloss, mini_loss, loss_a.minimun_loss, valid_loss, v_accuracy, oaa, (tt-t1)))
                    if laflag:
                        file_record = logfile(file_record, runs=runs, OAA=oaa, afc=afc, valid_loss=valid_loss, valid_min_loss=loss_a.minimun_loss, 
                            final_train_loss=tloss, train_min_loss=mini_loss, TA=v_accuracy, TC=(tt-t1),ILR=cLR, FLR=clr, LS=lrstep, ites=i,
                            Epo=data.train.epochs_completed, cBS=batchSize, iBS=batchSize,
                            input=sys.argv, CM=confu_mat, T=time.localtime(tt), df=dfile)
                        if loss_a.minimun_loss < stcmwvlilttv:
                            saver.save(sess=sess, save_path=model_save_path)
     
                '''MODULE3 ENDS---------------------------------------------------------------------------------------------'''
        #
        #
        #
        elif Module==6:
            stcmwvlilttv=1.4054#value need to be determined. save_the_current_model_when_validation_loss_is_less_than_this_value
            '''MODULE6---------------------------------------------------------------------------------------------------- 
            Options for the fusion net of vgg inner_face and geometry input
            -------------------------------------------------------------------------------------------------------------'''
            print('Network Type: %s'%(NetworkType))
            if NetworkType==440:
                from Geometric_NET import Geometric_NET_2h1I as GEON
                geonfcdim=1024
                from VGG_NET import VGG_NET_o as APPN
                appnfcdim=4096
                from FintuneNet import FTN0 as FTN
            elif NetworkType==441:
                from Geometric_NET import Geometric_NET_2h1I as GEON
                geonfcdim=1024
                from VGG_NET import VGG_NET_o as APPN
                appnfcdim=4096
                from FintuneNet import FTN1 as FTN
            else:
                print("Usage: python finetune.py <GPUID> <Module> <NetworkType>\nWrong NetworkType, please check the NetworkType input again.")
                exit(-1)
            '''Here begins the implementation logic-------------------------------------------------------------------
            -------------------------------------------------------------------------------------------------------------'''
            #define geometry graph
            geo_G=tf.Graph()
            with geo_G.as_default():
                geo_features=tf.placeholder(tf.float32, [None,122])
                geo_net=GEON({'data':geo_features})
                geofc=geo_net.layers['gefc2']
                #print(geo_G.get_all_collection_keys())
                #print(geo_G.get_collection(name='trainable_variables'))
                #print(geo_G.get_collection(name='variables'))
                gsaver = tf.train.Saver()
                #exit()

            #define appearance graph
            app_G=tf.Graph()
            with app_G.as_default():
                images = tf.placeholder(tf.float32, m1shape)
                app_net=APPN({'data':images})
                appfc=app_net.layers['fc2']
                asaver = tf.train.Saver()
        
            #define fine-tuning graph
            fint_G=tf.Graph()
            with fint_G.as_default():
                geo_fc=tf.placeholder(tf.float32, [None, geonfcdim])
                app_fc=tf.placeholder(tf.float32, [None, appnfcdim])
                labels = tf.placeholder(tf.float32, labelshape)#the number of labels are to be determined
                fin_net=FTN({'appfc':app_fc, 'geofc':geo_fc})
                
                softmax=tf.nn.softmax('prob')
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=softmax),0)
                optm=tf.train.RMSPropOptimizer(lr)
                train_op=optm.minimize(loss)#for train
                #for test
                correcta_prediction = tf.equal(tf.argmax(softmax,1),tf.argmax(labels,1))
                accuracy = tf.reduce_mean(tf.cast(correcta_prediction, "float"))
                #print(fint_G.get_all_collection_keys())
                #print(fint_G.get_collection(name='variables'))
                #print(fint_G.get_collection(name='train_op'))
                #print(fint_G.get_collection(name='trainable_variables'))
            #exit()
        
            print('Geometry graph at: \t\t', geo_G)
            print('Appearance graph at: \t\t', app_G)
            print('Fine-tuning graph at: \t\t', fint_G)
            #exit()
            #different sessions have different graph
            geo_sess=tf.InteractiveSession(graph=geo_G)
            app_sess=tf.InteractiveSession(graph=app_G)
            fin_sess=tf.InteractiveSession(graph=fint_G)
            print('\n%%%%%%%Sessions are created\n')

            try:
                #must initialize the variables in the graph for compution or loading pretrained weights
                geo_sess.run(tf.variables_initializer(var_list=geo_G.get_collection(name='variables')))
                print('\nGeometry network variables initialized.')
                #the gsaver must define in the graph of its owner session, or it will occur error in restoration or saving
                gsaver.restore(sess=geo_sess, save_path=selectGeoModelPathForModule6_8G(TestID=TestID))
                print('Geometry Model loaded')
            except:
                print('Unable to load the pretrained network for geo_net')
                traceback.print_exc()
                
        
            try:
                #must initialize the variables in the graph for compution or loading pretrained weights
                app_sess.run(tf.variables_initializer(var_list=app_G.get_collection(name='variables')))
                print('\nAppearance network variables initialized.')
                #the asaver must define in the graph of its owner session, or it will occur error in restoration or saving
                asaver.restore(sess=app_sess, save_path=selectAppModelPathForModule6_8G(TestID=TestID))
                print('Appearance Model loaded\n')
            except:
                print('Unable to load the pretrained network for app_net')
                traceback.print_exc()
                exit(2)
            #exit()
            try:
                #besides the variables, the optimizer also need to be initialized.
                #fin_sess.run(tf.variables_initializer(var_list=fint_G.get_collection(name='trainable_variables')))
                fin_sess.run(tf.variables_initializer(var_list=fint_G.get_collection(name='variables')))
                saver = tf.train.Saver()
                print('\nFine-tuning network variables initialized.')
            except:
                print('Unable to initialize Fine-tuning network variables')
                traceback.print_exc()
                exit(3)
                
            '''MODULE6 ENDS---------------------------------------------------------------------------------------------'''
        #
        #
        #face patch CNN and Geometry original features fusion
        elif Module==7:
            from sklearn import metrics
            stcmwvlilttv=1.4054#value need to be determined. save_the_current_model_when_validation_loss_is_less_than_this_value
            '''MODULE7---------------------------------------------------------------------------------------------------- 
            Options for the fusion net of face patches and geometry input
            -------------------------------------------------------------------------------------------------------------'''
            print('Network Type: %s'%(NetworkType))
            if NetworkType//10==6:#using network 6 in face patches, get fusion_1 layer output
                from FacePatches_NET import FacePatches_NET_3Conv_3Inception_tflearn as FPN
                fpndim=9526
                #m3modelname='./M7models/D502_M3_N6_T2_V2_R1_20171110055149_1.18062_.ckpt'
                m3modelname='./M7models/D502_M3_N6_T2_V2_R1_20171110055149_1.18062_.ckpt'
                facepatchpreprocessdatafilename='./Pre-Datasets/D%d_N%dinM3_pre-data_with_%ddims_from_%s.pkl'%(DataSet,6,fpndim,os.path.basename(m3modelname))
            else:
                print("Usage: python finetune.py <GPUID> <Module> <NetworkType>\nWrong NetworkType, please check the NetworkType input again.")
                exit(-1)
            '''Here begins the implementation logic-------------------------------------------------------------------
            -------------------------------------------------------------------------------------------------------------'''
            ###load data from 
            print('Checking path:\n%s\n'%(facepatchpreprocessdatafilename))
            if os.path.exists(facepatchpreprocessdatafilename):
                print('Loading data from previous generated file......')
                with open(facepatchpreprocessdatafilename, 'rb') as datafile:
                    Apredata=pickle.load(datafile)
            else:
                print('Generating data......')
                #define appearance graph
                fp_G=tf.Graph()
                with fp_G.as_default():
                    eye_p_shape=[None, 26, 64, 1]
                    midd_p_shape=[None, 49, 28, 1]
                    mou_p_shape=[None, 30, 54, 1]
                    eye_p = tf.placeholder(tf.float32, eye_p_shape)
                    midd_p = tf.placeholder(tf.float32, midd_p_shape)
                    mou_p = tf.placeholder(tf.float32, mou_p_shape)
                    softmax=FPN(eye_p, midd_p, mou_p)
                    fpsaver = tf.train.Saver()
                    fusion1=tflearn.get_layer_by_name('fusion_1')        
            
                print('Facepatches graph at: \t\t', fp_G)
                #exit()
                #different sessions have different graph
                fp_sess=tf.InteractiveSession(graph=fp_G)
                print('\n%%%%%%%Sessions are created\n')

                try:
                    #must initialize the variables in the graph for compution or loading pretrained weights
                    fp_sess.run(tf.variables_initializer(var_list=fp_G.get_collection(name='variables')))
                    print('\nFace Patches network variables initialized.')
                    #the gsaver must define in the graph of its owner session, or it will occur error in restoration or saving
                    fpsaver.restore(sess=fp_sess, save_path=m3modelname)
                    print('Face Patches Network Model loaded')
                except:
                    print('Unable to load the pretrained network for geo_net')
                    traceback.print_exc()
                    exit()
                data10g=DataSetPrepare.loadPKLDataWithPartitions_v4(Dataset_Dictionary.get(DataSet), Geometry=True, Patches=True, cn=cn)
                Apredata=[]
                print('Data contains %d groups.'%len(data10g))
                for dg in data10g:
                    predata={'X':[], 'Y':[]}
                    fpeval=[]
                    ncount=len(dg['labels'])
                    print('Processing data with %d samples.'%(ncount))
                    #print(dg['eye_patch'][0].shape)
                    if ncount>TestNumLimit:
                        iters=np.floor_divide(ncount, test_bat)
                        print(iters)
                        for ite in range(iters):
                            #print(ite)
                            start=test_bat*ite
                            end=test_bat*(ite+1)
                            fcd=fusion1.eval(feed_dict={eye_p:dg['eye_patch'][start:end], midd_p:dg['middle_patch'][start:end],
                                                        mou_p:dg['mouth_patch'][start:end]})
                            fpeval.extend(fcd)
                            del fcd
                        if ncount%test_bat>0:
                            fcd=fusion1.eval(feed_dict={eye_p:dg['eye_patch'][test_bat*iters:ncount], midd_p:dg['middle_patch'][test_bat*iters:ncount],
                                                            mou_p:dg['mouth_patch'][test_bat*iters:ncount]})
                        fpeval.extend(fcd)
                        del fcd
                    else:
                        fcd=fusion1.eval(feed_dict={eye_p:dg['eye_patch'], midd_p:dg['middle_patch'], mou_p:dg['mouth_patch']})
                        fpeval.extend(fcd)
                        del fcd
                    for index_extend in range(ncount):
                        predata['X'].append(np.append(fpeval[index_extend], dg['geometry'][index_extend]))
                    del fpeval
                    predata['Y'].extend(dg['labels'])
                    del dg['labels'], dg['geometry'], dg['eye_patch'], dg['mouth_patch'], dg['middle_patch'] 
                    print('%d samples with %d dims.\n'%(len(predata['Y']), len(predata['X'][0])))
                    Apredata.append(predata)
                    del predata
                del data10g
                with open(facepatchpreprocessdatafilename, 'wb') as fin:
                    pickle.dump(Apredata, fin, 4)
                    print('File saved.')
                
            #print(Apredata)
            #exit()
            data=groupdata(Apredata, ValidID, TestID)
            overtimes=1
            if continue_test:
                overtimes=OverTimes

            #nel=[7, 10, 14, 18, 21, 25, 28, 32]
            #mssl=[4, 8, 10, 14, 18, 21, 27, 32]
            #msll=[1, 2, 3, 5, 8, 10, 14, 18, 24, 27]#, should not exceed 5. for this subject
            #loopflag=False
            #log=log.replace('./logs','./logs/M%dtests'%(Module))#use for tuning
            #for v_nel in nel:
            #    if loopflag:
            #        break
            #    if NetworkType==60 or NetworkType==61:
            #        loopflag=True
            #    for v_mss in mssl:
            #        for v_msl in msll:
            #            n_estimators=v_nel#10, estimators for random forest classifier
            #            min_samples_split=v_mss#10
            #            min_samples_leaf=v_msl#5, should not exceed 5. for this subject

            #n_estimators=14#10, estimators for random forest classifier
            #min_samples_split=10#10
            #min_samples_leaf=5#5, should not exceed 5. for this subject
            sst=SIMSTS(overtimes)
            for test_run in range(overtimes):
                ct=time.time()
                m7_model_save_path=model_save_path.replace('_R'+str(runs)+time.strftime('_%Y%m%d%H%M%S',time.localtime(t1)),
                                                            '_R'+str(test_run)+time.strftime('_%Y%m%d%H%M%S',time.localtime(ct)))
                

                if NetworkType%10==0:
                    from sklearn import tree
                    optm = tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=min_samples_split, 
                                                        min_samples_leaf=min_samples_leaf)
                elif NetworkType%10==1:
                    from sklearn import tree
                    optm = tree.DecisionTreeClassifier(criterion='gini', min_samples_split=min_samples_split, 
                                                        min_samples_leaf=min_samples_leaf)
                elif NetworkType%10==2:
                    n_estimators=14#10, estimators for random forest classifier
                    min_samples_split=4#10
                    min_samples_leaf=5#5, should not exceed 5. for this subject
                    from sklearn.ensemble import RandomForestClassifier
                    optm = RandomForestClassifier(n_estimators=n_estimators, criterion='entropy', 
                                                    min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
                elif NetworkType%10==3:
                    n_estimators=32#10, estimators for random forest classifier
                    min_samples_split=4#10
                    min_samples_leaf=5#5, should not exceed 5. for this subject
                    from sklearn.ensemble import RandomForestClassifier
                    optm = RandomForestClassifier(n_estimators=n_estimators, criterion='gini', 
                                                    min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
                else:
                    print('ERROR:::::$$$$: Unexpected networktype encount.')
                    exit(-1)
                if test_run==0:
                    print('n_estimators(RFC):%d\tmin_samples_split:%d\tmin_samples_leaf:%d'%(n_estimators, 
                                                                                                        min_samples_split, min_samples_leaf))
                logpostfix='_E%d_MSS%d_MSL%d_'%(n_estimators, min_samples_split, min_samples_leaf)
                m7_model_save_path=m7_model_save_path.replace('.ckpt', '_%s.ckpt'%(type(optm).__name__))
                optm.fit(data.train['X'], data.train['Y'])

                tY=optm.predict(data.train['X'])
                train_acc=metrics.accuracy_score(np.asarray(data.train['Y']), tY)
                tcm=calR(tY, data.train['Y'])
                toaa=overAllAccuracy(tcm)

                pY=optm.predict(data.test['X'])
                #print(pY.shape)
                #print((np.asarray(data.test['Y'])).shape)
                accuracy=metrics.accuracy_score(np.asarray(data.test['Y']), pY)
                cm=calR(pY, data.test['Y'])
                oaa=overAllAccuracy(cm)
                tt=time.time()
                print('OT:%2d\tOAA:%.8f\tAcc:%.8f\tTOAA:%.8f\tTAc:%.8f\t%s\tT:%fs'%(test_run, oaa, accuracy, toaa, train_acc, str(type(optm).__name__),(tt-ct)))
                sst.addFigure(oaa)
                file_record=logfileForSklearnModel(file_record,test_run, optm, accuracy, oaa, cm, facepatchpreprocessdatafilename, train_acc, toaa, tcm)
                loss_a.setMinimun_loss(oaa)
                modelname=m7_model_save_path.replace('.ckpt','_%s_.pkl'%(str(oaa)))
                with open(modelname, 'wb') as fin:
                    pickle.dump(optm, fin, 4)

                tt=time.time()
                logf=log.replace('.txt',('_'+str(type(optm).__name__)+logpostfix+'.txt'))
                filelog=open(logf,'a')
                filelog.write('%s\t\t TotalTimeConsumed: %f\tOptimizer: %s\n'%(file_record, (tt-ct), str(type(optm).__name__)))
                filelog.close()
            state=sst.getSTS()
            print('Mean:%f\tMax:%f\tMin:%f'%(state[0], state[1], state[2]))
            sst.logfile(Module, DataSet, NetworkType, n_estimators, min_samples_split, min_samples_leaf)
            '''MODULE7 ENDS---------------------------------------------------------------------------------------------'''
        #
        #
        #face patch CNN features
        elif Module==8:
            from sklearn import metrics
            #from multiprocessing import pool
            stcmwvlilttv=1.4054#value need to be determined. save_the_current_model_when_validation_loss_is_less_than_this_value
            '''MODULE8---------------------------------------------------------------------------------------------------- 
            Options for the fusion net of face patches and geometry input
            -------------------------------------------------------------------------------------------------------------'''
            print('Network Type: %s'%(NetworkType))
            if NetworkType//10==6:#using network 6 in face patches, get fusion_1 layer output
                from FacePatches_NET import FacePatches_NET_3Conv_3Inception_tflearn as FPN
                fpndim=9216
                #m3modelname='./M7models/D502_M3_N6_T2_V2_R1_20171110055149_1.18062_.ckpt'
                m3modelname='./M7models/D502_M3_N6_T2_V2_R1_20171110055149_1.18062_.ckpt'
                facepatchpreprocessdatafilename='./Pre-Datasets/D%d_N%dinM3_pre-data_with_%ddims_from_%s.pkl'%(DataSet,6,fpndim,os.path.basename(m3modelname))
            else:
                print("Usage: python finetune.py <GPUID> <Module> <NetworkType>\nWrong NetworkType, please check the NetworkType input again.")
                exit(-1)
            '''Here begins the implementation logic-------------------------------------------------------------------
            -------------------------------------------------------------------------------------------------------------'''
            ###load data from 
            print('Checking path:\n%s\n'%(facepatchpreprocessdatafilename))
            if os.path.exists(facepatchpreprocessdatafilename):
                print('Loading data from previous generated file......')
                with open(facepatchpreprocessdatafilename, 'rb') as datafile:
                    Apredata=pickle.load(datafile)
            else:
                print('Generating data......')
                #define appearance graph
                fp_G=tf.Graph()
                with fp_G.as_default():
                    eye_p_shape=[None, 26, 64, 1]
                    midd_p_shape=[None, 49, 28, 1]
                    mou_p_shape=[None, 30, 54, 1]
                    eye_p = tf.placeholder(tf.float32, eye_p_shape)
                    midd_p = tf.placeholder(tf.float32, midd_p_shape)
                    mou_p = tf.placeholder(tf.float32, mou_p_shape)
                    softmax=FPN(eye_p, midd_p, mou_p)
                    fpsaver = tf.train.Saver()
                    fusion1=tflearn.get_layer_by_name('fusion_1')        
            
                print('Facepatches graph at: \t\t', fp_G)
                #exit()
                #different sessions have different graph
                fp_sess=tf.InteractiveSession(graph=fp_G)
                print('\n%%%%%%%Sessions are created\n')

                try:
                    #must initialize the variables in the graph for compution or loading pretrained weights
                    fp_sess.run(tf.variables_initializer(var_list=fp_G.get_collection(name='variables')))
                    print('\nFace Patches network variables initialized.')
                    #the gsaver must define in the graph of its owner session, or it will occur error in restoration or saving
                    fpsaver.restore(sess=fp_sess, save_path=m3modelname)
                    print('Face Patches Network Model loaded')
                except:
                    print('Unable to load the pretrained network for geo_net')
                    traceback.print_exc()
                    exit()
                data10g=DataSetPrepare.loadPKLDataWithPartitions_v4(Dataset_Dictionary.get(DataSet), Patches=True, cn=cn)
                Apredata=[]
                print('Data contains %d groups.'%len(data10g))
                for dg in data10g:
                    predata={'X':[], 'Y':[]}
                    fpeval=[]
                    ncount=len(dg['labels'])
                    print('Processing data with %d samples.'%(ncount))
                    #print(dg['eye_patch'][0].shape)
                    if ncount>TestNumLimit:
                        iters=np.floor_divide(ncount, test_bat)
                        print(iters)
                        for ite in range(iters):
                            #print(ite)
                            start=test_bat*ite
                            end=test_bat*(ite+1)
                            fcd=fusion1.eval(feed_dict={eye_p:dg['eye_patch'][start:end], midd_p:dg['middle_patch'][start:end],
                                                        mou_p:dg['mouth_patch'][start:end]})
                            fpeval.extend(fcd)
                            del fcd
                        if ncount%test_bat>0:
                            fcd=fusion1.eval(feed_dict={eye_p:dg['eye_patch'][test_bat*iters:ncount], midd_p:dg['middle_patch'][test_bat*iters:ncount],
                                                            mou_p:dg['mouth_patch'][test_bat*iters:ncount]})
                        fpeval.extend(fcd)
                        del fcd
                    else:
                        fcd=fusion1.eval(feed_dict={eye_p:dg['eye_patch'], midd_p:dg['middle_patch'], mou_p:dg['mouth_patch']})
                        fpeval.extend(fcd)
                        del fcd
                    for index_extend in range(ncount):
                        #predata['X'].append(np.append(fpeval[index_extend], dg['geometry'][index_extend]))
                        predata['X'].append(np.asarray(fpeval[index_extend]))
                    del fpeval
                    predata['Y'].extend(dg['labels'])
                    del dg['labels'], dg['eye_patch'], dg['mouth_patch'], dg['middle_patch'] 
                    print('%d samples with %d dims.\n'%(len(predata['Y']), len(predata['X'][0])))
                    Apredata.append(predata)
                    del predata
                del data10g
                with open(facepatchpreprocessdatafilename, 'wb') as fin:
                    pickle.dump(Apredata, fin, 4)
                    print('File saved.')
                
            #print(Apredata)
            #exit()
            data=groupdata(Apredata, ValidID, TestID)
            overtimes=1
            if continue_test:
                overtimes=OverTimes

            #nel=[7, 10, 14, 18, 21, 25, 28, 32]
            #mssl=[2, 4, 8, 10, 14, 18, 21, 27, 32]
            #msll=[1, 2, 3, 5, 8, 10]
            ##nel=[10, 14, 18, 21, 25, 28, 32]
            ##mssl=[10, 14, 18, 21, 27, 32]
            ##msll=[3]
            #loopflag=False
            #log=log.replace('./logs','./logs/M%dtests/details'%(Module))#use for tuning
            #for v_nel in nel:
            #    if loopflag:
            #        break
            #    if NetworkType==60 or NetworkType==61:
            #        loopflag=True
            #    for v_mss in mssl:
            #        for v_msl in msll:
                        #n_estimators=v_nel#10, estimators for random forest classifier
                        #min_samples_split=v_mss#10
                        #min_samples_leaf=v_msl#5

            sst=SIMSTS(overtimes)
            for test_run in range(overtimes):
                ct=time.time()
                m8_model_save_path=model_save_path.replace('_R'+str(runs)+time.strftime('_%Y%m%d%H%M%S',time.localtime(t1)),
                                                            '_R'+str(test_run)+time.strftime('_%Y%m%d%H%M%S',time.localtime(ct)))
            
                if NetworkType%10==0:
                    from sklearn import tree
                    optm = tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=min_samples_split, 
                                                        min_samples_leaf=min_samples_leaf)
                elif NetworkType%10==1:
                    from sklearn import tree
                    optm = tree.DecisionTreeClassifier(criterion='gini', min_samples_split=min_samples_split, 
                                                        min_samples_leaf=min_samples_leaf)
                elif NetworkType%10==2:
                    n_estimators=49#10, estimators for random forest classifier
                    min_samples_split=5#10
                    min_samples_leaf=3#5
                    max_depth=50
                    oob_score=True
                    from sklearn.ensemble import RandomForestClassifier
                    optm = RandomForestClassifier(n_estimators=n_estimators, criterion='entropy', 
                                                    min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                                    max_depth=max_depth, oob_score=oob_score)
                elif NetworkType%10==3:
                    n_estimators=21#10, estimators for random forest classifier
                    min_samples_split=4#10
                    min_samples_leaf=2#5
                    from sklearn.ensemble import RandomForestClassifier
                    optm = RandomForestClassifier(n_estimators=n_estimators, criterion='gini', 
                                                    min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
                else:
                    print('ERROR:::::$$$$: Unexpected networktype encount.')
                    exit(-1)
                
                logpostfix='_E%d_MSS%d_MSL%d_'%(n_estimators, min_samples_split, min_samples_leaf)
                if test_run==0:
                    print('n_estimators(RFC):%d\tmin_samples_split:%d\tmin_samples_leaf:%d'%(n_estimators, 
                                                                                                    min_samples_split, min_samples_leaf))
                m8_model_save_path=m8_model_save_path.replace('.ckpt', '_%s.ckpt'%(type(optm).__name__))
                optm.fit(data.train['X'], data.train['Y'])

                tY=optm.predict(data.train['X'])
                train_acc=metrics.accuracy_score(np.asarray(data.train['Y']), tY)
                tcm=calR(tY, data.train['Y'])
                toaa=overAllAccuracy(tcm)

                pY=optm.predict(data.test['X'])
                #print(pY.shape)
                #print((np.asarray(data.test['Y'])).shape)
                accuracy=metrics.accuracy_score(np.asarray(data.test['Y']), pY)
                cm=calR(pY, data.test['Y'])
                oaa=overAllAccuracy(cm)
                tt=time.time()
                print('OT:%2d\tOAA:%.8f\tAcc:%.8f\tTOAA:%.8f\tTAc:%.8f\t%s\tT:%fs'%(test_run, oaa, accuracy, toaa, train_acc, str(type(optm).__name__),(tt-ct)))
                sst.addFigure(oaa)
                file_record=logfileForSklearnModel(file_record,test_run, optm, accuracy, oaa, cm, facepatchpreprocessdatafilename, train_acc, toaa, tcm)
                loss_a.setMinimun_loss(oaa)
                modelname=m8_model_save_path.replace('.ckpt','_%s_.pkl'%(str(oaa)))
                with open(modelname, 'wb') as fin:
                    pickle.dump(optm, fin, 4)

                tt=time.time()
                logf=log.replace('.txt',('_'+str(type(optm).__name__)+logpostfix+'.txt'))
                filelog=open(logf,'a')
                filelog.write('%s\t\t TotalTimeConsumed: %f\tOptimizer: %s\n'%(file_record, (tt-ct), str(type(optm).__name__)))
                filelog.close()
            '''n_estimators=10#10, estimators for random forest classifier
            min_samples_split=18#10
            min_samples_leaf=10#5
            #freeze_support()
            pool_processes=pool.Pool(processes=8)
            apply_result_list=[]
            for test_run in range(overtimes):
                apply_result_list.append(pool_processes.apply_async(multiprocessingUnitForModule8tests,
                            (metrics, pickle, sst, model_save_path, runs, t1, test_run, 
                            NetworkType, data,facepatchpreprocessdatafilename, log,
                            n_estimators, min_samples_split, min_samples_leaf,)))
            pool_processes.close()
            pool_processes.join()
            for v in apply_result_list:
                sst.addFigure(v.get())'''

            state=sst.getSTS()
            print('Mean:%f\tMax:%f\tMin:%f'%(state[0], state[1], state[2]))
            sst.logfile(Module, DataSet, NetworkType, n_estimators, min_samples_split, min_samples_leaf)
            '''MODULE8 ENDS---------------------------------------------------------------------------------------------'''

        
        if not Module==7 and not Module==8:
            #newmodelname=model_save_path.split('.ckpt')[0]+'_'+str(loss_a.minimun_loss)+'_.ckpt'
            newmodelname=model_save_path.replace('.ckpt','_%s_.ckpt'%(str(loss_a.minimun_loss)))
            if os.path.exists(model_save_path+'.data-00000-of-00001'):
                os.rename((model_save_path+'.data-00000-of-00001'),(newmodelname+'.data-00000-of-00001'))
                os.rename((model_save_path+'.index'),(newmodelname+'.index'))
                os.rename((model_save_path+'.meta'),(newmodelname+'.meta'))

            tt=time.time()
            log=log.replace('.txt',('_'+str(type(optm).__name__)+'.txt'))
            filelog=open(log,'a')
            filelog.write('%s\t\t TotalTimeConsumed: %f\tOptimizer: %s\n'%(file_record, (tt-t1), str(type(optm).__name__)))
            filelog.close()

            print(log)
            print(log.split('.txt')[0])
            losslog=log.split('.txt')[0]+'_Runs%d_%d_%d'%(runs, ValidID, TestID)+'.validationlosslist'
            losslog=losslog.replace('./logs/','./logs/VL/')
            loss_a.outputlosslist(losslog)

    except:
        try:
            if not Module==7 and not Module==8:
                tt=time.time()
                log=log.replace('.txt',('_'+str(type(optm).__name__)+'.txt'))
                filelog=open(log,'a')
                filelog.write('%s\t\t TotalTimeConsumed: %f\tOptimizer: %s\n'%(file_record, (tt-t1), str(type(optm).__name__)))
                filelog.close()
                print('\n\n>>>>>> Saving current run info after it crrupted or interrupted.\n\n')
                print(log)
                print(log.split('.txt')[0])
                losslog=log.split('.txt')[0]+'_Runs%d_%d_%d'%(runs, ValidID, TestID)+'.validationlosslist'
                losslog=losslog.replace('./logs/','./logs/VL/')
                loss_a.outputlosslist(losslog)
            print('>>>>>> Current run info has been saved after it crrupted or interrupted.\n\n')
        except:
            print('ERROR: Fail to save current run info. after it crrupted')
        ferror=open(errorlog,'w')
        traceback.print_exc()
        traceback.print_exc(file=ferror)
        ferror.close()


def second_save(model_save_path, model_save_path_second):
    if os.path.exists(model_save_path+'.data-00000-of-00001'):
        if os.path.exists(model_save_path_second+'.data-00000-of-00001'):
            os.remove(model_save_path_second+'.data-00000-of-00001')
            os.remove(model_save_path_second+'.index')
            os.remove(model_save_path_second+'.meta')
           
        os.rename((model_save_path+'.data-00000-of-00001'),(model_save_path_second+'.data-00000-of-00001'))
        os.rename((model_save_path+'.index'),(model_save_path_second+'.index'))
        os.rename((model_save_path+'.meta'),(model_save_path_second+'.meta'))
    return True
def runWithTestPKL(GPU_Device_ID, Module, 
        DataSet,PKLList, 
        NetworkType, runs
        ,cLR=0.0001,batchSize=15,loadONW=False,reshape=False):
    try:
        initialize_dirs()
        '''GPU Option---------------------------------------------------------------------------------------------
        Determine which GPU is going to be used
        ------------------------------------------------------------------------------------------------------------'''
        print('GPU Option: %s'%(GPU_Device_ID))
        if (0==GPU_Device_ID) or (1==GPU_Device_ID):
            os.environ["CUDA_VISIBLE_DEVICES"]=str(GPU_Device_ID)
            errorlog='./logs/errors_gpu'+str(GPU_Device_ID)+'.txt'
            templog='./logs/templogs_newSC_gpu'+str(GPU_Device_ID)+'_M'+str(Module)+'_D'+str(DataSet)+'.txt'
        else:
            print("Usage: python finetune.py <GPUID> <Module> <NetworkType>\nGPUID must be 0 or 1\nModule must be 1, 2, or 3\nNetworkType must be 0, 1, 2, 3")
            exit(-1)
        '''GPU Option ENDS---------------------------------------------------------------------------------------'''
        cn=7#category numbers
        if int(DataSet)>60000:
            cn=6
        lrstep=6000
        mini_loss=10000
        file_record=None
        t1=time.time()
        logprefix='./logs/'
        model_save_path=''

        labelshape=[None, 7]
        m1shape= [None, 128, 128, 1]
        if DataSet>500:
            m2d=310
        else:
            m2d=122
        global Mini_Epochs
        global show_threshold
        #
        #
        #
        '''Input Data-------------------------------------------------------------------------------------------------
        -------------------------------------------------------------------------------------------------------------'''
        #
        ##data set loading
        #
        D_f=False
        if Module==2 and NetworkType<3:
            D_f=True
        dfile=Dataset_Dictionary.get(DataSet, False)
        if dfile==False:
            print('\nERROR: Unexpected DatasetID %d encouted.\n\n'%(int(DataSet)))
            exit(-1)
        train_data = DataSetPrepare.loadPKLData_v4(dfile, Module, Df=D_f,reshape=reshape, cn=cn)

        PKLList=PKLList.split(',')
        print('Data to be tested: ', PKLList)
        testIDstr=''
        loss_a=[]
        test_data_list=[]
        laflag=[]
        pkl_test_num=len(PKLList)
        for v in PKLList:
            if Dataset_Dictionary.get(int(v), False)==False:
                print('\nWARNING: Unexpected DatasetID %d encouted.\n\n'%(int(v)))
                continue
            testIDstr=testIDstr+'D'+str(v)
            loss_a.append(LOSS_ANA())
            test_data_list.append(DataSetPrepare.loadPKLData_v4(Dataset_Dictionary.get(int(v)), Module, Df=D_f, reshape=reshape, cn=cn))
            laflag.append(False)

        if DataSet==2:
            logprefix="./logs/D2CKplus_newrescalemetric_8groups_gpu"
            print("Processing 8 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==3:
            logprefix="./logs/D3CKpluslogbslr_weberface_8groups_gpu"
            print("Processing 8 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==4:
            logprefix="./logs/D4CKpluslogbslr_weberReverse_8groups_gpu"
            print("Processing 8 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==5:
            logprefix="./logs/D5CKpluslogbslr_weberface25up_8groups_gpu"
            print("Processing 8 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==6:
            m2d=258
            logprefix="./logs/D6CKplus_GeoFeatureV2_8groups_gpu"
            print("Processing 8 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==7:
            logprefix="./logs/D7CKpluslogbslr_weberface_innerface48x36_8groups_gpu"
            print("Processing 8 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==8:
            logprefix="./logs/D8CKpluslogbslr_ELTFS_8groups_gpu"
            print("Processing 8 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==9:
            m1shape= [None, 224, 224, 1]
            logprefix="./logs/D9CKpluslogbslr_weberface224_8groups_gpu"
            print("Processing 8 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==10:
            logprefix="./logs/D10CKpluslogbslr_weberface_10groups_gpu"
            print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==11:
            m1shape= [None, 224, 224, 1]
            logprefix="./logs/D11CKpluslogbslr_weberface224_10groups_gpu"
            print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==12:
            logprefix="./logs/D12CKpluslogbslr_ELTFS_10groups_gpu"
            print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==13:
            m2d=258
            logprefix="./logs/D13_CKplus_8G_V4_Geo258_ELTFS128x128_gpu"
            print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==15:
            logprefix="./logs/D15_CKPLUS_10G_EnlargebyWEF_testonoriginal_gpu"
            print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==16:
            if runs%2==0:
                batchSize=30
            else:
                batchSize=15
            logprefix="./logs/D16_CKPLUS_10G_Enlargeby2015CCV_10T_testonoriginal_gpu"
            print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==17:
            logprefix="./logs/D17_CKplus_10G_V4_weberface128x128_gpu"
            print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==18:
            logprefix="./logs/D18_CKplus_10G_V5_formalized_weberface128x128_gpu"
            print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==19:
            logprefix="./logs/D19_CKplus_10G_V4_ELTFS128x128_gpu"
            print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==33:
            batchSize=35
            logprefix="./logs/D33_KDEF_weberface_10groups_gpu"
            print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==32:
            batchSize=70
            logprefix="./logs/D32_KDEF_10G_EnlargebyWEF_testonoriginal_gpu"
            print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==34:
            batchSize=70
            logprefix="./logs/D34_KDEF_10G_Enlargeby2015CCV_10T_testonoriginal_gpu"
            print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==42:
            batchSize=60
            logprefix="./logs/D42_JAFFE_10G_Enlargeby_WEF_testonoriginaldataset_gpu"
            print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==40:
            logprefix="./logs/D40_JAFFE_10G_gpu"
            print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==43:
            batchSize=60
            logprefix="./logs/D43_JAFFE_10G_Enlargeby2015CCV_10T_testonoriginaldataset_gpu"
            print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==111:
            batchSize=30
            logprefix="./logs/D111_MergeDataset_D10_D33_D40_10G_gpu"
            print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==222:
            batchSize=30
            logprefix="./logs/D222_MergeDataset_D16_D34_D43_10G_gpu"
            print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==333:
            batchSize=30
            logprefix="./logs/D333_MergeDataset_D16_D34_10G_gpu"
            print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==444:
            batchSize=30
            logprefix="./logs/D444_MergeDataset_D10_D33_10G_gpu"
            print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==501:
            if runs%2==0:
                batchSize=30
            else:
                batchSize=15
            logprefix="./logs/D501_gpu"
            print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==502:
            if runs%2==0:
                batchSize=30
            else:
                batchSize=15
            logprefix="./logs/D502_gpu"
            print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==531:
            if runs%2==0:
                batchSize=15
            else:
                batchSize=30
            logprefix="./logs/D531_gpu"
            print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==532:
            if runs%2==0:
                batchSize=15
            else:
                batchSize=30
            logprefix="./logs/D532_gpu"
            print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==551:
            if runs%2==0:
                batchSize=21
            else:
                batchSize=15
            logprefix="./logs/D551_gpu"
            print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==552:
            if runs%2==0:
                batchSize=21
            else:
                batchSize=15
            logprefix="./logs/D552_gpu"
            print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==553:
            if runs%2==0:
                batchSize=21
            else:
                batchSize=15
            logprefix="./logs/D553_gpu"
            print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==554:
            if runs%2==0:
                batchSize=21
            else:
                batchSize=15
            logprefix="./logs/D554_gpu"
            print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==600:
            if runs%2==0:
                batchSize=35
            else:
                batchSize=70
            cLR=0.00001
            logprefix="./logs/D600_gpu"
            print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==601:
            if runs%2==0:
                batchSize=35
            else:
                batchSize=70
            cLR=0.00001
            logprefix="./logs/D601_gpu"
            print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==610:
            if runs%3==0:
                batchSize=35
            elif runs%3==1:
                batchSize=70
            else:
                batchSize=128
                Mini_Epochs=Mini_Epochs*2
            cLR=0.00001
            logprefix="./logs/D610_gpu"
            print("Processing dataset>>>>>>>>\n%s"%(logprefix))
        elif DataSet==611:
            if runs%3==0:
                batchSize=35
            elif runs%3==1:
                batchSize=70
            else:
                batchSize=128
                Mini_Epochs=Mini_Epochs*2
            cLR=0.00001
            logprefix="./logs/D611_gpu"
            print("Processing dataset>>>>>>>>\n%s"%(logprefix))
        elif DataSet==620:
            if runs%3==0:
                batchSize=35
            elif runs%3==1:
                batchSize=70
            else:
                batchSize=128
                Mini_Epochs=Mini_Epochs*2
            cLR=0.00001
            logprefix="./logs/D620_gpu"
            print("Processing dataset>>>>>>>>\n%s"%(logprefix))
        elif DataSet==621:
            if runs%3==0:
                batchSize=35
            elif runs%3==1:
                batchSize=70
            else:
                batchSize=128
                Mini_Epochs=Mini_Epochs*2
            cLR=0.00001
            logprefix="./logs/D621_gpu"
            print("Processing dataset>>>>>>>>\n%s"%(logprefix))
        elif DataSet==1001:
            if runs%2==0:
                batchSize=30
            else:
                batchSize=15
            logprefix="./logs/D1001_gpu"
            print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==1002:
            if runs%2==0:
                batchSize=30
            else:
                batchSize=15
            logprefix="./logs/D1002_gpu"
            print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
        else:
            print('ERROR: Unexpeted Dataset ID')
            exit()
        #
        #
        #
        tt=time.time()
        if reshape:
            logprefix=logprefix+'_reshape64x64'
        if Module==6:
            log=logprefix+str(GPU_Device_ID)+"_M"+str(Module)+"_D"+str(DataSet)+"_N"+str(NetworkType)+'_FullDataForTrainingSubjectTo_'+testIDstr+"_newStopCriteriaV3.txt"
        elif loadONW:
            log=logprefix+str(GPU_Device_ID)+"_M"+str(Module)+"_D"+str(DataSet)+"_N"+str(NetworkType)+'_FullDataForTrainingSubjectTo_'+testIDstr+"_withPretrainModelWeight_newStopCriteriaV3.txt"
        else:
            log=logprefix+str(GPU_Device_ID)+"_M"+str(Module)+"_D"+str(DataSet)+"_N"+str(NetworkType)+'_FullDataForTrainingSubjectTo_'+testIDstr+"_noPretrain_newStopCriteriaV3.txt"
            #logfilename=time.strftime('%Y%m%d%H%M%S',time.localtime(tt))+str(sys.argv[2:4])
        print('Time used for loading data: %fs'%(tt-t1))
        
        if os.path.exists("J:/Models/saves/"):
            model_save_path=("J:/Models/saves/"+'M'+str(Module)+'/D'+str(DataSet)+'/N'+str(NetworkType)+'/')
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
            model_save_path=(model_save_path+'D'+str(DataSet)+'_M'+str(Module)+'_N'+str(NetworkType)+'_FullDataForTrainingSubjectTo_'+testIDstr+'_R'
                            +str(runs)+time.strftime('_%Y%m%d%H%M%S',time.localtime(t1))+".ckpt")
        else:
            model_save_path=("./saves/"+'M'+str(Module)+'/D'+str(DataSet)+'/N'+str(NetworkType)+'/')
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
            model_save_path=(model_save_path+'D'+str(DataSet)+'_M'+str(Module)+'_N'+str(NetworkType)+'_FullDataForTrainingSubjectTo_'+testIDstr+'_R'
                            +str(runs)+time.strftime('_%Y%m%d%H%M%S',time.localtime(t1))+".ckpt")
        model_save_path_second=model_save_path.replace('.ckpt','_second.ckpt')
        '''Input Data Ends-----------------------------------------------------------------------------------------'''
        #
        #
        #
        if reshape:
            m1shape=[None, 64, 64, 1]
            print('Module 1 images input shape has been set to %s'%str(m1shape))
            model_save_path=model_save_path.replace('.ckpt','_reshape.ckpt')
        #
        #
        #
        
        global_step = tf.Variable(0, trainable=False)
        lr=tf.train.exponential_decay(cLR, global_step, lrstep, lr_drate, staircase=True)

        if Module==1:
            stcmwvlilttv_for_mutilTest=1.4674#save_the_current_model_when_validation_loss_is_less_than_this_value
            if DataSet==554 or DataSet==551 or DataSet==552 or DataSet==553:
                stcmwvlilttv_for_mutilTest=1.7
            elif DataSet==610 or DataSet==611:
                stcmwvlilttv_for_mutilTest=1.70
            '''MODULE1---------------------------------------------------------------------------------------------------- 
            Options for the whole-face-network
            Only need to select one of the import options as the network for the whole face feature extraction.
            -------------------------------------------------------------------------------------------------------------'''
            print('Network Type: %s'%(NetworkType))
            if NetworkType==0:
                from VGG_NET import VGG_NET_20l_512o as WFN
            elif NetworkType==1:
                from VGG_NET import VGG_NET_20l_128o as WFN
            elif NetworkType==2:
                from VGG_NET import VGG_NET_16l_128o as WFN
            elif NetworkType==3:
                from VGG_NET import VGG_NET_16l_72o as WFN
            elif NetworkType==4:
                from VGG_NET import VGG_NET_o as WFN
            elif NetworkType==8:
                from VGG_NET import VGG_NET_Inception1 as WFN
            elif NetworkType==9:
                from VGG_NET import VGG_NET_Inception2 as WFN
            elif NetworkType==10:
                from VGG_NET import VGG_NET_O_tfl as WFN
            elif NetworkType==11:
                from VGG_NET import VGG_NET_I5 as WFN
            elif NetworkType==12:
                from VGG_NET import VGG_NET_I5_ELU as WFN

            else:
                print("Usage: python finetune.py <GPUID> <Module> <NetworkType>\nWith Module 1, NetworkType must be 0, 1, 2, 3")
                exit(-1)
            '''Here begins the implementation logic-------------------------------------------------------------------
            -------------------------------------------------------------------------------------------------------------'''
            #Holder for gray images with m1shape in a batch size of batch_size
            images = tf.placeholder(tf.float32, m1shape)
            #Holder for labels in a batch size of batch_size, number of labels are to be determined
            labels = tf.placeholder(tf.float32, labelshape)#the number of labels are to be determined

            if NetworkType==10 or NetworkType==11 or NetworkType==12:
                Mini_Epochs = 60
                softmax=WFN(images)
            else:
                whole_face_net = WFN({'data':images})
                softmax=whole_face_net.layers['prob']

            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=softmax),0)
            #optm=tf.train.RMSPropOptimizer(lr)
            optm=tf.train.AdamOptimizer(lr)
            #print(optm.get_name())
            #print(type(optm).__name__)
            #exit()
            train_op=optm.minimize(loss,global_step=global_step)#for train
            
            #for test
            correcta_prediction = tf.equal(tf.argmax(softmax,1),tf.argmax(labels,1))
            test_cast=tf.cast(correcta_prediction, "float")
            sum_test=tf.reduce_sum(test_cast)#for large test set
            accuracy = tf.reduce_mean(test_cast)#for small test set

            with tf.Session() as sess:

                sess.run(tf.global_variables_initializer())

                if loadONW:
                    if NetworkType==10 or NetworkType==11 or NetworkType==12:
                        restorevggModel(sess, NetworkType, tf.get_default_graph())
                    else:
                        loadPretrainedModel(NetworkType, whole_face_net, sess,Module)
                    print('Model has been restored.\n')

                saver = tf.train.Saver()
                iters=int((train_data.num_examples*Mini_Epochs)/batchSize)+1
                for i in range(iters):
                    afc=[]
                    batch=train_data.next_batch(batchSize, shuffle=False)
                    tloss, _=sess.run([loss, train_op], feed_dict={images:batch[0], labels:batch[5]})
                    if tloss<mini_loss:
                        mini_loss=tloss
                    if tloss > show_threshold:
                        clr=cLR*(lr_drate)**(i//lrstep)
                        tt=time.time()
                        print("CLR:%.8f Ite:%06d Bs:%03d Epo:%03d Los:%.8f mLo:%08f T:%fs"%
                                (clr,i,batchSize,train_data.epochs_completed, tloss, mini_loss, (tt-t1)))
                    else:
                        V_string='VALID>>'
                        cm_string='ConfusionMatrix>> '
                        for pkl_i in range(pkl_test_num):
                            afc=[]
                            v_accuracy, valid_loss, oaa, confu_mat = Valid_on_TestSet(cn, sess, accuracy, sum_test, loss, softmax,
                                                                                          images, test_data_list[pkl_i].res_images, labels, test_data_list[pkl_i].labels,afc=afc)
                            laflag[pkl_i] = loss_a[pkl_i].analyzeLossVariation(valid_loss)
                            V_string=V_string+'D%d OAA:%f VA:%f %s mVL:%.8f VL:%.8f  '%(int(PKLList[pkl_i]), oaa, v_accuracy, str(afc), loss_a[pkl_i].minimun_loss, valid_loss)
                            cm_string=cm_string+'D%d:'%(int(PKLList[pkl_i]))+str(confu_mat)+' '
                        clr=cLR*(lr_drate)**(i//lrstep)
                        tt=time.time()
                        print("CLR:%.8f Ite:%06d Bs:%03d Epo:%03d Los:%.8f mLo:%08f %s T:%fs"%
                                (clr,i,batchSize,train_data.epochs_completed, tloss, mini_loss, V_string, (tt-t1)))
                        if laflag[0]:
                            file_record = logfileV2(file_record, runs=runs, V_string=V_string, 
                                final_train_loss=tloss, train_min_loss=mini_loss, TC=(tt-t1), ILR=cLR, FLR=clr, LS=lrstep, ites=i,
                                Epo=train_data.epochs_completed, cBS=batchSize, iBS=batchSize,
                                input=sys.argv, CMstring=cm_string, T=time.localtime(tt), df=dfile)
                            if loss_a[0].minimun_loss < stcmwvlilttv_for_mutilTest:
                                second_save(model_save_path, model_save_path_second)
                                saver.save(sess=sess, save_path=model_save_path)
                '''MODULE1 ENDS---------------------------------------------------------------------------------------------'''
        #
        #
        #
        elif Module==2:
            show_threshold = 1.75
            Mini_Epochs = 100
            if DataSet==601:
                if runs%2==0:
                    batchSize = 35
                else:
                    batchSize = 70
            stcmwvlilttv_for_mutilTest=1.3854#value need to be determined. save_the_current_model_when_validation_loss_is_less_than_this_value
            '''MODULE2---------------------------------------------------------------------------------------------------- 
            Options for the Geometry-network
            Only need to select one of the import options as the network for the geometry feature extraction.
            -------------------------------------------------------------------------------------------------------------'''
            print('Geometry Network Type: %s'%(NetworkType))
            if NetworkType==0:
                from Geometric_NET import Geometric_NET_2c2l as GeN
            elif NetworkType==1:
                from Geometric_NET import Geometric_NET_2c2lcc1 as GeN
            elif NetworkType==2:
                from Geometric_NET import Geometric_NET_2c2lcc1l1 as GeN
            elif NetworkType==3:
                from Geometric_NET import Geometric_NET_1h as GeN
            elif NetworkType==4:
                from Geometric_NET import Geometric_NET_2h1I as GeN
            elif NetworkType==5:
                from Geometric_NET import Geometric_NET_3h1I as GeN
                clr=0.00001
                learningRate=0.00001
            elif NetworkType==6:
                from Geometric_NET import Geometric_NET_h1I as GeN
            else:
                print("Usage: python finetune.py <GPUID> <Module> <NetworkType>\nWith Module 2, NetworkType must be 0, 1, 2")
                exit(-1)
            '''Here begins the implementation logic-------------------------------------------------------------------
            -------------------------------------------------------------------------------------------------------------'''
            #Holder for geometry features with 122 in a batch size of batch_size
            if D_f:
                geo_features = tf.placeholder(tf.float32, [None, m2d, 1])
            else:
                geo_features = tf.placeholder(tf.float32, [None, m2d])

            #Holder for labels in a batch size of batch_size, number of labels are to be determined
            labels = tf.placeholder(tf.float32, labelshape)#the number of labels are to be determined
            
            Geometry_net = GeN({'data':geo_features})
            print(type(Geometry_net))
            softmax=Geometry_net.layers['geprob']

            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=softmax),0)
            #optm=tf.train.RMSPropOptimizer(lr)
            optm = tf.train.AdamOptimizer(lr)
            #optm=tf.train.RMSPropOptimizer(lr)
            train_op=optm.minimize(loss, global_step=global_step)#for train

            #for test
            correcta_prediction = tf.equal(tf.argmax(softmax,1),tf.argmax(labels,1))
            test_cast=tf.cast(correcta_prediction, "float")
            sum_test=tf.reduce_sum(test_cast)#for large test set
            accuracy = tf.reduce_mean(test_cast)#for small test set
        
            with tf.Session() as sess:

                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver()

                iters=int((train_data.num_examples*Mini_Epochs)/batchSize)+1
                for i in range(iters):
                    batch=train_data.next_batch(batchSize, shuffle=False)
                    tloss, _=sess.run([loss, train_op], feed_dict={geo_features:batch[1], labels:batch[5]})
                    if tloss<mini_loss:
                        mini_loss=tloss
                    if tloss > show_threshold:
                        clr=cLR*(lr_drate)**(i//lrstep)
                        tt=time.time()
                        print("CLR:%.8f Ite:%06d Bs:%03d Epo:%03d Los:%.8f mLo:%08f T:%fs"%
                                (clr,i,batchSize,train_data.epochs_completed, tloss, mini_loss, (tt-t1)))
                    else:
                        V_string='VALID>>'
                        cm_string='ConfusionMatrix>> '
                        for pkl_i in range(pkl_test_num):
                            v_accuracy, valid_loss, oaa, confu_mat = Valid_on_TestSet(cn, sess, accuracy, sum_test, loss, softmax,
                                                                                          geo_features, test_data_list[pkl_i].geometry, labels, test_data_list[pkl_i].labels)
                            laflag[pkl_i] = loss_a[pkl_i].analyzeLossVariation(valid_loss)
                            V_string=V_string+'D%d OAA:%f VA:%f mVL:%.8f VL:%.8f  '%(int(PKLList[pkl_i]), oaa, v_accuracy, loss_a[pkl_i].minimun_loss, valid_loss)
                            cm_string=cm_string+'D%d:'%(int(PKLList[pkl_i]))+str(confu_mat)+' '
                        clr=cLR*(lr_drate)**(i//lrstep)
                        tt=time.time()
                        print("CLR:%.8f Ite:%06d Bs:%03d Epo:%03d Los:%.8f mLo:%08f %s T:%fs"%
                                (clr,i,batchSize,train_data.epochs_completed, tloss, mini_loss, V_string, (tt-t1)))
                        if laflag[0]:
                            file_record = logfileV2(file_record, runs=runs, V_string=V_string, 
                                final_train_loss=tloss, train_min_loss=mini_loss, TC=(tt-t1), ILR=cLR, FLR=clr, LS=lrstep, ites=i,
                                Epo=train_data.epochs_completed, cBS=batchSize, iBS=batchSize,
                                input=sys.argv, CMstring=cm_string, T=time.localtime(tt), df=dfile)
                            if loss_a[0].minimun_loss < stcmwvlilttv_for_mutilTest:
                                second_save(model_save_path, model_save_path_second)
                                saver.save(sess=sess, save_path=model_save_path)
                '''MODULE2 ENDS---------------------------------------------------------------------------------------------'''
        #
        #
        #
        elif Module==3:
            stcmwvlilttv_for_mutilTest=1.4154#value need to be determined. save_the_current_model_when_validation_loss_is_less_than_this_value
            if DataSet==502 or DataSet==501:
                stcmwvlilttv_for_mutilTest=1.4854
            elif DataSet==532 or DataSet==531:
                stcmwvlilttv_for_mutilTest=1.4004
            elif DataSet==554 or DataSet==551 or DataSet==552 or DataSet==553:
                stcmwvlilttv_for_mutilTest=1.7
            elif DataSet==610 or DataSet==611:
                stcmwvlilttv_for_mutilTest=1.7
            '''MODULE3---------------------------------------------------------------------------------------------------- 
            Options for the face_patches-network
    
            -------------------------------------------------------------------------------------------------------------'''
            print('FacePatch Network Type: %s'%(NetworkType))
            if NetworkType==0:
                from FacePatches_NET import FacePatches_NET_2Inceptions as PaN
            elif NetworkType==1:
                from FacePatches_NET import FacePatches_NET_2Inceptions_4lrn as PaN
            elif NetworkType==2:
                from FacePatches_NET import FacePatches_NET_2Inceptions_4lrn2 as PaN
            elif NetworkType==3:
                from FacePatches_NET import FacePatches_NET_3Conv_2Inception as PaN
            elif NetworkType==4:
                #from FacePatches_NET import FacePatches_NET_3Conv_1Inception as PaN
                from FacePatches_NET import FacePatches_NET_3Conv_IInception_tflear as PaN
            elif NetworkType==5:
                from FacePatches_NET import FacePatches_NET_3Conv_2Inception_tflearn as PaN
            elif NetworkType==6:
                from FacePatches_NET import FacePatches_NET_3Conv_3Inception_tflearn as PaN
            elif NetworkType==7:
                from FacePatches_NET import FacePatches_NET_3Conv_3Inception_tflearn_ELU as PaN
            elif NetworkType==24:
                from FacePatches_NET import FacePatches_NET_3C_1I_2P as PaN
            elif NetworkType==25:
                from FacePatches_NET import FacePatches_NET_3C_2I_2P as PaN
            elif NetworkType==26:
                from FacePatches_NET import FacePatches_NET_3C_3I_2P as PaN
            else:
                print("Usage: python finetune.py <GPUID> <Module> <NetworkType>\nWith Module 2, NetworkType must be 0, 1")
                exit(-1)
            '''Here begins the implementation logic-------------------------------------------------------------------
            -------------------------------------------------------------------------------------------------------------'''
            #Holders for gray images
            eye_p_shape=[None, 26, 64, 1]
            midd_p_shape=[None, 49, 28, 1]
            mou_p_shape=[None, 30, 54, 1]

            eye_p = tf.placeholder(tf.float32, eye_p_shape)
            midd_p = tf.placeholder(tf.float32, midd_p_shape)
            mou_p = tf.placeholder(tf.float32, mou_p_shape)
            #Holder for labels in a batch size of batch_size, number of labels are to be determined
            labels = tf.placeholder(tf.float32, labelshape)#the number of labels are to be determined

            #FacePatch_net = PaN({'eyePatch_data':eye_p, 'middlePatch_data':midd_p, 'mouthPatch_data':mou_p})
            #print(type(FacePatch_net))
            #softmax=FacePatch_net.layers['prob']
            if NetworkType > 3 and NetworkType < 8:###current 4 5 6 7
                softmax=PaN(eye_p, midd_p, mou_p)
            elif NetworkType >23 and NetworkType <27:###using only eye patch and mouth patch
                softmax=PaN(eye_p, mou_p)
            else:
                FacePatch_net = PaN({'eyePatch_data':eye_p, 'middlePatch_data':midd_p, 'mouthPatch_data':mou_p})
                print(type(FacePatch_net))
                softmax=FacePatch_net.layers['prob']

            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=softmax),0)
            #optm=tf.train.RMSPropOptimizer(lr)
            optm=tf.train.AdamOptimizer(lr)
            train_op=optm.minimize(loss, global_step)#for train

            #for test
            correcta_prediction = tf.equal(tf.argmax(softmax,1),tf.argmax(labels,1))
            test_cast=tf.cast(correcta_prediction, "float")
            sum_test=tf.reduce_sum(test_cast)#for large test set
            accuracy = tf.reduce_mean(test_cast)#for small test set
    
            with tf.Session() as sess:
               
                sess.run(tf.global_variables_initializer())
                
                if loadONW:
                    restorefacepatchModel(DataSet, sess, NetworkType, tf.get_default_graph())

                saver = tf.train.Saver()
                iters=int((train_data.num_examples*Mini_Epochs)/batchSize)+1
                for i in range(iters):
                    batch=train_data.next_batch(batchSize, shuffle=False)
                    tloss, _=sess.run([loss, train_op], feed_dict={eye_p:batch[2], midd_p:batch[3], mou_p:batch[4], labels:batch[5]})
                    if tloss<mini_loss:
                        mini_loss=tloss
                    if tloss > show_threshold:
                        clr=cLR*(lr_drate)**(i//lrstep)
                        tt=time.time()
                        print("CLR:%.8f Ite:%06d Bs:%03d Epo:%03d Los:%.8f mLo:%08f T:%fs"%
                                (clr,i,batchSize,train_data.epochs_completed, tloss, mini_loss, (tt-t1)))
                    else:
                        V_string='VALID>>'
                        cm_string='ConfusionMatrix>> '
                        for pkl_i in range(pkl_test_num):
                            afc=[]
                            v_accuracy, valid_loss, oaa, confu_mat = Valid_on_TestSet_3NI(cn, sess, accuracy, sum_test, loss, softmax,
                                                                                          eye_p, test_data_list[pkl_i].eyep, midd_p, test_data_list[pkl_i].middlep,
                                                                                          mou_p, test_data_list[pkl_i].mouthp, labels, test_data_list[pkl_i].labels,afc=afc)
                            laflag[pkl_i] = loss_a[pkl_i].analyzeLossVariation(valid_loss)
                            V_string=V_string+'D%d OAA:%f VA:%f %s mVL:%.8f VL:%.8f  '%(int(PKLList[pkl_i]), oaa, v_accuracy, str(afc), loss_a[pkl_i].minimun_loss, valid_loss)
                            cm_string=cm_string+'D%d:'%(int(PKLList[pkl_i]))+str(confu_mat)+' '
                        clr=cLR*(lr_drate)**(i//lrstep)
                        tt=time.time()
                        print("CLR:%.8f Ite:%06d Bs:%03d Epo:%03d Los:%.8f mLo:%08f %s T:%fs"%
                                (clr,i,batchSize,train_data.epochs_completed, tloss, mini_loss, V_string, (tt-t1)))
                        if laflag[0]:
                            file_record = logfileV2(file_record, runs=runs, V_string=V_string, 
                                final_train_loss=tloss, train_min_loss=mini_loss, TC=(tt-t1), ILR=cLR, FLR=clr, LS=lrstep, ites=i,
                                Epo=train_data.epochs_completed, cBS=batchSize, iBS=batchSize,
                                input=sys.argv, CMstring=cm_string, T=time.localtime(tt), df=dfile)
                            if loss_a[0].minimun_loss < stcmwvlilttv_for_mutilTest:
                                second_save(model_save_path, model_save_path_second)
                                saver.save(sess=sess, save_path=model_save_path)
                '''MODULE3 ENDS---------------------------------------------------------------------------------------------'''
        #
        #
        #
        elif Module==6:
            stcmwvlilttv_for_mutilTest=1.4054#value need to be determined. save_the_current_model_when_validation_loss_is_less_than_this_value
            '''MODULE6---------------------------------------------------------------------------------------------------- 
            Options for the fusion net of vgg inner_face and geometry input
            -------------------------------------------------------------------------------------------------------------'''
            print('Network Type: %s'%(NetworkType))
            if NetworkType==440:
                from Geometric_NET import Geometric_NET_2h1I as GEON
                geonfcdim=1024
                from VGG_NET import VGG_NET_o as APPN
                appnfcdim=4096
                from FintuneNet import FTN0 as FTN
            elif NetworkType==441:
                from Geometric_NET import Geometric_NET_2h1I as GEON
                geonfcdim=1024
                from VGG_NET import VGG_NET_o as APPN
                appnfcdim=4096
                from FintuneNet import FTN1 as FTN
            else:
                print("Usage: python finetune.py <GPUID> <Module> <NetworkType>\nWrong NetworkType, please check the NetworkType input again.")
                exit(-1)
            '''Here begins the implementation logic-------------------------------------------------------------------
            -------------------------------------------------------------------------------------------------------------'''
            #define geometry graph
            geo_G=tf.Graph()
            with geo_G.as_default():
                geo_features=tf.placeholder(tf.float32, [None,122])
                geo_net=GEON({'data':geo_features})
                geofc=geo_net.layers['gefc2']
                #print(geo_G.get_all_collection_keys())
                #print(geo_G.get_collection(name='trainable_variables'))
                #print(geo_G.get_collection(name='variables'))
                gsaver = tf.train.Saver()
                #exit()

            #define appearance graph
            app_G=tf.Graph()
            with app_G.as_default():
                images = tf.placeholder(tf.float32, m1shape)
                app_net=APPN({'data':images})
                appfc=app_net.layers['fc2']
                asaver = tf.train.Saver()
        
            #define fine-tuning graph
            fint_G=tf.Graph()
            with fint_G.as_default():
                geo_fc=tf.placeholder(tf.float32, [None, geonfcdim])
                app_fc=tf.placeholder(tf.float32, [None, appnfcdim])
                labels = tf.placeholder(tf.float32, labelshape)#the number of labels are to be determined
                fin_net=FTN({'appfc':app_fc, 'geofc':geo_fc})
                
                softmax=fin_net.layers['prob']
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=softmax),0)
                optm=tf.train.RMSPropOptimizer(lr)
                train_op=optm.minimize(loss)#for train
                #for test
                correcta_prediction = tf.equal(tf.argmax(softmax,1),tf.argmax(labels,1))
                accuracy = tf.reduce_mean(tf.cast(correcta_prediction, "float"))
                #print(fint_G.get_all_collection_keys())
                #print(fint_G.get_collection(name='variables'))
                #print(fint_G.get_collection(name='train_op'))
                #print(fint_G.get_collection(name='trainable_variables'))
            #exit()
        
            print('Geometry graph at: \t\t', geo_G)
            print('Appearance graph at: \t\t', app_G)
            print('Fine-tuning graph at: \t\t', fint_G)
            #exit()
            #different sessions have different graph
            geo_sess=tf.InteractiveSession(graph=geo_G)
            app_sess=tf.InteractiveSession(graph=app_G)
            fin_sess=tf.InteractiveSession(graph=fint_G)
            print('\n%%%%%%%Sessions are created\n')

            try:
                #must initialize the variables in the graph for compution or loading pretrained weights
                geo_sess.run(tf.variables_initializer(var_list=geo_G.get_collection(name='variables')))
                print('\nGeometry network variables initialized.')
                #the gsaver must define in the graph of its owner session, or it will occur error in restoration or saving
                gsaver.restore(sess=geo_sess, save_path=selectGeoModelPathForModule6_8G(TestID=TestID))
                print('Geometry Model loaded')
            except:
                print('Unable to load the pretrained network for geo_net')
                traceback.print_exc()
                
        
            try:
                #must initialize the variables in the graph for compution or loading pretrained weights
                app_sess.run(tf.variables_initializer(var_list=app_G.get_collection(name='variables')))
                print('\nAppearance network variables initialized.')
                #the asaver must define in the graph of its owner session, or it will occur error in restoration or saving
                asaver.restore(sess=app_sess, save_path=selectAppModelPathForModule6_8G(TestID=TestID))
                print('Appearance Model loaded\n')
            except:
                print('Unable to load the pretrained network for app_net')
                traceback.print_exc()
                exit(2)
            #exit()
            try:
                #besides the variables, the optimizer also need to be initialized.
                #fin_sess.run(tf.variables_initializer(var_list=fint_G.get_collection(name='trainable_variables')))
                fin_sess.run(tf.variables_initializer(var_list=fint_G.get_collection(name='variables')))
                saver = tf.train.Saver()
                print('\nFine-tuning network variables initialized.')
            except:
                print('Unable to initialize Fine-tuning network variables')
                traceback.print_exc()
                exit(3)
                
            '''MODULE6 ENDS---------------------------------------------------------------------------------------------'''

        newmodelname=model_save_path.split('.ckpt')[0]+'_'+str(loss_a[0].minimun_loss)+'_.ckpt'
        if os.path.exists(model_save_path+'.data-00000-of-00001'):
            os.rename((model_save_path+'.data-00000-of-00001'),(newmodelname+'.data-00000-of-00001'))
            os.rename((model_save_path+'.index'),(newmodelname+'.index'))
            os.rename((model_save_path+'.meta'),(newmodelname+'.meta'))
        newmodelname_second=model_save_path_second.split('.ckpt')[0]+'_'+str(loss_a[0].second_minimun_loss)+'_.ckpt'
        if os.path.exists(model_save_path_second+'.data-00000-of-00001'):
            os.rename((model_save_path_second+'.data-00000-of-00001'),(newmodelname_second+'.data-00000-of-00001'))
            os.rename((model_save_path_second+'.index'),(newmodelname_second+'.index'))
            os.rename((model_save_path_second+'.meta'),(newmodelname_second+'.meta'))    

        tt=time.time()
        log=log.replace('.txt',('_'+type(optm).__name__+'.txt'))
        filelog=open(log,'a')
        filelog.write('%s\t\t TotalTimeConsumed: %f\tOptimizer: %s\n'%(file_record, (tt-t1), str(type(optm).__name__)))
        filelog.close()
        if not Module==7:
            print(log)
            print(log.split('.txt')[0])
            for log_index in range(pkl_test_num):
                losslog=log.split('.txt')[0]+'_Runs%d'%(runs)+'_T%d'%(log_index+1)+'.validationlosslist'
                losslog=losslog.replace('./logs/','./logs/VL/')
                loss_a[log_index].outputlosslist(losslog)
    except:
        try:
            tt=time.time()
            log=log.replace('.txt',('_'+str(type(optm).__name__)+'.txt'))
            filelog=open(log,'a')
            filelog.write('%s\t\t TotalTimeConsumed: %f\tOptimizer: %s\n'%(file_record, (tt-t1), str(type(optm).__name__)))
            filelog.close()
            print('\n\n>>>>>> Saving current run info after it crrupted or interrupted.\n\n')
            if not Module==7:
                print(log)
                print(log.split('.txt')[0])
                for log_index in range(pkl_test_num):
                    losslog=log.split('.txt')[0]+'_Runs%d'%(runs)+'_T%d'%(log_index+1)+'.validationlosslist'
                    losslog=losslog.replace('./logs/','./logs/VL/')
                    loss_a[log_index].outputlosslist(losslog)
            print('>>>>>> Current run info has been saved after it crrupted or interrupted.\n\n')
        except:
            print('ERROR: Fail to save current run info. after it crrupted')

        ferror=open(errorlog,'w')
        traceback.print_exc()
        traceback.print_exc(file=ferror)
        ferror.close()