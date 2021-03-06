#Here starts the depression estimation processes
##from 2017.09.26, the stop criteria of training is changing to 
##
##
import numpy as np
import tensorflow as tf
import os, pickle, time, sys, traceback
import DataSetPrepare

lr_step=0.8
c_i_c_loss_error=0.0000002
batchsize_step=0
show_it=100
test_bat=100
TestNumLimit = 200
Mini_Epochs = 60
ANALOG=False
Show_Summaries=False
summary_dir='summaries'
logtem=True

class LOSS_ANA:
    '''The LOSS_ANA class collects the training losses and analyzes them.
        The initial length should be divided by 50 with no remainder.'''
    def __init__(self, LastN=50, gradient_intervals=5):
        self.__Length = LastN * 2#length of the loss list
        self.__Half_Length = LastN#half length of the loss list
        self.__Validation_Loss_List = []
        self.__Current_Length = 0#indicates whether the Validation_Loss_List has reach the maximum Length
        self.__Min_Loss = 10000.0
        
        ###
        self.__Firsthalf_Std = 0.0
        self.__Secondhalf_Std = 0.0
        self.__Firsthalf_Mean = 0.0
        self.__Secondhalf_Mean = 0.0
        self.__Mean_Variance = 0.0
        self.__Last_Firsthalf_Std = 0.0
        self.__Last_Secondhalf_Std = 0.0
        self.__Last_Firsthalf_Mean = 0.0
        self.__Last_Secondhalf_Mean = 0.0
        self.__Last_Mean_Variance = 0.0
        self.__Intervals = 0

        ####parameters for interval gradients
        self.__gradient_interval_number = gradient_intervals
        self.__gradient_interval = self.__Length//self.__gradient_interval_number
        self.__Interval_Gradients = []
        self.__Last_Interval_Gradients = []

    @property
    def minimun_loss(self):
        return self.__Min_Loss
    @property
    def Interval(self):
        return self.__Intervals
    @property
    def first_half_std(self):
        return self.__Firsthalf_Std
    @property
    def first_half_mean(self):
        return self.__Firsthalf_Mean
    @property
    def second_half_std(self):
        return self.__Secondhalf_Std
    @property
    def second_half_mean(self):
        return self.__Secondhalf_Mean
    @property
    def mean_variance(self):
        return self.__Mean_Variance
    @property
    def last_first_half_std(self):
        return self.__Last_Firsthalf_Std
    @property
    def last_first_half_mean(self):
        return self.__Last_Firsthalf_Mean
    @property
    def last_second_half_std(self):
        return self.__Last_Secondhalf_Std
    @property
    def last_second_half_mean(self):
        return self.__Last_Secondhalf_Mean
    @property
    def last_mean_variance(self):
        return self.__Last_Mean_Variance
    @property
    def interval_gradients(self):
        return self.__Interval_Gradients
    @property
    def last_interval_gradients(self):
        return self.__Last_Interval_Gradients

    def __calculate_Interval_Gradients(self):
        inter1 = np.mean(self.__Validation_Loss_List[0:self.__gradient_interval])
        inter2 = np.mean(self.__Validation_Loss_List[self.__gradient_interval:(2*self.__gradient_interval)])
        inter3 = np.mean(self.__Validation_Loss_List[(2*self.__gradient_interval):(3*self.__gradient_interval)])
        inter4 = np.mean(self.__Validation_Loss_List[(3*self.__gradient_interval):(4*self.__gradient_interval)])
        inter5 = np.mean(self.__Validation_Loss_List[(4*self.__gradient_interval):self.__Length])
        return np.array([inter2-inter1, inter3-inter2, inter4-inter3, inter5-inter4])

    def show(self):
        print('INT: %010d LAST>> FHM: %0.6f SHM: %.6f FHS: %.6f SHS: %.6f MV: %.6f IG: %s'%
              (self.__Intervals, self.__Last_Firsthalf_Mean, self.__Last_Secondhalf_Mean, self.__Last_Firsthalf_Std, self.__Last_Secondhalf_Std,
               self.__Last_Mean_Variance, str(self.__Last_Interval_Gradients)))
        print('MIL: %.8f CERT>> FHM: %0.6f SHM: %.6f FHS: %.6f SHS: %.6f MV: %.6f IG: %s'%
              (self.__Min_Loss, self.__Firsthalf_Mean, self.__Secondhalf_Mean, self.__Firsthalf_Std, self.__Secondhalf_Std,
               self.__Mean_Variance, str(self.__Interval_Gradients)))
        return True

    def log(self, file_handle):
        file_handle.write('INT: %010d LAST>> FHM: %0.6f SHM: %.6f FHS: %.6f SHS: %.6f MV: %.6f IG: %s\n'%
              (self.__Intervals, self.__Last_Firsthalf_Mean, self.__Last_Secondhalf_Mean, self.__Last_Firsthalf_Std, self.__Last_Secondhalf_Std,
               self.__Last_Mean_Variance, str(self.__Last_Interval_Gradients)))
        file_handle.write('MIL: %.8f CERT>> FHM: %0.6f SHM: %.6f FHS: %.6f SHS: %.6f MV: %.6f IG: %s\n'%
              (self.__Min_Loss, self.__Firsthalf_Mean, self.__Secondhalf_Mean, self.__Firsthalf_Std, self.__Secondhalf_Std,
               self.__Mean_Variance, str(self.__Interval_Gradients)))
        return True

    def analyzeLossVariation(self, loss):
        '''Analize the LastN*2 validation losses, where LastN is defined in __init__
        Inputs:
            loss: float type, the current loss of the validation set
            
        Outputs:
            boolean type: indicates whether the Validation_Loss_List's length has reach Length
        '''
        self.__Current_Length = self.__Current_Length + 1
        showf=False
        flag=False
        if loss < self.__Min_Loss:
            self.__Min_Loss = loss
            flag=True
        if ANALOG:
            ###update the Validation_Loss_List while keeping the Length
            if self.__Current_Length > self.__Length:
                if (self.__Current_Length-1)%self.__Length == 0:
                    self.__Last_Firsthalf_Mean = self.__Firsthalf_Mean
                    self.__Last_Firsthalf_Std = self.__Firsthalf_Std
                    self.__Last_Secondhalf_Mean = self.__Secondhalf_Mean
                    self.__Last_Secondhalf_Std = self.__Secondhalf_Std
                    self.__Last_Mean_Variance = self.__Mean_Variance
                    self.__Last_Interval_Gradients = self.__Interval_Gradients
            
                self.__Validation_Loss_List.remove(self.__Validation_Loss_List[0])
                self.__Validation_Loss_List.append(loss)
                self.__Intervals = self.__Current_Length//self.__Length
            
                firsthalf = np.asarray(self.__Validation_Loss_List[0:self.__Half_Length])
                lasthalf = np.asarray(self.__Validation_Loss_List[self.__Half_Length:self.__Length])
                self.__Firsthalf_Mean = np.mean(firsthalf)
                self.__Firsthalf_Std = np.std(firsthalf)
                self.__Secondhalf_Mean = np.mean(lasthalf)
                self.__Secondhalf_Std = np.std(lasthalf)
                self.__Mean_Variance = self.__Secondhalf_Mean - self.__Firsthalf_Mean
                self.__Interval_Gradients = self.__calculate_Interval_Gradients()
                showf=True
            ##appending the Validation_Loss_List
            elif self.__Current_Length < self.__Length:
                self.__Validation_Loss_List.append(loss)
            else:
                self.__Validation_Loss_List.append(loss)
                self.__Intervals = self.__Current_Length//self.__Length
            
                firsthalf = np.asarray(self.__Validation_Loss_List[0:self.__Half_Length])
                lasthalf = np.asarray(self.__Validation_Loss_List[self.__Half_Length:self.__Length])
                self.__Firsthalf_Mean = np.mean(firsthalf)
                self.__Firsthalf_Std = np.std(firsthalf)
                self.__Secondhalf_Mean = np.mean(lasthalf)
                self.__Secondhalf_Std = np.std(lasthalf)
                self.__Mean_Variance = self.__Secondhalf_Mean - self.__Firsthalf_Mean
                self.__Interval_Gradients = self.__calculate_Interval_Gradients()
                showf=True
        return (showf, flag)

#def logfile(file_handle, runs, OAA, valid_loss, valid_min_loss, final_train_loss, train_min_loss, FHS, SHS, MV, VA, TA, TC, ILR, FLR, LS, ites, Epo, cBS, iBS, input, CM, T, df):
#    file_handle.write("Run%02d\tOverAllACC:%0.8f\tValidationAccuracy:%.8f\tTestAccuracy:%.8f\tFinalLoss:%.10f\tMinimunLoss:%.10f\tFinaltrainloss:%.10f\tMinimumtrainloss:%.10f\tFirsthalfStd:%.10f\tSecondhalfStd:%.10f\tMeanVariance:%.10f\tTimeComsumed:%08.6f\tInitialLearningRate:%.8f\tFinalLearningRate:%.8f\tLearningStepForDroppingMagnitude:%08d\tTotalIterations:%08d\tEpoches:%08d\tcurrentBatchSize:%05d\tinitialBatchSize:%05d\tInput:%s\t%s\tTime:%s\tDataFile:%s\n"%
#                      (runs, OAA, VA, TA, valid_loss, valid_min_loss, final_train_loss, train_min_loss, FHS, SHS, MV, TC, ILR,FLR, LS,ites,Epo,cBS,iBS,str(input),str(CM),time.strftime('%Y%m%d%H%M%S',T),df))
def logfile(file_record, runs, OAA, valid_loss, valid_min_loss, final_train_loss, train_min_loss, FHS, SHS, MV, VA, TA, TC, ILR, FLR, LS, ites, Epo, cBS, iBS, input, CM, T, df):
    file_record="Run%02d\tOverAllACC:%0.8f\tValidationAccuracy:%.8f\tTestAccuracy:%.8f\tFinalLoss:%.10f\tMinimunLoss:%.10f\tFinaltrainloss:%.10f\tMinimumtrainloss:%.10f\tFirsthalfStd:%.10f\tSecondhalfStd:%.10f\tMeanVariance:%.10f\tTimeComsumed:%08.6f\tInitialLearningRate:%.8f\tFinalLearningRate:%.8f\tLearningStepForDroppingMagnitude:%08d\tTotalIterations:%08d\tEpoches:%08d\tcurrentBatchSize:%05d\tinitialBatchSize:%05d\tInput:%s\t%s\tTime:%s\tDataFile:%s\n"%(runs, 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                              OAA, VA, TA, valid_loss, valid_min_loss, final_train_loss, train_min_loss, FHS, SHS, MV, TC, ILR,FLR, LS,ites,Epo,cBS,iBS,str(input),str(CM),time.strftime('%Y%m%d%H%M%S',T),df)
    return file_record                
def updateLR(cLR):
    if cLR>0.00001:
        cLR=cLR*lr_step
    return cLR
def updateBS(bS):
    bS=bS+batchsize_step
    return bS

def selectGeoModelPathForModule6_8G(TestID=-1):
    #network type of Geo must be N4
    if TestID==0:
        mp = './networkmodel/D5_M2_N4_T0_V0_R1_20170726125411.ckpt'
    elif TestID==1:
        mp = './networkmodel/D5_M2_N4_T1_V1_R2_20170726131500.ckpt'
    elif TestID==2:
        mp = './networkmodel/D5_M2_N4_T2_V2_R0_20170726132527.ckpt'
    elif TestID==3:
        mp = './networkmodel/D5_M2_N4_T3_V3_R3_20170720202422.ckpt'
    elif TestID==4:
        mp = './networkmodel/D5_M2_N4_T4_V4_R3_20170720203232.ckpt'
    elif TestID==5:
        mp = './networkmodel/D5_M2_N4_T5_V5_R0_20170720203537.ckpt'
    elif TestID==6:
        mp = './networkmodel/D5_M2_N4_T6_V6_R0_20170726134254.ckpt'
    elif TestID==7:
        mp = './networkmodel/D5_M2_N4_T7_V7_R2_20170726140721.ckpt'
    else:
        print('Unexpected TestID encount.')
        return exit(5)
    print('Loading model for Geo......\nModel from: %s'%(mp))
    return mp
def selectAppModelPathForModule6_8G(TestID=-1):
    #network type of App must be N4
    if TestID==0:#0.98069498
        mp = './networkmodel/D5_M1_N4_T0_V0_R4_20170726233725.ckpt'
    elif TestID==1:#0.85852257
        mp = './networkmodel/D5_M1_N4_T1_V1_R3_20170727023007.ckpt'
    elif TestID==2:#0.84218559
        mp = './networkmodel/D5_M1_N4_T2_V2_R2_20170727052217.ckpt'
    elif TestID==3:#0.93217893
        mp = './networkmodel/D5_M1_N4_T3_V3_R3_20170727094147.ckpt'
    elif TestID==4:#0.95280830	
        mp = './networkmodel/D5_M1_N4_T4_V4_R3_20170720172458.ckpt'
    elif TestID==5:#0.96071429
        mp = './networkmodel/D5_M1_N4_T5_V5_R1_20170720195004.ckpt'
    elif TestID==6:#0.89166389
        mp = './networkmodel/D5_M1_N4_T6_V6_R4_20170721021127.ckpt'
    elif TestID==7:#0.84931973
        mp = './networkmodel/D5_M1_N4_T7_V7_R4_20170721060140.ckpt'
    else:
        print('Unexpected TestID encount.')
        return exit(5)
    print('Loading model for App......\nModel from: %s'%(mp))
    return mp

def restorefacepatchModel(TrainID, sess, saver):
    mp=''
    if TrainID%100>30:
        saver.restore(sess, './FPPTM/EyePatch_TrainonD502_TestonD531_N4_R2_20171017210026.ckpt')#TestAccuracy:0.60510204	FinalLoss:1.5593829036
        saver.restore(sess, './FPPTM/MiddlePatch_TrainonD502_TestonD531_N4_R12_20171018061740.ckpt')#TestAccuracy:0.47959184	FinalLoss:1.6764027119
        saver.restore(sess, './FPPTM/MouthPatch_TrainonD502_TestonD531_N4_R14_20171018053755.ckpt')#TestAccuracy:0.59285714	FinalLoss:1.5720846534
    elif TrainID%100<20:
        saver.restore(sess, './FPPTM/')
        saver.restore(sess, './FPPTM/')
        saver.restore(sess, './FPPTM/')
    else:
        print('Unexpected case occurred when loading pretrain model in restorefacepatchModel')
        exit(-1)
def calR(predict_labels, groundtruth_labels):
    assert len(predict_labels)==len(groundtruth_labels), ('predict_labels length: %d groundtruth_labels length: %d' % (len(predict_labels), len(groundtruth_labels)))
    nc=len(groundtruth_labels)
    g_c=np.zeros([7])
    confusion_mat=[[0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0]]
    for i in range(nc):
        cmi=list(groundtruth_labels[i]).index(max(groundtruth_labels[i]))
        g_c[cmi]=g_c[cmi]+1
        pri=list(predict_labels[i]).index(max(predict_labels[i]))
        confusion_mat[cmi][pri]=confusion_mat[cmi][pri]+1
    for i in range(len(g_c)):
        if g_c[i]>0:
            confusion_mat[i]=list(np.asarray(confusion_mat[i])/g_c[i])
    return confusion_mat

def overAllAccuracy(conf_m):
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
    ac=ac/r
    return ac

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
    elif module==5:
        if NetworkType<2:
            try:
                print("Loading pretrained network model: VGGFACE.npy......")
                network.load('./networkmodel/VGGFACE.npy', session, ignore_missing=True)
                print('\nPreserved Model of VGGFACE was loaded.\n')
            except:
                print('ERROR: unable to load pretrained VGGFACE network weights')
                traceback.print_exc()
                exit(-1)
        else:
            print('No pretrain network weights are fit to the current network type. Please try another network type.')
            exit()
    else:
        print('Module %d has no pretrained model embedded. Please try another module or check the input again.'%(module))
        exit()

def getModelPath(mpprefix, tt, loss, reshape, best=False):
    loss_str='%.10f'%loss
    if reshape:
        mpprefix=mpprefix+'_reshape64x64_'
    if best:
        mp = mpprefix + time.strftime('%Y%m%d%H%M%S_',time.localtime(tt))+loss_str+"_.ckpt"
    else:
        mp = mpprefix + time.strftime('%Y%m%d%H%M%S_',time.localtime(tt))+loss_str+"_not_optimal.ckpt"
    return mp

def Eva_on_TestSet_M1(data, TestNumLimit, test_bat, 
                   sess, accuracy, sum_test, pred, images, labels):
    tlabels=[]
    ncount=data.test.num_examples
    if ncount > TestNumLimit:
        data.test.resetIndex()
        test_iter=np.floor_divide(ncount,test_bat)
        ac=0
        for ite in range(test_iter):
            bat_test=data.test.next_batch(test_bat, shuffle=False)
            tac, tlab=sess.run([sum_test, pred], feed_dict={images:bat_test[0], labels:bat_test[5]})
            ac=ac+tac
            tlabels.extend(list(tlab))
        tac, tlab=sess.run([sum_test, pred], feed_dict={images:data.test.res_images[test_bat*test_iter:ncount], 
                                                        labels:data.test.labels[test_bat*test_iter:ncount]})
        ac=ac+tac
        tlabels.extend(list(tlab))
        ac=ac/ncount
    else:
        ac, tlab=sess.run([accuracy, pred], feed_dict={images:data.test.res_images, labels:data.test.labels})
        tlabels.extend(list(tlab))
    confu_mat=calR(tlabels,data.test.labels)
    oaa=overAllAccuracy(confu_mat)
    return ac, oaa, confu_mat
def Eva_on_ValidSet_M1(data, TestNumLimit, test_bat,
                       sess, accuracy, sum_test, loss, images, labels):
    ncount=data.validation.num_examples
    if ncount>TestNumLimit:                            
        data.validation.resetIndex()
        test_iter=np.floor_divide(ncount,test_bat)
        v_accuracy=0
        valid_loss=0
        for ite in range(test_iter):
            bat_test=data.validation.next_batch(test_bat, shuffle=False)
            st, v_loss=sess.run([sum_test, loss], feed_dict={images:bat_test[0], labels:bat_test[5]})
            v_accuracy=v_accuracy+st
            valid_loss=valid_loss+v_loss
        st, v_loss=sess.run([sum_test, loss], feed_dict={images:data.validation.res_images[test_bat*test_iter:ncount], 
                                                        labels:data.validation.labels[test_bat*test_iter:ncount]})
        v_accuracy=v_accuracy+st
        valid_loss=valid_loss+v_loss
        v_accuracy=v_accuracy/ncount
        valid_loss=valid_loss/(test_iter+1)
    else:
        v_accuracy, valid_loss = sess.run([accuracy, loss], feed_dict={images:data.validation.res_images, labels:data.validation.labels})
    return v_accuracy, valid_loss
def Eva_on_TestSet_M2(data, sess, accuracy, pred, geo_features, labels):
    tlabels=[]
    ac, tlab=sess.run([accuracy, pred], feed_dict={geometry:data.test.geometry, labels:data.test.labels})
    tlabels.extend(list(tlab))
    confu_mat=calR(tlabels,data.test.labels)
    oaa=overAllAccuracy(confu_mat)
    return ac, oaa, confu_mat
def Eva_on_TestSet_M3(data, TestNumLimit, test_bat,
                      sess, accuracy, sum_test, pred, eye_p, midd_p, mou_p, labels):
    tlabels=[]
    ncount=data.test.num_examples
    if ncount>TestNumLimit:
        data.test.resetIndex()
        test_iter=np.floor_divide(ncount, test_bat)
        ac=0
        for ite in range(test_iter):
            bat_test=data.test.next_batch(test_bat, shuffle=False)
            tac, tlab=sess.run([sum_test, pred], feed_dict={eye_p:bat_test[2], midd_p:bat_test[3], mou_p:bat_test[4], labels:bat_test[5]})
            ac=ac+tac
            tlabels.extend(list(tlab))
        tac, tlab=sess.run([sum_test, pred], feed_dict={eye_p:data.test.eyep[test_bat*test_iter:ncount],
                                                        midd_p:data.test.middlep[test_bat*test_iter:ncount],
                                                        mou_p:data.test.mouthp[test_bat*test_iter:ncount],
                                                        labels:data.test.labels[test_bat*test_iter:ncount]})
        ac=ac+tac
        tlabels.extend(list(tlab))
        ac=ac/ncount
    else:
        ac, tlabels=sess.run([accuracy, pred],feed_dict={eye_p:data.test.eyep, 
                                                    midd_p:data.test.middlep, 
                                                    mou_p:data.test.mouthp, 
                                                    labels:data.test.labels})
    confu_mat=calR(tlabels,data.test.labels)
    oaa=overAllAccuracy(confu_mat)
    return ac, oaa, confu_mat
def Eva_on_ValidSet_M3(data, TestNumLimit, test_bat,
                       sess, accuracy, sum_test, loss, eye_p, midd_p, mou_p, labels):
    ncount=data.validation.num_examples
    if ncount>TestNumLimit:                            
        data.validation.resetIndex()
        test_iter=np.floor_divide(ncount,test_bat)
        v_accuracy=0
        valid_loss=0
        for ite in range(test_iter):
            bat_test=data.validation.next_batch(test_bat, shuffle=False)
            st, v_loss=sess.run([sum_test, loss], feed_dict={eye_p:bat_test[2], midd_p:bat_test[3], mou_p:bat_test[4], labels:bat_test[5]})
            v_accuracy=v_accuracy+st
            valid_loss=valid_loss+v_loss
        st, v_loss=sess.run([sum_test, loss], feed_dict={eye_p:data.validation.eyep[test_bat*test_iter:ncount], 
                                                        midd_p:data.validation.middlep[test_bat*test_iter:ncount],
                                                        mou_p:data.validation.mouthp[test_bat*test_iter:ncount],
                                                        labels:data.validation.labels[test_bat*test_iter:ncount]})
        v_accuracy=v_accuracy+st
        valid_loss=valid_loss+v_loss
        v_accuracy=v_accuracy/ncount
        valid_loss=valid_loss/(test_iter+1)
    else:
        v_accuracy, valid_loss = sess.run([accuracy, loss], feed_dict={eye_p:data.validation.eyep, 
                                                        midd_p:data.validation.middlep,
                                                        mou_p:data.validation.mouthp,
                                                        labels:data.validation.labels})
    return v_accuracy, valid_loss

def run(GPU_Device_ID, Module, 
        DataSet,ValidID,TestID, 
        NetworkType, runs
        ,learningRate=0.0001,IterationsC=4001,lrstep=3000,batchSize=15,saveM=True,log=True,loadONW=True,reshape=False):
    global Mini_Epochs
    c_i_c=30000
    MLTT=1
    file_record=None
    #if runs%2==0:
    #    saveM=False
    try:
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
        #currentLR=0.0005#start with 0.01 is not an option for it's poor performance in reducing the loss for RMSPropOptimizer
        currentLR=learningRate
        currentBatchSize=batchSize
        #IterationsC=5001
        #lrstep=2000#every x iterations the learning rate will drop an order of magnitude, when x=IterationsC it means learningRate will not change
        #currentBatchSize=30


        #
        #
        #
        '''Input Data-------------------------------------------------------------------------------------------------
        -------------------------------------------------------------------------------------------------------------'''
        t1=time.time()
        labelshape=[None, 7]
        mini_loss=10000
        logprefix='./logs/'
        D_f=False
        m2d=122
        if Module==2:
            if NetworkType==0 or NetworkType==1 or NetworkType==2:
                D_f=True
        #
        ##data set loading
        #
        if DataSet==2:
            dfile='./Datasets/D2_CKplus_8groups_groupedbythe_CKplus-group-details_preprocessdata_with_calibRotation_rescaleimgnewmetric0731_skip-contempV2.pkl'
            data = DataSetPrepare.loadCKplus10gdata_v2(dfile, ValidID, TestID, Df=D_f,reshape=reshape)
            m1shape= [None, 128, 128, 1]
            if log:
                logprefix="./logs/D2CKplus_newrescalemetric_8groups_gpu"
                print("Processing 8 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==3:
            dfile='./Datasets/D3_CKplus_8groups_groupedbythe_CKplus-group-details_preprocessdata_with_calibRotation_rescaleimg_geometricfeatures_facepatches_webberface_skip-contempV2.pkl'
            data = DataSetPrepare.loadCKplus10gdata_v2(dfile, ValidID, TestID, Df=D_f,reshape=reshape)
            m1shape= [None, 128, 128, 1]
            if log:
                logprefix="./logs/D3CKpluslogbslr_weberface_8groups_gpu"
                print("Processing 8 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==4:
            dfile='./Datasets/D4_CKplus_8groups_groupedbythe_CKplus-group-details_preprocessdata_with_calibRotation_rescaleimg_geometricfeatures_facepatches_webberfaceReverse_skip-contempV2.pkl'
            data = DataSetPrepare.loadCKplus10gdata_v2(dfile, ValidID, TestID, Df=D_f,reshape=reshape)
            m1shape= [None, 128, 128, 1]
            if log:
                logprefix="./logs/D4CKpluslogbslr_weberReverse_8groups_gpu"
                print("Processing 8 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==5:
            dfile='./Datasets/D5_CKplus_8groups_groupedbythe_CKplus-group-details_preprocessdata_with_calibRotation_rescaleimg_geometricfeatures_facepatches_webberface25up_skip-contempV2.pkl'
            data = DataSetPrepare.loadCKplus10gdata_v2(dfile, ValidID, TestID, Df=D_f,reshape=reshape)
            m1shape= [None, 128, 128, 1]
            if log:
                logprefix="./logs/D5CKpluslogbslr_weberface25up_8groups_gpu"
                print("Processing 8 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==6:
            dfile='./Datasets/D6_CKplus_8groups_groupedbythe_CKplus-group-details_preprocessdata_with_calibRotation_rescaleimg_geometricfeatureV2_skip-contempV2.pkl'
            data = DataSetPrepare.loadCKplus10gdata_v2(dfile, ValidID, TestID, Df=D_f,reshape=reshape)
            m1shape= [None, 128, 128, 1]
            m2d=258
            if log:
                logprefix="./logs/D6CKplus_GeoFeatureV2_8groups_gpu"
                print("Processing 8 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==7:
            dfile='./Datasets/D7_CKplus_8groups_groupedbythe_CKplus-group-details_preprocessdata_with_calibRotation_rescaleimg_geometricfeatures_facepatches_webberface_innerfaceSizew36xh48_skip-contempV2.pkl'
            data = DataSetPrepare.loadCKplus10gdata_v2(dfile, ValidID, TestID, Df=D_f,reshape=reshape)
            m1shape= [None, 128, 128, 1]
            if log:
                logprefix="./logs/D7CKpluslogbslr_weberface_innerface48x36_8groups_gpu"
                print("Processing 8 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==8:
            dfile='./Datasets/D8_CKplus_8groups_groupedbythe_CKplus-group-details_preprocessdata_with_calibRotation_rescaleimg_geometricfeatures_facepatches_ELTFS_skip-contempV2.pkl'
            data = DataSetPrepare.loadCKplus10gdata_v2(dfile, ValidID, TestID, Df=D_f,reshape=reshape)
            m1shape= [None, 128, 128, 1]
            if log:
                logprefix="./logs/D8CKpluslogbslr_ELTFS_8groups_gpu"
                print("Processing 8 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==9:
            dfile='./Datasets/D9_CKplus_8groups_groupedbythe_CKplus-group-details_preprocessdata_with_calibRotation_rescaleimg224x224_geometricfeatures_facepatches_weberface224x224_skip-contempV2.pkl'
            data = DataSetPrepare.loadCKplus10gdata_v2(dfile, ValidID, TestID, Df=D_f,reshape=reshape)
            m1shape= [None, 224, 224, 1]
            if log:
                logprefix="./logs/D9CKpluslogbslr_weberface224_8groups_gpu"
                print("Processing 8 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==10:
            dfile='./Datasets/D10_CKplus_10groups_groupedbythe_CKplus-group-details_preprocessdata_with_calibRotation_rescaleimg_geometricfeatures_facepatches_weberface_skip-contempV2.pkl'
            data = DataSetPrepare.loadCKplus10gdata_v2(dfile, ValidID, TestID, Df=D_f,reshape=reshape)
            m1shape= [None, 128, 128, 1]
            if log:
                logprefix="./logs/D10CKpluslogbslr_weberface_10groups_gpu"
                print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==11:
            dfile='./Datasets/D11_CKplus_10groups_groupedbythe_CKplus-group-details_preprocessdata_with_calibRotation_rescaleimg224x224_geometricfeatures_facepatches_weberface224x224_skip-contempV2.pkl'
            data = DataSetPrepare.loadCKplus10gdata_v2(dfile, ValidID, TestID, Df=D_f,reshape=reshape)
            m1shape= [None, 224, 224, 1]
            if log:
                logprefix="./logs/D11CKpluslogbslr_weberface224_10groups_gpu"
                print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==12:
            dfile='./Datasets/D12_CKplus_10groups_groupedbythe_CKplus-group-details_preprocessdata_with_calibRotation_rescaleimg_geometricfeatures_facepatches_ELTFS_skip-contempV2.pkl'
            data = DataSetPrepare.loadCKplus10gdata_v2(dfile, ValidID, TestID, Df=D_f,reshape=reshape)
            m1shape= [None, 128, 128, 1]
            if log:
                logprefix="./logs/D12CKpluslogbslr_ELTFS_10groups_gpu"
                print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==13:
            dfile='./Datasets/D13_CKplus_8G_V4_Geo258_ELTFS128x128.pkl'
            data = DataSetPrepare.loadCKplus10gdata_v2(dfile, ValidID, TestID, Df=D_f,reshape=reshape)
            m1shape= [None, 128, 128, 1]
            m2d=258
            if log:
                logprefix="./logs/D13_CKplus_8G_V4_Geo258_ELTFS128x128_gpu"
                print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==15:
            dfile='./Datasets/D15_CKPLUS_10G_Enlargeby_webface_ELTFS_flip.pkl'
            data = DataSetPrepare.loadCKplus10gdata_v2(dfile, ValidID, TestID, Df=D_f,reshape=reshape)
            dfilet='./Datasets/D10_CKplus_10groups_groupedbythe_CKplus-group-details_preprocessdata_with_calibRotation_rescaleimg_geometricfeatures_facepatches_weberface_skip-contempV2.pkl'
            datatest = DataSetPrepare.loadCKplus10gdata_v2(dfilet, ValidID, TestID, Df=D_f,reshape=reshape)
            print('Before reset: %d'%data.test.num_examples)
            data.test.reset(datatest.test.res_images, datatest.test.geometry, 
                            datatest.test.eyep, datatest.test.middlep, datatest.test.mouthp, datatest.test.innerf,
                            datatest.test.labels)
            data.validation.reset(datatest.validation.res_images, datatest.validation.geometry, 
                            datatest.validation.eyep, datatest.validation.middlep, datatest.validation.mouthp, datatest.validation.innerf,
                            datatest.validation.labels)
            print('After reset: %d'%data.test.num_examples)
            m1shape= [None, 128, 128, 1]
            batchSize=60
            currentBatchSize=batchSize
            if log:
                logprefix="./logs/D15_CKPLUS_10G_EnlargebyWEF_testonoriginal_gpu"
                print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==16:
            dfile='./Datasets/D16_CKPLUS_10G_Enlargeby2015CCV_10T.pkl'
            data = DataSetPrepare.loadCKplus10gdata_v2(dfile, ValidID, TestID, Df=D_f,reshape=reshape)
            dfilet='./Datasets/D10_CKplus_10groups_groupedbythe_CKplus-group-details_preprocessdata_with_calibRotation_rescaleimg_geometricfeatures_facepatches_weberface_skip-contempV2.pkl'
            datatest = DataSetPrepare.loadCKplus10gdata_v2(dfilet, ValidID, TestID, Df=D_f,reshape=reshape)
            print('Before reset: %d'%data.test.num_examples)
            data.test.reset(datatest.test.res_images, datatest.test.geometry, 
                            datatest.test.eyep, datatest.test.middlep, datatest.test.mouthp, datatest.test.innerf,
                            datatest.test.labels)
            data.validation.reset(datatest.validation.res_images, datatest.validation.geometry, 
                            datatest.validation.eyep, datatest.validation.middlep, datatest.validation.mouthp, datatest.validation.innerf,
                            datatest.validation.labels)
            print('After reset: %d'%data.test.num_examples)
            m1shape= [None, 128, 128, 1]
            if runs%2==0:
                batchSize=30
            else:
                batchSize=15
                IterationsC=IterationsC*2
                c_i_c=c_i_c*2
            currentBatchSize=batchSize
            if log:
                logprefix="./logs/D16_CKPLUS_10G_Enlargeby2015CCV_10T_testonoriginal_gpu"
                print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==17:
            dfile='./Datasets/D17_CKplus_10G_V4_weberface128x128.pkl'
            data = DataSetPrepare.loadCKplus10gdata_v2(dfile, ValidID, TestID, Df=D_f,reshape=reshape)
            m1shape= [None, 128, 128, 1]
            if log:
                logprefix="./logs/D17_CKplus_10G_V4_weberface128x128_gpu"
                print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==18:
            dfile='./Datasets/D18_CKplus_10G_V5_formalized_weberface128x128.pkl'
            data = DataSetPrepare.loadCKplus10gdata_v2(dfile, ValidID, TestID, Df=D_f,reshape=reshape)
            m1shape= [None, 128, 128, 1]
            if log:
                logprefix="./logs/D18_CKplus_10G_V5_formalized_weberface128x128_gpu"
                print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==19:
            dfile='./Datasets/D19_CKplus_10G_V4_ELTFS128x128.pkl'
            data = DataSetPrepare.loadCKplus10gdata_v2(dfile, ValidID, TestID, Df=D_f,reshape=reshape)
            m1shape= [None, 128, 128, 1]
            if log:
                logprefix="./logs/D19_CKplus_10G_V4_ELTFS128x128_gpu"
                print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==33:
            dfile='./Datasets/D33_KDEF_10G_rescaleimg_geometryfeature_patches_web.pkl'
            data = DataSetPrepare.loadCKplus10gdata_v2(dfile, ValidID, TestID, Df=D_f,reshape=reshape)
            m1shape= [None, 128, 128, 1]
            batchSize=35
            currentBatchSize=batchSize
            if log:
                logprefix="./logs/D33_KDEF_weberface_10groups_gpu"
                print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==32:
            dfile='./Datasets/D32_KDEF_10G_Enlargeby_webface_ELTFS_flip.pkl'
            data = DataSetPrepare.loadCKplus10gdata_v2(dfile, ValidID, TestID, Df=D_f,reshape=reshape)
            dfilet='./Datasets/D33_KDEF_10G_rescaleimg_geometryfeature_patches_web.pkl'
            datatest = DataSetPrepare.loadCKplus10gdata_v2(dfilet, ValidID, TestID, Df=D_f,reshape=reshape)
            print('Before reset: %d'%data.test.num_examples)
            data.test.reset(datatest.test.res_images, datatest.test.geometry, 
                            datatest.test.eyep, datatest.test.middlep, datatest.test.mouthp, datatest.test.innerf,
                            datatest.test.labels)
            data.validation.reset(datatest.validation.res_images, datatest.validation.geometry, 
                            datatest.validation.eyep, datatest.validation.middlep, datatest.validation.mouthp, datatest.validation.innerf,
                            datatest.validation.labels)
            print('After reset: %d'%data.test.num_examples)
            m1shape= [None, 128, 128, 1]
            batchSize=70
            currentBatchSize=batchSize
            if log:
                logprefix="./logs/D32_KDEF_10G_EnlargebyWEF_testonoriginal_gpu"
                print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==34:
            dfile='./Datasets/D34_KDEF_10G_Enlargeby2015CCV_10T.pkl'
            data = DataSetPrepare.loadCKplus10gdata_v2(dfile, ValidID, TestID, Df=D_f,reshape=reshape)
            dfilet='./Datasets/D33_KDEF_10G_rescaleimg_geometryfeature_patches_web.pkl'
            datatest = DataSetPrepare.loadCKplus10gdata_v2(dfilet, ValidID, TestID, Df=D_f,reshape=reshape)
            print('Before reset: %d'%data.test.num_examples)
            data.test.reset(datatest.test.res_images, datatest.test.geometry, 
                            datatest.test.eyep, datatest.test.middlep, datatest.test.mouthp, datatest.test.innerf,
                            datatest.test.labels)
            data.validation.reset(datatest.validation.res_images, datatest.validation.geometry, 
                            datatest.validation.eyep, datatest.validation.middlep, datatest.validation.mouthp, datatest.validation.innerf,
                            datatest.validation.labels)
            print('After reset: %d'%data.test.num_examples)
            m1shape= [None, 128, 128, 1]
            batchSize=70
            currentBatchSize=batchSize
            if log:
                logprefix="./logs/D34_KDEF_10G_Enlargeby2015CCV_10T_testonoriginal_gpu"
                print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==42:
            dfile='./Datasets/D42_JAFFE_10G_Enlargeby_webface_ELTFS_flip.pkl'
            data = DataSetPrepare.loadCKplus10gdata_v2(dfile, ValidID, TestID, Df=D_f,reshape=reshape)
            dfilet='./Datasets/D40_jaffe_10groups_groupedbysubjects_rescaleimg_geometricfeatures_facepatches_weber.pkl'
            datatest = DataSetPrepare.loadCKplus10gdata_v2(dfilet, ValidID, TestID, Df=D_f,reshape=reshape)
            print('Before reset: %d'%data.test.num_examples)
            data.test.reset(datatest.test.res_images, datatest.test.geometry, 
                            datatest.test.eyep, datatest.test.middlep, datatest.test.mouthp, datatest.test.innerf,
                            datatest.test.labels)
            data.validation.reset(datatest.validation.res_images, datatest.validation.geometry, 
                            datatest.validation.eyep, datatest.validation.middlep, datatest.validation.mouthp, datatest.validation.innerf,
                            datatest.validation.labels)
            print('After reset: %d'%data.test.num_examples)
            m1shape= [None, 128, 128, 1]
            batchSize=60
            currentBatchSize=batchSize
            if log:
                logprefix="./logs/D42_JAFFE_10G_Enlargeby_WEF_testonoriginaldataset_gpu"
                print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==40:
            dfile='./Datasets/D40_jaffe_10groups_groupedbysubjects_rescaleimg_geometricfeatures_facepatches_weber.pkl'
            data = DataSetPrepare.loadCKplus10gdata_v2(dfile, ValidID, TestID, Df=D_f,reshape=reshape)
            m1shape= [None, 128, 128, 1]
            if log:
                logprefix="./logs/D40_JAFFE_10G_gpu"
                print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==43:
            dfile='./Datasets/D43_JAFFE_10G_Enlargeby2015CCV_10T.pkl'
            data = DataSetPrepare.loadCKplus10gdata_v2(dfile, ValidID, TestID, Df=D_f,reshape=reshape)
            dfilet='./Datasets/D40_jaffe_10groups_groupedbysubjects_rescaleimg_geometricfeatures_facepatches_weber.pkl'
            datatest = DataSetPrepare.loadCKplus10gdata_v2(dfilet, ValidID, TestID, Df=D_f,reshape=reshape)
            print('Before reset: %d'%data.test.num_examples)
            data.test.reset(datatest.test.res_images, datatest.test.geometry, 
                            datatest.test.eyep, datatest.test.middlep, datatest.test.mouthp, datatest.test.innerf,
                            datatest.test.labels)
            data.validation.reset(datatest.validation.res_images, datatest.validation.geometry, 
                            datatest.validation.eyep, datatest.validation.middlep, datatest.validation.mouthp, datatest.validation.innerf,
                            datatest.validation.labels)
            print('After reset: %d'%data.test.num_examples)
            m1shape= [None, 128, 128, 1]
            batchSize=60
            currentBatchSize=batchSize
            if log:
                logprefix="./logs/D43_JAFFE_10G_Enlargeby2015CCV_10T_testonoriginaldataset_gpu"
                print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==111:
            dfile='./Datasets/D111_MergeDataset_D10_D33_D40_10G.pkl'
            data = DataSetPrepare.loadCKplus10gdata_v2(dfile, ValidID, TestID, Df=D_f,reshape=reshape)
            m1shape= [None, 128, 128, 1]
            batchSize=30
            currentBatchSize=batchSize
            if log:
                logprefix="./logs/D111_MergeDataset_D10_D33_D40_10G_gpu"
                print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==222:
            dfile='./Datasets/D222_MergeDataset_D16_D34_D43_10G.pkl'
            data = DataSetPrepare.loadCKplus10gdata_v2(dfile, ValidID, TestID, Df=D_f,reshape=reshape)
            dfilet='./Datasets/D111_MergeDataset_D10_D33_D40_10G.pkl'
            datatest = DataSetPrepare.loadCKplus10gdata_v2(dfilet, ValidID, TestID, Df=D_f,reshape=reshape)
            print('Before reset: %d'%data.test.num_examples)
            data.test.reset(datatest.test.res_images, datatest.test.geometry, 
                            datatest.test.eyep, datatest.test.middlep, datatest.test.mouthp, datatest.test.innerf,
                            datatest.test.labels)
            data.validation.reset(datatest.validation.res_images, datatest.validation.geometry, 
                            datatest.validation.eyep, datatest.validation.middlep, datatest.validation.mouthp, datatest.validation.innerf,
                            datatest.validation.labels)
            print('After reset: %d'%data.test.num_examples)
            m1shape= [None, 128, 128, 1]
            batchSize=30
            currentBatchSize=batchSize
            if log:
                logprefix="./logs/D222_MergeDataset_D16_D34_D43_10G_gpu"
                print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==333:
            dfile='./Datasets/D333_MergeDataset_D16_D34_10G.pkl'
            data = DataSetPrepare.loadCKplus10gdata_v2(dfile, ValidID, TestID, Df=D_f,reshape=reshape)
            dfilet='./Datasets/D444_MergeDataset_D10_D33_10G.pkl'
            datatest = DataSetPrepare.loadCKplus10gdata_v2(dfilet, ValidID, TestID, Df=D_f,reshape=reshape)
            print('Before reset: %d'%data.test.num_examples)
            data.test.reset(datatest.test.res_images, datatest.test.geometry, 
                            datatest.test.eyep, datatest.test.middlep, datatest.test.mouthp, datatest.test.innerf,
                            datatest.test.labels)
            data.validation.reset(datatest.validation.res_images, datatest.validation.geometry, 
                            datatest.validation.eyep, datatest.validation.middlep, datatest.validation.mouthp, datatest.validation.innerf,
                            datatest.validation.labels)
            print('After reset: %d'%data.test.num_examples)
            m1shape= [None, 128, 128, 1]
            batchSize=30
            currentBatchSize=batchSize
            if log:
                logprefix="./logs/D333_MergeDataset_D16_D34_10G_gpu"
                print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==444:
            dfile='./Datasets/D444_MergeDataset_D10_D33_10G.pkl'
            data = DataSetPrepare.loadCKplus10gdata_v2(dfile, ValidID, TestID, Df=D_f,reshape=reshape)
            m1shape= [None, 128, 128, 1]
            batchSize=30
            currentBatchSize=batchSize
            if log:
                logprefix="./logs/D444_MergeDataset_D10_D33_10G_gpu"
                print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==501:
            dfile='./Datasets/D501_CKplus_10G_V5_newGeo_newPatch.pkl'
            data = DataSetPrepare.loadCKplus10gdata_v2(dfile, ValidID, TestID, Df=D_f,reshape=reshape)
            m1shape= [None, 128, 128, 1]
            if runs%2==0:
                batchSize=9
                IterationsC=IterationsC*2
                c_i_c=c_i_c*2
            else:
                batchSize=15
            if log:
                logprefix="./logs/D501_gpu"
                print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==502:
            dfile='./Datasets/D502_CKPLUS_10G_Enlargeby2015ICCV_V5_newGeo_newPatches.pkl'
            data = DataSetPrepare.loadCKplus10gdata_v2(dfile, ValidID, TestID, Df=D_f,reshape=reshape)
            dfilet='./Datasets/D501_CKplus_10G_V5_newGeo_newPatch.pkl'
            datatest = DataSetPrepare.loadCKplus10gdata_v2(dfilet, ValidID, TestID, Df=D_f,reshape=reshape)
            print('Before reset: %d'%data.test.num_examples)
            data.test.reset(datatest.test.res_images, datatest.test.geometry, 
                            datatest.test.eyep, datatest.test.middlep, datatest.test.mouthp, datatest.test.innerf,
                            datatest.test.labels)
            data.validation.reset(datatest.validation.res_images, datatest.validation.geometry, 
                            datatest.validation.eyep, datatest.validation.middlep, datatest.validation.mouthp, datatest.validation.innerf,
                            datatest.validation.labels)
            print('After reset: %d'%data.test.num_examples)
            m1shape= [None, 128, 128, 1]
            Mini_Epochs=Mini_Epochs/2
            if runs%2==0:
                batchSize=9
                IterationsC=IterationsC*2
                c_i_c=c_i_c*2
            else:
                batchSize=15
            currentBatchSize=batchSize
            if log:
                logprefix="./logs/D502_gpu"
                print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==531:
            dfile='./Datasets/D531_KDEF_10G_V5_newGeo_newPatch.pkl'
            data = DataSetPrepare.loadCKplus10gdata_v2(dfile, ValidID, TestID, Df=D_f,reshape=reshape)
            m1shape= [None, 128, 128, 1]
            if runs%2==0:
                batchSize=15
            else:
                batchSize=7
                IterationsC=IterationsC*2
                c_i_c=c_i_c*2
            if log:
                logprefix="./logs/D531_gpu"
                print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==532:
            dfile='./Datasets/D532_KDEF_10G_Enlargeby2015ICCV_V5_newGeo_newPatches.pkl'
            data = DataSetPrepare.loadCKplus10gdata_v2(dfile, ValidID, TestID, Df=D_f,reshape=reshape)
            dfilet='./Datasets/D531_KDEF_10G_V5_newGeo_newPatch.pkl'
            datatest = DataSetPrepare.loadCKplus10gdata_v2(dfilet, ValidID, TestID, Df=D_f,reshape=reshape)
            print('Before reset: %d'%data.test.num_examples)
            data.test.reset(datatest.test.res_images, datatest.test.geometry, 
                            datatest.test.eyep, datatest.test.middlep, datatest.test.mouthp, datatest.test.innerf,
                            datatest.test.labels)
            data.validation.reset(datatest.validation.res_images, datatest.validation.geometry, 
                            datatest.validation.eyep, datatest.validation.middlep, datatest.validation.mouthp, datatest.validation.innerf,
                            datatest.validation.labels)
            print('After reset: %d'%data.test.num_examples)
            m1shape= [None, 128, 128, 1]
            Mini_Epochs=Mini_Epochs/2
            if runs%2==0:
                batchSize=15
            else:
                batchSize=7
                IterationsC=IterationsC*2
                c_i_c=c_i_c*2
            currentBatchSize=batchSize
            if log:
                logprefix="./logs/D532_gpu"
                print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==1001:
            dfile='./Datasets/D1001_MergeDataset_D501_D531_10G.pkl'
            data = DataSetPrepare.loadCKplus10gdata_v2(dfile, ValidID, TestID, Df=D_f,reshape=reshape)
            m1shape= [None, 128, 128, 1]
            if runs%2==0:
                batchSize=30
            else:
                batchSize=15
                IterationsC=IterationsC*2
                c_i_c=c_i_c*2
            currentBatchSize=batchSize
            if log:
                logprefix="./logs/D1001_gpu"
                print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
        elif DataSet==1002:
            dfile='./Datasets/D1002_MergeDataset_D502_D532_10G.pkl'
            data = DataSetPrepare.loadCKplus10gdata_v2(dfile, ValidID, TestID, Df=D_f,reshape=reshape)
            dfilet='./Datasets/D1001_MergeDataset_D501_D531_10G.pkl'
            datatest = DataSetPrepare.loadCKplus10gdata_v2(dfilet, ValidID, TestID, Df=D_f,reshape=reshape)
            print('Before reset: %d'%data.test.num_examples)
            data.test.reset(datatest.test.res_images, datatest.test.geometry, 
                            datatest.test.eyep, datatest.test.middlep, datatest.test.mouthp, datatest.test.innerf,
                            datatest.test.labels)
            data.validation.reset(datatest.validation.res_images, datatest.validation.geometry, 
                            datatest.validation.eyep, datatest.validation.middlep, datatest.validation.mouthp, datatest.validation.innerf,
                            datatest.validation.labels)
            print('After reset: %d'%data.test.num_examples)
            m1shape= [None, 128, 128, 1]
            Mini_Epochs=Mini_Epochs/2
            if runs%2==0:
                batchSize=30
            else:
                batchSize=15
                IterationsC=IterationsC*2
                c_i_c=c_i_c*2
            currentBatchSize=batchSize
            if log:
                logprefix="./logs/D1002_gpu"
                print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
        else:
            print('ERROR: Unexpeted Dataset ID')
            exit()
        #
        #
        #
        tt=time.time()
        if log:
            if reshape:
                logprefix=logprefix+'_reshape64x64'
            if Module==6:
                logbslr=logprefix+str(GPU_Device_ID)+"_M"+str(Module)+"_D"+str(DataSet)+"_N"+str(NetworkType)+"_newStopCriteria.txt"
            elif loadONW:
                logbslr=logprefix+str(GPU_Device_ID)+"_M"+str(Module)+"_D"+str(DataSet)+"_N"+str(NetworkType)+"_withPretrainModelWeight_newStopCriteria.txt"
            else:
                logbslr=logprefix+str(GPU_Device_ID)+"_M"+str(Module)+"_D"+str(DataSet)+"_N"+str(NetworkType)+"_noPretrain_newStopCriteria.txt"
            #logfilename=time.strftime('%Y%m%d%H%M%S',time.localtime(tt))+str(sys.argv[2:4])
        print('Time used for loading data: %fs'%(tt-t1))
        if saveM:
            model_save_path=("J:/Models/saves/"+'D'+str(DataSet)+'_M'+str(Module)+'_N'+str(NetworkType)+'_T'+str(TestID)+'_V'+str(ValidID)+'_R'+str(runs)+'_')
        '''Input Data Ends-----------------------------------------------------------------------------------------'''
        #
        #
        #
        mini_loss_track=0
        loss_a=LOSS_ANA()
        laflag=(False, False)
        if reshape:
            m1shape=[None, 64, 64, 1]
            print('Module 1 images input shape has been set to %s'%str(m1shape))
        #
        #
        #
        if Module==1:
            stcmwvlilttv=1.1674#save_the_current_model_when_validation_loss_is_less_than_this_value
            
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
                learningRate=0.00005
                currentLR=0.00005
            elif NetworkType==4:
                from VGG_NET import VGG_NET_o as WFN
                learningRate=0.00005
                currentLR=0.00005
            elif NetworkType==8:
                from VGG_NET import VGG_NET_Inception1 as WFN
            elif NetworkType==9:
                from VGG_NET import VGG_NET_Inception2 as WFN

            elif NetworkType==30:
                from PopularNets import ResNet50 as WFN
                learningRate=0.00002
                currentLR=0.00002
                batchSize=10
                currentBatchSize=10
            elif NetworkType==31:
                from PopularNets import ResNet101 as WFN
            elif NetworkType==32:
                from PopularNets import ResNet150 as WFN
            elif NetworkType==33:
                from PopularNets import AlexNet as WFN
            elif NetworkType==34:
                from PopularNets import CaffeNet as WFN
            elif NetworkType==35:
                from PopularNets import NiN as WFN
            elif NetworkType==36:
                from PopularNets import GoogleNet as WFN
            else:
                print("Usage: python finetune.py <GPUID> <Module> <NetworkType>\nWith Module 1, NetworkType must be 0, 1, 2, 3")
                exit(-1)
            '''Here begins the implementation logic-------------------------------------------------------------------
            -------------------------------------------------------------------------------------------------------------'''
            #Holder for gray images with m1shape in a batch size of batch_size
            images = tf.placeholder(tf.float32, m1shape)
            #Holder for labels in a batch size of batch_size, number of labels are to be determined
            labels = tf.placeholder(tf.float32, labelshape)#the number of labels are to be determined

            whole_face_net = WFN({'data':images})
            pred=tf.nn.softmax('prob')

            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=pred),0)
            #optm=tf.train.RMSPropOptimizer(LR)
            optm=tf.train.RMSPropOptimizer(LR)
            train_op=optm.minimize(loss)#for train
            
            #for test
            correcta_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(labels,1))
            test_cast=tf.cast(correcta_prediction, "float")
            sum_test=tf.reduce_sum(test_cast)#for large test set
            accuracy = tf.reduce_mean(test_cast)#for small test set
            if logtem:
                ftl=open(templog,'a')
                ftl.write('\n\n\n%s\n'%str(sys.argv))
            with tf.Session() as sess:
            
                if Show_Summaries:
                    tf.summary.scalar('Cross_entropy', loss)
                    tf.summary.scalar('Accuracy', accuracy)
                    merged=tf.summary.merge_all()
                    trainw=tf.summary.FileWriter(summary_dir, sess.graph)

                sess.run(tf.global_variables_initializer())

                if loadONW:
                    loadPretrainedModel(NetworkType, whole_face_net, sess,Module)

                saver = tf.train.Saver()
                for i in range(IterationsC):#iteration goes wrong at 287009 with no clue
                    #train phase
                    batch=data.train.next_batch(currentBatchSize, shuffle=False)
                    if Show_Summaries:#write summaries in the file for tensorboard visualization
                        np_loss, np_pred, sies, _=sess.run([loss, pred, merged, train_op],feed_dict={images:batch[0], labels:batch[5]})
                        trainw.add_summary(sies, i)
                    else:
                        np_loss, np_pred, _=sess.run([loss, pred, train_op],feed_dict={images:batch[0], labels:batch[5]})
                    if np_loss<mini_loss:
                        mini_loss=np_loss
                    #validation and analyse
                    v_accuracy, valid_loss = Eva_on_ValidSet_M1(data, TestNumLimit, test_bat, sess, accuracy, sum_test, loss, images, labels)
                    laflag = loss_a.analyzeLossVariation(valid_loss) 
                    if laflag[0]:
                        loss_a.show()
                    tt=time.time()
                    print("CLR:%.8f Ite:%06d Bs:%03d Epo:%04d Los:%.8f mLo:%08f\tVALID>> mVL: %.8f\tVL: %.8f\tVA: %f\tT: %fs"%
                          (currentLR,i,currentBatchSize,data.train.epochs_completed, np_loss, mini_loss, loss_a.minimun_loss, valid_loss, v_accuracy, (tt-t1)))
                    if laflag[1]:
                        #update the file_record if a new minimun value of valid_loss is reached 
                        ac, oaa, confu_mat = Eva_on_TestSet_M1(data, TestNumLimit, test_bat, sess, accuracy, sum_test, pred, images, labels)
                        file_record=logfile(file_record, runs=runs, OAA=oaa, valid_loss=valid_loss, valid_min_loss=loss_a.minimun_loss, 
                            final_train_loss=np_loss, train_min_loss=mini_loss,
                            FHS=loss_a.first_half_std, SHS=loss_a.second_half_std, MV=loss_a.mean_variance,
                            VA=v_accuracy, TA=ac, TC=(tt-t1),ILR=learningRate, FLR=currentLR, LS=lrstep, ites=i,
                            Epo=data.train.epochs_completed, cBS=currentBatchSize, iBS=batchSize,
                            input=sys.argv, CM=confu_mat, T=time.localtime(tt), df=dfile)
                        if loss_a.minimun_loss < stcmwvlilttv:
                            #save the model if condition is satisfied
                            saver.save(sess=sess, save_path=getModelPath(model_save_path, tt, valid_loss, best=laflag[1]), global_step=i)
                    if logtem:
                        ftl.write("CLR:%.8f Ite:%06d Bs:%03d Epo:%04d Los:%.8f mLo:%08f\tVALID>> mVL: %.8f\tVL: %.8f\tVA: %f\tT: %fs\n"%
                          (currentLR,i,currentBatchSize,data.train.epochs_completed, np_loss, mini_loss, loss_a.minimun_loss, valid_loss, v_accuracy, (tt-t1)))
                        loss_a.log(ftl)
                    if (i+1)%lrstep == 0:
                        currentLR=updateLR(currentLR)
                        currentBatchSize=updateBS(currentBatchSize)
                for j in range(c_i_c):###provide a harder condition for exiting the training
                    print('Finding the optimal exit point......')
                    ###train phase
                    batch=data.train.next_batch(currentBatchSize, shuffle=False)
                    if Show_Summaries:#write summaries in the file for tensorboard visualization
                        np_loss, np_pred, sies, _=sess.run([loss, pred, merged, train_op],feed_dict={images:batch[0], labels:batch[5]})
                        trainw.add_summary(sies, i+j)
                    else:
                        np_loss, np_pred, _=sess.run([loss, pred, train_op],feed_dict={images:batch[0], labels:batch[5]})
                    if np_loss<mini_loss:
                        mini_loss=np_loss

                    ###validation and analyse
                    v_accuracy, valid_loss = Eva_on_ValidSet_M1(data, TestNumLimit, test_bat, sess, accuracy, sum_test, loss, images, labels)
                    laflag = loss_a.analyzeLossVariation(valid_loss) 
                    if laflag[0]:
                        loss_a.show()
                    tt=time.time()
                    print("CLR:%.8f Ite:%06d Bs:%03d Epo:%04d Los:%.8f mLo:%08f\tVALID>> mVL: %.8f\tVL: %.8f\tVA: %f\tT: %fs"%
                          (currentLR,j+IterationsC,currentBatchSize,data.train.epochs_completed, np_loss, mini_loss, loss_a.minimun_loss, valid_loss, v_accuracy, (tt-t1)))
                    if logtem:
                        ftl.write("CLR:%.8f Ite:%06d Bs:%03d Epo:%04d Los:%.8f mLo:%08f\tVALID>> mVL: %.8f\tVL: %.8f\tVA: %f\tT: %fs\n"%
                          (currentLR,j+IterationsC,currentBatchSize,data.train.epochs_completed, np_loss, mini_loss, loss_a.minimun_loss, valid_loss, v_accuracy, (tt-t1)))
                        loss_a.log(ftl)
                    if (valid_loss-loss_a.minimun_loss)<c_i_c_loss_error:
                        mini_loss_track = mini_loss_track + 1
                        if mini_loss_track < MLTT or data.train.epochs_completed < Mini_Epochs:
                            continue
                        else:
                            break
                    else:
                        mini_loss_track=0
                    if laflag[1]:
                        ac, oaa, confu_mat = Eva_on_TestSet_M1(data, TestNumLimit, test_bat, sess, accuracy, sum_test, pred, images, labels)
                        file_record=logfile(file_record, runs=runs, OAA=oaa, valid_loss=valid_loss, valid_min_loss=loss_a.minimun_loss, 
                            final_train_loss=np_loss, train_min_loss=mini_loss,
                            FHS=loss_a.first_half_std, SHS=loss_a.second_half_std, MV=loss_a.mean_variance,
                            VA=v_accuracy, TA=ac, TC=(tt-t1),ILR=learningRate, FLR=currentLR, LS=lrstep, ites=j+IterationsC,
                            Epo=data.train.epochs_completed, cBS=currentBatchSize, iBS=batchSize,
                            input=sys.argv, CM=confu_mat, T=time.localtime(tt), df=dfile)
                        if loss_a.minimun_loss < stcmwvlilttv:
                            saver.save(sess=sess, save_path=getModelPath(model_save_path, tt, valid_loss, laflag[1]), global_step=j+IterationsC)
                    if (j+1)%lrstep == 0:
                        currentLR=updateLR(currentLR)
                        currentBatchSize=updateBS(currentBatchSize)
                IterationsC=IterationsC+j
                tt=time.time()

                ac, oaa, confu_mat = Eva_on_TestSet_M1(data, TestNumLimit, test_bat, sess, accuracy, sum_test, pred, images, labels)
                print('Test Accuracy: %f\tTime used: %fs'%(ac,(tt-t1)))
                print('Over All Accuracy: %f'%(oaa))

                tt=time.time()
                if logtem:
                    ftl.write('Test Accuracy: %f\tTime used: %fs\n'%(ac,(tt-t1)))
                    ftl.write('Over All Accuracy: %f'%(oaa))
                    ftl.write('Time comsumed: %fs\t%s\n'%(tt-t1,time.strftime('%Y%m%d%H%M%S',time.localtime(tt))))
                    ftl.close()
                print('Time comsumed: %fs\t%s'%(tt-t1,time.strftime('%Y%m%d%H%M%S',time.localtime(tt))))
                if saveM:
                    if laflag[1] or runs%3==0:
                        try:
                            saver.save(sess=sess,save_path=getModelPath(model_save_path, tt, valid_loss, reshape, best=laflag[1]),
                                       global_step=IterationsC)
                        except:
                            traceback.print_exc()
            
                if log:
                    tlog=open('./logs/depressrunninglogvggface.txt','a')
                    if laflag[1]:
                        file_record=logfile(file_record, runs=runs, OAA=oaa, valid_loss=valid_loss, valid_min_loss=loss_a.minimun_loss, 
                            final_train_loss=np_loss, train_min_loss=mini_loss,
                            FHS=loss_a.first_half_std, SHS=loss_a.second_half_std, MV=loss_a.mean_variance,
                            VA=v_accuracy, TA=ac, TC=(tt-t1),ILR=learningRate, FLR=currentLR, LS=lrstep, ites=IterationsC,
                            Epo=data.train.epochs_completed, cBS=currentBatchSize, iBS=batchSize,
                            input=sys.argv, CM=confu_mat, T=time.localtime(tt), df=dfile)
                    tlog.write(file_record)
                    #logfile(file_handle=tlog, runs=runs, OAA=oaa, valid_loss=valid_loss, valid_min_loss=loss_a.minimun_loss, 
                    #        final_train_loss=np_loss, train_min_loss=mini_loss,
                    #        FHS=loss_a.first_half_std, SHS=loss_a.second_half_std, MV=loss_a.mean_variance,
                    #        VA=v_accuracy, TA=ac, TC=(tt-t1),ILR=learningRate, FLR=currentLR, LS=lrstep, ites=IterationsC,
                    #        Epo=data.train.epochs_completed, cBS=currentBatchSize, iBS=batchSize,
                    #        input=sys.argv, CM=confu_mat, T=time.localtime(tt), df=dfile)
                    tlog.close()
                    bslrf=open(logbslr,'a')
                    bslrf.write(file_record)
                    #logfile(file_handle=bslrf, runs=runs, OAA=oaa, valid_loss=valid_loss, valid_min_loss=loss_a.minimun_loss, 
                    #        final_train_loss=np_loss, train_min_loss=mini_loss,
                    #        FHS=loss_a.first_half_std, SHS=loss_a.second_half_std, MV=loss_a.mean_variance,
                    #        VA=v_accuracy, TA=ac, TC=(tt-t1),ILR=learningRate, FLR=currentLR, LS=lrstep, ites=IterationsC,
                    #        Epo=data.train.epochs_completed, cBS=currentBatchSize, iBS=batchSize,
                    #        input=sys.argv, CM=confu_mat, T=time.localtime(tt), df=dfile)
                    bslrf.close()
                    
                '''MODULE1 ENDS---------------------------------------------------------------------------------------------'''
        #
        #
        #
        elif Module==2:
            stcmwvlilttv=1.1854#value need to be determined. save_the_current_model_when_validation_loss_is_less_than_this_value
            '''MODULE2---------------------------------------------------------------------------------------------------- 
            Options for the Geometry-network
            Only need to select one of the import options as the network for the geometry feature extraction.
            -------------------------------------------------------------------------------------------------------------'''
            IterationsC=23001
            c_i_c=18000
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
                currentLR=0.00001
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
            pred=tf.nn.softmax('prob')

            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=pred),0)
            optm=tf.train.RMSPropOptimizer(LR)
            #optm=tf.train.RMSPropOptimizer(LR)
            train_op=optm.minimize(loss)#for train

            #for test
            correcta_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(labels,1))
            accuracy = tf.reduce_mean(tf.cast(correcta_prediction, "float"))
        
            save = tf.train.Saver()

            if logtem:
                ftl=open(templog,'a')
                ftl.write('\n\n\n%s\n'%str(sys.argv))
 
            with tf.Session() as sess:

                if Show_Summaries:
                    tf.summary.scalar('Cross_entropy', loss)
                    tf.summary.scalar('Accuracy', accuracy)
                    merged=tf.summary.merge_all()
                    trainw=tf.summary.FileWriter(summary_dir, sess.graph)

                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver()
                for i in range(IterationsC):
                    batch=data.train.next_batch(currentBatchSize, shuffle=False)
                    if Show_Summaries:#write summaries in the file for tensorboard visualization
                        np_loss, np_pred, sies, _=sess.run([loss, pred, merged, train_op], feed_dict={geo_features:batch[1], labels:batch[5]})
                        trainw.add_summary(sies, i)
                    else:
                        np_loss, np_pred, _=sess.run([loss, pred, train_op], feed_dict={geo_features:batch[1], labels:batch[5]})
                    if np_loss<mini_loss:
                        mini_loss=np_loss
                    v_accuracy, valid_loss = sess.run([accuracy, loss],feed_dict={geo_features:data.validation.geometry, labels:data.validation.labels})
                    laflag = loss_a.analyzeLossVariation(valid_loss)
                    if laflag[0]:
                        loss_a.show()
                    tt=time.time()
                    print("CLR:%.8f Ite:%06d Bs:%03d Epo:%04d Los:%.8f mLo:%08f\tVALID>> mVL: %.8f\tVL: %.8f\tVA: %f\tT: %fs"%
                          (currentLR,i,currentBatchSize,data.train.epochs_completed, np_loss, mini_loss, loss_a.minimun_loss, valid_loss, v_accuracy, (tt-t1)))
                    if laflag[1]:
                        ac, oaa, confu_mat = Eva_on_TestSet_M2(data, sess, accuracy, pred, geo_features, labels)
                        file_record=logfile(file_record, runs=runs, OAA=oaa, valid_loss=valid_loss, valid_min_loss=loss_a.minimun_loss, 
                            final_train_loss=np_loss, train_min_loss=mini_loss,
                            FHS=loss_a.first_half_std, SHS=loss_a.second_half_std, MV=loss_a.mean_variance,
                            VA=v_accuracy, TA=ac, TC=(tt-t1),ILR=learningRate, FLR=currentLR, LS=lrstep, ites=i,
                            Epo=data.train.epochs_completed, cBS=currentBatchSize, iBS=batchSize,
                            input=sys.argv, CM=confu_mat, T=time.localtime(tt), df=dfile)
                        if loss_a.minimun_loss < stcmwvlilttv:
                            #save the model if condition is satisfied
                            saver.save(sess=sess, save_path=getModelPath(model_save_path, tt, valid_loss, best=laflag[1]), global_step=i)
                    if logtem:
                        ftl.write("CLR:%.8f Ite:%06d Bs:%03d Epo:%04d Los:%.8f mLo:%08f\tVALID>> mVL: %.8f\tVL: %.8f\tVA: %f\tT: %fs\n"%
                          (currentLR,i,currentBatchSize,data.train.epochs_completed, np_loss, mini_loss, loss_a.minimun_loss, valid_loss, v_accuracy, (tt-t1)))
                        loss_a.log(ftl)
                    if (i+1)%lrstep == 0:
                        currentLR=updateLR(currentLR)
                        currentBatchSize=updateBS(currentBatchSize)
                for j in range(c_i_c):###select a better final loss
                    print('Finding the optimal exit point......')
                    batch=data.train.next_batch(currentBatchSize, shuffle=False)
                    if Show_Summaries:#write summaries in the file for tensorboard visualization
                        np_loss, np_pred, sies, _=sess.run([loss, pred, merged, train_op], feed_dict={geo_features:batch[1], labels:batch[5]})
                        trainw.add_summary(sies, i+j)
                    else:
                        np_loss, np_pred, _=sess.run([loss, pred, train_op], feed_dict={geo_features:batch[1], labels:batch[5]})
                    if np_loss<mini_loss:
                        mini_loss=np_loss

                    v_accuracy, valid_loss = Eva_on_TestSet_M2(data, sess, accuracy, pred, geo_features, labels)
                    laflag = loss_a.analyzeLossVariation(valid_loss)
                    if laflag[0]:
                        loss_a.show()
                    tt=time.time()
                    print("CLR:%.8f Ite:%06d Bs:%03d Epo:%04d Los:%.8f mLo:%08f\tVALID>> mVL: %.8f\tVL: %.8f\tVA: %f\tT: %fs"%
                          (currentLR,j+IterationsC,currentBatchSize,data.train.epochs_completed, np_loss, mini_loss, loss_a.minimun_loss, valid_loss, v_accuracy, (tt-t1)))
                    if logtem:
                        ftl.write("CLR:%.8f Ite:%06d Bs:%03d Epo:%04d Los:%.8f mLo:%08f\tVALID>> mVL: %.8f\tVL: %.8f\tVA: %f\tT: %fs\n"%
                          (currentLR,j+IterationsC,currentBatchSize,data.train.epochs_completed, np_loss, mini_loss, loss_a.minimun_loss, valid_loss, v_accuracy, (tt-t1)))
                        loss_a.log(ftl)
                    if (valid_loss-loss_a.minimun_loss)<c_i_c_loss_error:
                        mini_loss_track=mini_loss_track+1
                        if mini_loss_track < MLTT or data.train.epochs_completed < Mini_Epochs:
                            continue
                        else:
                            break
                    else:
                        mini_loss_track=0
                    if laflag[1]:
                        ac, oaa, confu_mat = Eva_on_TestSet_M2(data, sess, accuracy, pred, geo_features, labels)
                        file_record=logfile(file_record, runs=runs, OAA=oaa, valid_loss=valid_loss, valid_min_loss=loss_a.minimun_loss, 
                            final_train_loss=np_loss, train_min_loss=mini_loss,
                            FHS=loss_a.first_half_std, SHS=loss_a.second_half_std, MV=loss_a.mean_variance,
                            VA=v_accuracy, TA=ac, TC=(tt-t1),ILR=learningRate, FLR=currentLR, LS=lrstep, ites=j+IterationsC,
                            Epo=data.train.epochs_completed, cBS=currentBatchSize, iBS=batchSize,
                            input=sys.argv, CM=confu_mat, T=time.localtime(tt), df=dfile)
                        if loss_a.minimun_loss < stcmwvlilttv:
                            saver.save(sess=sess, save_path=getModelPath(model_save_path, tt, valid_loss, laflag[1]), global_step=j+IterationsC)
                    if (j+1)%lrstep == 0:
                        currentLR=updateLR(currentLR)
                        currentBatchSize=updateBS(currentBatchSize)
                IterationsC=IterationsC+j
                tt=time.time()
                
                ac, oaa, confu_mat = Eva_on_TestSet_M2(data, sess, accuracy, pred, geo_features, labels)
                print('Over All Accuracy: %f'%(oaa))
                print('Test Accuracy: %f\tTime used: %fs'%(ac,(tt-t1)))

                tt=time.time()
                print('Time comsumed: %fs\t%s'%(tt-t1,time.strftime('%Y%m%d%H%M%S',time.localtime(tt))))
                if saveM:
                    if laflag[1] or runs%3==0:
                        try:
                            saver.save(sess=sess,save_path=getModelPath(model_save_path, tt, valid_loss, reshape, best=laflag[1]),
                                       global_step=IterationsC)
                        except:
                            traceback.print_exc()
            
                if log:
                    tlog=open('./logs/depressrunninglogGeometry.txt','a')
                    if laflag[1] or (file_record is None):
                        file_record=logfile(file_record, runs=runs, OAA=oaa, valid_loss=valid_loss, valid_min_loss=loss_a.minimun_loss, 
                            final_train_loss=np_loss, train_min_loss=mini_loss,
                            FHS=loss_a.first_half_std, SHS=loss_a.second_half_std, MV=loss_a.mean_variance,
                            VA=v_accuracy, TA=ac, TC=(tt-t1),ILR=learningRate, FLR=currentLR, LS=lrstep, ites=IterationsC,
                            Epo=data.train.epochs_completed, cBS=currentBatchSize, iBS=batchSize,
                            input=sys.argv, CM=confu_mat, T=time.localtime(tt), df=dfile)
                    tlog.write(file_record)
                    #logfile(file_handle=tlog, runs=runs, OAA=oaa, valid_loss=valid_loss, valid_min_loss=loss_a.minimun_loss, 
                    #        final_train_loss=np_loss, train_min_loss=mini_loss,
                    #        FHS=loss_a.first_half_std, SHS=loss_a.second_half_std, MV=loss_a.mean_variance,
                    #        VA=v_accuracy, TA=ac, TC=(tt-t1),ILR=learningRate, FLR=currentLR, LS=lrstep, ites=IterationsC,
                    #        Epo=data.train.epochs_completed, cBS=currentBatchSize, iBS=batchSize,
                    #        input=sys.argv, CM=confu_mat, T=time.localtime(tt), df=dfile)
                    tlog.close()
                    bslrf=open(logbslr,'a')
                    bslrf.write(file_record)
                    #logfile(file_handle=bslrf, runs=runs, OAA=oaa, valid_loss=valid_loss, valid_min_loss=loss_a.minimun_loss, 
                    #        final_train_loss=np_loss, train_min_loss=mini_loss,
                    #        FHS=loss_a.first_half_std, SHS=loss_a.second_half_std, MV=loss_a.mean_variance,
                    #        VA=v_accuracy, TA=ac, TC=(tt-t1),ILR=learningRate, FLR=currentLR, LS=lrstep, ites=IterationsC,
                    #        Epo=data.train.epochs_completed, cBS=currentBatchSize, iBS=batchSize,
                    #        input=sys.argv, CM=confu_mat, T=time.localtime(tt), df=dfile)
                    bslrf.close()
                '''MODULE2 ENDS---------------------------------------------------------------------------------------------'''
        #
        #
        #
        elif Module==3:
            stcmwvlilttv=1.2154#value need to be determined. save_the_current_model_when_validation_loss_is_less_than_this_value
            if DataSet==502 or DataSet==501:
                stcmwvlilttv=1.2004
            elif DataSet==532 or DataSet==531:
                stcmwvlilttv=1.2004
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
                from FacePatches_NET import FacePatches_NET_3Conv_1Inception as PaN
            else:
                print("Usage: python finetune.py <GPUID> <Module> <NetworkType>\nWith Module 2, NetworkType must be 0, 1")
                exit(-1)
            learningRate=0.00005
            currentLR=0.00005
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
            global_step = tf.Variable(0, trainable=False)
            lr=tf.train.exponential_decay(cLR, global_step, lrstep, lr_drate, staircase=True)

            FacePatch_net = PaN({'eyePatch_data':eye_p, 'middlePatch_data':midd_p, 'mouthPatch_data':mou_p})
            print(type(FacePatch_net))
            pred=tf.nn.softmax('prob')

            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=pred),0)
            optm=tf.train.RMSPropOptimizer(lr)
            train_op=optm.minimize(loss, global_step)#for train

            #for test
            correcta_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(labels,1))
            test_cast=tf.cast(correcta_prediction, "float")
            sum_test=tf.reduce_sum(test_cast)#for large test set
            accuracy = tf.reduce_mean(test_cast)#for small test set
    
            
            saver = tf.train.Saver()
            if logtem:
                ftl=open(templog,'a')
                ftl.write('\n\n\n%s\n'%str(sys.argv))
            with tf.Session() as sess:
                if Show_Summaries:
                    tf.summary.scalar('Cross_entropy', loss)
                    tf.summary.scalar('Accuracy', accuracy)
                    merged=tf.summary.merge_all()
                    trainw=tf.summary.FileWriter(summary_dir, sess.graph)

                sess.run(tf.global_variables_initializer())
                
                if loadONW:
                    restorefacepatchModel(DataSet, sess, saver)

                for i in range(IterationsC):
                    batch=data.train.next_batch(currentBatchSize, shuffle=False)
                    if Show_Summaries:#write summaries in the file for tensorboard visualization
                        np_loss, np_pred, sies, _=sess.run([loss, pred, merged, train_op], feed_dict={eye_p:batch[2], midd_p:batch[3], mou_p:batch[4], labels:batch[5]})
                        trainw.add_summary(sies, i)
                    else:
                        np_loss, np_pred, _=sess.run([loss, pred, train_op], feed_dict={eye_p:batch[2], midd_p:batch[3], mou_p:batch[4], labels:batch[5]})
                    if np_loss<mini_loss:
                        mini_loss=np_loss
                    v_accuracy, valid_loss = Eva_on_ValidSet_M3(data, TestNumLimit, test_bat, sess, accuracy, sum_test, loss, eye_p, midd_p, mou_p, labels)
                    laflag = loss_a.analyzeLossVariation(valid_loss) 
                    if laflag[0]:
                        loss_a.show()
                    tt=time.time()
                    print("CLR:%.8f Ite:%06d Bs:%03d Epo:%04d Los:%.8f mLo:%08f\tVALID>> mVL: %.8f\tVL: %.8f\tVA: %f\tT: %fs"%
                          (currentLR,i,currentBatchSize,data.train.epochs_completed, np_loss, mini_loss, loss_a.minimun_loss, valid_loss, v_accuracy, (tt-t1)))
                    if laflag[1]:
                        ac, oaa, confu_mat = Eva_on_TestSet_M3(data, TestNumLimit, test_bat, sess, accuracy, sum_test, pred, eye_p, midd_p, mou_p, labels)
                        file_record = logfile(file_record, runs=runs, OAA=oaa, valid_loss=valid_loss, valid_min_loss=loss_a.minimun_loss, 
                            final_train_loss=np_loss, train_min_loss=mini_loss,
                            FHS=loss_a.first_half_std, SHS=loss_a.second_half_std, MV=loss_a.mean_variance,
                            VA=v_accuracy, TA=ac, TC=(tt-t1),ILR=learningRate, FLR=currentLR, LS=lrstep, ites=IterationsC,
                            Epo=data.train.epochs_completed, cBS=currentBatchSize, iBS=batchSize,
                            input=sys.argv, CM=confu_mat, T=time.localtime(tt), df=dfile)
                        if loss_a.minimun_loss < stcmwvlilttv:
                            saver.save(sess=sess, save_path=getModelPath(model_save_path, tt, valid_loss, laflag[1]), global_step=i)
                    if logtem:
                        ftl.write("CLR:%.8f Ite:%06d Bs:%03d Epo:%04d Los:%.8f mLo:%08f\tVALID>> mVL: %.8f\tVL: %.8f\tVA: %f\tT: %fs\n"%
                          (currentLR,i,currentBatchSize,data.train.epochs_completed, np_loss, mini_loss, loss_a.minimun_loss, valid_loss, v_accuracy, (tt-t1)))
                        loss_a.log(ftl)
                    if (i+1)%lrstep == 0:
                        currentLR=updateLR(currentLR)
                        currentBatchSize=updateBS(currentBatchSize)
                for j in range(c_i_c):###select a better final loss
                    print('Finding the optimal exit point......')
                    batch=data.train.next_batch(currentBatchSize, shuffle=False)
                    if Show_Summaries:#write summaries in the file for tensorboard visualization
                        np_loss, np_pred, sies, _=sess.run([loss, pred, merged, train_op], feed_dict={eye_p:batch[2], midd_p:batch[3], mou_p:batch[4], labels:batch[5]})
                        trainw.add_summary(sies, i+j)
                    else:
                        np_loss, np_pred, _=sess.run([loss, pred, train_op], feed_dict={eye_p:batch[2], midd_p:batch[3], mou_p:batch[4], labels:batch[5]})
                    if np_loss<mini_loss:
                        mini_loss=np_loss
                    v_accuracy, valid_loss = Eva_on_ValidSet_M3(data, TestNumLimit, test_bat, sess, accuracy, sum_test, loss,eye_p,midd_p,mou_p,labels)
                    laflag = loss_a.analyzeLossVariation(valid_loss) 
                    if laflag[0]:
                        loss_a.show()
                    tt=time.time()
                    print("CLR:%.8f Ite:%06d Bs:%03d Epo:%04d Los:%.8f mLo:%08f\tVALID>> mVL: %.8f\tVL: %.8f\tVA: %f\tT: %fs"%
                          (currentLR,j+IterationsC,currentBatchSize,data.train.epochs_completed, np_loss, mini_loss, loss_a.minimun_loss, valid_loss, v_accuracy, (tt-t1)))
                    if logtem:
                        ftl.write("CLR:%.8f Ite:%06d Bs:%03d Epo:%04d Los:%.8f mLo:%08f\tVALID>> mVL: %.8f\tVL: %.8f\tVA: %f\tT: %fs\n"%
                          (currentLR,j+IterationsC,currentBatchSize,data.train.epochs_completed, np_loss, mini_loss, loss_a.minimun_loss, valid_loss, v_accuracy, (tt-t1)))
                        loss_a.log(ftl)
                    if (valid_loss-loss_a.minimun_loss)<c_i_c_loss_error:
                        mini_loss_track = mini_loss_track + 1
                        if mini_loss_track < MLTT or data.train.epochs_completed < Mini_Epochs:
                            continue
                        else:
                            break
                    else:
                        mini_loss_track=0
                    if (j+1)%lrstep == 0:
                        currentLR=updateLR(currentLR)
                        currentBatchSize=updateBS(currentBatchSize)
                    if laflag[1]:
                        ac, oaa, confu_mat = Eva_on_TestSet_M3(data, TestNumLimit, test_bat, sess, accuracy, sum_test, pred, eye_p, midd_p, mou_p, labels)
                        file_record = logfile(file_record, runs=runs, OAA=oaa, valid_loss=valid_loss, valid_min_loss=loss_a.minimun_loss, 
                            final_train_loss=np_loss, train_min_loss=mini_loss,
                            FHS=loss_a.first_half_std, SHS=loss_a.second_half_std, MV=loss_a.mean_variance,
                            VA=v_accuracy, TA=ac, TC=(tt-t1),ILR=learningRate, FLR=currentLR, LS=lrstep, ites=IterationsC,
                            Epo=data.train.epochs_completed, cBS=currentBatchSize, iBS=batchSize,
                            input=sys.argv, CM=confu_mat, T=time.localtime(tt), df=dfile)
                        if loss_a.minimun_loss < stcmwvlilttv:
                            saver.save(sess=sess, save_path=getModelPath(model_save_path, tt, valid_loss, laflag[1]), global_step=j+IterationsC)
                IterationsC=IterationsC+j
                tt=time.time()
                ac, oaa, confu_mat = Eva_on_TestSet_M3(data, TestNumLimit, test_bat, sess, accuracy, sum_test, pred, eye_p, midd_p, mou_p, labels)
                print('Over All Accuracy: %f'%(oaa))
                print('Test Accuracy: %f\tTime used: %fs'%(ac,(tt-t1)))
                tt=time.time()
                print('Time comsumed: %fs\t%s'%(tt-t1,time.strftime('%Y%m%d%H%M%S',time.localtime(tt))))
                if saveM:
                    if laflag[1] or runs%3==0:
                        try:
                            saver.save(sess=sess,save_path=getModelPath(model_save_path, tt, valid_loss, reshape, best=laflag[1]),
                                       global_step=IterationsC)
                        except:
                            traceback.print_exc()
                if log:
                    tlog=open('./logs/depressrunninglogFacePatch.txt','a')
                    if laflag[1] or (file_record is None):
                        file_record=logfile(file_record, runs=runs, OAA=oaa, valid_loss=valid_loss, valid_min_loss=loss_a.minimun_loss, 
                            final_train_loss=np_loss, train_min_loss=mini_loss,
                            FHS=loss_a.first_half_std, SHS=loss_a.second_half_std, MV=loss_a.mean_variance,
                            VA=v_accuracy, TA=ac, TC=(tt-t1),ILR=learningRate, FLR=currentLR, LS=lrstep, ites=IterationsC,
                            Epo=data.train.epochs_completed, cBS=currentBatchSize, iBS=batchSize,
                            input=sys.argv, CM=confu_mat, T=time.localtime(tt), df=dfile)
                    tlog.write(file_record)
                    tlog.close()
                    bslrf=open(logbslr,'a')
                    bslrf.write(file_record)
                    bslrf.close()
                    
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
                
                pred=tf.nn.softmax('prob')
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=pred),0)
                optm=tf.train.RMSPropOptimizer(LR)
                train_op=optm.minimize(loss)#for train
                #for test
                correcta_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(labels,1))
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

            for i in range(IterationsC):#iteration goes wrong at 287009 with no clue
                batch=data.train.next_batch(currentBatchSize, shuffle=False)
                d1geofc=geo_sess.run(geofc, feed_dict={geo_features:batch[1]})
                d2appfc=app_sess.run(appfc, feed_dict={images:batch[0]})
                np_loss, np_pred, _=fin_sess.run([loss, pred, train_op],feed_dict={geo_fc:d1geofc, app_fc:d2appfc, labels:batch[5]})
                print("CLR:%.8f Ite:%06d Bs:%03d Epo:%04d Los:%.8f mLo:%08f"%(currentLR,i,currentBatchSize,data.train.epochs_completed, np_loss, mini_loss))
                if np_loss<mini_loss:
                    mini_loss=np_loss
                if i%show_it == 0:
                    tt=time.time()
                    d1geofc_v=geo_sess.run(geofc, feed_dict={geo_features:data.validation.geometry})
                    d2appfc_v=app_sess.run(appfc, feed_dict={images:data.validation.res_images})
                    v_accuracy = (fin_sess.run([accuracy], feed_dict={geo_fc:d1geofc_v, app_fc:d2appfc_v, labels:data.validation.labels}))[0]
                    print('NT: %s\tIte: %d\tVA: %f\tT: %fs'%(type(fin_net),i, v_accuracy, (tt-t1)))
                if (i+1)%lrstep == 0:
                    currentLR=updateLR(currentLR)
                    currentBatchSize=updateBS(currentBatchSize)
            for j in range(c_i_c):###select a better final loss
                print('Finding the optimal exit point......')
                batch=data.train.next_batch(currentBatchSize, shuffle=False)
                d1geofc=geo_sess.run(geofc, feed_dict={geo_features:batch[1]})
                d2appfc=app_sess.run(appfc, feed_dict={images:batch[0]})
                np_loss, np_pred, _=fin_sess.run([loss, pred, train_op],feed_dict={geo_fc:d1geofc, app_fc:d2appfc, labels:batch[5]})
                print("CLR:%.8f Ite:%06d Bs:%03d Epo:%04d Los:%.8f mLo:%08f"%(currentLR,i,currentBatchSize,data.train.epochs_completed, np_loss, mini_loss))
                if (valid_loss-loss_a.minimun_loss)<c_i_c_loss_error:
                    mini_loss_track=mini_loss_track+1
                    if mini_loss_track < MLTT:
                        continue
                    d1geofc_v=geo_sess.run(geofc, feed_dict={geo_features:data.validation.geometry})
                    d2appfc_v=app_sess.run(appfc, feed_dict={images:data.validation.res_images})
                    v_accuracy = (fin_sess.run([accuracy], feed_dict={geo_fc:d1geofc_v, app_fc:d2appfc_v, labels:data.validation.labels}))[0]
                    print('NT: %s\tIte: %d\tVA: %f\tT: %fs'%(type(fin_net),i, v_accuracy, (tt-t1)))
                    break
                else:
                    mini_loss_track=0
                if i%show_it == 0:
                    tt=time.time()
                    d1geofc_v=geo_sess.run(geofc, feed_dict={geo_features:data.validation.geometry})
                    d2appfc_v=app_sess.run(appfc, feed_dict={images:data.validation.res_images})
                    v_accuracy = (fin_sess.run([accuracy], feed_dict={geo_fc:d1geofc_v, app_fc:d2appfc_v, labels:data.validation.labels}))[0]
                    print('NT: %s\tIte: %d\tVA: %f\tT: %fs'%(type(fin_net),i, v_accuracy, (tt-t1)))
                if (j+1)%lrstep == 0:
                    currentLR=updateLR(currentLR)
                    currentBatchSize=updateBS(currentBatchSize)
            IterationsC=IterationsC+j
            tt=time.time()
        
            d1geofc_v=geo_sess.run(geofc, feed_dict={geo_features:data.test.geometry})
            d2appfc_v=app_sess.run(appfc, feed_dict={images:data.test.res_images})

            tlabels=list((fin_sess.run([pred], feed_dict={geo_fc:d1geofc_v, app_fc:d2appfc_v, labels:data.test.labels}))[0])
            confu_mat=calR(tlabels,data.test.labels)
            oaa=overAllAccuracy(confu_mat)
            print('Over All Accuracy: %f'%(oaa))
        
            ac=(fin_sess.run([accuracy], feed_dict={geo_fc:d1geofc_v, app_fc:d2appfc_v, labels:data.test.labels}))[0]
            print('Test Accuracy: %f\tTime used: %fs'%(ac,(tt-t1)))
            tt=time.time()
            print('Time comsumed: %fs\t%s'%(tt-t1,time.strftime('%Y%m%d%H%M%S',time.localtime(tt))))
            if saveM:
                try:
                    DataSetPrepare.saveModels(fin_sess,model_save_path)
                except:
                    traceback.print_exc()
            
            if log:
                tlog=open('./logs/depressrunninglog_AppandGeo_fusion.txt','a')
                if laflag[1] or (file_record is None):
                        file_record=logfile(file_record, runs=runs, OAA=oaa, valid_loss=valid_loss, valid_min_loss=loss_a.minimun_loss, 
                            final_train_loss=np_loss, train_min_loss=mini_loss,
                            FHS=loss_a.first_half_std, SHS=loss_a.second_half_std, MV=loss_a.mean_variance,
                            VA=v_accuracy, TA=ac, TC=(tt-t1),ILR=learningRate, FLR=currentLR, LS=lrstep, ites=IterationsC,
                            Epo=data.train.epochs_completed, cBS=currentBatchSize, iBS=batchSize,
                            input=sys.argv, CM=confu_mat, T=time.localtime(tt), df=dfile)
                        tlog.write(file_record)
                        #logfile(file_handle=tlog, runs=runs, OAA=oaa, valid_loss=valid_loss, valid_min_loss=loss_a.minimun_loss, 
                        #        final_train_loss=np_loss, train_min_loss=mini_loss,
                        #        FHS=loss_a.first_half_std, SHS=loss_a.second_half_std, MV=loss_a.mean_variance,
                        #        VA=v_accuracy, TA=ac, TC=(tt-t1),ILR=learningRate, FLR=currentLR, LS=lrstep, ites=IterationsC,
                        #        Epo=data.train.epochs_completed, cBS=currentBatchSize, iBS=batchSize,
                        #        input=sys.argv, CM=confu_mat, T=time.localtime(tt), df=dfile)
                        tlog.close()
                        bslrf=open(logbslr,'a')
                        bslrf.write(file_record)
                        #logfile(file_handle=bslrf, runs=runs, OAA=oaa, valid_loss=valid_loss, valid_min_loss=loss_a.minimun_loss, 
                        #        final_train_loss=np_loss, train_min_loss=mini_loss,
                        #        FHS=loss_a.first_half_std, SHS=loss_a.second_half_std, MV=loss_a.mean_variance,
                        #        VA=v_accuracy, TA=ac, TC=(tt-t1),ILR=learningRate, FLR=currentLR, LS=lrstep, ites=IterationsC,
                        #        Epo=data.train.epochs_completed, cBS=currentBatchSize, iBS=batchSize,
                        #        input=sys.argv, CM=confu_mat, T=time.localtime(tt), df=dfile)
                        bslrf.close()
                
            '''MODULE6 ENDS---------------------------------------------------------------------------------------------'''
    except:
        ferror=open(errorlog,'a')
        traceback.print_exc()
        traceback.print_exc(file=ferror)
        ferror.close()
        