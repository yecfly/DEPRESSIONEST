#Here starts the depression estimation processes
##from 2017.09.26, the stop criteria of training is changing to 
##
##
import tensorflow as tf
import os, time, sys, traceback
import DataSetPrepare
import tflearn
import numpy as np
Dataset_Dictionary={1001:'./Datasets\D1001_MergeDataset_D501_D531_10G.pkl', 
                    1002:'./Datasets\D1002_MergeDataset_D502_D532_10G.pkl', 
                    10:'./Datasets\D10_CKplus_10groups_groupedbythe_CKplus-group-details_preprocessdata_with_calibRotation_rescaleimg_geometricfeatures_facepatches_weberface_skip-contempV2.pkl', 
                    11:'./Datasets\D11_CKplus_10groups_groupedbythe_CKplus-group-details_preprocessdata_with_calibRotation_rescaleimg224x224_geometricfeatures_facepatches_weberface224x224_skip-contempV2.pkl', 
                    12:'./Datasets\D12_CKplus_10groups_groupedbythe_CKplus-group-details_preprocessdata_with_calibRotation_rescaleimg_geometricfeatures_facepatches_ELTFS_skip-contempV2.pkl', 
                    13:'./Datasets\D13_CKplus_8G_V4_Geo258_ELTFS128x128.pkl', 
                    16:'./Datasets\D16_CKPLUS_10G_Enlargeby2015CCV_10T.pkl', 
                    17:'./Datasets\D17_CKplus_10G_V4_weberface128x128.pkl', 
                    18:'./Datasets\D18_CKplus_10G_V5_formalized_weberface128x128.pkl', 
                    19:'./Datasets\D19_CKplus_10G_V4_ELTFS128x128.pkl', 
                    2:'./Datasets\D2_CKplus_8groups_groupedbythe_CKplus-group-details_preprocessdata_with_calibRotation_rescaleimgnewmetric0731_skip-contempV2.pkl', 
                    33:'./Datasets\D33_KDEF_10G_rescaleimg_geometryfeature_patches_web.pkl', 
                    34:'./Datasets\D34_KDEF_10G_Enlargeby2015CCV_10T.pkl', 
                    3:'./Datasets\D3_CKplus_8groups_groupedbythe_CKplus-group-details_preprocessdata_with_calibRotation_rescaleimg_geometricfeatures_facepatches_webberface_skip-contempV2.pkl', 
                    40:'./Datasets\D40_jaffe_10groups_groupedbysubjects_rescaleimg_geometricfeatures_facepatches_weber.pkl', 
                    43:'./Datasets\D43_JAFFE_10G_Enlargeby2015CCV_10T.pkl', 
                    44:'./Datasets\D44_jaffe_10G_V4_weber128x128.pkl', 
                    4:'./Datasets\D4_CKplus_8groups_groupedbythe_CKplus-group-details_preprocessdata_with_calibRotation_rescaleimg_geometricfeatures_facepatches_webberfaceReverse_skip-contempV2.pkl', 
                    501:'./Datasets\D501_CKplus_10G_V5_newGeo_newPatch.pkl', 
                    502:'./Datasets\D502_CKPLUS_10G_Enlargeby2015ICCV_V5_newGeo_newPatches.pkl', 
                    531:'./Datasets\D531_KDEF_10G_V5_newGeo_newPatch.pkl', 
                    532:'./Datasets\D532_KDEF_10G_Enlargeby2015ICCV_V5_newGeo_newPatches.pkl', 
                    5:'./Datasets\D5_CKplus_8groups_groupedbythe_CKplus-group-details_preprocessdata_with_calibRotation_rescaleimg_geometricfeatures_facepatches_webberface25up_skip-contempV2.pkl', 
                    6:'./Datasets\D6_CKplus_8groups_groupedbythe_CKplus-group-details_preprocessdata_with_calibRotation_rescaleimg_geometricfeatureV2_skip-contempV2.pkl', 
                    7:'./Datasets\D7_CKplus_8groups_groupedbythe_CKplus-group-details_preprocessdata_with_calibRotation_rescaleimg_geometricfeatures_facepatches_webberface_innerfaceSizew36xh48_skip-contempV2.pkl', 
                    8:'./Datasets\D8_CKplus_8groups_groupedbythe_CKplus-group-details_preprocessdata_with_calibRotation_rescaleimg_geometricfeatures_facepatches_ELTFS_skip-contempV2.pkl', 
                    9:'./Datasets\D9_CKplus_8groups_groupedbythe_CKplus-group-details_preprocessdata_with_calibRotation_rescaleimg224x224_geometricfeatures_facepatches_weberface224x224_skip-contempV2.pkl'}


lr_drate=0.85
batchsize_step=0
test_bat=100
TestNumLimit = 200

class LOSS_ANA:
    '''The LOSS_ANA class collects the training losses and analyzes them.
        The initial length should be divided by 50 with no remainder.'''
    def __init__(self):
        self.__Validation_Loss_List = []
        self.__Current_Length = 0#indicates whether the Validation_Loss_List has reach the maximum Length
        self.__Min_Loss = 10000.0

    @property
    def minimun_loss(self):
        return self.__Min_Loss
    @property
    def loss_length(self):
        return self.__Current_Length

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
def Valid_on_TestSet(sess, accuracy, sum_test, loss, softmax,
                       placeholder1_imgs, placeholder1_input, 
                       placeholder2_labels, placeholder2_input):
    ncount=len(placeholder2_input)
    tlabels=[]
    if ncount>TestNumLimit:               
        test_iter=np.floor_divide(ncount,test_bat)
        v_accuracy=0
        valid_loss=0
        for ite in range(test_iter):
            start=test_bat*ite
            end=test_bat*(ite+1)
            st, v_loss, tlab=sess.run([sum_test, loss, softmax], feed_dict={placeholder1_imgs:placeholder1_input[start:end],
                                                            placeholder2_labels:placeholder2_input[start:end]})
            v_accuracy=v_accuracy+st
            valid_loss=valid_loss+v_loss
            tlabels.extend(tlab)
        st, v_loss, tlab=sess.run([sum_test, loss, softmax], feed_dict={placeholder1_imgs:placeholder1_input[test_bat*test_iter:ncount], 
                                                        placeholder2_labels:placeholder2_input[test_bat*test_iter:ncount]})
        v_accuracy=v_accuracy+st
        valid_loss=valid_loss+v_loss
        v_accuracy=v_accuracy/ncount
        valid_loss=valid_loss/(test_iter+1)
        tlabels.extend(tlab)
    else:
        v_accuracy, valid_loss, tlabels = sess.run([accuracy, loss, softmax], feed_dict={placeholder1_imgs:placeholder1_input, 
                                                                       placeholder2_labels:placeholder2_input})
    confu_mat=calR(tlabels, placeholder2_input)
    oaa=overAllAccuracy(confu_mat)
    return v_accuracy, valid_loss, oaa, confu_mat
def logfile(file_record, runs, OAA, valid_loss, valid_min_loss, final_train_loss, train_min_loss, TA, TC, ILR, FLR, LS, ites, Epo, cBS, iBS, input, CM, T, df):
    file_record="Run%02d\tOverAllACC:%0.8f\tTestAccuracy:%.8f\tFinalLoss:%.10f\tMinimunLoss:%.10f\tFinaltrainloss:%.10f\tMinimumtrainloss:%.10f\tTimeComsumed:%08.6f\tInitialLearningRate:%.8f\tFinalLearningRate:%.8f\tLearningStepForDroppingMagnitude:%08d\tTotalIterations:%08d\tEpoches:%08d\tcurrentBatchSize:%05d\tinitialBatchSize:%05d\tInput:%s\t%s\tTime:%s\tDataFile:%s"%(runs, 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                              OAA, TA, valid_loss, valid_min_loss, final_train_loss, train_min_loss, TC, ILR,FLR, LS,ites,Epo,cBS,iBS,str(input),str(CM),time.strftime('%Y%m%d%H%M%S',T),df)
    return file_record    
def savelistcontent(filename, list):
    fw=open(filename, 'w')
    for v in list:
        print(v.name)
        print(type(v))
        print(v.eval())
        print()
        fw.write('%s\n'%(str(v)))
    fw.close()
def printgraph(name, graph, sess):
    #print('\n\n>>>>>>>>>>>>>>all collection keys')
    #print(graph.get_all_collection_keys())
    #savelistcontent('./fpit_%s_all_collection_keys.txt'%(name),graph.get_all_collection_keys())
    #print('\n\n>>>>>>>>>>>>>>all variables')
    #print(graph.get_collection(name='variables'))
    #savelistcontent('./fpit_%s_all_variables.txt'%(name),graph.get_collection(name='variables'))
    #print('\n\n>>>>>>>>>>>>>>all train_op')
    #print(graph.get_collection(name='train_op'))
    #savelistcontent('./fpit_%s_train_op.txt'%(name),graph.get_collection(name='train_op'))
    #print('\n\n>>>>>>>>>>>>>>all trainable variables')
    #print(graph.get_collection(name='trainable_variables'))
    savelistcontent('./fpit_%s_trainable_variables.txt'%(name), graph.get_collection(name='trainable_variables'))
def restorefacepatchModel(TrainID, sess, saver, FPI):
    if TrainID%100<20:
        if FPI==1:
            saver.restore(sess, './FPPTM/EyePatch_TrainonD532_TestonD501_N4_R9_20171019103910_1.51784744629_.ckpt')#OverAllACC:0.57857271	TestAccuracy:0.65412330	FinalLoss:1.5178474463
        elif FPI==2:
            saver.restore(sess, './FPPTM/MiddlePatch_TrainonD532_TestonD501_N4_R11_20171019080535_1.66863813767_.ckpt')#OverAllACC:0.44420250	TestAccuracy:0.49079263	FinalLoss:1.6686381377
        elif FPI==3:
            saver.restore(sess, './FPPTM/MouthPatch_TrainonD532_TestonD501_N4_R1_20171018224312_1.42820624205_.ckpt')#OverAllACC:0.68346248	TestAccuracy:0.74299440	FinalLoss:1.4282062420
        else:
            exit(-1)
    else:
        exit(-1)
def savefacepatchmodelwithonlytrainablevariables(TrainID, sess, saver2, FPI):
    if TrainID%100<20:
        if FPI==1:
            saver2.save(sess, './FPPTM/EyePatch_TrainonD532_TestonD501_N4_R9_20171019103910_1.51784744629_only_trainable_variables.ckpt')#OverAllACC:0.57857271	TestAccuracy:0.65412330	FinalLoss:1.5178474463
        elif FPI==2:
            saver2.save(sess, './FPPTM/MiddlePatch_TrainonD532_TestonD501_N4_R11_20171019080535_1.66863813767_only_trainable_variables.ckpt')#OverAllACC:0.44420250	TestAccuracy:0.49079263	FinalLoss:1.6686381377
        elif FPI==3:
            saver2.save(sess, './FPPTM/MouthPatch_TrainonD532_TestonD501_N4_R1_20171018224312_1.42820624205_only_trainable_variables.ckpt')#OverAllACC:0.68346248	TestAccuracy:0.74299440	FinalLoss:1.4282062420
        else:
            exit(-1)
def savegraph(sess, saver, saver2, FacePatchID, TrainID=531):
    restorefacepatchModel(TrainID, sess, saver, FacePatchID)
    savefacepatchmodelwithonlytrainablevariables(TrainID, sess, saver2, FacePatchID)
def setSavertoTrainableVariables(graph):
    tv=graph.get_collection(name='trainable_variables')
    dict={}
    length=len(tv)
    for i in range(length-1):
        dict[tv[i].name]=tv[i]
        print(type(dict[tv[i].name]))
        print(type(tv[i]))
    saver=tf.train.Saver(dict)
    return saver

def runPatch(GPU_Device_ID, FacePatchID, trainpklID, testpklID, NetworkType, runs,
         Epoch=45, batchSize=15, cLR=0.0001):
    try:
        '''GPU Option---------------------------------------------------------------------------------------------
        Determine which GPU is going to be used
        ------------------------------------------------------------------------------------------------------------'''
        print('GPU Option: %s'%(GPU_Device_ID))
        if (0==int(GPU_Device_ID)) or (1==int(GPU_Device_ID)):
            os.environ["CUDA_VISIBLE_DEVICES"]=str(GPU_Device_ID)
            errorlog='./logs/errors_gpu'+str(GPU_Device_ID)+'.txt'
        '''GPU Option ENDS---------------------------------------------------------------------------------------'''
        if testpklID==trainpklID:
            print('Warning: train and test phases use the same dataset.')
            strinput=input('Press Y to continue and any other key to exit')
            if not strinput=='y' and not strinput=='Y':
                exit(-1)

        if Dataset_Dictionary.get(int(testpklID), False) and Dataset_Dictionary.get(int(trainpklID), False):
            traindata=DataSetPrepare.loadPKLData_v2(Dataset_Dictionary.get(int(trainpklID)))
            testdata=DataSetPrepare.loadPKLData_v2(Dataset_Dictionary.get(int(testpklID)))
        else:
            print('Unsupported pkl ID')
            exit(-1)

        lrstep=6000
        mini_loss=10000
        loss_a=LOSS_ANA()
        file_record=None
        t1=time.time()
        log='./logs/facepatchpretrain/'
        model_save_path=''
        
        global_step = tf.Variable(0, trainable=False)
        lr=tf.train.exponential_decay(cLR, global_step, lrstep, lr_drate, staircase=True)
        ilr=cLR
        if FacePatchID==1:
            log=log+'EyePatch_TrainonD'+str(trainpklID)+'_TestonD'+str(testpklID)+'_N'+str(NetworkType)+'.txt'
            model_save_path=("J:/Models/saves/facepatchpretrain/"+'EyePatch_TrainonD'+str(trainpklID)+'_TestonD'+str(testpklID)+'_N'+str(NetworkType)+'_R'+str(runs)+
                             time.strftime('_%Y%m%d%H%M%S',time.localtime(t1))+".ckpt")

            stcmwvlilttv=1.6000#save_the_current_model_when_validation_loss_is_less_than_this_value
            '''FacePatchID1---------------------------------------------------------------------------------------------------- 
            Eye patch pretraining
            -------------------------------------------------------------------------------------------------------------'''
            print('Network Type: %s'%(NetworkType))
            x=tf.placeholder(tf.float32, [None, 26, 64,1])
            y=tf.placeholder(tf.float32, [None, 7])
            net=None
            if NetworkType==4:#4
                net=tflearn.conv_2d(x, 8, 3, activation='relu',name='eye_conv1_1_3x3')
                net=tflearn.conv_2d(net, 8, 3, activation='relu',name='eye_conv1_2_3x3')
                net=tflearn.max_pool_2d(net,2,2,name='eye_pool1')
                net=tflearn.conv_2d(net, 32, 3, activation='relu', name='eye_conv2_1_3x3')
                net=tflearn.conv_2d(net, 32, 3, activation='relu', name='eye_conv2_2_3x3')
                net=tflearn.max_pool_2d(net, 2, 2, name='eye_pool2')
                net=tflearn.conv_2d(net, 128, 3, activation='relu', name='eye_conv3_1_3x3')
                net=tflearn.conv_2d(net, 128, 3, activation='relu', name='eye_conv3_2_3x3')
                net=tflearn.max_pool_2d(net, 2, 2, name='eye_pool3')
                net=tflearn.fully_connected(net, 1024, activation='tanh', name='eye_fc1')
                net=tflearn.dropout(net, 0.8)
            elif NetworkType==2:
                exit(1)
            else:
                print("UNVALID INPUT ")
                exit(-1)
            '''Here begins the implementation logic-------------------------------------------------------------------
            -------------------------------------------------------------------------------------------------------------'''
            softmax=tflearn.fully_connected(net, 7, activation='softmax', restore=False)
            loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=softmax, labels=y))
            optimizer=tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, global_step=global_step)

            #for test
            correcta_prediction = tf.equal(tf.argmax(softmax,1),tf.argmax(y,1))
            test_cast=tf.cast(correcta_prediction, "float")
            sum_test=tf.reduce_sum(test_cast)#for large test set
            accuracy = tf.reduce_mean(test_cast)#for small test set

            saver=tf.train.Saver()
            saver2=setSavertoTrainableVariables(tf.get_default_graph())
            ibs=batchSize
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                #tv=tf.get_default_graph().get_collection(name='trainable_variables')
                #print('\n\nBefore')
                #print(tv[-1].eval())
                #saver.restore(sess, './FPPTM/EyePatch_TrainonD502_TestonD531_N4_R2_20171017210026.ckpt')#TestAccuracy:0.60510204	FinalLoss:1.5593829036
                #print('\n\nAfter')
                #print(tv[-1].eval())
                #printgraph('eyep', tf.get_default_graph(), sess, saver, FacePatchID)
                savegraph(sess, saver, saver2, FacePatchID, trainpklID)

            '''ENDS---------------------------------------------------------------------------------------------'''
        #
        #
        #
        elif FacePatchID==2:
            log=log+'MiddlePatch_TrainonD'+str(trainpklID)+'_TestonD'+str(testpklID)+'_N'+str(NetworkType)+'.txt'
            model_save_path=("J:/Models/saves/facepatchpretrain/"+'MiddlePatch_TrainonD'+str(trainpklID)+'_TestonD'+str(testpklID)+'_N'+str(NetworkType)+'_R'+str(runs)+
                             time.strftime('_%Y%m%d%H%M%S',time.localtime(t1))+".ckpt")
            batchSize=30
            stcmwvlilttv=1.7000#save_the_current_model_when_validation_loss_is_less_than_this_value
            '''FacePatchID1---------------------------------------------------------------------------------------------------- 
            Middle patch pretraining
            -------------------------------------------------------------------------------------------------------------'''
            print('Network Type: %s'%(NetworkType))
            x=tf.placeholder(tf.float32, [None, 49, 28,1])
            y=tf.placeholder(tf.float32, [None, 7])
            net=None
            if NetworkType==4:#4
                net=tflearn.conv_2d(x, 8, 3, activation='relu',name='middle_conv1_1_3x3')
                net=tflearn.conv_2d(net, 8, 3, activation='relu',name='middle_conv1_2_3x3')
                net=tflearn.max_pool_2d(net,2,2,name='middle_pool1')
                net=tflearn.conv_2d(net, 32, 3, activation='relu', name='middle_conv2_1_3x3')
                net=tflearn.conv_2d(net, 32, 3, activation='relu', name='middle_conv2_2_3x3')
                net=tflearn.max_pool_2d(net, 2, 2, name='middle_pool2')
                net=tflearn.conv_2d(net, 128, 3, activation='relu', name='middle_conv3_1_3x3')
                net=tflearn.conv_2d(net, 128, 3, activation='relu', name='middle_conv3_2_3x3')
                net=tflearn.max_pool_2d(net, 2, 2, name='middle_pool3')
                net=tflearn.fully_connected(net, 1024, activation='tanh', name='middle_fc1')
                net=tflearn.dropout(net, 0.8)
            elif NetworkType==2:
                exit(1)
            else:
                print("UNVALID INPUT ")
                exit(-1)
            '''Here begins the implementation logic-------------------------------------------------------------------
            -------------------------------------------------------------------------------------------------------------'''
            softmax=tflearn.fully_connected(net, 7, activation='softmax', restore=False)
            loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=softmax, labels=y))
            optimizer=tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, global_step=global_step)

            #for test
            correcta_prediction = tf.equal(tf.argmax(softmax,1),tf.argmax(y,1))
            test_cast=tf.cast(correcta_prediction, "float")
            sum_test=tf.reduce_sum(test_cast)#for large test set
            accuracy = tf.reduce_mean(test_cast)#for small test set

            saver=tf.train.Saver()
            saver2=setSavertoTrainableVariables(tf.get_default_graph())
            ilr=cLR
            ibs=batchSize
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                #printgraph('middlep', tf.get_default_graph(), sess)
                savegraph(sess, saver, saver2, FacePatchID, trainpklID)

            '''ENDS---------------------------------------------------------------------------------------------'''
        #
        #
        #
        elif FacePatchID==3:
            log=log+'MouthPatch_TrainonD'+str(trainpklID)+'_TestonD'+str(testpklID)+'_N'+str(NetworkType)+'.txt'
            model_save_path=("J:/Models/saves/facepatchpretrain/"+'MouthPatch_TrainonD'+str(trainpklID)+'_TestonD'+str(testpklID)+'_N'+str(NetworkType)+'_R'+str(runs)+
                             time.strftime('_%Y%m%d%H%M%S',time.localtime(t1))+".ckpt")

            stcmwvlilttv=1.6074#save_the_current_model_when_validation_loss_is_less_than_this_value
            '''MouthPatchID3---------------------------------------------------------------------------------------------------- 
            Mouth patch pretraining
            -------------------------------------------------------------------------------------------------------------'''
            print('Network Type: %s'%(NetworkType))
            x=tf.placeholder(tf.float32, [None, 30, 54,1])
            y=tf.placeholder(tf.float32, [None, 7])
            net=None
            if NetworkType==4:#4
                net=tflearn.conv_2d(x, 8, 3, activation='relu',name='mouth_conv1_1_3x3')
                net=tflearn.conv_2d(net, 8, 3, activation='relu',name='mouth_conv1_2_3x3')
                net=tflearn.max_pool_2d(net,2,2,name='mouth_pool1')
                net=tflearn.conv_2d(net, 32, 3, activation='relu', name='mouth_conv2_1_3x3')
                net=tflearn.conv_2d(net, 32, 3, activation='relu', name='mouth_conv2_2_3x3')
                net=tflearn.max_pool_2d(net, 2, 2, name='mouth_pool2')
                net=tflearn.conv_2d(net, 128, 3, activation='relu', name='mouth_conv3_1_3x3')
                net=tflearn.conv_2d(net, 128, 3, activation='relu', name='mouth_conv3_2_3x3')
                net=tflearn.max_pool_2d(net, 2, 2, name='mouth_pool3')
                net=tflearn.fully_connected(net, 1024, activation='tanh', name='mouth_fc1')
                net=tflearn.dropout(net, 0.8)
            elif NetworkType==2:
                exit(1)
            else:
                print("UNVALID INPUT ")
                exit(-1)
            '''Here begins the implementation logic-------------------------------------------------------------------
            -------------------------------------------------------------------------------------------------------------'''
            softmax=tflearn.fully_connected(net, 7, activation='softmax', restore=False)
            loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=softmax, labels=y))
            optimizer=tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, global_step=global_step)

            #for test
            correcta_prediction = tf.equal(tf.argmax(softmax,1),tf.argmax(y,1))
            test_cast=tf.cast(correcta_prediction, "float")
            sum_test=tf.reduce_sum(test_cast)#for large test set
            accuracy = tf.reduce_mean(test_cast)#for small test set

            saver=tf.train.Saver()
            saver2=setSavertoTrainableVariables(tf.get_default_graph())
            ilr=cLR
            ibs=batchSize
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                #printgraph('mouthp', tf.get_default_graph(), sess)
                savegraph(sess, saver, saver2, FacePatchID, trainpklID)

            '''ENDS---------------------------------------------------------------------------------------------'''
        else:
            print('Unexpected facepatchID')
            exit(-1)
        #
        #
        #
        newmodelname=model_save_path.split('.ckpt')[0]+'_'+str(loss_a.minimun_loss)+'_.ckpt'
        if os.path.exists(model_save_path+'.data-00000-of-00001'):
            os.rename((model_save_path+'.data-00000-of-00001'),(newmodelname+'.data-00000-of-00001'))
            os.rename((model_save_path+'.index'),(newmodelname+'.index'))
            os.rename((model_save_path+'.meta'),(newmodelname+'.meta'))

    except:
        ferror=open(errorlog,'w')
        traceback.print_exc()
        traceback.print_exc(file=ferror)
        ferror.close()

if __name__=='__main__':
    runPatch(int(sys.argv[1]), int(sys.argv[2]), 501, 531, 4, 0)


