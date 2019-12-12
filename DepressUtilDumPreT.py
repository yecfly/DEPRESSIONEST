#Here starts the depression estimation processes
import numpy as np
import tensorflow as tf
import os, pickle, time, sys, traceback
import DataSetPrepare
from skimage import transform

lr_step=0.8
c_i_c=3000
c_i_c_loss_error=0.0000001
batchsize_step=0

Show_Summaries=True
summary_dir='summaries'


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
        if NetworkType==4 or NetworkType<9:
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

def run(GPU_Device_ID, Module, 
        DataSet,ValidID,TestID, 
        NetworkType, runs
        ,learningRate=0.0001,IterationsC=19001,lrstep=3000,batchSize=30,saveM=False,log=False,loadONW=False):
    '''GPU Option---------------------------------------------------------------------------------------------
    Determine which GPU is going to be used
    ------------------------------------------------------------------------------------------------------------'''
    print('GPU Option: %s'%(GPU_Device_ID))
    if (0==GPU_Device_ID) or (1==GPU_Device_ID):
        os.environ["CUDA_VISIBLE_DEVICES"]=str(GPU_Device_ID)
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
    logprefix='./logs/log'
    D_f=False
    if Module==2:
        if NetworkType==0:
            D_f=True
        elif NetworkType==1:
            D_f=True
        elif NetworkType==2:
            D_f=True
    #
    ##data set loading
    #
    if DataSet==3:
        dfile='./Datasets/D3_CKplus_8groups_groupedbythe_CKplus-group-details_preprocessdata_with_calibRotation_rescaleimg_geometricfeatures_facepatches_webberface_skip-contempV2.pkl'
        data = DataSetPrepare.loadCKplus10gdata_v2(dfile, ValidID, TestID, Df=D_f)
        m1shape= [None, 128, 128, 1]
        m4shape= [None, 96, 72, 1]
        if log:
            logprefix="./logs/D3CKpluslogbslr_weberface_8groups_gpu"
            print("Processing 8 groups>>>>>>>>\n%s"%(logprefix))
    elif DataSet==4:
        dfile='./Datasets/D4_CKplus_8groups_groupedbythe_CKplus-group-details_preprocessdata_with_calibRotation_rescaleimg_geometricfeatures_facepatches_webberfaceReverse_skip-contempV2.pkl'
        data = DataSetPrepare.loadCKplus10gdata_v2(dfile, ValidID, TestID, Df=D_f)
        m1shape= [None, 128, 128, 1]
        m4shape= [None, 96, 72, 1]
        if log:
            logprefix="./logs/D4CKpluslogbslr_weberReverse_8groups_gpu"
            print("Processing 8 groups>>>>>>>>\n%s"%(logprefix))
    elif DataSet==5:
        dfile='./Datasets/D5_CKplus_8groups_groupedbythe_CKplus-group-details_preprocessdata_with_calibRotation_rescaleimg_geometricfeatures_facepatches_webberface25up_skip-contempV2.pkl'
        data = DataSetPrepare.loadCKplus10gdata_v2(dfile, ValidID, TestID, Df=D_f, one_hot=False)
        m1shape= [None, 128, 128, 1]
        m4shape= [None, 96, 72, 1]
        if log:
            logprefix="./logs/D5CKpluslogbslr_weberface25up_8groups_gpu"
            print("Processing 8 groups>>>>>>>>\n%s"%(logprefix))
    elif DataSet==7:
        dfile='./Datasets/D7_CKplus_8groups_groupedbythe_CKplus-group-details_preprocessdata_with_calibRotation_rescaleimg_geometricfeatures_facepatches_webberface_innerfaceSizew36xh48_skip-contempV2.pkl'
        data = DataSetPrepare.loadCKplus10gdata_v2(dfile, ValidID, TestID, Df=D_f)
        m1shape= [None, 128, 128, 1]
        m4shape= [None, 48, 36, 1]
        if log:
            logprefix="./logs/D7CKpluslogbslr_weberface_innerface48x36_8groups_gpu"
            print("Processing 8 groups>>>>>>>>\n%s"%(logprefix))
    elif DataSet==8:
        dfile='./Datasets/D8_CKplus_8groups_groupedbythe_CKplus-group-details_preprocessdata_with_calibRotation_rescaleimg_geometricfeatures_facepatches_ELTFS_skip-contempV2.pkl'
        data = DataSetPrepare.loadCKplus10gdata_v2(dfile, ValidID, TestID, Df=D_f)
        m1shape= [None, 128, 128, 1]
        m4shape= [None, 96, 72, 1]
        if log:
            logprefix="./logs/D8CKpluslogbslr_ELTFS_8groups_gpu"
            print("Processing 8 groups>>>>>>>>\n%s"%(logprefix))
    elif DataSet==9:
        dfile='./Datasets/D9_CKplus_8groups_groupedbythe_CKplus-group-details_preprocessdata_with_calibRotation_rescaleimg224x224_geometricfeatures_facepatches_weberface224x224_skip-contempV2.pkl'
        data = DataSetPrepare.loadCKplus10gdata_v2(dfile, ValidID, TestID, Df=D_f)
        m1shape= [None, 224, 224, 1]
        m4shape= [None, 224, 224, 1]
        if log:
            logprefix="./logs/D9CKpluslogbslr_weberface224_8groups_gpu"
            print("Processing 8 groups>>>>>>>>\n%s"%(logprefix))
    elif DataSet==10:
        dfile='./Datasets/D10_CKplus_10groups_groupedbythe_CKplus-group-details_preprocessdata_with_calibRotation_rescaleimg_geometricfeatures_facepatches_weberface_skip-contempV2.pkl'
        data = DataSetPrepare.loadCKplus10gdata_v2(dfile, ValidID, TestID, Df=D_f, one_hot=False)
        m1shape= [None, 128, 128, 1]
        m4shape= [None, 96, 72, 1]
        if log:
            logprefix="./logs/D10CKpluslogbslr_weberface_10groups_gpu"
            print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
    elif DataSet==11:
        dfile='./Datasets/D11_CKplus_10groups_groupedbythe_CKplus-group-details_preprocessdata_with_calibRotation_rescaleimg224x224_geometricfeatures_facepatches_weberface224x224_skip-contempV2.pkl'
        data = DataSetPrepare.loadCKplus10gdata_v2(dfile, ValidID, TestID, Df=D_f)
        m1shape= [None, 224, 224, 1]
        m4shape= [None, 224, 224, 1]
        if log:
            logprefix="./logs/D11CKpluslogbslr_weberface224_10groups_gpu"
            print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
    elif DataSet==12:
        dfile='./Datasets/D12_CKplus_10groups_groupedbythe_CKplus-group-details_preprocessdata_with_calibRotation_rescaleimg_geometricfeatures_facepatches_ELTFS_skip-contempV2.pkl'
        data = DataSetPrepare.loadCKplus10gdata_v2(dfile, ValidID, TestID, Df=D_f)
        m1shape= [None, 128, 128, 1]
        m4shape= [None, 96, 72, 1]
        if log:
            logprefix="./logs/D12CKpluslogbslr_ELTFS_10groups_gpu"
            print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
    elif DataSet==20:
        dfile='./Datasets/D20_MMI_8groups_groupedbythe_MMI-group-details_preprocessdata_with_calibRotation_rescaleimg_geometricfeatures_facepatches_webberface_innerface_skip-contempV.pkl'
        data = DataSetPrepare.loadCKplus10gdata_v2(dfile, ValidID, TestID, Df=D_f)
        m1shape= [None, 128, 128, 1]
        m4shape= [None, 96, 72, 1]
        if log:
            logprefix="./logs/D20_MMI_weberface_8groups_gpu"
            print("Processing 8 groups>>>>>>>>\n%s"%(logprefix))
    elif DataSet==21:
        dfile='./Datasets/D21_MMI_10groups_groupedbythe_MMI-group-details_preprocessdata_with_calibRotation_rescaleimg_geometricfeatures_facepatches_webberface_innerface_skip-contempV.pkl'
        data = DataSetPrepare.loadCKplus10gdata_v2(dfile, ValidID, TestID, Df=D_f)
        m1shape= [None, 128, 128, 1]
        m4shape= [None, 96, 72, 1]
        if log:
            logprefix="./logs/D21_MMI_weberface_10groups_gpu"
            print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
    elif DataSet==30:
        dfile='./Datasets/D30_KDEF_10groups_groupedbythe_KDEF-group-details_preprocessdata_with_calibRotation_rescaleimg_geometricfeatures_facepatches_webberface_innerface_skip-contempV.pkl'
        data = DataSetPrepare.loadCKplus10gdata_v2(dfile, ValidID, TestID, Df=D_f)
        m1shape= [None, 128, 128, 1]
        m4shape= [None, 96, 72, 1]
        if log:
            logprefix="./logs/D30_KDEF_weberface_10groups_gpu"
            print("Processing 10 groups>>>>>>>>\n%s"%(logprefix))
    else:
        print('ERROR: Unexpeted Dataset ID')
        exit()
    #
    #
    #
    tt=time.time()
    if log:
        if Module==6:
            logbslr=logprefix+str(GPU_Device_ID)+"_M"+str(Module)+"_D"+str(DataSet)+"_N"+str(NetworkType)+".txt"
        elif loadONW:
            logbslr=logprefix+str(GPU_Device_ID)+"_M"+str(Module)+"_D"+str(DataSet)+"_N"+str(NetworkType)+"_withPretrainModelWeight.txt"
        else:
            logbslr=logprefix+str(GPU_Device_ID)+"_M"+str(Module)+"_D"+str(DataSet)+"_N"+str(NetworkType)+"_noPretrain.txt"
        #logfilename=time.strftime('%Y%m%d%H%M%S',time.localtime(tt))+str(sys.argv[2:4])
    print('Time used for loading data: %fs'%(tt-t1))
    '''Input Data Ends-----------------------------------------------------------------------------------------'''
    #
    #
    #
    if Module==1:
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
        LR=tf.placeholder(tf.float32, shape=[])

        whole_face_net = WFN({'data':images})
        fc7 = whole_face_net.layers['fc7']
        pred=tf.nn.softmax(fc7)

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=pred),0)
        optm=tf.train.RMSPropOptimizer(currentLR)
        train_op=optm.minimize(loss)#for train

        #for test
        correcta_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(labels,1))
        accuracy = tf.reduce_mean(tf.cast(correcta_prediction, "float"))

        with tf.Session() as sess:
            
            tf.summary.scalar('Cross_entropy', loss)
            tf.summary.scalar('Accuracy', accuracy)
            merged=tf.summary.merge_all()
            trainw=tf.summary.FileWriter(summary_dir, sess.graph)

            sess.run(tf.global_variables_initializer())

            if loadONW:
                loadPretrainedModel(NetworkType, whole_face_net, sess,Module)

            for i in range(IterationsC):#iteration goes wrong at 287009 with no clue
                batch=data.train.next_batch(currentBatchSize, shuffle=False)
                np_loss, np_pred, sies, _=sess.run([loss, pred, merged, train_op],feed_dict={images:batch[0], labels:batch[5], LR:currentLR})
                print("Current Learning Rate: %.8f\tIteration: %08d    Loss: %f    miniLoss:%f"%(currentLR,i, np_loss, mini_loss))
                if Show_Summaries:#write summaries in the file for tensorboard visualization
                    trainw.add_summary(sies, i)
                if np_loss<mini_loss:
                    mini_loss=np_loss
                if i%50 == 0:
                    tt=time.time()
                    v_accuracy = accuracy.eval(feed_dict={images:data.validation.res_images, labels:data.validation.labels})
                    print('Network type: %s\nIteration: %d\tValidation Accuracy: %f\tTime used: %fs'%(type(whole_face_net),i, v_accuracy, (tt-t1)))
                if (i+1)%lrstep == 0:
                    currentLR=currentLR*lr_step
                    currentBatchSize=currentBatchSize+batchsize_step
            for j in range(c_i_c):###select a better final loss
                batch=data.train.next_batch(currentBatchSize, shuffle=False)
                np_loss, np_pred, sies, _=sess.run([loss, pred, merged, train_op],feed_dict={images:batch[0], labels:batch[5], LR:currentLR})
                print("Current Learning Rate: %.8f\tIteration: %08d    Loss: %f    miniLoss:%f"%(currentLR,j+IterationsC, np_loss, mini_loss))
                if Show_Summaries:#write summaries in the file for tensorboard visualization
                    trainw.add_summary(sies, i+j)
                if (np_loss-mini_loss)<c_i_c_loss_error:
                    tt=time.time()
                    v_accuracy = accuracy.eval(feed_dict={images:data.validation.res_images, labels:data.validation.labels})
                    print('Network type: %s\nIteration: %d\tValidation Accuracy: %f\tTime used: %fs'%(type(whole_face_net),j+IterationsC, v_accuracy, (tt-t1)))
                    break
                if j%50 == 0:
                    tt=time.time()
                    v_accuracy = accuracy.eval(feed_dict={images:data.validation.res_images, labels:data.validation.labels})
                    print('Network type: %s\nIteration: %d\tValidation Accuracy: %f\tTime used: %fs'%(type(whole_face_net),j+IterationsC, v_accuracy, (tt-t1)))
                if (j+1)%lrstep == 0:
                    currentLR=currentLR*lr_step
                    currentBatchSize=currentBatchSize+batchsize_step
            IterationsC=IterationsC+j
            tt=time.time()
            tlables=list(pred.eval(feed_dict={images:data.test.res_images, labels:data.test.labels}))
            confu_mat=calR(tlables,data.test.labels)
            oaa=overAllAccuracy(confu_mat)
            print('Over All Accuracy: %f'%(oaa))
            ac=accuracy.eval(feed_dict={images:data.test.res_images, labels:data.test.labels})
            print('Test Accuracy: %f\tTime used: %fs'%(ac,(tt-t1)))
            t2=time.time()
            print('Time comsumed: %fs\t%s'%(t2-t1,time.strftime('%Y%m%d%H%M%S',time.localtime(t2))))
            if saveM:
                try:
                    DataSetPrepare.saveModels(sess,("./saves/"+time.strftime('%Y%m%d%H%M%S',time.localtime(t2))+'_D'+str(DataSet)+'_M'+str(Module)+'_N'+str(NetworkType)+'_T'+str(TestID)+'_V'+str(ValidID)+'_R'+str(runs)+".ckpt"))
                except:
                    traceback.print_exc()
            
            if log:
                tlog=open('./logs/depressrunninglogvggface.txt','a')
                tlog.write("Run%02d\tFinalLoss:%.10f\tValidationAccuracy:%.8f\tTestAccuracy:%.8f\tOverAllACC:%0.8f\tTimeComsumed:%08.6f\tInitialLearningRate:%.8f\tFinalLearningRate:%.8f\tLearningStepForDroppingMagnitude:%08d\tTotalIterations:%08d\tEpoches:%08d\tcurrentBatchSize:%05d\tinitialBatchSize:%05d\tInput:%s\t%s\tTime:%s\tDataFile:%s\n"%((runs+1), 
                                                                                                                                                                                                                                                                                                                    np_loss, v_accuracy, ac, oaa, (t2-t1), learningRate,currentLR, lrstep,IterationsC,data.train.epochs_completed,currentBatchSize,batchSize,
                                                                                                                                                                                                                                                                                                                    str(sys.argv),str(confu_mat),time.strftime('%Y%m%d%H%M%S',time.localtime(t2)),dfile))
                tlog.close()
                bslrf=open(logbslr,'a')
                bslrf.write("Run%02d\tFinalLoss:%.10f\tValidationAccuracy:%.8f\tTestAccuracy:%.8f\tOverAllACC:%0.8f\tTimeComsumed:%08.6f\tInitialLearningRate:%.8f\tFinalLearningRate:%.8f\tLearningStepForDroppingMagnitude:%08d\tTotalIterations:%08d\tEpoches:%08d\tcurrentBatchSize:%05d\tinitialBatchSize:%05d\tInput:%s\t%s\tTime:%s\t%s\n"%((runs+1), 
                                                                                                                                                                                                                                                                                                                     np_loss, v_accuracy, ac, oaa, (t2-t1), learningRate,currentLR, lrstep,IterationsC,data.train.epochs_completed,currentBatchSize,batchSize,
                                                                                                                                                                                                                                                                                                                     str(sys.argv),str(confu_mat),time.strftime('%Y%m%d%H%M%S',time.localtime(t2)),dfile))
                bslrf.close()
            '''MODULE1 ENDS---------------------------------------------------------------------------------------------'''
    #
    #
    #
    elif Module==4:
        '''MODULE4---------------------------------------------------------------------------------------------------- 
        Options for the whole-face-network with inner_face input
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
            learningRate=0.00005
            currentLR=0.00005
        elif NetworkType==5:
            from VGG_NET import PRANDRE as WFN
        elif NetworkType==6:
            from VGG_NET import PRANDRE1 as WFN
        elif NetworkType==7:
            from VGG_NET import PRANDRE2 as WFN
        
        elif NetworkType==30:
            from PopularNets import ResNet50 as WFN
            #learningRate=0.001
            #currentLR=0.001
            #batchSize=10
            #currentBatchSize=10
        elif NetworkType==31:
            from PopularNets import ResNet101 as WFN
        elif NetworkType==32:
            from PopularNets import ResNet152 as WFN
        elif NetworkType==33:
            from PopularNets import AlexNet as WFN
            #learningRate=0.001
            #currentLR=0.001
            #batchSize=60
            #currentBatchSize=60
        elif NetworkType==34:
            from PopularNets import CaffeNet as WFN
        elif NetworkType==35:
            from PopularNets import NiN as WFN
        elif NetworkType==36:
            from PopularNets import GoogleNet as WFN
        else:
            print("Usage: python finetune.py <GPUID> <Module> <NetworkType>\nWrong NetworkType, please check the NetworkType input again.")
            exit(-1)
        '''Here begins the implementation logic-------------------------------------------------------------------
        -------------------------------------------------------------------------------------------------------------'''
        
        images = tf.placeholder(tf.float32, m4shape)
        #Holder for labels in a batch size of batch_size, number of labels are to be determined
        labels = tf.placeholder(tf.float32, labelshape)#the number of labels are to be determined
        LR=tf.placeholder(tf.float32, shape=[])
        
        whole_face_net = WFN({'data':images})
        fc7 = whole_face_net.layers['fc7']

        pred=tf.nn.softmax(fc7)
 
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=pred),0)
        optm=tf.train.RMSPropOptimizer(currentLR)
        train_op=optm.minimize(loss)#for train

        #for test
        correcta_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(labels,1))
        accuracy = tf.reduce_mean(tf.cast(correcta_prediction, "float"))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            if loadONW:
                loadPretrainedModel(NetworkType, whole_face_net, sess,Module)

            for i in range(IterationsC):#iteration goes wrong at 287009 with no clue
                batch=data.train.next_batch(currentBatchSize, shuffle=False)
                np_loss, np_pred, _=sess.run([loss, pred, train_op],feed_dict={images:batch[6], labels:batch[5], LR:currentLR})
                print("Current Learning Rate: %.8f\tIteration: %08d    Loss: %f    miniLoss:%f"%(currentLR,i, np_loss, mini_loss))
                if np_loss<mini_loss:
                    mini_loss=np_loss
                if i%50 == 0:
                    tt=time.time()
                    v_accuracy = accuracy.eval(feed_dict={images:data.validation.innerf, labels:data.validation.labels})
                    print('Network type: %s\nIteration: %d\tValidation Accuracy: %f\tTime used: %fs'%(type(whole_face_net),i, v_accuracy, (tt-t1)))
                if (i+1)%lrstep == 0:
                    currentLR=currentLR*lr_step
                    currentBatchSize=currentBatchSize+batchsize_step
            for j in range(c_i_c):###select a better final loss
                batch=data.train.next_batch(currentBatchSize, shuffle=False)
                np_loss, np_pred, _=sess.run([loss, pred, train_op],feed_dict={images:batch[6], labels:batch[5], LR:currentLR})
                print("Current Learning Rate: %.8f\tIteration: %08d    Loss: %f    miniLoss:%f"%(currentLR,j+IterationsC, np_loss, mini_loss))
                if (np_loss-mini_loss)<c_i_c_loss_error:
                    tt=time.time()
                    v_accuracy = accuracy.eval(feed_dict={images:data.validation.innerf, labels:data.validation.labels})
                    print('Network type: %s\nIteration: %d\tValidation Accuracy: %f\tTime used: %fs'%(type(whole_face_net),j+IterationsC, v_accuracy, (tt-t1)))
                    break
                if j%50 == 0:
                    tt=time.time()
                    v_accuracy = accuracy.eval(feed_dict={images:data.validation.innerf, labels:data.validation.labels})
                    print('Network type: %s\nIteration: %d\tValidation Accuracy: %f\tTime used: %fs'%(type(whole_face_net),j+IterationsC, v_accuracy, (tt-t1)))
                if (j+1)%lrstep == 0:
                    currentLR=currentLR*lr_step
                    currentBatchSize=currentBatchSize+batchsize_step
            IterationsC=IterationsC+j
            tt=time.time()
            tlables=list(pred.eval(feed_dict={images:data.test.innerf, labels:data.test.labels}))
            confu_mat=calR(tlables,data.test.labels)
            oaa=overAllAccuracy(confu_mat)
            print('Over All Accuracy: %f'%(oaa))
            ac=accuracy.eval(feed_dict={images:data.test.innerf, labels:data.test.labels})
            print('Test Accuracy: %f\tTime used: %fs'%(ac,(tt-t1)))
            t2=time.time()
            print('Time comsumed: %fs\t%s'%(t2-t1,time.strftime('%Y%m%d%H%M%S',time.localtime(t2))))
            if saveM:
                try:
                    DataSetPrepare.saveModels(sess,("./saves/"+time.strftime('%Y%m%d%H%M%S',time.localtime(t2))+'_D'+str(DataSet)+'_M'+str(Module)+'_N'+str(NetworkType)+'_T'+str(TestID)+'_V'+str(ValidID)+'_R'+str(runs)+".ckpt"))
                except:
                    traceback.print_exc()
            
            if log:
                tlog=open('./logs/depressrunninglogvggface_innerface.txt','a')
                tlog.write("Run%02d\tFinalLoss:%.10f\tValidationAccuracy:%.8f\tTestAccuracy:%.8f\tOverAllACC:%0.8f\tTimeComsumed:%08.6f\tInitialLearningRate:%.8f\tFinalLearningRate:%.8f\tLearningStepForDroppingMagnitude:%08d\tTotalIterations:%08d\tEpoches:%08d\tcurrentBatchSize:%05d\tinitialBatchSize:%05d\tInput:%s\t%s\tTime:%s\tDataFile:%s\n"%((runs+1), 
                                                                                                                                                                                                                                                                                                                    np_loss, v_accuracy, ac, oaa, (t2-t1), learningRate,currentLR, lrstep,IterationsC,data.train.epochs_completed,currentBatchSize,batchSize,
                                                                                                                                                                                                                                                                                                                    str(sys.argv),str(confu_mat),time.strftime('%Y%m%d%H%M%S',time.localtime(t2)),dfile))
                tlog.close()
                bslrf=open(logbslr,'a')
                bslrf.write("Run%02d\tFinalLoss:%.10f\tValidationAccuracy:%.8f\tTestAccuracy:%.8f\tOverAllACC:%0.8f\tTimeComsumed:%08.6f\tInitialLearningRate:%.8f\tFinalLearningRate:%.8f\tLearningStepForDroppingMagnitude:%08d\tTotalIterations:%08d\tEpoches:%08d\tcurrentBatchSize:%05d\tinitialBatchSize:%05d\tInput:%s\t%s\tTime:%s\t%s\n"%((runs+1), 
                                                                                                                                                                                                                                                                                                                     np_loss, v_accuracy, ac, oaa, (t2-t1), learningRate,currentLR, lrstep,IterationsC,data.train.epochs_completed,currentBatchSize,batchSize,
                                                                                                                                                                                                                                                                                                                     str(sys.argv),str(confu_mat),time.strftime('%Y%m%d%H%M%S',time.localtime(t2)),dfile))
                bslrf.close()
            '''MODULE4 ENDS---------------------------------------------------------------------------------------------'''
    #
    #
    #
    elif Module==2:
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
            currentLR=0.00001
            learningRate=0.00001
        else:
            print("Usage: python finetune.py <GPUID> <Module> <NetworkType>\nWith Module 2, NetworkType must be 0, 1, 2")
            exit(-1)
        '''Here begins the implementation logic-------------------------------------------------------------------
        -------------------------------------------------------------------------------------------------------------'''
        #Holder for geometry features with 122 in a batch size of batch_size
        if D_f:
            geo_features = tf.placeholder(tf.float32, [None, 122, 1])
        else:
            geo_features = tf.placeholder(tf.float32, [None, 122])
        #Holder for labels in a batch size of batch_size, number of labels are to be determined
        labels = tf.placeholder(tf.float32, labelshape)#the number of labels are to be determined
        LR=tf.placeholder(tf.float32, shape=[])
        Geometry_net = GeN({'data':geo_features})
        print(type(Geometry_net))
        fc7 = Geometry_net.layers['gefc7']
        pred=tf.nn.softmax(fc7)

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=pred),0)
        optm=tf.train.RMSPropOptimizer(currentLR)
        #optm=tf.train.RMSPropOptimizer(LR)
        train_op=optm.minimize(loss)#for train

        #for test
        correcta_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(labels,1))
        accuracy = tf.reduce_mean(tf.cast(correcta_prediction, "float"))
        
 
        with tf.Session() as sess:

            tf.summary.scalar('Cross_entropy', loss)
            tf.summary.scalar('Accuracy', accuracy)
            merged=tf.summary.merge_all()
            trainw=tf.summary.FileWriter(summary_dir, sess.graph)

            sess.run(tf.global_variables_initializer())
            for i in range(IterationsC):
                batch=data.train.next_batch(currentBatchSize, shuffle=False)
                np_loss, np_pred, sies, _=sess.run([loss, pred, merged, train_op], feed_dict={geo_features:batch[1], labels:batch[5], LR:currentLR})
                print("Current Learning Rate: %.8f\tIteration: %08d    Loss: %f    miniLoss:%f"%(currentLR,i, np_loss, mini_loss))
                if Show_Summaries:#write summaries in the file for tensorboard visualization
                    trainw.add_summary(sies, i)
                if np_loss<mini_loss:
                    mini_loss=np_loss
                if i%50 == 0:
                    tt=time.time()
                    v_accuracy = accuracy.eval(feed_dict={geo_features:data.validation.geometry, labels:data.validation.labels})
                    print('Network type: %s\nIteration: %d\tValidation Accuracy: %f\tTime used: %fs'%(type(Geometry_net),i, v_accuracy, (tt-t1)))
                if (i+1)%lrstep == 0:
                    currentLR=currentLR*lr_step
                    currentBatchSize=currentBatchSize+batchsize_step
            for j in range(c_i_c):###select a better final loss
                batch=data.train.next_batch(currentBatchSize, shuffle=False)
                np_loss, np_pred, sies, _=sess.run([loss, pred, merged, train_op], feed_dict={geo_features:batch[1], labels:batch[5], LR:currentLR})
                print("Current Learning Rate: %.8f\tIteration: %08d    Loss: %f    miniLoss:%f"%(currentLR,j+IterationsC, np_loss, mini_loss))
                if Show_Summaries:#write summaries in the file for tensorboard visualization
                    trainw.add_summary(sies, i+j)
                if (np_loss-mini_loss)<c_i_c_loss_error:
                    tt=time.time()
                    v_accuracy = accuracy.eval(feed_dict={geo_features:data.validation.geometry, labels:data.validation.labels})
                    print('Network type: %s\nIteration: %d\tValidation Accuracy: %f\tTime used: %fs'%(type(Geometry_net),j+IterationsC, v_accuracy, (tt-t1)))
                    break
                if j%50 == 0:
                    tt=time.time()
                    v_accuracy = accuracy.eval(feed_dict={geo_features:data.validation.geometry, labels:data.validation.labels})
                    print('Network type: %s\nIteration: %d\tValidation Accuracy: %f\tTime used: %fs'%(type(Geometry_net),j+IterationsC, v_accuracy, (tt-t1)))
                if (j+1)%lrstep == 0:
                    currentLR=currentLR*lr_step
                    currentBatchSize=currentBatchSize+batchsize_step
            IterationsC=IterationsC+j
            tt=time.time()
            tlables=list(pred.eval(feed_dict={geo_features:data.test.geometry, labels:data.test.labels}))
            confu_mat=calR(tlables,data.test.labels)
            oaa=overAllAccuracy(confu_mat)
            print('Over All Accuracy: %f'%(oaa))
            ac=accuracy.eval(feed_dict={geo_features:data.test.geometry, labels:data.test.labels})
            print('Test Accuracy: %f\tTime used: %fs'%(ac,(tt-t1)))
            t2=time.time()
            print('Time comsumed: %fs\t%s'%(t2-t1,time.strftime('%Y%m%d%H%M%S',time.localtime(t2))))
            if saveM:
                try:
                    DataSetPrepare.saveModels(sess,("./saves/"+time.strftime('%Y%m%d%H%M%S',time.localtime(t2))+'_D'+str(DataSet)+'_M'+str(Module)+'_N'+str(NetworkType)+'_T'+str(TestID)+'_V'+str(ValidID)+'_R'+str(runs)+".ckpt"))
                except:
                    traceback.print_exc()
            
            if log:
                tlog=open('./logs/depressrunninglogGeometry.txt','a')
                tlog.write("Run%02d\tFinalLoss:%.10f\tValidationAccuracy:%.8f\tTestAccuracy:%.8f\tOverAllACC:%0.8f\tTimeComsumed:%08.6f\tInitialLearningRate:%.8f\tFinalLearningRate:%.8f\tLearningStepForDroppingMagnitude:%08d\tTotalIterations:%08d\tEpoches:%08d\tcurrentBatchSize:%05d\tinitialBatchSize:%05d\tInput:%s\t%s\tTime:%s\tDataFile:%s\n"%((runs+1), 
                                                                                                                                                                                                                                                                                                                    np_loss, v_accuracy, ac, oaa, (t2-t1), learningRate,currentLR, lrstep,IterationsC,data.train.epochs_completed,currentBatchSize,batchSize,
                                                                                                                                                                                                                                                                                                                    str(sys.argv),str(confu_mat),time.strftime('%Y%m%d%H%M%S',time.localtime(t2)),dfile))
                tlog.close()
                bslrf=open(logbslr,'a')
                bslrf.write("Run%02d\tFinalLoss:%.10f\tValidationAccuracy:%.8f\tTestAccuracy:%.8f\tOverAllACC:%0.8f\tTimeComsumed:%08.6f\tInitialLearningRate:%.8f\tFinalLearningRate:%.8f\tLearningStepForDroppingMagnitude:%08d\tTotalIterations:%08d\tEpoches:%08d\tcurrentBatchSize:%05d\tinitialBatchSize:%05d\tInput:%s\t%s\tTime:%s\t%s\n"%((runs+1), 
                                                                                                                                                                                                                                                                                                                     np_loss, v_accuracy, ac, oaa, (t2-t1), learningRate,currentLR, lrstep,IterationsC,data.train.epochs_completed,currentBatchSize,batchSize,
                                                                                                                                                                                                                                                                                                                     str(sys.argv),str(confu_mat),time.strftime('%Y%m%d%H%M%S',time.localtime(t2)),dfile))
                bslrf.close()
            '''MODULE2 ENDS---------------------------------------------------------------------------------------------'''
    #
    #
    #
    elif Module==3:
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
        else:
            print("Usage: python finetune.py <GPUID> <Module> <NetworkType>\nWith Module 2, NetworkType must be 0, 1")
            exit(-1)
        learningRate=0.00005
        currentLR=0.00005
        '''Here begins the implementation logic-------------------------------------------------------------------
        -------------------------------------------------------------------------------------------------------------'''
        #Holders for gray images with 64*26 in a batch size of batch_size
        eye_p = tf.placeholder(tf.float32, [None, 64, 26, 1])
        midd_p = tf.placeholder(tf.float32, [None, 64, 26, 1])
        mou_p = tf.placeholder(tf.float32, [None, 64, 26, 1])
        #Holder for labels in a batch size of batch_size, number of labels are to be determined
        labels = tf.placeholder(tf.float32, labelshape)#the number of labels are to be determined
        LR=tf.placeholder(tf.float32, shape=[])
        FacePatch_net = PaN({'eyePatch_data':eye_p, 'middlePatch_data':midd_p, 'mouthPatch_data':mou_p})
        print(type(FacePatch_net))
        fc7 = FacePatch_net.layers['fc7']
        pred=tf.nn.softmax(fc7)

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=pred),0)
        optm=tf.train.RMSPropOptimizer(currentLR)
        #optm=tf.train.RMSPropOptimizer(LR)
        train_op=optm.minimize(loss)#for train

        #for test
        correcta_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(labels,1))
        accuracy = tf.reduce_mean(tf.cast(correcta_prediction, "float"))
    

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(IterationsC):
                batch=data.train.next_batch(currentBatchSize, shuffle=False)
                np_loss, np_pred, _=sess.run([loss, pred, train_op], feed_dict={eye_p:batch[2], midd_p:batch[3], mou_p:batch[4], labels:batch[5], LR:currentLR})
                print("Current Learning Rate: %.8f\tIteration: %08d    Loss: %f    miniLoss:%f"%(currentLR,i, np_loss, mini_loss))
                if np_loss<mini_loss:
                    mini_loss=np_loss
                if i%50 == 0:
                    tt=time.time()
                    v_accuracy = accuracy.eval(feed_dict={eye_p:data.validation.eyep, 
                                                            midd_p:data.validation.middlep, 
                                                            mou_p:data.validation.mouthp, 
                                                            labels:data.validation.labels})
                    print('Network type: %s\nIteration: %d\tValidation Accuracy: %f\tTime used: %fs'%(type(FacePatch_net),i, v_accuracy, (tt-t1)))
                if (i+1)%lrstep == 0:
                    currentLR=currentLR*lr_step
                    currentBatchSize=currentBatchSize+batchsize_step
            for j in range(c_i_c):###select a better final loss
                batch=data.train.next_batch(currentBatchSize, shuffle=False)
                np_loss, np_pred, _=sess.run([loss, pred, train_op], feed_dict={eye_p:batch[2], midd_p:batch[3], mou_p:batch[4], labels:batch[5], LR:currentLR})
                print("Current Learning Rate: %.8f\tIteration: %08d    Loss: %f    miniLoss:%f"%(currentLR,j+IterationsC, np_loss, mini_loss))
                if (np_loss-mini_loss)<c_i_c_loss_error:
                    tt=time.time()
                    v_accuracy = accuracy.eval(feed_dict={eye_p:data.validation.eyep, 
                                                            midd_p:data.validation.middlep, 
                                                            mou_p:data.validation.mouthp, 
                                                            labels:data.validation.labels})
                    print('Network type: %s\nIteration: %d\tValidation Accuracy: %f\tTime used: %fs'%(type(FacePatch_net),j+IterationsC, v_accuracy, (tt-t1)))
                    break
                #train_op.run(feed_dict={geo_features:batch[0], labels:batch[1]})
                if j%50 == 0:
                    tt=time.time()
                    v_accuracy = accuracy.eval(feed_dict={eye_p:data.validation.eyep, 
                                                            midd_p:data.validation.middlep, 
                                                            mou_p:data.validation.mouthp, 
                                                            labels:data.validation.labels})
                    print('Network type: %s\nIteration: %d\tValidation Accuracy: %f\tTime used: %fs'%(type(FacePatch_net),j+IterationsC, v_accuracy, (tt-t1)))
                if (j+1)%lrstep == 0:
                    currentLR=currentLR*lr_step
                    currentBatchSize=currentBatchSize+batchsize_step
            IterationsC=IterationsC+j
            tt=time.time()
            tlables=pred.eval(feed_dict={eye_p:data.test.eyep, 
                                                            midd_p:data.test.middlep, 
                                                            mou_p:data.test.mouthp, 
                                                            labels:data.test.labels})
            confu_mat=calR(tlables,data.test.labels)
            oaa=overAllAccuracy(confu_mat)
            print('Over All Accuracy: %f'%(oaa))
            ac=accuracy.eval(feed_dict={eye_p:data.test.eyep, 
                                                            midd_p:data.test.middlep, 
                                                            mou_p:data.test.mouthp, 
                                                            labels:data.test.labels})
            print('Test Accuracy: %f\tTime used: %fs'%(ac,(tt-t1)))
            t2=time.time()
            print('Time comsumed: %fs\t%s'%(t2-t1,time.strftime('%Y%m%d%H%M%S',time.localtime(t2))))
            if saveM:
                try:
                    DataSetPrepare.saveModels(sess,("./saves/"+time.strftime('%Y%m%d%H%M%S',time.localtime(t2))+'_D'+str(DataSet)+'_M'+str(Module)+'_N'+str(NetworkType)+'_T'+str(TestID)+'_V'+str(ValidID)+'_R'+str(runs)+".ckpt"))
                except:
                    traceback.print_exc()
            if log:
                tlog=open('./logs/depressrunninglogFacePatch.txt','a')
                tlog.write("Run%02d\tFinalLoss:%.10f\tValidationAccuracy:%.8f\tTestAccuracy:%.8f\tOverAllACC:%0.8f\tTimeComsumed:%08.6f\tInitialLearningRate:%.8f\tFinalLearningRate:%.8f\tLearningStepForDroppingMagnitude:%08d\tTotalIterations:%08d\tEpoches:%08d\tcurrentBatchSize:%05d\tinitialBatchSize:%05d\tInput:%s\t%s\tTime:%s\tDataFile:%s\n"%((runs+1), 
                                                                                                                                                                                                                                                                                                                    np_loss, v_accuracy, ac, oaa, (t2-t1), learningRate,currentLR, lrstep,IterationsC,data.train.epochs_completed,currentBatchSize,batchSize,
                                                                                                                                                                                                                                                                                                                    str(sys.argv),str(confu_mat),time.strftime('%Y%m%d%H%M%S',time.localtime(t2)),dfile))
                tlog.close()
                bslrf=open(logbslr,'a')
                bslrf.write("Run%02d\tFinalLoss:%.10f\tValidationAccuracy:%.8f\tTestAccuracy:%.8f\tOverAllACC:%0.8f\tTimeComsumed:%08.6f\tInitialLearningRate:%.8f\tFinalLearningRate:%.8f\tLearningStepForDroppingMagnitude:%08d\tTotalIterations:%08d\tEpoches:%08d\tcurrentBatchSize:%05d\tinitialBatchSize:%05d\tInput:%s\t%s\tTime:%s\t%s\n"%((runs+1), 
                                                                                                                                                                                                                                                                                                                     np_loss, v_accuracy, ac, oaa, (t2-t1), learningRate,currentLR, lrstep,IterationsC,data.train.epochs_completed,currentBatchSize,batchSize,
                                                                                                                                                                                                                                                                                                                     str(sys.argv),str(confu_mat),time.strftime('%Y%m%d%H%M%S',time.localtime(t2)),dfile))
                bslrf.close()
            '''MODULE3 ENDS---------------------------------------------------------------------------------------------'''
    #
    #
    #
    elif Module==5:
        '''MODULE5---------------------------------------------------------------------------------------------------- 
        Options for the fusion net of vgg inner_face and geometry input
        -------------------------------------------------------------------------------------------------------------'''
        print('Network Type: %s'%(NetworkType))
        if NetworkType==0:
            from FusionNets import VGG_Geo_NET as AGN
        if NetworkType==1:
            from FusionNets import VGG_Geo_NET_N1 as AGN
        else:
            print("Usage: python finetune.py <GPUID> <Module> <NetworkType>\nWrong NetworkType, please check the NetworkType input again.")
            exit(-1)
        '''Here begins the implementation logic-------------------------------------------------------------------
        -------------------------------------------------------------------------------------------------------------'''
        
        images = tf.placeholder(tf.float32, m4shape)
        geo_features=tf.placeholder(tf.float32, [None,122])
        #Holder for labels in a batch size of batch_size, number of labels are to be determined
        labels = tf.placeholder(tf.float32, labelshape)#the number of labels are to be determined

        LR=tf.placeholder(tf.float32, shape=[])
        
        app_geo_net = AGN({'image':images, 'geometry':geo_features})
        fc7 = app_geo_net.layers['fc7']
        pred=tf.nn.softmax(fc7)
 
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=pred),0)
        optm=tf.train.RMSPropOptimizer(currentLR)
        train_op=optm.minimize(loss)#for train

        #for test
        correcta_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(labels,1))
        accuracy = tf.reduce_mean(tf.cast(correcta_prediction, "float"))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            if loadONW:
                loadPretrainedModel(NetworkType, app_geo_net, sess,Module)

            for i in range(IterationsC):#iteration goes wrong at 287009 with no clue
                batch=data.train.next_batch(currentBatchSize, shuffle=False)
                np_loss, np_pred, _=sess.run([loss, pred, train_op],feed_dict={images:batch[6], geo_features:batch[1], labels:batch[5], LR:currentLR})
                print("Current Learning Rate: %.8f\tIteration: %08d    Loss: %f    miniLoss:%f"%(currentLR,i, np_loss, mini_loss))
                if np_loss<mini_loss:
                    mini_loss=np_loss
                if i%50 == 0:
                    tt=time.time()
                    v_accuracy = accuracy.eval(feed_dict={images:data.validation.innerf, geo_features:data.validation.geometry, labels:data.validation.labels})
                    print('Network type: %s\nIteration: %d\tValidation Accuracy: %f\tTime used: %fs'%(type(app_geo_net),i, v_accuracy, (tt-t1)))
                if (i+1)%lrstep == 0:
                    currentLR=currentLR*lr_step
                    currentBatchSize=currentBatchSize+batchsize_step
            for j in range(c_i_c):###select a better final loss
                batch=data.train.next_batch(currentBatchSize, shuffle=False)
                np_loss, np_pred, _=sess.run([loss, pred, train_op],feed_dict={images:batch[6], geo_features:batch[1], labels:batch[5], LR:currentLR})
                print("Current Learning Rate: %.8f\tIteration: %08d    Loss: %f    miniLoss:%f"%(currentLR,j+IterationsC, np_loss, mini_loss))
                if (np_loss-mini_loss)<c_i_c_loss_error:
                    tt=time.time()
                    v_accuracy = accuracy.eval(feed_dict={images:data.validation.innerf, geo_features:data.validation.geometry, labels:data.validation.labels})
                    print('Network type: %s\nIteration: %d\tValidation Accuracy: %f\tTime used: %fs'%(type(app_geo_net),j+IterationsC, v_accuracy, (tt-t1)))
                    break
                if j%50 == 0:
                    tt=time.time()
                    v_accuracy = accuracy.eval(feed_dict={images:data.validation.innerf, geo_features:data.validation.geometry, labels:data.validation.labels})
                    print('Network type: %s\nIteration: %d\tValidation Accuracy: %f\tTime used: %fs'%(type(app_geo_net),j+IterationsC, v_accuracy, (tt-t1)))
                if (j+1)%lrstep == 0:
                    currentLR=currentLR*lr_step
                    currentBatchSize=currentBatchSize+batchsize_step
            IterationsC=IterationsC+j
            tt=time.time()
            tlables=list(pred.eval(feed_dict={images:data.test.innerf, geo_features:data.test.geometry, labels:data.test.labels}))
            confu_mat=calR(tlables,data.test.labels)
            oaa=overAllAccuracy(confu_mat)
            print('Over All Accuracy: %f'%(oaa))
            ac=accuracy.eval(feed_dict={images:data.test.innerf, geo_features:data.test.geometry, labels:data.test.labels})
            print('Test Accuracy: %f\tTime used: %fs'%(ac,(tt-t1)))
            t2=time.time()
            print('Time comsumed: %fs\t%s'%(t2-t1,time.strftime('%Y%m%d%H%M%S',time.localtime(t2))))
            if saveM:
                try:
                    DataSetPrepare.saveModels(sess,("./saves/"+time.strftime('%Y%m%d%H%M%S',time.localtime(t2))+'_D'+str(DataSet)+'_M'+str(Module)+'_N'+str(NetworkType)+'_T'+str(TestID)+'_V'+str(ValidID)+'_R'+str(runs)+".ckpt"))
                except:
                    traceback.print_exc()
            
            if log:
                tlog=open('./logs/depressrunninglogvggface_innerface.txt','a')
                tlog.write("Run%02d\tFinalLoss:%.10f\tValidationAccuracy:%.8f\tTestAccuracy:%.8f\tOverAllACC:%0.8f\tTimeComsumed:%08.6f\tInitialLearningRate:%.8f\tFinalLearningRate:%.8f\tLearningStepForDroppingMagnitude:%08d\tTotalIterations:%08d\tEpoches:%08d\tcurrentBatchSize:%05d\tinitialBatchSize:%05d\tInput:%s\t%s\tTime:%s\tDataFile:%s\n"%((runs+1), 
                                                                                                                                                                                                                                                                                                                    np_loss, v_accuracy, ac, oaa, (t2-t1), learningRate,currentLR, lrstep,IterationsC,data.train.epochs_completed,currentBatchSize,batchSize,
                                                                                                                                                                                                                                                                                                                    str(sys.argv),str(confu_mat),time.strftime('%Y%m%d%H%M%S',time.localtime(t2)),dfile))
                tlog.close()
                bslrf=open(logbslr,'a')
                bslrf.write("Run%02d\tFinalLoss:%.10f\tValidationAccuracy:%.8f\tTestAccuracy:%.8f\tOverAllACC:%0.8f\tTimeComsumed:%08.6f\tInitialLearningRate:%.8f\tFinalLearningRate:%.8f\tLearningStepForDroppingMagnitude:%08d\tTotalIterations:%08d\tEpoches:%08d\tcurrentBatchSize:%05d\tinitialBatchSize:%05d\tInput:%s\t%s\tTime:%s\t%s\n"%((runs+1), 
                                                                                                                                                                                                                                                                                                                     np_loss, v_accuracy, ac, oaa, (t2-t1), learningRate,currentLR, lrstep,IterationsC,data.train.epochs_completed,currentBatchSize,batchSize,
                                                                                                                                                                                                                                                                                                                     str(sys.argv),str(confu_mat),time.strftime('%Y%m%d%H%M%S',time.localtime(t2)),dfile))
                bslrf.close()
            '''MODULE5 ENDS---------------------------------------------------------------------------------------------'''

    #
    #
    #
    elif Module==6:
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
        #    #exit()

        #define appearance graph
        app_G=tf.Graph()
        with app_G.as_default():
            images = tf.placeholder(tf.float32, m1shape)
            app_net=APPN({'data':images})
            appfc=app_net.layers['fc2']
            asaver = tf.train.Saver()
        
        ##define fine-tuning graph
        #fint_G=tf.Graph()
        #with fint_G.as_default():
        #    geo_fc=tf.placeholder(tf.float32, [None, geonfcdim])
        #    app_fc=tf.placeholder(tf.float32, [None, appnfcdim])
        #    labels = tf.placeholder(tf.float32, labelshape)#the number of labels are to be determined
        #    fin_net=FTN({'appfc':app_fc, 'geofc':geo_fc})
        #    LR=tf.placeholder(tf.float32, shape=[])
        #    fc7 = fin_net.layers['fin_fc7']
        #    pred=tf.nn.softmax(fc7)
        #    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=pred),0)
        #    optm=tf.train.RMSPropOptimizer(currentLR)
        #    train_op=optm.minimize(loss)#for train
        #    #for test
        #    correcta_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(labels,1))
        #    accuracy = tf.reduce_mean(tf.cast(correcta_prediction, "float"))
        #    #print(fint_G.get_all_collection_keys())
        #    #print(fint_G.get_collection(name='variables'))
        #    #print(fint_G.get_collection(name='train_op'))
        #    #print(fint_G.get_collection(name='trainable_variables'))
        ##exit()
        
        #print('Geometry graph at: \t\t', geo_G)
        #print('Appearance graph at: \t\t', app_G)
        #print('Fine-tuning graph at: \t\t', fint_G)
        #exit()
        #different sessions have different graph
        geo_sess=tf.InteractiveSession(graph=geo_G)
        app_sess=tf.InteractiveSession(graph=app_G)
        #fin_sess=tf.InteractiveSession(graph=fint_G)
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
            #exit(1)
        
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
        #exit()
        #try:
        #    #besides the variables, the optimizer also need to be initialized.
        #    #fin_sess.run(tf.variables_initializer(var_list=fint_G.get_collection(name='trainable_variables')))
        #    fin_sess.run(tf.variables_initializer(var_list=fint_G.get_collection(name='variables')))
        #    print('\nFine-tuning network variables initialized.')
        #except:
        #    print('Unable to initialize Fine-tuning network variables')
        #    traceback.print_exc()
        #    exit(3)

        #for i in range(IterationsC):#iteration goes wrong at 287009 with no clue
        #    batch=data.train.next_batch(currentBatchSize, shuffle=False)
        #    d1geofc=geo_sess.run(geofc, feed_dict={geo_features:batch[1]})
        #    d2appfc=app_sess.run(appfc, feed_dict={images:batch[0]})
        #    np_loss, np_pred, _=fin_sess.run([loss, pred, train_op],feed_dict={geo_fc:d1geofc, app_fc:d2appfc, labels:batch[5], LR:currentLR})
        #    print("Current Learning Rate: %.8f\tIteration: %08d    Loss: %f    miniLoss:%f"%(currentLR,i, np_loss, mini_loss))
        #    if np_loss<mini_loss:
        #        mini_loss=np_loss
        #    if i%50 == 0:
        #        tt=time.time()
        #        d1geofc_v=geo_sess.run(geofc, feed_dict={geo_features:data.validation.geometry})
        #        d2appfc_v=app_sess.run(appfc, feed_dict={images:data.validation.res_images})
        #        v_accuracy = (fin_sess.run([accuracy], feed_dict={geo_fc:d1geofc_v, app_fc:d2appfc_v, labels:data.validation.labels}))[0]
        #        print('Network type: %s\nIteration: %d\tValidation Accuracy: %f\tTime used: %fs'%(type(fin_net),i, v_accuracy, (tt-t1)))
        #    if (i+1)%lrstep == 0:
        #        currentLR=currentLR*lr_step
        #        currentBatchSize=currentBatchSize+batchsize_step
        #for j in range(c_i_c):###select a better final loss
        #    batch=data.train.next_batch(currentBatchSize, shuffle=False)
        #    d1geofc=geo_sess.run(geofc, feed_dict={geo_features:batch[1]})
        #    d2appfc=app_sess.run(appfc, feed_dict={images:batch[0]})
        #    np_loss, np_pred, _=fin_sess.run([loss, pred, train_op],feed_dict={geo_fc:d1geofc, app_fc:d2appfc, labels:batch[5], LR:currentLR})
        #    print("Current Learning Rate: %.8f\tIteration: %08d    Loss: %f    miniLoss:%f"%(currentLR,i, np_loss, mini_loss))
        #    if (np_loss-mini_loss)<c_i_c_loss_error:
        #        d1geofc_v=geo_sess.run(geofc, feed_dict={geo_features:data.validation.geometry})
        #        d2appfc_v=app_sess.run(appfc, feed_dict={images:data.validation.res_images})
        #        v_accuracy = (fin_sess.run([accuracy], feed_dict={geo_fc:d1geofc_v, app_fc:d2appfc_v, labels:data.validation.labels}))[0]
        #        print('Network type: %s\nIteration: %d\tValidation Accuracy: %f\tTime used: %fs'%(type(fin_net),i, v_accuracy, (tt-t1)))
        #        break
        #    if j%50 == 0:
        #        tt=time.time()
        #        d1geofc_v=geo_sess.run(geofc, feed_dict={geo_features:data.validation.geometry})
        #        d2appfc_v=app_sess.run(appfc, feed_dict={images:data.validation.res_images})
        #        v_accuracy = (fin_sess.run([accuracy], feed_dict={geo_fc:d1geofc_v, app_fc:d2appfc_v, labels:data.validation.labels}))[0]
        #        print('Network type: %s\nIteration: %d\tValidation Accuracy: %f\tTime used: %fs'%(type(fin_net),i, v_accuracy, (tt-t1)))
        #    if (j+1)%lrstep == 0:
        #        currentLR=currentLR*lr_step
        #        currentBatchSize=currentBatchSize+batchsize_step
        #IterationsC=IterationsC+j
        #tt=time.time()
        
        d1geofc_v=geo_sess.run(geofc, feed_dict={geo_features:data.test.geometry})
        d2appfc_v=app_sess.run(appfc, feed_dict={images:data.test.res_images})
        
        labels=data.test.labels
        print(type(labels))
        assert len(d2appfc_v)==len(labels), ('ERROR')
        insts=[]
        for i, v in enumerate(d2appfc_v):
            inst=[]
            inst.append(labels[i])
            inst.extend(v)
            inst.extend(d1geofc_v[i])
            insts.append(inst)
        vggfc4096geofc1024=np.array(insts)
        print(vggfc4096geofc1024.shape)
        filename='ckplus8G_label+vggfc4096+geometryfc1024_g%d.pkl'%(TestID)
        with open(filename,'wb') as fin:
            pickle.dump(vggfc4096geofc1024,fin,4)
        print('File saved. %s'%(filename))


        #tlables=list((fin_sess.run([pred], feed_dict={geo_fc:d1geofc_v, app_fc:d2appfc_v, labels:data.test.labels}))[0])
        #confu_mat=calR(tlables,data.test.labels)
        #oaa=overAllAccuracy(confu_mat)
        #print('Over All Accuracy: %f'%(oaa))
        
        #ac=(fin_sess.run([accuracy], feed_dict={geo_fc:d1geofc_v, app_fc:d2appfc_v, labels:data.test.labels}))[0]
        #print('Test Accuracy: %f\tTime used: %fs'%(ac,(tt-t1)))
        #t2=time.time()
        #print('Time comsumed: %fs\t%s'%(t2-t1,time.strftime('%Y%m%d%H%M%S',time.localtime(t2))))
        #if saveM:
        #    try:
        #        DataSetPrepare.saveModels(sess,("./saves/"+time.strftime('%Y%m%d%H%M%S',time.localtime(t2))+'_D'+str(DataSet)+'_M'+str(Module)+'_N'+str(NetworkType)+'_T'+str(TestID)+'_V'+str(ValidID)+'_R'+str(runs)+".ckpt"))
        #    except:
        #        traceback.print_exc()
            
        #if log:
        #    tlog=open('./logs/depressrunninglog_AppandGeo_fusion.txt','a')
        #    tlog.write("Run%02d\tFinalLoss:%.10f\tValidationAccuracy:%.8f\tTestAccuracy:%.8f\tOverAllACC:%0.8f\tTimeComsumed:%08.6f\tInitialLearningRate:%.8f\tFinalLearningRate:%.8f\tLearningStepForDroppingMagnitude:%08d\tTotalIterations:%08d\tEpoches:%08d\tcurrentBatchSize:%05d\tinitialBatchSize:%05d\tInput:%s\t%s\tTime:%s\tDataFile:%s\n"%((runs+1), 
        #                                                                                                                                                                                                                                                                                                            np_loss, v_accuracy, ac, oaa, (t2-t1), learningRate,currentLR, lrstep,IterationsC,data.train.epochs_completed,currentBatchSize,batchSize,
        #                                                                                                                                                                                                                                                                                                            str(sys.argv),str(confu_mat),time.strftime('%Y%m%d%H%M%S',time.localtime(t2)),dfile))
        #    tlog.close()
        #    bslrf=open(logbslr,'a')
        #    bslrf.write("Run%02d\tFinalLoss:%.10f\tValidationAccuracy:%.8f\tTestAccuracy:%.8f\tOverAllACC:%0.8f\tTimeComsumed:%08.6f\tInitialLearningRate:%.8f\tFinalLearningRate:%.8f\tLearningStepForDroppingMagnitude:%08d\tTotalIterations:%08d\tEpoches:%08d\tcurrentBatchSize:%05d\tinitialBatchSize:%05d\tInput:%s\t%s\tTime:%s\t%s\n"%((runs+1), 
        #                                                                                                                                                                                                                                                                                                            np_loss, v_accuracy, ac, oaa, (t2-t1), learningRate,currentLR, lrstep,IterationsC,data.train.epochs_completed,currentBatchSize,batchSize,
        #                                                                                                                                                                                                                                                                                                            str(sys.argv),str(confu_mat),time.strftime('%Y%m%d%H%M%S',time.localtime(t2)),dfile))
        #    bslrf.close()
        '''MODULE6 ENDS---------------------------------------------------------------------------------------------'''
gr=sys.argv[1]#change the group
run(GPU_Device_ID=0, Module=6, DataSet=5,
        ValidID=int(gr),TestID=int(gr), 
        NetworkType=440, runs=0,learningRate=0.0001,IterationsC=19001,lrstep=3000,batchSize=30,saveM=False,log=False,loadONW=False)