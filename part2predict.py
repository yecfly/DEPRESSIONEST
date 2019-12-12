#Here starts the depression estimation processes
import numpy as np
import tensorflow as tf
import os, time, traceback, ntpath, glob, pickle
import DataSetPrepare
import VGG_NET
from sklearn import metrics
from matplotlib import pyplot as plt

Dataset_Dictionary={1001:'./Datasets\D1001_MergeDataset_D501_D531_10G.pkl'
	, 1002:'./Datasets\D1002_MergeDataset_D502_D532_10G.pkl'
	, 10:'./Datasets\D10_CKplus_10groups_groupedbythe_CKplus-group-details_preprocessdata_with_calibRotation_rescaleimg_geometricfeatures_facepatches_weberface_skip-contempV2.pkl'
	, 11:'./Datasets\D11_CKplus_10groups_groupedbythe_CKplus-group-details_preprocessdata_with_calibRotation_rescaleimg224x224_geometricfeatures_facepatches_weberface224x224_skip-contempV2.pkl'
	, 12:'./Datasets\D12_CKplus_10groups_groupedbythe_CKplus-group-details_preprocessdata_with_calibRotation_rescaleimg_geometricfeatures_facepatches_ELTFS_skip-contempV2.pkl'
	, 13:'./Datasets\D13_CKplus_8G_V4_Geo258_ELTFS128x128.pkl'
	, 16:'./Datasets\D16_CKPLUS_10G_Enlargeby2015CCV_10T.pkl'
	, 17:'./Datasets\D17_CKplus_10G_V4_weberface128x128.pkl'
	, 18:'./Datasets\D18_CKplus_10G_V5_formalized_weberface128x128.pkl'
	, 19:'./Datasets\D19_CKplus_10G_V4_ELTFS128x128.pkl'
	, 2:'./Datasets\D2_CKplus_8groups_groupedbythe_CKplus-group-details_preprocessdata_with_calibRotation_rescaleimgnewmetric0731_skip-contempV2.pkl'
	, 33:'./Datasets\D33_KDEF_10G_rescaleimg_geometryfeature_patches_web.pkl'
	, 34:'./Datasets\D34_KDEF_10G_Enlargeby2015CCV_10T.pkl'
	, 3:'./Datasets\D3_CKplus_8groups_groupedbythe_CKplus-group-details_preprocessdata_with_calibRotation_rescaleimg_geometricfeatures_facepatches_webberface_skip-contempV2.pkl'
	, 40:'./Datasets\D40_jaffe_10groups_groupedbysubjects_rescaleimg_geometricfeatures_facepatches_weber.pkl'
	, 43:'./Datasets\D43_JAFFE_10G_Enlargeby2015CCV_10T.pkl'
	, 44:'./Datasets\D44_jaffe_10G_V4_weber128x128.pkl'
	, 4:'./Datasets\D4_CKplus_8groups_groupedbythe_CKplus-group-details_preprocessdata_with_calibRotation_rescaleimg_geometricfeatures_facepatches_webberfaceReverse_skip-contempV2.pkl'
	, 501:'./Datasets\D501_CKplus_10G_V5_newGeo_newPatch.pkl'
	, 502:'./Datasets\D502_CKPLUS_10G_Enlargeby2015ICCV_V5_newGeo_newPatches.pkl'
	, 503:'./Datasets\D503_CKplus_8G_V5_newGeo_newPatch.pkl'
	, 531:'./Datasets\D531_KDEF_10G_V5_newGeo_newPatch.pkl'
	, 532:'./Datasets\D532_KDEF_10G_Enlargeby2015ICCV_V5_newGeo_newPatches.pkl'
	, 551:'./Datasets\D551_OuluCASIA_Weak_10G_V5_newGeo_newPatch.pkl'
	, 552:'./Datasets\D552_OuluCASIA_Strong_10G_V5_newGeo_newPatch.pkl'
	, 553:'./Datasets\D553_OuluCASIA_Dark_10G_V5_newGeo_newPatch.pkl'
	, 554:'./Datasets\D554_MergeDataset_D551_D553_D552_10G_OuluCASIA_Weak_Dark_Strong_10G_V5_newGeo_newPatch.pkl'
	, 5:'./Datasets\D5_CKplus_8groups_groupedbythe_CKplus-group-details_preprocessdata_with_calibRotation_rescaleimg_geometricfeatures_facepatches_webberface25up_skip-contempV2.pkl'
	, 600:'./Datasets\D600_Collectedbywlc_filteredby435_data_with_geometry_and_facepatches.pkl'
	, 601:'./Datasets\D601_Collectedbywlc_filteredby435_data_with_geometry_and_facepatches_FF.pkl'
	, 6:'./Datasets\D6_CKplus_8groups_groupedbythe_CKplus-group-details_preprocessdata_with_calibRotation_rescaleimg_geometricfeatureV2_skip-contempV2.pkl'
	, 700:'./Datasets\D700_front_with827samples_data_with_geometry_and_facepatches.pkl'
	, 7:'./Datasets\D7_CKplus_8groups_groupedbythe_CKplus-group-details_preprocessdata_with_calibRotation_rescaleimg_geometricfeatures_facepatches_webberface_innerfaceSizew36xh48_skip-contempV2.pkl'
	, 8:'./Datasets\D8_CKplus_8groups_groupedbythe_CKplus-group-details_preprocessdata_with_calibRotation_rescaleimg_geometricfeatures_facepatches_ELTFS_skip-contempV2.pkl'
	, 99999:'./Datasets\D99999_MergeDataset_data_undetected_1G.pkl'
	, 9:'./Datasets\D9_CKplus_8groups_groupedbythe_CKplus-group-details_preprocessdata_with_calibRotation_rescaleimg224x224_geometricfeatures_facepatches_weberface224x224_skip-contempV2.pkl'
	}
TestNumLimit=200
test_bat=100
reverse=np.asarray([[0], [1], [2], [3], [4], [5], [6]])
MAP = {'neutral':0, 'angry':1, 'surprise':2, 'disgust':3, 'fear':4, 'happy':5, 'sad':6}
labels = ['neutral', 'angry', 'surprise', 'disgust', 'fear', 'happy', 'sad']
RMAP = {0:'neutral', 1:'angry', 2:'surprise', 3:'disgust', 4:'fear', 5:'happy', 6:'sad'}
def plot_confusion_matrix(cm, savedir, title='Confusion Matrix', cmap = plt.cm.binary):
    global labels
    plt.figure(figsize=(8.8, 6.4))
    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if x_val==y_val:
            if (c > 0.001):
                plt.text(x_val, y_val, "%0.3f" %(c,), color='red', fontsize=14, va='center', ha='center')
            elif c>0:
                plt.text(x_val, y_val, "%0.3f" %(c,), color='red', fontsize=14, va='center', ha='center')
            else:
                plt.text(x_val, y_val, "%d" %(c,), color='red', fontsize=14, va='center', ha='center')
        else:
            if (c > 0.001):
                plt.text(x_val, y_val, "%0.3f" %(c,), color='blue', fontsize=14, va='center', ha='center')
            elif c>0:
                plt.text(x_val, y_val, "%0.3f" %(c,), color='blue', fontsize=14, va='center', ha='center')
            else:
                plt.text(x_val, y_val, "%d" %(c,), color='blue', fontsize=14, va='center', ha='center')
    tick_marks = np.array(range(len(labels)))+1.0
    plt.gca().set_xticks(tick_marks, minor = True)
    plt.gca().set_yticks(tick_marks, minor = True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    #plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels)
    #plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('GroundTruth')
    plt.xlabel('Predict')
    plt.savefig(os.path.join(savedir,'confusion_matrix.jpg'))
def calR(predict_labels_in, groundtruth_labels_in):
    #print(len(predict_labels_in.shape))
    #print(len(predict_labels_in))
    #print(len(np.asarray(groundtruth_labels_in).shape))
    #print(len(groundtruth_labels_in))
    #exit()
    if len(np.asarray(predict_labels_in).shape)==1:
        predict_labels=DataSetPrepare.dense_to_one_hot(predict_labels_in, 7)
        #print(predict_labels.shape)
    else:
        predict_labels=predict_labels_in
    if len(np.asarray(groundtruth_labels_in).shape)==1:
        groundtruth_labels=DataSetPrepare.dense_to_one_hot(groundtruth_labels_in, 7)
        #print(groundtruth_labels.shape)
    else:
        groundtruth_labels=groundtruth_labels_in
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
def logImagesAndProb(predict, ground_truth, images, log_dir, confusion_matrix=None):
    import cv2
    if len(np.asarray(predict).shape)==1:
        predict_labels=DataSetPrepare.dense_to_one_hot(predict, 7)
        #print(predict_labels.shape)
    else:
        predict_labels=predict
    if len(np.asarray(ground_truth).shape)==1:
        groundtruth_labels=DataSetPrepare.dense_to_one_hot(ground_truth, 7)
        #print(groundtruth_labels.shape)
    else:
        groundtruth_labels=ground_truth
    assert len(predict_labels)==len(groundtruth_labels), ('predict_labels length: %d groundtruth_labels length: %d' % (len(predict_labels), len(groundtruth_labels)))
    nc=len(groundtruth_labels)
    false_predict=log_dir+'/'+'FalsePredict'
    true_predict=log_dir+'/'+'TruePredict'
    if not os.path.exists(false_predict):
        os.makedirs(false_predict)
    if not os.path.exists(true_predict):
        os.makedirs(true_predict)
    predict_prob_log=log_dir+'/probs_log.txt'
    predict_log=log_dir+'/log.txt'
    np.savetxt(predict_prob_log, np.asarray(predict_labels))
    print('\nPredict probabilities have been saved in %s\n'%(predict_prob_log))
    fin=open(predict_log,'w')
    for i in range(nc):
        gti=list(groundtruth_labels[i]).index(max(groundtruth_labels[i]))
        pri=list(predict_labels[i]).index(max(predict_labels[i]))
        fin.write('%05d %s %s %s\n'%(i, str(gti==pri),RMAP.get(gti), RMAP.get(pri)))
        if not images is None:
            if gti==pri:
                imagename=true_predict+'/'+'G_%s___P_%s__index%05d.jpg'%(str(RMAP.get(gti, None)),str(RMAP.get(pri, None)), i)
            else:
                imagename=false_predict+'/'+'G_%s___P_%s__index%5d.jpg'%(str(RMAP.get(gti, None)),str(RMAP.get(pri, None)), i)
            cv2.imwrite(imagename, np.reshape(images[i], (128,128))*255)
    fin.close()
    if not confusion_matrix is None:
        plot_confusion_matrix(confusion_matrix, log_dir)

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
def extractConfusionMatrix(cm):
    scm=[]
    for i in range(len(cm)):
        scm.append(cm[i][i])
    return scm
def Test_on_TestSet(sess, softmax,
                       placeholder1_imgs, placeholder1_input, 
                       placeholder_labels, placeholder_labels_input, imagelog, afc=None, saveSamples=False):
    ncount=len(placeholder_labels_input)
    '''return v_accuracy, valid_loss, oaa, confu_mat'''
    tlabels=[]
    if ncount>TestNumLimit:               
        test_iter=np.floor_divide(ncount,test_bat)
        v_accuracy=0
        for ite in range(test_iter):
            start=test_bat*ite
            end=test_bat*(ite+1)
            tlab=sess.run([softmax], feed_dict={placeholder1_imgs:placeholder1_input[start:end],
                                                            placeholder_labels:placeholder_labels_input[start:end]})
            tlabels.extend(list(tlab[0]))
        if ncount%test_bat>0:
            tlab=sess.run([softmax], feed_dict={placeholder1_imgs:placeholder1_input[test_bat*test_iter:ncount], 
                                                            placeholder_labels:placeholder_labels_input[test_bat*test_iter:ncount]})
            tlabels.extend(list(tlab[0]))
    else:
        tlabels = sess.run([softmax], feed_dict={placeholder1_imgs:placeholder1_input, 
                                                                       placeholder_labels:placeholder_labels_input})
    #print(np.asarray(placeholder_labels_input).shape)
    #print(np.asarray(tlabels).shape)
    pl=[]
    for v in tlabels:
        va=list(v)
        pl.append(va.index(max(va)))
    gt=np.reshape(np.dot(np.asarray(placeholder_labels_input), reverse),(len(placeholder_labels_input)))
    v_accuracy=metrics.accuracy_score(gt, pl)
    confu_mat=calR(tlabels, placeholder_labels_input)
    oaa=overAllAccuracy(confu_mat, afc)
    if saveSamples:
        logImagesAndProb(tlabels, placeholder_labels_input, placeholder1_input, imagelog, confu_mat)

    return v_accuracy, oaa, confu_mat
def Test_on_TestSet_3NI(sess, softmax,
                       placeholder1, placeholder1_input, 
                       placeholder2, placeholder2_input,
                       placeholder3, placeholder3_input,
                       placeholder_labels, placeholder_labels_input, imagelog, afc=None, saveSamples=False):
    '''Test the data with 3 network inputs in the session input
Inputs:
    sess:
    softmax:

    
Outputs:
    v_accuracy:
    oaa:
    confu_mat'''
    ncount=len(placeholder_labels_input)
    tlabels=[]
    if ncount>TestNumLimit:               
        test_iter=np.floor_divide(ncount,test_bat)
        v_accuracy=0
        for ite in range(test_iter):
            start=test_bat*ite
            end=test_bat*(ite+1)
            #tlab is length 1 ndarray
            tlab=sess.run([softmax], feed_dict={placeholder1:placeholder1_input[start:end],
                                                                            placeholder2:placeholder2_input[start:end],
                                                                            placeholder3:placeholder3_input[start:end],
                                                                            placeholder_labels:placeholder_labels_input[start:end]})
            tlabels.extend(list(tlab[0]))
        if ncount%test_bat>0:
            tlab=sess.run([softmax], feed_dict={placeholder1:placeholder1_input[test_bat*test_iter:ncount],
                                                                            placeholder2:placeholder2_input[test_bat*test_iter:ncount],
                                                                            placeholder3:placeholder3_input[test_bat*test_iter:ncount],
                                                                            placeholder_labels:placeholder_labels_input[test_bat*test_iter:ncount]})
            tlabels.extend(list(tlab[0]))
    else:
        tlab = sess.run([softmax], feed_dict={placeholder1:placeholder1_input, 
                                                                        placeholder2:placeholder2_input,
                                                                        placeholder3:placeholder3_input,
                                                                        placeholder_labels:placeholder_labels_input})
        tlabels.extend(list(tlab[0]))
    #print(np.asarray(placeholder_labels_input).shape)
    #print(np.asarray(tlabels).shape)
    #print(tlabels)
    pl=[]
    for v in tlabels:
        va=list(v)
        pl.append(va.index(max(va)))
    gt=np.reshape(np.dot(np.asarray(placeholder_labels_input), reverse),(len(placeholder_labels_input)))
    #print(gt)
    #print(pl)
    #print(tlabels)
    v_accuracy=metrics.accuracy_score(gt, pl)
    #confu_mat=metrics.confusion_matrix(gt,pl)
    #print(confu_mat)
    confu_mat=calR(tlabels, placeholder_labels_input)
    oaa=overAllAccuracy(confu_mat, afc)
    if saveSamples:
        logImagesAndProb(tlabels, placeholder_labels_input, None, imagelog, confu_mat)
    return v_accuracy, oaa, confu_mat
def regroupdata(Apredata):
    data={'X':[], 'Y':[]}
    for v in Apredata:
        data['X'].extend(v['X'])
        data['Y'].extend(v['Y'])
    del Apredata
    return data
overcheck=True
over_times=5

def TestPreprocessPKL(GPU_Device_ID,
        Dataset=0, 
        NetworkType=0,
        TrainDataset=0,
        Module=1, saveSamples=False, ModelName=None):
    try:
        ###GPU Option---------------------------------------------------------------------------------------------
        #Determine which GPU is going to be used
        ###------------------------------------------------------------------------------------------------------------
        print('GPU Option: %s'%(GPU_Device_ID))
        if (0==GPU_Device_ID) or (1==GPU_Device_ID):
            os.environ["CUDA_VISIBLE_DEVICES"]=str(GPU_Device_ID)
        else:
            print("Tips: GPUID must be 0 or 1")
            exit(-1)
        ###GPU Option ENDS---------------------------------------------------------------------------------------
        #
        #
        #
        ###Input Data-------------------------------------------------------------------------------------------------
        ###-------------------------------------------------------------------------------------------------------------
        t1=time.time()
        labelshape=[None, 7]
        logprefix='./logs/testlog/M%d'%(Module)
        if not os.path.exists(logprefix):
            os.makedirs(logprefix)
        logprefix=logprefix+'/TEST_log'
        m1shape= [None, 128, 128, 1]
        m4shape= [None, 96, 72, 1]
        pklname=Dataset_Dictionary.get(Dataset)
        tt=time.time()
        logbslr=logprefix+"_M"+str(Module)+"_N"+str(NetworkType)+'_D'+str(TrainDataset)+'_'+'TestOn_D'+str(Dataset)+".txt"
        errorlog=logbslr.replace('.txt','_error.txt')
        imagelog=logbslr.replace('.txt','')
        ###Input Data Ends-----------------------------------------------------------------------------------------

        ###Loading data
        ddir=("J:/Models/saves/"+'M'+str(Module)+'/D'+str(TrainDataset)+'/N'+str(NetworkType)+'/')
        if os.path.exists(ddir):
            print('Checking Pre-saved models in %s'%(ddir))
        else:
            print('ERROR Unexpected Model Path Occurred.')
            exit(-1)
        if Module<6:
            testdata=DataSetPrepare.loadPKLData_v4(pklname, Module)
            print('Time used for loading data: %fs'%(tt-t1))
            print('Network Type: %s'%(NetworkType))
            print('Samples Number: %d'%(testdata.num_examples))
        if Module==1:
            if NetworkType==4:
                from VGG_NET import VGG_NET_o as APPN
            elif NetworkType==9:
                from VGG_NET import VGG_NET_Inception2 as APPN
            elif NetworkType==10:
                from VGG_NET import VGG_NET_O_tfl as APPN
            elif NetworkType==11:
                from VGG_NET import VGG_NET_I5 as APPN
            else:
                print('ERROR: Unexpected Module and NetworkType')
                exit(-1)
            
            #define appearance graph
            app_G=tf.Graph()
            with app_G.as_default():
                images = tf.placeholder(tf.float32, m1shape)
                labels = tf.placeholder(tf.float32, labelshape)
                if NetworkType==10 or NetworkType==11:
                    pred=APPN(images)
                else:
                    app_net=APPN({'data':images})
                    pred=app_net.layers['prob']

                asaver = tf.train.Saver()


            print('Appearance graph at: \t\t', app_G)

            app_sess=tf.InteractiveSession(graph=app_G)

            print('\n%%%%%%%Sessions are created\n')
            if ModelName is None:
                modellist=glob.glob(os.path.join(ddir,'*.index'))
            else:
                modellist=[]
                modellist.append(ModelName)
            print(modellist)
            for v in modellist:
                modelname=v.replace('.index','')
                imglog=imagelog+'_'+os.path.basename(modelname)
                print('\n\nMODEL: %s'%(modelname))
                
                try:
                    #must initialize the variables in the graph for compution or loading pretrained weights
                    app_sess.run(tf.variables_initializer(var_list=app_G.get_collection(name='variables')))
                    print('\nAppearance network variables initialized.')
                    #the asaver must define in the graph of its owner session, or it will occur error in restoration or saving
                    asaver.restore(sess=app_sess, save_path=modelname)
                    print('Appearance Model loaded from %s\n'%(modelname))
                except:
                    print('ERROR: Unable to load the pretrained network from %s'%(modelname))
                    traceback.print_exc()
                    continue

                ncount=testdata.num_examples
                print('Total test samples: %d'%(ncount))
                iter=1
                if overcheck:
                    if NetworkType==11 or NetworkType==10:
                        iter=1
                    else:
                        iter=over_times
                if saveSamples:
                    iter=1
                for i in range(iter):
                    scm=[]
                    try:
                        v_accuracy, oaa, cm=Test_on_TestSet(app_sess, pred, images, testdata.res_images, labels, testdata.labels, imglog, afc=scm, saveSamples=saveSamples)
                    except:
                        bslrf=open(errorlog,'a')
                        bslrf.write('\n\nR%02d Networktype: %s\nPKLFile: %s\nModelName: %s\n'%(i+1, str(NetworkType),
                                                                                                                                    pklname, modelname))
                        traceback.print_exc(file=bslrf)
                        traceback.print_exc()
                        print('UNEXPECTED CASE OCCURRED.\nSKIP to next model.')
                        break
                    tt=time.time()
                    print('R%02d Network type: %s\tTest Accuracy: %f\tOAA: %f\nACs: %s\tTime used: %fs'%(i+1, str(NetworkType),v_accuracy, oaa, str(scm), (tt-t1)))
  
                    bslrf=open(logbslr,'a')
                    bslrf.write('R%02d Networktype: %s TestAccuracy: %f OverAllAccuracy: %f ACs: %s TimeUsed: %f PKLFile: %s ModelName: %s ConfusionMat: %s Timestamp: %s\n'%(i+1, str(NetworkType),
                                                                                                                                    v_accuracy, oaa, str(scm), (tt-t1), 
                                                                                                                                    pklname, modelname, str(cm), time.strftime('%Y%m%d%H%M%S',time.localtime(tt))))
                    bslrf.close()    

        ########Module 3
        elif Module==3:
            if NetworkType==5:
                from FacePatches_NET import FacePatches_NET_3Conv_2Inception_tflearn as FPN
            elif NetworkType==4:
                from FacePatches_NET import FacePatches_NET_3Conv_IInception_tflear as FPN
            elif NetworkType==6:
                from FacePatches_NET import FacePatches_NET_3Conv_3Inception_tflearn as FPN
            elif NetworkType==24:
                from FacePatches_NET import FacePatches_NET_3C_1I_2P as FPN
            elif NetworkType==25:
                from FacePatches_NET import FacePatches_NET_3C_2I_2P as FPN
            elif NetworkType==26:
                from FacePatches_NET import FacePatches_NET_3C_3I_2P as FPN
            else:
                print('ERROR: Unexpected Module and NetworkType')
                exit(-1)
            
            #define appearance graph
            fp_G=tf.Graph()
            with fp_G.as_default():
                eye_p_shape=[None, 26, 64, 1]
                midd_p_shape=[None, 49, 28, 1]
                mou_p_shape=[None, 30, 54, 1]

                eye_p = tf.placeholder(tf.float32, eye_p_shape)
                midd_p = tf.placeholder(tf.float32, midd_p_shape)
                mou_p = tf.placeholder(tf.float32, mou_p_shape)
                labels = tf.placeholder(tf.float32, labelshape)

                #pred=FPN(eye_p, midd_p, mou_p)
                if NetworkType > 3 and NetworkType < 7:###current 4 5 6
                    pred=FPN(eye_p, midd_p, mou_p)
                elif NetworkType >23 and NetworkType <27:###using only eye patch and mouth patch
                    pred=FPN(eye_p, mou_p)
                else:
                    print('ERROR Unexpected Networktype occur.')
                    exit()
                asaver = tf.train.Saver()

            print('Appearance graph at: \t\t', fp_G)

            fp_sess=tf.InteractiveSession(graph=fp_G)

            print('\n%%%%%%%Sessions are created\n')

            if ModelName is None:
                modellist=glob.glob(os.path.join(ddir,'*.index'))
            else:
                modellist=[]
                modellist.append(ModelName)
            print(modellist)
            for v in modellist:
                modelname=v.replace('.index','')
                imglog=imagelog+'_'+os.path.basename(modelname)
                print('\n\nMODEL: %s'%(modelname))

                try:
                    #must initialize the variables in the graph for compution or loading pretrained weights
                    fp_sess.run(tf.variables_initializer(var_list=fp_G.get_collection(name='variables')))
                    print('\nAppearance network variables initialized.')
                    #the asaver must define in the graph of its owner session, or it will occur error in restoration or saving
                    asaver.restore(sess=fp_sess, save_path=modelname)
                    print('FacePatch Model loaded from %s\n'%(modelname))
                except:
                    print('ERROR: Unable to load the pretrained network from %s'%(modelname))
                    traceback.print_exc()
                    continue

                ncount=testdata.num_examples
                print('Total test samples: %d'%(ncount))
                iter=1
                if overcheck:
                    if NetworkType>3:
                        iter=2
                    else:
                        iter=over_times
                if saveSamples:
                    iter=1
                for i in range(iter):
                    scm=[]
                    try:
                        v_accuracy, oaa, cm=Test_on_TestSet_3NI(fp_sess, pred, eye_p, testdata.eyep, midd_p, testdata.middlep, mou_p, testdata.mouthp, 
                                                            labels, testdata.labels, imglog, afc=scm, saveSamples=saveSamples)
                    except:
                        bslrf=open(errorlog,'a')
                        bslrf.write('\n\nR%02d Networktype: %s\nPKLFile: %s\nModelName: %s\n'%(i+1, str(NetworkType),
                                                                                                                                    pklname, modelname))
                        traceback.print_exc(file=bslrf)
                        traceback.print_exc()
                        print('UNEXPECTED CASE OCCURRED.\nSKIP to next model.')
                        exit()
                        break
                    tt=time.time()
                    print('R%02d Network type: %s\tTest Accuracy: %f\tOAA: %f\nACs: %s\tTime used: %fs'%(i+1, str(NetworkType),v_accuracy, oaa, str(scm), (tt-t1)))

                    bslrf=open(logbslr,'a')
                    bslrf.write('R%02d Networktype: %s TestAccuracy: %f OverAllAccuracy: %f ACs: %s TimeUsed: %f PKLFile: %s ModelName: %s ConfusionMat: %s Timestamp: %s\n'%(i+1, str(NetworkType),
                                                                                                                                    v_accuracy, oaa, str(scm), (tt-t1), 
                                                                                                                                    pklname, modelname, str(cm), time.strftime('%Y%m%d%H%M%S',time.localtime(tt))))
                    bslrf.close()    

        ########Module 8
        elif Module==8:
            from sklearn import metrics
            import tflearn
            if NetworkType==62:
                from FacePatches_NET import FacePatches_NET_3Conv_3Inception_tflearn as FPN
                fpndim=9216
                m3modelname='./M7models/D502_M3_N6_T2_V2_R1_20171110055149_1.18062_.ckpt'
                facepatchpreprocessdatafilename='./Pre-Datasets/D%d_N%dinM3_pre-data_with_%ddims_from_%s.pkl'%(Dataset,6,fpndim,os.path.basename(m3modelname))
            else:
                print('ERROR: Unexpected Module and NetworkType')
                exit(-1)
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
                data10g=DataSetPrepare.loadPKLDataWithPartitions_v4(Dataset_Dictionary.get(Dataset), Patches=True)
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
            data=regroupdata(Apredata)
            overtimes=1
            if overcheck:
                overtimes=1
            if ModelName is None:
                print('Checking models from %s'%(ddir))
                modellist=glob.glob(os.path.join(ddir,'*.pkl'))
            else:
                modellist=[]
                modellist.append(ModelName)
            for modelname in modellist:
                imglog=imagelog+'_'+os.path.basename(modelname)
                model=None
                with open(modelname,'rb') as fread:
                    model=pickle.load(fread)
                if model is None:
                    print('Unexpected case occurred while loading %s\nSKIP to next model'%(modelname))
                    continue
                print('\n\nChecking model: %s'%(modelname))
                print(str(model).replace('\n',' '))
                if saveSamples:
                    overtimes=1
                for i in range(overtimes):
                    tt=time.time()
                    pY=model.predict(data['X'])
                    #print(pY)
                    pbY=model.predict_proba(data['X'])
                    #print(pbY)
                    #exit()
                    v_accuracy=metrics.accuracy_score(np.asarray(data['Y']),pY)
                    cm=calR(pY, data['Y'])
                    scm=[]
                    oaa=overAllAccuracy(cm, scm)
                    print('R%02d Network type: %s\tTest Accuracy: %f\tOAA: %f\nACs: %s\tTime used: %fs'%(i+1, str(NetworkType),v_accuracy, oaa, str(scm), (tt-t1)))
                    if saveSamples:
                        logImagesAndProb(pbY, data['Y'], None, imglog, cm)
                    bslrf=open(logbslr,'a')
                    bslrf.write('R%02d Networktype: %s TestAccuracy: %f OverAllAccuracy: %f ACs: %s TimeUsed: %f PKLFile: %s ModelName: %s ConfusionMat: %s ModelConfig: %s\n Timestamp: %s'%(i+1, str(NetworkType),
                                                                                                                                    v_accuracy, oaa, str(scm), (tt-t1), 
                                                                                                                                    pklname, modelname, str(cm), str(model).replace('\n',' '), time.strftime('%Y%m%d%H%M%S',time.localtime(tt))))
                    bslrf.close()
        else:
            print("Usage: python finetune.py <GPUID> <Module> <NetworkType>\nWrong NetworkType, please check the NetworkType input again.")
            exit(-1)
        ###Here begins the implementation logic-------------------------------------------------------------------
        ###-------------------------------------------------------------------------------------------------------------
  

    except:
        traceback.print_exc()

