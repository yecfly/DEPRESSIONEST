from tensorflow.python.framework import dtypes
import numpy, collections, os, pickle, cv2
import tensorflow as tf

Dataset_Dictionary={1001:'./Datasets\D1001_MergeDataset_D501_D531_10G.pkl'
	,1002:'./Datasets\D1002_MergeDataset_D502_D532_10G.pkl'
	,10:'./Datasets\D10_CKplus_10groups_groupedbythe_CKplus-group-details_preprocessdata_with_calibRotation_rescaleimg_geometricfeatures_facepatches_weberface_skip-contempV2.pkl'
	,11:'./Datasets\D11_CKplus_10groups_groupedbythe_CKplus-group-details_preprocessdata_with_calibRotation_rescaleimg224x224_geometricfeatures_facepatches_weberface224x224_skip-contempV2.pkl'
	,12:'./Datasets\D12_CKplus_10groups_groupedbythe_CKplus-group-details_preprocessdata_with_calibRotation_rescaleimg_geometricfeatures_facepatches_ELTFS_skip-contempV2.pkl'
	,13:'./Datasets\D13_CKplus_8G_V4_Geo258_ELTFS128x128.pkl'
	,16:'./Datasets\D16_CKPLUS_10G_Enlargeby2015CCV_10T.pkl'
	,17:'./Datasets\D17_CKplus_10G_V4_weberface128x128.pkl'
	,18:'./Datasets\D18_CKplus_10G_V5_formalized_weberface128x128.pkl'
	,19:'./Datasets\D19_CKplus_10G_V4_ELTFS128x128.pkl'
	,2:'./Datasets\D2_CKplus_8groups_groupedbythe_CKplus-group-details_preprocessdata_with_calibRotation_rescaleimgnewmetric0731_skip-contempV2.pkl'
	,33:'./Datasets\D33_KDEF_10G_rescaleimg_geometryfeature_patches_web.pkl'
	,34:'./Datasets\D34_KDEF_10G_Enlargeby2015CCV_10T.pkl'
	,3:'./Datasets\D3_CKplus_8groups_groupedbythe_CKplus-group-details_preprocessdata_with_calibRotation_rescaleimg_geometricfeatures_facepatches_webberface_skip-contempV2.pkl'
	,40:'./Datasets\D40_jaffe_10groups_groupedbysubjects_rescaleimg_geometricfeatures_facepatches_weber.pkl'
	,43:'./Datasets\D43_JAFFE_10G_Enlargeby2015CCV_10T.pkl'
	,44:'./Datasets\D44_jaffe_10G_V4_weber128x128.pkl'
	,4:'./Datasets\D4_CKplus_8groups_groupedbythe_CKplus-group-details_preprocessdata_with_calibRotation_rescaleimg_geometricfeatures_facepatches_webberfaceReverse_skip-contempV2.pkl'
	,501:'./Datasets\D501_CKplus_10G_V5_newGeo_newPatch.pkl'
	,502:'./Datasets\D502_CKPLUS_10G_Enlargeby2015ICCV_V5_newGeo_newPatches.pkl'
	,503:'./Datasets\D503_CKplus_8G_V5_newGeo_newPatch.pkl'
	,531:'./Datasets\D531_KDEF_10G_V5_newGeo_newPatch.pkl'
	,532:'./Datasets\D532_KDEF_10G_Enlargeby2015ICCV_V5_newGeo_newPatches.pkl'
	,551:'./Datasets\D551_OuluCASIA_Weak_10G_V5_newGeo_newPatch.pkl'
	,552:'./Datasets\D552_OuluCASIA_Strong_10G_V5_newGeo_newPatch.pkl'
	,553:'./Datasets\D553_OuluCASIA_Dark_10G_V5_newGeo_newPatch.pkl'
	,554:'./Datasets\D554_MergeDataset_D551_D553_D552_10G_OuluCASIA_Weak_Dark_Strong_10G_V5_newGeo_newPatch.pkl'
	,5:'./Datasets\D5_CKplus_8groups_groupedbythe_CKplus-group-details_preprocessdata_with_calibRotation_rescaleimg_geometricfeatures_facepatches_webberface25up_skip-contempV2.pkl'
	,600:'./Datasets\D600_Collectedbywlc_filteredby435_data_with_geometry_and_facepatches.pkl'
	,601:'./Datasets\D601_Collectedbywlc_filteredby435_data_with_geometry_and_facepatches_FF.pkl'
	,610:'./Datasets\D610_FusionProject_MMI_CKplus_CasIA_with33137samples_data_with_geometry_and_facepatches_randomorder.pkl'
	,611:'./Datasets\D611_FusionProject_MMI_CKplus_CasIA_with33137samples_data_with_geometry_and_facepatches_randomorder1.pkl'
	,620:'./Datasets\D620_FusionProject_MMI_CKplus_CasIA_KDEF_JAFFE_with_geometry_and_facepatches_categoryOrder.pkl'
	,621:'./Datasets\D621_FusionProject_MMI_CKplus_CasIA_KDEF_JAFFE_with_geometry_and_facepatches_randomOrder.pkl'
	,66501:'./Datasets\D66501_CKplus_10G_V5_newGeo_newPatch_reproducelabel0to5.pkl'
	,66504:'./Datasets\D66504_CKplus_8G_V5_newGeo_newPatch_producebyreprocessimages.pkl'
	,66505:'./Datasets\D66505_CKplus_10G_V5_newGeo_newPatch_reproducelabel0to6_withContempt.pkl'
	,66531:'./Datasets\D66531_KDEF_10G_V5_newGeo_newPatch_reproducelabel0to5.pkl'
	,66554:'./Datasets\D66554_OuluCASIA_fusion_10G_V5_newGeo_newPatch_reproducelabel0to5.pkl'
	,66555:'./Datasets\D66555_OuluCASIA_Normal_10G_V5_newGeo_newPatch_reproducelabel0to5.pkl'
	,6:'./Datasets\D6_CKplus_8groups_groupedbythe_CKplus-group-details_preprocessdata_with_calibRotation_rescaleimg_geometricfeatureV2_skip-contempV2.pkl'
	,700:'./Datasets\D700_front_with827samples_data_with_geometry_and_facepatches.pkl'
	,7:'./Datasets\D7_CKplus_8groups_groupedbythe_CKplus-group-details_preprocessdata_with_calibRotation_rescaleimg_geometricfeatures_facepatches_webberface_innerfaceSizew36xh48_skip-contempV2.pkl'
	,8:'./Datasets\D8_CKplus_8groups_groupedbythe_CKplus-group-details_preprocessdata_with_calibRotation_rescaleimg_geometricfeatures_facepatches_ELTFS_skip-contempV2.pkl'
	,99999:'./Datasets\D99999_MergeDataset_data_undetected_1G.pkl'
	,9:'./Datasets\D9_CKplus_8groups_groupedbythe_CKplus-group-details_preprocessdata_with_calibRotation_rescaleimg224x224_geometricfeatures_facepatches_weberface224x224_skip-contempV2.pkl'
}


#Dataset = collections.namedtuple('Dataset', ['data', 'target'])
Datasets = collections.namedtuple('Datasets', ['train', 'test', 'validation'])

def saveModels(sess, saver, modelName, checkPoint=False, iterations=None):
    if checkPoint:
        return True#To be finished
    else:
        #saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)
        #saver = tf.train.Saver()
        saver.save(sess, modelName,)
        #saver.restore(sess,modelName)
        return True

def loadModels(sess, saver, ModelName, checkPoint=False):
    if checkPoint:
        return True#To be finished
    else:
        saver.restore(sess, ModelName)
        return True

def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = len(labels_dense)
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    #labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    #print('Total %d'%num_labels)
    for i in range(num_labels):
        labels_one_hot[i,int(labels_dense[i])]=1
        #print('%d\t%s\t%s'%(i, labels_dense[i], str(labels_one_hot[i])))
    return labels_one_hot

def listShuffle(listV, ind):
    tm=listV[:]
    for i, v in enumerate(ind):
        tm[i]=listV[v]
    return tm

class DataSetFor3KindsDataV2(object):
    """nextbatch function returns a tuple of lists (res_images, geometry, eyep, middlep, mouthp, labels),
each with length of batch size
Class Adapted from tensorflow.mnist 
Input res_images, geometry, eyep, middlep, mouthp, and labels must be list objects
res_images datatype should be unit8
`dtype` can be either `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into `[0, 1]`"""
    def __init__(self, res_images, geometry, eyep, middlep, mouthp, innerf, labels, 
                 one_hot=True, dtype=dtypes.float32, num_Classes=7, Df=True,
                 reshape=False, reshape_size=(64, 64)):
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %dtype)
        assert len(res_images) == len(labels), ('res_images length: %s labels length: %s' % (len(res_images), len(labels)))
        assert len(geometry) == len(labels), ('geometry length: %s labels length: %s' % (len(geometry), len(labels)))
        assert len(eyep) == len(labels), ('eye_patch length: %s labels length: %s' % (len(eyep), len(labels)))
        assert len(middlep) == len(labels), ('middle_patch length: %s labels length: %s' % (len(middlep), len(labels)))
        assert len(mouthp) == len(labels), ('mouth_patch length: %s labels length: %s' % (len(mouthp), len(labels)))
        assert len(innerf) == len(labels), ('inner_face length: %s labels length: %s' % (len(innerf), len(labels)))
        self._num_examples = len(res_images)

        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns] (assuming depth == 1)
            
        self._res_images = res_images[:]
        self._eyep = eyep[:]
        self._geometry = geometry[:]
        self._middlep = middlep[:]
        self._mouthp = mouthp[:]
        self._innerf = innerf[:]
        if dtype == dtypes.float32:
            # Convert from [0, 255] -> [0.0, 1.0].
            for i in range(self._num_examples):
                if reshape:
                    self._res_images[i] = cv2.resize(self._res_images[i], reshape_size, interpolation = cv2.INTER_CUBIC)

                self._res_images[i] = self._res_images[i].astype(numpy.float32)
                self._res_images[i] = numpy.multiply(self._res_images[i], 1.0 / 255.0)
                r,c=self._res_images[i].shape
                self._res_images[i] = numpy.reshape(self._res_images[i], [r, c, 1])
                
                if Df:
                    c=len(self._geometry[i])
                    self._geometry[i] = numpy.reshape(self._geometry[i], [c, 1])
                
                self._eyep[i] = self._eyep[i].astype(numpy.float32)
                self._eyep[i] = numpy.multiply(self._eyep[i], 1.0/255.0)
                r,c=self._eyep[i].shape
                self._eyep[i] = numpy.reshape(self._eyep[i], [r, c, 1])
                
                self._middlep[i] = self._middlep[i].astype(numpy.float32)
                self._middlep[i] = numpy.multiply(self._middlep[i], 1.0/255.0)
                r,c=self._middlep[i].shape
                self._middlep[i] = numpy.reshape(self._middlep[i], [r, c, 1])
                
                self._mouthp[i] = self._mouthp[i].astype(numpy.float32)
                self._mouthp[i] = numpy.multiply(self._mouthp[i], 1.0/255.0)
                r,c=self._mouthp[i].shape
                self._mouthp[i] = numpy.reshape(self._mouthp[i], [r, c, 1])
                
                self._innerf[i] = self._innerf[i].astype(numpy.float32)
                self._innerf[i] = numpy.multiply(self._innerf[i], 1.0/255.0)
                r,c=self._innerf[i].shape
                self._innerf[i] = numpy.reshape(self._innerf[i], [r, c, 1])

        
        if one_hot:
            self._labels=dense_to_one_hot(labels, num_Classes)
        else:
            self._labels = labels[:]
        self._epochs_completed = 0
        self._index_in_epoch = 0

    def reset(self, res_images, geometry, eyep, middlep, mouthp, innerf, labels):
        assert len(res_images) == len(labels), ('res_images length: %s labels length: %s' % (len(res_images), len(labels)))
        assert len(geometry) == len(labels), ('geometry length: %s labels length: %s' % (len(geometry), len(labels)))
        assert len(eyep) == len(labels), ('eye_patch length: %s labels length: %s' % (len(eyep), len(labels)))
        assert len(middlep) == len(labels), ('middle_patch length: %s labels length: %s' % (len(middlep), len(labels)))
        assert len(mouthp) == len(labels), ('mouth_patch length: %s labels length: %s' % (len(mouthp), len(labels)))
        assert len(innerf) == len(labels), ('inner_face length: %s labels length: %s' % (len(innerf), len(labels)))
        self._num_examples = len(res_images)

        self._res_images = res_images[:]
        self._eyep = eyep[:]
        self._geometry = geometry[:]
        self._middlep = middlep[:]
        self._mouthp = mouthp[:]
        self._innerf = innerf[:]
        self._labels = labels[:]
        self._epochs_completed = 0
        self._index_in_epoch = 0

    def resetIndex(self):
        self._index_in_epoch = 0
        print('_index_in_epoch has been reset.')
        return True

    @property
    def res_images(self):
        return self._res_images

    @property
    def geometry(self):
        return self._geometry

    @property
    def eyep(self):
        return self._eyep

    @property
    def middlep(self):
        return self._middlep

    @property
    def mouthp(self):
        return self._mouthp
    
    @property
    def innerf(self):
        return self._innerf

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=True):
        """nextbatch function returns a tuple of lists (res_images, geometry, eyep, middlep, mouthp, labels), each with length of batch_size from this data set."""
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm0)
            #self._res_images = self.res_images[perm0]
            #self._geometry = self.geometry[perm0]
            #self._eyep = self.eyep[perm0]
            #self._middlep = self.middlep[perm0]
            #self._mouthp = self.mouthp[perm0]
            #self._labels = self.labels[perm0]
            self._res_images = listShuffle(self.res_images, perm0)
            self._geometry = listShuffle(self.geometry, perm0)
            self._eyep = listShuffle( self.eyep, perm0)
            self._middlep = listShuffle( self.middlep, perm0)
            self._mouthp = listShuffle( self.mouthp, perm0)
            self._innerf = listShuffle( self._innerf, perm0)
            self._labels = listShuffle( self.labels, perm0)
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            #print('Epoche: %d'%self._epochs_completed)
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start

            images_rest_part = []
            geometry_rest_part = []
            eyep_rest_part = []
            middlep_rest_part = []
            mouthp_rest_part = []
            innerf_rest_part = []
            labels_rest_part = []

            images_rest_part.extend(self._res_images[start:self._num_examples])
            geometry_rest_part.extend(self._geometry[start:self._num_examples])
            eyep_rest_part.extend(self._eyep[start:self._num_examples])
            middlep_rest_part.extend(self._middlep[start:self._num_examples])
            mouthp_rest_part.extend(self._mouthp[start:self._num_examples])
            innerf_rest_part.extend(self._innerf[start:self._num_examples])
            labels_rest_part.extend(self._labels[start:self._num_examples])

            # Shuffle the data
            if shuffle:
                perm = numpy.arange(self._num_examples)
                numpy.random.shuffle(perm)
                self._res_images = listShuffle(self.res_images, perm)
                self._geometry = listShuffle(self.geometry, perm)
                self._eyep = listShuffle( self.eyep, perm)
                self._middlep = listShuffle( self.middlep, perm)
                self._mouthp = listShuffle( self.mouthp, perm)
                self._innerf = listShuffle( self._innerf, perm)
                self._labels = listShuffle( self.labels, perm)
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch

            images_rest_part.extend(self._res_images[start:end])
            geometry_rest_part.extend(self._geometry[start:end])
            eyep_rest_part.extend(self._eyep[start:end])
            middlep_rest_part.extend(self._middlep[start:end])
            mouthp_rest_part.extend(self._mouthp[start:end])
            innerf_rest_part.extend(self._innerf[start:end])
            labels_rest_part.extend(self._labels[start:end])

            return images_rest_part, geometry_rest_part, eyep_rest_part, middlep_rest_part, mouthp_rest_part, labels_rest_part, innerf_rest_part
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            #print('%d\t%d'%(len(self._res_images[start:end]),len(self._labels[start:end])))
            return self._res_images[start:end], self._geometry[start:end], self._eyep[start:end], self._middlep[start:end], self._mouthp[start:end], self._labels[start:end], self._innerf[start:end]

def loadCKplus10gdata_v2(datafilepath, validation_no=1, test_no=0, Df=False, one_hot=True, reshape=False, cn=7):
    '''return the CKplus preprocessed data in a class with convinient function to access it
    validation_no and test_no must be integer values from 0 to 9'''
    if(os.path.exists(datafilepath)):
        print('Loading data from file: %s'%datafilepath)
    else:
        print('Cannot find the data file: %s'%datafilepath)
        exit(-1)
    with open(datafilepath,'rb') as datafile:
        ckplus10g=pickle.load(datafile)
    nL=len(ckplus10g)
    print('%s preprocessed data loaded %d'%(datafilepath,nL))

    tl_rescaleimgs=[]
    tl_geometry=[]
    tl_eye_patch=[]
    tl_mouth_patch=[]
    tl_middle_patch=[]
    tl_innerf=[]
    tl_labels=[]

    for i in range(nL):
        if i==int(validation_no):
            print('Initializing validation and test dataset......')
            valid=DataSetFor3KindsDataV2(ckplus10g[i]['imgs'], ckplus10g[i]['geometry'],
                                        ckplus10g[i]['eye_patch'], ckplus10g[i]['middle_patch'],
                                        ckplus10g[i]['mouth_patch'], ckplus10g[i]['inner_face'], ckplus10g[i]['labels'], 
                                        num_Classes=cn, Df=Df, one_hot=one_hot, reshape=reshape)
            if validation_no==test_no:
                test=valid
        elif i==int(test_no):
            print('Initializing test dataset......')
            test=DataSetFor3KindsDataV2(ckplus10g[i]['imgs'], ckplus10g[i]['geometry'],
                                       ckplus10g[i]['eye_patch'], ckplus10g[i]['middle_patch'],
                                       ckplus10g[i]['mouth_patch'], ckplus10g[i]['inner_face'], ckplus10g[i]['labels'], 
                                       num_Classes=cn, Df=Df, one_hot=one_hot, reshape=reshape)
        else:
            tl_rescaleimgs.extend(ckplus10g[i]['imgs'])
            tl_geometry.extend(ckplus10g[i]['geometry'])
            tl_eye_patch.extend(ckplus10g[i]['eye_patch'])
            tl_mouth_patch.extend(ckplus10g[i]['mouth_patch'])
            tl_middle_patch.extend(ckplus10g[i]['middle_patch'])
            tl_innerf.extend( ckplus10g[i]['inner_face'])
            tl_labels.extend(ckplus10g[i]['labels'])
            #print(ckplus10g[i]['labels'])
    print('Initializing train dataset......')
    train=DataSetFor3KindsDataV2(tl_rescaleimgs, tl_geometry,
                                tl_eye_patch, tl_middle_patch,
                                tl_mouth_patch, tl_innerf, tl_labels, 
                                num_Classes=cn, Df=Df, one_hot=one_hot, reshape=reshape)
    return Datasets(train=train, test = test, validation = valid)
def loadandMergeData_v2(datafilepathlist, ID, Df=False, one_hot=True, only_geometry = False, Path = None, posfix=None, cn=7):
    '''Load the data from datafilepathlist, and merge them as one.'''
    overAll=[]
    if Path==None:
        filename='./Datasets/D%s_MergeDataset_'%str(ID)
    else:
        filename='%s/D%s_MergeDataset_'%(Path, str(ID))
    g=0
    for datafilepath in datafilepathlist:
        if(os.path.exists(datafilepath)):
            print('\n\nLoading data from file: %s'%datafilepath)
        else:
            print('Cannot find the data file: %s'%datafilepath)
            exit(-1)
        with open(datafilepath,'rb') as datafile:
            ckplus10g=pickle.load(datafile)
        nL=len(ckplus10g)
        if g==0:
            g=nL
        filename=filename+datafilepath.split('/')[-1].split('_')[0]+'_'
        print('%s preprocessed data loaded %d'%(datafilepath, nL))
        assert nL==g, 'Groups are not equal'
        for i in range(nL):
            if i < len(overAll):
                print('Dataset: %s\nextend part %d......'%(datafilepath, i))
                if only_geometry:
                    overAll[i]['geometry'].extend(ckplus10g[i]['geometry'])
                    overAll[i]['labels'].extend(ckplus10g[i]['labels'])
                else:
                    overAll[i]['imgs'].extend(ckplus10g[i]['imgs'])
                    overAll[i]['geometry'].extend(ckplus10g[i]['geometry'])
                    overAll[i]['eye_patch'].extend(ckplus10g[i]['eye_patch'])
                    overAll[i]['mouth_patch'].extend(ckplus10g[i]['mouth_patch'])
                    overAll[i]['middle_patch'].extend(ckplus10g[i]['middle_patch'])
                    overAll[i]['inner_face'].extend(ckplus10g[i]['inner_face'])
                    overAll[i]['labels'].extend(ckplus10g[i]['labels'])
            else:
                print('Dataset: %s\nappend part %d......'%(datafilepath, i))
                overAll.append(ckplus10g[i])
    if only_geometry:
        filename=filename+'only_geometry_'
    filename=filename+'%dG.pkl'%g
    if posfix:
        filename=filename.replace('.pkl', ('_'+posfix+'.pkl'))
    if(os.path.exists(filename)):
        print('Filename %s already exists'%filename)
        print('Try another merge.')
    else:
        print('Saving file in %s'%filename)
        with open(filename, 'wb') as fin:
            pickle.dump(overAll, fin, 4)
        print('File has been saved.')
def loadTestData_v2(datafilepath, Df=False, one_hot=True, reshape=False, cn=7):
    '''Use loadPKLData_v2 instead.
    return the preprocessed data in a class with convinient function to access it
    The data returned were formed into one list'''
    return loadPKLData_v2(datafilepath, Df, one_hot, reshape)
def loadPKLData_v2(datafilepath, Df=False, one_hot=True, reshape=False, cn=7):
    '''Load the entire pkl data file without partitionning.
    return the preprocessed data in a class with convinient function to access it
    The data returned were formed into one list'''
    if(os.path.exists(datafilepath)):
        print('Loading data from file: %s'%datafilepath)
    else:
        print('Cannot find the data file: %s'%datafilepath)
        exit(-1)
    with open(datafilepath,'rb') as datafile:
        ckplus10g=pickle.load(datafile)
    nL=len(ckplus10g)
    print('%s preprocessed data loaded %d'%(datafilepath,nL))

    tl_rescaleimgs=[]
    tl_geometry=[]
    tl_eye_patch=[]
    tl_mouth_patch=[]
    tl_middle_patch=[]
    tl_innerf=[]
    tl_labels=[]

    for i in range(nL):
        tl_rescaleimgs.extend(ckplus10g[i]['imgs'])
        tl_geometry.extend(ckplus10g[i]['geometry'])
        tl_eye_patch.extend(ckplus10g[i]['eye_patch'])
        tl_mouth_patch.extend(ckplus10g[i]['mouth_patch'])
        tl_middle_patch.extend(ckplus10g[i]['middle_patch'])
        tl_innerf.extend( ckplus10g[i]['inner_face'])
        tl_labels.extend(ckplus10g[i]['labels'])
        #print(ckplus10g[i]['labels'])
    print('Initializing train dataset......')
    test=DataSetFor3KindsDataV2(tl_rescaleimgs, tl_geometry,
                                tl_eye_patch, tl_middle_patch,
                                tl_mouth_patch, tl_innerf, tl_labels, 
                                num_Classes=cn, Df=Df, one_hot=one_hot, reshape=reshape)
    #return Datasets(train=None, test = test, validation = None)
    return test

class DataSetFor3KindsDataV3(object):
    """nextbatch function returns a tuple of lists (res_images, geometry, eyep, middlep, mouthp, labels),
each with length of batch size
Class Adapted from tensorflow.mnist 
Input res_images, geometry, eyep, middlep, mouthp, and labels must be list objects
res_images datatype should be unit8
`dtype` can be either `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into `[0, 1]`"""
    def __init__(self, res_images, geometry, labels, one_hot=True, dtype=dtypes.float32, num_Classes=7, Df=True):
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %dtype)
        assert len(res_images) == len(labels), ('res_images length: %s labels length: %s' % (len(res_images), len(labels)))
        assert len(geometry) == len(labels), ('geometry length: %s labels length: %s' % (len(geometry), len(labels)))
        self._num_examples = len(res_images)

        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns] (assuming depth == 1)
        if dtype == dtypes.float32:
            # Convert from [0, 255] -> [0.0, 1.0].
            for i in range(self._num_examples):
                res_images[i] = res_images[i].astype(numpy.float32)
                res_images[i] = numpy.multiply(res_images[i], 1.0 / 255.0)
                r,c=res_images[i].shape
                res_images[i] = numpy.reshape(res_images[i], [r, c, 1])
                
                if Df:
                    c=len(geometry[i])
                    geometry[i] = numpy.reshape(geometry[i], [c, 1])
            
        self._res_images = res_images[:]
        self._geometry = geometry[:]
        if one_hot:
            self._labels=dense_to_one_hot(labels, num_Classes)
        else:
            self._labels = labels[:]
        self._epochs_completed = 0
        self._index_in_epoch = 0

    def resetIndex(self):
        self._index_in_epoch = 0
        print('_index_in_epoch has been reset.')
        return True

    @property
    def res_images(self):
        return self._res_images

    @property
    def geometry(self):
        return self._geometry

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=True):
        """nextbatch function returns a tuple of lists (res_images, geometry, eyep, middlep, mouthp, labels), each with length of batch_size from this data set."""
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm0)
            self._res_images = listShuffle(self.res_images, perm0)
            self._geometry = listShuffle(self.geometry, perm0)
            self._labels = listShuffle( self.labels, perm0)
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            #print('Epoche: %d'%self._epochs_completed)
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start

            images_rest_part = []
            geometry_rest_part = []
            labels_rest_part = []

            images_rest_part.extend(self._res_images[start:self._num_examples])
            geometry_rest_part.extend(self._geometry[start:self._num_examples])
            labels_rest_part.extend(self._labels[start:self._num_examples])

            # Shuffle the data
            if shuffle:
                perm = numpy.arange(self._num_examples)
                numpy.random.shuffle(perm)
                self._res_images = listShuffle(self.res_images, perm)
                self._geometry = listShuffle(self.geometry, perm)
                self._labels = listShuffle( self.labels, perm)
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch

            images_rest_part.extend(self._res_images[start:end])
            geometry_rest_part.extend(self._geometry[start:end])
            labels_rest_part.extend(self._labels[start:end])

            return images_rest_part, geometry_rest_part, None, None, None, labels_rest_part, None
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            #print('%d\t%d'%(len(self._res_images[start:end]),len(self._labels[start:end])))
            return self._res_images[start:end], self._geometry[start:end], None, None, None, self._labels[start:end], None
def loadCKplus10gdata_v3(datafilepath, validation_no=1, test_no=0, Df=False, one_hot=True, cn=7):
    '''This version only contains the whole face and geometry features, plus the labels. No pathes and no innerface were included.'''
    if(os.path.exists(datafilepath)):
        print('Loading data from file: %s'%datafilepath)
    else:
        print('Cannot find the data file: %s'%datafilepath)
        exit(-1)
    with open(datafilepath,'rb') as datafile:
        ckplus10g=pickle.load(datafile)
    nL=len(ckplus10g)
    print('CKplus preprocessed data loaded %d'%(nL))

    tl_rescaleimgs=[]
    tl_geometry=[]
    tl_labels=[]

    for i in range(nL):

        if i==int(validation_no):
            if validation_no==test_no:
                valid=DataSetFor3KindsDataV3(ckplus10g[i]['imgs'], ckplus10g[i]['geometry'],
                                           ckplus10g[i]['labels'], 
                                           num_Classes=cn, Df=Df, one_hot=one_hot)
                test=valid
            else:
                valid=DataSetFor3KindsDataV3(ckplus10g[i]['imgs'], ckplus10g[i]['geometry'],
                                           ckplus10g[i]['labels'], 
                                           num_Classes=cn, Df=Df, one_hot=one_hot)
            #print(ckplus10g[i]['labels'])
        elif i==int(test_no):
            test=DataSetFor3KindsDataV3(ckplus10g[i]['imgs'], ckplus10g[i]['geometry'],
                                       ckplus10g[i]['labels'], 
                                       num_Classes=cn, Df=Df, one_hot=one_hot)
        else:
            tl_rescaleimgs.extend(ckplus10g[i]['imgs'])
            tl_geometry.extend(ckplus10g[i]['geometry'])
            tl_labels.extend(ckplus10g[i]['labels'])
    train=DataSetFor3KindsDataV3(tl_rescaleimgs, tl_geometry,
                                tl_labels, 
                                num_Classes=cn, Df=Df, one_hot=one_hot)
    return Datasets(train=train, test = test, validation = valid)

class DataSetFor3KindsDataV4(object):
    """nextbatch function returns a tuple of lists (res_images, geometry, eyep, middlep, mouthp, labels),
each with length of batch size
Class Adapted from tensorflow.mnist 
Input res_images, geometry, eyep, middlep, mouthp, and labels must be list objects
res_images datatype should be unit8
`dtype` can be either `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into `[0, 1]`"""
    def __init__(self, res_images=None, labels=None, geometry=None, eyep=None, middlep=None, mouthp=None, innerf=None, 
                 one_hot=True, dtype=dtypes.float32, num_Classes=7, Df=False, loadImg=False, loadGeo=False, loadPat=False, loadInnerF=False,
                 reshape=False, reshape_size=(64, 64)):
        '''Becareful, the inputs of this function will be deleted.'''
        if loadImg or loadGeo or loadPat or loadInnerF:
            dtype = dtypes.as_dtype(dtype).base_dtype
            if dtype not in (dtypes.uint8, dtypes.float32):
                raise TypeError('Invalid image dtype %r, expected uint8 or float32' %dtype)
            if loadImg:
                assert len(res_images) == len(labels), ('res_images length: %s labels length: %s' % (len(res_images), len(labels)))
                self._res_images = res_images[:]
                del res_images
            else:
                self._res_images = None
            if loadGeo:
                assert len(geometry) == len(labels), ('geometry length: %s labels length: %s' % (len(geometry), len(labels)))
                self._geometry = geometry[:]
                del geometry
            else:
                self._geometry = None
            if loadPat:
                assert len(eyep) == len(labels), ('eye_patch length: %s labels length: %s' % (len(eyep), len(labels)))
                assert len(middlep) == len(labels), ('middle_patch length: %s labels length: %s' % (len(middlep), len(labels)))
                assert len(mouthp) == len(labels), ('mouth_patch length: %s labels length: %s' % (len(mouthp), len(labels)))
                self._eyep = eyep[:]
                del eyep
                self._middlep = middlep[:]
                del middlep
                self._mouthp = mouthp[:]
                del mouthp
            else:
                self._eyep = None
                self._middlep = None
                self._mouthp = None
            if loadInnerF:
                if loadImg or loadGeo or loadPat:
                    print('ERROR in __init__ of DataSetFor3KindsDataV4: Unexpected case for nextbatch logic')
                    exit(-1)
                assert len(innerf) == len(labels), ('inner_face length: %s labels length: %s' % (len(innerf), len(labels)))
                self._innerf = innerf[:]
                del innerf
            else:
                self._innerf = None
            if one_hot:
                self._labels=dense_to_one_hot(labels, num_Classes)
            else:
                self._labels = numpy.asarray(labels[:])
            self._num_examples = len(labels)
            del labels
            self._epochs_completed = 0
            self._index_in_epoch = 0

            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, rows*columns] (assuming depth == 1)
            self.__loadGeo=loadGeo
            self.__loadPat=loadPat   
            self.__loadImg=loadImg
            self.__loadInnerF=loadInnerF
            if dtype == dtypes.float32:
                # Convert from [0, 255] -> [0.0, 1.0].
                for i in range(self._num_examples):
                    if loadImg:
                        if reshape:
                            self._res_images[i] = cv2.resize(self._res_images[i], reshape_size, interpolation = cv2.INTER_CUBIC)

                        self._res_images[i] = self._res_images[i].astype(numpy.float32)
                        self._res_images[i] = numpy.multiply(self._res_images[i], 1.0 / 255.0)
                        r,c=self._res_images[i].shape
                        self._res_images[i] = numpy.reshape(self._res_images[i], [r, c, 1])
                
                    if loadGeo:
                        if Df:
                            c=len(self._geometry[i])
                            self._geometry[i] = numpy.reshape(self._geometry[i], [c, 1])
                    if loadPat:
                        self._eyep[i] = self._eyep[i].astype(numpy.float32)
                        self._eyep[i] = numpy.multiply(self._eyep[i], 1.0/255.0)
                        r,c=self._eyep[i].shape
                        self._eyep[i] = numpy.reshape(self._eyep[i], [r, c, 1])
                
                        self._middlep[i] = self._middlep[i].astype(numpy.float32)
                        self._middlep[i] = numpy.multiply(self._middlep[i], 1.0/255.0)
                        r,c=self._middlep[i].shape
                        self._middlep[i] = numpy.reshape(self._middlep[i], [r, c, 1])
                
                        self._mouthp[i] = self._mouthp[i].astype(numpy.float32)
                        self._mouthp[i] = numpy.multiply(self._mouthp[i], 1.0/255.0)
                        r,c=self._mouthp[i].shape
                        self._mouthp[i] = numpy.reshape(self._mouthp[i], [r, c, 1])
                    if loadInnerF:
                        self._innerf[i] = self._innerf[i].astype(numpy.float32)
                        self._innerf[i] = numpy.multiply(self._innerf[i], 1.0/255.0)
                        r,c=self._innerf[i].shape
                        self._innerf[i] = numpy.reshape(self._innerf[i], [r, c, 1])
        else:
            print('%*%*%**%*%*%*ERROR: Must load one of the feature Module')
            exit(-1)


    def reset(self, res_images=None, labels=None, geometry=None, eyep=None, middlep=None, mouthp=None, innerf=None):
        if self.__loadImg:
            assert len(res_images) == len(labels), ('res_images length: %s labels length: %s' % (len(res_images), len(labels)))
            self._res_images = res_images[:]
        if self.__loadGeo:
            assert len(geometry) == len(labels), ('geometry length: %s labels length: %s' % (len(geometry), len(labels)))
            self._geometry = geometry[:]
        if self.__loadPat:
            assert len(eyep) == len(labels), ('eye_patch length: %s labels length: %s' % (len(eyep), len(labels)))
            assert len(middlep) == len(labels), ('middle_patch length: %s labels length: %s' % (len(middlep), len(labels)))
            assert len(mouthp) == len(labels), ('mouth_patch length: %s labels length: %s' % (len(mouthp), len(labels)))
            self._eyep = eyep[:]
            self._middlep = middlep[:]
            self._mouthp = mouthp[:]
        if self.__loadInnerF:
            assert len(innerf) == len(labels), ('inner_face length: %s labels length: %s' % (len(innerf), len(labels)))
            self._innerf = innerf[:]
        self._num_examples = len(labels)
        self._labels = labels[:]
        self._epochs_completed = 0
        self._index_in_epoch = 0

    def resetIndex(self):
        self._index_in_epoch = 0
        print('_index_in_epoch has been reset.')
        return True

    @property
    def res_images(self):
        return self._res_images

    @property
    def geometry(self):
        return self._geometry

    @property
    def eyep(self):
        return self._eyep

    @property
    def middlep(self):
        return self._middlep

    @property
    def mouthp(self):
        return self._mouthp
    
    @property
    def innerf(self):
        return self._innerf

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=True):
        """nextbatch function returns a tuple of lists (res_images, geometry, eyep, middlep, mouthp, labels), each with length of batch_size from this data set."""
        start = self._index_in_epoch
        if self.__loadGeo and self.__loadPat and self.__loadImg:
            # Shuffle for the first epoch
            if self._epochs_completed == 0 and start == 0 and shuffle:
                perm0 = numpy.arange(self._num_examples)
                numpy.random.shuffle(perm0)
                self._res_images = listShuffle(self.res_images, perm0)
  
                self._geometry = listShuffle(self.geometry, perm0)
  
                self._eyep = listShuffle( self.eyep, perm0)
                self._middlep = listShuffle( self.middlep, perm0)
                self._mouthp = listShuffle( self.mouthp, perm0)
                self._innerf = listShuffle( self._innerf, perm0)
                self._labels = listShuffle( self.labels, perm0)
            # Go to the next epoch
            if start + batch_size > self._num_examples:
                # Finished epoch
                self._epochs_completed += 1
                #print('Epoche: %d'%self._epochs_completed)
                # Get the rest examples in this epoch
                rest_num_examples = self._num_examples - start

                images_rest_part = []
                labels_rest_part = []
                images_rest_part.extend(self._res_images[start:self._num_examples])
                labels_rest_part.extend(self._labels[start:self._num_examples])

                geometry_rest_part = []
                geometry_rest_part.extend(self._geometry[start:self._num_examples])


                eyep_rest_part = []
                middlep_rest_part = []
                mouthp_rest_part = []
                innerf_rest_part = []
                eyep_rest_part.extend(self._eyep[start:self._num_examples])
                middlep_rest_part.extend(self._middlep[start:self._num_examples])
                mouthp_rest_part.extend(self._mouthp[start:self._num_examples])
                innerf_rest_part.extend(self._innerf[start:self._num_examples])

                # Shuffle the data
                if shuffle:
                    perm = numpy.arange(self._num_examples)
                    numpy.random.shuffle(perm)
                    self._res_images = listShuffle(self.res_images, perm)
                    self._labels = listShuffle( self.labels, perm)

                    self._geometry = listShuffle(self.geometry, perm)

                    self._eyep = listShuffle( self.eyep, perm)
                    self._middlep = listShuffle( self.middlep, perm)
                    self._mouthp = listShuffle( self.mouthp, perm)
                    self._innerf = listShuffle( self._innerf, perm)
                # Start next epoch
                start = 0
                self._index_in_epoch = batch_size - rest_num_examples
                end = self._index_in_epoch

                images_rest_part.extend(self._res_images[start:end])
                labels_rest_part.extend(self._labels[start:end])

                geometry_rest_part.extend(self._geometry[start:end])

                eyep_rest_part.extend(self._eyep[start:end])
                middlep_rest_part.extend(self._middlep[start:end])
                mouthp_rest_part.extend(self._mouthp[start:end])
                innerf_rest_part.extend(self._innerf[start:end])

                return images_rest_part, geometry_rest_part, eyep_rest_part, middlep_rest_part, mouthp_rest_part, labels_rest_part, innerf_rest_part
            else:
                self._index_in_epoch += batch_size
                end = self._index_in_epoch
                #print('%d\t%d'%(len(self._res_images[start:end]),len(self._labels[start:end])))
                return self._res_images[start:end], self._geometry[start:end], self._eyep[start:end], self._middlep[start:end], self._mouthp[start:end], self._labels[start:end], self._innerf[start:end]
        elif self.__loadPat and self.__loadImg:
            # Shuffle for the first epoch
            if self._epochs_completed == 0 and start == 0 and shuffle:
                perm0 = numpy.arange(self._num_examples)
                numpy.random.shuffle(perm0)
                self._res_images = listShuffle(self.res_images, perm0)
  
                self._eyep = listShuffle( self.eyep, perm0)
                self._middlep = listShuffle( self.middlep, perm0)
                self._mouthp = listShuffle( self.mouthp, perm0)
                self._innerf = listShuffle( self._innerf, perm0)
                self._labels = listShuffle( self.labels, perm0)
            # Go to the next epoch
            if start + batch_size > self._num_examples:
                # Finished epoch
                self._epochs_completed += 1
                #print('Epoche: %d'%self._epochs_completed)
                # Get the rest examples in this epoch
                rest_num_examples = self._num_examples - start

                images_rest_part = []
                labels_rest_part = []
                images_rest_part.extend(self._res_images[start:self._num_examples])
                labels_rest_part.extend(self._labels[start:self._num_examples])

                eyep_rest_part = []
                middlep_rest_part = []
                mouthp_rest_part = []
                innerf_rest_part = []
                eyep_rest_part.extend(self._eyep[start:self._num_examples])
                middlep_rest_part.extend(self._middlep[start:self._num_examples])
                mouthp_rest_part.extend(self._mouthp[start:self._num_examples])
                innerf_rest_part.extend(self._innerf[start:self._num_examples])

                # Shuffle the data
                if shuffle:
                    perm = numpy.arange(self._num_examples)
                    numpy.random.shuffle(perm)
                    self._res_images = listShuffle(self.res_images, perm)
                    self._labels = listShuffle( self.labels, perm)

                    self._eyep = listShuffle( self.eyep, perm)
                    self._middlep = listShuffle( self.middlep, perm)
                    self._mouthp = listShuffle( self.mouthp, perm)
                    self._innerf = listShuffle( self._innerf, perm)
                # Start next epoch
                start = 0
                self._index_in_epoch = batch_size - rest_num_examples
                end = self._index_in_epoch

                images_rest_part.extend(self._res_images[start:end])
                labels_rest_part.extend(self._labels[start:end])

                eyep_rest_part.extend(self._eyep[start:end])
                middlep_rest_part.extend(self._middlep[start:end])
                mouthp_rest_part.extend(self._mouthp[start:end])
                innerf_rest_part.extend(self._innerf[start:end])

                return images_rest_part, None, eyep_rest_part, middlep_rest_part, mouthp_rest_part, labels_rest_part, innerf_rest_part
            else:
                self._index_in_epoch += batch_size
                end = self._index_in_epoch
                #print('%d\t%d'%(len(self._res_images[start:end]),len(self._labels[start:end])))
                return self._res_images[start:end], None, self._eyep[start:end], self._middlep[start:end], self._mouthp[start:end], self._labels[start:end], self._innerf[start:end]
        elif self.__loadGeo and self.__loadImg:
            # Shuffle for the first epoch
            if self._epochs_completed == 0 and start == 0 and shuffle:
                perm0 = numpy.arange(self._num_examples)
                numpy.random.shuffle(perm0)
                self._res_images = listShuffle(self.res_images, perm0)
  
                self._geometry = listShuffle(self.geometry, perm0)
  
                self._labels = listShuffle( self.labels, perm0)
            # Go to the next epoch
            if start + batch_size > self._num_examples:
                # Finished epoch
                self._epochs_completed += 1
                #print('Epoche: %d'%self._epochs_completed)
                # Get the rest examples in this epoch
                rest_num_examples = self._num_examples - start

                images_rest_part = []
                labels_rest_part = []
                images_rest_part.extend(self._res_images[start:self._num_examples])
                labels_rest_part.extend(self._labels[start:self._num_examples])

                geometry_rest_part = []
                geometry_rest_part.extend(self._geometry[start:self._num_examples])

                # Shuffle the data
                if shuffle:
                    perm = numpy.arange(self._num_examples)
                    numpy.random.shuffle(perm)
                    self._res_images = listShuffle(self.res_images, perm)
                    self._labels = listShuffle( self.labels, perm)

                    self._geometry = listShuffle(self.geometry, perm)

                # Start next epoch
                start = 0
                self._index_in_epoch = batch_size - rest_num_examples
                end = self._index_in_epoch

                images_rest_part.extend(self._res_images[start:end])
                labels_rest_part.extend(self._labels[start:end])

                geometry_rest_part.extend(self._geometry[start:end])

                return images_rest_part, geometry_rest_part, None, None, None, labels_rest_part, None
            else:
                self._index_in_epoch += batch_size
                end = self._index_in_epoch
                #print('%d\t%d'%(len(self._res_images[start:end]),len(self._labels[start:end])))
                return self._res_images[start:end], self._geometry[start:end], None, None, None, self._labels[start:end], None
        elif self.__loadGeo and self.__loadPat:
            # Shuffle for the first epoch
            if self._epochs_completed == 0 and start == 0 and shuffle:
                perm0 = numpy.arange(self._num_examples)
                numpy.random.shuffle(perm0)
  
                self._geometry = listShuffle(self.geometry, perm0)
  
                self._eyep = listShuffle( self.eyep, perm0)
                self._middlep = listShuffle( self.middlep, perm0)
                self._mouthp = listShuffle( self.mouthp, perm0)
                self._innerf = listShuffle( self._innerf, perm0)
                self._labels = listShuffle( self.labels, perm0)
            # Go to the next epoch
            if start + batch_size > self._num_examples:
                # Finished epoch
                self._epochs_completed += 1
                #print('Epoche: %d'%self._epochs_completed)
                # Get the rest examples in this epoch
                rest_num_examples = self._num_examples - start

                labels_rest_part = []
                labels_rest_part.extend(self._labels[start:self._num_examples])

                geometry_rest_part = []
                geometry_rest_part.extend(self._geometry[start:self._num_examples])

                eyep_rest_part = []
                middlep_rest_part = []
                mouthp_rest_part = []
                innerf_rest_part = []
                eyep_rest_part.extend(self._eyep[start:self._num_examples])
                middlep_rest_part.extend(self._middlep[start:self._num_examples])
                mouthp_rest_part.extend(self._mouthp[start:self._num_examples])
                innerf_rest_part.extend(self._innerf[start:self._num_examples])

                # Shuffle the data
                if shuffle:
                    perm = numpy.arange(self._num_examples)
                    numpy.random.shuffle(perm)
                    self._labels = listShuffle( self.labels, perm)

                    self._geometry = listShuffle(self.geometry, perm)

                    self._eyep = listShuffle( self.eyep, perm)
                    self._middlep = listShuffle( self.middlep, perm)
                    self._mouthp = listShuffle( self.mouthp, perm)
                    self._innerf = listShuffle( self._innerf, perm)
                # Start next epoch
                start = 0
                self._index_in_epoch = batch_size - rest_num_examples
                end = self._index_in_epoch

                labels_rest_part.extend(self._labels[start:end])

                geometry_rest_part.extend(self._geometry[start:end])

                eyep_rest_part.extend(self._eyep[start:end])
                middlep_rest_part.extend(self._middlep[start:end])
                mouthp_rest_part.extend(self._mouthp[start:end])
                innerf_rest_part.extend(self._innerf[start:end])

                return None, geometry_rest_part, eyep_rest_part, middlep_rest_part, mouthp_rest_part, labels_rest_part, innerf_rest_part
            else:
                self._index_in_epoch += batch_size
                end = self._index_in_epoch
                return None, self._geometry[start:end], self._eyep[start:end], self._middlep[start:end], self._mouthp[start:end], self._labels[start:end], self._innerf[start:end]
        elif self.__loadGeo:
            if self._epochs_completed == 0 and start == 0 and shuffle:
                perm0 = numpy.arange(self._num_examples)
                numpy.random.shuffle(perm0)
                self._labels = listShuffle( self.labels, perm0)
                self._geometry = listShuffle(self.geometry, perm0)
  
            # Go to the next epoch
            if start + batch_size > self._num_examples:
                # Finished epoch
                self._epochs_completed += 1
                #print('Epoche: %d'%self._epochs_completed)
                # Get the rest examples in this epoch
                rest_num_examples = self._num_examples - start

                labels_rest_part = []
                labels_rest_part.extend(self._labels[start:self._num_examples])

                geometry_rest_part = []
                geometry_rest_part.extend(self._geometry[start:self._num_examples])

                # Shuffle the data
                if shuffle:
                    perm = numpy.arange(self._num_examples)
                    numpy.random.shuffle(perm)
                    self._labels = listShuffle( self.labels, perm)
                    self._geometry = listShuffle(self.geometry, perm)

                # Start next epoch
                start = 0
                self._index_in_epoch = batch_size - rest_num_examples
                end = self._index_in_epoch

                labels_rest_part.extend(self._labels[start:end])

                geometry_rest_part.extend(self._geometry[start:end])

                return None, geometry_rest_part, None, None, None, labels_rest_part, None
            else:
                self._index_in_epoch += batch_size
                end = self._index_in_epoch
                #print('%d\t%d'%(len(self._res_images[start:end]),len(self._labels[start:end])))
                return None, self._geometry[start:end], None, None, None, self._labels[start:end], None
        elif self.__loadImg:
            if self._epochs_completed == 0 and start == 0 and shuffle:
                perm0 = numpy.arange(self._num_examples)
                numpy.random.shuffle(perm0)
                self._res_images = listShuffle(self.res_images, perm0)
                self._labels = listShuffle( self.labels, perm0)
            # Go to the next epoch
            if start + batch_size > self._num_examples:
                # Finished epoch
                self._epochs_completed += 1
                #print('Epoche: %d'%self._epochs_completed)
                # Get the rest examples in this epoch
                rest_num_examples = self._num_examples - start

                images_rest_part = []
                labels_rest_part = []
                images_rest_part.extend(self._res_images[start:self._num_examples])
                labels_rest_part.extend(self._labels[start:self._num_examples])

                # Shuffle the data
                if shuffle:
                    perm = numpy.arange(self._num_examples)
                    numpy.random.shuffle(perm)
                    self._res_images = listShuffle(self.res_images, perm)
                    self._labels = listShuffle( self.labels, perm)

                # Start next epoch
                start = 0
                self._index_in_epoch = batch_size - rest_num_examples
                end = self._index_in_epoch

                images_rest_part.extend(self._res_images[start:end])
                labels_rest_part.extend(self._labels[start:end])

                return images_rest_part, None, None, None, None, labels_rest_part, None
            else:
                self._index_in_epoch += batch_size
                end = self._index_in_epoch
                #print('%d\t%d'%(len(self._res_images[start:end]),len(self._labels[start:end])))
                return self._res_images[start:end], None, None, None, None, self._labels[start:end], None
        elif self.__loadPat:
            # Shuffle for the first epoch
            if self._epochs_completed == 0 and start == 0 and shuffle:
                perm0 = numpy.arange(self._num_examples)
                numpy.random.shuffle(perm0)
                self._eyep = listShuffle( self.eyep, perm0)
                self._middlep = listShuffle( self.middlep, perm0)
                self._mouthp = listShuffle( self.mouthp, perm0)
                self._labels = listShuffle( self.labels, perm0)
            # Go to the next epoch
            if start + batch_size > self._num_examples:
                # Finished epoch
                self._epochs_completed += 1
                #print('Epoche: %d'%self._epochs_completed)
                # Get the rest examples in this epoch
                rest_num_examples = self._num_examples - start

                labels_rest_part = []
                labels_rest_part.extend(self._labels[start:self._num_examples])

                eyep_rest_part = []
                middlep_rest_part = []
                mouthp_rest_part = []
                eyep_rest_part.extend(self._eyep[start:self._num_examples])
                middlep_rest_part.extend(self._middlep[start:self._num_examples])
                mouthp_rest_part.extend(self._mouthp[start:self._num_examples])

                # Shuffle the data
                if shuffle:
                    perm = numpy.arange(self._num_examples)
                    numpy.random.shuffle(perm)
                    self._labels = listShuffle( self.labels, perm)

                    self._eyep = listShuffle( self.eyep, perm)
                    self._middlep = listShuffle( self.middlep, perm)
                    self._mouthp = listShuffle( self.mouthp, perm)
                # Start next epoch
                start = 0
                self._index_in_epoch = batch_size - rest_num_examples
                end = self._index_in_epoch

                labels_rest_part.extend(self._labels[start:end])

                eyep_rest_part.extend(self._eyep[start:end])
                middlep_rest_part.extend(self._middlep[start:end])
                mouthp_rest_part.extend(self._mouthp[start:end])

                return None, None, eyep_rest_part, middlep_rest_part, mouthp_rest_part, labels_rest_part, None
            else:
                self._index_in_epoch += batch_size
                end = self._index_in_epoch
                #print('%d\t%d'%(len(self._res_images[start:end]),len(self._labels[start:end])))
                return None, None, self._eyep[start:end], self._middlep[start:end], self._mouthp[start:end], self._labels[start:end], None
        elif self.__loadInnerF:
            # Shuffle for the first epoch
            if self._epochs_completed == 0 and start == 0 and shuffle:
                perm0 = numpy.arange(self._num_examples)
                numpy.random.shuffle(perm0)
                self._innerf = listShuffle( self._innerf, perm0)
                self._labels = listShuffle( self.labels, perm0)
            # Go to the next epoch
            if start + batch_size > self._num_examples:
                # Finished epoch
                self._epochs_completed += 1
                #print('Epoche: %d'%self._epochs_completed)
                # Get the rest examples in this epoch
                rest_num_examples = self._num_examples - start

                labels_rest_part = []
                labels_rest_part.extend(self._labels[start:self._num_examples])

                innerf_rest_part = []
                innerf_rest_part.extend(self._innerf[start:self._num_examples])

                # Shuffle the data
                if shuffle:
                    perm = numpy.arange(self._num_examples)
                    numpy.random.shuffle(perm)
                    self._labels = listShuffle( self.labels, perm)

                    self._innerf = listShuffle( self._innerf, perm)
                # Start next epoch
                start = 0
                self._index_in_epoch = batch_size - rest_num_examples
                end = self._index_in_epoch

                labels_rest_part.extend(self._labels[start:end])

                innerf_rest_part.extend(self._innerf[start:end])

                return None, None, None, None, None, labels_rest_part, innerf_rest_part
            else:
                self._index_in_epoch += batch_size
                end = self._index_in_epoch
                #print('%d\t%d'%(len(self._res_images[start:end]),len(self._labels[start:end])))
                return None, None, None, None, None, self._labels[start:end], self._innerf[start:end]
def loadPKLData_v4(datafilepath, Module=0, Df=False, one_hot=True, reshape=False,cn=7):
    '''Load the entire pkl data file without partitionning.
    return the preprocessed data in a class with convinient function to access it
    The data returned were formed into one list'''
    if(os.path.exists(datafilepath)):
        print('Loading data from file: %s'%datafilepath)
    else:
        print('Cannot find the data file: %s'%datafilepath)
        exit(-1)
    with open(datafilepath,'rb') as datafile:
        ckplus10g=pickle.load(datafile)
    nL=len(ckplus10g)
    print('%s preprocessed data loaded %d'%(datafilepath,nL))
    if Module==0:
        tl_rescaleimgs=[]
        tl_geometry=[]
        tl_eye_patch=[]
        tl_mouth_patch=[]
        tl_middle_patch=[]
        tl_innerf=[]
        tl_labels=[]

        for i in range(nL):
            tl_rescaleimgs.extend(ckplus10g[i]['imgs'])
            del ckplus10g[i]['imgs']
            tl_geometry.extend(ckplus10g[i]['geometry'])
            del ckplus10g[i]['geometry']
            tl_eye_patch.extend(ckplus10g[i]['eye_patch'])
            del ckplus10g[i]['eye_patch']
            tl_mouth_patch.extend(ckplus10g[i]['mouth_patch'])
            del ckplus10g[i]['mouth_patch']
            tl_middle_patch.extend(ckplus10g[i]['middle_patch'])
            del ckplus10g[i]['middle_patch']
            tl_innerf.extend( ckplus10g[i]['inner_face'])
            del ckplus10g[i]['inner_face']
            tl_labels.extend(ckplus10g[i]['labels'])
            del ckplus10g[i]['labels']
            #print(ckplus10g[i]['labels'])
        print('Initializing train dataset......')
        test=DataSetFor3KindsDataV4(res_images=tl_rescaleimgs, labels=tl_labels,geometry = tl_geometry,
                                    eyep= tl_eye_patch, middlep= tl_middle_patch,
                                    mouthp= tl_mouth_patch, innerf= tl_innerf, 
                                    num_Classes=cn, Df=Df, one_hot=one_hot, reshape=reshape,
                                    loadGeo=True, loadImg=True, loadPat=True)
    elif Module==1:
        tl_rescaleimgs=[]
        tl_labels=[]

        for i in range(nL):
            tl_rescaleimgs.extend(ckplus10g[i]['imgs'])
            del ckplus10g[i]['imgs']
            tl_labels.extend(ckplus10g[i]['labels'])
            del ckplus10g[i]['labels']
            del ckplus10g[i]['geometry']
            del ckplus10g[i]['eye_patch']
            del ckplus10g[i]['mouth_patch']
            del ckplus10g[i]['middle_patch']
            del ckplus10g[i]['inner_face']
        print('Initializing train dataset......')
        test=DataSetFor3KindsDataV4(res_images=tl_rescaleimgs, labels=tl_labels,
                                    num_Classes=cn, Df=Df, one_hot=one_hot, reshape=reshape,
                                    loadGeo=False, loadImg=True, loadPat=False)
    elif Module==2:
        tl_geometry=[]
        tl_labels=[]
        print('\nGeometry feature dimension: %d\n'%(len(ckplus10g[0]['geometry'][0])))
        for i in range(nL):
            tl_geometry.extend(ckplus10g[i]['geometry'])
            del ckplus10g[i]['geometry']
            tl_labels.extend(ckplus10g[i]['labels'])
            del ckplus10g[i]['labels']
            del ckplus10g[i]['imgs']
            del ckplus10g[i]['eye_patch']
            del ckplus10g[i]['mouth_patch']
            del ckplus10g[i]['middle_patch']
            del ckplus10g[i]['inner_face']
        print('Initializing train dataset......')
        test=DataSetFor3KindsDataV4(labels=tl_labels,geometry = tl_geometry,
                                    num_Classes=cn, Df=Df, one_hot=one_hot, reshape=reshape,
                                    loadGeo=True, loadImg=False, loadPat=False)
    elif Module==3:
        tl_eye_patch=[]
        tl_mouth_patch=[]
        tl_middle_patch=[]
        tl_labels=[]

        for i in range(nL):
            tl_eye_patch.extend(ckplus10g[i]['eye_patch'])
            del ckplus10g[i]['eye_patch']
            tl_mouth_patch.extend(ckplus10g[i]['mouth_patch'])
            del ckplus10g[i]['mouth_patch']
            tl_middle_patch.extend(ckplus10g[i]['middle_patch'])
            del ckplus10g[i]['middle_patch']
            tl_labels.extend(ckplus10g[i]['labels'])
            del ckplus10g[i]['labels']
            del ckplus10g[i]['imgs']
            del ckplus10g[i]['geometry']
            del ckplus10g[i]['inner_face']
            #print(ckplus10g[i]['labels'])
        print('Initializing train dataset......')
        test=DataSetFor3KindsDataV4(labels=tl_labels,
                                    eyep= tl_eye_patch, middlep= tl_middle_patch,
                                    mouthp= tl_mouth_patch,
                                    num_Classes=cn, Df=Df, one_hot=one_hot, reshape=reshape,
                                    loadGeo=False, loadImg=False, loadPat=True)
    #return Datasets(train=None, test = test, validation = None)
    elif Module==4:
        tl_innerf=[]
        tl_labels=[]

        for i in range(nL):
            tl_innerf.extend( ckplus10g[i]['inner_face'])
            del ckplus10g[i]['inner_face']
            tl_labels.extend(ckplus10g[i]['labels'])
            del ckplus10g[i]['labels']
            del ckplus10g[i]['imgs']
            del ckplus10g[i]['geometry']
            del ckplus10g[i]['eye_patch']
            del ckplus10g[i]['mouth_patch']
            del ckplus10g[i]['middle_patch']
            #print(ckplus10g[i]['labels'])
        print('Initializing train dataset......')
        test=DataSetFor3KindsDataV4(labels=tl_labels,innerf= tl_innerf, 
                                    num_Classes=cn, Df=Df, one_hot=one_hot, reshape=reshape,
                                    loadGeo=False, loadImg=False, loadPat=False, loadInnerF=True)
    
    else:
        print('ERROR: Unexpected Module in loadPKLData_V4')
        exit(1)
    return test
def loadPKLDataWithPartitions_v4(datafilepath, res_img=False, Patches=False, Geometry=False, Df=False, one_hot=False, reshape=False, cn=7):
    '''Load the entire pkl data file with original partitions.
    return the preprocessed data in a list'''
    if(os.path.exists(datafilepath)):
        print('Loading data from file: %s'%datafilepath)
    else:
        print('Cannot find the data file: %s'%datafilepath)
        exit(-1)
    with open(datafilepath,'rb') as datafile:
        ckplus10g=pickle.load(datafile)
    nL=len(ckplus10g)
    print('%s preprocessed data loaded %d'%(datafilepath,nL))

    for i in range(nL):
        if res_img:
            r, c=ckplus10g[i]['imgs'][0].shape
            for j in range(len(ckplus10g[i]['imgs'])):
                ckplus10g[i]['imgs'][j]=ckplus10g[i]['imgs'][j].astype(numpy.float32)
                ckplus10g[i]['imgs'][j]=numpy.multiply(ckplus10g[i]['imgs'][j], 1.0/255.0)
                ckplus10g[i]['imgs'][j]=numpy.reshape(ckplus10g[i]['imgs'][j], [r, c, 1])
        else:
            del ckplus10g[i]['imgs']
        if Patches:
            r, c=ckplus10g[i]['eye_patch'][0].shape
            for j in range(len(ckplus10g[i]['eye_patch'])):
                ckplus10g[i]['eye_patch'][j]=ckplus10g[i]['eye_patch'][j].astype(numpy.float32)
                ckplus10g[i]['eye_patch'][j]=numpy.multiply(ckplus10g[i]['eye_patch'][j], 1.0/255.0)
                ckplus10g[i]['eye_patch'][j]=numpy.reshape(ckplus10g[i]['eye_patch'][j], [r, c, 1])
            r, c=ckplus10g[i]['mouth_patch'][0].shape
            for j in range(len(ckplus10g[i]['mouth_patch'])):
                ckplus10g[i]['mouth_patch'][j]=ckplus10g[i]['mouth_patch'][j].astype(numpy.float32)
                ckplus10g[i]['mouth_patch'][j]=numpy.multiply(ckplus10g[i]['mouth_patch'][j], 1.0/255.0)
                ckplus10g[i]['mouth_patch'][j]=numpy.reshape(ckplus10g[i]['mouth_patch'][j], [r, c, 1])
            r, c=ckplus10g[i]['middle_patch'][0].shape
            for j in range(len(ckplus10g[i]['middle_patch'])):
                ckplus10g[i]['middle_patch'][j]=ckplus10g[i]['middle_patch'][j].astype(numpy.float32)
                ckplus10g[i]['middle_patch'][j]=numpy.multiply(ckplus10g[i]['middle_patch'][j], 1.0/255.0)
                ckplus10g[i]['middle_patch'][j]=numpy.reshape(ckplus10g[i]['middle_patch'][j], [r, c, 1])
            r, c=ckplus10g[i]['inner_face'][0].shape
            for j in range(len(ckplus10g[i]['inner_face'])):
                ckplus10g[i]['inner_face'][j]=ckplus10g[i]['inner_face'][j].astype(numpy.float32)
                ckplus10g[i]['inner_face'][j]=numpy.multiply(ckplus10g[i]['inner_face'][j], 1.0/255.0)
                ckplus10g[i]['inner_face'][j]=numpy.reshape(ckplus10g[i]['inner_face'][j], [r, c, 1])
        else:
            del ckplus10g[i]['eye_patch']
            del ckplus10g[i]['mouth_patch']
            del ckplus10g[i]['middle_patch']
            del ckplus10g[i]['inner_face']
        if not Geometry:
            del ckplus10g[i]['geometry']
        if one_hot:
            ckplus10g[i]['labels']=dense_to_one_hot(ckplus10g[i]['labels'], num_classes=7)
        
        #print(ckplus10g[i]['labels'])
    return ckplus10g
def loadCKplus10gdata_v4(datafilepath, validation_no=1, test_no=0, Module=0, Df=False, one_hot=True, reshape=False, cn=7):
    '''return the CKplus preprocessed data in a class with convinient function to access it
    validation_no and test_no must be integer values from 0 to 9'''
    if(os.path.exists(datafilepath)):
        print('Loading data from file: %s'%datafilepath)
    else:
        print('Cannot find the data file: %s'%datafilepath)
        exit(-1)
    with open(datafilepath,'rb') as datafile:
        ckplus10g=pickle.load(datafile)
    nL=len(ckplus10g)
    print('%s preprocessed data loaded %d'%(datafilepath,nL))
    if Module==0:
        tl_rescaleimgs=[]
        tl_geometry=[]
        tl_eye_patch=[]
        tl_mouth_patch=[]
        tl_middle_patch=[]
        tl_innerf=[]
        tl_labels=[]

        for i in range(nL):
            if i==test_no:
                test=DataSetFor3KindsDataV4(res_images=ckplus10g[i]['imgs'], labels=ckplus10g[i]['labels'],
                                            geometry=ckplus10g[i]['geometry'], eyep=ckplus10g[i]['eye_patch'],
                                            middlep=ckplus10g[i]['middle_patch'], mouthp=ckplus10g[i]['mouth_patch'],
                                            innerf=ckplus10g[i]['inner_face'], one_hot=one_hot, num_Classes=cn,Df=Df,reshape=reshape,
                                            loadGeo=True, loadImg=True, loadPat=True, loadInnerF=True)
                if test_no==validation_no:
                    valid=test
            elif i==validation_no:
                valid=DataSetFor3KindsDataV4(res_images=ckplus10g[i]['imgs'], labels=ckplus10g[i]['labels'],
                                            geometry=ckplus10g[i]['geometry'], eyep=ckplus10g[i]['eye_patch'],
                                            middlep=ckplus10g[i]['middle_patch'], mouthp=ckplus10g[i]['mouth_patch'],
                                            innerf=ckplus10g[i]['inner_face'], one_hot=one_hot, num_Classes=cn,Df=Df,reshape=reshape,
                                            loadGeo=True, loadImg=True, loadPat=True, loadInnerF=True)
            else:
                tl_rescaleimgs.extend(ckplus10g[i]['imgs'])
                del ckplus10g[i]['imgs']
                tl_geometry.extend(ckplus10g[i]['geometry'])
                del ckplus10g[i]['geometry']
                tl_eye_patch.extend(ckplus10g[i]['eye_patch'])
                del ckplus10g[i]['eye_patch']
                tl_mouth_patch.extend(ckplus10g[i]['mouth_patch'])
                del ckplus10g[i]['mouth_patch']
                tl_middle_patch.extend(ckplus10g[i]['middle_patch'])
                del ckplus10g[i]['middle_patch']
                tl_innerf.extend( ckplus10g[i]['inner_face'])
                del ckplus10g[i]['inner_face']
                tl_labels.extend(ckplus10g[i]['labels'])
                del ckplus10g[i]['labels']
            #print(ckplus10g[i]['labels'])
        print('Initializing train dataset......')
        train=DataSetFor3KindsDataV4(res_images=tl_rescaleimgs, labels=tl_labels,geometry = tl_geometry,
                                    eyep= tl_eye_patch, middlep= tl_middle_patch,
                                    mouthp= tl_mouth_patch, innerf= tl_innerf, 
                                    num_Classes=cn, Df=Df, one_hot=one_hot, reshape=reshape,
                                    loadGeo=True, loadImg=True, loadPat=True, loadInnerF=False)
    elif Module==1:
        tl_rescaleimgs=[]
        tl_labels=[]

        for i in range(nL):
            if i==test_no:
                test=DataSetFor3KindsDataV4(res_images=ckplus10g[i]['imgs'], labels=ckplus10g[i]['labels'],
                                            num_Classes=cn,Df=Df, one_hot=one_hot, reshape=reshape,
                                            loadGeo=False, loadImg=True, loadPat=False, loadInnerF=False)
                if test_no==validation_no:
                    valid=test
            elif i==validation_no:
                valid=DataSetFor3KindsDataV4(res_images=ckplus10g[i]['imgs'], labels=ckplus10g[i]['labels'],
                                             num_Classes=cn, Df=Df, one_hot=one_hot, reshape=reshape,
                                             loadGeo=False, loadImg=True, loadPat=False, loadInnerF=False)
            else:
                tl_rescaleimgs.extend(ckplus10g[i]['imgs'])
                del ckplus10g[i]['imgs']
                tl_labels.extend(ckplus10g[i]['labels'])
                del ckplus10g[i]['labels']
                del ckplus10g[i]['geometry']
                del ckplus10g[i]['eye_patch']
                del ckplus10g[i]['mouth_patch']
                del ckplus10g[i]['middle_patch']
                del ckplus10g[i]['inner_face']
        print('Initializing train dataset......')
        train=DataSetFor3KindsDataV4(res_images=tl_rescaleimgs, labels=tl_labels,
                                    num_Classes=cn, Df=Df, one_hot=one_hot, reshape=reshape,
                                    loadGeo=False, loadImg=True, loadPat=False, loadInnerF=False)
    elif Module==2:
        tl_geometry=[]
        tl_labels=[]
        print('\nGeometry feature dimension: %d\n'%(len(ckplus10g[0]['geometry'][0])))
        for i in range(nL):
            if i==test_no:
                test=DataSetFor3KindsDataV4(labels=ckplus10g[i]['labels'], geometry=ckplus10g[i]['geometry'],
                                            num_Classes=cn, Df=Df, one_hot=one_hot, reshape=reshape,
                                            loadGeo=True, loadImg=False, loadPat=False, loadInnerF=False)
                if test_no==validation_no:
                    valid=test
            elif i==validation_no:
                valid=DataSetFor3KindsDataV4(labels=ckplus10g[i]['labels'], geometry=ckplus10g[i]['geometry'],
                                            num_Classes=cn, Df=Df, one_hot=one_hot, reshape=reshape,
                                            loadGeo=True, loadImg=False, loadPat=False, loadInnerF=False)
            else:
                tl_geometry.extend(ckplus10g[i]['geometry'])
                del ckplus10g[i]['geometry']
                tl_labels.extend(ckplus10g[i]['labels'])
                del ckplus10g[i]['labels']
                del ckplus10g[i]['imgs']
                del ckplus10g[i]['eye_patch']
                del ckplus10g[i]['mouth_patch']
                del ckplus10g[i]['middle_patch']
                del ckplus10g[i]['inner_face']
        print('Initializing train dataset......')
        train=DataSetFor3KindsDataV4(labels=tl_labels,geometry = tl_geometry,
                                    num_Classes=cn, Df=Df, one_hot=one_hot, reshape=reshape,
                                    loadGeo=True, loadImg=False, loadPat=False, loadInnerF=False)
    elif Module==3:
        tl_eye_patch=[]
        tl_mouth_patch=[]
        tl_middle_patch=[]
        tl_labels=[]

        for i in range(nL):
            if i==test_no:
                test=DataSetFor3KindsDataV4(labels=ckplus10g[i]['labels'], eyep=ckplus10g[i]['eye_patch'],
                                            middlep=ckplus10g[i]['middle_patch'], mouthp=ckplus10g[i]['mouth_patch'],
                                            one_hot=one_hot, num_Classes=cn,Df=Df,reshape=reshape,
                                            loadGeo=False, loadImg=False, loadPat=True, loadInnerF=False)
                if test_no==validation_no:
                    valid=test
            elif i==validation_no:
                valid=DataSetFor3KindsDataV4(labels=ckplus10g[i]['labels'], eyep=ckplus10g[i]['eye_patch'],
                                            middlep=ckplus10g[i]['middle_patch'], mouthp=ckplus10g[i]['mouth_patch'],
                                            one_hot=one_hot, num_Classes=cn,Df=Df,reshape=reshape,
                                            loadGeo=False, loadImg=False, loadPat=True, loadInnerF=False)
            else:
                tl_eye_patch.extend(ckplus10g[i]['eye_patch'])
                del ckplus10g[i]['eye_patch']
                tl_mouth_patch.extend(ckplus10g[i]['mouth_patch'])
                del ckplus10g[i]['mouth_patch']
                tl_middle_patch.extend(ckplus10g[i]['middle_patch'])
                del ckplus10g[i]['middle_patch']
                tl_labels.extend(ckplus10g[i]['labels'])
                del ckplus10g[i]['labels']
                del ckplus10g[i]['imgs']
                del ckplus10g[i]['geometry']
                del ckplus10g[i]['inner_face']
                #print(ckplus10g[i]['labels'])
        print('Initializing train dataset......')
        train=DataSetFor3KindsDataV4(labels=tl_labels,
                                    eyep= tl_eye_patch, middlep= tl_middle_patch,
                                    mouthp= tl_mouth_patch,
                                    num_Classes=cn, Df=Df, one_hot=one_hot, reshape=reshape,
                                    loadGeo=False, loadImg=False, loadPat=True, loadInnerF=False)
    elif Module==4:
        tl_innerf=[]
        tl_labels=[]

        for i in range(nL):
            if i==test_no:
                test=DataSetFor3KindsDataV4(labels=ckplus10g[i]['labels'],innerf=ckplus10g[i]['inner_face'], 
                                            one_hot=one_hot, num_Classes=cn,Df=Df,reshape=reshape,
                                            loadGeo=False, loadImg=False, loadPat=False, loadInnerF=True)
                if test_no==validation_no:
                    valid=test
            elif i==validation_no:
                valid=DataSetFor3KindsDataV4(labels=ckplus10g[i]['labels'],innerf=ckplus10g[i]['inner_face'], 
                                            one_hot=one_hot, num_Classes=cn,Df=Df,reshape=reshape,
                                            loadGeo=False, loadImg=False, loadPat=False, loadInnerF=True)
            else:
                tl_innerf.extend( ckplus10g[i]['inner_face'])
                del ckplus10g[i]['inner_face']
                tl_labels.extend(ckplus10g[i]['labels'])
                del ckplus10g[i]['labels']
                del ckplus10g[i]['imgs']
                del ckplus10g[i]['geometry']
                del ckplus10g[i]['eye_patch']
                del ckplus10g[i]['mouth_patch']
                del ckplus10g[i]['middle_patch']
                #print(ckplus10g[i]['labels'])
        print('Initializing train dataset......')
        train=DataSetFor3KindsDataV4(labels=tl_labels,innerf= tl_innerf, 
                                    num_Classes=cn, Df=Df, one_hot=one_hot, reshape=reshape,
                                    loadGeo=False, loadImg=False, loadPat=False, loadInnerF=True)
    else:
        print('ERROR: Unexpected Module in loadPKLData_V4')
        exit(1)

    
    return Datasets(train=train, test = test, validation = valid)

def loadandConvertTo6LabelsWithoutNatual(datalist):
    for file in datalist:
        if (os.path.exists(file)):
            print('\n\nLoading data from file: %s'%file)
        else:
            print('\n\nCan not find %s'%file)
            continue
        did=file.split('_')[0]
        newname=file.replace(did, did.replace('\\D','\\D66'))
        newname=newname.replace('.pkl','_label0to5.pkl')
        print('New file path: %s'%newname)
        with open(file,'rb') as datafile:
            data=pickle.load(datafile)
        nL=len(data)
        count=0
        for i in range(nL):
            for index, value in enumerate(data[i]['labels']):
                if int(value)==0:
                    count=count+1
                    print('locate and remove %d'%count)
                    #data[i]['labels'].remove(value)
                    #data[i]['imgs'].remove(data[i]['imgs'][index])
                    #data[i]['geometry'].remove(data[i]['geometry'][index])
                    #data[i]['eye_patch'].remove(data[i]['eye_patch'][index])
                    #data[i]['mouth_patch'].remove(data[i]['mouth_patch'][index])
                    #data[i]['middle_patch'].remove(data[i]['middle_patch'][index])
                    #data[i]['inner_face'].remove(data[i]['inner_face'][index])
                    data[i]['labels'].pop(index)
                    data[i]['imgs'].pop(index)
                    data[i]['geometry'].pop(index)
                    data[i]['eye_patch'].pop(index)
                    data[i]['mouth_patch'].pop(index)
                    data[i]['middle_patch'].pop(index)
                    data[i]['inner_face'].pop(index)
                    if index< len(data[i]['labels']):
                        while data[i]['labels'][index]==value:
                            count=count+1
                            print('locate and remove %d'%count)
                            data[i]['labels'].pop(index)
                            data[i]['imgs'].pop(index)
                            data[i]['geometry'].pop(index)
                            data[i]['eye_patch'].pop(index)
                            data[i]['mouth_patch'].pop(index)
                            data[i]['middle_patch'].pop(index)
                            data[i]['inner_face'].pop(index)
                            if index==len(data[i]['labels']):
                                break
            for index in range(len(data[i]['labels'])):#labels starts from 0
                data[i]['labels'][index]=int(data[i]['labels'][index])-1
                    
        with open(newname,'wb') as fin:
            pickle.dump(data, fin, 4)
        del data