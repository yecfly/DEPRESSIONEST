import os, pickle, time, sys, traceback
import DataSetPrepare as DSP

datalist=[]
#dfile='./Datasets/D10_CKplus_10groups_groupedbythe_CKplus-group-details_preprocessdata_with_calibRotation_rescaleimg_geometricfeatures_facepatches_weberface_skip-contempV2.pkl'
#datalist.append(dfile)
#dfile='./Datasets/D33_KDEF_10G_rescaleimg_geometryfeature_patches_web.pkl'
#datalist.append(dfile)
#dfile='./Datasets/D40_jaffe_10groups_groupedbysubjects_rescaleimg_geometricfeatures_facepatches_weber.pkl'
#datalist.append(dfile)

#dfile='./Datasets/D16_CKPLUS_10G_Enlargeby2015CCV_10T.pkl'
#datalist.append(dfile)
#dfile='./Datasets/D34_KDEF_10G_Enlargeby2015CCV_10T.pkl'
#datalist.append(dfile)
#dfile='./Datasets/D43_JAFFE_10G_Enlargeby2015CCV_10T.pkl'
#datalist.append(dfile)

#pklname='I:/Data/detected_records/detected/data_with_geometry_and_facepatches.pkl'
#datalist.append(pklname)
#pklname='I:/Data/detected_records/undetected/undetected_data_with_geometry_and_facepatches.pkl'
#datalist.append(pklname)

pklname='I:/Data/OuluCasIA/D551_OuluCASIA_Weak_10G_V5_newGeo_newPatch.pkl'
datalist.append(pklname)
pklname='I:/Data/OuluCasIA/D553_OuluCASIA_Dark_10G_V5_newGeo_newPatch.pkl'
datalist.append(pklname)
pklname='I:/Data/OuluCasIA/D552_OuluCASIA_Strong_10G_V5_newGeo_newPatch.pkl'
filepath='I:/Data/OuluCasIA'
posfix='OuluCASIA_Weak_Dark_Strong_10G_V5_newGeo_newPatch'
datalist.append(pklname)

t1=time.time()
DSP.loadandMergeData_v2(datalist, 554, Df=False, Path = filepath, posfix=posfix)
t2=time.time()
print('Time consumed: %fs'%(t2-t1))