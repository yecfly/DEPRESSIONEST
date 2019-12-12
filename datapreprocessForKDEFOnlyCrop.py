import os, shutil
import numpy as np
import glob
import cv2
import time
import ntpath
import sys
import pickle
import FaceProcessUtilOnlycrop as fpu

MAPPINGfromSTRtoINTv2 = {'ne':0, 'an':1, 'su':2, 'di':3, 'fe':4, 'ha':5, 'sa':6}
MAPPINGfromINTtoSTR = {0:'ne', 1:'an', 2:'su', 3:'di', 4:'fe', 5:'ha', 6:'sa'}
LOGP=True

#### Initiate ################################
print("Length:%d"%(len(sys.argv)))
print("Remenber to change the path of tmdect to log the process.")
print("AND MAKE SURE............\nyou have changed the value of filenametosave at the end of the source code to save the final output data.") 
cv2.waitKey(5000)
if len(sys.argv) != 1 :
		print("Usage: python datapreprocessForKDEF.py")
		exit(1)


kdef_groups_folder_path = kdefdir'\KDEF'
tmdect = outputdir
if not os.path.exists(tmdect):
    os.makedirs(tmdect)

###########Read the emotion labels
tm=time.time()
fi = open(labeldir+'\label.txt')
flist = fi.readlines()
print('Total records: %d'%len(flist))
ipl=[]
lal=[]
gl=[]
for fn in flist:
    n, l, g=fn.replace('\n','').split(' ')
    #print(n,l)
    ipl.append(n)
    lal.append(int(l))
    gl.append(int(g))
print(len(ipl))
#################
##### Prepare images ##############################
cums=[]
imglist=[]
labellist=[]
afl=[]
lab=[]
groups=0
gc=0
for i, v in enumerate(ipl):
    if groups==0:
        groups=gl[i]
        afl.append(v)
        lab.append(lal[i])
        gc=gc+1
    elif not groups==gl[i]:
        print('\n\nGroup %d\n'%(groups))
        print(afl)
        imglist.append(afl)
        labellist.append(lab)
        cums.append(gc)
        gc=0
        afl=[]
        lab=[]
        groups=gl[i]
        afl.append(v)
        lab.append(lal[i])
        gc=gc+1
    else:
        afl.append(v)
        lab.append(lal[i])
        gc=gc+1
print('\n\n\nGroup %d\n'%(gl[i]))
print(afl)
imglist.append(afl)
labellist.append(lab)
cums.append(gc)
print(cums)
print(len(imglist))
print("Total training images:%d"%(sum(cums)))


count=0
feature_group_of_subject=[]
for id in range(len(imglist)):
    print("\nProcess Group %d>>>>>>>>\n"%(id+1))
    imagelist=imglist[id]
    lablist=labellist[id]
    kdef_rim=[]
    kdef_gf=[]
    kdef_ep=[]
    kdef_mp=[]
    kdef_moup=[]
    kdef_inner=[]
    kdef_label=[]
    gc=0
    for i in range(len(imagelist)):
        image_path=imagelist[i]
        tm1=time.time()
        print("\n> Prepare image "+image_path + ":")
        imname = ntpath.basename(image_path)
        label=lablist[i]
        count=count+1
        print("Name: %s            Label: %s"%(imname, label))
        image_path=kdef_groups_folder_path+'/Group'+str(id+1)+'/'+image_path
        #img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        #imgr = fpu.getLandMarkFeatures_and_ImgPatches(img, True, True, True)
        flag, img=fpu.calibrateImge(image_path)
        if flag:
            imgr = fpu.getLandMarkFeatures_and_ImgPatches(img)
        else:
            print('Unexpected case while calibrating for:'+str(image_path))
            exit(1)

        gc=gc+1
        kdef_rim.append(imgr)
        kdef_label.append(label)
       
        if LOGP:#log the process
            img2 = imgr[:]
            cv2.imwrite(tmdect+'/'+MAPPINGfromINTtoSTR.get(int(label))+'_'+imname+".jpg",img2)
        tm2=time.time()
        dtm=tm2-tm1
        print("Time comsuming: %f"%(dtm))
    print("Group %d has %d samples"%(id+1, gc))
    kdef={}
    kdef['imgs']=kdef_rim
    kdef['labels']=kdef_label
    feature_group_of_subject.append(kdef)

filenametosave='/D_KDEF_10G_only_rescale_images_with_RBG.pkl'
tm2=time.time()

with open(filenametosave,'wb') as fin:
    pickle.dump(feature_group_of_subject,fin,4)
dtm=tm2-tm

print(len(feature_group_of_subject))
print('File saved: %s'%(filenametosave))
print("Total time comsuming: %fs for %d images"%(dtm, count))
