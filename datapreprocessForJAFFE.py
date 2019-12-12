import os
import numpy as np
import glob
#import cv2.cv2 as cv2
import cv2
import time
import ntpath
import sys
import dlib
import pickle, traceback
import FaceProcessUtil as fpu

#rescaleImg = [1.4441, 1.6943, 1.4567, 1.2065] #1.4441+1.4567==1.6943+1.2065
#target_size = 128

#eye_patch_size = (64, 26)
#eye_width_height_ratio = 0.40625 #alternative 0.5-0.6 including forehead

#middle_width_height_ratio = 1.25
#middle_patch_size = (32, 40)

#mouth_width_height_ratio = 0.5
#mouth_patch_size = (48, 24)

LMP1_keys=[17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 27]
LMP1_keys=[17, 19, 21, 26, 24, 22, 37, 41, 44, 46, 27]
LMP1_triangle=[[17, 19, 21], [27, 37, 41], [17, 19, 27], [41, 19, 27], [41, 17, 21], [41, 21, 27], [17, 41, 27], [26, 24, 22], [27, 44, 46], [26, 24, 27], [46, 24, 27], [46, 26, 22], [46, 22, 27], [26, 46, 27]]
LMP2_keys=[4, 5, 48, 12, 11, 54, 49, 59, 51, 57, 53, 55, 62, 66] 
LMP2_triangle=[[4, 5, 48], [12, 11, 54], [51, 57, 48], [51, 57, 54], [62, 66, 48], [62, 66, 54]]

LabelDict={'NE':0, 'AN':1, 'SU':2, 'DI':3, 'FE':4, 'HA':5, 'SA':6, 'CO':7}

SETLM=True
SETPM=False
LOGP=False

#### Initiate ################################
print("Length:%d"%(len(sys.argv)))
if len(sys.argv) != 1 :
		print("Usage: python datapreprocessForJAFFE.py")
		exit(1)
faces_folder_path = 'I:/Data/jaffe'
tmdect = 'I:/Data/jaffe/feat_log'
if not os.path.exists(tmdect):
    os.makedirs(tmdect)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../shape_predictor_68_face_landmarks.dat")


##### Prepare images ##############################
tm=time.time()
Pcount=0
Gcount=0
count=0

imagelist=[]
#for image_path in glob.glob(os.path.join(faces_folder_path, "*.tiff")):
#    imagelist.append(image_path)
imagelist.extend(glob.glob(os.path.join(faces_folder_path, '*.tiff')))
imagelist=sorted(imagelist)
for i in range(len(imagelist)):
    print(imagelist[i])
print("\n\n\nAfter sorted\n\n\n")
for i in range(len(imagelist)):
    print(imagelist[i])


feature_group_of_subject=[]
jaffe_rim=[]
jaffe_gf=[]
jaffe_ep=[]
jaffe_mp=[]
jaffe_moup=[]
jaffe_innerf=[]
jaffe_label=[]
sname=None
#for image_path in glob.glob(os.path.join(faces_folder_path, "*.tiff")):
for i in range(len(imagelist)):
    image_path=imagelist[i]
    count=count+1
    tm1=time.time()
    print("\n> Prepare image "+image_path + ":")
    imname = ntpath.basename(image_path)

    #print(imname.split('.')[-1])
    #print(imname.split('.')[0])
    #print(imname.split('.')[1][0:2])
    #print(imname.split('.')[2])
    cnt=imname.split('.')[0]
    label=imname.split('.')[1][0:2]
    imname = imname.split(imname.split('.')[-1])[0][0:-1]
    print("Name: %s                   Label: %s"%(imname, label))

    #img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    try:
        img=fpu.calibrateImge(image_path)
    except:
        traceback.print_exc()
        continue
    if img[0]:
        imgr = fpu.getLandMarkFeatures_and_ImgPatches(img[1], True, True, True)

    if imgr[3] and imgr[1]:
        if sname == cnt:#same person
            jaffe_rim.append(imgr[0])
            if SETLM:
                print("Get Geometry>>>>>>>>>>>>>>")
                Gcount +=1
                jaffe_gf.append(imgr[2])
            if SETPM:
                print("Get Patches>>>>>>>>>>>>>>")
                Pcount +=1
                jaffe_ep.append(imgr[4])
                jaffe_mp.append(imgr[5])
                jaffe_moup.append(imgr[6])
                jaffe_innerf.append(imgr[7])
            jaffe_label.append(LabelDict[label])
        elif sname==None:
            sname = cnt
            jaffe_rim.append(imgr[0])
            if SETLM:
                print("Get Geometry>>>>>>>>>>>>>>")
                Gcount +=1
                jaffe_gf.append(imgr[2])
            if SETPM:
                print("Get Patches>>>>>>>>>>>>>>")
                Pcount +=1
                jaffe_ep.append(imgr[4])
                jaffe_mp.append(imgr[5])
                jaffe_moup.append(imgr[6])
                jaffe_innerf.append(imgr[7])
            jaffe_label.append(LabelDict[label])
        else:#different person
            jaffe={}
            jaffe['imgs']=jaffe_rim
            jaffe['labels']=jaffe_label
            jaffe['geometry']=jaffe_gf
            jaffe['eye_patch']=jaffe_ep
            jaffe['middle_patch']=jaffe_mp
            jaffe['mouth_patch']=jaffe_moup
            jaffe['inner_face']=jaffe_innerf
            feature_group_of_subject.append(jaffe)

            sname = cnt
            jaffe_rim=[]
            jaffe_gf=[]
            jaffe_ep=[]
            jaffe_mp=[]
            jaffe_moup=[]
            jaffe_innerf=[]
            jaffe_label=[]

            jaffe_rim.append(imgr[0])
            if SETLM:
                print("Get Geometry>>>>>>>>>>>>>>")
                Gcount +=1
                jaffe_gf.append(imgr[2])
            if SETPM:
                print("Get Patches>>>>>>>>>>>>>>")
                Pcount +=1
                jaffe_ep.append(imgr[4])
                jaffe_mp.append(imgr[5])
                jaffe_moup.append(imgr[6])
                jaffe_innerf.append(imgr[7])
            jaffe_label.append(LabelDict[label])
    if not imgr[3]:
        print('Unexpected case while getting patches for:'+str(image_path))
    if not imgr[1]:
        print('Unexpected case while getting geometry for:'+str(image_path))

    if LOGP:#log the process
        img=imgr[0]
        #print(img.shape)
        img2 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        #img2 = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)
        dets = detector(img, 1)

        print(">     Number of faces detected: {}".format(len(dets)))
        if len(dets) == 0:
            print('> Could not detect the face, skipping the image...' + image_path)
        else:
            if len(dets) > 1:
                print("> Process only the first detected face!")
            detected_face = dets[0]
  
            print("> cropByLM ")
            shape = predictor(img, detected_face)
            #nLM = shape.num_parts
            nLM = len(LMP1_keys)
            for i in range(0,nLM):
                cv2.circle(img2, (shape.part(LMP1_keys[i]).x, shape.part(LMP1_keys[i]).y), 2, (0,0,255),2)
                #cv2.putText(img2,str(LMP1_keys[i]),(shape.part(LMP1_keys[i]).x,shape.part(LMP1_keys[i]).y),0,1,(0,255,0),1)
                #fileE.write(' %d %d'%(shape.part(i).x, shape.part(i).y))
            nLM = len(LMP1_triangle)
            for i in range(nLM):
                cv2.line(img2, (shape.part(LMP1_triangle[i][0]).x, shape.part(LMP1_triangle[i][0]).y), 
                         (shape.part(LMP1_triangle[i][1]).x, shape.part(LMP1_triangle[i][1]).y),
                         (0,255,0))
                cv2.line(img2, (shape.part(LMP1_triangle[i][1]).x,shape.part(LMP1_triangle[i][1]).y),
                         (shape.part(LMP1_triangle[i][2]).x,shape.part(LMP1_triangle[i][2]).y),
                         (0,255,0))
                cv2.line(img2, (shape.part(LMP1_triangle[i][0]).x,shape.part(LMP1_triangle[i][0]).y), 
                         (shape.part(LMP1_triangle[i][2]).x,shape.part(LMP1_triangle[i][2]).y),
                         (0,255,0))
            nLM = len(LMP2_keys)
            for i in range(nLM):
                cv2.circle(img2, (shape.part(LMP2_keys[i]).x, shape.part(LMP2_keys[i]).y), 2, (0,0,255),2)
            nLM = len(LMP2_triangle)
            for i in range(nLM):
                cv2.line(img2, (shape.part(LMP2_triangle[i][0]).x, shape.part(LMP2_triangle[i][0]).y), 
                         (shape.part(LMP2_triangle[i][1]).x, shape.part(LMP2_triangle[i][1]).y),
                         (0,255,0))
                cv2.line(img2, (shape.part(LMP2_triangle[i][1]).x,shape.part(LMP2_triangle[i][1]).y),
                         (shape.part(LMP2_triangle[i][2]).x,shape.part(LMP2_triangle[i][2]).y),
                         (0,255,0))
                cv2.line(img2, (shape.part(LMP2_triangle[i][0]).x,shape.part(LMP2_triangle[i][0]).y), 
                         (shape.part(LMP2_triangle[i][2]).x,shape.part(LMP2_triangle[i][2]).y),
                         (0,255,0))
        cv2.imwrite(tmdect+'/'+imname+"_detect.jpg",img2)
    tm2=time.time()
    dtm=tm2-tm1
    print("Time comsuming: %f"%(dtm))
jaffe={}
jaffe['imgs']=jaffe_rim
jaffe['labels']=jaffe_label
if SETLM:
    jaffe['geometry']=jaffe_gf
if SETPM:
    jaffe['eye_patch']=jaffe_ep
    jaffe['middle_patch']=jaffe_mp
    jaffe['mouth_patch']=jaffe_moup
    jaffe['inner_face']=jaffe_innerf
feature_group_of_subject.append(jaffe)

if SETLM and SETPM:
    filenametosave='I:/Data/jaffe/D44_jaffe_10G_V4_weber128x128.pkl'
elif SETLM:
    filenametosave='I:\Python\wlcTESTGEO\Datasets/D463_JAFFE_10G_V5_withoutPatches.pkl'
elif SETPM:
    filenametosave='I:\Python\wlcTESTGEO\Datasets/D400_JAFFE_10G_newM_withouGeometry.pkl'


with open(filenametosave,'wb') as fin:
    pickle.dump(feature_group_of_subject,fin,4)
dtm=tm2-tm
print(len(feature_group_of_subject))
#print(feature_group_of_subject)
#for i in feature_group_of_subject:
#    print("\n\n")
#    print(i['geometry'])
#print(jaffe_gf)
print('Saved file in %s'%filenametosave)
print("Total time comsuming: %fs for %d images"%(dtm, count))
print("Patches: %d\tGeometry: %d"%(Pcount, Gcount))