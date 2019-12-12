import cv2
import time
import ntpath
import sys, os
import pickle
import FaceProcessUtil as fpu

SETLM=True
SETPM=True
LOGP=False

ContinueWhileUnexpectedCaseOccur=True

#### Initiate ################################
print("Length:%d"%(len(sys.argv)))
print("AND MAKE SURE............\nyou have changed the value of filenametosave at the end of the source code to save the final output data.") 
cv2.waitKey(1000)
if len(sys.argv) != 1 :
    print("Usage: python producetestpkl.py")
    exit(1)
tm=time.time()
#change this value to the folder containing the images you want to preprocess as pkl file
#imagedir = 'I:/Python/test_imgs'
#imagedir = 'I:/Data/detected_records/detected'
#imagedir = 'I:/Data/toprocess/results/illumination'
#imagedir = 'I:/Data/toprocess/results/intrinsic'
#imagedir = 'I:/Data/toprocess/results/weberface'
#imagedir = 'I:/Data/checked_front_regroup_oneframe'
#filelist=imagedir+'/front_file_list_oneframe.txt'
#imagedir = 'I:/Data/FusionProject'
#filelist=imagedir+'/V1_label_random.txt'
imagedir = 'H:/435Shares/YYS/FusionProject'
filelist=imagedir+'/V2_label_random.txt'
tailtag='_randomOrder'
headtag='D_data'
#filelist=imagedir+'/V2_label.txt'
#tailtag='_categoryOrder'
fi = open(filelist)##try to find generatelabeltxt.py in Learning dir to generate label.txt
tmdect = imagedir+'/feat_log_new'
if not os.path.exists(tmdect):
    os.makedirs(tmdect)

errorlog=imagedir+'/unexpected case while preprocessing.txt'
flist = fi.readlines()
print('Total records: %d'%len(flist))
ipl=[]
lal=[]
for fn in flist:
    n, l=fn.replace('\n','').split(' ')
    ipl.append(n)
    lal.append(int(l))
print(len(ipl))
#################
##### Prepare images ##############################
cums=[]
imglist=[]
labellist=[]
for i, v in enumerate(ipl):
    if lal[i]<7:
        imglist.append(v)
        labellist.append(lal[i])
print(len(imglist))
print("Total training images:%d"%(len(imglist)))


count=0
feature_group_of_subject=[]
imagelist=imglist
lablist=labellist
ckplus_rim=[]
ckplus_gf=[]
ckplus_ep=[]
ckplus_mp=[]
ckplus_moup=[]
ckplus_inner=[]
ckplus_label=[]
gc=0
fc=0
for i, v in enumerate(imagelist):
    image_path=v
    tm1=time.time()
    print("\n> Prepare image "+image_path + ":")
    imname = ntpath.basename(image_path)
    #complement the absolute path
    image_path=imagedir+'/'+image_path
    label=lablist[i]
    count=count+1
    print("Name: %s            Label: %s\n%s"%(imname, label, image_path))
    #img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    #imgr = fpu.getLandMarkFeatures_and_ImgPatches(img, SETLM, SETPM, True)

    flag, img=fpu.calibrateImge(image_path)
    if flag:
        imgr = fpu.getLandMarkFeatures_and_ImgPatches(img, SETLM, SETPM, True)
    else:
        print('Unexpected case while calibrating for:'+str(image_path))
        if ContinueWhileUnexpectedCaseOccur:
            fo=open(errorlog,'a')
            fo.write('No face was detected in %s\n'%(str(image_path)))
            fo.close()
            fc=fc+1
            continue
        else:
            exit(1)

    if imgr[3] or imgr[1]:
        gc=gc+1
        ckplus_rim.append(imgr[0])
        if SETLM:
            print("Get Geometry>>>>>>>>>>>>>>")
            ckplus_gf.append(imgr[2])
        if SETPM:
            print("Get Patches>>>>>>>>>>>>>>")
            ckplus_ep.append(imgr[4])
            ckplus_mp.append(imgr[5])
            ckplus_moup.append(imgr[6])
            ckplus_inner.append(imgr[7])
        ckplus_label.append(label)
    if not imgr[3] and SETPM:
        print('Unexpected case while getting patches for:'+str(image_path))
        if ContinueWhileUnexpectedCaseOccur:
            fo=open(errorlog,'a')
            fo.write('Cannot extract Patches from %s\n'%(str(image_path)))
            fo.close()
            fc=fc+1
            continue
        else:
            exit(1)
    if not imgr[1] and SETLM:
        print('Unexpected case while getting geometry for:'+str(image_path))
        if ContinueWhileUnexpectedCaseOccur:
            fo=open(errorlog,'a')
            fo.write('Cannot extract LandMarks from %s\n'%(str(image_path)))
            fo.close()
            fc=fc+1
            continue
        else:
            exit(1)

    if LOGP:#log the process
        '''
        #img=imgr[0]
        ##print(img.shape)
        #img2 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        ##img2 = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)
        #dets = detector(img, 1)

        #print(">     Number of faces detected: {}".format(len(dets)))
        #if len(dets) == 0:
        #    print('> Could not detect the face, skipping the image...' + image_path)
        #else:
        #    if len(dets) > 1:
        #        print("> Process only the first detected face!")
        #    detected_face = dets[0]
  
        #    print("> cropByLM ")
            #shape = predictor(img, detected_face)
            #nLM = shape.num_parts
            #nLM = len(LMP1_keys)
            #for i in range(0,nLM):
            #    cv2.circle(img2, (shape.part(LMP1_keys[i]).x, shape.part(LMP1_keys[i]).y), 2, (0,0,255),2)
                #cv2.putText(img2,str(LMP1_keys[i]),(shape.part(LMP1_keys[i]).x,shape.part(LMP1_keys[i]).y),0,1,(0,255,0),1)
                #fileE.write(' %d %d'%(shape.part(i).x, shape.part(i).y))
            #nLM = len(LMP1_triangle)
            #for i in range(nLM):
            #    cv2.line(img2, (shape.part(LMP1_triangle[i][0]).x, shape.part(LMP1_triangle[i][0]).y), 
            #                (shape.part(LMP1_triangle[i][1]).x, shape.part(LMP1_triangle[i][1]).y),
            #                (0,255,0))
            #    cv2.line(img2, (shape.part(LMP1_triangle[i][1]).x,shape.part(LMP1_triangle[i][1]).y),
            #                (shape.part(LMP1_triangle[i][2]).x,shape.part(LMP1_triangle[i][2]).y),
            #                (0,255,0))
            #    cv2.line(img2, (shape.part(LMP1_triangle[i][0]).x,shape.part(LMP1_triangle[i][0]).y), 
            #                (shape.part(LMP1_triangle[i][2]).x,shape.part(LMP1_triangle[i][2]).y),
            #                (0,255,0))
            #nLM = len(LMP2_keys)
            #for i in range(nLM):
            #    cv2.circle(img2, (shape.part(LMP2_keys[i]).x, shape.part(LMP2_keys[i]).y), 2, (0,0,255),2)
            #nLM = len(LMP2_triangle)
            #for i in range(nLM):
            #    cv2.line(img2, (shape.part(LMP2_triangle[i][0]).x, shape.part(LMP2_triangle[i][0]).y), 
            #                (shape.part(LMP2_triangle[i][1]).x, shape.part(LMP2_triangle[i][1]).y),
            #                (0,255,0))
            #    cv2.line(img2, (shape.part(LMP2_triangle[i][1]).x,shape.part(LMP2_triangle[i][1]).y),
            #                (shape.part(LMP2_triangle[i][2]).x,shape.part(LMP2_triangle[i][2]).y),
            #                (0,255,0))
            #    cv2.line(img2, (shape.part(LMP2_triangle[i][0]).x,shape.part(LMP2_triangle[i][0]).y), 
            #                (shape.part(LMP2_triangle[i][2]).x,shape.part(LMP2_triangle[i][2]).y),
            #                (0,255,0))
        #cv2.imwrite(tmdect+'/'+imname+"_detect.jpg",img2)'''
        cv2.imwrite(tmdect+'/'+imname+"_res.jpg",imgr[0])
        cv2.imwrite(tmdect+'/'+imname+"_eye.jpg",imgr[4])
        cv2.imwrite(tmdect+'/'+imname+"_mid.jpg",imgr[5])
        cv2.imwrite(tmdect+'/'+imname+"_mou.jpg",imgr[6])
        #cv2.imwrite(tmdect+'/'+imname+"_innerface.jpg",imgr[7])

    tm2=time.time()
    dtm=tm2-tm1
    print("Time comsuming: %f"%(dtm))
print("Has %d samples"%(gc))
ckplus={}
ckplus['imgs']=ckplus_rim
ckplus['labels']=ckplus_label
if SETLM:
    ckplus['geometry']=ckplus_gf
if SETPM:
    ckplus['eye_patch']=ckplus_ep
    ckplus['middle_patch']=ckplus_mp
    ckplus['mouth_patch']=ckplus_moup
    ckplus['inner_face']=ckplus_inner
feature_group_of_subject.append(ckplus)
if SETLM and SETPM:
    filenametosave=imagedir+'/%s_with_geometry_and_facepatches%s.pkl'%(headtag, tailtag)
elif SETLM:
    filenametosave=imagedir+'/%s_without_facepatches%s.pkl'%(headtag, tailtag)
elif SETPM:
    filenametosave=imagedir+'/%s_without_geometry%s.pkl'%(headtag, tailtag)
else:
    filenametosave=imagedir+'/%s_with_only_res_image%s.pkl'%(headtag, tailtag)

with open(filenametosave,'wb') as fin:
    pickle.dump(feature_group_of_subject,fin,4)
dtm=tm2-tm
logfile=filenametosave.replace('.pkl','_info.txt')
fin=open(logfile,'w')
fin.write('File saved: %s\n'%(filenametosave))
fin.write('Total images: %d\tGet: %d\tFail: %d\n'%(count, gc, fc))
fin.write("Total time comsuming: %fs for %d images\n"%(dtm, count))
fin.close()
print(len(feature_group_of_subject))
print('File saved: %s'%(filenametosave))
print('Total images: %d\tGet: %d\tFail: %d'%(count, gc, fc))
print("Total time comsuming: %fs for %d images"%(dtm, count))
