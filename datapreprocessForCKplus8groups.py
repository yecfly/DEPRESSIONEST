import os
import cv2
import time
import ntpath
import sys
import dlib
import pickle
import FaceProcessUtil as fpu

LMP1_keys=[17, 19, 21, 26, 24, 22, 37, 41, 44, 46, 27]
LMP1_Dists=[[37,41],[41,19],[37,19],[41,21],[37,21],
            [44,46],[44,24],[46,24],[44,22],[46,22]]
LMP1_triangle=[[17, 19, 21], [27, 37, 41], [17, 19, 27], 
               [41, 19, 27], [41, 17, 21], [41, 21, 27], 
               [17, 41, 27], [26, 24, 22], [27, 44, 46], 
               [26, 24, 27], [46, 24, 27], [46, 26, 22], 
               [46, 22, 27], [26, 46, 27]]

LMP2_keys=[4, 5, 48, 12, 11, 54, 49, 59, 51, 57, 53, 55, 62, 66] 
LMP2_Dists=[[51,57],[62,66],[49,59],[53,55]]
LMP2_triangle=[[4, 5, 48], [12, 11, 54], [51, 57, 48], 
               [51, 57, 54], [62, 66, 48], [62, 66, 54],
               [4, 5, 66], [12, 11, 66], [4, 5, 51], [12, 11, 51],
               [4, 5, 62], [12, 11, 62], [4, 5, 57], [12, 11, 57]]

LabelDict={'NE':0, 'AN':1, 'SU':2, 'DI':3, 'FE':4, 'HA':5, 'SA':6, 'CO':7}
CKLabelDict={'NE':0, 'AN':1, 'SU':7, 'DI':3, 'FE':4, 'HA':5, 'SA':6, 'CO':2}
SETLM=True
SETPM=True
LOGP=False
#### Initiate ################################
print("Length:%d"%(len(sys.argv)))
print("Remenber to change the path of tmdect to log the process.")
print("AND MAKE SURE............\nyou have changed the value of filenametosave at the end of the source code to save the final output data.") 
cv2.waitKey(5000)
if len(sys.argv) != 1 :
    print("Usage: python datapreprocessForCKplus.py")
    exit(1)
ckp_groups_folder_path = 'I:\Data\CK+\CKplus8G'
tmdect = 'I:/Data/CK+/feat_log8groups_new'
if not os.path.exists(tmdect):
    os.makedirs(tmdect)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./dlibmodel/shape_predictor_68_face_landmarks.dat")


tm=time.time()
cn=6
if cn==6:
    fi = open('I:\Data\CK+\CKplus8G\label6.txt')
flist = fi.readlines()
print('Total records: %d'%len(flist))
ipl=[]
lal=[]
for fn in flist:
    n, l=fn.replace('\n','').split(' ')
    #print(n,l)
    ipl.append(n)
    if cn==6:
        lal.append((int(l)-1))
    elif cn==7:
        lal.append(int(l))
print(len(ipl))
#################
##### Prepare images ##############################
cums=[]
imglist=[]
labellist=[]
afl=[]
lab=[]
group=None
gc=0
for i, v in enumerate(ipl):
    cg=v.split('/')[0]
    if group==None:
        gc=gc+1
        group=cg
        afl.append(v)
        lab.append(lal[i])
    elif group==cg:
        gc=gc+1
        afl.append(v)
        lab.append(lal[i])
    else:
        imglist.append(afl)
        labellist.append(lab)
        print('Group %s'%group)
        print(afl)
        afl=[]
        lab=[]
        cums.append(gc)
        gc=1
        group=cg
        afl.append(v)
        lab.append(lal[i])
imglist.append(afl)
labellist.append(lab)
print('Group %s'%group)
print(afl)
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
    ckplus_rim=[]
    ckplus_gf=[]
    ckplus_ep=[]
    ckplus_mp=[]
    ckplus_moup=[]
    ckplus_inner=[]
    ckplus_label=[]
    gc=0
    for i, v in enumerate(imagelist):
        image_path=v
        tm1=time.time()
        print("\n> Prepare image "+image_path + ":")
        imname = ntpath.basename(image_path)
        #complement the absolute path
        image_path=ckp_groups_folder_path+'/'+image_path
        label=lablist[i]
        count=count+1
        print("Name: %s            Label: %s"%(imname, label))

        #img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        #imgr = fpu.getLandMarkFeatures_and_ImgPatches(img, True, True, True)
        flag, img=fpu.calibrateImge(image_path)
        if flag:
            imgr = fpu.getLandMarkFeatures_and_ImgPatches(img, SETLM, SETPM, True)
        else:
            print('Unexpected case while calibrating for:'+str(image_path))
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
            exit(1)
        if not imgr[1] and SETLM:
            print('Unexpected case while getting geometry for:'+str(image_path))
            exit(1)

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
            cv2.imwrite(tmdect+'/'+imname+"_res.jpg",imgr[0])
            cv2.imwrite(tmdect+'/'+imname+"_eye.jpg",imgr[4])
            cv2.imwrite(tmdect+'/'+imname+"_mid.jpg",imgr[5])
            cv2.imwrite(tmdect+'/'+imname+"_mou.jpg",imgr[6])
            cv2.imwrite(tmdect+'/'+imname+"_innerface.jpg",imgr[7])
        tm2=time.time()
        dtm=tm2-tm1
        print("Time comsuming: %f"%(dtm))
    print("Group %d has %d samples"%(id+1, gc))
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
    filenametosave='I:/Data/CK+/D66504_CKplus_8G_V5_newGeo_newPatch_producebyreprocessimages.pkl'
elif SETLM:
    filenametosave='I:\Python\wlcTESTGEO\Datasets/D5xx_CKplus_8G_V5_newGeo_withoutPatch.pkl'
elif SETPM:
    filenametosave='I:\Python\wlcTESTGEO\Datasets/D5xx_CKplus_8G_V5_withoutGeo_newPatch.pkl'
else:
    filenametosave='I:/Data/CK+/D5xx_CKplus_8G_V5_newGeo_newPatch.pkl'

with open(filenametosave,'wb') as fin:
    pickle.dump(feature_group_of_subject,fin,4)
dtm=tm2-tm
#print(feature_group_of_subject)
cms=0
for i in feature_group_of_subject:
    if SETLM:
        print(len(i['geometry']))
        cms=cms+len(i['geometry'])
    elif SETPM:
        print(len(i['mouth_patch']))
        cms=cms+len(i['mouth_patch'])
print(len(feature_group_of_subject))
print('File saved: %s'%(filenametosave))
print("Total time comsuming: %fs for %d images"%(dtm, count))
