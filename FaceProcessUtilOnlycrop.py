import math
from datetime import datetime

import cv2
import dlib
import numpy as np
from PIL import Image as IM
from scipy import ndimage
import time
# --------------------------------------------------------------------------- #
# Usage: python facepatches.py <inputDir> <outputDir>
# --------------------------------------------------------------------------- #

#---------------------------------------------------------------------------#

#rescaleImg = [1.4504, 1.6943, 1.4504, 1.2065]
#mpoint = [63.1902642394822, 47.2030047734627]

rescaleImg = [1.4504, 1.5843, 1.4504, 1.3165]
mpoint = [63.78009, 41.66620]
target_size = 128


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./dlibmodel/shape_predictor_68_face_landmarks.dat")

def __cropImg(img, shape=None, trg_size=target_size, rescale=rescaleImg):
    """Rescale, adjust, and crop the images.
    If shape is None, it will recale the img without croping and return __genLMFandIP"""
    if not shape==None:
        nLM = shape.num_parts
        lms_x = np.asarray([shape.part(i).x for i in range(0,nLM)])
        lms_y = np.asarray([shape.part(i).y for i in range(0,nLM)])

        tlx = float(min(lms_x))#top left x
        tly = float (min(lms_y))#top left y
        ww = float (max(lms_x) - tlx)
        hh = float(max(lms_y) - tly)
        # Approximate LM tight BB
        h = img.shape[0]
        w = img.shape[1]
        cx = tlx + ww/2
        cy = tly + hh/2
        #tsize = max(ww,hh)/2
        tsize = ww/2

        # Approximate expanded bounding box
        btlx = int(round(cx - rescale[0]*tsize))
        btly = int(round(cy - rescale[1]*tsize))
        bbrx = int(round(cx + rescale[2]*tsize))
        bbry = int(round(cy + rescale[3]*tsize))
        nw = int(bbrx-btlx)
        nh = int(bbry-btly)

        #adjust relative location
        x0=(np.mean(lms_x[36:42])+np.mean(lms_x[42:48]))/2
        y0=(np.mean(lms_y[36:42])+np.mean(lms_y[42:48]))/2
        Mpx=int(round((mpoint[0]*nw/float(target_size))-x0+btlx))
        Mpy=int(round((mpoint[1]*nh/float(target_size))-y0+btly))
        btlx=btlx-Mpx
        bbrx=bbrx-Mpx
        bbry=bbry-Mpy
        btly=btly-Mpy
        print('coordinate adjustment')
        print(Mpx, Mpy)
        Xa=np.round((lms_x-btlx)*trg_size/nw)
        Ya=np.round((lms_y-btly)*trg_size/nh)
        
        #few=open(eyelog,'a')
        #few.write('%lf %lf\n'%((np.mean(Xa[36:42])+np.mean(Xa[42:48]))/2,(np.mean(Ya[36:42])+np.mean(Ya[42:48]))/2))
        #few.close()

        imcrop = np.zeros((nh,nw,3), dtype = "uint8")

        blxstart = 0
        if btlx < 0:
            blxstart = -btlx
            btlx = 0
        brxend = nw
        if bbrx > w:
            brxend = w+nw - bbrx#brxend=nw-(bbrx-w)
            bbrx = w
        btystart = 0
        if btly < 0:
            btystart = -btly
            btly = 0
        bbyend = nh
        if bbry > h:
            bbyend = h+nh - bbry#bbyend=nh-(bbry-h)
            bbry = h
        imcrop[btystart:bbyend, blxstart:brxend, 0:3] = img[btly:bbry, btlx:bbrx, 0:3]
        im_rescale=cv2.resize(imcrop,(trg_size, trg_size))
        return im_rescale
    else:
        im_rescale=cv2.resize(img, (trg_size, trg_size))
        return im_rescale

def getLandMarkFeatures_and_ImgPatches(img):
    """Input:
    img: image to be processed.
Outputs: 
    rescaleimg
    rescaleimg: the rescale image of the input img
    """
    g_img = img
    td1= time.time()
    #f_ds=detector(g_img, 1)#1 represents upsample the image 1 times for detection
    f_ds=detector(g_img, 0)
    td2 = time.time()
    print('Time in detecting face: %fs'%(td2-td1))
    if len(f_ds) == 0:
        #pl.write('0')
        print(">>>***%%%Warning [getLandMarkFeatures_and_ImgPatches()]: No face was detected from the image")
        return __cropImg(g_img)
    elif len(f_ds) > 1:
        print(">>>***%%%Warning [getLandMarkFeatures_and_ImgPatches()]: Only process the first face detected.")
    f_shape = predictor(g_img, f_ds[0])
    #pl.write('1')
    return __cropImg(g_img)
def calibrateImge(imgpath):
    '''Calibrate the image of the face'''
    tm=time.time()
    imgcv=cv2.imread(imgpath, cv2.IMRead_COLOR)
    if imgcv is None:
        print('Unexpected ERROR: The value read from the imagepath is None. No image was loaded')
        exit(-1)
    dets = detector(imgcv,1)
    if len(dets)==0:
        print("No face was detected^^^^^^^^^^^^^^")
        return False, imgcv
    lmarks=[]
    for id, det in enumerate(dets):
        if id > 0:
            print("ONLY process the first face>>>>>>>>>")
            break
        shape = predictor(imgcv, det)
        x, y = __shape_to_np(shape)
    lmarks = np.asarray(lmarks, dtype='float32')
    pilimg=IM.fromarray(imgcv)
    rtimg=__RotaFace(pilimg, eye_left=(np.mean(x[36:42]),np.mean(y[36:42])),
                                           eye_right=(np.mean(x[42:48]),np.mean(y[42:48])))
    imgcv=np.array(rtimg)
    return True, imgcv

######
#
#The followings are for calibrate the image
def __RotateTranslate(image, angle, center =None, new_center =None, resample=IM.BICUBIC):
    '''Rotate the image according to the angle'''
    if center is None:  
        return image.rotate(angle=angle, resample=resample)  
    nx,ny = x,y = center  
    if new_center:  
        (nx,ny) = new_center  
    cosine = math.cos(angle)  
    sine = math.sin(angle)  
    c = x-nx*cosine-ny*sine  
    d =-sine  
    e = cosine
    f = y-nx*d-ny*e  
    return image.transform(image.size, IM.AFFINE, (cosine,sine,c,d,e,f), resample=resample)
def __RotaFace(image, eye_left=(0,0), eye_right=(0,0)):
    '''Rotate the face according to the eyes'''
    # get the direction from two eyes
    eye_direction = (eye_right[0]- eye_left[0], eye_right[1]- eye_left[1])
    # calc rotation angle in radians
    rotation =-math.atan2(float(eye_direction[1]),float(eye_direction[0]))
    # rotate original around the left eye  
    image = __RotateTranslate(image, center=eye_left, angle=rotation)
    return image
def __shape_to_np(shape):
    '''Transform the shape points into numpy array of 68*2'''
    nLM = shape.num_parts
    x = np.asarray([shape.part(i).x for i in range(0,nLM)])
    y = np.asarray([shape.part(i).y for i in range(0,nLM)])
    return x,y

### system module
crop_size=0.7
def __getLandMarkFeatures_and_ImgPatches_for_Facelist(img_list, withLM=True, withPatches=True):
    """Input:
    img_list: face image list to be processed.
    
Outputs: 
    rescaleimg
    
    rescaleimg: the rescale image of the input img
    """
    RT=[]
    for img in img_list:
        g_img = img
        f_ds=detector(g_img, 1)
        if len(f_ds) == 0:
            #pl.write('0')
            print(">>>***%%%Warning [getLandMarkFeatures_and_ImgPatches()]: No face was detected, and return None values")
            RT.append(None)
        else:
            max_area=0
            for i in range(len(f_ds)):
                f_shape = predictor(g_img, f_ds[i])
                curr_area = (f_ds[i].right()-f_ds[i].left()) * (f_ds[i].bottom()-f_ds[i].top())
                if curr_area > max_area:
                    max_area = curr_area
                    rescaleimg=__cropImg(g_img, shape=f_shape)
            RT.append((rescaleimg))
    return RT

def __calibrateImageWithArrayInput(img):
    '''Calibrate the image of the face'''
    if img is None:
        print('Unexpected ERROR: The value input is None. No image was loaded')
        return False, None, None
    imgcv=img[:]

    dets = detector(imgcv,1)
    img_face_list=[]
    rectPoint=[]
    if len(dets)==0:
        print("No face was detected^^^^^^^^^^^^^^")
        return False, img_face_list, rectPoint
    h=imgcv.shape[0]
    w=imgcv.shape[1]
    for id, det in enumerate(dets):
        shape = predictor(imgcv, det)
        x, y = __shape_to_np(shape)
        top=[]
        top.append((det.left(),det.top()))
        top.append((det.right(),det.bottom()))
        rectPoint.append(top)

        #crop face
        tlx=float(min(x))
        tly=float(min(y))
        ww=float(max(x)-tlx)
        hh=float(max(y)-tly)
        cx=tlx+ww/2
        cy=tly+hh/2
        tsize=ww*crop_size
        # Approximate expanded bounding box
        btlx = int(round(cx - rescaleImg[0]*tsize))
        btly = int(round(cy - rescaleImg[1]*tsize))
        bbrx = int(round(cx + rescaleImg[2]*tsize))
        bbry = int(round(cy + rescaleImg[3]*tsize))
        nw = int(bbrx-btlx)
        nh = int(bbry-btly)
        imcrop = np.zeros((nh,nw,3), dtype = "uint8")
        blxstart = 0
        if btlx < 0:
            blxstart = -btlx
            btlx = 0
        brxend = nw
        if bbrx > w:
            brxend = w+nw - bbrx#brxend=nw-(bbrx-w)
            bbrx = w
        btystart = 0
        if btly < 0:
            btystart = -btly
            btly = 0
        bbyend = nh
        if bbry > h:
            bbyend = h+nh - bbry#bbyend=nh-(bbry-h)
            bbry = h
        imcrop[btystart:bbyend, blxstart:brxend, 0:3] = imgcv[btly:bbry, btlx:bbrx, 0:3]
        pilimg=IM.fromarray(imcrop)
        rtimg=__RotaFace(pilimg, eye_left=(np.mean(x[36:42]),np.mean(y[36:42])),
                                           eye_right=(np.mean(x[42:48]),np.mean(y[42:48])))
        img_face_list.append(np.array(rtimg))
        im=cv2.
    return True, img_face_list, rectPoint

def preprocessImage(img):
    """process image as input for model, extract all human faces in the image and their corresponding coordinate points
        
    Args:
        img (ndarray): input image represent in numpy.ndarray
    
    Returns: a dictionnary contains the following information
        detected(boolean): bool type to indicates whether the there are human faces in the input
        rescaleimg(list of ndarray): a list of rescaled and cropped image of the detected face
        originalPoints(list of tuple): a list tuple corresponding to rescaleimg, each tuple contains tow points that represent human faces
        gf: bool type for geometry features flag, indicating whether there would be meaning values in geo_features or a just a None value
        geo_features: geometryf features or None value
        pf: bool type indicates whether the following features are meaningful or meaningless
        eyepatch: eye patch of the recaleimg
        foreheadpatch: forehead patch of the rescaleimg
        mouthpatch: mouthpatch of the rescaleimg
        innerface: croped face from the rescaleimg
    """
    crop_part = ((500, 1450), (1500, 2000)) # 4000 * 3000
    crop_part = ((120, 1050), (1400, 1700)) # 3072 * 2048
    cropped = False
    left_top, right_bottom = crop_part

    r, c, ch = img.shape
    if r >= right_bottom[0] and c >= right_bottom[1]:
        cropped = True
        print('cropping image........')
        img = img[left_top[0] : right_bottom[0], left_top[1] : right_bottom[1], 0]
        # cv2.imwrite('./crop_imgs/crop_{0}.jpeg'.format(datetime.now().strftime("%Y%m%d%H%M%S")), img)
    
    # pack the features and return   
    features = {}
    detected, face_list, originalPoints = __calibrateImageWithArrayInput(img)
    features['detected'] = detected
    if detected: # detect human face
        processedFeature = __getLandMarkFeatures_and_ImgPatches_for_Facelist(face_list, False, False)
        
        rescaleimg, detectedOriginalPoints = [], []
        for i in range(len(processedFeature)):
            if processedFeature[i]:
                # order of features
                # rescaleimg, gf, geo_features, pf, eyepatch, foreheadpatch, mouthpatch, innerface, rotatedPoints 
                rescaleimg.append(processedFeature[i][0].reshape(1, 128, 128, 1))
                detectedOriginalPoints.append(originalPoints[i])

        print('detect {0} human faces'.format(len(detectedOriginalPoints)))
        
        # save the cropped image
        # print('cropping img with face to shape {0}'.format(img.shape))
        # cv2.imwrite('./crop_imgs/crop_{0}.jpeg'.format(datetime.now().strftime("%Y%m%d%H%M%S")), img)

        # if cropping image, move the square surrounding human face to the right place 
        if cropped:
            tmp = []
            for face in detectedOriginalPoints:
                modified_left_top = (face[0][0] + left_top[1], face[0][1] + left_top[0])
                modified_right_bottom = (face[1][0] + left_top[1], face[1][1] + left_top[0])
                tmp.append((modified_left_top, modified_right_bottom))
            detectedOriginalPoints = tmp
        
        assert len(rescaleimg) == len(detectedOriginalPoints), 'the number of human faces do not equal the number of face points'
        features['rescaleimg'] = rescaleimg
        features['originalPoints'] = detectedOriginalPoints
    return features
    
########image calibration ends here
#------------------------------------------------------------------------#
