import cv2 as cv
import numpy as np
import base64
from io import BytesIO
import json

def GetImage(imageData):
    try:
        starter = imageData.find(',')
        base64ImgData = imageData[starter+1:]
        base64ImgData = bytes(base64ImgData, encoding="ascii")
        img = BytesIO(base64.b64decode(base64ImgData))
        return ({'message':'Image Data Found', 'status':True, 'data':img})
    except Exception as e:
        # Return a Message that showing the need of base64 image data
        return ({'message':'Image Data Not Found', 'status':False})

def CheckValidImage(imageData):
    try:
        format, imgStr = imageData.split(';base64,')
        ext = format.split('/')[-1]
        return ({'message':'Valid Image Data', 'status':True, 'data':imgStr})
    except Exception as e:
        return ({'message':'Invalid Image Data', 'status':False})
        

def ObjectRegistration(imageData):
    img = CheckValidImage(imageData)
    if not img['status']:
        return ({'message': img['message'], 'status': False})
    return ({'message': img['message'], 'status': True, 'data': img['data']})

def ObjectMatching(imageData, imageDataFromDB):
    imgFromReq = GetImage(imageData)
    if not imgFromReq['status']:
        return ({'message': imgFromReq['message'], 'status': False})
    try:
        imgFromReq = cv.imdecode(np.frombuffer(imgFromReq['data'].read(), np.uint8), cv.IMREAD_UNCHANGED)
        grayImg1 = cv.cvtColor(imgFromReq, cv.COLOR_BGR2GRAY)

        imgFromDB = bytes(imageDataFromDB, encoding="ascii")
        imgFromDB = BytesIO(base64.b64decode(imgFromDB))
        imgFromDB = cv.imdecode(np.frombuffer(imgFromDB.read(), np.uint8), cv.IMREAD_UNCHANGED)
        grayImg2 = cv.cvtColor(imgFromDB, cv.COLOR_BGR2GRAY)

        # Resize both the images
        resize_ = (520,520)
        # Check size of both the images
        if grayImg1.shape[0] > resize_[0] and grayImg1.shape[1] > resize_[1]:
            grayImg1 = cv.resize(grayImg1, resize_)
        if grayImg2.shape[0] > resize_[0] and grayImg2.shape[1] > resize_[1]:
            grayImg2 = cv.resize(grayImg2, resize_)

        # cv.imshow('Image 1', grayImg1)
        # cv.imshow('Image 2', grayImg2)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        # sift = cv.SIFT_create(1000)
        orb = cv.ORB_create(1000)
        kp1, des1 = orb.detectAndCompute(grayImg1, None)
        kp2, des2 = orb.detectAndCompute(grayImg2, None)
        
        # FLANN -> FAST Library for Approximate Nearest Neighbors
        # For ORB
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        # For SIFT
        # FLANN_INDEX_KDTREE = 1
        # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        # search_params = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        goodMatches = 0
        for i, match in enumerate(matches):
            if len(match) == 2:
                (m, n) = match
                if m.distance < 0.75 * n.distance:
                    goodMatches += 1
            else:
                pass
        # print(goodMatches)
        if goodMatches > 50:
            return ({'message': 'Object Matched', 'status': True, 'matches': goodMatches, 'flag': True})
        else:
            return ({'message': 'Object Not Matched', 'status': False, 'flag': True})
    except Exception as e:
        return ({'message': 'Something Went Wrong', 'flag': False})
