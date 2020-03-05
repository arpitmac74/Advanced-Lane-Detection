"""
MIT License

Copyright (c) 2020 amrish1222

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.signal import find_peaks
import warnings

def undistort(calibration_matrix,dist_coeff,image):
    h,  w = image.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(calibration_matrix,dist,(w,h),0,(w,h))
    undistorted = cv2.undistort(image, calibration_matrix, dist_coeff, None, newcameramtx)
    return undistorted

def lanes(inputImg, winWidth):
    segments = 20
    h = inputImg.shape[0]
    step = int(np.ceil(inputImg.shape[1]/segments))
    laneLineLx = []
    laneLineLy = []
    laneLineRx = []
    laneLineRy = []
    
    xNzR = []
    yNzR = []
    
    xNzL = []
    yNzL = []
    
    for row in reversed(range(0,h,step)):
        ImgSeg =  inputImg[row:row+step,:]
#        print(row,row+step)
        histogram =  np.sum(ImgSeg[:,:], axis=0)
        peaks,_ = find_peaks(histogram, distance=10)
        peaksR = peaks[peaks >= histogram.shape[0]/2]
        peaksL = peaks[peaks < histogram.shape[0]/2]
#        plt.plot(histogram)
        peaksLtemp = histogram.shape[0]/2
        if peaksL.size>0 :
#            plt.plot(peaksL[-1],histogram[peaksL[-1]], 'r+')
            laneLineLx.append(peaksL[-1])
            laneLineLy.append(row + step/2)
            nzXY = ImgSeg[:, peaksL[-1]-winWidth : peaksL[-1]+winWidth].nonzero()
            xNzL.append(nzXY[1] + peaksL[-1]-winWidth)
            yNzL.append(nzXY[0] + row)
            peaksLtemp = peaksL[-1]
        if peaksR.size>0 :
#                plt.plot(peaksR[0],histogram[peaksR[0]], 'bo')
            if 450+peaksLtemp > peaksR[0]:
#                print("dist = " , peaksLtemp - peaksR[0])
                laneLineRx.append(peaksR[0])
                laneLineRy.append(row + step/2)
                nzXY = ImgSeg[:, peaksR[0]-winWidth : peaksR[0]+winWidth].nonzero()
                xNzR.append(nzXY[1] + peaksR[0]-winWidth)
                yNzR.append(nzXY[0]+ row)
#        plt.show()    
                
    if len(xNzL)<=0 or len(xNzR)<=0:
        return inputImg,False,0, "Straight"
    
    inputImg = cv2.cvtColor(inputImg,cv2.COLOR_GRAY2BGR)
    for i,j in zip(np.hstack(xNzL),np.hstack(yNzL)):
        cv2.circle(inputImg,(i,j), 2, (0,0,255), -1)
    for i,j in zip(np.hstack(xNzR),np.hstack(yNzR)):
            cv2.circle(inputImg,(i,j), 2, (0,255,0), -1)
    
    with warnings.catch_warnings():
        warnings.filterwarnings('error')    
        try:
            left = np.polyfit(np.hstack(yNzL),np.hstack(xNzL) , 2)
            right = np.polyfit(np.hstack(yNzR),np.hstack(xNzR) , 2)
            y = np.linspace(0, h, h).astype(int)
            xL = [int(left[2] + left[1]*(yy) + left[0]*(yy**2)) for yy in y]
            xR = [int(right[2] + right[1]*(yy) + right[0]*(yy**2)) for yy in y]
        except np.RankWarning:
            return inputImg,False,0, "Straight"
        
    pts = []
    pts = np.column_stack((xL,y))
    pts = np.vstack((pts,np.flip(np.column_stack((xR,y)),0)))
    laneShade = np.zeros_like(inputImg)
    for i,j,k in zip(xL, xR ,y):
        cv2.circle(inputImg,(i,k), 2, (255,0,0), -1)
        cv2.circle(inputImg,(j,k), 2, (255,0,0), -1) 
    
    pts2 = []
    pts2.append(pts)
    
    cv2.fillPoly(laneShade, np.array(pts2),(100,100,255))
    for i,j,k in zip(xL, xR ,y):
        cv2.circle(laneShade,(i,k), 5, (255,0,0), -1)
        cv2.circle(laneShade,(j,k), 5, (255,0,0), -1)
    inputImg = cv2.resize(inputImg,(0,0),fx =0.5, fy = 0.4)
#    cv2.imshow("cor", laneShade)
    
#    cv2.imshow("cor1", inputImg)
    
    curvature = rad_of_curvature(np.column_stack((xL,y)),np.column_stack((xR,y)))
    
    d1 = direction(xL[0],y[0],xL[int(len(xL)/2)],y[int(len(xL)/2)],xL[-1],y[-1])
    d2 = direction(xR[0],y[0],xR[int(len(xR)/2)],y[int(len(xR)/2)],xR[-1],y[-1])
    dT = (d1+d2)/2
    print(dT)
    if abs(dT)<7000:
        dirTurn = "Straight"
    elif dT>7000:
        dirTurn = "Right Turn"
    else:
        dirTurn = "Left Turn"
        
    return laneShade, True, curvature, dirTurn


def direction(x1,y1,x2,y2,x3,y3):
    val = (y2 - y1)*(x3 - x2) - (y3 - y2)*(x2 - x1)
    return val

def rad_of_curvature(left_line, right_line):

    ploty = left_line[:,1]
    leftx, rightx = left_line[:,0], right_line[:,0]

    leftx = leftx[::-1]
    rightx = rightx[::-1]

    width_lanes = abs(right_line[0,0] - left_line[0,0])
    ym_per_pix = 30 / 720 
    xm_per_pix = 3.7*(720/1280) / width_lanes

    y_eval = np.max(ploty)

    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])
    
    return (left_curverad+right_curverad)/2



#Camera Matrix
K = np.array([[1.15422732e+03, 0.00000000e+00, 6.71627794e+02], 
              [0.00000000e+00,1.14818221e+03,3.86046312e+02],
              [0.00000000e+00,0.00000000e+00,1.00000000e+00]])

#Distortion Coefficients
dist = np.array([[ -2.42565104e-01, -4.77893070e-02, -1.31388084e-03,
                  -8.79107779e-05, 2.20573263e-02]])
    
cap = cv2.VideoCapture('inputVideo.mp4')

#cap.set(cv2.CAP_PROP_POS_FRAMES,350)

while(cap.isOpened()):
    
    _, frame = cap.read()
    
    # Undistorting Image    
    frameCopy = copy.deepcopy(frame)
    undistorted_image = undistort(K,dist,frameCopy)
    
    # Homography  
    pts1 = np.float32([[554,489],[751,487],[961,599],[416,602]])
    pts2 = np.float32([[200,800],[600,800],[600,1400],[200,1400]])

    M = cv2.getPerspectiveTransform(pts1,pts2)
    wf,hf = undistorted_image.shape[:2]
    perspectiveImg = cv2.warpPerspective(undistorted_image,M,(800,1500))     

    
    # Denoise
    blur = cv2.GaussianBlur(perspectiveImg,(3,3),3)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray,(0,0),fx =0.5, fy = 0.4)
    #plt.hist(gray.ravel(),256,[0,256]); plt.show()

    # Color thresholding
    hsl1 = cv2.cvtColor(blur, cv2.COLOR_BGR2HLS)
    lower_yellow = np.array([9, 128, 154])
    upper_yellow = np.array([33, 203, 255])
    
    lower_white = np.array([0, 208, 0])
    upper_white = np.array([255, 255, 255])
    
    maskYellow = cv2.inRange(hsl1, lower_yellow, upper_yellow)
    maskwhite = cv2.inRange(hsl1, lower_white, upper_white)
    totalMask = cv2.add(maskYellow,maskwhite)
    result1 = cv2.bitwise_and(hsl1, hsl1, mask=totalMask)
    result1 = cv2.cvtColor(result1,cv2.COLOR_HLS2BGR)
    result1 = cv2.cvtColor(result1,cv2.COLOR_BGR2GRAY)
    (T, result1_) = cv2.threshold(result1, 190, 255, cv2.THRESH_BINARY)
    
    
    laneShaded, success , curve, turn = lanes(result1_,10)
    if success:
        shadePersp = cv2.warpPerspective(laneShaded,np.linalg.inv(M),(undistorted_image.shape[1],undistorted_image.shape[0]))
        detectedCurvature = curve
        dirTurn = turn
    detected = cv2.addWeighted(undistorted_image,1,shadePersp,0.7,0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "Curvature= "+str(round(detectedCurvature,3))
    cv2.putText(detected,text,(int(detected.shape[1]/2),int(detected.shape[0]*0.9)), font, 0.5,(150,30,30),1,cv2.LINE_AA)
    cv2.putText(detected,dirTurn,(int(detected.shape[1]/2),int(detected.shape[0]*0.9)+13), font, 0.5,(150,30,30),1,cv2.LINE_AA)
    
    # Display Images with Overlay
    cv2.imshow("Lane Detection",detected)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    #print(cap.get(cv2.CAP_PROP_POS_FRAMES),cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT)-20:
        break

cap.release()
cv2.destroyAllWindows()