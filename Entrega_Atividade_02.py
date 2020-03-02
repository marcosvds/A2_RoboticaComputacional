# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 21:11:56 2020

@author: Enrico Damiani e Marcos Vinícius
"""
import cv2
import numpy as np
import sys
import math
import auxiliar as aux
if (sys.version_info > (3, 0)): 
    # Modo Python 3
    import importlib
    importlib.reload(aux) # Para garantir que o Jupyter sempre relê seu trabalho
else:
    # Modo Python 2
    reload(aux)

def calc_dist(frame,l_x,l_y):
            cv2.line(frame,(l_x[0],l_y[0]),(l_x[1],l_y[1]),(0,255,0),3)
            dist_real = 13.9
            dist_pix30 = 400
            dist_obs = 30
            coef_obs = (dist_pix30*dist_obs)/(dist_real)
            coef_final = (13.9*coef_obs)
            dist_pix = ((((l_y[0] - l_y[1])*(l_y[0] - l_y[1])) + ((l_x[0] - l_x[1])*(l_x[0] - l_x[1])))**(1/2))
            dist_calc = coef_final/dist_pix
            if math.isnan(dist_calc):  
                cv2.putText(final,'Distance = Searching...',(10,100), font, 1,(0,255,0),2,cv2.LINE_AA)
                cv2.putText(final, "Degrees = Searching...", (10,150), font, 1,(0,255,0),2,cv2.LINE_AA)
            else:
                dist = str(round(dist_calc,3))
                cv2.putText(final,'Distance = '+ dist + "cm",(10,100), font, 1,(0,255,0),2,cv2.LINE_AA)
                Ang= math.degrees(math.atan2((l_y[0] - l_y[1]), (l_x[0] - l_x[1])))
                Ang_final = str(round(abs(Ang),3))
                cv2.putText(final, "Degrees = " + Ang_final, (10,150), font, 1,(0,255,0),2,cv2.LINE_AA)

def auto_canny(image, sigma=0.33):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged

def find_homography_draw_box(kp1, kp2, img_cena):
    out = img_cena
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist() 
    h,w = img_original.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    img2b = cv2.polylines(out,[np.int32(dst)],True,(0,255,0),5, cv2.LINE_AA)
    return img2b   

hsv1_b = np.array([97,50,50], dtype=np.uint8)
hsv2_b = np.array([145, 255, 255], dtype=np.uint8)
hsv1_p = np.array([145,  50,  50], dtype=np.uint8)
hsv2_p = np.array([179, 255, 255], dtype=np.uint8)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

font = cv2.FONT_HERSHEY_SIMPLEX