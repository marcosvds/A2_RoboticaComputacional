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
#Parte_3-----------------------------------
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
#-----------------------------------------
#Parte_4----------------------------------                
            else:
                dist = str(round(dist_calc,3))
                cv2.putText(final,'Distance = '+ dist + "cm",(10,100), font, 1,(0,255,0),2,cv2.LINE_AA)
                Ang= math.degrees(math.atan2((l_y[0] - l_y[1]), (l_x[0] - l_x[1])))
                Ang_final = str(round(abs(Ang),3))
                cv2.putText(final, "Degrees = " + Ang_final, (10,150), font, 1,(0,255,0),2,cv2.LINE_AA)
#------------------------------------------
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

while(True):
    ret, frame = cap.read()
#Parte_2--------------------------------------------
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    mask_b = cv2.inRange(img_hsv, hsv1_b, hsv2_b)
    mask_p = cv2.inRange(img_hsv, hsv1_p, hsv2_p) 
    maskf = mask_b + mask_p
    not_mask = cv2.bitwise_not(maskf)
    res = cv2.bitwise_and(img_gray,img_gray, mask = not_mask)
    res2 = cv2.bitwise_and(frame,frame, mask= maskf)
    img_mask = cv2.cvtColor(res, cv2.COLOR_GRAY2RGB)
    final = cv2.bitwise_or(res2, img_mask)
#---------------------------------------------------
# Parte_5-------------------------------------------
    cv2.putText(final,'Press Q to quit',(10,50), font, 1,(0,255,0),2,cv2.LINE_AA)
    
    blur = cv2.GaussianBlur(img_gray,(5,5),0)
    bordas = auto_canny(blur)
    
    circles = []
    circles = None
    circles=cv2.HoughCircles(bordas,cv2.HOUGH_GRADIENT,2,40,param1=70,param2=100,minRadius=5,maxRadius=70)

    if circles is not None:   
        circles = np.uint16(np.around(circles))
        lista_raio = []
        lista_posix = []
        lista_posiy = []
        for i in circles[0,:]:
            if maskf[i[1]][i[0]] == 255:
                lista_raio.append(int(i[2]))
                lista_posix.append(int(i[0]))
                lista_posiy.append(int(i[1]))
                cv2.circle(final,(i[0],i[1]),i[2],(0,255,0),2)
                cv2.circle(final,(i[0],i[1]),2,(0,255,0),3)
        if len(lista_raio) >= 2:
            calc_dist(final,lista_posix,lista_posiy)
    else:
        cv2.putText(final,'Distance = Searching...',(10,100), font, 1,(0,255,0),2,cv2.LINE_AA)
        cv2.putText(final, "Degrees = Searching...", (10,150), font, 1,(0,255,0),2,cv2.LINE_AA)
#----------------------------------------------------
#Parte_6---------------------------------------------
    MIN_MATCH_COUNT = 10
    original_bgr = cv2.imread('folha_atividade_insper.png')
    img_original = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
    img_cena = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    brisk = cv2.BRISK_create()
    kp1, des1 = brisk.detectAndCompute(img_original ,None)
    kp2, des2 = brisk.detectAndCompute(img_cena,None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1,des2,k=2)
    
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    
    if len(good)>MIN_MATCH_COUNT:
        cv2.putText(final,'Matches found',(10,200), font, 1,(0,255,0),2,cv2.LINE_AA)    
        find_homography_draw_box(kp1, kp2, final)
    else:
        val_1 = str(len(good))
        val_2 = str(MIN_MATCH_COUNT)
        cv2.putText(final,'Not enough matches are found ' + val_1 + '/' + val_2,(10,200), font, 1,(0,255,0),2,cv2.LINE_AA)      
#----------------------------------------------------
    cv2.imshow('frame', final)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()











