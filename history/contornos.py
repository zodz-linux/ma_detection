import matplotlib.pyplot as plt
import numpy as np
import cv2


im = cv2.imread('lesion.png')
tmp = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

for t in range(45,76,5):
    tmp2=tmp.copy()
    f,c=tmp2.shape
    img=np.zeros((f,c*2),dtype=np.int)
    imgray=tmp.copy()
    imgray[imgray<t]=255
    imgray[imgray<255]=0
    thresh=imgray.copy()
    img[0:f,0:c]=thresh.copy()
    cv2.imwrite("lesion_binaria_"+str(t)+".png",thresh)
    _,contours, hierarchy  = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(tmp2, contours, -1, (0,255,0), 3)
    cv2.imwrite("contornos"+str(t)+".png",tmp2)
    img[0:f,c:c*2]=tmp2.copy()

    cv2.imwrite("comparacion"+str(t)+".png",img)

    cnt = contours[0]
    area = cv2.contourArea(cnt)
    print "area: ",area
    perimeter = cv2.arcLength(cnt,True)
    print "perimetro",perimeter








#for i in contours:
#    print type(i),len(i)
#print contours[-1]
#print type(contours)
#cnt = contours[2]
#cv2.drawContours(tmp, [cnt], 0, (0,255,0), 3)
