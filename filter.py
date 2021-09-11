import cv2
import numpy as np
glass = cv2.imread('glass.png',-1)
must = cv2.imread('mustache.png',-1)
cigar = cv2.imread('cigar.png',-1)

def transparentOverlay(src,overlay,pos = (0,0),scale = 1):
    overlay = cv2.resize(overlay,(0,0),fx = scale,fy = scale)
    h,w,_ = overlay.shape
    row,col,_ = src.shape
    y,x = pos[0],pos[1]

    for i in range(h):
        for j in range(w):
            if x+i>=row or y+j>=col:
                continue
            alpha = float(overlay[i][j][3]/255.0)
            src[x+i][y+j] = alpha*overlay[i][j][:3]+(1-alpha)*src[x+i][y+j]
    return src

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
while True:
    ret,frame = cap.read()
    if ret == False:
        continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        if w>0 and h>0:
            glass_symin = int(y+1.5*h/5)
            glass_symax = int(y+2.5*h/5)
            sh_glass = glass_symax-glass_symin

            cigar_symin = int(y+4*h/6)
            cigar_symax = int(y+5.5*h/6)
            sh_cigar = cigar_symax-cigar_symin

            mus_symin = int(y+3.5*h/6)
            mus_symax = int(y+5*h/6)
            sh_mus = mus_symax-mus_symin

            glass_roi = frame[glass_symin:glass_symax,x:x+w]
            cigar_roi = frame[cigar_symin:cigar_symax,x:x+w]
            mus_roi = frame[mus_symin:mus_symax,x:x+w]
            spec= cv2.resize(glass, (w,sh_glass),interpolation=cv2.INTER_CUBIC)
            musta = cv2.resize(must,(w,sh_mus),interpolation=cv2.INTER_CUBIC)
            ciga = cv2.resize(cigar,(w,sh_cigar),interpolation=cv2.INTER_CUBIC)

            transparentOverlay(glass_roi,spec)
            #transparentOverlay(mus_roi, musta)
            transparentOverlay(cigar_roi, ciga,(int(w/2),int(sh_cigar/2)))
    cv2.imshow('Glasses',frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()