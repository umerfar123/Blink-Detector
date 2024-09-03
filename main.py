import cv2
import mediapipe as mp
import time
import faceMesh as fm
from scipy.spatial.distance import euclidean

ctime,ptime=0,0
face = cv2.imread('demo\mukam.jpg')

cap = cv2.VideoCapture('demo\o1.mp4')
faceMesh = fm.faceContour(mode=False)

while True:
    
    st,frame=cap.read()
    frame = cv2.resize(frame,(720,400))
    
    frame = faceMesh.meshDetector(frame,Draw=False)
    lmList = faceMesh.meshPoints(frame)
    
    if len(lmList)!=0:
        
        
        lmpoints_Reye = [385,387,359,373,380,398]
        lmpoints_Leye = [160,158,173,153,144,7]
        
        frame = faceMesh.eyePlotter(frame,lmList,lmpoints_Leye)
        frame = faceMesh.eyePlotter(frame,lmList,lmpoints_Reye)
        
        p2_p6 = euclidean(lmList[160][1:],lmList[144][1:])
        p3_p5 = euclidean(lmList[158][1:],lmList[153][1:])
        p1_p4 = euclidean(lmList[7][1:],lmList[173][1:])
        
        ear = (p2_p6 + p3_p5) / (2.0 * p1_p4)
        if ear < 0.2:
            print("Blinked")
        else:
            print('0')
        
        
        
        
        
    
    ctime= time.time()
    fps = 1/(ctime-ptime)
    ptime=ctime
    cv2.putText(frame,text=str(int(fps)),color=(0,255,0),org=(0,30),fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=3,thickness=1)
    
    cv2.imshow("Face Mesh",frame)
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break
    

cv2.destroyAllWindows()
cap.release()