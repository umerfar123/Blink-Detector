import cv2
import mediapipe as mp
import time
import faceMesh as fm
from scipy.spatial.distance import euclidean

ctime,ptime=0,0

#Video Source
cap = cv2.VideoCapture('demo\o4.mp4')
faceMesh = fm.faceContour(mode=False)

while True:
    
    st,frame=cap.read()
    frame = cv2.resize(frame,(720,400))
    
    #facemesh plotting
    frame = faceMesh.meshDetector(frame,Draw=False)
    #facemesh points
    lmList = faceMesh.meshPoints(frame)
    
    if len(lmList)!=0:
        
        #Eye points landmarks
        lmpoints_Reye = [385,387,359,373,380,398]
        lmpoints_Leye = [160,158,173,153,144,7]
        
        #plotting the eyepoints
        frame = faceMesh.eyePlotter(frame,lmList,lmpoints_Leye)
        frame = faceMesh.eyePlotter(frame,lmList,lmpoints_Reye)
        
        #Ear calculation
        p2_p6 = euclidean(lmList[160][1:],lmList[144][1:])
        p3_p5 = euclidean(lmList[158][1:],lmList[153][1:])
        p1_p4 = euclidean(lmList[7][1:],lmList[173][1:])
        
        ear = (p2_p6 + p3_p5) / (2.0 * p1_p4)
        
        #blink detection using ear
        if ear < 0.28:
            print("Blinked")
            cv2.putText(frame,text='Blinked',org=(100,300),fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale=2,thickness=3,color=(0,255,0))
        else:
            print('0')
     
    #fps calculation   
    ctime= time.time()
    fps = 1/(ctime-ptime)
    ptime=ctime
    cv2.putText(frame,text=str(int(fps)),color=(0,255,0),org=(0,30),fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=3,thickness=1)
    
    #output
    cv2.imshow("Face Mesh",frame)
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break
    

cv2.destroyAllWindows()
cap.release()