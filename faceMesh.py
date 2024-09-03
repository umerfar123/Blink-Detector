import mediapipe as mp
import cv2

class faceContour():
    
    def __init__(self,mode=False, maxFace=2, detConf=0.5,trackConf=0.5):
        
        self.mode = mode
        self.maxFace = maxFace
        self.detConf = detConf
        self.trackConf = trackConf
        
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFacemesh =  mp.solutions.face_mesh
        self.faceMesh = self.mpFacemesh.FaceMesh(max_num_faces=2)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)
        
    def meshDetector(self,frame,Draw=True):
        
        frameRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(frameRGB)
        
        if self.results.multi_face_landmarks:
            for self.faceLM in self.results.multi_face_landmarks:
                if Draw:
                    self.mpDraw.draw_landmarks(frame,self.faceLM, self.mpFacemesh.FACEMESH_CONTOURS,self.drawSpec,self.drawSpec)
         
        return frame  
    
    def meshPoints(self,frame):
        
        lmList=[]
        for id,lm in enumerate(self.faceLM.landmark):
            ih ,iw, c = frame.shape
            
            x,y = int(lm.x*iw), int(lm.y*ih)
            lmList.append([id,x,y])
            
        return lmList
        
    def eyePlotter(self,frame,lmlist,lmpoints):
        
        x1,y1 = lmlist[lmpoints[0]][1],lmlist[lmpoints[0]][2] # red
        x2,y2 = lmlist[lmpoints[1]][1],lmlist[lmpoints[1]][2] # green
        x3,y3 = lmlist[lmpoints[2]][1],lmlist[lmpoints[2]][2] # blue
        x4,y4 = lmlist[lmpoints[3]][1],lmlist[lmpoints[3]][2] # yellow
        x5,y5 = lmlist[lmpoints[4]][1],lmlist[lmpoints[4]][2] # pink
        x6,y6 = lmlist[lmpoints[5]][1],lmlist[lmpoints[5]][2] # Black
        

        cv2.circle(frame,center=(x1,y1),color=(0,0,255),radius=2,thickness=1)
        cv2.circle(frame,center=(x2,y2),color=(0,255,0),radius=2,thickness=1)
        cv2.circle(frame,center=(x3,y3),color=(255,0,0),radius=2,thickness=1)
        cv2.circle(frame,center=(x4,y4),color=(0,255,255),radius=2,thickness=1)
        cv2.circle(frame,center=(x5,y5),color=(255,0,255),radius=2,thickness=1)
        cv2.circle(frame,center=(x6,y6),color=(0,0,0),radius=2,thickness=1)
        
        return frame
          
    
    
    
    
    
def main():
    pass

if __name__ == '__main__':
    main()
