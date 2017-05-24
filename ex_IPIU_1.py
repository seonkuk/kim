#-*- coding: utf-8 -*-
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import cv2
from PIL import Image
from numpy import *
from webcam import Webcam
import camera
from Objectcut import Objectcut
import math

def my_calibration(sz):
    row,col=sz
    fx=983
    fy=983
    K=diag([fx,fy,1])
    K[0,2]=331
    K[1,2]=232
    return K


############################################################### H-matrix 찾기
def match_images(img1, img2):
    """Given two images, returns the matches"""
    detector = cv2.SURF(400, 5, 5)
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    kp1, desc1 = detector.detectAndCompute(img1, None)
    kp2, desc2 = detector.detectAndCompute(img2, None)

    raw_matches = matcher.knnMatch(desc1,desc2, k=2)  # 2
    matches=[]

    matA, matB=[],[]
    for m in raw_matches:
        if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
            matA.append(kp1[m[0].queryIdx])
            matB.append(kp2[m[0].trainIdx])

    
    if len(matA) > 50:
        ptsA = float32([m.pt for m in matA])
        ptsB = float32([n.pt for n in matB])
        H=[]
        H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,3.0)
        
        return H
    else:
        return None

class OpenGLGlyphs:
  

##############################################################초기화
    def __init__(self):
        # initialise webcam and start thread
        self.webcam = Webcam()
        self.webcam.start()
        self.wid=640
        self.hei=480
        # initialise cube
        self.d_obj = None
        self.setting=0
        # initialise texture
        self.texture_background = None
        self.cut_im=None
        self.marker_im=None
        self.K=None
        #self.Rt=None
        self.wait=0
        self.cut=None
        self.x1=self.wid/4
        self.y1=self.hei/4
        self.x2=self.wid/4*3
        self.y2=self.hei/4*3
        self.trans=ones((3,3))
        self.homo=eye(3)
##############################################################카메라 세팅
    def _init_gl(self, Width, Height):
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClearDepth(1.0)
        glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)
        glShadeModel(GL_SMOOTH)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        
        self.K=my_calibration((Height,Width))
        fx = self.K[0,0]
        fy = self.K[1,1]
        fovy = 2*arctan(0.5*Height/fy)*180/pi
        aspect = (float)(Width*fy)/(Height*fx)
        # define the near and far clipping planes
        near = 0.1
        far = 100.0
        # set perspective
        gluPerspective(fovy,aspect,near,far)
        
        glMatrixMode(GL_MODELVIEW)
        #self.d_obj=[OBJ('Rocket.obj')]
        glEnable(GL_TEXTURE_2D)
        self.texture_background = glGenTextures(1)
        #gluPerspective(33.7, 1.3, 0.1, 100.0)
        

##############################################################배경화면 설정
    def _draw_scene(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
 
        # get image from webcam
        image = self.webcam.get_current_frame()
        back_im=cv2.imread("wait.jpg")
        
        # convert image to OpenGL texture format
        cv2.rectangle(image,(self.x1,self.y1),(self.x2,self.y2),(255,0,0),3)


        if self.setting==1:
            bg_image = cv2.flip(back_im, 0)
        else:
            bg_image = cv2.flip(image, 0)
                  

        
        bg_image = Image.fromarray(bg_image)
        
        ix = bg_image.size[0]
        iy = bg_image.size[1]
        bg_image = bg_image.tostring("raw", "BGRX", 0, -1)
  
        # create background texture
        glBindTexture(GL_TEXTURE_2D, self.texture_background)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,self.wid, self.hei, 0, GL_RGBA, GL_UNSIGNED_BYTE, bg_image)
         
        # draw background
        glBindTexture(GL_TEXTURE_2D, self.texture_background)
        glPushMatrix()
        #glTranslatef(0.0,0.0,0.0)
        gluLookAt (0.0, 0.0, 14.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        self._draw_background()
        glPopMatrix()
        box=zeros((self.hei,self.wid),uint8)
        
            
        if self.setting==2: ################Rt를 구해서 매칭되는 이미지가 있는지 판단
            Rt=self._my_cal(self.cut_im,image)
            if Rt!=None:
                self._set_modelview_from_camera(Rt)
                glEnable(GL_LIGHTING)
                glEnable(GL_LIGHT0)
                glEnable(GL_DEPTH_TEST)
                glEnable(GL_NORMALIZE)
                glClear(GL_DEPTH_BUFFER_BIT)
                glMaterialfv(GL_FRONT, GL_AMBIENT, [0.25,0.25,0.0,1.0])
                glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.9,0.9,0.0,1.0])
                glMaterialfv(GL_FRONT, GL_SPECULAR, [1.0,1.0,1.0,1.0])
                glMaterialfv(GL_FRONT, GL_SHININESS, 0.25*128.0)
                glutSolidTeapot(0.05)



                X=array([self.x1,self.y1,1])
                Y=array([self.x2,self.y2,1])
                X=dot(self.trans,X.T)
                Y=dot(self.trans,Y.T)

                X=X/X[-1]
                Y=Y/Y[-1]
                      
   
                if int(X[0])<0:
                      self.x1=0
                elif int(X[0])>self.wid:
                      self.x1=self.wid
                else:
                      self.x1=int(X[0])      
                if int(X[1])<0:
                      self.y1=0
                elif int(X[1])>self.hei:
                      self.y1=self.hei
                else:
                      self.y1=int(X[1])
                        
                if int(Y[0])<0:
                      self.x2=0
                elif int(Y[0])>self.wid:
                      self.x2=self.wid
                else:
                      self.x2=int(Y[0])
                        
                if int(Y[1])<0:
                      self.y2=0
                elif int(Y[1])>self.hei:
                      self.y2=self.hei
                else:
                      self.y2=int(Y[1])
                box[self.y1:self.y2,self.x1:self.x2]=1
                self.cut_im=image*box[:,:,newaxis]
                
        elif self.setting==1:
            box[self.y1:self.y2,self.x1:self.x2]=1
            self.cut_im=image*box[:,:,newaxis]
            self.cut_im=image
            self.marker_im=image
            self.setting=2

                
        glutSwapBuffers()

        
##############################################################OpenGL용 Rt변환
    def _set_modelview_from_camera(self,Rt):

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        Rx = array([[1,0,0],[0,0,1],[0,1,0]])
    
        # set rotation to best approximation
        R = Rt[:,:3]

        #U,S,V = linalg.svd(R)
        #R = dot(U,V)
    
        # change sign of x-axis
        R[0,:] = -R[0,:]
        # set translation
        t = Rt[:,3]
        t[0]=-t[0]

        # setup 4*4 model view matrix
        M = eye(4)
        M[:3,:3] = dot(R,Rx)
        M[:3,3] = t
        M[3,:3] = t
        
        # transpose and flatten to get column order
        M = M.T
    
        m = M.flatten()
        # replace model view with the new matrix
        glLoadMatrixf(m)
    
##############################################################Rt반환    
    def _my_cal(self,im1,iamge):
        
        img1=im1
        img2=iamge
        
        H=match_images(img1,img2)
        #H1=match_images(self.marker_im,img2)
        if H!=None:
            
            if cv2.norm(H,self.trans) > 1.0:
                
                self.trans=H
                self.homo=dot(H,self.homo)
            else:
                self.trans=eye(3)
                self.homo=dot(self.trans,self.homo)

                
            cam1 = camera.Camera( hstack((self.K,dot(self.K,array([[0],[0],[-1]])) )) )
            cam2 = camera.Camera(dot(self.homo,cam1.P))
            A = dot(linalg.inv(self.K),cam2.P[:,:3])
            A = array([A[:,0],A[:,1],cross(A[:,0],A[:,1])]).T
            cam2.P[:,:3] = dot(self.K,A)
            Rt=dot(linalg.inv(self.K),cam2.P)
        
            return Rt
        else:
            return None



    def _draw_background(self):
        # draw background
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 1.0); glVertex3f(-4.0, -3.0, 0.0)
        glTexCoord2f(1.0, 1.0); glVertex3f( 4.0, -3.0, 0.0)
        glTexCoord2f(1.0, 0.0); glVertex3f( 4.0,  3.0, 0.0)
        glTexCoord2f(0.0, 0.0); glVertex3f(-4.0,  3.0, 0.0)
        glEnd( )
        glDeleteTextures(1)
        
    def keyboard(self,*args):
        if args[0]==GLUT_KEY_UP:
            glutDestroyWindow(self.window_id)
            sys.exit()
        elif args[0]==GLUT_KEY_DOWN:
            self.setting=1
        elif args[0]==GLUT_KEY_RIGHT:
            self.setting=0

##############################################################OpenGL창 초기
 
    def main(self):
        # setup and run OpenGL
        glutInit()
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(self.wid, self.hei)
        glutInitWindowPosition(400, 400)
        self.window_id = glutCreateWindow("OpenGL Glyphs")
        self._init_gl(self.wid, self.hei)
        glutDisplayFunc(self._draw_scene)
        glutIdleFunc(self._draw_scene)
        glutSpecialFunc(self.keyboard)
        glutMainLoop()
  
# run an instance of OpenGL Glyphs 
openGLGlyphs = OpenGLGlyphs()
openGLGlyphs.main()
