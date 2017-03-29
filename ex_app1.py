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


def my_calibration(sz):
    row,col=sz
    fx=2782*col/3264
    fy=2840*row/2448
    K=diag([fx,fy,1])
    K[0,2]=0.5*col
    K[1,2]=0.5*row
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
    """
    kp1 = float32([kp.pt for kp in kp1])
    kp2 = float32([kp.pt for kp in kp2])
    """
    matA, matB=[],[]
    for m in raw_matches:
        if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
            matA.append(kp1[m[0].queryIdx])
            matB.append(kp2[m[0].trainIdx])

    
    if len(matA) > 50:
        ptsA = float32([m.pt for m in matA])
        ptsB = float32([n.pt for n in matB])
        H=[]
        H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,5.0)
        
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
        self.K=None
        #self.Rt=None
        self.wait=0
        self.cut=None
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
        

##############################################################K값 구하기
    def _draw_scene(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
 
        # get image from webcam
        image = self.webcam.get_current_frame()
        back_im=cv2.imread("wait.jpg")
        
        # convert image to OpenGL texture format
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
        gluLookAt (0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        self._draw_background()
        glPopMatrix()
        
        
            
        if self.setting==2:
            ################Rt를 구해서 매칭되는 이미지가 있는지 판단
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
        elif self.setting==1:
            if self.wait==0:
                self.cut=Objectcut(image)
                self.cut.start()
                self.wait=1
            if self.wait==1:
                if self.cut.confirm()[0]==2:
                    self.cut_im=self.cut.confirm()[1]
                    self.setting =2
                    self.wait=0
                elif self.cut.confirm()[0]==0:
                    self.setting =0
                    self.wait=0
                else:
                    self.setting =1
                
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
        if H!=None:
            cam1 = camera.Camera( hstack((self.K,dot(self.K,array([[0],[0],[-1]])) )) )
            #Rt1=dot(linalg.inv(self.K),cam1.P)
            cam2 = camera.Camera(dot(H,cam1.P))

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
