from PIL import Image
from scipy import linalg
from numpy import *


def rotation_matrix(a):
    from scipy.linalg import expm
    R = eye(4)
    R[:3,:3] = expm([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])
    return R

class Camera(object):
    """ Class for representing pin-hole cameras. """
    def __init__(self,P):
	""" Initialize P = K[R|t] camera model. """
	self.P = P
	self.K = None # calibration matrix
	self.R = None # rotation
	self.t = None # translation
	self.c = None # camera center
	
    def project(self,X):
	""" Project points in X (4*n array) and normalize coordinates. """
	x = dot(self.P,X)
	for i in range(3):
		x[i] /= x[2]
	return x

    def factor(self):
        # factor first 3*3 part
        from scipy.linalg import rq
        K,R = rq(self.P[:,:3])
        # make diagonal of K positive
        T = diag(sign(diag(K)))
        
        if linalg.det(T) < 0:
            T[1,1] *= -1
            
        self.K = dot(K,T)
        self.R = dot(T,R) # T is its own inverse
        self.t = dot(linalg.inv(self.K),self.P[:,3])
        
        return self.K, self.R, self.t
    def center(self):
        """ Compute and return the camera center. """
        if self.c is not None:
            return self.c
        else:
            # compute c by factoring
            self.factor()
            self.c = -dot(self.R.T,self.t)
            return self.c

def my_calibration(sz):
    row,col=sz
    fx=3913.0*col/4128
    fy=3920.0*row/2322
    K=diag([fx,fy,1])
    K[0,2]=0.5*col
    K[1,2]=0.5*row
    return K
