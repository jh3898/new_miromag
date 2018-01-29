import numpy as np
import scipy.sparse as sparse

class OperatorFiniteDiff1DPeriodic(object):
    def __init__(self, shape, lengths= None):
        if lengths is None:
            Lx=1.
        else:
            Lx= float(lengths[0])
        nx= int(shape[0])
        self.nx= nx
        self.size =nx
        self.shape= [nx]
        self.Lx= Lx
        self.dx = Lx/nx
        dx= self.dx

        self.xs = np.linspace(0,Lx,nx)

        self.sparse_px= sparse.diags(
            diagonals= [-np.ones(nx-1),np.ones(nx-1),-1,1],
            offsets= [-1,1,nx-1,(-nx+1)])
        self.sparse_px = self.sparse_px/(2*dx)

        self.sparse_pxx = sparse.diags(diagonals= [np.ones(nx-1),-2*np.ones(nx),np.ones(nx-1),1,1],
                                       offsets= [-1,0,1,nx-1,-(nx-1)])
    def px(self,a):
            return self.sparse_px.dot(a.flat)

    def pxx(self,a):
            return self.sparse_pxx.dot(a.flat)

    def identity(self):
            return sparse.identity(self.size)

if __name__ == '__main__':
    nx=4
    oper = OperatorFiniteDiff1DPeriodic([nx],[nx/2.])
    a= np.arange(nx)












