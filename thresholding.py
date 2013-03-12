from pylab import *

import sys
import Image


def mle(x,mu,sig):
    ndim = mu.shape[0]
    assert sig.shape[0] == ndim
    assert sig.shape[1] == ndim
    
    nf = (2 * pi) ** (-ndim/2.0) * det(sig) ** -0.5
    xmu = x - mu
    isi = pinv(sig)

    return nf * exp(-0.5 * dot(dot(xmu, isi), xmu.T))

class WingImage():

    def __init__(self, filename):
        image_file = Image.open(filename).convert('RGB')
        self.image = array(image_file, dtype=np.float)
        self.image += 2.0 * random(self.image.shape) - 1.0
        self.shape = self.image.shape[:-1]
        
        self.pcn = ones(r_[self.shape, [3]]) / 3.0
        self.pc = ones(3) / 3.0

        self.mu = outer([0.0,196.0,256.0],[1.0,1.0,1.0])
        self.sig = (16.0**2) * array(3*[identity(3)])

    def e_step(self):
        for j in range(self.shape[0]):
            for k in range(self.shape[1]):
                for c in range(3):
                    self.pcn[j,k,c] = mle(self.image[j,k], self.mu[c], self.sig[c])
                self.pcn[j,k,:] = self.pcn[j,k,:] / self.pcn[j,k,:].sum()


    def m_step(self):
        for c in range(3):
            self.mu = zeros((3,3))
            self.sig = zeros((3,3,3))
        denom = zeros(3)
        for j in range(self.shape[0]):
            for k in range(self.shape[1]):
                cp = outer(self.image[j,k],self.image[j,k])
                denom += self.pcn[j,k]
                for c in range(3):
                    self.mu[c] += self.pcn[j,k,c] * self.image[j,k]
                    self.sig[c] += self.pcn[j,k,c] * cp
        print self.mu
        print self.sig
        print 70*'-'

        for c in range(3):
            self.mu[c] /= denom[c]
            self.sig[c] /= denom[c]
            self.sig[c] -= outer(self.mu[c], self.mu[c])
            



if __name__ == '__main__':
    
    wing = WingImage(sys.argv[1])
    wing.image

    ion()
    figure(1)
    imshow(wing.image / 256.0)


    wing.e_step()

    figure(2)
    imshow(copy(wing.pcn))

    N_em_iter = 3
    for kk in range(N_em_iter):
        wing.m_step()
        wing.e_step()

    figure(3)
    imshow(wing.pcn)

    figure()
    imshow(wing.pcn[:,:,0] / wing.pcn[:,:,0].max())
    figure()
    imshow(wing.pcn[:,:,1] / wing.pcn[:,:,1].max())
    figure()
    imshow(wing.pcn[:,:,2] / wing.pcn[:,:,2].max())
