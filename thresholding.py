from pylab import *

import sys
import Image


def mle(x,mu,sig):
    nf = (2 * pi) ** (-1/2.0) * sig
    xmu = x - mu

    return nf * exp(-0.5 * (xmu / sig) ** 2)

class WingImage():

    def __init__(self, filename):
        image_file = Image.open(filename).convert('L')
        self.image = array(image_file, dtype=np.float)
        self.image += 2.0 * random(self.image.shape) - 1.0
        self.shape = self.image.shape[0:2]
        
        self.pcn = ones(r_[self.shape, [3]]) / 3.0
        self.pc = ones(3) / 3.0

        self.mu = array([0.0, 220.0, 245.0])
        self.sig = 6.0 * ones(3)

    def e_step(self):
        for j in range(self.shape[0]):
            for k in range(self.shape[1]):
                for c in range(3):
                    self.pcn[j,k,c] = self.pc[c] * mle(self.image[j,k], self.mu[c], self.sig[c])
                self.pcn[j,k,:] = self.pcn[j,k,:] / self.pcn[j,k,:].sum()


    def m_step(self):
        for c in range(3):
            self.mu = zeros(3)
            self.sig = zeros(3)
        denom = zeros(3)
        for j in range(self.shape[0]):
            for k in range(self.shape[1]):
                cp = self.image[j,k] * self.image[j,k]
                denom += self.pcn[j,k]
                for c in range(3):
                    self.mu[c] += self.pcn[j,k,c] * self.image[j,k]
                    self.sig[c] += self.pcn[j,k,c] * cp

        for c in range(3):
            self.mu[c] /= denom[c]
            self.sig[c] /= denom[c]
            self.sig[c] = sqrt(self.sig[c] - self.mu[c] * self.mu[c])

        self.pc = mean(self.pcn.reshape(-1,3),0)

        print self.mu
        print self.sig
        print self.pc
        print 70*'-'



if __name__ == '__main__':

    wing = WingImage(sys.argv[-1])
    wing.image

    ion()
    figure(1)
    imshow(wing.image / 256.0)

    wing.e_step()

    figure(2)
    imshow(copy(wing.pcn))

    N_em_iter = 10
    for kk in range(N_em_iter):
        wing.m_step()
        wing.e_step()

    figure(3)
    imshow(wing.pcn)

    figure()
    imshow(wing.pcn[:,:,0] / wing.pcn[:,:,0].max(), cmap=cm.bone)
    figure()
    imshow(wing.pcn[:,:,1] / wing.pcn[:,:,1].max(), cmap=cm.bone)
    figure()
    imshow(wing.pcn[:,:,2] / wing.pcn[:,:,2].max(), cmap=cm.bone)
