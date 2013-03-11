from pylab import *

import sys
import Image


def mle(x,mu,sig):
    ndim = mu.shape[0]
    assert sig.shape[0] == ndim
    assert sig.shape[1] == ndim
    
    nf = (2 * pi) ** (-ndim/2.0) * det(sig) ** -0.5
    xmu = x - mu
    isi = inv(sig)

    return nf * exp(-0.5 * dot(dot(xmu, isi), xmu.T))

class WingImage():

    def __init__(self, filename):
        self.image = array(Image.open(filename), dtype=np.float)
        self.shape = self.image.shape[:-1]
        
        self.pcn = ones(r_[self.shape, [3]]) / 3.0
        self.pc = ones(3) / 3.0

        self.mu = outer([0.0,128.0,256.0],[1.0,1.0,1.0])
        self.sig = (32.0**2) * array(3*[identity(3)])

    def e_step(self):
        for j in range(self.shape[0]):
            for k in range(self.shape[1]):
                for c in range(3):
                    self.pcn[j,k,c] = multinomial_likelihood(self.image[j,k],
                                                           self.mu[c], self.sig[c])
                self.pcn[j,k,:] = self.pcn[j,k,:] / self.pcn[j,k,:].sum()


    def m_step(self):
        for c in range(3):
            newmu = zeros(3)
xxx            newpc = 0.0

            denom = 0.0
            for j in range(self.shape[0]):
                for k in range(self.shape[1]):
                    newmu[c] += self.pc[j,k,c] * self.image[j,k]
                    newpc += 
                    denom += self.pc[j,k,c]
            newmu[c] /= denom



if __name__ == '__main__':
    
    wing = WingImage(sys.argv[1])
    wing.image

    ion()
    figure(1)
    imshow(wing.image)

    figure(2)
    imshow(wing.pcn)

    wing.e_step()
    wing.m_step()

    wing.e_step()
    wing.m_step()

    wing.e_step()
    wing.m_step()

    wing.e_step()
    wing.m_step()

    figure(3)
    imshow(wing.pcn)

    figure()
    imshow(wing.pcn[:,:,0] / wing.pcn[:,:,0].max())
    figure()
    imshow(wing.pcn[:,:,1] / wing.pcn[:,:,1].max())
    figure()
    imshow(wing.pcn[:,:,2] / wing.pcn[:,:,2].max())
