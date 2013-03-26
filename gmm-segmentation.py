from pylab import *

import sys
import Image

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

    def mixture_components(self, x):
        out = zeros(3)
        for c in range(3):
            nf = ((2 * pi) ** (1/2.0) * self.sig[c]) ** -1
            xn = (x - self.mu[c])/self.sig[c]
            out[c] = self.pc[c] * nf * exp(-0.5 * (xn) ** 2)
        return out


    def e_step(self):
        for j in range(self.shape[0]):
            for k in range(self.shape[1]):
                for c in range(3):
                    self.pcn[j,k,:] = self.mixture_components(self.image[j,k])
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



    figure()
    dx = 4.0
    xh = mgrid[:256:dx]
    hh = hist(wing.image.ravel(), xh)[0]
    hh = (hh + 0.0) / hh.sum() / dx

    xx = mgrid[0.0:256.1:0.1]
    comp = array([wing.mixture_components(x) for x in xx])

    figure()
    ## plot(xh[1:], hh, '-o')
    # plot(c_[xh, xh].ravel()[:-1], r_[0, c_[hh,hh].ravel()])
    # plot(xx, comp[:,0])
    # plot(xx, comp[:,1])
    # plot(xx, comp[:,2])
    semilogy(c_[xh, xh].ravel()[:-1], r_[0, c_[hh,hh].ravel()])
    semilogy(xx, comp[:,0])
    semilogy(xx, comp[:,1])
    semilogy(xx, comp[:,2])
