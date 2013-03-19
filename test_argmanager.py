from pylab import *
import venatexll


if __name__ == '__main__':


    aa = rand(100,100)

    bb = zeros(aa.shape)

    venatexll.smooth(aa, bb)

    figure()
    imshow(aa)

    figure()
    imshow(bb)

    show()
