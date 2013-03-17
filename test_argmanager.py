from pylab import *
import lowlevel

def f(x,y):
    return 2+3


if __name__ == '__main__':
    a = 1
    b = 2
    c = 3

    d = 1.0

    print 'a:', a
    print 'b:', b
    z = lowlevel.tf1(a, b)
    print 'z=', z

    ## Test insufficient number of args
    try:
        z = lowlevel.tf1()
    except TypeError as err:
        print 'OK, TypeError with message "' + err.message + '"'
    try:
        z = lowlevel.tf1(a)
    except TypeError as err:
        print 'OK, TypeError with message "' + err.message + '"'

    ## Test incorrect type
    try:
        z = lowlevel.tf1(d,a)
    except TypeError as err:
        print 'OK, TypeError with message "' + err.message + '"'
    try:
        z = lowlevel.tf1(d,d)
    except TypeError as err:
        print 'OK, TypeError with message "' + err.message + '"'
    try:
        z = lowlevel.tf1(a,d)
    except TypeError as err:
        print 'OK, TypeError with message "' + err.message + '"'


