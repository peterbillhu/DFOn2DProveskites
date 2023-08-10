## Import packages
import numpy as np
from numpy import sin, cos, sqrt, pi
from os import listdir
from os.path import isfile, isdir, join, splitext

## The input elements scales and angles can be defined as follows:
## scales = np.array([a, b, c], dtype = 'float')
## angles = np.array([alp, bet, gam], dtype = 'float')
def get_unit_cell(scales, angles):
    ## In python, elements separated by ',' can recieve information from np.array
    ## Set scales (a, b, c) and angles (alp, bet, gam)
    a,b,c = scales
    ## alpha, beta, gamma
    alp, bet, gam = angles/360*2*pi
    ## In python the notation ** denotes the Exponentiation, e.g.,	x ** y
    Omg = a*b*c*sqrt(1-cos(alp)**2-cos(bet)**2-cos(gam)**2+2*cos(alp)*cos(bet)*cos(gam))
    ##　"Row" vectors: v1, v2, v3
    v1 = np.array([a, 0, 0])
    v2 = np.array([b*cos(gam), b*sin(gam), 0])
    v3 = np.array([c*cos(bet), c*(cos(alp)-cos(bet)*cos(gam))/sin(gam), Omg/(a*b*sin(gam))])
    ## The unit_cell consists of "row" vectors v1, v2, and v3
    unit_cell = np.array([v1, v2, v3])
    return unit_cell
