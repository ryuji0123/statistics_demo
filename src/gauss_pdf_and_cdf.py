import numpy as np
import matplotlib.pyplot as plt

from sympy import *
from sympy import pi, E
#import sympy as sp

mu = 0
sigma = 1

t = Symbol('t')
x = Symbol('x')

#ガウス分布の確率密度関数
y = E**(-(t-mu) ** 2 / 2 * sigma ** 2) / sqrt(2 * pi * (sigma ** 2))
plot(y, (t,-10,10))

#確率密度関数を積分
f = integrate(y, (t,-oo,x))
plot(f, (x,-10,10))
