import numpy as np
from math import asinh, atan, sqrt, pi
eps = 1e-18
def f(p):
  x, y, z = (abs(p[0]), abs(p[1]), abs(p[2]))
  return + y / 2.0 * (z**2 - x**2) * asinh(y / (sqrt(x**2 + z**2) + eps)) \
         + z / 2.0 * (y**2 - x**2) * asinh(z / (sqrt(x**2 + y**2) + eps)) \
         - x*y*z * atan(y*z / (x * sqrt(x**2 + y**2 + z**2) + eps))       \
         + 1.0 / 6.0 * (2*x**2 - y**2 - z**2) * sqrt(x**2 + y**2 + z**2)

# newell g
def g(p):
  x, y, z = (p[0], p[1], abs(p[2]))
  return + x*y*z * asinh(z / (sqrt(x**2 + y**2) + eps))                         \
         + y / 6.0 * (3.0 * z**2 - y**2) * asinh(x / (sqrt(y**2 + z**2) + eps)) \
         + x / 6.0 * (3.0 * z**2 - x**2) * asinh(y / (sqrt(x**2 + z**2) + eps)) \
         - z**3 / 6.0 * atan(x*y / (z * sqrt(x**2 + y**2 + z**2) + eps))        \
         - z * y**2 / 2.0 * atan(x*z / (y * sqrt(x**2 + y**2 + z**2) + eps))    \
         - z * x**2 / 2.0 * atan(y*z / (x * sqrt(x**2 + y**2 + z**2) + eps))    \
         - x*y * sqrt(x**2 + y**2 + z**2) / 3.0

for i, t in enumerate(((f, 0, 1, 2), (g, 0, 1, 2), (g, 0, 2, 1), (f, 1, 2, 0), (g, 1, 2, 0), (f, 2, 0, 1))):
   print(i,t)
   # set_n_demag(i, t[1], t[0])
print(f, 0, 1, 2)

n     = (2, 2, 1)
axes= list(filter(lambda i: n[i] > 1, range(3)))
print(axes)

dx    = (5e-9, 5e-9, 3e-9)
m = np.ones(n + (3,))
h_ex = - 2 *m* sum([1/x**2 for x in dx])

print(h_ex.shape)
for i in range(6):
    mn =np.repeat(m,1 if n[i%3] == 1 else [i/3*2] + [1]*(n[i%3]-2) + [2-i/3*2], axis = i%3) / dx[i%3]**2
    # print(mn.shape)
    mm= np.array([i/3*2] + [1]*(n[i%3]-2)+ [2-i/3*2])
    #print(i,mn.shape)

    print(mn)
    #
    mmm= np.repeat(m,1,axis= 0)
    #print(mmm.shape)