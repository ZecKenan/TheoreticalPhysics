#Dependencies
from pylab import *
import numpy as np
import matplotlib.pyplot as plt

#Initial data
a=1.0e-9           # well width a=1 nm
hbar=1.0546e-34    # Plancks constant
m=9.1094e-31       # electron mass
e=1.6022e-19       # electron charge=-e
c=2.0*m/hbar**2    # constant in Schrödinger equation
N=10**5             # number of mesh points
dx=4*a/N             # step length
dx2=dx**2          # step length squared
#Initial electrostatics
EeV = 0.03466966*-5     # input energy in eV: test 0.3 , 0.4 , 0.3760 , 1.5
E = EeV*e          # input energy in J

#Print all correct solutions as comparison
print('Exact solution:')
print('E1=',(hbar*np.pi/a)**2/(2.0*m)/e,'eV')
print('E2=',(hbar*2.0*np.pi/a)**2/(2.0*m)/e,'eV')
print('E3=', (hbar*3*np.pi/a)**2/(2.0*m)/e, 'eV')
print('E4=', (hbar*4*np.pi/a)**2/(2.0*m)/e, 'eV')
print('E5=', (hbar*5*np.pi/a)**2/(2.0*m)/e, 'eV')


# potential energy function
def V(x):
    y = 0.0
    #y=e*5.*x/a # use this for triangular potential
    if x>0. and x<2. : y=e*5.*(x/a-1.) # finite triangular potential
    return y

# initial values and lists
x = -2*a            # initial value of position x
psi = 0.0           # wave function at initial position
dpsi = 1.0          # derivative of wave function at initial position
x_tab = []          # list to store positions for plot
psi_tab = []        # list to store wave function for plot
x_tab.append(x/a)
psi_tab.append(psi)

#Computing the wave function for def V(x)
for i in range(N) :
    d2psi = c*(V(x)-E)*psi
    d2psinew = c*(V(x+dx)-E)*psi
    psi += dpsi*dx + 0.5*d2psi*dx2
    dpsi += 0.5*(d2psi+d2psinew)*dx
    x += dx
    x_tab.append(x/a)
    psi_tab.append(psi)

#Printing the simulated wave function 
print('E=',EeV,'eV , psi(x=a)=',psi)
grad = np.gradient(psi_tab)

plt.close()
plt.figure(num=None, figsize=(8,8), dpi=80, facecolor='w', edgecolor='k')
plt.plot(x_tab, psi_tab, linewidth=1, color='red')
#plt.xlim(0, 1)
#limit=1.e-9
#plt.ylim(0, limit)
#plt.ylim(-limit, limit)
#plt.autoscale(False)
plt.title('Wavefunction $\psi$')
plt.xlabel('x/a')
plt.ylabel('$\psi$')
plt.savefig('psi.pdf')
plt.figure(2)
plt.xlabel('x/a')
plt.ylabel('d$\psi$/dx')
plt.plot(x_tab, grad)
plt.show()
