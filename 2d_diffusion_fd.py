import numpy as np
import matplotlib.pyplot as plt
plt.interactive(False)
plt.show(block= True)

w= h= 10
dx= dy=0.1
D= 4  # thermal difussivity of steel
nx,ny = int(w/dx), int(h/dy)
T_high= 700
T_cool = 300
dt= dx**2*dy**2/(2*D*(dx**2+dy**2))
U0= np.ones([nx, ny])* T_cool
U= np.empty([nx,ny])
# initial condition: center in c with radius 2
c= (5,5)
rd= 5
for i in range(nx):
    for j in range(ny):
        r= np.sqrt((i*dx -c[0])**2 +(j*dy -c[1])**2)
        if r < 5:
            U0[i,j]= T_high


def df_timestep(U0,U):
    U[1:-1,1:-1]= U0[1:-1,1:-1]+ D* dt *((U0[2:,1:-1]-2*U0[1:-1,1:-1]+U0[:-2,1:-1])/dx**2+(U0[1:-1,2:]-2*U0[1:-1,1:-1]+U0[1:-1,:-2])/dy**2)
    U0= U.copy()
    return U0,U


# number of steps
nstep =101
mfig= [0,10,50,100]
fignum =0
fig= plt.figure()

for m in range(nstep):
    U0,U= df_timestep(U0,U)
    if m in mfig:
        fignum +=1
        print(m, fignum)
        ax= fig.add_subplot(220+fignum)
        im = ax.imshow(U.copy(),cmap= plt.get_cmap('hot'),vmin= T_cool,vmax= T_high)
        ax.set_axis_off()
        ax.set_title('{:.1f} ms'.format(m*dt*1000))
fig.subplots_adjust(right= 0.85)
cbar_ax= fig.add_axes([0.9,0.15,0.03,0.7])
cbar_ax.set_xlabel('$T$ /K',labelpad= 20)
fig.colorbar(im, cax= cbar_ax)
plt.show()

plt.plot([1,2,3,4])
plt.show()






