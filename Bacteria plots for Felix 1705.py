
# coding: utf-8

# In[10]:

import numpy
import scipy
import time
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import prettyplotlib as ppl
from prettyplotlib import brewer2mpl

set2 = brewer2mpl.get_map('Set2', 'qualitative', 8).mpl_colors
mpl.rc('axes', color_cycle = set2)
mpl.rc('axes', linewidth=3)
mpl.rc('xtick', labelsize=20)
mpl.rc('xtick.major', width=3)
mpl.rc('xtick.major', size=10)
mpl.rc('ytick', labelsize=20)
mpl.rc('ytick.major', size=10)
mpl.rc('ytick.major', width=3)
mpl.rc('lines', linewidth=3)


# In[11]:

#Figure 4+5 Consolidated

svbvds_id2 = zeros([10,10,2,2])
for i in range(1,20):
    filename = 'D:\\Po-Yi\\Grad\\yProgress\\1704\\AD\\Results0425\\Po_Yi_results\\svbvds_id2_rep_%d.npy'%i
    svbvds = load(filename)
    svbvds_id2 = svbvds_id2 + svbvds
svbvds_id2 = svbvds_id2/20

svbvds_id = zeros([10,10,2,2])
for i in range(1,20):
    filename = 'D:\\Po-Yi\\Grad\\yProgress\\1704\\AD\\Results0425\\Po_Yi_results\\svbvds_id_rep_%d.npy'%i
    svbvds = load(filename)
    svbvds_id = svbvds_id + svbvds
svbvds_id = svbvds_id/20

n = 10
m = 10
x = linspace(0.01, 0.30, num=n)
y = linspace(0.01, 0.30, num=m)

figure(figsize(20,20))
smin = 0
smax = 2

subplot(2,2,1)
pcolor(x, y, transpose(svbvds_id[:,:,0,0]), cmap='RdBu', vmin=smin, vmax=smax)
title('$f = 0.5$, $(C+D)/t_{db} = 0.7$',fontsize=22)
ylabel('$\sigma_{CD}/CD$',fontsize=22)
xlabel('$\sigma_i/\Delta$',fontsize=22)
axis([x.min(), x.max(), y.min(), y.max()])
colorbar()

filename = 'D:\\Po-Yi\\Fig4_1_1.npy'
z = transpose(svbvds_id[:,:,0,0])
save(filename,z)


subplot(2,2,2)
pcolor(x, y, transpose(svbvds_id[:,:,0,1]), cmap='RdBu', vmin=smin, vmax=smax)
title('$f = 0.5$, $(C+D)/t_{db} = 0.9$',fontsize=22)
ylabel('$\sigma_{CD}/CD$',fontsize=22)
xlabel('$\sigma_i/\Delta$',fontsize=22)
axis([x.min(), x.max(), y.min(), y.max()])
colorbar()

filename = 'D:\\Po-Yi\\Fig4_1_2.npy'
z = transpose(svbvds_id[:,:,0,1])
save(filename,z)


subplot(2,2,3)
pcolor(x, y, transpose(svbvds_id2[:,:,0,0]), cmap='RdBu', vmin=smin, vmax=smax)
title('$f = 0.5$, $(C+D)/t_{db} = 0.7$',fontsize=22)
ylabel('$\sigma_{CD}/CD$',fontsize=22)
xlabel('$\sigma_i/\Delta$',fontsize=22)
axis([x.min(), x.max(), y.min(), y.max()])
colorbar()

filename = 'D:\\Po-Yi\\Fig4_2_1.npy'
z = transpose(svbvds_id[:,:,1,0])
save(filename,z)


subplot(2,2,4)
pcolor(x, y, transpose(svbvds_id2[:,:,0,1]), cmap='RdBu', vmin=smin, vmax=smax)
title('$f = 0.5$, $(C+D)/t_{db} = 0.9$',fontsize=22)
ylabel('$\sigma_{CD}/CD$',fontsize=22)
xlabel('$\sigma_i/\Delta$',fontsize=22)
axis([x.min(), x.max(), y.min(), y.max()])
colorbar()

filename = 'D:\\Po-Yi\\Fig4_2_2.npy'
z = transpose(svbvds_id[:,:,1,1])
save(filename,z)

#savefig('D:\Po-Yi\\bacteria_noiseless.pdf', bbox_inches='tight')


# In[12]:

#Figure 8

svbvds_ia = zeros([10,10,2,2])
for i in range(1,20):
    filename = 'D:\\Po-Yi\\Grad\\yProgress\\1704\\AD\\Results0425\\Po_Yi_results\\svbvds_ia_rep_%d.npy'%i
    svbvds = load(filename)
    svbvds_ia = svbvds_ia + svbvds
svbvds_ia = svbvds_ia/20


n = 10
m = 10
x = linspace(0.01, 0.30, num=n)
y = linspace(0.01, 0.30, num=m)

figure(figsize(20,10))
smin = 0
smax = 2

subplot(1,2,1)
pcolor(x, y, transpose(svbvds_ia[:,:,0,0]), cmap='RdBu', vmin=smin, vmax=smax)
title('$f = 0.5$, $(C+D)/t_{db} = 0.7$',fontsize=22)
ylabel('$\sigma_{CD}/CD$',fontsize=22)
xlabel('$\sigma_i/\Delta$',fontsize=22)
axis([x.min(), x.max(), y.min(), y.max()])
colorbar()

subplot(1,2,2)
pcolor(x, y, transpose(svbvds_ia[:,:,0,1]), cmap='RdBu', vmin=smin, vmax=smax)
title('$f = 0.5$, $(C+D)/t_{db} = 0.9$',fontsize=22)
ylabel('$\sigma_{CD}/CD$',fontsize=22)
xlabel('$\sigma_i/\Delta$',fontsize=22)
axis([x.min(), x.max(), y.min(), y.max()])
colorbar()

#savefig('D:\Po-Yi\\bacteria_a.pdf', bbox_inches='tight')


# In[13]:

svbvds_ia = zeros([10,10,2,2])
for i in range(1,20):
    filename = 'D:\\Po-Yi\\Grad\\yProgress\\1704\\AD\\Results0425\\Po_Yi_results\\svbvds_ia_rep_%d.npy'%i
    svbvds = load(filename)
    svbvds_ia = svbvds_ia + svbvds
svbvds_ia = svbvds_ia/20

n = 10
m = 10
x = linspace(0.01, 0.30, num=n)
y = linspace(0.01, 0.30, num=m)

fig = figure(figsize(12,8))
ax = fig.gca();
smin = 0
smax = 2

ri = 0
cmap = cm.get_cmap('Blues')
plot(x, svbvds_ia[:,0,ri,ri], 'o', color=cmap(4/5), markeredgewidth=0, markersize=12, label='A $\sigma_T = 0.1$')
plot(x, svbvds_ia[:,3,ri,ri], 'o', color=cmap(3/5), markeredgewidth=0, markersize=12, label='A $\sigma_T = 0.10$')
plot(x, svbvds_ia[:,6,ri,ri], 'o', color=cmap(2/5), markeredgewidth=0, markersize=12, label='A $\sigma_T = 0.20$')
plot(x, svbvds_ia[:,9,ri,ri], 'o', color=cmap(1/5), markeredgewidth=0, markersize=12, label='A $\sigma_T = 0.30$')

xx = linspace(0,0.40,1e3)
b = exp(log(2)**2/2*(0.01*0.7)**2)
plot(xx, xx**2/((3*b**2+1)/4+xx**2-1), '-', color=cmap(4/5))
plot(xx, 1/(b**2+3/4*(b**2-1)/xx**2), '--', color=cmap(4/5))
b = exp(log(2)**2/2*(0.10*0.7)**2)
plot(xx, xx**2/((3*b**2+1)/4+xx**2-1), '-', color=cmap(3/5))
plot(xx, 1/(b**2+3/4*(b**2-1)/xx**2), '--', color=cmap(3/5))
b = exp(log(2)**2/2*(0.20*0.7)**2)
plot(xx, xx**2/((3*b**2+1)/4+xx**2-1), '-', color=cmap(2/5))
plot(xx, 1/(b**2+3/4*(b**2-1)/xx**2), '--', color=cmap(2/5))
b = exp(log(2)**2/2*(0.30*0.7)**2)
plot(xx, xx**2/((3*b**2+1)/4+xx**2-1), '-', color=cmap(1/5))
plot(xx, 1/(b**2+3/4*(b**2-1)/xx**2), '--', color=cmap(1/5))

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
ax.legend(fontsize=16, numpoints=1, loc='center left', bbox_to_anchor=(1, 0.5))

ylabel(r'$S(v_b,v_d)$', fontsize=22)
xlabel(r'$\sigma_t$', fontsize=22)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

#xlim([0,0.3])
#ylim([0,0.5])
#xticks([0,0.1,0.2,0.3])
#yticks([0,0.1,0.2,0.3,0.4,0.5])

#savefig('D:\Po-Yi\\bacteria_a.pdf', bbox_inches='tight')

