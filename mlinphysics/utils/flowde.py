# ----------------------------------------------------------------------------
# # GSOC PINNDE Project
# > Sijil Jose, Pushpalatha C. Bhat, Sergei Gleyzer, Harrison B. Prosper<br>
# > Created: Tue May 27 2025<br>
# > Updated: Thu May 29 2025 HBP: generalize sigma(t)<br>
# > Updated: Sat Jul 19 2025 HBP: inspired by Sijil's logsumexp suggestion, 
#                                 try using torch.softmax to compute weights.
# ----------------------------------------------------------------------------
# standard system modules
import os, sys, re

# standard module for array manipulation
import numpy as np

# standard research-level machine learning toolkit from Meta (FKA: FaceBook)
import torch
import torch.nn as nn

from tqdm import tqdm
import scipy.stats as st

import matplotlib as mp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation

import shutil
# update fonts
plt.rcParams.update({
  "text.usetex": shutil.which('latex') is not None,
  "font.family": "sans-serif",
  "font.sans-serif": "Helvetica",
  "font.size": 14
  })
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
class qVectorField(nn.Module):    
    '''
    Compute a Monte Carlo approximation of the q vector field of the reverse-time 
    diffusion equation, 
    
     [sigma_0 + (1 - sigma_0) * t] * dx/dt = (1-sigma0) * x - q(t, x),

    where xt = x(t) is a d-dimensional vector and q(t, xt) is a d-dimensional 
    time-dependent vector field and s0 is that value of sigma(t) at t=0.
     
    The vector field q(t, x) is defined by a d-dimensional integral which is 
    approximated with a Monte Carlo (MC)-generated sample, x0, of shape (M, d),
    where M is the sample size and d is the dimension of the vector space.

    Example
    -------

    q = qVectorField(x0)
        :  :
    qt = q(t, x)
    '''
    def __init__(self, x0, sigma0=1e-3, debug=False):

        super().__init__()

        assert(x0.ndim==2)
        # x0.shape: (M, d)
        
        # change shape of x0 from (M, d) to (1, M, d)
        # so that broadcasting works correctly later.
        self.x0 = x0.unsqueeze(0)

        self.sigma0 = sigma0
        self.debug  = debug

        if debug:
            print('qVectorField.__init__: x0.shape', x0.shape)

    def set_debug(self, debug=True):
        self.debug = debug

    def sigma(self, t):
        return self.sigma0 + (1 - self.sigma0) * t

    def forward(self, t, x):
        assert(x.ndim==2)

        if type(t) == type(x):
            assert(t.ndim==2)

            # change shape of t so that broadcasting works correctly
            t = t.unsqueeze(1)
            # t.shape: (N, 1) => (N, 1, 1)
            
        debug = self.debug
        x0 = self.x0
        sigma0 = self.sigma0
        
        # change shape xt so that broadcasting works correctly
        x = x.unsqueeze(1)
        # x.shape: (N, d) => (N, 1, d)

        if debug:
            print('qVectorField(BEGIN)')
            print('  x.shape: ', x.shape)
            print('  x0.shape:', x0.shape)

        alphat = 1 - t
        sigmat = self.sigma(t)
        y  = x - alphat * x0
        vt = y / sigmat
        # vt.shape: (N, M, d)
        
        if debug:
            print('  qVectorField: vt.shape', vt.shape)
            print('                vt', vt)

        if torch.isnan(vt).any():
            if torch.isnan(y):
                print('NAN in y')
            if torch.isnan(sigmat):
                print('NAN in sigmat')
            raise ValueError("vt contains at least one NAN")

        if torch.isinf(vt).any():
            raise ValueError("vt contains at least one INF")

        # sum over arguments of exponential, that is, over
        # over the d-dimensions of each element in x0, so
        # that we get the product of d normal densities for
        # each Monte Carlo-sampled point and for each batch
        # instance. 
        u = (vt*vt).sum(dim=-1)/2
        # u.shape: (N, M)
        
        # for each row, find minimum value of u, umin,
        # and change shape of umin from (N,) => (N, 1) so
        # that broadcasting with u (of shape (N, M)) will
        # work correctly. Note: min(...) returns (min(u), pos(u))
        # where "min(u)" is the minimum value of u and pos(u) its
        # ordinal value (position).
        umin = u.min(dim=-1)[0].unsqueeze(-1)
        
        if debug:
            print('  qVectorField: u.shape, umin.shape', u.shape, umin.shape)

        # compute weights
        # ---------------
        # note: by adding umin to -u, we guarantee that at least
        # one term in the sum exp will be equal to unity, while all 
        # the rest will be < 1.  The softmax is performed along the
        # Monte Carlo sample direction.
        wt = torch.softmax(-u + umin, dim=-1)
        # wt.shape: (N, M)

        # compute effective count
        #neff = 1.0 / (wt*wt).sum(dim=-1)
        
        if debug:
            print('  qVectorField: wt.shape', wt.shape)
            print('              : wt.sum', wt.sum(dim=-1))
  
        if torch.isnan(wt).any():
            raise ValueError("wt contains at least one NAN")

        if torch.isinf(wt).any():
            raise ValueError("wt contains at least one INF")

      
        # now sum over the Monte Carlo sample of M weighted 
        # elements of x0
        # x0.shape: (1, M, d)
        # wt.shape: (N, M) => (N, M, 1)
        x0_wt = x0 * wt.unsqueeze(-1)
        # x0_wt.shape: (N, M, d)

        if debug:
            print('  qVectorField: x0_wt.shape', x0_wt.shape)

        if torch.isnan(x0_wt).any():
            raise ValueError("x0_wt contains at least one NAN")

        # sum over the MC sample dimension of x0_wt
        qt = x0_wt.sum(dim=1)
        # qt.shape: (N, d)

        if debug:
            print('  qVectorField: qt.shape', qt.shape)
            print('qVectorField(END)')
 
        if torch.isnan(qt).any():
            raise ValueError("qt contains at least one NAN")

        return qt

    def Gprime(self, t, x):
        qt = self(t, x)
        return (1-self.sigma0)*x - qt
        
    def G(self, t, x):
        return self.Gprime(t, x) / self.sigma(t)
# ----------------------------------------------------------------------------
def get_normal_sample(x):
    try:
        x = x.cpu()
    except:
        pass
    means = np.zeros_like(x)
    scales = np.ones_like(x)
    return torch.Tensor(np.random.normal(loc=means, scale=scales))
        
def get_target_sample(x, size=5000):
    ii = np.random.randint(0, len(x)-1, size)
    return torch.Tensor(x[ii])

class FlowDE(nn.Module):    
    '''
    Given standard normal vectors z = x(t=1), compute target vectors 
    x0 = x(t=0) by mapping z to x0 deterministically. x0, which is 
    of shape (M, d), where M is the Monte Carlo (MC) sample size and d 
    the dimension of the vector x0 = x(0), is used to compute a MC 
    approximation of the q vector field. The tensor z is of shape (N, d), 
    where N is the number of points sampled from a d-dimensionbal 
    standard normal. 

    Utility functions
    =================
    1. get_normal_sample(X0) returns a tensor z = x(1), with the same shape 
    as X0, whose elements are sampled from a diagonal d-dimensional Gaussian. 

    2. get_target_sample(X0, M) returns a sample of points, x0, of size M 
    from X0, which will be used to approximate the q vector field.

    Example
    -------
    N = 4000
    M = 4000
    
    z  = get_normal_sample(X0[:N]).to(DEVICE)
    x0 = get_target_sample(X0, M).to(DEVICE)
    
    flow = FlowDE(x0)

    y = flow(z)
    
    '''
    def __init__(self, x0, sigma0=1e-3, T=500, savepath=False, step=5, debug=False, 
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        
        # x0: MC sample of shape (M, d)
        # T:  number of time steps in [1, 0]
        
        super().__init__()

        assert(x0.ndim==2)
        
        self.q = qVectorField(x0, sigma0, debug).to(device)
        self.sigma0 = sigma0
        
        if T < 4: T = 4

        self.T = T
        self.h = 1/T # step size
        self.savepath = savepath
        self.step = step
        self.debug = debug

    def set_debug(self, debug=True):
        self.debug = debug

    def G(self, t, xt):
        # t is either a float or a 2D tensor of shape (N, 1)
        # xt.shape: (N, d)
        return  self.q.G(t, xt)

    def forward(self, z):
        assert(z.ndim==2)
        
        debug = self.debug

        savepath = self.savepath
        T = self.T
        h = self.h
        t = 1      # initial "time"
        xt= z
        dim = xt.shape[-1]
        
        if debug:
            print('FlowDE.forward: xt.shape', xt.shape)
            print('FlowDE.forward: t', t)

        if savepath:
            y = [xt]
            t_y = [t]
            
        G1 = self.G(t, xt)
        
        for i in tqdm(range(T-1)):
            t = (T-i-1) * h
            #t -= h
            if t < 0: 
                break
                
            if debug:
                print('FlowDE.forward: t', t)
                print('FlowDE.forward: xt.shape', xt.shape) 
                print('FlowDE.forward: G1.shape', G1.shape)
                
            G2 = self.G(t, xt - G1 * h)

            xt = xt - (G1 + G2) * h / 2
            
            G1 = G2.detach().clone()

            if savepath:
                if (i+1) % self.step == 0:
                    y.append(xt)
                    t_y.append(t)

        if savepath:
            return torch.Tensor(t_y), torch.concat(y, dim=0).view(len(y), *xt.shape)
        else:
            return xt
            
class Flow2DAnimation:
    '''
    Given a Python list, y, of arrays (or tensors), each of shape (N, d) where N is the
    number of data points and d=2 is the dimensionality of the space, animate the flow
    of points from a diagonal 2D, zero mean, unit variance Gaussian to the target
    distribution.

    anim = Flow2DAnimation(y)

    The animation can be saved as a gif or, if ffmpeg is available, in mp4 format using

    anim.save('flow.gif')
    anim.save('flow.mp4')

    Parameters
    ----------
        data :      Python list of data objects. Each object is table (either a 2D numpy 
                    array or a 2D tensor) or shape (N, d). Each data object corresponds
                    to a time step.
        xmin, xmax: x limits [-5, 5]
        ymin, ymax: y limits [-5, 5]
        nframes :   number of frames [50]
        interval :  time between frames in milliseconds [100 ms]
        mcolor :    marker color ['blue']
        msize :     marker size [1]
        fgsize :    size of figure [(4, 4)]
        ftsize :    font size [16 pt]
    '''
    def __init__(self, data,
                 xmin=-5, xmax=5, ymin=-5, ymax=5,
                 nframes=50,     # number of frames
                 interval=100,   # time between frames in milliseconds
                 mcolor='blue',  # color of points
                 msize=1,
                 fgsize=(4, 4), 
                 ftsize=16):
        
        # cache inputs
        self.data = np.array(data)
        self.nframes = nframes
        self.interval = interval
        self.T = len(self.data)
        self.npoints, self.d = self.data[0].shape
        self.factor = self.T / nframes
        
        # set size of figure
        self.fig = plt.figure(figsize=fgsize)

        # create area for a plot 
        nrows, ncols, index = 1, 1, 1
        ax  = plt.subplot(nrows, ncols, index)
        
        self.ax = ax   # cache plot
        
        ax.set_title('Reverse Time Diffusion', pad=14)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        
        # create a scatter plot 
        x = self.data[0, :, 0] # initial x positions
        y = self.data[0, :, 1] # initial y positions
       
        self.scatter = ax.scatter(x, y, s=msize, c=mcolor, alpha=1)

        # matplotlib refers to the objects scatter, etc., as artists.
        # we need to place artists in a list since we need to return all of
        # them in the function update() for the animation to work correctly
        self.artists = []
        self.artists.append(self.scatter)

        # IMPORTANT: Turn off Latex processing of text; it is far too slow!
        mp.rc('text', usetex=False)

        # create a text object to display the days since the start
        self.text = ax.text(0.95*xmin, 0.80*ymin, f't: {0:5.2f}')
        self.artists.append(self.text)

        # fix the layout so that labels are a bit
        # closer to the plot
        self.fig.tight_layout()

        # don't show the above plots. Show only the animated version
        plt.close()

        # initialize animated plot
        self.ani = FuncAnimation(fig=self.fig,           # animation figure
                                 func=self.update,       # function to update plot in figure
                                 repeat=False,           # don't repeat animation
                                 frames=self.nframes,    # number of frames
                                 interval=self.interval) # time between frames (ms)

    # this is the function that updates the plot    
    def update(self, frame):
        
        print(f'\rframe: {frame:d}', end='')
        
        # get data and artists!
        data, scatter, text = self.data, self.scatter, self.text
        
        # compute index into data array (i.e., time step)
        index = np.ceil(frame * self.factor).astype(int)
        t = 1 - frame / (self.nframes-1)
        
        # display the number of days
        self.text.set_text(f't: {t:5.2f}')
        
        # display the current position of points
        x = data[index, :, 0]
        y = data[index, :, 1]
        a = np.array([x, y]).T   # we need x and y to be columns
        scatter.set_offsets(a)

        # must return artists for animation to work
        return self.artists
    
    def show(self):
        '''
        Construct and show the animation.
        '''
        plt.show()
        return self.ani
    
    def save(self, filename):
        '''
        Construct and save the animation to a file. The format of the file is
        determined from the file extension. For example, a filename with 
        extension ".gif" saves the animation as an animated gif, while one with
        extension ".mp4" saves it in mp4 format. Note: the latter requires the
        module ffmpeg.
        
    Parameters
    ----------
        filename :      Name of graphics file.
        '''
        self.ani.save(filename)