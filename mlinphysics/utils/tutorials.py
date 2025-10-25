# Used in GNN and GAE tutorials
# -------------------------------------------
import os, sys
import numpy as np

# standard module for high-quality plots
import matplotlib as mp
import matplotlib.pyplot as plt

# module to access data in 
# Hierarchical Data Format (HDF or H5 format)
import h5py

# standard research-level machine learning 
# toolkit from Meta (FKA: FaceBook)
import torch

from tqdm import tqdm
# -------------------------------------------
SM, SIG = -1, 1
# -------------------------------------------
def load_events(filename, pTcut=4.0, which=0):
    hdf = h5py.File(filename, "r")

    targets = np.array(hdf['targets'])

    events = []
    trgs = []
    mincount = float('inf')
    maxcount = 0
    ptmean = 0.0
    m1 = 0.0
    m2 = 0.0
    
    for i in tqdm(range(len(targets))):
        
        if which != 0:
            if targets[i] * which < 0:
                continue
                
        # create event key into hdf file
        key = f'{i:d}'

        # get array of particles
        event = np.array(hdf[key])

        # apply lower pT threshold (pTCut)
        pT = event[:, 0]
        select = pT > pTcut
        event  = event[select]
        if len(event)==0:
            continue

        ptmean += pT[select].mean()
        n = len(event)
        m1 += n
        m2 += n**2
        
        # make sure data are ok
        if np.isnan(event).sum() > 0:
            raise ValueError(f'event {i:d} contains at least one NAN')

        if len(event) < mincount:
            mincount = len(event)

        if len(event) > maxcount:
            maxcount = len(event)

        # cache events in a Python list
        events.append(event)
        trgs.append(targets[i])

    m1 /= len(events)
    m2 /= len(events)
    m2 = np.sqrt(m2-m1**2)
    ptmean /= len(events)
    
    print()
    print(f'\tsample size:            {len(events):6d}')
    print(f'\tmin(multiplicity):      {mincount:6d}')
    print(f'\tmax(multiplicity):      {maxcount:6d}')
    print(f'\tE[pT]:                  {ptmean:8.1f} GeV')
    print(f'\tE[multiplicity] +/-std: {m1:8.1f} +/-{m2:<8.1f}')

    return events, np.array(trgs)

def plot_events(events, targets, 
                ndata=12, ptscale=25, scale=80,
                xmin=-6.0, xmax=6.0,
                ymin=-4.0, ymax=4.0,
                filename='events.png',
                ncols=4,
                fgscale=1.65,
                xpos_frac=0.14,
                ypos_frac=0.36,
                ftsize=14):

    plt.rcParams.update({'font.size': 10})

    # work out number of columns and number of plots
    nrows = ndata // ncols
    ndata = nrows *  ncols
    fgsize= (fgscale*ncols, fgscale*nrows)

    # create an empty figure
    fig = plt.figure(figsize=fgsize)

    # loop over number of point clouds, ndata
    for i in range(ndata):
        index = i+1
        ax = fig.add_subplot(nrows, ncols, index)

        # setup axes
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        if i > (nrows-1) * ncols-1:
            ax.set_xlabel(r'$\eta$', fontsize=ftsize)

        if i % ncols == 0:
            ax.set_ylabel(r'$\phi$', fontsize=ftsize)

        # get point cloud (i.e., event).
        # area of points in (eta, phi) proportional to pT of particle
        pt, eta, phi = events[i].T
        y = targets[i]

        size = scale * np.sqrt(pt / ptscale)
        cmap = mp.colormaps['rainbow']
        colors = cmap(size / size.max(), alpha=0.4)

        ax.scatter(eta, phi, s=size, c=colors)

        xpos = xpos_frac * (xmax-xmin)
        ypos = ypos_frac * (ymax-ymin)
        
        ax.text(xpos, ypos, f'$y = {y:d}$', fontsize=12)

    fig.tight_layout()
    plt.savefig(filename)
    
def delta_phi(phi2, phi1):
    '''
    Compute the difference in phi accounting for wraparound.
    '''
    deltaphi = phi2 - phi1
    abs_deltaphi = torch.abs(deltaphi)

    # handle wraparound
    deltaphi = torch.where(
        abs_deltaphi > torch.pi,
        2 * torch.pi - abs_deltaphi,
        abs_deltaphi)
    
    return deltaphi

def delta_Rsquared(x, edge_index, ieta=1, iphi=2):
    '''
    Compute square of distance between particles in (eta, phi) space.
    '''
    eta, phi = x[:, ieta], x[:, iphi]
    trg_nodes, src_nodes = edge_index

    # 1. Compute all possible eta differences
    deta = eta[trg_nodes] - eta[src_nodes]

    # 2. Compute all possible phi differences
    dphi = delta_phi(phi[trg_nodes], phi[src_nodes])

    # 3. Compute square of Euclidean distance in
    #    (eta, phi) space dR^2 = deta^2 + dphi^2
    return deta**2 + dphi**2
        