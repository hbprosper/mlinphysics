# Used in GNN and GAE tutorials
# ----------------------------------------------------------------
import os, sys
import numpy as np

# standard module for high-quality plots
import matplotlib as mp
import matplotlib.pyplot as plt

# module to access data in 
# Hierarchical Data Format (HDF or H5 format)
import h5py

# PyTorch
import torch
import torch.nn as nn

from tqdm import tqdm
# ----------------------------------------------------------------
SM, SIG = -1, 1
# ----------------------------------------------------------------
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
# ----------------------------------------------------------------
def load_events(filename, pTcut=0.5, which=0):
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
        pT = pT[select]
        
        ptmean += pT.mean()
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
    print(f'\tsample size:              {len(events):6d}')
    print(f'\tmin[multiplicity]:        {mincount:6d}')
    print(f'\tavg[multiplicity] +/-std: {m1:8.1f} +/-{m2:<8.1f}')
    print(f'\tmax[multiplicity]:        {maxcount:6d}')
    print(f'\tavg[pT]:                  {ptmean:8.1f} GeV')

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

def plot_graphs(loader, ptscale=25, scale=80,
                xmin=-6.0, xmax=6.0,
                ymin=-4.0, ymax=4.0,
                filename='events_as_graphs.png',
                ftsize=14):

    plt.rcParams.update({'font.size': 10})

    # work out number of columns and number of plots
    ncols = 2
    nrows = 2
    ndata = ncols * nrows
    fgsize= (7, 7)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=fgsize)
    
    axs = axs.flatten()
    
    gedge = IceCubeAdjacencyMatrix()
    
    # loop over number of point clouds, ndata
    for i, (event, y) in enumerate(loader):
        if i > ndata-1: break

        ax = axs[i]

        ax.set_xlabel(f'$y = {int(y.detach()):d}$')
        
        # get point cloud (i.e., event).
        # area of points in (eta, phi) proportional to pT of particle
        
        pt, eta, phi = event[0,:,:].detach().numpy().T[:]
        y = y.detach().numpy()[0]
        evt = event.squeeze().view(-1,len(pt), 3)
        A   = gedge(evt).squeeze()

        print(f'number of particles with pT > 4 GeV: {len(pt):4d}')
        
        size = scale * np.sqrt(pt / ptscale)
        cmap = mp.colormaps['rainbow']
        colors = cmap(size / size.max(), alpha=1)

        # create a graph
        G = nx.Graph()
        n = 0
        widths = []
        for j in range(len(pt)):
            G.add_node(j, value=pt[j])
            for k in range(len(pt)):
                if k == j: continue
                w = float(A[j, k].detach())
                n += 1
                if n < 500:
                    G.add_edge(j, k, weight=w)
                    widths.append(10*w)
                    
        widths = np.array(widths)

        # Position nodes
        pos = nx.spring_layout(G, seed=42)

        # Draw the graph
        nodes = nx.draw_networkx_nodes(G, pos, 
                                       node_size=5*size, 
                                       node_color=colors, ax=ax)
        
        edges = nx.draw_networkx_edges(G, pos, 
                                       width=widths, ax=ax)
        plt.plot()

    #fig.tight_layout()
    plt.savefig(filename)
    
class IceCubeAdjacencyMatrix(nn.Module):
    '''
    Given a set of vertices V, compute the IceCube adjacency matrix of shape (n, n).
    '''
    def __init__(self, alpha=1):

        super().__init__()

        # Note use of Parameter to tell PyTorch that the parameter
        # alpha is to be fitted.
        self.alpha = nn.Parameter(alpha * torch.rand(1))

    def forward(self, x):
        # shape of tensor x: (batch size, number of particles, number of attributes)
        _, size, d = x.shape

        # 1. compute square of Euclidean distance in (eta, phi) space
        #    dR^2 = deta^2 + dphi^2
        eta, phi = x[:, :, 1], x[:, :, 2]

        # use broadcasting to compute all possible differences
        deta = eta.view(-1, size, 1) - eta.view(-1, 1, size)
        dphi = tut.delta_phi(phi.view(-1, size, 1), phi.view(-1, 1, size))
        dRdR = deta**2 + dphi**2

        # 2. compute exp(- alpha * dR)**2)
        A = torch.exp(-self.alpha * dRdR)

        # 3. apply softmax in "horizontal" direction.
        A = torch.softmax(A, dim=-1)
        return A
        
        