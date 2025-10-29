# Used in GNN and GAE tutorials
# ----------------------------------------------------------------
CHECK = "\u2705"
FAIL  = "\u274C"
WARN  = "\u26A0"
SM, SIG = -1, 1
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

try:
    # PyG
    from torch_geometric.nn import global_mean_pool
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.utils import scatter, softmax, subgraph
except:
    print(f'''
{WARN}: PyTorch Geometric is needed!

    Please use either
        conda install torch_geometric
    or 
        pip install torch_geometric

    to install the module. 
    ''')

# graph G = (V, E) plots
import networkx as nx

from tqdm import tqdm

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
# ----------------------------------------------------------------
# plotters
# ----------------------------------------------------------------
def plot_events(events, targets, 
                ndata=9, ptscale=25, scale=80,
                xmin=-6.0, xmax=6.0,
                ymin=-4.0, ymax=4.0,
                filename='events.png',
                ncols=3,
                fgscale=2.0,
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

def plot_graphs(loader, 
                ndata=9, ptscale=25, scale=8,
                ncols=3,
                fgscale=2.5,
                filename='events_as_graphs.png',
                ftsize=14):

    plt.rcParams.update({'font.size': 14})

    # work out number of columns and number of plots
    nrows = ndata // ncols
    ndata = nrows *  ncols
    fgsize= (fgscale*ncols, fgscale*nrows)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=fgsize)
    
    axs = axs.flatten()
    
    gedge = IceCubeAdjacencyMatrix()
    
    # loop over number of point clouds, ndata
    for i, (event, y) in enumerate(loader):
        if i > ndata-1: break

        ax = axs[i]
        
        # get point cloud (i.e., event).
        # area of points in (eta, phi) proportional to pT of particle
        pt, eta, phi = event[0,:,:].detach().numpy().T[:]            
        y = y.detach().numpy()[0]
        evt = event.squeeze().view(-1,len(pt), 3)
        A   = gedge(evt).squeeze()

        ax.set_xlabel(f'$y = {int(y):d}$ - count: {len(pt):5d}')
      
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

def plot_confusion_matrix(targets, predictions, 
                          fgsize=(4, 4),
                          gfile='confusion_matrix.png'):
    from sklearn.metrics import confusion_matrix

    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(y_true=targets, y_pred=predictions)

    # plot the confusion matrix 
    fig, ax = plt.subplots(figsize=fgsize)
    ax.matshow(conf_matrix, cmap=plt.cm.rainbow, alpha=0.4)

    # annotate each element of matrix with count
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i, s=conf_matrix[i, j],
                    va='center', ha='center', size='x-large')

    plt.xlabel('Predicted Labels', fontsize=16)
    plt.ylabel('True Labels', fontsize=16)
    plt.title(f'Confusion Matrix', fontsize=16)

    fig.tight_layout()
    plt.savefig(gfile)


def histogram_classifier_outputs(targets, y_hat,
                 xbins=50, xmin=0, xmax=1,
                 ymin=0, ymax=None,
                 filename='outputs.png',
                 fgsize=(5, 4),
                 ftsize=14):

    s = y_hat[targets > 0.5]
    b = y_hat[targets < 0.5]
    
    # create an empty figure
    fig = plt.figure(figsize=fgsize)

    nrows, ncols, index = 1, 1, 1
    ax = fig.add_subplot(nrows, ncols, index)

    # setup axes
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel(r'$\\hat{y} = D(G)$', fontsize=ftsize)
    ax.set_ylabel('density($y$)', fontsize=ftsize)

    cs, _, _ = ax.hist(s, bins=xbins, range=(xmin, xmax), 
            color='blue', alpha=0.3, label='signal', density=True)
 
    cb, _, _ = ax.hist(b, bins=xbins, range=(xmin, xmax), 
            color='red', alpha=0.3, label='background', density=True)

    if ymax is None:
        ymax = 2 * np.ceil(max(cs.max(), cb.max())/2)
        ax.set_ylim(ymin, ymax)

    ax.legend()

    fig.tight_layout()
    plt.savefig(filename)

def plot_roc(targets, y_hat, 
             fgsize=(4, 4), filename='ROC.png'):
    # standard measures of model performance
    from sklearn.metrics import roc_curve, auc

    bkg, sig, _ = roc_curve(targets, y_hat)

    roc_auc = auc(bkg, sig)

    fig = plt.figure(figsize=fgsize)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('$P(\\hat{y} > y | B)$', fontsize=14)
    plt.ylabel('$P(\\hat{y} > y | S)$', fontsize=14)

    plt.plot(bkg, sig, color='red',
             lw=1, label='ROC curve, AUC = %0.2f)' % roc_auc)

    plt.plot([0, 1], [0, 1], color='blue', lw=1, linestyle='--')

    plt.legend(loc="lower right", fontsize=11)
    fig.tight_layout()
    plt.savefig("ROC.png")
# ----------------------------------------------------------------
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
        dphi = delta_phi(phi.view(-1, size, 1), phi.view(-1, 1, size))
        dRdR = deta**2 + dphi**2

        # 2. compute exp(- alpha * dR)**2)
        A = torch.exp(-self.alpha * dRdR)

        # 3. apply softmax in "horizontal" direction.
        A = torch.softmax(A, dim=-1)
        return A
# ----------------------------------------------------------------
# PyTorch Geometric utilities
# ----------------------------------------------------------------
def edge_weights(nodes, edge_index,
                 alpha=1,
                 ieta=1,
                 iphi=2,
                 debug=False):
    '''
    Compute weights between two particles separated by deltaR in
    (eta, phi)-space.

    nodes      : tensor[number of particles, number of features] - nodes...duh!
    edge_index : tensor[2, number of edges] - indices of edges (i, j)
                 i = edge_index[0] - target node indices
                 j = edge_index[1] - source node indices
    alpha      : float - Scale defining size of particle neighborhood

    ieta       : int - Feature index of eta [1]
    iphi       : int - Feature index of phi [2]
    '''

    # 1. Compute dR**2 between all pairs of particles
    dR2 = delta_Rsquared(nodes, edge_index, ieta, iphi).to(nodes.device)

    # 2. Compute unnormalized edge weights exp(-alpha * dR**2)
    weights = torch.exp(-alpha * dR2).to(nodes.device)

    # 3. Normalize weights for each target node
    trg_nodes, src_nodes = edge_index
    weights = softmax(weights, trg_nodes)

    if debug:
        print('\tnormalized edge weights')
        num_nodes = len(nodes)
        from copy import copy
        wgt = copy(weights)
        for k in range(num_nodes):
            start= k * num_nodes
            end  = start + num_nodes
            wgt[start:end] /= wgt[start:end].sum()
            wsum = wgt[start:end].sum()
            wgt[start:end] /= wsum

        for k, (i, j) in enumerate(zip(trg_nodes, src_nodes)):
            print(f'{i:4d}, {j:4d}\t{weights[k]:8.4f}\t{wgt[k]:8.4f}')
            if k > 49:
                break
        print()

    return weights

class FullConnection(nn.Module):
    def __init__(self, self_loops=True):
        super().__init__()
        self.self_loops = self_loops

    def forward(self, nodes):
        '''
        Given tensor nodes[n_nodes, n_features] construct all
        possible edge connections, modeled as a two 2D tensor
        edge_index of shape (2, n_edges).

          I = edge_index[0] - target node indices
          J = edge_index[1] - source node indices
        '''
        num_nodes = len(nodes) # number of nodes

        # I = 0, 1, ... num_nodes - 1
        I = np.arange(0, num_nodes, 1)

        # create a numpy array of all possible tuples (i, j)
        I, J = np.meshgrid(I, I)
        J, I = np.stack((I, J), axis=2).reshape(-1, 2).T

        if not self.self_loops:
            # exclude pairs with i = j
            keep = J != I
            I = I[keep]
            J = J[keep]

        # create a tensor of shape [2, n_edges]
        return torch.tensor(np.array([I, J]), dtype=torch.long) 

class GraphDataset(list):

    def __init__(self, data, start, end,
                 targets=None,
                 random_sample_size=None,
                 connection=FullConnection(),  # define node connectivity
                 device=torch.device('cuda'
                                     if torch.cuda.is_available() else 'cpu'),
                 verbose=1):
        '''
    Create a dataset comprising graphs with potentially differing numbers
    of nodes. The input data and targets are converted to tensors.

    data: list[np.arrays]    Node data
    start, end : int         start and end of tensor
    targets : np.array       Target(s) associated with each element of data
    random_sample_size : int If specified, select a random sample from data
                             of size random_sample_size within given range.
    connection : function    Given a tensor x[n_nodes, n_features] of nodes,
                             this function should return a tensor of shape
                             [2, n_edges] specifying which target nodes and
                             source nodes are connected.
                             [default: FullConnection()]
    device : device-type
    verbose : int
        '''
        import numpy

        super().__init__()

        self.has_targets = type(targets) != type(None)

        # check data type
        if not isinstance(data[0], numpy.ndarray):
            raise TypeError("data should be a list of numpy arrays")

        if self.has_targets:
            if not isinstance(targets, numpy.ndarray):
                raise TypeError("targets should be a numpy array")

        self.connection = connection
        self.device = device
        self.verbose = verbose

        y = None
        if random_sample_size == None:
            x = data[start:end]
            if self.has_targets:
                y = targets[start:end]
        else:
            # create a random sample from items in the specified
            # range (start, end)
            assert(type(random_sample_size) == type(0))

            length  = end - start
            assert(length > 0)

            # we have a list of possibly inhomogeous arrays
            indices = torch.randint(start, end-1, size=(random_sample_size,))
            x = [data[i] for i in indices]
            if self.has_targets:
                y = targets[indices]

        # convert to tensors, cache data and
        # send to computational device
        self.x = [torch.tensor(z).to(device) for z in x]
        if self.has_targets:
            self.y = torch.tensor(y).to(device)

        dataset = self.__build_graphs()

        # NB: remember to initialize the list
        super().__init__(dataset)

        if verbose:
            print('Dataset')
            try:
                print(f"  shape of x: {self.x.shape}")
                if self.has_targets:
                    print(f"  shape of y: {self.y.shape}")
            except:
                print(f"  shape of x: {len(self.x)}")
                if self.has_targets:
                    print(f"  shape of y: {len(self.y)}")
            print()

    def __build_graphs(self):
        # build list of graphs, each modeled with the PyG class Data
        device = self.x[0].device
        dataset = []
        for i in tqdm(range(len(self.x))):
            dataset.append(
                Data(x=self.x[i],
                     edge_index=self.connection(self.x[i]).to(device),
                     y=torch.Tensor([self.y[i]]))
            )
        return dataset