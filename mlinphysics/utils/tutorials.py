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
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

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
    PYG_IS_AVAILABLE = True
except:
    PYG_IS_AVAILABLE = False

def PYG_WARN():
    print(f'''
{WARN}: PyTorch Geometric is needed!

    Please use either
        conda install torch_geometric
    or 
        pip install torch_geometric

    to install the module. 
    ''')

try:
    import sklearn
    SKLEARN_IS_AVAILABLE = True
except:
    SKLEARN_IS_AVAILABLE = False

def SKLEARN_WARN():
    print(f'''
{WARN}:  scikit-learn is needed!

    Please use either
        conda install scikit-learn
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
                xmin=-4.0, xmax=4.0,
                ymin=-4.0, ymax=4.0,
                filename='events.png',
                ncols=3,
                fgscale=2.0,
                xpos_frac=0.14,
                ypos_frac=0.36,
                ftsize=18, 
                fgsize=(5, 5), dR_max=None):

    # work out number of columns and number of plots
    if len(events) == 1:
        plt.rcParams.update({'font.size': ftsize})
        ndata  = 1
        nrows  = 1
        ncols  = 1
    else:
        plt.rcParams.update({'font.size': 10})
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
        colors = cmap(size / size.max(), alpha=0.5)

        ax.scatter(eta, phi, s=size, c=colors)

        xpos = xpos_frac * (xmax-xmin)
        ypos = ypos_frac * (ymax-ymin)
        
        ax.text(xpos, ypos, f'$y = {y:d}$', fontsize=12)

        if len(events) == 1 and dR_max is not None:
            candidates = np.where(np.abs(phi) < 2.2)[0]
            k = candidates[np.argmax(pt[candidates])]      # the target particle
            
            t = np.linspace(0, 2*np.pi, 300)
            ax.plot(eta[k] + dR_max*np.cos(t), phi[k] + dR_max*np.sin(t), color='black', lw=2)
            ax.annotate(rf'$\Delta R_\mathrm{{max}}={dR_max}$',
                        (eta[k] + 0.71*dR_max, phi[k] + 0.71*dR_max),
                        xytext=(eta[k] + 1.3, phi[k] + 0.5), color='black', fontsize=15,
                        arrowprops=dict(arrowstyle='->', color='black'))

    
    fig.tight_layout()
    plt.savefig(filename)

def plot_adjacency_matrix(A, 
                   filename='adjacency_matrix.png',
                   fgsize=(6, 6),
                   ftsize=18,
                   figdir='figures', 
                   dpi=200):

    plt.rcParams.update({'font.size': ftsize})
    
    fig, ax = plt.subplots(figsize=fgsize)

    # fraction of matrix elements that are zero
    zero_fraction = (A == 0).mean()

    Am   = np.ma.masked_where(A == 0, A)   # masked elements are drawn white
    cmap = plt.get_cmap('Blues').copy()
    cmap.set_bad('white')
    
    im = ax.imshow(Am, cmap=cmap, interpolation='nearest',
                   norm=LogNorm(vmin=Am.min(), vmax=Am.max()))
    
    ax.set_xlabel(fr'particles sorted by $\phi$')
    ax.set_ylabel(fr'particles sorted by $\phi$')
    ax.set_title(rf'$A$: {100*zero_fraction:.0f}\% of the matrix elements are zero',
                 fontsize=14, color='black')
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('$A_{ij}$')
    
    fig.tight_layout()
    #os.makedirs(figdir, exist_ok=True)
    fig.savefig(filename)
    plt.show()

def plot_graphs(V, Y, 
                ndata=9, ptscale=25, scale=8,
                ncols=3,
                fgscale=2.5,
                filename='events_as_graphs.png',
                dR_max=0.8,
                ftsize=14, 
                fgsize=(5, 5), 
                layout='kamada_kawai'):

    layouts = ['circular', 'kamada_kawai', 'spiral', 'spring', 'shell']
       
    plt.rcParams.update({'font.size': 14})

    # work out number of columns and number of plots
    if len(V) < 2:
        plt.rcParams.update({'font.size': ftsize})
        ndata = 1
        nrows = 1
        ncols = 1
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=fgsize)
        axs = [axs]
    else:
        plt.rcParams.update({'font.size': 10})
        nrows = ndata // ncols
        ndata = nrows *  ncols
        fgsize= (fgscale*ncols, fgscale*nrows)

        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=fgsize)
        axs = axs.flatten()
    
    gedges  = GaussianAdjacencyMatrix(alpha=1.0, dR_max=dR_max) 
    
    # loop over number of point clouds, ndata
    for i, (event, y) in enumerate(zip(V, Y)):
        if i > ndata-1: break

        ax = axs[i]
        
        # get point cloud (i.e., event).
        # area of points in (eta, phi) proportional to pT of particle
        pt = event[:, 0].detach().cpu().numpy()           
        A  = gedges(event.unsqueeze(dim=0)).squeeze()

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
                if n < 500:
                    if w > 0:
                        n += 1
                        G.add_edge(j, k, weight=w)
                        widths.append(10*w)
                    
        widths = np.array(widths)

        # Position nodes
        if layout in layouts:
            print('graph layout', layout)
            pos = eval(f'nx.{layout}_layout(G)')
        else:
            pos = nx.planar_layout(G)

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
    ax.set_xlabel('$y = D(G)$', fontsize=ftsize)
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
    plt.xlabel('$P(Y > y | B)$', fontsize=14)
    plt.ylabel('$P(Y > y | S)$', fontsize=14)

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
    if not PYG_IS_AVAILABLE:
        raise ModuleNotFoundError(PYG_WARN())
    
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
        
        if not PYG_IS_AVAILABLE:
            raise ModuleNotFoundError(PYG_WARN())

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
            # note: the upper limit of randint is exclusive, so use end
            indices = torch.randint(start, end, size=(random_sample_size,))
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
                    print(f"  shape of y: {self.y.shape}, device: {self.y.device}")
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
                     y=torch.Tensor([self.y[i]]).to(device))
            )
        return dataset
# ================================================================
# Sparse graphs with a finite "attention domain"   (added Nov 2026)
# ================================================================
# The utilities below build graphs in which a particle is connected
# only to particles within a distance dR_max in (eta, phi) space,
#
#     dR_ij = sqrt( (eta_i - eta_j)**2 + dphi(phi_i, phi_j)**2 ) < dR_max,
#
# and weight the edges with a normalized Gaussian kernel
#
#     e_ij = softmax_j( -alpha * dR_ij**2 ),
#
# where the softmax is taken over all source nodes j connected to
# the target node i. Distances are always computed from the
# UNSCALED (eta, phi) so that dR has its usual physical meaning
# and the azimuthal wraparound phi = +pi <-> phi = -pi is handled
# correctly.
#
# Two equivalent implementations are provided:
#   GaussianAdjacencyMatrix - dense n x n matrix (plain PyTorch)
#   RadiusConnection +
#   gaussian_edge_weights   - sparse edge list (PyTorch Geometric)
# ----------------------------------------------------------------
def delta_R_matrix(eta, phi):
    '''
    Return the matrix of squared distances dR_ij**2 between all
    pairs of particles, computed with broadcasting.

    eta, phi : tensors of shape [..., n] (UNSCALED values)
    returns  : tensor of shape [..., n, n]
    '''
    deta = eta.unsqueeze(-1) - eta.unsqueeze(-2)   # [..., n, n]
    dphi = delta_phi(phi.unsqueeze(-1), phi.unsqueeze(-2))
    return deta**2 + dphi**2

class GaussianAdjacencyMatrix(nn.Module):
    '''
    Dense adjacency matrix A of shape (batch, n, n) with elements

        A_ij = softmax_j( -alpha * dR_ij**2 )   if dR_ij < dR_max
             = 0                                otherwise.

    alpha     : initial value of the (learnable) inverse squared
                width of the Gaussian. It is kept positive by fitting
                log(alpha) rather than alpha.
    dR_max    : radius of the attention domain. None means no cut,
                that is, a fully-connected graph.
    scale     : the factors by which (eta, phi) were divided when the
                data were scaled, e.g., (5, pi). They are used to undo
                the scaling so that dR is computed in physical units.
    ieta, iphi: feature indices of eta and phi.
    '''
    def __init__(self, alpha=1.0, dR_max=None, scale=(1.0, 1.0),
                 ieta=1, iphi=2):
        super().__init__()
        # nn.Parameter tells PyTorch that log(alpha) is to be fitted
        self.log_alpha = nn.Parameter(torch.log(torch.tensor([float(alpha)])))
        self.dR_max = dR_max
        self.scale  = scale
        self.ieta, self.iphi = ieta, iphi

    def alpha(self):
        return torch.exp(self.log_alpha)

    def forward(self, x):
        # x: tensor of shape (batch size, number of particles, features)

        # 1. undo the scaling of eta and phi
        eta = x[..., self.ieta] * self.scale[0]
        phi = x[..., self.iphi] * self.scale[1]

        # 2. squared distances between all pairs of particles
        dR2 = delta_R_matrix(eta, phi)           # (batch, n, n)

        # 3. logits of the Gaussian kernel
        logits = -self.alpha() * dR2

        # 4. particles outside the attention domain get logit -infinity,
        #    which the softmax turns into an exact zero. (The diagonal,
        #    dR = 0, is always inside, so no row is entirely -infinity.)
        if self.dR_max is not None:
            logits = logits.masked_fill(dR2 >= self.dR_max**2,
                                        float('-inf'))

        # 5. normalize each row (target node) with a softmax
        return torch.softmax(logits, dim=-1)

class RadiusConnection(nn.Module):
    '''
    Sparse analog of GaussianAdjacencyMatrix's cut: connect target
    node i to source node j only if dR_ij < dR_max. Each node is
    connected to itself (dR_ii = 0).

    Given pos[n, 2] = UNSCALED (eta, phi) of the n particles, return
    edge_index[2, K], where K is the number of connected pairs and

        edge_index[0] - target node indices i
        edge_index[1] - source node indices j

    Note: to find the pairs we compute all n**2 distances of ONE
    graph, but we keep only the K pairs inside the attention domain.
    It is the storage of the edges of ALL graphs that must be sparse.
    '''
    def __init__(self, dR_max=0.8):
        super().__init__()
        self.dR_max = dR_max

    def forward(self, pos):
        dR2 = delta_R_matrix(pos[:, 0], pos[:, 1])   # [n, n]
        # nonzero() returns the (row, column) = (target, source)
        # indices of the True elements as a tensor of shape [K, 2];
        # transpose it to get PyG's shape [2, K]
        return (dR2 < self.dR_max**2).nonzero().T.contiguous()

def gaussian_edge_weights(pos, edge_index, alpha, num_nodes=None):
    '''
    Sparse analog of GaussianAdjacencyMatrix: compute one weight per
    edge (i, j),

        e_ij = softmax_j( -alpha * dR_ij**2 ),

    where the softmax runs over all edges that share the same target
    node i.

    pos        : tensor[n, 2]  UNSCALED (eta, phi) of every node
    edge_index : tensor[2, K]  (target, source) node indices
    alpha      : positive tensor or float
    returns    : tensor[K]
    '''
    if not PYG_IS_AVAILABLE:
        raise ModuleNotFoundError(PYG_WARN())

    trg_nodes, src_nodes = edge_index

    # 1. squared distance for every edge
    deta = pos[trg_nodes, 0] - pos[src_nodes, 0]
    dphi = delta_phi(pos[trg_nodes, 1], pos[src_nodes, 1])
    dR2  = deta**2 + dphi**2

    # 2. softmax of -alpha * dR**2 over the edges of each target node
    return softmax(-alpha * dR2, trg_nodes, num_nodes=num_nodes)

class SparseGraphDataset(torch.utils.data.Dataset):
    '''
    A dataset of PyG graphs whose edges are built "lazily", that is,
    only when a graph is requested (by the DataLoader). This avoids
    storing the edges of every graph in memory: for a fully-connected
    graph of n nodes, the edge_index alone occupies 16 n**2 bytes,
    which for our 36,000 events amounts to about 14 GB!

    data   : list[np.array] of SCALED node features, one per event
    pos    : list[np.array] of UNSCALED (eta, phi), one per event
    targets: np.array of targets
    start, end         : range of events to use
    random_sample_size : if specified, a random sample of this size
                         is drawn from the range (start, end)
    connection         : function that maps pos[n, 2] to edge_index
                         [default: RadiusConnection(dR_max=0.8)]
    '''
    def __init__(self, data, pos, targets, start, end,
                 random_sample_size=None,
                 connection=RadiusConnection(0.8),
                 verbose=1):

        if not PYG_IS_AVAILABLE:
            raise ModuleNotFoundError(PYG_WARN())

        super().__init__()

        if random_sample_size is None:
            indices = range(start, end)
        else:
            indices = torch.randint(start, end,
                                    size=(random_sample_size,)).tolist()

        # cache node features, positions and targets as tensors
        self.x   = [torch.tensor(data[i], dtype=torch.float32) for i in indices]
        self.pos = [torch.tensor(pos[i],  dtype=torch.float32) for i in indices]
        self.y   = torch.tensor(np.asarray(targets)[list(indices)],
                                dtype=torch.float32)

        self.connection = connection

        if verbose:
            print(f'SparseGraphDataset: {len(self):d} graphs')

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        # build graph i on demand
        return Data(x=self.x[i],
                    pos=self.pos[i],
                    edge_index=self.connection(self.pos[i]),
                    y=self.y[i:i+1])
