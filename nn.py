# ----------------------------------------------------------------------------
# Machine Learning in Physics Course at Florida State University.
# This contains work developed with Claire David and Tlotlo Oepeng in the
# contect PINN black hole project.
#
# Harrison B. Prosper
# Created: Mon Aug 25 2025
# ----------------------------------------------------------------------------
import os, sys, re
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as td
from torch.optim.lr_scheduler import MultiStepLR

import scipy.stats as st
import json
# ----------------------------------------------------------------------------
# Simple utilities
# ----------------------------------------------------------------------------
# Note: there are several average loss functions available 
# in PyTorch, such as nn.CrossEntropyLoss(), but it's useful 
# to know how to create your own.
def average_quadratic_loss(f, y):
    # f and t must be of the same shape
    losses = (f - y)**2
    return torch.mean(losses)

def average_binary_cross_entropy_loss(f, y):
    # f and t must be of the same shape
    # Note: because of our use of the "where" function, the 
    # precise values of the targets doesn't matter so long as for
    # one class y < 0.5 and the other y > 0.5
    losses = -torch.where(y > 0.5, torch.log(f), torch.log(1 - f))
    return torch.mean(losses)
        
def number_of_parameters(model):
    '''
    Get number of trainable parameters in a model.
    '''
    return sum(param.numel() 
               for param in model.parameters() 
               if param.requires_grad)

# This function assumes that the len(loader) is the same as
# the batch size given when the loader is instantiated
def compute_avg_loss(objective, loader):    
    assert(len(loader)==1)
    objective.eval()
    for x, y in loader:
        # Detach from computation tree and send to CPU (if on a GPU)
        avg_loss = float(objective(x, y).detach().cpu())
    return avg_loss

def elapsed_time(now, start):
    etime = now() - start    
    t = etime
    hours = int(t / 3600)
    t = t - 3600 * hours
    minutes = int(t / 60)
    seconds = t - 60 * minutes
    etime_str = "%2.2d:%2.2d:%2.2d" % (hours, minutes, seconds)
    return etime_str, etime, (hours, minutes, seconds)

def get_steplr_scheduler(optimizer, config):
    # Number of milestones in multistep LR schedule
    n_steps = config('n_steps')
    n_milestones = n_steps - 1
    print(f'number of milestones: {n_milestones:10d}\n')

    # Learning rate milestones
    n_iters_per_step = config('n_iters_per_step')
    milestones = [n * n_iters_per_step for n in range(n_steps)]

    # learning rates
    base_lr = config('base_lr')
    gamma = config('gamma')
    lrs = [base_lr * gamma**i for i in range(n_steps)]
    
    print("Step | Milestone | LR")
    print("-----------------------------")
    for i in range(n_steps):
        print(f"{i:>4} | {milestones[i]:>9} | {lrs[i]:<10.1e}")
        if i < 1:
            print("-----------------------------")
    print()
    
    n_iters = n_steps * n_iters_per_step
    print(f'number of iterations:     {n_iters:10d}\n')
    
    # drop first entry of milestones list because it contains the base LR    
    return MultiStepLR(optimizer, milestones=milestones[1:], gamma=gamma)

def plot_loss_curve(losses):
    
    xx, yy_t, yy_v = losses
    
    # create an empty figure
    fig = plt.figure(figsize=(6, 3.8))
    fig.tight_layout()
    
    # add a subplot to it
    nrows, ncols, index = 1,1,1
    ax  = fig.add_subplot(nrows,ncols,index)
    
    ax.plot(xx, yy_t, color='red',  lw=1, label='training loss')
    ax.plot(xx, yy_v, color='blue', lw=1, label='validation loss')
    ax.legend()
    
    ax.set_xlabel('iterations', fontsize=FONTSIZE)
    ax.set_ylabel('average loss', fontsize=FONTSIZE)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, which="both", linestyle='-')

    plt.show()
# ----------------------------------------------------------------------------
# Classes 
# ----------------------------------------------------------------------------
class Sin(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)
        
class FCNN(nn.Module):
    '''
    Model a fully-connected neural network (FCNN).
    '''
    
    def __init__(self, 
                 n_inputs=2, 
                 n_hidden=4, 
                 n_width=32, 
                 nonlinearity=Sin):
        
        super().__init__()
        
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_width  = n_width
        
        cmd  = 'nn.Sequential(nn.Linear(n_inputs, n_width), nonlinearity(), '
        cmd += ', '.join(['nn.Linear(n_width, n_width), nonlinearity()' 
                          for _ in range(n_hidden-1)])
        cmd += ', nn.Linear(n_width, 1))'
        cmd  = cmd.replace(', ,', ', ') # Hack!
        
        self.net = eval(cmd)

    def save(self, dictfile):
        # save parameters of neural network
        torch.save(self.state_dict(), dictfile)

    def load(self, dictfile):
        # load parameters of neural network and set to eval mode
        self.load_state_dict(torch.load(dictfile, 
                                        weights_only=True,
                                        map_location=torch.device('cpu')))
        self.eval()

    def forward(self, x, p=None):
        assert(x.ndim==2)
        
        if type(p) != type(None):
            p = p.repeat(len(x), 1) if p.ndim < 2 else p
            x = torch.concat((x, p), dim=-1)

        y = self.net(x)   
        return y
# ----------------------------------------------------------------------------
# Extreme learning machine
# experimental
# ----------------------------------------------------------------------------
class ELM(nn.Module):
    '''
    Extreme learning machine (ELM)
    
    n_inputs, n_width, n_outputs: architecture. 
    Number of free parameters: n_width * n_outputs

    '''
    
    def __init__(self, n_inputs, n_width, n_outputs, 
                 nonlinearity=Sin):
        
        # WORK IN PROGRESS - DON'T USE!
        
        super().__init__()

        self.n_inputs = n_inputs
        self.n_width  = n_width
        self.n_outputs= n_outputs
        
        # linear layer of fixed random weights and biases
        self.weights = nn.Parameter(torch.randn(n_inputs, n_width), 
                                          requires_grad=False)
        self.biases  = nn.Parameter(torch.randn(n_width), 
                                    requires_grad=False)

        self.nonlinearity = nonlinearity

        # trainable linear layer
        self.free = nn.Linear(n_width, n_outputs, bias=False)
        
    def save(self, dictfile):
        # save parameters of neural network
        torch.save(self.state_dict(), dictfile)

    def load(self, dictfile):
        # load parameters of neural network and set latter to eval mode
        self.load_state_dict(torch.load(dictfile, weights_only=True,
                                        map_location=torch.device('cpu')))
        self.eval()
    
    def copy(self, x):
        # copy x into the parameter beta. we need to detach the tensor "free"
        # from the computation graph before we can copy data to it.
        self.free.weight.detach().copy_(torch.Tensor(x))
        
    def forward(self, x, p=None):
        assert(x.ndim==2)
        
        # check whether to concatenate inputs
        if type(p) != type(None):
            p = p.repeat(len(y), 1) if p.ndim < 2 else p
            x = torch.concat((x, p), dim=1)

        # calculate the output of the hidden layer
        y = self.nonlinearity(torch.mm(x, self.weights) + self.biases)

        # calculate the output of trainable layer
        y = self.free(y)

        return y

    def fit(self, x, y):
        # calculate the output of the hidden layer
        output = self.forward(x)

        # calculate the output weights
        pseudo_inverse = torch.pinverse(output)
        self.output_weights = torch.mm(pseudo_inverse, y)
# ---------------------------------------------------------------------------
class Config:
    '''
        Manage simple ML application configuration

          name:      name stub for all files, including the json file
          batchsize: 
          niter:     number of iterations
          base_lr:   base learning rate
          network:   network structure (n_hidden, n_width)
            :
          etc.
    '''
    def __init__(self, name, verbose=0):
        '''
        name:   name stub for all files, including the json file, or 
                the name of a json file. A json file is identified 
                by the extension .json
                
                    1. if name is a name stub, create a new json object.
                
                    2. if name is a json filename, create the json object
                       from the file.
        '''

        # check if a json file has been specified
        if name.endswith('.json'):
            self.cfg_filename = name # cache filename
            self.load(name)
        else:
            # this not a json file specification, assume it is a name stub
            # and build a json object
            cfg = []
            cfg.append(('name', name))
    
            # construct output file names    
            fcg = {}
            fcg['losses']     = f'{name}_losses.csv'
            fcg['params']     = f'{name}_params.pth'
            fcg['initparams'] = f'{name}_init_params.pth'
            
            cfg.append(('file', fcg))
    
            # create a default name for json configuration file
            # this name will be used if a filename is not
            # specified in the save method
            self.cfg_filename = f'{name}_config.json'
    
            # create partially filled json object
            # the rest will be filled with calls to __call__(...)
            self.cfg = dict(cfg)

        if verbose:
            print(self.__str__())
            
    def load(self, filename):
        # make sure file exists
        if not os.path.exists(filename):
            raise FileNotFoundError(f'{filename}')
        
        # read json file and cache as Python dictionary
        with open(filename, mode="r", encoding="utf-8") as file:
            self.cfg = json.load(file)

    def save(self, filename=None):
        # if no filename specified use default filename
        if filename == None:
            filename = self.cfg_filename

        # require .json extension
        if not filename.endswith('.json'):
            raise NameError('the output file must have extension .json')
            
        # save to JSON file
        open(filename, 'w').write(self.__str__())
        
    def __call__(self, key, value=None):
        # this method can be used to fill out the rest
        # of the json object
        keys = key.split('/')
        
        # if key exists, return its value
        cfg = self.cfg
        
        for ii, lkey in enumerate(keys):
            depth = ii + 1
            
            if lkey in cfg:
                val = cfg[lkey]
                if depth < len(keys):
                    # recursion
                    cfg = val
                else:
                    value = val
                    break
            else:
                # key is not in json object, so add it to json file
                if value == None:
                    # no value specified, so we can't add this key
                    raise KeyError(f'key "{lkey}" not found')
                    
                elif depth < len(keys):
                    cfg[lkey] = {}
                    cfg = cfg[lkey]
                else:
                    try:
                        cfg[lkey] = value
                    except:
                        pkey = keys[ii-1]
                        print(
                            f'''
    Warning: key '{key}' not created because '{pkey}' is 
    of type {str(type(pkey))}
                        ''')
        return value

    def __str__(self):
        # return a pretty printed string of the json object
        return str(json.dumps(self.cfg, indent=4))
# ---------------------------------------------------------------------------
class LRStepScheduler:
    def __init__(self, optimizer, scheduler, verbose=True):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.verbose   = verbose
        self.curr_lr   = -1.0

    def step(self):
        self.scheduler.step()
        
    def lr(self):
        lrate = self.optimizer.param_groups[0]['lr']
        if lrate != self.curr_lr:
            self.curr_lr = lrate
            if self.verbose:
                print()
                print(f'\t\tlearning rate: {lrate:10.3e}')
        return lrate
# ---------------------------------------------------------------------------
class Objective(nn.Module):

    def __init__(self, model, avgloss):
        super().__init__()
        self.model = model
        self.avgloss = avgloss
        
    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def save(self, paramsfile):
        self.model.save(paramsfile)
    
    def forward(self, x, y):
        f = self.model(x)          # compute model output
        return self.avgloss(f, y)  # compute average loss