# ----------------------------------------------------------------------------
# Machine Learning in Physics Course at Florida State University.
# This contains work developed with Dr. Claire David and Tlotlo Oepeng in the
# context of the AIMS PINN black hole project.
#
# Harrison B. Prosper
# Created: Mon Aug 25 2025
# ----------------------------------------------------------------------------
import os, sys, re
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as td
import scipy.stats as st
# ----------------------------------------------------------------------------
# Using a Sobol sequence to created a sample of points
# ----------------------------------------------------------------------------
class SobolSample(np.ndarray):
    def __new__(cls,
                 lower_bounds,
                 upper_bounds,
                 num_points_exp=17, # of points = 2^num_points_exp
                 verbose=1):
       
        # Generate Sobol points in the unit D-cube and scale to bounds
        D = len(lower_bounds)
        sampler = st.qmc.Sobol(d=D, scramble=True)
        sample  = sampler.random_base2(m=num_points_exp) 
        sample  = st.qmc.scale(sample, lower_bounds, upper_bounds)

        if verbose:
            print("SobolSample")
            print(f"  {2**num_points_exp} Sobol points created in unit {D}-cube.")

        # Cast the numpy array to the type SobolSample
        sample = np.asarray(sample).view(cls)
        return sample
# ----------------------------------------------------------------------------
# Use uniform sampling to create a sample of points
# ----------------------------------------------------------------------------
class UniformSample(np.ndarray):
    def __new__(cls,
                 lower_bounds,
                 upper_bounds,
                 num_points,   # of points
                 verbose=1):

        # Generate points in the unit D-cube and scale to bounds
        D = len(lower_bounds)
        sample = np.random.uniform(0, 1, D*num_points).reshape((num_points, D))
        sample = st.qmc.scale(sample, lower_bounds, upper_bounds)
        
        if verbose:
            print("UniformSample")
            print(f"  {num_points} uniformly sampled points created in unit {D}-cube.")

        # Cast the numpy array to the type UniformSample
        sample = np.asarray(sample).view(cls)
        return sample
# ---------------------------------------------------------------------------
# Custom Dataset that takes (N, D) array of N points in the unit D-cube,
# Taken from AIMS PINN project
# ---------------------------------------------------------------------------
class Dataset(td.Dataset):
    
    def __init__(self, data, start, end,
                 targets=None,         # can specify targets explicitly
                 split_data=None,      # split data: [cols], [ncols-cols]
                 requires_grad=False,  # if True and split_data specified, 
                 random_sample_size=None,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 verbose=1):
        
        super().__init__()

        self.verbose  = verbose
        targets_given = type(targets) != type(None)
        split         = type(split_data) != type(None)
        
        if random_sample_size == None:
            x = torch.Tensor(data[start:end])
            if targets_given:
                if targets.dtype == int:
                    y = torch.tensor(targets[start:end])
                else:
                    y = torch.Tensor(targets[start:end])
        else:
            # create a random sample from items in the specified range (start, end)
            assert(type(random_sample_size) == type(0))
            
            length  = end - start
            assert(length > 0)
            
            indices = torch.randint(0, length-1, size=(random_sample_size,))
            x   = torch.Tensor(data[indices])
            if targets_given:
                if targets.dtype == int:
                    y = torch.tensor(targets[indices])
                else:
                    y = torch.Tensor(targets[indices])

        if split or targets_given:

            self.split = True

            if split:
                cols = split_data
                y = x[:, cols:]           
                x = x[:, :cols]
                
            if requires_grad:
                self.x = x.requires_grad_().to(device)
            else:
                self.x = x.to(device)

            self.y = y.to(device)

        else:
            # neither targets nor split specified
            self.split = False
            # do not split data
            self.x = x.to(device)

        if verbose:
            print('Dataset')
            print(f"  shape of x: {self.x.shape}")
            if self.split:
                print(f"  shape of y: {self.y.shape}")
            print()
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.split:
            return self.x[idx], self.y[idx]
        else:
            return self.x[idx]
# ---------------------------------------------------------------------------
# Custom DataLoader that is much faster than the default usage of the PyTorch
# DataLoader.
# ---------------------------------------------------------------------------
class DataLoader:
    '''
    A data loader that is much faster than the default PyTorch DataLoader.
    
    Notes:
           
       If num_iterations is specified, it is assumed that this is the
       desired maximum number of iterations, maxiter, per for-loop. 
       The flag shuffle is automatically set to True and an internal 
       count, defined by shuffle_step = floor(len(dataset) / batch_size) 
       is computed. The indices for accessing items from the dataset 
       are shuffled every time the following condition is True

           itnum % shuffle_step == 0,

       where itnum is an internal counter that keeps track of the iteration
       number. If num_iterations is not specified (the default), then
       the maximum number of iterations, maxiter = shuffle_step.
       
       This data loader, unlike the PyTorch data loader does not provide the 
       option to return the last batch if the latter is shorter than batch_size.
    '''
    def __init__(self, dataset, 
                 batch_size=None,
                 num_iterations=None,
                 verbose=1,
                 debug=0,
                 shuffle=False):

        self.dataset = dataset
        self.batch_size = batch_size
        self.niterations = num_iterations
        self.verbose = verbose
        self.debug   = debug
        self.shuffle = shuffle

        self.size = len(dataset)
        
        # need batch_size
        if self.batch_size == None:
            raise ValueError("you must specify a batch_size!")
            
        # If shuffle, then shuffle the dataset every shuffle_step iterations
        self.shuffle_step = int(len(dataset) / self.batch_size)

        if self.verbose:
            print('DataLoader')      
        
        if self.niterations != None:
            # The user has specified the maximum number of iterations 
            assert(type(self.niterations)==type(0))
            assert(self.niterations > 0)
            
            self.maxiter = self.niterations
            
            # IMPORTANT: shuffle indices every self.shuffle_step iterations
            self.shuffle = True

            if self.verbose:
                print('  Maximum number of iterations has been specified')
                
        elif self.size > self.batch_size:
            self.maxiter = self.shuffle_step
            
        else:
            # Note: this could be = 2 for a 2-tuple of tensors!
            self.size = len(dataset)
            self.shuffle_step = 1
            self.maxiter = self.shuffle_step

        if self.verbose:
            print(f'  maxiter:      {self.maxiter:10d}')
            print(f'  batch_size:   {self.batch_size:10d}')
            print(f'  shuffle_step: {self.shuffle_step:10d}')
            print()

        assert(self.maxiter > 0)
        
        # initialize iteration number
        # IMPORTANT: must start at -1 so that itnum goes from
        # 0 to batch_size - 1
        self.itnum = -1

        # initial indices for dataset (useful for debugging)
        self.indices = torch.tensor(np.linspace(0, 
                                                self.size-1,
                                                self.size).astype(int))

    # Tell Python to make objects of type DataLoader iterable
    def __iter__(self):
        return self

    # This method implements and terminates iterations
    def __next__(self): 

        # IMPORTANT: increment iteration number here, since we reset to itnum=-1
        self.itnum += 1

        if self.itnum < self.maxiter:

            if self.shuffle:
                # Create a new tensor indexing dataset via a random
                # sequence of indices
                jtnum = self.itnum % self.shuffle_step
                if self.itnum > 0 and (jtnum == 0):
                    self.indices = torch.randperm(self.size)
                    if self.debug > 0:
                        print(f'DataLoader shuffled indices @ index {self.itnum}')
                        
                start = jtnum * self.batch_size
                end = start + self.batch_size
                indices = self.indices[start:end]
                return self.dataset[indices]
                
            else:
                # Create a new tensor directly indexing dataset
                start = self.itnum * self.batch_size
                end = start + self.batch_size
                return self.dataset[start:end]
        else:
            # Terminate iteration and reset iteration counter
            # IMPORTANT: reset iteration number
            # 0 to size - 1
            self.itnum = -1
            raise StopIteration

    def __len__(self):
        return self.maxiter

    def __call__(self, itnum=0):
        self.itnum = itnum-1
        return next(self)
        
    def reset(self):
        self.itnum = -1