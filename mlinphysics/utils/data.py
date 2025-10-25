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
                 split_col=None,       # split data: [cols], [ncols-cols]
                 requires_grad=False,  # if True and split_data specified,
                 random_sample_size=None,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 verbose=1):

        super().__init__()

        self.verbose  = verbose
        self.device   = device
        has_targets   = type(targets) != type(None)
        split_data    = type(split_col) != type(None)

        # store_as_tensor will be false if data is a list of tensors
        try:
            tmp = torch.Tensor(data[start:end])
            self.store_as_tensor = True
        except:
            self.store_as_tensor = False

        y = None
        if random_sample_size == None:

            if self.store_as_tensor:
                x = torch.Tensor(data[start:end])
            else:
                # assume we have a list of possibly inhomogeneous tensors
                x = data[start:end]

            if has_targets:
                if targets.dtype == int:
                    y = torch.tensor(targets[start:end])
                else:
                    y = torch.Tensor(targets[start:end])
        else:
            # create a random sample from items in the specified range (start, end)
            assert(type(random_sample_size) == type(0))

            length  = end - start
            assert(length > 0)

            indices = torch.randint(start, end-1,
                                        size=(random_sample_size,))
            if self.store_as_tensor:
                x = torch.Tensor(data[indices])
            else:
                # assume we have a list of possibly inhomogeous tensors
                x = [data[i] for i in indices]

            if has_targets:
                if targets.dtype == int:
                    y = torch.tensor(targets[indices])
                else:
                    y = torch.Tensor(targets[indices])

        # perhaps we should split?
        if split_data:
            has_targets = True # important!
            y = x[:, split_col:]
            x = x[:, :split_col].view(-1, split_col)

        if requires_grad:
            if self.store_as_tensor:
                x = x.requires_grad_(True)
            else:
                # assume we have a list of tensors
                x = [d.requires_grad_(True) for d in x]

        # cache, needed later
        self.has_targets = has_targets

        # cache data
        if self.store_as_tensor:
            self.x = x.to(device)
        else:
            self.x = [d.to(device) for d in x]

        # assume y is a tensor
        if self.has_targets:
            self.y = y.to(device)

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

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.has_targets:
            try:
                return self.x[idx], self.y[idx]
            except:
                return [self.x[i] for i in idx], [self.y[i] for i in idx]                
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

       This class uses the Python generator pattern
    '''
    def __init__(self, dataset,
                 batch_size=None,
                 num_iterations=None,
                 verbose=1,
                 debug=0,
                 shuffle=False):

        self.dataset = dataset
        self.batch_size = batch_size
        self.num_iterations = num_iterations
        self.verbose = verbose
        self.debug   = debug
        self.shuffle = shuffle

        self.size = len(dataset)

        # need batch_size
        if self.batch_size is None:
            raise ValueError("you must specify a batch_size!")

        # if shuffle, then shuffle the dataset every shuffle_step iterations
        self.shuffle_step = max(1, self.size // self.batch_size)

        if self.verbose:
            print('DataLoader')

        if self.num_iterations is not None:
            if self.verbose:
                print('  Number of iterations has been specified')

            # the user has specified the number of iterations
            assert(type(self.num_iterations)==type(0))
            assert(self.num_iterations > 0)

            self.maxiter = self.num_iterations

            # IMPORTANT: shuffle indices every self.shuffle_step iterations
            self.shuffle = True  
            
        elif self.size > self.batch_size:
            self.maxiter = self.shuffle_step
            
        else:
            # Note: this could be = 2 for a 2-tuple of tensors!
            self.shuffle_step = 1
            self.maxiter = self.shuffle_step

        if self.verbose:
            print(f'  maxiter:      {self.maxiter:10d}')
            print(f'  batch_size:   {self.batch_size:10d}')
            print(f'  shuffle_step: {self.shuffle_step:10d}')
            print()

        assert(self.maxiter > 0)

        # initialize iteration number
        self.itnum = 0

        # initial indices for dataset (useful for debugging)
        self.indices = torch.arange(self.size)
        
    # This method implements the Python generator pattern.
    # The for loop
    #  for batch in loader:
    #          : :
    # is logically equivalent to:
    #
    #  iterator = iter(loader) # call __iter__(self) once
    #  while True
    #     try:
    #        batch = next(iterator) # which resumes execution at yield call
    #     except StopIteration:
    #        break
    
    def __iter__(self):

        self.itnum = 0
        while self.itnum < self.maxiter:

            if self.shuffle:
                # create a new tensor indexing dataset via a random
                # sequence of indices
                jtnum = self.itnum % self.shuffle_step
                if self.itnum > 0 and jtnum == 0:
                    self.indices = torch.randperm(self.size)
                    if self.debug > 0:
                        print(f'DataLoader shuffled indices @ index {self.itnum}')

                start   = jtnum * self.batch_size
                end     = start + self.batch_size
                indices = self.indices[start:end]
                batch   = self.dataset[indices]
            else:
                # create a new tensor directly indexing dataset
                start   = self.itnum * self.batch_size
                end     = start + self.batch_size
                batch   = self.dataset[start:end]

            # increment iteration number
            self.itnum += 1

            # pause function and return a value
            yield batch

    def __len__(self):
        return self.maxiter
