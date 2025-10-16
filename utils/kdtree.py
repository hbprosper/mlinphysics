# ---------------------------------------------------------------------------
# Description: Implement a KD-tree using recursive binary partitioning.
# Created: Sep. 06 2025, Harrison B. Prosper in collaboration with ChatGPT3
# ---------------------------------------------------------------------------
import numpy as np
import math
# ---------------------------------------------------------------------------
# Model a node in the KD-tree. If the node has no children, then this node is
# a leaf.
class KDNode:
    '''
    Model a node (aka a bin) in a KD-tree. If a node has no child nodes, 
    then the node is a leaf.

    Leaf attributes
    ---------------
      ID: int                 Leaf identifier (0 to K-1), where K is the number 
                              of leaves in the tree.
      
      points: (n, m)-ndarray  The points within the leaf, where
                                n is the number of points in leaf
                                m is the dimension of the vector space in which 
                                the point cloud resides.
                                
      indices: (n, )-ndarray  The index (ordinal value of each point)
                  
      bounds: (m, 2)-ndarray  min and max limits of bounding box.
      
      density: float          Estimate of probability density: n / volume / size, 
                              where size is the total number of points in the 
                              KD-tree.

      volume: float           Volume of bounding box.
    '''
    def __init__(self, points, indices, axis, bounds, left=None, right=None):
        self.points = points        # None, or all points within the leaf
        self.indices= indices       #
        self.bounds = bounds        # bounding box [(min_0, max_0),(min_1, max_2),...] 
        self.density= None          # estimate of probability density
        self.volume = None          # volume of bounding box
        
        # for internal use
        
        self.split_point_ = None    # median point
        self.axis_   = axis         # dimension (axis) along which to 
        self.left_   = left         # "left" child node or None if this is a leaf
        self.right_  = right        # "right" child node of None if this is a leaf

    def is_leaf(self):
        return (self.left_ is None) and (self.right_ is None)

    def __str__(self):
        s = f'''leaf: {self.ID}
  number of points: {len(self.points):10d}
  density:          {self.density:10.3e}
  volume:           {self.volume:10.3e}
        '''
        return s

class KDTree:
    '''
    Implement recursive binary partitioning of n points in m dimensions.

    points: (n, m)-ndarray    Points to be binned into leaves (aka bins).'
    
    leaf_size: int            Upper bound on number of points / leaf.
                              Note: If len(points) = 2**P and leaf_size = 2**Q,
                              then the number of points/leaf will be 
                              exactly = leaf_count.
    '''
    
    def __init__(self, points, leaf_size=1, store_nodeinfo=False):

        # sample size (number of points)
        self.size = len(points)

        # index of each point
        indices = np.linspace(0, self.size-1, self.size).astype(int)
        
        self.leaf_size = leaf_size

        self.store_nodeinfo = store_nodeinfo
        
        # find lower bounds in all m dimensions
        mins = np.min(points, axis=0)

        # find upper bounds in all m dimensions
        maxs = np.max(points, axis=0)

        # cache leaves in a list
        self.leaves = []

        # cache nodes (mostly for debugging)
        self.nodeinfo = []
        
        # build tree recursively
        self.node = self._build(points, indices, bounds=(mins, maxs), depth=0)

        # sort leaves in order of decreasing probability density
        # Note: argsort() returns the indices that would sort the array
        ii = np.array([leaf.density for leaf in self.leaves]).argsort()
        ii = np.flip(ii) # reverse order of ode nindices
        self.leaves = list(np.array(self.leaves)[ii])
        
        # set leaf IDs
        for i in range(len(self.leaves)):
            self.leaves[i].ID = i 
            
    def _build(self, points, indices, bounds, depth):
        
        # get  number of points and dimensionality of vector space
        n_points, n_dim = points.shape

        # for current depth in tree decide along which dimension 
        # to split bins. (cycle through each dimension.)
        axis = depth % n_dim

        # if the number of points within node <= self.leaf_size
        # then make this node a leaf 
        if n_points <= self.leaf_size:

            bounds = np.array(bounds).T # [[min_0, max_0], [min_1, max_1], ...]
            
            leaf = KDNode(points=points, indices=indices, axis=axis, bounds=bounds)

            # add more information to leaf
            leaf.volume  = float(math.prod([xmax-xmin for xmin, xmax in bounds]))
            leaf.density = len(leaf.points) / leaf.volume / self.size

            # cache leaf in a list
            self.leaves.append(leaf)
                        
            return leaf
            
        # ...otherwise split current node at median point along current axis

        ii = points[:, axis].argsort() # get indices that would sort points along given axis
        
        sorted_points = points[ii]
        sorted_indices= indices[ii]
        median_idx    = n_points // 2
        split_point   = sorted_points[median_idx]
  
        # create bounds for children
        mins, maxs = bounds

        if self.store_nodeinfo:
            self.nodeinfo.append((depth, axis, np.array(bounds).T, split_point))
        
        # the split point along current axis is the upper bound for the left node
        left_maxs = maxs.copy() # initialize with the current upper bounds...
        left_maxs[axis] = split_point[axis] # ...and update the upper bound for current axis

        # the split point along current axis is the lower bound for right node
        right_mins = mins.copy()
        right_mins[axis] = split_point[axis]

        # build children
        left  = self._build(sorted_points[:median_idx], 
                            sorted_indices[:median_idx],
                            (mins.copy(), left_maxs),  depth + 1)
        
        right = self._build(sorted_points[median_idx:],
                            sorted_indices[median_idx:],
                            (right_mins, maxs.copy()), depth + 1)

        node  = KDNode(points=None, indices=None, 
                       axis=axis, bounds=bounds, 
                       left=left, right=right)
        
        node.split_point_ = split_point 
        
        return node

    def find_leaf(self, point):
        """Return the leaf (aka bin) into which the given point falls."""
        node = self.node
        while not node.is_leaf():
            axis = node.axis_
            if point[axis] < node.split_point_[axis]:
                node = node.left_
            else:
                node = node.right_
        return node

    def get_leaves(self): 
        return self.leaves

    def get_nodeinfo(self):
        I = [(self.nodeinfo[i][0], i) for i in range(len(self.nodeinfo))]
        I.sort()
        depth, index = list(zip(*I))
        return np.array(depth), np.array(index)
        
    def number_of_leaves():
        return self.__len__()
        
    def __len__(self):
        return len(self.leaves)