import numpy as np
import random

from utils import mat_reader
from manifolds import deep_mlp

from deep_mlp import *
from diagonal_local_pca_metric import *
from geomedic_funcs import *
from geodesic_solver_bvp5c import *
from geodesic_solver_fp import *

# data inputs
param_file = 'example2_data_parametric.mat'
muNet_params = ('W0', 'W1', 'W2', 'b0', 'b1', 'b2')
sigmaNet_params = ('Wrbf', 'gammas', 'centers', 'zeta')

# parameters of the expected metric
muNet = {param : mat_reader(param_file, param) for param in muNet_params}
sigmaNet = {param : mat_reader(param_file, param) for param in sigmaNet_params}

# the manifold structure
manifold = deep_mlp(muNet, sigmaNet)

# randomly sample two points
data = mat_reader(param_file, 'data')
row_range, _ = data.shape
row_indices = list(range(row_range))
rand_row_indices = random.sample(row_indices, 2)
c0 = data[rand_row_indices[0], :]       # 1x2
c1 = data[rand_row_indices[1], :]       # 1x2

# define the fixed-point shortest path solver with default parameters
cov = np.cov(data)
dim = 2

# define the bvp5 solver with default options
bvp5c_options=[]
# bvp5c_options.tol = 1e-3;
solver_bvp5c=geodesic_solver_bvp5c(bvp5c_options)

# compute the shortest paths
curve_fp,logmap_fp,len_fp,failed_fp,solution_fp=compute_geodesic(solver_fp,manifold,c0,c1,nargout=5)
curve_bvp5c,logmap_bvp5c,len_bvp5c,failed_bvp5c,solution_bvp5c=compute_geodesic(solver_bvp5c,manifold,c0,c1,nargout=5)

# compute the exponential maps
curve_expmap_fp,solution_expmap_fp=expmap(manifold,c0,logmap_fp,nargout=2)
curve_expmap_bvp5c,solution_expmap_bvp5c=expmap(manifold,c0,logmap_bvp5c,nargout=2)

# plot the results
h_fp=plot_curve(curve_fp,'g','LineWidth',2)
h_bvp5c=plot_curve(curve_bvp5c,'r','LineWidth',2)
legend(concat([h_fp,h_bvp5c]),'Fixed-Point','bvp5c')
hq_bvp5c=quiver(c0(1),c0(2),logmap_bvp5c(1),logmap_bvp5c(2),0.1)
hq_bvp5c.Color = copy('r')
hq_fp=quiver(c0(1),c0(2),logmap_fp(1),logmap_fp(2),0.1)
hq_fp.Color = copy('g')
