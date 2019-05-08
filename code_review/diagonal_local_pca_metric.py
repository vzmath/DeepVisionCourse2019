import numpy as np

def diagonal_local_pca_metric(data=None,sigma=None,rho=None,*args,**kwargs):
    varargin = diagonal_local_pca_metric.varargin
    nargin = diagonal_local_pca_metric.nargin

    ## Create manfold object
    manifold=struct()
    manifold.X = copy(data)

    manifold.sigma2 = copy(sigma ** 2)
    manifold.rho = copy(rho)
    manifold.dimension = copy(size(data,2))
    manifold=class_(manifold,'diagonal_local_pca_metric')
    return manifold

# Determine if a Riemannian metric is diagonal
# returns true if the Riemannian metric is always diagonal.
def isdiagonal(manifold=None,*args,**kwargs):
    varargin = isdiagonal.varargin
    nargin = isdiagonal.nargin

    retval=copy(true)
    return retval

# This function evaluates the Riemannian metric as well as its derivative.
# Input:
#   c - point D x N
# Output:
#   M    - A matrix NxD, each row has the diagonal elements of the metric.
#   dMdc - A matrix NxDxD, the M(n,:,j) are the dM(z)/dz_j.
def metric_tensor(manifold=None,c=None,*args,**kwargs):
    varargin = metric_tensor.varargin
    nargin = metric_tensor.nargin

    ## Get problem dimensions
    X=manifold.X
    R,D=size(X,nargout=2)
    N=size(c,2)
    sigma2=manifold.sigma2
    rho=manifold.rho
    M=NaN(N,D)

    if (nargout > 1):
        dMdc=zeros(N,D,D)

    for n in arange(1,N).reshape(-1):
        ## Compute metric
        cn=c(arange(),n)
        delta=bsxfun(minus,X,cn.T)
        delta2=delta ** 2
        dist2=sum(delta2,2)
        w_n=exp(dot(- 0.5,dist2) / sigma2) / ((dot(dot(2,pi),sigma2)) ** (D / 2))
        S=dot((delta ** 2).T,w_n) + rho
        m=1 / S
        M[n,arange()]=m
        if (nargout > 1):
            dSdc=dot(2,diag(dot(delta.T,w_n)))
            weighted_delta=bsxfun(times,(w_n / sigma2),delta)
            dSdc=dSdc - dot(weighted_delta.T,delta2)
            dMdc[n,arange(),arange()]=bsxfun(times,dSdc.T,m ** 2)

    return M,dMdc
