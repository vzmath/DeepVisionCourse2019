import numpy as np

def deep_mlp(muNet=None,sigmaNet=None,*args,**kwargs):
    varargin = deep_mlp.varargin
    nargin = deep_mlp.nargin

    ## Create manifold object
    manifold=struct()
    manifold.W2 = copy(muNet.W2)
    manifold.b2 = copy(muNet.b2)
    manifold.W1 = copy(muNet.W1)
    manifold.b1 = copy(muNet.b1)
    manifold.W0 = copy(muNet.W0)
    manifold.b0 = copy(muNet.b0)

    # The parameters of the variance network
    manifold.Wrbf = copy(sigmaNet.Wrbf)
    manifold.gammas = copy(sigmaNet.gammas)
    manifold.C = copy(sigmaNet.centers)
    manifold.zeta = copy(sigmaNet.zeta)
    manifold.dimension = copy(size(manifold.W0,2))

    manifold=class_(manifold,'deep_mlp')
    return manifold

def f_mu(manifold=None,z=None,*args,**kwargs):
    varargin = f_mu.varargin
    nargin = f_mu.nargin

    ## Define the decoding functions
    f2=lambda x=None: tanh(x)
    f1=lambda x=None: softplus(x)
    f0=lambda x=None: softplus(x)

    layer0=f0(dot(manifold.W0,z) + manifold.b0)
    layer1=f1(dot(manifold.W1,layer0) + manifold.b1)
    layer2=f2(dot(manifold.W2,layer1) + manifold.b2)
    val=copy(layer2)
    return val

def f_sigma(manifold=None,z=None,*args,**kwargs):
    varargin = f_sigma.varargin
    nargin = f_sigma.nargin

    val=bsxfun(rdivide,1,sqrt(rbf(z,manifold)))
    return val

def rbf(z=None,manifold=None,*args,**kwargs):
    varargin = rbf.varargin
    nargin = rbf.nargin

    dist=sum(z ** 2,1) + sum(manifold.C ** 2,2) - dot(dot(2,manifold.C),z)
    val=dot(manifold.Wrbf,exp(- bsxfun(times,manifold.gammas,dist))) + manifold.zeta
    return val

def isdiagonal(manifold=None,*args,**kwargs):
    varargin = isdiagonal.varargin
    nargin = isdiagonal.nargin

    retval=copy(false)
    return retval

# This function implements the analytic expected Riemannian metric
# from a deep generator (VAE). The metric is expressed in the latent space.

# Input:
#   z - The latent point d x N.

# Output:
#   M        - The metric tensor for N points Nx(DxD).
#   dMdz     - The metric tensor derivatives Nx(DxD)xD. For each point N
#              the M(n,:,:,j) is the dM(z)/dz_j (derivative wrt dz_j).
#   Jacobian - The Jacobian matrix Nx2xDxd. For each N the J(n,j,:,:) is
#              the Jacobian of the mean (j=1) and sigma (j=2) respectively.

def metric_tensor(manifold=None,z=None,*args,**kwargs):
    varargin = metric_tensor.varargin
    nargin = metric_tensor.nargin

    # Get the parameters of the network
    W0=manifold.W0
    b0=manifold.b0
    W1=manifold.W1
    b1=manifold.b1
    W2=manifold.W2
    b2=manifold.b2
    Wrbf=manifold.Wrbf
    C=manifold.C
    gammas=manifold.gammas
    zeta=manifold.zeta

#     f2 = @(x)linearFun(x); df2 = @(x)dlinearFun(x); ddf2 = @(x)ddlinearFun(x);
#     f2 = @(x)sigmoid(x); df2 = @(x)dsigmoid(x); ddf2 = @(x)ddsigmoid(x);
#     f1 = @(x)tanh(x); df1 = @(x)dtanh(x); ddf1 = @(x)ddtanh(x);
#     f0 = @(x)tanh(x); df0 = @(x)dtanh(x); ddf0 = @(x)ddtanh(x);

    f2=lambda x=None: tanh(x)
    df2=lambda x=None: dtanh(x)
    ddf2=lambda x=None: ddtanh(x)
    f1=lambda x=None: softplus(x)
    df1=lambda x=None: dsoftplus(x)
    ddf1=lambda x=None: ddsoftplus(x)
    f0=lambda x=None: softplus(x)
    df0=lambda x=None: dsoftplus(x)
    ddf0=lambda x=None: ddsoftplus(x)

    D=size(Wrbf,1)
    d,N=size(z,nargout=2)

    ## Predefine the output matrices
    M=zeros(N,d,d)
    if (nargout > 1):
        dMdz=zeros(N,d,d,d)

    # Return the Jacobian matrix?
    if (nargout > 2):
        Jacobian=zeros(N,2,D,d)

    ## For speed-up do these computations
    W0Z0b0=dot(W0,z) + b0
    f0W0Z0b0=f0(W0Z0b0)
    W1f0W0Z0b0b1=dot(W1,f0W0Z0b0) + b1
    W2f1W1f0W0Z0b0b1b2=dot(W2,f1(W1f0W0Z0b0b1)) + b2

    for n in arange(1,N).reshape(-1):
        ## The mu network
        # The derivative of the mu network, Jacobian D x d
        dmudz=dot(bsxfun(times,df2(W2f1W1f0W0Z0b0b1b2(arange(),n)),W2),(dot(bsxfun(times,df1(W1f0W0Z0b0b1(arange(),n)),W1),bsxfun(times,df0(W0Z0b0(arange(),n)),W0))))
        # The derivative of the sigma network, Jacobian D x d
        dsigmadz=dot(- 0.5,bsxfun(rdivide,drbf(z(arange(),n),C,gammas,Wrbf),sqrt(bsxfun(power,rbf(z(arange(),n),C,gammas,Wrbf,zeta),3))))
        if (nargout > 2):
            Jacobian[n,1,arange(),arange()]=dmudz
            Jacobian[n,2,arange(),arange()]=dsigmadz
        # The metric tensor expectation d x d
        M[n,arange(),arange()]=dot(dmudz.T,dmudz) + dot(dsigmadz.T,dsigmadz)
        if (nargout > 1):
            for dd in arange(1,d).reshape(-1):
                dJmudzd=dot(bsxfun(times,dot(dot(bsxfun(times,ddf2(W2f1W1f0W0Z0b0b1b2(arange(),n)),W2),bsxfun(times,df1(W1f0W0Z0b0b1(arange(),n)),W1)),bsxfun(times,df0(W0Z0b0(arange(),n)),W0(arange(),dd))),W2),(dot(bsxfun(times,df1(W1f0W0Z0b0b1(arange(),n)),W1),bsxfun(times,df0(W0Z0b0(arange(),n)),W0)))) + dot(bsxfun(times,df2(W2f1W1f0W0Z0b0b1b2(arange(),n)),W2),(dot(bsxfun(times,dot(bsxfun(times,ddf1(W1f0W0Z0b0b1(arange(),n)),W1),bsxfun(times,df0(W0Z0b0(arange(),n)),W0(arange(),dd))),W1),bsxfun(times,df0(W0Z0b0(arange(),n)),W0)))) + dot(bsxfun(times,df2(W2f1W1f0W0Z0b0b1b2(arange(),n)),W2),(dot(bsxfun(times,df1(W1f0W0Z0b0b1(arange(),n)),W1),bsxfun(times,bsxfun(times,ddf0(W0Z0b0(arange(),n)),W0(arange(),dd)),W0))))
                dJsigmadzd=ddrbf(Wrbf,C,gammas,z(arange(),n),dd,zeta)
                dMdz[n,arange(),arange(),dd]=dot(dJmudzd.T,dmudz) + dot(dmudz.T,dJmudzd) + dot(dJsigmadzd.T,dsigmadz) + dot(dsigmadz.T,dJsigmadzd)

    return M,dMdz,Jacobian

# The derivative of the Jacobian of the variance network
def ddrbf(Wrbf=None,C=None,gammas=None,z=None,d=None,zeta=None,*args,**kwargs):
    varargin = ddrbf.varargin
    nargin = ddrbf.nargin

    temp=zeros(size(C))
    temp[arange(),d]=1
    dist=sum(z ** 2,1) + sum(C ** 2,2) - dot(dot(2,C),z)
    rbfVal=exp(multiply(- gammas,dist))
    output=dot(Wrbf,exp(multiply(- gammas,dist))) + zeta
    val=dot(- 0.5,bsxfun(times,bsxfun(times,dot(- 1.5,bsxfun(rdivide,1,sqrt(bsxfun(power,output,5)))),dot(Wrbf,(dot(- 2,bsxfun(times,gammas,bsxfun(times,(z(d) - C(arange(),d)),rbfVal)))))),dot(dot(- 2,Wrbf),bsxfun(times,bsxfun(times,gammas,(z.T - C)),rbfVal)))) + dot(- 0.5,bsxfun(times,bsxfun(rdivide,1,sqrt(bsxfun(power,output,3))),(dot(Wrbf,(dot(- 2,bsxfun(times,gammas,(bsxfun(times,temp,rbfVal) - dot(2,bsxfun(times,bsxfun(times,(z(d) - C(arange(),d)),gammas),bsxfun(times,(z.T - C),rbfVal)))))))))))
    return val

def drbf(z=None,C=None,gammas=None,Wrbf=None,*args,**kwargs):
    varargin = drbf.varargin
    nargin = drbf.nargin

    dist=sum(z ** 2,1) + sum(C ** 2,2) - dot(dot(2,C),z)
    rbfVal=exp(- bsxfun(times,gammas,dist))
    val=dot(dot(- 2,Wrbf),bsxfun(times,bsxfun(times,gammas,(z.T - C)),rbfVal))
    return val

def rbf(z=None,C=None,gammas=None,Wrbf=None,beta=None,*args,**kwargs):
    varargin = rbf.varargin
    nargin = rbf.nargin

    dist=sum(z ** 2,1) + sum(C ** 2,2) - dot(dot(2,C),z)
    val=dot(Wrbf,exp(- bsxfun(times,gammas,dist))) + beta
    return val

# The activation functions and their derivatives
def linearFun(z=None,*args,**kwargs):
    varargin = linearFun.varargin
    nargin = linearFun.nargin

    val=copy(z)
    return val

def dlinearFun(z=None,*args,**kwargs):
    varargin = dlinearFun.varargin
    nargin = dlinearFun.nargin

    val=ones(size(z))
    return val

def ddlinearFun(z=None,*args,**kwargs):
    varargin = ddlinearFun.varargin
    nargin = ddlinearFun.nargin

    val=zeros(size(z))
    return val

def softplus(z=None,*args,**kwargs):
    varargin = softplus.varargin
    nargin = softplus.nargin

    val=log(1 + exp(z))
    return val

def dsoftplus(z=None,*args,**kwargs):
    varargin = dsoftplus.varargin
    nargin = dsoftplus.nargin

    val=bsxfun(rdivide,1,1 + exp(- z))
    return val

def ddsoftplus(z=None,*args,**kwargs):
    varargin = ddsoftplus.varargin
    nargin = ddsoftplus.nargin

    val=bsxfun(rdivide,exp(z),bsxfun(power,1 + exp(z),2))
    return val

def relu(z=None,*args,**kwargs):
    varargin = relu.varargin
    nargin = relu.nargin

    val=bsxfun(max,z,0)
    return val

def drelu(z=None,*args,**kwargs):
    varargin = drelu.varargin
    nargin = drelu.nargin

    val=bsxfun(max,z,0)
    val=double(bsxfun(gt,val,0))
    return val

def ddrelu(z=None,*args,**kwargs):
    varargin = ddrelu.varargin
    nargin = ddrelu.nargin

    val=zeros(size(z))
    return val

def sigmoid(z=None,*args,**kwargs):
    varargin = sigmoid.varargin
    nargin = sigmoid.nargin

    val=sigmf(z,concat([1,0]))
    return val

def dsigmoid(z=None,*args,**kwargs):
    varargin = dsigmoid.varargin
    nargin = dsigmoid.nargin

    val=bsxfun(times,sigmoid(z),1 - sigmoid(z))
    return val

def ddsigmoid(z=None,*args,**kwargs):
    varargin = ddsigmoid.varargin
    nargin = ddsigmoid.nargin

    val=bsxfun(times,dsigmoid(z),1 - dot(2,sigmoid(z)))
    return val

def dtanh(z=None,*args,**kwargs):
    varargin = dtanh.varargin
    nargin = dtanh.nargin

    val=(1 - bsxfun(power,tanh(z),2))
    return val

def ddtanh(z=None,*args,**kwargs):
    varargin = ddtanh.varargin
    nargin = ddtanh.nargin

    val=dot(- 2,bsxfun(times,tanh(z),1 - bsxfun(power,tanh(z),2)))
    return val
