import numpy as np 

# This function computes all the shortest paths between the point mu and the data.
def compute_all_geodesics(solver=None,manifold=None,mu=None,data=None,*args,**kwargs):
    varargin = compute_all_geodesics.varargin
    nargin = compute_all_geodesics.nargin

    N,D=size(data,nargout=2)
    Curves=cell(N,1)
    Lens=NaN(N,1)
    Logs=NaN(N,D)
    Fails=false(N,1)
    Solutions=cell(N,1)
    t=tic()

    for n in arange(1,N).reshape(-1):
        fprintf('Computing geodesic %d / %d\n',n,N)
        curve,logmap,len_,failed,solution=compute_geodesic(solver,manifold,mu,data(n,arange()),nargout=5)
        Curves[n]=curve
        Lens[n]=len_
        Logs[n,arange()]=logmap
        Fails[n]=failed
        Solutions[n]=solution

    time_elapsed=toc(t)

    return Curves,Logs,Lens,Fails,Solutions,time_elapsed

# This function computes the length of curve along the Riemannian manifold.
# The input curve must be parametric type: curve = @(t)...

# Input:
#   manifold - An object of any manifold class.
#   curve    - A parametric curve c(t):[0,1]->M.
#   a, b     - The interval boundaries.
#   tol      - If the interal is smaller than tol then do not integrate.

# Output:
#   len      - A scalar, the curve length on the Riemannian manifold.
def curve_length(manifold=None,curve=None,a=None,b=None,tol=None,*args,**kwargs):
    varargin = curve_length.varargin
    nargin = curve_length.nargin

    ## Supply default arguments
    if (nargin < 3):
        a=0
    if (nargin < 4):
        b=1
    if (nargin < 5):
        tol=1e-06

    ## Integrate
    if (abs(a - b) > 1e-06):
        if (logical_not(isnumeric(curve))):
            len_=integral(lambda t=None: local_length(manifold,curve,t),a,b,'AbsTol',tol)

    return len_

## The infinitesimal local lenght at a point c(t).
def local_length(manifold=None,curve=None,t=None,*args,**kwargs):
    varargin = local_length.varargin
    nargin = local_length.nargin

    c,dc=curve(t,nargout=2)

    M=metric_tensor(manifold,c)

    if (isdiagonal(manifold)):
        d=sqrt(sum(multiply(M.T,(dc ** 2)),1))
    else:
        v1=squeeze(sum(bsxfun(times,M,dc.T),2))
        d=sqrt(sum(multiply(dc,v1.T),1))

    return d

def evaluate_solution(solution=None,t=None,t_scale=None,*args,**kwargs):
    varargin = evaluate_solution.varargin
    nargin = evaluate_solution.nargin

    cdc=deval(solution,dot(t,t_scale))

    D=size(cdc,1) / 2
    c=cdc(arange(1,D),arange()).T
    dc=dot(cdc(arange((D + 1),end()),arange()).T,t_scale)
    c=c.T
    dc=dc.T

    return c,dc

# Compute the exponential map of v expressed in the tangent space at mu.
# The used ODE is in
#   "Latent Space Oddity: on the Curvature of Deep Generative Models",
#       International Conference on Learning Representations (ICLR) 2018.

# Inputs:
#   manifold - any class which implements the metric_tensor method.
#     mu     - the origin of the tangent space. This should be a D-vector.
#     v      - a D-vector expressed in the tangent space at mu. This vector
#              is often a linear combination of points produced by the
#              'compute_geodesic' function.

#   Output:
#     curve    - a parametric function c:[0,1] -> M
#     solution - the output structure from the ODE solver. This is mostly
#                used for debugging.
def expmap(manifold=None,mu=None,v=None,*args,**kwargs):
    varargin = expmap.varargin
    nargin = expmap.nargin

    mu=ravel(mu)
    v=ravel(v)
    odefun=lambda x=None,y=None: second2first_order(manifold,y)

    if (norm(v) > 1e-05):
        curve,solution,failed=solve_ode(odefun,mu,v,manifold,nargout=3)
    else:
        curve=lambda t=None: repmat(ravel(mu),1,numel(t))
        solution=struct()
        failed=copy(false)

    return curve,solution,failed

# The actual intial value problem solver
def solve_ode(odefun=None,mu=None,v=None,manifold=None,*args,**kwargs):
    varargin = solve_ode.varargin
    nargin = solve_ode.nargin

    ## Compute how long the geodesic should be
    D=numel(mu)
    req_len=norm(v)
    init=concat([[ravel(mu)],[ravel(v)]])

    prev_t=0
    t=1
    solution=ode45(odefun,concat([0,t]),init)
    curve=lambda tt=None: evaluate_solution(solution,tt,1)
    sol_length=curve_length(manifold,curve,0,t)

    max_iter=1000
    for k in arange(1,max_iter).reshape(-1):
        if (sol_length > req_len):
            break
        prev_t=copy(t)
        t=dot(dot(1.1,(req_len / sol_length)),t)
        solution=odextend(solution,odefun,t)
        curve=lambda tt=None: evaluate_solution(solution,tt,1)
        sol_length=curve_length(manifold,curve,0,t)

    if (sol_length > req_len):
        ## Shorten the geodesic to have the required length
        t=fminbnd(lambda tt=None: (curve_length(manifold,curve,0,tt) - req_len) ** 2,prev_t,t)
        #t = fzero (@(tt) curve_length (manifold, curve, 0, tt) - req_len, [prev_t, t]);
        ## Create the final solution
        curve=lambda tt=None: evaluate_solution(solution,tt,t)
        failed=copy(false)
    else:
        failed=copy(true)
        warning('expmap: unable to make the solution long enough')

    return curve,solution,failed

def geodesic_system(manifold=None,c=None,dc=None,*args,**kwargs):
    varargin = geodesic_system.varargin
    nargin = geodesic_system.nargin

    D,N=size(c,nargout=2)
    if (size(dc,1) != D or size(dc,2) != N):
        error('geodesic_system: second and third input arguments must have same dimensionality')

    # Evaluate metric
    M,dM=metric_tensor(manifold,c,nargout=2)
    ddc=zeros(D,N)

    # Separate cases diagonal or not metric
    if (isdiagonal(manifold)):
        # Each row is dM = [dM/dc1, dM/dc2, ..., dM/dcD], where each column
        # dM/dcj is the derivatives of the diagonal metric elements wrt cj.
        # This has dimension Dx1.
        for n in arange(1,N).reshape(-1):
            dMn=reshape(dM(n,arange(),arange()),D,D)
            ddc[arange(),n]=dot(- 0.5,(dot(dot(2,(multiply(dMn,dc(arange(),n)))),dc(arange(),n)) - dot(dMn.T,(dc(arange(),n) ** 2)))) / (M(n,arange()).T)
    else:
        # Each row is dM = [dM/dc1| dM/dc2| ...| dM/dcD], where each slice
        # dM/dcj is the derivatives of the metric elements wrt cj.
        # This has dimension (DxD)xD.
        for n in arange(1,N).reshape(-1):
            M_n=reshape(M(n,arange(),arange()),D,D)
            if (rcond(M_n) < 1e-15):
                disp('Bad Condition Number of the metric')
                error('Bad Condition Number of the metric')
            # This is the dvec[M]/dc,
            # Each slice dM(n, :, :, d) is the derivative dM/dc_d for the nth point
            dvecMdc_n=reshape(dM(n,arange(),arange(),arange()),dot(D,D),D)
            # Construct the block diagonal matrix
            blck=kron(eye(D),dc(arange(),n).T)
            ddc[arange(),n]=dot(- 0.5,(numpy.linalg.solve(M_n,(dot(dot(dot(2,blck),dvecMdc_n),dc(arange(),n)) - dot(dvecMdc_n.T,kron(dc(arange(),n),dc(arange(),n)))))))

    return ddc

def second2first_order(manifold=None,state=None,*args,**kwargs):
    varargin = second2first_order.varargin
    nargin = second2first_order.nargin

    # Dimensions:
    # state: (2D)xN
    # y: (2D)xN
    D=size(state,1) / 2
    c=state(arange(1,D),arange())
    cm=state(arange((D + 1),end()),arange())
    cmm=geodesic_system(manifold,c,cm)
    y=concat([[cm],[cmm]])

    return y

def vec(A=None,*args,**kwargs):
    varargin = vec.varargin
    nargin = vec.nargin
    v=ravel(A)
    return v
