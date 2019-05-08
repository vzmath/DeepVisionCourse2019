import numpy as np

def compute_geodesic(solver=None,manifold=None,p0=None,p1=None,init_curve=None,*args,**kwargs):
    varargin = compute_geodesic.varargin
    nargin = compute_geodesic.nargin

    p0=ravel(p0)
    p1=ravel(p1)
    D=numel(p0)

    bc = lambda a = None, b = None: boundary(a,b,p0,p1)
    odefun = lambda x = None, y = None: second2first_order(manifold,y)

    options=bvpset('Vectorized','on','RelTol',solver.tol,'NMax',solver.NMax)

    if (nargin < 5):
        x_init=linspace(0,1,10)
        y_init=lambda t=None: init_solution(p0,p1,t)
        init=bvpinit(x_init,y_init)
    else:
        x_init=linspace(0,1,10)
        y_init=lambda t=None: init_solution_given(init_curve,t)
        init=bvpinit(x_init,y_init)

    ## Try to solve the ODE
    tic
    try:
        #         solution = bvp4c(odefun, bc, init, options);
        solution=bvp5c(odefun,bc,init,options)
        if (isfield(solution,'stats') and isfield(solution.stats,'maxerr')):
            if (isfield(options,'RelTol') and isscalar(options.RelTol)):
                reltol=options.RelTol
            else:
                reltol=0.001
            failed=(solution.stats.maxerr > reltol)
        else:
            failed=copy(false)
    finally:
        pass

    solution.time_elapsed = copy(toc)

    if (failed):
        curve=lambda t=None: evaluate_failed_solution(p0,p1,t)
        logmap=(p1 - p0)
    else:
        curve=lambda t=None: evaluate_solution(solution,t,1)
        logmap=solution.y(arange((D + 1),end()),1)

    if (nargout > 1):
        len_=curve_length(manifold,curve)
        logmap=dot(len_,logmap) / norm(logmap)

    return curve,logmap,len_,failed,solution

def boundary(p0=None,p1=None,p0_goal=None,p1_goal=None,*args,**kwargs):
    varargin = boundary.varargin
    nargin = boundary.nargin

    D=numel(p0_goal)
    d1=p0(arange(1,D)) - ravel(p0_goal)
    d2=p1(arange(1,D)) - ravel(p1_goal)
    bc=concat([[d1],[d2]])
    return bc

def evaluate_failed_solution(p0=None,p1=None,t=None,*args,**kwargs):
    varargin = evaluate_failed_solution.varargin
    nargin = evaluate_failed_solution.nargin

    t=ravel(t)
    c=dot((1 - t),p0.T) + dot(t,p1.T)
    dc=repmat((p1 - p0).T,numel(t),1)

    c=c.T
    dc=dc.T
    return c,dc

def init_solution(p0=None,p1=None,t=None,*args,**kwargs):
    varargin = init_solution.varargin
    nargin = init_solution.nargin

    c,dc=evaluate_failed_solution(p0,p1,t,nargout=2)
    state=cat(1,c,dc)

    return state

def init_solution_given(solution=None,t=None,*args,**kwargs):
    varargin = init_solution_given.varargin
    nargin = init_solution_given.nargin

    c,dc=solution(t,nargout=2)
    state=cat(1,c,dc)

    return state

def geodesic_solver_bvp5c(options=None,*args,**kwargs):
    varargin = geodesic_solver_bvp5c.varargin
    nargin = geodesic_solver_bvp5c.nargin

    solver.options = copy(struct())
    if (isfield(options,'NMax')):
        solver.NMax = copy(options.NMax)
    else:
        solver.NMax = copy(1000)

    if (isfield(options,'tol')):
        solver.tol = copy(options.tol)
    else:
        solver.tol = copy(0.1)

    solver=class_(solver,'geodesic_solver_bvp5c')
    return solver
