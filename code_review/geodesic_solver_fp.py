import numpy as np

def compute_geodesic(solver=None,manifold=None,c0=None,c1=None,solution=None,*args,**kwargs):
    varargin = compute_geodesic.varargin
    nargin = compute_geodesic.nargin

    ## Get boundary points and ensure column vectors
    c0=ravel(c0)
    c1=ravel(c1)
    D=numel(c0)

    t_min=0
    t_max=1

    N=solver.N
    max_iter=solver.max_iter

    ell=solver.ell
    T=solver.T.T

    tol=solver.tol
    gp_kernel=solver.gp_kernel

    v=c1 - c0
    Veta=(dot((dot(dot(v.T,solver.Sdata),v)),solver.Sdata))

    # ----- The prior mean functions
    m=lambda t=None: c0.T + dot(t,(c1 - c0).T)
    dm=lambda t=None: dot(ones(size(t,1),1),(c1 - c0).T)
    ddm=lambda t=None: dot(zeros(size(t,1),1),(c1 - c0).T)

    # ----- The kernel components for the GP ( KernelMatrix = [A, C; C', B + R] )
    I_D=eye(D)
    R=lambda sigma2=None: blkdiag(sigma2[arange()],dot(I_D,1e-10),dot(I_D,1e-10))

    Ctrain=concat([[kdxddy(gp_kernel,T,T),kdxy(gp_kernel,T,concat([[t_min],[t_max]]))],[kxddy(gp_kernel,T,T),kxy(gp_kernel,T,concat([[t_min],[t_max]]))]])
    Btrain=concat([[kddxddy(gp_kernel,T,T),kddxy(gp_kernel,T,concat([[t_min],[t_max]]))],[kxddy(gp_kernel,concat([[t_min],[t_max]]),T),kxy(gp_kernel,concat([[t_min],[t_max]]),concat([[t_min],[t_max]]))]])

    y_hat=concat([[ddm(T)],[m(concat([[t_min],[t_max]]))]])
    y_obs=lambda ddc=None: concat([[ddc],[ravel(c0).T],[ravel(c1).T]])

    # ----- The initial guess for the parameters c'' we want to learn
    if (nargin < 5):
        DDC=ddm(T)
    else:
        # If the geodesic has been solved previously, we will initialize
        # the current parameters with the ones found before.
        if (logical_not(isempty(solution))):
            DDC=solution.ddc

    # ----- Initialize the noise of the GP
    sigma2_blkdiag=cell(N,1)

    for n in arange(1,N).reshape(-1):
        sigma2_blkdiag[n]=dot(solver.sigma2,I_D)

    # ----- The posterior mean of the dcurve & curve with size 2*(N*D) x 1
    Btrain_R_inv=(kron(Btrain,Veta) + R(sigma2_blkdiag))
    Btrain_R_inv=numpy.linalg.solve(Btrain_R_inv,eye(size(Btrain_R_inv)))
    kronCtrainVeta=kron(Ctrain,Veta)
    dmu_mu_post=lambda t=None,ddc=None: vec(concat([[dm(t)],[m(t)]]).T) + dot(kronCtrainVeta,(dot(Btrain_R_inv,vec((y_obs(ddc) - y_hat).T))))

    iter=1
    tic
    try:
        while (1):

            # ----- Compute current c, dc posterior
            dmu_mu_post_curr=reshape(dmu_mu_post(T,DDC),concat([D,dot(2,N)]))
            mu_post_curr=dmu_mu_post_curr(arange(),arange(N + 1,end()))
            dmu_post_curr=dmu_mu_post_curr(arange(),arange(1,N))
            # ----- Evaluate the c'' = f(c, c') the fixed-point
            DDC_new=geodesic_system(manifold,mu_post_curr,dmu_post_curr).T
            cost_current=(DDC - DDC_new) ** 2
            #             disp(['Iter: ',num2str(iter),', cost: ', num2str(sum(sum(cost_current)))]);
            condition_1=all(all((cost_current) < tol))
            condition_2=(iter > max_iter)
            if (condition_1 or condition_2):
                if (condition_1):
                    convergence_cond=1
                if (condition_2):
                    convergence_cond=2
                break
            # ----- Compute the "gradient"
            grad=DDC - DDC_new
            alpha=1
            for i in arange(1,3).reshape(-1):
                # Update the DDC temporary & check if update is good
                # DDC_temp = (1 - alpha) * DDC + alpha * DDC_new;
                DDC_temp=DDC - dot(alpha,grad)
                dmu_mu_post_curr_temp=reshape(dmu_mu_post(T,DDC_temp),concat([D,dot(2,N)]))
                mu_post_curr_temp=dmu_mu_post_curr_temp(arange(),arange(N + 1,end()))
                dmu_post_curr_temp=dmu_mu_post_curr_temp(arange(),arange(1,N))
                # If cost is reduced keep the step-size
                cost_temp=(DDC_temp - geodesic_system(manifold,mu_post_curr_temp,dmu_post_curr_temp).T) ** 2
                if (sum[sum(cost_temp)]=sum(sum(cost_current))
                    break
                else:
                    alpha=dot(alpha,0.33)
            # ----- Update the parameteres
            DDC=DDC - dot(alpha,grad)
            iter=iter + 1
    finally:
        pass

    ## Prepare the output.
    if (convergence_cond == 2 or convergence_cond == 3):
        disp('Geodesic solver (fp) failed!')
        curve=lambda t=None: evaluate_failed_solution(c0,c1,t)
        len_=curve_length(manifold,curve)
        logmap=(c1 - c0)
        logmap=dot(len_,logmap) / norm(logmap)
        failed=copy(true)
        solution=[]
        solution.time_elapsed = copy(toc)
    else:
        # This is a GP-posterior mean function, evaluated at new points t
        curve=lambda t=None: curve_eval_gp(t,T,t_min,t_max,Btrain,R(sigma2_blkdiag),Veta,dm,m,DDC,y_hat,y_obs,gp_kernel)
        len_=curve_length(manifold,curve)
        __,logmap=curve(0,nargout=2)
        logmap=dot(len_,logmap) / norm(logmap)
        failed=copy(false)
        solution=[]
        solution.ddc = copy(DDC)
        solution.Sigma2 = copy(sigma2_blkdiag)
        solution.total_iter = copy(iter)
        solution.cost = copy(cost_current)
        solution.ell = copy(ell)
        solution.T = copy(T)
        solution.time_elapsed = copy(toc)
        solution.cdc_spline = copy(lambda t=None: curve_eval_spline(T,mu_post_curr,dmu_post_curr,t))

    return curve,logmap,len_,failed,solution

# Computes the covariance function of a GP.
# function [diagVal, val] = var_post(A, B, R, C, Veta)
#     val = kron(A, Veta) - kron(C, Veta) * ((kron(B, Veta) + R) \ (kron(C, Veta)'));
#     val = 0.5 * (val + val'); # ensure symmetry!!
#     val = val + eye(size(val)) * 1e-10; # ensure PSD matrix
#     diagVal = diag(val);
# end # function

# Curve evaluation using spline model
def curve_eval_spline(T=None,c=None,dc=None,t=None,*args,**kwargs):
    varargin = curve_eval_spline.varargin
    nargin = curve_eval_spline.nargin

    D=size(c,2)
    N=size(ravel(t),1)
    c_t=zeros(N,D)
    dc_t=zeros(N,D)
    for d in arange(1,D).reshape(-1):
        c_t[arange(),d]=spline(T,c(arange(),d),t)
        dc_t[arange(),d]=spline(T,dc(arange(),d),t)

    return c_t,dc_t

# Curve evaluation for GP
def curve_eval_gp(Ts=None,T=None,t_min=None,t_max=None,Btrain=None,R=None,Veta=None,dm=None,m=None,DDC=None,y_hat=None,y_obs=None,gp_kernel=None,*args,**kwargs):
    varargin = curve_eval_gp.varargin
    nargin = curve_eval_gp.nargin

    Ts=ravel(Ts)
    D=size(DDC,2)
    Ns=numel(Ts)

    # ----- The kernel components
#     Atest = [kdxdy(gp_kernel, Ts, Ts), kdxy(gp_kernel, Ts, Ts); ...
#             kxdy(gp_kernel, Ts, Ts), kxy(gp_kernel, Ts, Ts)]; # 2Ns x 2Ns

    Ctest=concat([[kdxddy(gp_kernel,Ts,T),kdxy(gp_kernel,Ts,concat([[t_min],[t_max]]))],[kxddy(gp_kernel,Ts,T),kxy(gp_kernel,Ts,concat([[t_min],[t_max]]))]])

    # ----- Evaluate the c, dc posterior for the t_star.
    dmu_mu_Ts=vec(concat([[dm(Ts)],[m(Ts)]]).T) + dot(kron(Ctest,Veta),(numpy.linalg.solve((kron(Btrain,Veta) + R),vec((y_obs(DDC) - y_hat).T))))
    dmu_mu_Ts=reshape(dmu_mu_Ts,concat([D,dot(2,Ns)]))

    # The values to return
    c_t=dmu_mu_Ts(arange(),arange(Ns + 1,end()))
    dc_t=dmu_mu_Ts(arange(),arange(1,Ns))

    #     # If we need the variance around the posterior
#     if nargout > 2
#         [~, var_post_curr_Ts] = var_post(Atest, Btrain, R, Ctest, Veta);
#         var_c = var_post_curr_Ts((2 * Ns) + 1:end, (2 * Ns) + 1:end);
#         var_dc = var_post_curr_Ts(1:(2 * Ns), 1:(2 * Ns));
#     end

    return c_t,dc_t

def evaluate_failed_solution(p0=None,p1=None,t=None,*args,**kwargs):
    varargin = evaluate_failed_solution.varargin
    nargin = evaluate_failed_solution.nargin

    t=ravel(t)
    c=dot((1 - t),p0.T) + dot(t,p1.T)
    dc=repmat((p1 - p0).T,numel(t),1)

    c=c.T
    dc=dc.T
    return c,dc

# This function constructs an object from the specified solver.
#
# Options:
#   N        - number of mesh points [0,1] including boundaries.
#   tol      - the tolerance of the algorithm.
#   ell      - the length scale.
#   max_iter - the maximum iterations.
#   Sdata    - the amplitude of the kernel.
#   kernel   - the kernel for the GP.
#   sigma    - the std of the noise for the GP parameters.
def geodesic_solver_fp(D=None,options=None,*args,**kwargs):
    varargin = geodesic_solver_fp.varargin
    nargin = geodesic_solver_fp.nargin

    # If the dimensionality is not given the solver does not work.
    if (nargin == 0):
        error('Solver dimensionality has not been set...')

    # If the default parameters to be used.
    if (nargin == 1):
        options=[]

    # Initialize the solver options
    solver.options = copy(struct())
    if (isfield(options,'N')):
        solver.N = copy(options.N)
    else:
        solver.N = copy(10)

    if (isfield(options,'tol')):
        solver.tol = copy(options.tol)
    else:
        solver.tol = copy(0.1)

    # The mesh size
    T=linspace(0,1,solver.N)
    solver.T = copy(T)
    if (isfield(options,'ell')):
        solver.ell = copy(options.ell)
    else:
        solver.ell = copy(sqrt(dot(0.5,(T(2) - T(1)))))


    if (isfield(options,'max_iter')):
        solver.max_iter = copy(options.max_iter)
    else:
        solver.max_iter = copy(1000)

    if (isfield(options,'Sdata')):
        solver.Sdata = copy(options.Sdata)
    else:
        warning('Kernel amplitude has not been specified!')
        solver.Sdata = copy(eye(D))

    if (isfield(options,'kernel_type')):
        error('Kernel type is not given... ')
    else:
        warning('Kernel type has not been specified (default: Squared Exponential).')
        solver.gp_kernel = copy(se_kernel(solver.ell,1))

    # This is the fixed noise of the parameters that we learn during training.
    if (isfield(options,'sigma')):
        solver.sigma2 = copy(options.sigma ** 2)
    else:
        solver.sigma2 = copy((0.0001) ** 2)

    # Initialize the object
    solver=class_(solver,'geodesic_solver_fp')

    return solver
