function [x,flag,relres,iter,resvec,resveccg] = minres_for_IP(funAx,b,tol,maxit,funPrec,M2,x0,varargin)

% From Mathwork code updated by MK for POEMA and Interior Point algorithm

%MINRES   Minimum Residual Method.
%   Copyright 1984-2017 The MathWorks, Inc.

if (nargin < 2)
    error(message('MATLAB:minres:NotEnoughInputs'));
end

ma = size(b,1);
n = ma;

% Assign default values to unspecified parameters
if (nargin < 3) || isempty(tol)
    tol = 1e-6;
end
warned = 0;
if tol <= eps
    warning(message('MATLAB:minres:tooSmallTolerance'));
    warned = 1;
    tol = eps;
elseif tol >= 1
    warning(message('MATLAB:minres:tooBigTolerance'));
    warned = 1;
    tol = 1-eps;
end
if (nargin < 4) || isempty(maxit)
    maxit = min(n,20);
end
maxit = max(maxit, 1);

% Check for all zero right hand side vector => all zero solution
n2b = norm(b);                     % Norm of rhs vector, b
if (n2b == 0)                      % if    rhs vector is all zeros
    x = zeros(n,1);                % then  solution is all zeros
    flag = 0;                      % a valid solution has been obtained
    relres = 0;                    % the relative residual is actually 0/0
    iter = 0;                      % no iterations need be performed
    resvec = 0;                    % resvec(1) = norm(b-A*x) = norm(0)
    if nargout >= 6	               % resveccg(1) = norm(b-A*xcg) = norm(0)
        resveccg = 0;
    end
    if (nargout < 2)
        itermsg('minres',tol,maxit,0,flag,iter,NaN);
    end
    return
end

if ((nargin >= 7) && ~isempty(x0))
    if ~isequal(size(x0),[n,1])
        error(message('MATLAB:minres:WrongInitGuessSize', n));
    else
        x = x0;
    end
else
    x = zeros(n,1);
end

if ((nargin > 7) && strcmp(atype,'matrix') && ...
        strcmp(m1type,'matrix') && strcmp(m2type,'matrix'))
    error(message('MATLAB:minres:TooManyInputs'));
end

% Set up for the method
flag = 1;
iter = 0;
xmin = x;                          % Iterate which has minimal residual so far
imin = 0;                          % Iteration at which xmin was computed
tolb = tol * n2b;                  % Relative tolerance
%r = b - iterapp('mtimes',afun,atype,afcnstr,x,varargin{:});
r = b - funAx(x);
normr = norm(r);                   % Norm of residual

if (normr <= tolb)                 % Initial guess is a good enough solution
    flag = 0;
    relres = normr / n2b;
    resvec = normr;
    if nargout >= 6
        resveccg = normr;
    end
    if (nargout < 2)
        itermsg('minres',tol,maxit,0,flag,iter,relres);
    end
    return
end

resvec = zeros(maxit+1,1);         % Preallocate vector for MINRES residuals
resvec(1) = normr;                 % resvec(1) = norm(b-A*x0)
if nargout >= 6
    resveccg = zeros(maxit+2,1);   % Preallocate vector for CG residuals
    resveccg(1) = normr;           % resveccg(1) = norm(b-A*x0)
end
normrmin = normr;                  % Norm of minimum residual

vold = r;
u = funPrec(vold);
if ~all(isfinite(u))
    flag = 2;
    relres = normr / n2b;
    resvec = resvec(1);
    if nargout >= 6, resveccg = resveccg(1);    end
    if nargout < 2
        itermsg('minres',tol,maxit,0,flag,iter,relres);
    end
    return
end

v = u;
beta1 = vold' * v;
if (beta1 <= 0)
    flag = 5;
    relres = normr / n2b;
    resvec = resvec(1);
    if nargout >= 6, resveccg = resveccg(1); end
    if nargout < 2
        itermsg('minres',tol,maxit,0,flag,iter,relres);
    end
    return
end
beta1 = sqrt(beta1);
snprod = beta1;
vv = v / beta1;
% v = iterapp('mtimes',afun,atype,afcnstr,vv,varargin{:});
v = funAx(vv);
Amvv = v;
alpha = vv' * v;
v = v - (alpha/beta1) * vold;

% Local reorthogonalization
numer = vv' * v;
denom = vv' * vv;
v = v - (numer/denom) * vv;
volder = vold;
vold = v;

u = funPrec(vold);
if ~all(isfinite(u))
    flag = 2;
    relres = normr / n2b;
    resvec = resvec(1);
    if nargout >= 6, resveccg = resveccg(1);    end
    if nargout < 2
        itermsg('minres',tol,maxit,0,flag,iter,relres);
    end
    return
end
v = u;
betaold = beta1;
beta = vold' * v;
if (beta < 0)
    flag = 5;
    relres = normr / n2b;
    resvec = resvec(1);
    if nargout >= 6, resveccg = resveccg(1);    end
    if nargout < 2
        itermsg('minres',tol,maxit,0,flag,iter,relres);
    end
    return
end
iter = 1;
beta = sqrt(beta);
gammabar = alpha;
epsilon = 0;
deltabar = beta;
gamma = sqrt(gammabar^2 + beta^2);
mold = zeros(n,1);
Amold = mold;
m = vv / gamma;
Am = Amvv / gamma;
cs = gammabar / gamma;
sn = beta / gamma;
x = x + snprod * cs * m;
snprodold = snprod;
snprod = snprod * sn;

% This recurrence produces CG iterates.
% Enable the following statement to see xcg.
%xcg = x + snprod * (sn/cs) * m;

r = r - snprodold * cs * Am;
normr = norm(r);
resvec(2,1) = normr;
if nargout >= 6
    if (cs == 0)
        % It's possible that this cs value is zero (CG iterate does not exist)
        normrcg = Inf;
    else
        rcg = r - snprod*(sn/cs)*Am;
        normrcg = norm(rcg);
    end
    resveccg(2,1) = normrcg;
end

% Check for convergence after first step.
if normr <= tolb
    flag = 0;
    relres = normr / n2b;
    resvec = resvec(1:2);
    if nargout >= 6,    resveccg = resveccg(1:2);    end
    if (nargout < 2)
        itermsg('minres',tol,maxit,1,flag,iter,relres);
    end
    return
end

stag = 0;                          % stagnation of the method
moresteps = 0;
maxmsteps = min([floor(n/50),5,n-maxit]);
maxmsteps = maxit/1;
maxstagsteps = 3;

% loop over maxit iterations (unless convergence or failure)

for ii = 2 : maxit

    vv = v * (1/beta);
    %     v = iterapp('mtimes',afun,atype,afcnstr,vv,varargin{:});
    v = funAx(vv);
    Amolder = Amold;
    Amold = Am;
    Am = v;
    v = v - (beta / betaold) * volder;
    alpha = vv' * v;
    v = v - (alpha / beta) * vold;
    volder = vold;
    vold = v;
    u = funPrec(vold);
    if ~all(isfinite(u))
        flag = 2;
        break
    end
    v = u;
    betaold = beta;
    beta = vold' * v;
    if (beta < 0)
        flag = 5;
        break
    end
    beta = sqrt(beta);
    delta = cs * deltabar + sn * alpha;
    molder = mold;
    mold = m;
    m = vv - delta * mold - epsilon * molder;
    Am = Am - delta * Amold - epsilon * Amolder;
    gammabar = sn * deltabar - cs * alpha;
    epsilon = sn * beta;
    deltabar = - cs * beta;
    gamma = sqrt(gammabar^2 + beta^2);
    m = m / gamma;
    Am = Am / gamma;
    cs = gammabar / gamma;
    sn = beta / gamma;
    % Check for stagnation of the method
    if (snprod*cs == 0) || (abs(snprod*cs)*norm(m) < eps*norm(x))
        % increment the number of consecutive iterates which are the same
        stag = stag + 1;
    else
        stag = 0;
    end
    x = x + (snprod * cs) * m;
    snprodold = snprod;
    snprod = snprod * sn;
    % This recurrence produces CG iterates.
    % Enable the following statement to see xcg.
    %xcg = x + snprod * (sn/cs) * m;

    normr = abs(snprod);
    resvec(ii+1,1) = normr;
    if nargout >= 6
        % It's possible that this cs value is zero (CG iterate does not exist).
        if (cs == 0)
            normrcg = Inf;
        else
            rcg = r - snprod*(sn/cs)*Am;
            normrcg = norm(rcg);
        end
        resveccg(ii+2,1) = normrcg;
    end

    % check for convergence
    if (normr <= tolb || stag >= maxstagsteps || moresteps)
        % double check residual norm is less than tolerance
        %         r = b - iterapp('mtimes',afun,atype,afcnstr,x,varargin{:});
        r = b - funAx(x);
        normr = norm(r);
        resvec(ii+1,1) = normr;
        if (normr <= tolb)
            flag = 0;
            iter = ii;
            break
        else
            if stag >= maxstagsteps && moresteps == 0
                stag = 0;
            end
            moresteps = moresteps + 1;
            if moresteps >= maxmsteps
                if ~warned
                    warning(message('MATLAB:minres:tooSmallTolerance'));
                end
                flag = 3;
                iter = ii;
                break;
            end
        end
    end

    if (normr < normrmin)      % update minimal norm quantities
        normrmin = normr;
        xmin = x;
        imin = ii;
    end

    if (stag >= maxstagsteps)      % 3 iterates are the same
        flag = 3;
        break
    end
end                                % for ii = 1 : maxit
if isempty(ii)
    ii = 1;
end

% returned solution is first with minimal residual
if (flag == 0)
    relres = normr / n2b;
else
    %     r_comp = b - iterapp('mtimes',afun,atype,afcnstr,xmin,varargin{:});
    r_comp = b - funAx(xmin);
    if norm(r_comp) <= normr
        x = xmin;
        iter = imin;
        relres = norm(r_comp) / n2b;
    else
        iter = ii;
        relres = normr / n2b;
    end
end

% truncate the zeros from resvec
if ((flag <= 1) || (flag == 3))
    resvec = resvec(1:ii+1);
    if nargout >= 6,    resveccg = resveccg(1:ii+2);    end
else
    resvec = resvec(1:ii);
    if nargout >= 6,    resveccg = resveccg(1:ii+1);      end
end

% only display a message if the output flag is not used
if (nargout < 2)
    itermsg('minres',tol,maxit,ii,flag,iter,relres);
end
