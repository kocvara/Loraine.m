function [y,X,S,X_lin,S_lin] = loraine(d,options)
%
% Version ncluding L1 penalty for lin equality constraints and rank-one
% data recognition.
%
% TBD: Treating a mix of linear equality and inequality contraints, check
% that it handles the general situation of mixing equalities and
% inequalities
%
% Interior point method for low-rank linear SDP
%
% max_X tr(B,X)
% s.t.
% tr(A_i,X) = c_i, i=1,...,n
% X >=0 (positive semidefinite)
%
% where the solution X^* is assumed very low rank.
%
% Version working with linear inequality and EQUALITY constraints (in the
% primal formulation); equality constraints treated by l1-penalty method.
%
% NT direction as in SDPT3
% Schur system solved by either directly by Cholesky factorization or by 
% the preconditioned CG method Low-rank preconditioners 
% by RY Zhang and J. Lavaei: "Modified interior-point method for 
% large-and-sparse low-rank semidefinite programs." 2017 IEEE 56th Annual 
% Conference on Decision and Control, with modifications by Soodeh Habibi, 
% Arefeh Kavand, Michal Kocvara and Michael Stingl (POEMA ITN)
%
% uses full vectorization (not the symmetric one)
%
% INPUT:
% d      ... SDP problem in POEMA structure format
% options.xxx where xxx stands for:
% kit    ... 0 for direct solver
%            1 for preconditioned CG
% tol_cg     ... stopping tolerance for the CG solver
% tol_cg_up  ... update of tol after every IP iteration
% tol_cg_min ... minimal stopping tolerance for the CG solver
% eDIMACS    ... epsilon for DIMACS error stopping criterion
% prec       ... preconditioner type:
%               ... 0 no preconditioner
%               ... 1 Schur complement preconditioner with SMW formula
%               ... 2 preconditioner with augmented system
%               ... 3 Schur complement preconditioner, direct inverse
%               ... 4 hybrid preconditioner, first 2 then 1
% erank  ... estimated upper ound on rank of the dual solution
% aamat  ... choice of A^TA or identity in the preconditioner
%        ... 0 A^TA
%        ... 1 diag(A^TA)
%        ... 2 identity
%        ... 3 = 1+2
% fig_ev ... 0 no figure
%        ... 1 figure with eigenvalues of H, H_til, etc in every iteration,
%              switches to preconditioner = 3 !!!
% mup    ... penalty parameter for linear equality constraints (1000)
% datarank ...  0 low-rank data used if in the input file
%          ... -1 input file contains rank-1 matrices A_i which
%                 are decomposed into vectors b_i s.t. A_i = b_i b_i^T

% Programmed by Soodeh Habibi and Michal Kocvara, University of Birmingham,
% and Michael Stingl, University of Erlangen, for H2020 ITN POEMA

if (nargin<2)
    kit = 1;
    tol_cg = 1.0e-2; tol_cg_up = 0.5; tol_cg_min = 1.0e-6;
    cg_type = 'minres';
    eDIMACS = 1.0e-5;
    preconditioner = 4;
    erank = 1;
    aamat = 2;
    fig_ev = 0;
    verb = 1;
    mup = 1000.0;
    datarank = 0;
    initpoint = 0;
else
    if isfield(options,'kit'), kit = options.kit; else, kit = 1; end
    if isfield(options,'tol_cg'), tol_cg = options.tol_cg; else, tol_cg = 1e-2; end
    if isfield(options,'tol_cg_up'), tol_cg_up = options.tol_cg_up; else, tol_cg_up = 0.5; end
    if isfield(options,'tol_cg_min'), tol_cg_min = options.tol_cg_min; else, tol_cg_min = 1e-6; end
    if isfield(options,'cg_type'), cg_type = options.cg_type; else, cg_type = 'minres'; end
    if isfield(options,'eDIMACS'), eDIMACS = options.eDIMACS; else, eDIMACS = 1e-5; end
    if isfield(options,'prec'), preconditioner = options.prec; else, preconditioner = 4; end
    if isfield(options,'erank'), erank = options.erank; else, erank = 1; end
    if isfield(options,'aamat'), aamat = options.aamat; else, aamat = 2; end
    if isfield(options,'fig_ev'), fig_ev = options.fig_ev; else, fig_ev = 0; end
    if isfield(options,'verb'), verb = options.verb; else, verb = 1; end
    if isfield(options,'mup'), mup = options.mup; else, mup = 1000; end
    if isfield(options,'datarank'), datarank = options.datarank; else, datarank = 0; end
    if isfield(options,'initpoint'), initpoint = options.initpoint; else, initpoint = 0; end
end

if fig_ev==1, preconditioner = 3; aamat = 0; end

IPtol = eDIMACS;   % stopping tolerance for IP algorithm based on DIMACS error

if verb > 0
    fprintf('\n *** Loraine v0.1, version with L1-penalty ***\n')
    fprintf(' *** Initialisation STARTS\n')
end

%% Data preparation
n = d.nvar; msizes = d.msizes; nlin = d.nlin; nlmi = d.nlmi; eqc = 0;

tic;

% LOW-RANK DATA CHECKING/PREPARING
% find rank of LMI data by checking existence and size of Avec
rank_lmi = zeros(nlmi,1);
if isfield(d,'Avec')==1
    for i=1:nlmi
        for j=1:n
            if nnz(d.Avec{i,j})>0
                rank_lmi(i) = size(d.Avec{i,j},2);
            end
        end
    end
end

for k=1:nlmi
    for i=1:n
        if rank_lmi(k) == 0
        elseif rank_lmi(k) == 1
            B{k}(:,i) = d.Avec{k,i+1}';
            A{k,i} = -B{k}(:,i)*(B{k}(:,i)');
        else
            for j=1:rank_lmi(k)
                B{k,j}(i,:) = d.Avec{k,i+1}(:,j)';
            end
        end      
    end
    if rank_lmi(k) == 1
        B{k} = B{k}';
    end
end
 
% covert rank-1 matrices A_i into vectors b, store in matrix B
if datarank == -1 && isfield(d,'Avec') == 0
    for k=1:nlmi
        m = msizes(k);
        B{k} = sparse(n,m);
        for i=1:n
            [ii,jj,vv] = find(d.A{k,i+1});
            bidx = unique(ii);
            if ~isempty(bidx)
                tmp = full(d.A{k,i+1}(bidx,bidx));
                [vtmp,utmp] = eig((tmp+tmp') ./ 2.0);
                bbb = sign(vtmp(:,end)) .* sqrt(diag(tmp));
                tmp2 = bbb*bbb';
                if norm(tmp-tmp2) > 5.0e-6
                    datarank = -2;
                    warning('data conversion problem, swith to datrank=0');
                    break
                end
                B{k}(i,bidx) = bbb;
            end
        end
        rank_lmi(k) = 1;
    end
end

if max(rank_lmi) && datarank ~= -2
    datarank = max(rank_lmi); 
end
if datarank == -2, datarank = 0; rank_lmi = zeros(nlmi,1); end
% END OF LOW-RANK DATA BLOCK

% LINEAR CONSTRAINTS CHECKING/PREPARING
if isfield(d,'lsi_op')
    key_lsi = 1;
else
    key_lsi = 0;
end

sumlin = 0;
if nlin > 0
    if key_lsi == 1
        sumlin = sum(d.lsi_op);
        eqc = nlin - sumlin;
        if sumlin < nlin
            gamma = zeros(eqc,1); d_lin = zeros(nlin,1); C_lin=[];
            ieq = 1; iin = 1;
            for i=1:d.nlin
                if d.lsi_op(i) == 0
                    indieq(ieq) = i;
                    gamma(ieq) = -d.d(i);
                    ieq = ieq + 1;
                else
                    C_lin(:,iin) = -d.C(:,i);
                    d_lin(iin) = -d.d(i);
                    iin = iin + 1;
                end
            end
            Gamma(:,1:ieq-1) = -d.C(:,indieq) ;
        else
            d_lin = -d.d(:);
            C_lin = -d.C;
        end
    else
        d_lin = -d.d(:);
        C_lin = -d.C;
    end
end

if nlin > 0
    if eqc > 0
        if sumlin > 0
            C_linex = sparse(n+eqc,nlin+eqc);
            C_linex(1:n,1:sumlin) = C_lin;
            C_linex(1:n,sumlin+1:sumlin+eqc) = -Gamma;
            C_linex(end-eqc+1:end, sumlin+1:sumlin+eqc) = -speye(eqc);
            C_linex(end-eqc+1:end,end-eqc+1:end) = -speye(eqc);
            
            C_lin = C_linex;
            d_lin = [d_lin(1:sumlin);-gamma;sparse(eqc,1)];
        else
            C_linex = sparse(n+eqc,2*eqc);
            C_linex(1:n,1:eqc) = -Gamma;
            C_linex(end-eqc+1:end, 1:eqc) = -speye(eqc);
            C_linex(end-eqc+1:end,end-eqc+1:end) = -speye(eqc);
            
            C_lin = C_linex;
            d_lin = [-gamma;sparse(eqc,1)];           
        end
    end
end

nlin = sumlin + 2*eqc;
% END OF LINEAR CONSTRAINTS BLOCK

% PREPARE DATA ARRAYS
msizesex = msizes;

for i=1:nlmi
    A0{i} = sparse(msizesex(i),msizesex(i));
    if rank_lmi(i) == 1  
        A0{i} = sparse(d.A{i,1});
    else
        A0{i}(1:msizes(i),1:msizes(i)) = d.A{i,1};
    end
    for j=1:n+eqc
        Ai{i,j} = sparse(msizesex(i),msizesex(i));
    end
    for j=1:n
        if rank_lmi(i) == 1   && datarank > 1
            Ai{i,j}(1:msizes(i),1:msizes(i)) = A{i,j};
        else
            Ai{i,j}(1:msizes(i),1:msizes(i)) = d.A{i,j+1};
        end
    end
end

% A = Ai;

ddnvar = d.nvar;
ddc = d.c;
clear d

for i=1:nlmi
    C{i} = -sparse(A0{i});

    nnzall = 0;
    nnzs = zeros(n,1);
    for j=1:n
        if ~isempty(Ai{i,j})
            nnzs(j) = nnz(Ai{i,j});
            nnzall = nnzall+nnzs(j);
        end
    end

    ii = zeros(nnzall,1);
    jj = zeros(nnzall,1);
    vv = zeros(nnzall,1);
    start = 1;

    for j=1:n
        if rank_lmi(i) == 0
            if ~isempty(Ai{i,j})
                [i_,j_,v_] = find(vec(Ai{i,j}));
                ii(start:start+nnzs(j)-1) = i_;
                jj(start:start+nnzs(j)-1) = j;
                vv(start:start+nnzs(j)-1) = -v_;
                start = start + nnzs(j);
            else
                error('Empty input data')
            end
        elseif rank_lmi(i) == 1
%             if kit==0
%                 A{i,j} = -B{i}(j,:)'*(B{i}(j,:));
%             end
            [i_,j_,v_] = find(vec(-B{i}(j,:)'*(B{i}(j,:))));
            ii(start:start+nnzs(j)-1) = i_;
            jj(start:start+nnzs(j)-1) = j;
            vv(start:start+nnzs(j)-1) = v_;
            start = start + nnzs(j);
        else
            error('Empty input data')
        end
    end

    AAA = sparse(ii,jj,vv,size(A0{i},1)*size(A0{i},2),n+eqc);
    AA{i} = AAA';
    if aamat<2 || aamat==3
        AAAAT{i} = AAA'*AAA;
    else
        AAAAT{i} = [];
    end

    vecC{i} = vec(C{i});
    normC(i) = sqrt(vecC{i}'*vecC{i});
end

if kit > 0
    clear Ai
end

b = -ddc(:);
if eqc > 0
    b = b - mup*sum(Gamma,2);
    for i = ddnvar+1 : ddnvar+eqc
        b(i) = -2*mup;
    end
end
% END OF DATA READING AND PREPARING

if verb > 0
    fprintf(' Number of variables: %5d\n',n)
    fprintf(' Matrix size(s)     :'), fprintf(' %5d',msizes), fprintf('\n')
    fprintf(' Linear constraints : %5d\n',nlin)
    if kit > 0
        fprintf(' Preconditioner     : %5d\n',preconditioner)
    else
        fprintf(' Preconditioner     :  none, using direct solver\n')
    end
    fprintf(' Expected rank      : %5d\n',erank)
end
tottime = toc;
if verb > 1
    fprintf(' CPU preparation    : %8.2f\n',tottime)
end
tic

n = n + eqc;

%% Initialization

%%%%%
if initpoint == 1
    % Variables initialization ala SDPT3
    if nlin > 0
        [X,S,y,X_lin,S_lin,S_lin_inv] = initial(msizes,nlmi,AA,b,C,nlin,C_lin,d_lin);
    else
        [X,S,y] = initial(msizes,nlmi,AA,b,C,nlin,[],[]);
    end
else
    % Simple variables initialization (better e.g. for tru/vib problems)
    for i=1:nlmi
        X{i} = 1.*eye(msizes(i));      %X{i} = 10.*eye(msizes(i));
        S{i} = n*1e0.*eye(msizes(i));  %S{i} = 10.*eye(msizes(i));
    end
    if nlin>0
        X_lin = 1.*ones(nlin,1); S_lin = 1.*ones(nlin,1); S_lin_inv = 1./S_lin;
    else
        X_lin = []; S_lin = [];
    end
    y = 1.*zeros(n,1); yold = y;
end
%%%%%

if nlin == 0
    C_lin=[]; X_lin=[]; S_lin=[]; S_lin_inv=[];
end

mu = 1.1;
for i=1:nlmi
    mu = mu + trace(X{i}*S{i})/msizes(i);
end
if nlin > 0
    mu = mu + trace(X_lin'*S_lin)/nlin;
end

sigma = 3;

%%%%%% These two parameters may influence convergence:
tau = 0.9;  %lower value, such as 0.9 leads to more iterations but more robust algo
expon = 3;
%%%%%%

Xo = X; So = S;
DIMACS_error = 1; iter = 0; cg_iter = 0; 
if eqc > 0
    eq_norm = 1; eq_count = 0;
else
    eq_norm = 0; eq_count = 0;
end

pencount = 0;

print_header(verb,eqc,kit)

%% the main iteration loop
TimeIni = tic;
while abs(DIMACS_error ) > IPtol || eq_norm > IPtol
    tic
    if iter > 100, break; end
    iter = iter + 1;
    
    mu = btrace(nlmi,X,S);
    if nlin>0
        mu = mu + trace(X_lin'*S_lin);
    end
    mu = mu/(sum(msizes)+nlin);
    
    for i=1:nlmi
        [Ltmp,flag] = chol(X{i},'lower');
        if flag > 0
            icount = 0;
            while flag > 0
                X{i} = X{i} + 1.0e-6 .* eye(size(X{i},1));
                [Ltmp,flag] = chol(X{i},'lower');
                icount = icount + 1;
                if icount > 100
                    error('X cannot be made positive definite')
                end
            end
        end
        L{i} = Ltmp;
        
        S{i} = (S{i}+S{i}') ./ 2.0;
        [Rtmp,flag] = chol(S{i},'lower');
        if flag > 0
            icount = 0;
            while flag > 0
                S{i} = S{i} + 1.0e-6 .* eye(size(S{i},1));
                [Rtmp,flag] = chol(S{i},'lower');
                icount = icount + 1;
                if icount>100
                    error('S cannot be made positive definite')
                end
            end
        end
        R{i} = Rtmp;
        [U,D{i},V] = svd(R{i}'*L{i});
        
        Di{i} = diag(1./(diag(D{i}))); 
        Di2 = diag(1./sqrt(diag(D{i})));
        G{i} = Ltmp*V*Di2; 
        Gi{i} = inv(G{i});
        W{i} = G{i}*G{i}';
        E{i} = my_kron(Gi{i},G{i}',S{i});
        
        Si{i} = inv(S{i});
        DD{i} = G{i}'*S{i}*G{i};
        DDs{i} = diag(sqrt(diag(DD{i}))); 
        DDsi{i} = diag(1./sqrt(diag(DD{i})));
    end
    if nlin>0
        Si_lin = 1./S_lin;
    end
    
    precond = preconditioner;
    arank = erank;
    
    %% predictor
    
    Rp = b;
    Rd{nlmi,1} = []; Rc{nlmi,1} = [];
    for i=1:nlmi
        Rp = Rp - AA{i}*vec(X{i});
        Rd{i} = C{i} - S{i} - mat(AA{i}'*y);
        Rc{i} = sigma*mu .* eye(msizesex(i)) - D{i}.^2;
    end
    if nlin > 0
        Rp = Rp - C_lin*X_lin;
        Rd_lin = d_lin - S_lin - C_lin'*y;
        Rc_lin = sigma*mu.*ones(nlin,1) - X_lin.*S_lin;
    end

    if kit == 0   % if direct solver, compute the Hessian matrix
        BBBB = zeros(n,n);
        for ilmi=1:nlmi
            if rank_lmi(ilmi) == 0
%                             Gilmi = G{ilmi};
%                             BB = zeros(size(Gilmi,1)^2,n);
%                             for i = 1:n
%                                 BB(:,i) = vec((Gilmi'*Ai{ilmi,i})*Gilmi);
%                             end
%                             BBBB = BBBB + BB'*BB;
                Hnn = zeros(n,n);
                Wilmi = W{ilmi};
                for i = 1:n
                    Fkok = -vec((Wilmi*Ai{ilmi,i})*Wilmi);
                    Hnn(i,:) = AA{ilmi}*Fkok;
                end
                BBBB = BBBB + Hnn;
            elseif rank_lmi(ilmi) == 1
                Gilmi = G{ilmi};
                BB = B{ilmi}*Gilmi;
                BBBB = BBBB + (BB*BB').^2;
            else   % this branch is very slow, ZZ big and dense
                Gilmi = G{ilmi};
                for i=1:n
                    for j=1:rank_lmi(ilmi)
                        Z(:,j) = Gilmi'*B{ilmi,j}(i,:)';
                    end
                    ZZtmp = Z*Z';
                    ZZ(:,i) = ZZtmp(:);
                end
                BBBB = BBBB + ZZ'*ZZ;
            end
        end

        if nlin>0
            BBBB = BBBB + C_lin*spdiags(X_lin.*S_lin_inv,0,nlin,nlin)*C_lin';
        end
        if rank_lmi(ilmi) ~= 1
            BBBB = ((BBBB+BBBB')./2);
        end
    end

    h = Rp;  % RHS for the Hessian equation
    for i=1:nlmi
        h = h + AA{i}*my_kron(G{i},G{i},(G{i}'*Rd{i}*G{i} + D{i}));
    end
    if nlin>0
        h = h + C_lin*(spdiags(X_lin.*Si_lin,0,nlin,nlin)*Rd_lin + X_lin);
    end

    % solving the linear system
    if kit == 0   % direct solver
        %dely = (BBBB)\h;
%         [Rchol,flag,Pchol] = chol(BBBB); Pchol = speye(size(BBBB));
        [Rchol,flag] = chol(BBBB); Pchol = speye(size(BBBB));
        if flag > 0
            icount = 0;
            while flag > 0
                BBBB = BBBB + 1.0e-4.*speye(size(BBBB,1));
                [Rchol,flag] = chol(BBBB);
                icount = icount + 1;
                if icount>100
                    error('System matrix cannot be made positive definite')
                end
            end
        end
        dely = Pchol*(Rchol\(Rchol'\(Pchol'*h)));
        cg_iter1 = 0;
    else   % iterative solver
        switch preconditioner
            case 0
                % matrix and function for no preconditioner
                M1x = @(x)(No_Prec(x));
                maxcgit = 10000;
            case 1
                % matrix and function for U_tilda preconditioner, SMV formula
                [Umat,Zp,Lp,AAAATtau,Rchol,Pchol] = ...
                    Prec_for_CG_tilS_prep(W,AA,AAAAT,arank,aamat,nlin,C_lin,X_lin,S_lin_inv);
                M1x = @(x)(Prec_for_CG_tilS(x,AA,AAAATtau,Rchol,Pchol,Umat,Zp,Lp));
                maxcgit = 1000;
            case 2
                % matrix and function for U_hat preconditioner
                [Prec] = Prec_for_CG_aug_prep(W,AA,AAAAT,arank,aamat,nlin,C_lin,X_lin,S_lin_inv);
                M1x = @(x)(Prec_for_CG_aug(x,Prec));
                maxcgit = 10000;
            case 3
                % matrix and function for U_tilda preconditioner
                [Prec] = Prec_for_CG_til_prep(W,AA,AAAAT,arank,aamat,nlin,C_lin,X_lin,S_lin,fig_ev);
                M1x = @(x)(Prec_for_CG_til(x,Prec));
                maxcgit = 1000;
            case 4
                % matrix and function for U_hat precond.-hybrid, stage 1
                [Prec] = Prec_for_CG_aug_prep(W,AA,AAAAT,arank,aamat,nlin,C_lin,X_lin,S_lin_inv);
                M1x = @(x)(Prec_for_CG_aug(x,Prec));
                maxcgit = 10000;
        end
        
        Ax = @(x)(Ax_for_CG(x,W,AA,nlin,C_lin,X_lin,S_lin_inv)); % matrix vector multiplication function

        % calling the solver
        switch cg_type
            case 'cg'
                [dely, ~ , cg_iter1] = cg_for_IP(Ax, zeros(n,1), h, M1x, maxcgit, tol_cg);
            case 'minres'
                %[dely, ~ , cg_iter1] = minres_for_IP(Ax, zeros(n,1), h, M1x, 1000, tol_cg);
                [dely,flag,~,cg_iter1] = minres_for_IP(Ax, h, tol_cg, maxcgit,M1x);
                if flag > 0
                    warning('MINRES raised flag:',flag)
                end
            otherwise
                warning('Selected iterative solver not implemented or does not exist')
        end
    end
    cg_iter = cg_iter + cg_iter1;
    
    for i=1:nlmi
        delS{i} = Rd{i} - mat(AA{i}'*dely);
        delX{i} = mat(vec(-X{i}) - my_kron(W{i},W{i},delS{i}));
    end
    if nlin > 0
        delS_lin = Rd_lin - C_lin'*dely;
        delX_lin = -X_lin - (X_lin).*(Si_lin).*delS_lin;
    end
    
    % determining steplength to stay feasible
    for i=1:nlmi
        delSb = G{i}'*delS{i}*G{i}; delXb = Gi{i}*delX{i}*Gi{i}';
        
        XXX = DDsi{i}*delXb*DDsi{i}; XXX = (XXX+XXX')./2;
        mimiX = min(eig(XXX));
        if mimiX > -1.0e-6
            alpha(i) = .99;
        else
            alpha(i) = min(1.0,-tau/mimiX);
        end
        XXX = DDsi{i}*delSb*DDsi{i}; XXX = (XXX+XXX')./2;
        mimiS = min(eig(XXX));
        if mimiS > -1.0e-6
            beta(i) = .99;
        else
            beta(i) = min(1.0,-tau/mimiS);
        end
        
        % solution update
        Xn{i} = X{i} + alpha(i).*delX{i};
        Sn{i} = S{i} + beta(i).*delS{i};
    end
    if nlin > 0
        mimiX_lin = min(delX_lin./X_lin);
        if mimiX_lin > -1.0e-6
            alpha_lin = .99;
        else
            alpha_lin = min(1.0,-tau/mimiX_lin);
        end
        mimiS_lin = min(delS_lin./S_lin);
        if mimiS_lin > -1.0e-6
            beta_lin = .99;
        else
            beta_lin = min(1.0,-tau/mimiS_lin);
        end
        
        % solution update
        Xn_lin = X_lin + alpha_lin.*delX_lin;
        Sn_lin = S_lin + beta_lin.*delS_lin;
    else
        alpha_lin = 1.0; beta_lin = 1.0;
    end
    
    % sigma update
    step_pred = min(min([alpha,alpha_lin]),min([beta,beta_lin]));
    if (mu > 1.0e-6)
        if (step_pred < 1.0/sqrt(3.0));
            expon_used = 1.0;
        else
            expon_used = max(expon,3*step_pred^2);
        end
    else
        expon_used = max(1,min(expon,3*step_pred^2));
    end
    if btrace(nlmi,Xn,Sn) < 0
        sigma = 0.8;
    else
        expon_used;
        tmp1 = btrace(nlmi,Xn,Sn);
        if nlin > 0, tmp2 = Xn_lin'*Sn_lin; else tmp2 = 0.0; end
        tmp12 = (tmp1+tmp2)/(sum(msizes)+nlin);
        sigma = min(1,((tmp12)/mu)^expon_used);
    end
    
    %% corrector
    
    h = Rp; %RHS for the linear system
    for i=1:nlmi
        diD = diag(D{i}); deed = diD*ones(1,msizes(i)) + ones(msizes(i),1)*diD';
        RNT{i} = -(Gi{i}*(delX{i}*delS{i})*G{i} + G{i}'*(delS{i}*delX{i})*Gi{i}')./deed;
        h = h + AA{i}*my_kron(G{i},G{i},(G{i}'*Rd{i}*G{i} + D{i} - (sigma*mu).*Di{i}  - RNT{i}));         % RHS using my_kron
    end
    if nlin > 0
        tmp = (delX_lin.*delS_lin) .* Si_lin - (sigma*mu) .* Si_lin;
        h = h + C_lin*(spdiags(X_lin.*Si_lin,0,nlin,nlin)*Rd_lin + X_lin + tmp);
    end
    
    % solving the linear system
    if kit == 0   % direct solver
        dely = Pchol*(Rchol\(Rchol'\(Pchol'*h)));
        cg_iter2 = 0;
    else   % iterative solver
        switch cg_type
            case 'cg'
                [dely, ~ , cg_iter2] = cg_for_IP(Ax, zeros(n,1), h, M1x, maxcgit, tol_cg);
            case 'minres'
                [dely,~,~,cg_iter2] = minres_for_IP(Ax, h, tol_cg, maxcgit, M1x);
            otherwise
                warning('Selected iterative solver not implemented or does not exist')
        end
    end
    cg_iter = cg_iter + cg_iter2;
    
    for i=1:nlmi
        delS{i} = Rd{i} - mat(AA{i}'*dely);
        delX{i} = mat(vec((sigma*mu).*Si{i}-X{i}) + my_kron(G{i},G{i},RNT{i}) - my_kron(W{i},W{i},delS{i}));
        delX{i} = (delX{i}+delX{i}') ./ 2.0;
    end
    if nlin > 0
        RNT_lin = -(delX_lin.*delS_lin).*(Si_lin);
        delS_lin = Rd_lin - C_lin'*dely;
        delX_lin = -X_lin - (X_lin).*(Si_lin).*delS_lin + (sigma*mu).*(Si_lin) + RNT_lin;
    end;
    
    for i=1:nlmi
        % determining steplength to stay feasible
        delSb = G{i}'*delS{i}*G{i}; delXb = Gi{i}*delX{i}*Gi{i}';
        delXb = (delXb+delXb') ./ 2.0;
        
        XXX = DDsi{i}*delXb*DDsi{i}; XXX = (XXX+XXX') ./ 2.0;
        mimiX = min(eig(XXX));
        if mimiX > -1.0e-6
            alpha(i) = .99;
        else
            alpha(i) = min(1.0,-tau/mimiX);
        end
        XXX = DDsi{i}*delSb*DDsi{i}; XXX = (XXX+XXX') ./ 2.0;
        mimiS = min(eig(XXX));
        if mimiS > -1.0e-6
            beta(i) = .99;
        else
            beta(i) = min(1.0,-tau/mimiS);
        end
    end
    
    if nlin>0
        mimiX_lin = min(delX_lin./X_lin);
        if mimiX_lin > -1.0e-6
            alpha_lin = .99;
        else
            alpha_lin = min(1.0,-tau/mimiX_lin);
        end
        mimiS_lin = min(delS_lin./S_lin);
        if mimiS_lin > -1.0e-6
            beta_lin = .99;
        else
            beta_lin = min(1.0,-tau/mimiS_lin);
        end
    else
        alpha_lin = 1.0; beta_lin = 1.0;
    end
    
    % solution update
    yold = y;
    y = y + min([beta,beta_lin])*dely;
    for i=1:nlmi
        X{i} = X{i} + min([alpha,alpha_lin]).*delX{i}; X{i} = (X{i}'+X{i})./2;
        S{i} = S{i} + min([beta,beta_lin]).*delS{i};
    end
    if nlin>0
        X_lin = X_lin + min([alpha,alpha_lin]) .* delX_lin;
        S_lin = S_lin + min([beta,beta_lin]) .* delS_lin;
    end
    S_lin_inv = 1.0 ./ S_lin;
    
    tol_cg = max(tol_cg*tol_cg_up,tol_cg_min); % update tol for CG

    %% DIMACS error evaluation
    err1 = norm(Rp)/(1+norm(b));
    
    [err2,err3,err4,err5,err6] = deal(0.0,0.0,0.0,0.0,0.0);
    for i=1:nlmi
        err2 = err2 + max(0.0,-min(eig(X{i}))/(1+norm(b)));
        err3 = err3 + norm(Rd{i},'fro')/(1+normC(i));
        err4 = err4 + max(0.0,-min(eig(S{i}))/(1+normC(i)));
        err6 = err6 + ((vec(S{i}))'*vec(X{i}))/(1+abs(vecC{i}'*vec(X{i}))+abs(b'*y));
    end
    err5 = (btrace(nlmi,C,X)-b'*y)/(1+abs(btrace(nlmi,C,X))+abs(b'*y));
    if nlin>0
        err2 = err2 + max(0.0,-min(X_lin)/(1+norm(b)));
        err3 = err3 + norm(Rd_lin)/(1+norm(d_lin));
        err4 = err4 + max(0.0,-min(S_lin)/(1+norm(d_lin)));
        err5 = (btrace(nlmi,C,X) + d_lin'*X_lin - b'*y)/(1+abs(btrace(nlmi,C,X))+abs(b'*y));
        err6 = err6 + (S_lin'*X_lin)/(1+abs(d_lin'*X_lin)+abs(b'*y));
    end
    DIMACS_error = err1 + err2 + err3 + err4 + abs(err5) + err6;

    if eqc > 0
        eq_norm1 = norm(abs(Gamma'*y(1:end-eqc)-gamma));
        if eq_norm1 > eq_norm, eq_count = eq_count+1; end
        eq_norm = eq_norm1;
        if eq_count > 10
            error('eq_norm not decreasing, STOP')
            break
        end
    end
    
    titi = toc;
    tottime = tottime + titi;

    % print output
    print_progress(verb,eqc,kit,iter, y(1:ddnvar)'*ddc(:), ...
        DIMACS_error,err1,err2,err3,err4,err5,err6,eq_norm,cg_iter1,cg_iter2,titi)
    
    % hybrid preconditioner 
    if preconditioner==4
%         if (cg_iter2>erank*nlmi*sqrt(n)/1 && iter>sqrt(n)/60)||cg_iter2>100 %for SNL problems
        if (cg_iter2>erank*nlmi*sqrt(n)/10 && iter>sqrt(n)/60)||cg_iter2>40
            preconditioner = 1; aamat = 2; clear Prec
            if verb > 0
                fprintf(' Switching to preconditioner 1\n')
            end
        end
    end
    
%     % l1-penalty update (does not really work)
%     pencount = pencount + 1;
%     if DIMACS_error<eq_norm/10 && pencount > 5
%         mup = min(mup*10.0,10e9) %mu for penalty approach
%         b= -ddc(:);
%         if eqc>0
%             for i=1:eqc
%                 b = b - mup * (Gamma(:,i));
%             end
%             for i = ddnvar+1:ddnvar+eqc
%                 b(i) = -mup;
%             end
%         end
%         pencount = 0;
%     end
    
end

if verb > 0
    fprintf('*** Total CG iterations: %8.0d\n', cg_iter)
    fprintf('*** Total CPU time: %13.2f seconds\n', tottime)
end
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   SUPPORTING FUNCTIONS Linear Algebra
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function D = my_kron(A,B,C)
D = vec(B*(C*A'));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function vecA = vec(A)
vecA = A(:);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function A = mat(vecA)
n = sqrt(length(vecA));
A = reshape(vecA,n,n);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function trXS = btrace(nlmi,X,S)
% compute sum of traces of products of block matrices

trXS = 0;
for i = 1:nlmi
    trXS = trXS + sum(sum(X{i}.*S{i}));
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   SUPPORTING FUNCTIONS Preconditioner
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Ax = Ax_for_CG(x,W,AA,nlin,C_lin,X_lin,S_lin_inv)
% system matrix times a vector
nlmi = size(AA,2);
m = size(AA{1},1);
Ax = zeros(m,1);
for ilmi = 1:nlmi
    ax = AA{ilmi}'*x;
    waxw = vec(W{ilmi}*mat(ax)*W{ilmi});
    Ax = Ax + AA{ilmi}*waxw;
end
if nlin > 0
    Ax = Ax + C_lin*( (X_lin.*S_lin_inv) .* (C_lin'*x) );
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Prec] = Prec_for_CG_aug_prep(W,AA,AAAAT,kkk,aamat,nlin,C_lin,X_lin,S_lin_inv)
% prepare the matrix "Prec" for diagoanal preconditioner

nlmi = size(AA,2);
kk = kkk.*ones(nlmi,1);  %kk(1)=9;
nvar = size(AA{1},1);
AAAATtau = sparse(nvar,nvar);

ntot=0; for ilmi=1:nlmi, ntot = ntot+size(W{ilmi},1); end
sizeS=0; for ilmi=1:nlmi, sizeS = sizeS+kk(ilmi)*size(W{ilmi},1); end
S = zeros(sizeS,sizeS);

lbt = 1; lbs=1;
for ilmi = 1:nlmi
    n = size(W{ilmi},1);
    k = kk(ilmi);
    
    [vectf,lambdaf] = eig(W{ilmi}); dlambdaf = diag(lambdaf);
    [lambda1,iind] = maxk(dlambdaf,k); vect_l = vectf(:,iind);
    lambda_l = diag(lambda1);
    jind = setdiff(1:n,iind);
    [lambda1] = dlambdaf(jind); vect_s = vectf(:,jind);
    lambda_s = diag(lambda1);
%     ttau = .99*mean(lambda1); %ttau = 1.1*min(lambda1);
    ttau = (min(lambda1)+mean(lambda1))/2;
    
    Umat{ilmi} = sqrt(lambda_l-ttau.*speye(k));
    Umat{ilmi} = vect_l*Umat{ilmi};
    m = size(Umat{ilmi},1);
    V{ilmi} = sqrt(ttau).*AA{ilmi}*kron(Umat{ilmi},speye(m));
    
    switch aamat
        case 0
            ZZZ = AAAAT{ilmi};
        case 1
            ZZZ=spdiags(diag(AAAAT{ilmi}),0,nvar,nvar);
        case 2
            ZZZ=spdiags(ones(nvar,1),0,nvar,nvar);
        case 3
            ZZZ=sparse(nvar,nvar);
    end
    AAAATtau = AAAATtau + ttau^2.*ZZZ;
end

if nlin>0
    AAAATtau = AAAATtau + C_lin*spdiags(X_lin.*S_lin_inv,0,nlin,nlin)*C_lin';
end

% Prec.b12 = sparse([V{:}]);
Prec = sparse(AAAATtau);
% Prec.b22 = -speye(size(Prec.b12,2));

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function M1x = Prec_for_CG_aug(b,Prec)
% diagonal preconditioner
x = Prec\b;
M1x = x;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function M1x = No_Prec(x)
% no preconditioning
M1x = 1.*x;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Htil] = Prec_for_CG_til_prep(W,AA,AAAAT,kkk,aamat,nlin,C_lin,X_lin,S_lin,fig_ev)
% preconditioner U_tilda, prepare the matrix (without SMW formula, so far)

nlmi = size(AA,2);
kk=kkk.*ones(nlmi,1);  %kk(1)=10;

m = size(AA{1},1);
Htil = zeros(m,m);
if(fig_ev == 1)
    Htila = zeros(m,m);
    HHHH = zeros(m,m);
    HUHU = zeros(m,m);
end

if nlin>0
    Htil = Htil + C_lin*spdiags(X_lin./S_lin,0,nlin,nlin)*C_lin';
    if(fig_ev == 1)
        Htila = Htila + C_lin*spdiags(X_lin./S_lin,0,nlin,nlin)*C_lin';
        HHHH = HHHH + C_lin*spdiags(X_lin./S_lin,0,nlin,nlin)*C_lin';
        HUHU = HUHU + C_lin*spdiags(X_lin./S_lin,0,nlin,nlin)*C_lin';
    end
end

for ilmi = 1:nlmi
    n = size(W{ilmi},1);
    k = kk(ilmi);
    
    if k == 0   % This is for the linear constraints written as LMI
        
        if isdiag(W{ilmi})
            Htil = Htil + AA{ilmi}*kron(sparse(W{ilmi}),sparse(W{ilmi}))*AA{ilmi}';
        else
            Htil = Htil + AA{ilmi}*kron(W{ilmi},W{ilmi})*AA{ilmi}';
        end
        
    else
        
        [vectf,lambdaf] = eig(W{ilmi}); dlambdaf = diag(lambdaf);
        [lambda1,iind] = maxk(dlambdaf,k); vect_l = vectf(:,iind);
        lambda_l = diag(lambda1);
        jind = setdiff(1:n,iind);
        [lambda1] = dlambdaf(jind); vect_s = vectf(:,jind);
        lambda_s = diag(lambda1);
        ttau = .99.*mean(lambda1);
        ttau = (min(lambda1)+mean(lambda1))/2;   %ttau=1e-7;
%         ttau = 2*mean(lambda1);
        
        %     Umat = sqrt(lambda)-sqrt(ttau).*speye(k);
        Umat = sqrt(lambda_l-ttau.*speye(k));
        %Umat = sqrt(lambda_l-(min(ttau,0.9*lambda_l)).*speye(k));
        Umat = vect_l*Umat;
        %m = size(Umat,1);
        
        W0 = [vect_s vect_l]*[lambda_s sparse(n-k,k); sparse(k,n-k) ttau*speye(k)]...
            *[vect_s vect_l]';
        
        Z = chol(2.*W0 + Umat*Umat','lower');
        
        BB = zeros(size(Umat,1)*size(Umat,2),size(AA{ilmi},1));
        for i=1:size(AA{ilmi},1);
            BB(:,i) = my_kron(Umat',Z',mat(AA{ilmi}(i,:)));
        end
        UUUUT = BB'*BB;
        
        nnn = size(AA{1},1);
        switch aamat
            case 0
                ZZZ = AAAAT{ilmi};
            case 1
                ZZZ = spdiags(diag(AAAAT{ilmi}),0,nnn,nnn);
            case 2
                ZZZ = spdiags(ones(nnn,1),0,nnn,nnn);
            case 3
                ZZZ = sparse(nnn,nnn);
        end
        
        Htil = Htil + ttau^2.*ZZZ + UUUUT;
        if(fig_ev == 1)
            Htila = Htila + ttau^2.*spdiags(ones(nnn,1),0,nnn,nnn) + UUUUT;
            HUHU = HUHU + ttau^2.*spdiags(ones(nnn,1),0,nnn,nnn);
        end
        
    end
    %Htila = (Htila+Htila')./2;
    
    if(fig_ev == 1)
        %H1 = HHHH; H1 = (H1+H1')./2;
        HHHH = HHHH + AA{ilmi}*kron(W0,W0)*AA{ilmi}'+UUUUT;%HHHH=(HHHH+HHHH')./2;
        H0 = AA{ilmi}*kron(W0,W0)*AA{ilmi}';% H0 = (H0+H0')./2;
        %H2 = AA{ilmi}*kron(W{1},W{1})*AA{ilmi}'; H2 = (H2+H2')./2;
        %Halpha = ttau^2.*spdiags(ones(nnn,1),0,nnn,nnn) + UUUUT;
        %H3a = sqrtm(inv(Halpha));H3 = H3a*(H0-ttau^2.*spdiags(ones(nnn,1)))*H3a;
    end
end
%norm(HHHH-(H0-ttau^2.*spdiags(ones(nnn,1),0,nnn,nnn)+Htila))
%norm(HHHH-(C_lin*spdiags(X_lin./S_lin,0,nlin,nlin)*C_lin'+AA{ilmi}*kron(W0,W0)*AA{ilmi}'+UUUUT))
%
if(fig_ev == 1)
    %     figure
    %     %plot(log10(sort(abs(eig(Htila)))),'LineWidth',2);hold on
    %     %plot(log10(sort(abs(eig(Htila-C_lin*spdiags(X_lin./S_lin,0,nlin,nlin)*C_lin')))),'g','LineWidth',2);hold on
    %     %  plot(log10(sort(abs(eig(HUHU)))),'k','LineWidth',2);
    %     %plot(log10(sort(abs(eig(H2)))),'r','LineWidth',2);hold on
    %     plot(log10(sort(abs(eig(HHHH-C_lin*spdiags(X_lin./S_lin,0,nlin,nlin)*C_lin')))),'b','LineWidth',2);%hold on
    %     plot(log10(sort(abs(eig(C_lin*spdiags(X_lin./S_lin,0,nlin,nlin)*C_lin')))),'r','LineWidth',2);hold on
    %     %plot(log10(sort(abs(eig(HHHH)))),'r','LineWidth',2);hold on
    %     %plot(log10(sort(abs(eig(H0)))),'m','LineWidth',2);
    %     %plot(log10(sort(abs(eig(H3a)))),'k','LineWidth',2);
    %     plot(log10(sort(abs(eig(W{1})))),'r','LineWidth',2);
    %     pause(0.01);
    
    %figure
     iHtila=pinv(Htila);iHtila=(iHtila+iHtila')./2;iHtil=pinv(Htil);iHtil=(iHtil+iHtil')./2;
     llla=sqrtm(iHtila); lll=sqrtm(inv((UUUUT+AA{ilmi}*kron(W0,W0)*AA{ilmi}'+C_lin*spdiags(X_lin./S_lin,0,nlin,nlin)*C_lin'))); %lll=sqrtm(iHtil);
     plot((sort(real(eig(llla*HHHH*llla')))),'b','LineWidth',2); %hold on
    %  max(abs(eig(llla*HHHH*llla')))
    %  plot((sort(real(eig(lll*HHHH*lll')))),'b','LineWidth',2);
    ea=eig(Htila);eb=eig(Htil);ee=eig(HHHH);
    %     min(eig(HHHH*inv(Htila)))
    %     min((ee)./(ea))
    %     %min(ee)/max(ea)
    %     min(ee)/min(ea)
    pause(0.01);
    Htil = Htila;
    cond(HHHH)
    cond(Htila)
    cond(llla*HHHH*llla')
    %min(eig(HHHH))
    %min(eig(AA{ilmi}*kron(W0,W0)*AA{ilmi}' + C_lin*spdiags(X_lin./S_lin,0,nlin,nlin)*C_lin'))
end

%Htil=Htila;

%cond(llla*HHHH*llla')
%cond(W0)

%L = C_lin*sqrtm(spdiags(X_lin./S_lin,0,nlin,nlin));
%F=[(ttau)*eye(m),BB',L]'; Q=F*sqrtm(inv(full(F'*F)));
% H0 = AA{ilmi}*kron(W0,W0)*AA{ilmi}'; H0=(H0+H0')./2;
% Hgamma = L*L'+UUUUT+ttau*ttau*eye(m); Hgamma = L*L';
% llla=sqrtm(inv(Hgamma));
%EEE1=llla*HHHH*llla'; %EEE2=Q'*[(1/(ttau*ttau)).*H0,zeros(300,641);zeros(641,300),eye(641)]*Q;
%EEE1=(EEE1+EEE1')./2;
%1+max(eig(llla*H0*llla))
%cond(llla*HHHH*llla')
%lambda_l - max(lambda1)

%max(eig(AA{ilmi}*kron(W{1},W{1})*AA{ilmi}',UUUUT+C_lin*spdiags(X_lin./S_lin,0,nlin,nlin)*C_lin'))
%max(eig(AA{ilmi}*kron(W0,W0)*AA{ilmi}',UUUUT+C_lin*spdiags(X_lin./S_lin,0,nlin,nlin)*C_lin'))
%norm(W{1}-Umat*Umat','fro')
%H0 = AA{ilmi}*kron(W0,W0)*AA{ilmi}'; H0 = (H0+H0')./2;
%Htila = L*L' + ttau^2.*spdiags(ones(nnn,1),0,nnn,nnn) + UUUUT;
%iHtila=inv(Htila);iHtila=(iHtila+iHtila')./2;iHtil=inv(Htil);iHtil=(iHtil+iHtil')./2;
%  llla=sqrtm(iHtila); lll=sqrtm(iHtil);
%HHHH = L*L' + AA{ilmi}*kron(W0,W0)*AA{ilmi}'+UUUUT;HHHH=(HHHH+HHHH')./2;

%H01 = llla*(H0-ttau*ttau*spdiags(ones(nnn,1),0,nnn,nnn))*llla'; H01 = (H01+H01')./2;
%H02 = llla*(H0-0.*ttau*ttau*spdiags(ones(nnn,1),0,nnn,nnn))*llla; H02 = (H02+H02')./2;
%max(eig(H01)),%/(1-min(eig(H01)))
%(1+max(eig(H01)))/(1+min(eig(H01)))
%ttau*ttau
%global condi
%condi = [condi max(abs(eig(llla*HHHH*llla')))/min(abs(eig(llla*HHHH*llla')))];
%min(eig(H01))


%global fileID
%fprintf(fileID,'%20.10f\n',max(eig(H01)));


end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function M1x = Prec_for_CG_til(x,Htil)
% preconditioner U_tilda, solve the linear system

warning('off','all')
M1x = Htil\x;
warning('on','all')

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Umat,Z,L,AAAATtau,Rchol,Pchol] = ...
    Prec_for_CG_tilS_prep(W,AA,AAAAT,kkk,aamat,nlin,C_lin,X_lin,S_lin_inv)
% preconditioner U_tilda, prepare the matrix

nlmi = size(AA,2);
kk = kkk.*ones(nlmi,1); %kk(1)=12;

Rchol=[];
nvar = size(AA{1},1);

AAAATtau = sparse(nvar,nvar);

ntot = 0; for ilmi=1:nlmi, ntot = ntot+size(W{ilmi},1); end
sizeS = 0; for ilmi=1:nlmi, sizeS = sizeS+kk(ilmi)*size(W{ilmi},1); end
S = zeros(sizeS,sizeS);

lbt = 1; lbs=1;
for ilmi = 1:nlmi
    n = size(W{ilmi},1);
    k = kk(ilmi);

    [vectf,lambdaf] = eig(W{ilmi});
    dlambdaf = diag(lambdaf);
    [lambda1,iind] = maxk(dlambdaf,k);
    vect_l = vectf(:,iind);
    lambda_l = diag(lambda1);
    jind = setdiff(1:n,iind);
    [lambda1] = dlambdaf(jind);
    vect_s = vectf(:,jind);
    lambda_s = diag(lambda1);
    if aamat == 0
        ttau = 1.0*min(lambda1);
    else
        ttau = min(1*(min(lambda1) + mean(lambda1))/2, max(lambda1)/2);
    end

%     if max(dlambdaf)/min(dlambdaf) > 1e10
%         ttau = max(mean(lambda1), max(lambda1)/4);
%     end

    if lambda_l-ttau.*speye(k) > 0
        Umat{ilmi} = sqrt(lambda_l-ttau.*speye(k));
    else
        Umat{ilmi} = sqrt(lambda_l-min(lambda1).*speye(k));
    end
    Umat{ilmi} = (vect_l*Umat{ilmi});
    m = size(Umat{ilmi},1);

    W0 = [vect_s vect_l]*[lambda_s sparse(n-k,k); sparse(k,n-k) ttau*speye(k)]...
        *[vect_s vect_l]';

    Z{ilmi} = chol(2.*W0 + Umat{ilmi}*Umat{ilmi}','lower');

    switch aamat
        case 0
            ZZZ = AAAAT{ilmi};
        case 1
            ZZZ = spdiags(diag(AAAAT{ilmi}),0,nvar,nvar);
        case 2
            ZZZ = spdiags(ones(nvar,1),0,nvar,nvar);
        case 3
            ZZZ = spdiags(diag(AAAAT{ilmi}),0,nvar,nvar)... 
                + spdiags(ones(nvar,1),0,nvar,nvar);
    end
    AAAATtau = AAAATtau + ttau^2.*ZZZ;
end

if nlin>0
    AAAATtau = AAAATtau + C_lin*spdiags(X_lin.*S_lin_inv,0,nlin,nlin)*C_lin';
end
AAAATtau = (AAAATtau+AAAATtau')./2;

% if 0 %max(kk)>1 %slow formula
%     for ilmi = 1:nlmi
%         n = size(W{ilmi},1);
%         k = kk(ilmi);
%         TT = kron (Umat{ilmi},Z{ilmi});
%         t(:,lbt:lbt+k*n-1) = (AA{ilmi}*TT);
%         lbt = lbt + k*n;
%     end
%     clear TT
%
% %less memory needed but much slower
% %     for ilmi = 1:nlmi
% %         n = size(W{ilmi},1);
% %         k = kk(ilmi);
% %         for ii = 1:size(AA{ilmi},1)
% %             tmp = my_kron(Umat{ilmi}',Z{ilmi}',Ai{ilmi,ii});
% %             t(ii,lbt:lbt+k*n-1) = tmp';
% %         end
% %         lbt = lbt + k*n;
% %     end
%
%     [Rchol,flag,Pchol] = chol(AAAATtau);
%     if size(t,2) > 1
%         S = t'* (AAAATtau\t);
%     else
%         S = t'*(Pchol*(Rchol\(Rchol'\(Pchol'*t))));
%     end
%
% else %fast formula
lbt = 1;
[Rchol,flag,Pchol] = chol(AAAATtau);
%[Rchol1,flag] = chol(AAAATtau);
if flag > 0
    icount = 0;
    while flag>0
        AAAATtau = AAAATtau + ttau.*speye(size(AAAATtau));
        [Rchol,flag] = chol(AAAATtau);
        icount = icount + 1;
        if icount > 100
            error('AAAATtau cannot be made positive definite')
        end
    end
end
AAAATtau_d = Rchol';

if nlmi>1
    for ilmi = 1:nlmi
        if kk(ilmi) == 0
            continue;
        end
        n = size(W{ilmi},1);
        k = kk(ilmi);
        AAs = AAAATtau_d\(Pchol'*AA{ilmi});
        %AAs = AAAATtau_d\(AA{ilmi});
        if 1
            [jj_,ii_,aa_] = find(AAs');
            qq_ = fix((jj_-1)./n)+1;
            pp_ = mod(jj_-1,n)+1;
            for kkk=1:k
                UU = Umat{ilmi}(qq_,kkk);
                aau = aa_.*UU;
                AU = sparse(ii_,pp_,aau,nvar,n);
                if nlmi>1 || k>1
                    t1(:,lbt+(kkk-1)*n:lbt+kkk*n-1) = AU*Z{ilmi};
                else
                    t1 = AU*Z{1};
                end
            end
        end
        lbt = lbt + k*n;
    end
    S = t1'*t1;

else
    clear AU; lbt = 1;
    for ilmi = 1:nlmi
        if kk(ilmi) == 0
            continue;
        end
        n = size(W{ilmi},1);
        k = kk(ilmi);
        AAs = AAAATtau_d\(Pchol'*AA{ilmi});
        %AAs = AAAATtau_d\(AA{ilmi});
        if 1
            [jj_,ii_,aa_] = find(AAs');
            qq_ = fix((jj_-1)./n)+1;
            pp_ = mod(jj_-1,n)+1;
            for kkk=1:k
                UU = Umat{ilmi}(qq_,kkk);
                aau = aa_.*UU;
                AU{kkk} = sparse(ii_,pp_,aau,nvar,n);
            end
        end

        for iii=1:k
            tmp = full(AU{iii}'*AU{iii});
            tmp = Z{ilmi}'*tmp*Z{ilmi};
            S(lbt+(iii-1)*n:lbt+iii*n-1,lbt+(iii-1)*n:lbt+iii*n-1) = tmp;
            for jjj=iii+1:k
                tmp = full(AU{iii}'*AU{jjj});
                tmp = Z{ilmi}'*tmp*Z{ilmi};
                S(lbt+(jjj-1)*n:lbt+jjj*n-1,lbt+(iii-1)*n:lbt+iii*n-1) = tmp';
                S(lbt+(iii-1)*n:lbt+iii*n-1,lbt+(jjj-1)*n:lbt+jjj*n-1) = tmp;
            end
        end
        lbt = lbt + k*n;
    end
end
% end


% Schur complement for the SMW formula
S = (S+S')./2 + eye(sizeS);
[L,flag] = chol(S,'lower');
if flag>0
    icount = 0;
    while flag > 0
        S = S + ttau.*eye(sizeS);
        [L,flag] = chol(S,'lower');
        icount = icount + 1;
        if icount > 100
            error('Schur complement cannot be made positive definite')
            return
        end
    end
end

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function M1x = Prec_for_CG_tilS(x,AA,AAAATtau,Rchol,Pchol,Umat,Z,L)
% preconditioner U_tilda, solve the linear system using SMW formula

nvar = size(x,1);
nlmi = size(AA,2);

yy2 = zeros(nvar,1);
y3 = []; y33 = [];

if size(x,2) > 1 || isempty(Rchol)==1
    AAAAinvx = AAAATtau\x;
else
    AAAAinvx = Pchol*(Rchol\(Rchol'\(Pchol'*x)));
end

for ilmi = 1:nlmi
    n = size(Umat{ilmi},1);
    k = size(Umat{ilmi},2);
    y22 = AA{ilmi}'*AAAAinvx;
    y33 = [y33; vec(Z{ilmi}'*mat(y22)*Umat{ilmi})];
end

y33 = (L'\(L\y33));

ii=0;
for ilmi = 1:nlmi
    n = size(Umat{ilmi},1);
    k = size(Umat{ilmi},2);
    yy = zeros(n*n,1);
    for i=1:k
        xx = Z{ilmi}*y33(ii+1:ii+n,1);
        yy = yy + kron(Umat{ilmi}(:,i),xx);
        ii = ii+n;
    end
    yy2 = yy2 + AA{ilmi}*yy;
end

if size(x,2) > 1 || isempty(Rchol)==1
    yyy2 = AAAATtau\yy2;
else
    yyy2 = Pchol*(Rchol\(Rchol'\(Pchol'*yy2)));
end

M1x = AAAAinvx - yyy2;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   SUPPORTING FUNCTIONS Printout
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function print_header(verb,eqc,kit)
if verb > 0
    fprintf(' *** IP STARTS\n')
    if eqc > 0
        if verb < 2
            if kit == 0
                fprintf(' it        obj         error     err_eq    CPU/it\n')
            else
                fprintf(' it        obj         error     err_eq  cg_iter   CPU/it\n')
            end
        else
            if kit == 0
                fprintf(' it        obj         error      err1      err2      err3      err4      err5      err6     err_eq   CPU/it\n')
            else
                fprintf(' it        obj         error      err1      err2      err3      err4      err5      err6     err_eq  cg_pre cg_cor  CPU/it\n')
            end
        end
    else
        if verb < 2
            if kit == 0
                fprintf(' it        obj         error     CPU/it\n')
            else
                fprintf(' it        obj         error     cg_iter   CPU/it\n')
            end
        else
            if kit == 0
                fprintf(' it        obj         error      err1      err2      err3      err4      err5      err6     CPU/it\n')
            else
                fprintf(' it        obj         error      err1      err2      err3      err4      err5      err6     cg_pre cg_cor  CPU/it\n')
            end
        end
    end
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function print_progress(verb,eqc,kit,iter, yddc, DIMACS_error, err1, err2, err3, err4, err5, err6, eq_norm, cg_iter1, cg_iter2, titi)
if verb > 0
    if eqc > 0
        if verb < 2
            if kit == 0
                fprintf('%3.0d %16.8e %9.2e %9.2e %8.2f\n', iter, yddc, DIMACS_error, eq_norm, titi)
            else
                fprintf('%3.0d %16.8e %9.2e %9.2e %7.0d %8.2f\n', iter, yddc, DIMACS_error, eq_norm, cg_iter1+cg_iter2, titi)
            end
        else
            if kit == 0
                fprintf('%3.0d %16.8e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %7.2f\n', iter, yddc, DIMACS_error, err1, err2, err3, err4, err5, err6, eq_norm, titi)
            else
                fprintf('%3.0d %16.8e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %6.0d %6.0d %7.2f\n', iter, yddc, DIMACS_error, err1, err2, err3, err4, err5, err6, eq_norm, cg_iter1, cg_iter2, titi)
            end
        end
    else
        if verb < 2
            if kit == 0
                fprintf('%3.0d %16.8e %9.2e %8.2f\n', iter, yddc, DIMACS_error, titi)
            else
                fprintf('%3.0d %16.8e %9.2e %7.0d %8.2f\n', iter, yddc, DIMACS_error, cg_iter1+cg_iter2, titi)
            end
        else
            if kit == 0
                fprintf('%3.0d %16.8e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %7.2f\n', iter, yddc, DIMACS_error, err1, err2, err3, err4, err5, err6, titi)
            else
                fprintf('%3.0d %16.8e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %6.0d %6.0d %7.2f\n', iter, yddc, DIMACS_error, err1, err2, err3, err4, err5, err6, cg_iter1, cg_iter2, titi)
            end
        end
    end
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [X0,S0,y0,X_lin,S_lin,S_lin_inv] = initial(msizesex,nlmi,AA,b,C,nlin,C_lin,d_lin)

A = AA;
C_lin = C_lin';

n = length(b);
y0= 1.*zeros(n,1);

X0 = cell(size(C));
S0 = cell(size(C));
for i=1:nlmi
    X0{i} = eye(msizesex(i));
    S0{i} = eye(msizesex(i));
end
b2 = 1 + abs(b');
f = zeros(1,n);
ff = zeros(1,n);
for i=1:nlmi
%     for j=1:n
%         normA = 1+norm(A{i}(j,:));
%         f(j)=b2(j)./normA;
%     end
%     f = max(f);
    f = norm(b2)/(1+normest(A{i},1e-3)); %f = normest(A{i});
    Eps{i} = sqrt(msizesex(i)).* max(1,sqrt(msizesex(i)).* f);
    X0{i} = Eps{i} * X0{i};
    
%     for j=1:n
%         ff(j)=norm(A{i}(j,:));
%     end
%     ff = max(ff);
    ff = normest(A{i},1e-3);
    mf = max(f,norm(C{i},'fro'));
    mf = (1 + mf)./ sqrt(msizesex(i));
    Eta{i} = sqrt(msizesex(i)).* max(1,mf);
    S0{i} = Eta{i} * S0{i};
end

p = zeros(1,n);
pp = zeros(1,n);
dd = size(d_lin,1);
if nlin>0
    for j=1:n
        normClin = 1+norm(C_lin(:,j));
        p(j)=b2(j)./normClin;
    end
    Epss = max(1, max(p));
    X_lin = 1.*Epss * ones(dd,1);
    
    for j=1:n
        pp(j)=norm(C_lin(:,j));
    end
    mf = max(max(pp),norm(d_lin));
    mf = (1 + mf)./ sqrt(dd);
    Etaa =  max(1,mf);
    S_lin = 1.*Etaa * ones(dd,1);
    S_lin_inv = 1./S_lin;
else
    X_lin = []; S_lin = [];
end
if nlin==0
    C_lin=[]; X_lin=[]; S_lin=[]; S_lin_inv=[];
end
end

