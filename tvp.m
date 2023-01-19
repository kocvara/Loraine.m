% calling script for Loraine
% Programmed by Soodeh Habibi and Michal Kocvara, University of Birmingham,
% and Michael Stingl, University of Erlangen, for H2020 ITN POEMA

addpath database

% d = sdpa2poema('database/problems/SDPA/buck2.dat-s');
d = sdpa2poema('../../../sdplib/gpp250-1.dat-s');
d = poema2sparse(d);

% load('database/problems/MATLAB_POEMA/tru3');
% load('database/problems/MATLAB_POEMA/maxcut_unw20');
% d=poema2sparse(sdp);
%   
%  load('database/problems/SPARSE_POEMA/SNL16'); 
%  load('database/problems/MATLAB_SPARSE/maxcut_w20'); d=dd;
% load('database/problems/SPARSE_POEMA_LR/tru15lr'); 


%% OPTIONS FOR Loraine

options.kit = 0;           % kit = 0 for direct solver; kit = 1 for CG
options.tol_cg = 1e-2;     % tolerance for CG solver (1e-2)
options.tol_cg_up = 0.5;   % tolerance update (0.5)
options.tol_cg_min = 1e-6; % minimal tolerance for CG solver (1e-6)
options.cg_type = 'minres';  % 'minres' or 'cg' implemented
options.eDIMACS = 1e-5;    % epsilon for DIMACS error stopping criterion (1e-5)

options.prec  = 1;      % 0...no; 1...H_tilde with SMW; 2...H_hat; 3...H_tilde inverse; 4...hybrid
options.erank = 1;      % estimated rank

options.mup = 100;      % initial penalty parameter mu for l1-penalization
options.aamat = 2;      % 0 ... A^TA; 1 ... diag(A^TA); 2 ... identity

options.verb = 1;       % 2..full output, 1..short output, 0..no output

options.datarank = 0;   % 0..full rank matrices expected, 
                        % -1..rank-1 matrices expected, coverted to vectors, if possible
                        % 1..vector expected for low-rank data matrices
options.initpoint = 0;  % 0..Loraine heuristics, 1..SDPT3-like heuristics

% Use fig_ev=1 only for small problems!!! switches to preconditioner = 3 !!!
options.fig_ev = 0; % 0...no figure; 1.1.figure with eigenvalues of H, H_til, etc in every IP iteration

%% CALLING loraine

[y2,X2,S2] = loraine(d,options);

clear eee
for i=1:d.nlmi
    eee{i} = eig(X2{i}); eee{i} = sort(eee{i},'descend');
    % estimation of rank of X2
    eta = 10000;
    eee1 = circshift(eee{i},-1); eee1(end) = eee1(end-1);
    e = eee{i}./eee1;
    ee=find(e>eta);
    if length(ee)>0, ee1 = ee(1); else, ee1=0; end
    fprintf('*** Estimated rank of X*: %3.0d\n', ee1)
end



