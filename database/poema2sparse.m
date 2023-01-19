function sdpdata = poema2sparse(poema_struct);
%POEMA2SPARSE converts SDP POEMA Matlab structure to sparse matrices
%
% Input: SDP problem in the POEMA Matlab structure
%
% Output: a Matlab structure with data matrices stored as Matlab sparse
% matrices
%
% This file is a part of POEMA database distributed under GPLv3 license
% Copyright (c) 2020 by EC H2020 ITN 813211 (POEMA)
% Coded by Michal Kocvara, m.kocvara@bham.ac.uk
% Last Modified: 2 July 2020

sdpdata = poema_struct;

    if isfield(sdpdata.constraints,'lsi_op')
        eqcon=1;
    else
        eqcon=0;
    end

if sdpdata.type=='sdp'  %primal SDP
    %%
    if isfield(sdpdata.constraints,'nlsi'), alin = 1; else, alin=0; end
    
    sdpdata = rmfield(sdpdata,'objective');
    sdpdata.constraints = rmfield(sdpdata.constraints,'lmi_symat');
    if isfield(sdpdata.constraints,'lsi_mat'), sdpdata.constraints = rmfield(sdpdata.constraints,'lsi_mat'); end
    if isfield(sdpdata.constraints,'lsi_vec'), sdpdata.constraints = rmfield(sdpdata.constraints,'lsi_vec'); end
    if isfield(sdpdata.constraints,'nlsi'), sdpdata.constraints = rmfield(sdpdata.constraints,'nlsi'); end
    if isfield(sdpdata.constraints,'nlmi'), sdpdata.constraints = rmfield(sdpdata.constraints,'nlmi'); end
    if isfield(sdpdata.constraints,'msizes'), sdpdata.constraints = rmfield(sdpdata.constraints,'msizes'); end
    if isfield(sdpdata.constraints,'lsi_op'), sdpdata.constraints = rmfield(sdpdata.constraints,'lsi_op'); end
    sdpdata = rmfield(sdpdata,'constraints');
    
    
    sdpdata.c = sparse(poema_struct.objective);
    
    % convert sparse LMI matrices into Matlab sparse format
    if isfield(poema_struct.constraints,'lmi_symat')
        alldata = poema_struct.constraints.lmi_symat';
        matblk = find(poema_struct.constraints.msizes);
        na = length(matblk);
        nx = poema_struct.nvar;
        sdpdata.nlmi = poema_struct.constraints.nlmi;
        sdpdata.msizes = poema_struct.constraints.msizes;
        if alin==1
            sdpdata.nlin = poema_struct.constraints.nlsi;
            sdpdata.lsi_op = poema_struct.constraints.lsi_op;
        else
            sdpdata.nlin = 0;
            sdpdata.lsi_op = [];
        end
        A = cell(na,nx+1);
        for iii=1:length(matblk)
            iblk = matblk(iii);
            dim = poema_struct.constraints.msizes(iblk);
            idxentries = find(alldata(3,:)==iblk);
            thisblock = alldata(:,idxentries);
            thisblock = circshift(thisblock,-1);
            if (any(thisblock(3,:)<1 | thisblock(3,:)>dim | thisblock(4,:)<1 | ...
                    thisblock(4,:)>dim))
                error(sprintf('Block %d have indices not matching its dim=%d.',iblk,dim));
            end
            % if i>j --> lower triangle which is not allowed
        % zz=thisblock(3,:); thisblock(3,:) = thisblock(4,:); thisblock(4,:) = zz;
            if (any(thisblock(3,:)>thisblock(4,:)))
                error(sprintf('Block %d have elements outside upper triangle.',iblk));
            end
            % extract each of the matrices in this block
            for i = 0:nx
                idx = find(thisblock(1,:)==i);
                if (isempty(idx))
                    A{iblk,i+1} = sparse(dim,dim);
                else
                    M = sparse(thisblock(3,idx),thisblock(4,idx),thisblock(5,idx),dim,dim);
                    A{iblk,i+1} = M+triu(M,1)';
                end
            end
        end
        sdpdata.A = A;
    end
    
    % convert low-rank LMI matrices into Matlab sparse format
    if isfield(poema_struct.constraints,'lmi_lrmat')
        alldata = poema_struct.constraints.lmi_lrmat';
        matblk = find(poema_struct.constraints.msizes);
        na = length(matblk);
        nx = poema_struct.nvar;
        sdpdata.nlmi = poema_struct.constraints.nlmi;
        sdpdata.msizes = poema_struct.constraints.msizes;
        if alin==1
            sdpdata.nlin = poema_struct.constraints.nlsi;
            sdpdata.lsi_op = poema_struct.constraints.lsi_op;
        end
        Avec = cell(na,nx+1);
        for iii=1:length(matblk)
            iblk = matblk(iii);
            dim = poema_struct.constraints.msizes(iblk);
            idxentries = find(alldata(3,:)==iblk);
            thisblock = alldata(:,idxentries);
            thisblock = circshift(thisblock,-1);
            if (any(thisblock(3,:)<1 | thisblock(3,:)>dim | thisblock(4,:)<1 | ...
                    thisblock(4,:)>dim))
                error(sprintf('Block %d have indices not matching its dim=%d.',iblk,dim));
            end
%             % if i>j --> lower triangle which is not allowed
%             if (any(thisblock(3,:)>thisblock(4,:)))
%                 error(sprintf('Block %d have elements outside upper triangle.',iblk));
%             end
            % extract each of the matrices in this block
            for i = 0:nx
                idx = find(thisblock(1,:)==i);
                if (isempty(idx))
                    Avec{iblk,i+1} = sparse(dim,1);
                else
                    Avec{iblk,i+1} = sparse(thisblock(4,idx),thisblock(3,idx),thisblock(5,idx),dim,max(thisblock(3,idx)));
                end
            end
        end
        sdpdata.Avec = Avec;
    end

    if alin==1
        l = poema_struct.constraints.lsi_mat;
        %  sdpdata.C = sparse(l(:,2),l(:,3),l(:,1),sdpdata.nlin,nx);
        sdpdata.C = sparse(l(:,2),l(:,3),l(:,1),nx,sdpdata.nlin);
        sdpdata.d = sparse(poema_struct.constraints.lsi_vec);
    end

    if eqcon == 0
        % turning double inequalities into equalities
        ind = ones(sdpdata.nlin,1);
        k = 2;
        if sdpdata.nlin>0
            while k<=sdpdata.nlin
                if abs(sdpdata.C(:,k) + sdpdata.C(:,k-1)) <= 1e-8;
                    ind(k-1) = 2; ind(k) = 0;
                    k = k+2;
                else
                    k = k+1;
                end
            end
        end
    end
    
    epsi = 0;
    
    if eqcon == 0
    if sdpdata.nlin>0
        sdpdata.C = sdpdata.C(:,find(ind));
        sdpdata.d= sdpdata.d(:)';
        sdpdata.d = sdpdata.d(:,find(ind));
        
            sdpdata.lsi_op = mod(ind(ind>0),2);

        sdpdata.nlin = length(sdpdata.lsi_op);
    end  
    end
    % end of this block
    
elseif sdpdata.type=='sdp_relax'  %dual SDP_relax with ONE matrix variable!
    %%
    sdpdata = rmfield(sdpdata,'objective');
    sdpdata = rmfield(sdpdata,'constraints');
    
    sdpdata.c = sparse(poema_struct.constraints.rhs);
    sdpdata.msizes = [poema_struct.objective.msizes];
    sdpdata.nlmi = 1;
    sdpdata.nvar = poema_struct.constraints.ncon;
    
    % convert sparse LMI matrices into Matlab sparse format
    if isfield(poema_struct.objective,'symat')||isfield(poema_struct.constraints,'symat')
        thisblock=[];
        nx = sdpdata.nvar;
        dim = sdpdata.msizes(1);
        iblk = 1;
        if isfield(poema_struct.constraints,'symat')
            alldata = poema_struct.constraints.symat';
            matblk = find(sdpdata.msizes);
            na = length(matblk);
                   
            A = cell(na,nx+1);
            
            idxentries = find(alldata(3,:)==iblk);
            thisblock = alldata(:,idxentries);
            thisblock = circshift(thisblock,-1);
            if (any(thisblock(3,:)<1 | thisblock(3,:)>dim | thisblock(4,:)<1 | ...
                    thisblock(4,:)>dim))
                error(sprintf('Block %d have indices not matching its dim=%d.',iblk,dim));
            end
        end
        if isfield(poema_struct.objective,'symat')
            A0 = poema_struct.objective.symat';
            A0 = circshift(A0,-1);
            thisblock = [A0,thisblock];
        end
        
        % if i>j --> lower triangle which is not allowed
        if (any(thisblock(3,:)>thisblock(4,:)))
            error(sprintf('Block %d have elements outside upper triangle.',iblk));
        end
        % extract each of the matrices in this block
        for i = 0:nx
            idx = find(thisblock(1,:)==i);
            if (isempty(idx))
                A{iblk,i+1} = sparse(dim,dim);
            else
                M = sparse(thisblock(3,idx),thisblock(4,idx),thisblock(5,idx),dim,dim);
                A{iblk,i+1} = M+triu(M,1)';
            end
        end
        sdpdata.A = A;
    end
    % convert low-rank LMI matrices into Matlab sparse format
    if isfield(poema_struct.objective,'lrmat')||isfield(poema_struct.constraints,'lrmat')
        thisblock=[];
        nx = sdpdata.nvar;
        dim = sdpdata.msizes(1);
        iblk = 1;
        if isfield(poema_struct.constraints,'lrmat')
            alldata = poema_struct.constraints.lrmat';
            matblk = find(sdpdata.msizes);
            na = length(matblk);
           
            Avec = cell(na,nx+1);
            
            idxentries = find(alldata(3,:)==iblk);
            thisblock = alldata(:,idxentries);
            thisblock = circshift(thisblock,-1);
            if (any(thisblock(3,:)<1 | thisblock(3,:)>dim | thisblock(4,:)<1 | ...
                    thisblock(4,:)>dim))
                error(sprintf('Block %d have indices not matching its dim=%d.',iblk,dim));
            end
        end
        if isfield(poema_struct.objective,'lrmat')
            A0 = poema_struct.objective.lrmat';
            A0 = circshift(A0,-1);
            thisblock = [A0,thisblock];
        end
        
        % if i>j --> lower triangle which is not allowed
        if (any(thisblock(3,:)>thisblock(4,:)))
            error(sprintf('Block %d have elements outside upper triangle.',iblk));
        end
        % extract each of the matrices in this block
        for i = 0:nx
            idx = find(thisblock(1,:)==i);
            if (isempty(idx))
                Avec{iblk,i+1} = sparse(dim,1);
            else
                Avec{iblk,i+1} = sparse(thisblock(4,idx),thisblock(3,idx),thisblock(5,idx),dim,length(thisblock(3,idx)));
            end
        end
        sdpdata.Avec = Avec;
    end
    
    if isfield(poema_struct.constraints,'op')
        nlin = sum(poema_struct.constraints.op);
        if nlin > 0
            sdpdata.nlin = nlin;
            sdpdata.C = sparse(1:nlin,find(poema_struct.constraints.op),ones(nlin,1),nlin,nx);
            sdpdata.d = sparse(nlin,1);
        end
    end
else
    error('The file does not contain "sdp" or "sdp_dual" data type.')
end

end