function sdp=sdpa2poema(filename);
%SDP2POEMA reads SDPA sparse format and returns POEMA Matlab structure
%
% Read linear SDP problem from a Sparse SDPA file, separate the linear
% constraint matrix, return the problem in the POEMA Matlab structure
%
% This file is a part of POEMA database distributed under GPLv3 license
% Copyright (c) 2020 by EC H2020 ITN 813211 (POEMA)
% Coded by Michal Kocvara, m.kocvara@bham.ac.uk
% Last Modified: 18 Apr 2020

% open file
fid = fopen(filename,'r');
if (fid == -1)
    error(sprintf('Cannot open file "%s"',filename));
end

nline = 0;
phase = 0;
nx = 0;
nblocks = 0;
msizes = [];
c = [];
nentries = 0;
maxentries = 0;
alldata = [];
while 1
    line = fgetl(fid);
    nline = nline+1;
    if (~ischar(line))
        % end of file
        break;
    end
    
    % skip comments or empty lines
    if (isempty(line) || line(1)=='*' || line(1) =='"')
        continue;
    end
    
    switch (phase)
        case 0
            % expecting number of variables
            [nx,count] = sscanf(line,'%d',1);
            if (count ~= 1)
                error(sprintf('Line %d, cannot read number of variables.',nline));
            elseif (nx <= 0)
                error(sprintf('Line %d, wrong number of variables.',nline));
            end
            phase=phase+1;
            %nx
            
        case 1
            % expecting number of matrix constraints (=blocks)
            [nblocks,count] = sscanf(line,'%d',1);
            if (count ~= 1)
                error(sprintf('Line %d, cannot read number of blocks.',nline));
            elseif (nblocks <= 0)
                error(sprintf('Line %d, wrong number of blocks.',nline));
            end
            phase = phase+1;
            %nblocks
            
        case 2
            % expecting block sizes
            [msizes,count] = sscanf(line,'%d',nblocks);
            if (count ~= nblocks)
                error(sprintf('Line %d, cannot read block sizes.',nline));
            elseif (any(msizes == 0))
                error(sprintf('Line %d, block sizes are incorrect.',nline));
            end
            phase = phase+1;
            
        case 3
            % expecting the objective vector
            
            special_chars = ',{}()';
            cumidx=[];
            for i=1:length(special_chars)
                idx = findstr(line,special_chars(i));
                cumidx = [cumidx,idx];
            end
            line(cumidx) = blanks(length(cumidx));
            
            [c,count] = sscanf(line,'%g',nx);
            if (count~=nx)
                error(sprintf('Line %d, cannot read the objective vector.',nline));
            end
            phase = phase+1;
            
        case 4
            % expecting a data line
            [data,count] = sscanf(line,'%i %i %i %i %g',5);
            if (count ~= 5)
                error(sprintf('Line %d, cannot read the data line.',nline));
            end
            nentries = nentries + 1;
            alldata(:,nentries) = data;
    end
end

% close file
fclose(fid);

if (nx <= 0 || nblocks <= 0 || isempty(msizes) || isempty(c) || nentries <= 0)
    error('The file seems to be incomplete.');
end

% remove empty entries
alldata = alldata(:,1:nentries);

% check the correctness of the data lines
if (any(alldata(1,:) < 0 | alldata(1,:) > nx))
    error('Some of the data lines have matrix_number out of range.');
end
if (any(alldata(2,:) < 1 | alldata(2,:) > nblocks))
    error('Some of the data lines have block_number out of range.');
end

% extract the linear constraints
% turn 1-size matrix blocks into linear constraints
idx = find(msizes == 1);
msizes(idx) = -1;
linblk = find(msizes < 0);
nlin = sum(abs(msizes(linblk)));
nnzlin = length(find(msizes(alldata(2,:)) < 0));
% accummulate data into B matrix
B = sparse([],[],[],nx,nlin,nnzlin);
d = zeros(nlin,1);
ng = 0;    % no of the constraint written so far
for iblk = linblk'
    dim = -msizes(iblk);
    idxentries = find(alldata(2,:) == iblk);
    thisblock = alldata(:,idxentries);
    if (any(thisblock(3,:) < 1 | thisblock(3,:) > dim | ...
            thisblock(3,:) ~= thisblock(4,:)))
        error(sprintf('Diagonal block %d have indices nondiag. or out of range elements.',iblk));
    end
    % extract RHS
    idx = find(thisblock(1,:) == 0);
    if (~isempty(idx))
        d(ng+thisblock(3,idx)) = thisblock(5,idx);
    end
    % extract linear constraints bodies
    idx = find(thisblock(1,:) > 0);
    if (~isempty(idx))
        B(:,ng+1:ng+dim) = sparse(thisblock(1,idx),thisblock(3,idx),thisblock(5,idx),nx,dim);
    end
    ng = ng+dim;
end

% extract matrix constraints
M = [];
matblk = find(msizes>0); iiblk = 1;
na = length(matblk);
A = cell(na,nx+1);
for iblk = matblk'
    dim = msizes(iblk);
    idxentries = find(alldata(2,:) == iblk);
    thisblock = alldata(:,idxentries);
    if (any(thisblock(3,:) < 1 | thisblock(3,:) > dim | thisblock(4,:) < 1 | ...
            thisblock(4,:) > dim))
        error(sprintf('Block %d have indices not matching its dim=%d.',iblk,dim));
    end
    % if i>j --> lower triangle which is not allowed
    if (any(thisblock(3,:) > thisblock(4,:)))
        error(sprintf('Block %d have elements outside upper triangle.',iblk));
    end
    thisblock = circshift(thisblock,1);
    thisblock(3,:) = iiblk*ones(1,length(idxentries));
    iiblk = iiblk+1;
    M = [M;thisblock'];
end

[ir,ic,val] = find(B);
 C(:,1) = val; C(:,2) = ir; C(:,3) = ic;
msizes = msizes(msizes>0);

lsi_op = ones(nlin,1);

sdp = struct(string('type'),'sdp', ...
    string('name'),filename, ...
    string('nvar'), nx,...
    string('objective'), c(:)'...
    );
sdp.constraints = struct(...
    string('nlmi'), na,...
    string('msizes'), msizes(:)',...
    string('lmi_symat'), M,...
    string('nlsi'), nlin,...
    string('lsi_mat'), C,...
    string('lsi_vec'), d(:)',...
    string('lsi_op'), lsi_op ...
    );

end
