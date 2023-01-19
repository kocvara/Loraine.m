# Loraine.m

Loraine is a general-purpose solver for any linear SDP with linear equality and inequality constraints. Compared to other general-purpose SDP software, it particularly targets two classes of problems.

*Problems with low-rank solutions:* Loraine is using the preconditioned conjugate gradient method. In particular, the user can choose between the direct and iterative solver, the type of preconditioner and the expected rank of the solution. The direct solver relies on the implementation of the (sparse or dense) Cholesky factorization provided by MATLAB.

*Problems with low-rank data input:* This feature is only useful when a direct solver is used. In this case, the bottleneck of an interior-point algorithm is the computation of the Schur complement matrix. When the data matrices are of low rank (typically of rank one), the complexity can be drastically reduced. The user has a choice

- to provide vectors defining data matrices;
    
- to indicate that the input data matrices are all expected to be of rank one. Then their decomposition to a vector-vector product can be automatically computed by Loraine.

To solve a problem from the database, identify the problem name in `rloraine.m`. To solve a problem in SDPA input file, uncomment the corresponding part in `rloraine.m` and identify the name of your `*.dat-s` file.

Then, in MATLAB environment, run
  
`>> rloraine`

Loraine was developed by Soodeh Habibi and Michal Kocvara, University of Birmingham, and Michael Stingl, University of Erlangen, for H2020 ITN POEMA
and is distributed under the GNU General Public License 3.0. For commercial applications that may be incompatible with this license, please contact the authors to discuss alternatives. 

*Copyright (c) 2023 by EC-H2020-ITN POEMA*
