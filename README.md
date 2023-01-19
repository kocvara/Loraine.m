# Loraine.m

Loraine is a general-purpose solver for any linear SDP with linear equality and inequality constraints. Compared to other general-purpose SDP software, it particularly targets two classes of problems.

Problems with low-rank solutions: Loraine is using the preconditioned conjugate gradient method. In particular, the user can choose between the direct and iterative solver, the type of preconditioner and the expected rank of the solution. The direct solver relies on the implementation of the (sparse or dense) Cholesky factorization provided by MATLAB.

Problems with low-rank data input: This feature is only useful when a direct solver is used. In this case, the bottleneck of an interior-point algorithm is the computation of the Schur complement matrix in \cref{eq:sced} and \cref{eq:schurcomp}. When the data matrices are of low rank (typically of rank one), the complexity can be drastically reduced. The user has a choice
    - to provide vectors $(a_i^j)_k,\ k=1,\ldots,r$, defining matrices $A_j^{(i)} = \sum_{k=1}^r (a_i^j)_k (a_i^j)_k^\top$ (field \verb|Avec| in the Data input section below);
    - to indicate that the input matrices $A_j^{(i)}$ are all expected to be of rank one. Then their decomposition to a vector-vector product can be automatically computed by Loraine.

