### The following functions are from
### https://github.com/mhauru/MPS-in-Julia-minicourse
### and with some modifications partially used in our own implementation

using TensorOperations
using LinearMaps


"""
    rand_UMPS(d, D; keep_it_real=true)

Return a random three-valent tensor A, that defines a uniform MPS (UMPS).
The bond dimension of the physical leg should be d, and the bond dimension
of the two "virtual" legs (the horizontal ones) should be D.
keep_it_real is keyword argument, for whether the matrix should be real or
complex.

This means you can call
`rand_UMPS(2, 9)`
or
`rand_UMPS(2, 9; keep_it_real=true)`
and they both give a you real A, but you can also call
`rand_UMPS(2, 9; keep_it_real=false)`
to get a complex A.
"""
function rand_UMPS(d, D; keep_it_real=true)
    shp = (D, d, D)
    if keep_it_real
        A = randn(shp)
    else
        A_real = randn(shp)
        A_imag = randn(shp)
        A = complex.(A_real, A_imag) / sqrt(2)
    end
    return A
end


"""
    tm(A)

Return the transfer matrix of A:
 --A---
   |
 --A*--
"""
function tm(A)
    @tensor T[i1,i2,j1,j2] := A[i1,p,j1]*conj(A)[i2,p,j2]
end


function eig_and_trunc(T, nev; by=identity, rev=false)
    S, U = eig(T)
    perm = sortperm(S; by=by, rev=rev)
    S = S[perm]
    U = U[:, perm]
    S = S[1:nev]
    U = U[:, 1:nev]
    return S, U
end

"""
    tm_eigs(A, dirn, nev)

Return some of the eigenvalues and vectors of the transfer matrix of A.
dirn should be "L", "R" or "BOTH", and determines which eigenvectors to return.
nev is the number of eigenpairs to return (starting with the eigenvalues with
largest magnitude).
"""
function tm_eigs_dense(A, dirn, nev)
    T = tm(A)
    D = size(T, 1)
    T = reshape(T, (D^2, D^2))
    nev = min(nev, D^2)

    result = ()
    if dirn == "R" || dirn == "BOTH"
        SR, UR = eig_and_trunc(T, nev; by=abs, rev=true)
        UR = [reshape(UR[:,i], (D, D)) for i in 1:nev]
        result = tuple(result..., SR, UR)
    end
    if dirn == "L" || dirn == "BOTH"
        SL, UL = eig_and_trunc(T', nev; by=abs, rev=true)
        UL = [reshape(UL[:,i], (D, D)) for i in 1:nev]
        result = tuple(result..., SL, UL)
    end
    return result
end


"""
    tm_l(A, x)

Return y, where
/------   /------A--
|       = |      |
\- y* -   \- x* -A*-
"""
function tm_l(A, x)
    @tensor y[i, j] := (x[a, b] * A[b, p, j]) * conj(A[a, p, i])
    return y
end


"""
    tm_r(A, x)

Return y, where
-- y -\   --A-- x -\
      | =   |      |
------/   --A*-----/
"""
function tm_r(A, x)
    @tensor y[i, j] := A[i, p, a] * (conj(A[j, p, b]) * x[a, b])
    return y
end


function tm_eigs_sparse(A, dirn, nev)
    if dirn == "BOTH"
        SR, UR = tm_eigs_sparse(A, "R", nev)
        SL, UL = tm_eigs_sparse(A, "L", nev)
        return SR, UR, SL, UL
    else
        D = size(A, 1)
        x = zeros(eltype(A), (D, D))
        if dirn == "L"
            f = v -> vec(tm_l(A, copy!(x, v)))
        else
            f = v -> vec(tm_r(A, copy!(x, v)))
        end

        fmap = LinearMap{eltype(A)}(f, D^2)
        S, U, nconv, niter, nmult, resid = eigs(fmap, nev=nev, which=:LM, ritzvec=true)
        U = [reshape(U[:,i], (D, D)) for i in 1:size(U, 2)]

        return S, U
    end
end


function tm_eigs(A, dirn, nev; max_dense_D=10)
    D = size(A, 1)
    if D <= max_dense_D || nev >= D^2
        return tm_eigs_dense(A, dirn, nev)
    else
        return tm_eigs_sparse(A, dirn, nev)
    end
end


"""
    normalize_it(A)

Normalize the UMPS defined by A, and return the dominant left and right
eigenvectors l and r of its transfer matrix, normalized so that they are
both Hermitian and positive semi-definite (when thought of as matrices),
and l'*r = 1.
"""
function normalize_it(A; do_normalization=true)
    SR, UR, SL, UL = tm_eigs(A, "BOTH", 1)
    S1 = SR[1]
    if do_normalization
        # println("normalization")
        A ./= sqrt(S1)
    end

    l = UL[1]
    r = UR[1]
    # We want both l and r to be Hermitian and pos. semi-def.
    # We know they are that, up to a phase.
    # We can find this phase, and divide it away, because it is also the
    # phase of the trace of l (respectively r).
    r_tr = trace(r)
    phase_r = r_tr/abs(r_tr)
    r ./= phase_r
    l_tr = trace(l)
    phase_l = l_tr/abs(l_tr)
    l ./= phase_l
    # Finally divide them by a real scalar that makes
    # their inner product be 1.
    n = vec(l)'*vec(r)
    abs_n = abs(n)
    phase_n = n/abs_n
    (phase_n ??? 1) && warn("In normalize_it phase_n = ", phase_n, " ??? 1")
    sfac = sqrt(abs_n)
    if do_normalization
        l ./= sfac
        r ./= sfac
    end
    # println("n = ",vec(l)'*vec(r))
    return l, r
end


"""
    tm_l_op(A, O, x)

Return y, where
/------   /------A--
|         |      |
|       = |      O
|         |      |
\- y* -   \- x* -A*-
"""
function tm_l_op(A, O, x)
    @tensor y[i, j] := (x[a, b] * A[b, p2, j]) * (conj(A[a, p1, i]) * conj(O[p1, p2]))
    return y
end


"""
    tm_r_op(A, O, x)

Return y, where
-- y -\   --A-- x -\
      |     |      |
      | =   O      |
      |     |      |
------/   --A*-----/
"""
function tm_r_op(A, O, x)
    @tensor y[i, j] := (A[i, p1, a] * O[p1, p2]) * (conj(A[j, p2, b]) * x[a, b])
    return y
end


"""
    expect_local(A, O, l, r)

Return the expectation value of the one-site operator O for the UMPS state
defined by the tensor A.
"""
function expect_local(A, O, l, r)
    l = tm_l_op(A, O, l)
    expectation = vec(l)'*vec(r)
    return expectation
end


"""
    correlator_twopoint(A, O1, O2, m, l, r)

Return the (connected) two-point correlator of operators O1 and O2 for the
state UMPS(A), when O1 and O2 are i sites apart, where i ranges from 1 to m. In
other words, return <O1_0 O2_i> - <O1> <O2>, for all i = 1,...,m, where the
expectation values are with respect to the state |UMPS(A)>.
"""
function correlator_twopoint(A, O1, O2, m, l, r)
    local_O1 = expect_local(A, O1, l, r)
    local_O2 = expect_local(A, O2, l, r)
    disconnected = local_O1 * local_O2

    l = tm_l_op(A, O1, l)
    r = tm_r_op(A, O2, r)

    result = zeros(eltype(A), m)
    result[1] = vec(l)'*vec(r) - disconnected
    for i in 1:m
        r = tm_r(A, r)
        result[i] = vec(l)'*vec(r) - disconnected
    end
    return result
end


"""
    correlation_length(A)

Return the correlation length ?? of the UMPS defined by A. ?? = - 1/ln(|lambda[2]|),
where lambda[2] is the eigenvalue of the MPS transfer matrix with second largest
magnitude. (We assume here that UMPS(A) is normalized.)
"""
function correlation_length(A)
    S, U = tm_eigs(A, "L", 2)
    s2 = S[2]
    ?? = -1/log(abs(s2))
    return ??
end


"""
    canonical_form(A, l, r)

Return a three-valent tensor ?? and a vector ??, that define the canonical
of the UMPS defined by A. l and r should be the normalized dominant
left and right eigenvectors of A.
"""
function canonical_form(A, l, r, eval_threshold=0)
    # println("\ns(A): ",size(A))
    l_H = 0.5*(l + l')
    r_H = 0.5*(r + r')
    (l_H ??? l) && warn("In canonical_form, l is not Hermitian: ", vecnorm(l_H - l))
    (r_H ??? r) && warn("In canonical_form, r is not Hermitian: ", vecnorm(r_H - r))
    evl, Ul = eig(Hermitian(l_H))
    evr, Ur = eig(Hermitian(r_H))
    # println("Ul:",size(Ul))
    # println("evl: ",evl,size(evl))
    # println("Ur:",size(Ur))
    # println("evr: ",evr,size(evr))

    ## exclude tiny eigenvalues which are problematic in inversion
    f(x) = x < eval_threshold
    inds_l = find(f,evl)
    inds_r = find(f,evr)
    if eval_threshold != 0 && length(inds_l)>0
        println("inds_l: ",inds_l)
        evl = evl[inds_l[end]+1:end]
        Ul = Ul[:,inds_l[end]+1:end]
        println("s(Ul) cut: ",size(Ul))
        # println("evl: ",evl)
    end
    if eval_threshold != 0 && length(inds_r)>0
        println("inds_r: ",inds_r)
        evr = evr[inds_r[end]+1:end]
        Ur = Ur[:,inds_r[end]+1:end]
        println("s(Ur) cut: ",size(Ur))
        # println("evr: ",evr)
    end

    X = Ur * Diagonal(sqrt.(complex.(evr)))
    YT = Diagonal(sqrt.(complex.(evl))) * Ul'
    U, ??, V = svd(YT*X)
    # println("?? in YT*X: ",??)
    Xi = Diagonal(sqrt.(complex.(1./evr))) * Ur'
    YTi = Ul * Diagonal(sqrt.(complex.(1 ./ evl)))
    @tensor ??[x,i,y] := (V'[x,a] * Xi[a,b]) * A[b,i,c] * (YTi[c,d] * U[d,y])
    # println("s(??): ",size(??),"\n")
    return ??, ??
end

function canonical_form_chol(A, l, r)
    println("\ns(A): ",size(A))
    l_H = 0.5*(l + l')
    r_H = 0.5*(r + r')
    (l_H ??? l) && warn("In canonical_form, l is not Hermitian: ", vecnorm(l_H - l))
    (r_H ??? r) && warn("In canonical_form, r is not Hermitian: ", vecnorm(r_H - r))
    Ul = chol(Hermitian(l_H))
    Ur = chol(Hermitian(r_H))
    println("Ul:",size(Ul))
    println("Ur:",size(Ur))

    X = Ur'
    YT = Ul
    U, ??, V = svd(YT*X)
    # println("?? in YT*X: ",??)
    Xi = inv(X)
    YTi = inv(YT)
    println(typeof(V),typeof(Xi),typeof(A),typeof(YTi),typeof(U))
    @tensor ??[x,i,y] := (V'[x,a] * Xi[a,b]) * A[b,i,c] * (YTi[c,d] * U[d,y])
    println("s(??): ",size(??),"\n")
    return ??, ??
end




################################################################################
#                                   iTEBD functions
################################################################################
"""
    truncate_svd(U, S, V, D)

Given an SVD of some matrix M as M = U*diagm(S)*V', truncate this
SVD, keeping only the D largest singular values.
"""
# TODO Add an optional parameter for a threshold ??, such that if
# the truncation error is below this, a smaller bond dimension can
# be used.
function truncate_svd_static(U, S, V, D)
    U = U[:, 1:D]
    S = S[1:D]
    V = V[:, 1:D]
    return U, S, V
end


"""
    double_canonicalize(??A, ??A, ??B, ??B)

Given ??A, ??A, ??B, ??B that define an infinite MPS with two-site
translation symmetry (the ??s are the tensors and the ??s are the
vectors of diagonal weights on the virtual legs), return an MPS
defined by ??A', ??A', ??B', ??B', that represents the same state,
but has been gauge transformed into the canonical form.
See Figure 4 of https://arxiv.org/pdf/0711.3960.pdf.
"""
function double_canonicalize(??A, ??A, ??B, ??B; do_normalization=true)
    # Note that we don't quite follow Figure 4 of
    # https://arxiv.org/pdf/0711.3960.pdf: In order
    # to make maximal use of the old code we have
    # above, we build a tensor C, that includes both
    # ?? and ?? of part (i) and (ii) in Figure 4.
    D_out, d, D_in = size(??A) # D on two sides might be different from dynamical truncation
    # println("D_out, d, D_in in canon: ",D_out,", ",d,", ",D_in)
    # The next two lines are equivalent to
    # @tensor A[x,i,y] := ??A[x,i,a] * diagm(??A)[a,y]
    # @tensor B[x,i,y] := ??B[x,i,a] * diagm(??B)[a,y]
    A = ??A .* reshape(??A, (1,1,D_in))
    B = ??B .* reshape(??B, (1,1,D_out))
    @tensor C[x,i,j,y] := A[x,i,a] * B[a,j,y]
    C = reshape(C, (D_out, d*d, D_out))
    l, r = normalize_it(C, do_normalization=do_normalization)
    ??, ??B = canonical_form(C, l, r)
    # The next line is equivalent to
    @tensor ??[x,i,y] := diagm(??B)[x,a] * ??[a,i,b] * diagm(??B)[b,y]
    # ?? .*= reshape(??B, (D,1,1)) .* reshape(??B, (1,1,D))
    ?? = reshape(??, (D_out*d, d*D_out))
    ??A, ??A, ??B = svd(??)
    # println("size(??A, ??A, ??B): ",size(??A),size(??A),size(??B))
    ??A, ??A, ??B = truncate_svd_static(??A, ??A, ??B, D_in)  # This always causes effectively zero error!
    ??A = reshape(??A, (D_out, d, D_in))
    ??B = reshape(??B', (D_in, d, D_out))
    ??Binv = 1. ./ ??B
    # The next two lines are equivalent to
    @tensor ??A[x,i,y] := diagm(??Binv)[x,a] * ??A[a,i,y]
    @tensor ??B[x,i,y] := ??B[x,i,a] * diagm(??Binv)[a,y]
    # ??A .*= reshape(??Binv, (D,1,1))
    # ??B .*= reshape(??Binv, (1,1,D))
    return ??A, ??A, ??B, ??B
end


"""
    itebd_halfstep(??A, ??A, ??B, ??B, U, pars)

Absorb a two-site gate U (not necessarily unitary) into an
MPS defined by ??A, ??A, ??B, ??B, and split the result back
into an MPS of the same form, returning ??A', ??A', ??B', ??B'.
The bond dimension of the MPS is truncated to pars["D"],
where pars is a dictionary.

This is called a "half-step" because we only absorb a U
operating on every second pair of neighbouring sites.
"""
function itebd_halfstep(??A, ??A, ??B, ??B, U, Dmax)
    D, d = size(??A, 1, 2)
    # The next four lines are equivalent to
    # @tensor lump[x,i,j,y] := (((((diagm(??B)[x,a] * ??A[a,m,b]) * diagm(??A)[b,c]) * ??B[c,n,d]) * diagm(??B)[d,y]) * U[m,n,i,j])
    A = ??A .* reshape(??B, (D,1,1))
    B = ??B .* reshape(??B, (1,1,D))
    A .*= reshape(??A, (1,1,D))
    @tensor lump[x,i,j,y] := (A[x,m,a] * B[a,n,y]) * U[m,n,i,j]
    lump = reshape(lump, (D*d, d*D))
    ??A, ??A, ??B = svd(lump)
    ??A, ??A, ??B = truncate_svd(??A, ??A, ??B, Dmax)
    ??A = reshape(??A, (D, d, D))
    ??B = reshape(??B', (D, d, D))
    ??Binv = 1 ./ ??B
    # The next two lines are equivalent to
    # @tensor ??A[x,i,y] := diagm(??Binv)[x,a] * ??A[a,i,y]
    # @tensor ??B[x,i,y] := ??B[x,i,a] * diagm(??Binv)[a,y]
    ??A .*= reshape(??Binv, (D,1,1))
    ??B .*= reshape(??Binv, (1,1,D))
    return ??A, ??A, ??B, ??B
end


"""
    itebd_step(??A, ??A, ??B, ??B, U, pars)

Apply a step of iTEBD into an MPS represented by
??A, ??A, ??B, ??B, with U being the two-site gate that
defines a layer of (imaginary) time-evolution.
Return a new MPS, ??A', ??A', ??B', ??B'.
See https://arxiv.org/pdf/cond-mat/0605597.pdf,
especially Figure 3. pars is a dictionary of parameters,
that most notably should include the bond dimension
pars["D"] to which the MPS should be truncated.
"""
function itebd_step(??A, ??A, ??B, ??B, U, Dmax)
    ??A, ??A, ??B, ??B = itebd_halfstep(??A, ??A, ??B, ??B, U, Dmax)
    ??B, ??B, ??A, ??A = itebd_halfstep(??B, ??B, ??A, ??A, U, Dmax)
    return ??A, ??A, ??B, ??B
end


"""
    itebd_random_initial(d, D)

Return ??A, ??A, ??B, ??B that define an MPS with two-site
translation invariance in the canonical form, with the
tensor chosen randomly.
"""
function itebd_random_initial(d, D)
    ?? = randn(D, d, D)
    ?? = randn(D)
    ??A, ??A, ??B, ??B = double_canonicalize(??, ??, ??, ??)
    return ??A, ??A, ??B, ??B
end


"""
    trotter_gate(h, ??)

Given a two-site gate h (a 4-valent tensor),
return the gate U = e^(-?? h).
"""
function trotter_gate(h, ??)
    d = size(h, 1)
    h = reshape(h, (d*d, d*d))
    U = expm(-??*h)
    U = reshape(U, (d,d,d,d))
    return U
end


"""
    itebd_optimize(h, pars; evalfunc=nothing)

Apply the iTEBD algorithm to find the ground state of the Hamiltonian
defined by the local Hamiltonian term h. h is assumed to operate on
nearest-neighbours only, and translation invariance is assumed. Return
??A, ??A, ??B, ??B that define an MPS with two-site translation invariance,
which is guaranteed to be in the canonical form. This MPS approximates
the ground state.
See https://arxiv.org/pdf/cond-mat/0605597.pdf.

pars is a dictionary, where each key-value pair is some parameter
that the algorithm takes. The parameters that should be provided are
"??_min and ??_step":
    Every time convergence has been reached, the Trotter
    parameter ?? is multiplied by ??_step and the optimization
    is restarted, until ?? falls below ??_min. ?? initially starts
    from 0.1.
"D":
    The bond dimension of the MPS.
"max_iters":
    The maximum number of iTEBD iterations that is done before moving
    on to the next value of ??.
"convergence_eps":
    A threshold for convergence. If the relative difference in the
    vectors of Schmidt values before and after the latest iTEBD
    iteration falls below convergence_eps, we move on to the next value
    of ??.
"inner_iters":
    At every iTEBD iteration, several layers of e^(-?? h) are absorbed
    into the MPS before recanonicalizing and checking for convergence.
    inner_iters specifies how many. Note that the total number of layers
    absorbed during the optimization for a given ?? may reach
    inner_iters * max_iters.

evalfunc is an optional function, that should take as arguments
??A, ??A, ??B, ??B that define the (canonical-form) MPS, and return a string.
This string is then printed after every iTEBD step, in addition to other
information such as the measure of convergence and the current iteration
count. Can be used, for instance, for printing the energy at every
iteration.
"""
function itebd_optimize(h, pars; evalfunc=nothing)
    d = size(h, 1)
    ??A, ??A, ??B, ??B = itebd_random_initial(d, pars["D"])
    ?? = 0.1
    while ?? > pars["??_min"]
        @printf("In iTEBD, evolving with ?? = %.3e.\n", ??)
        eps = Inf
        counter = 0
        U = trotter_gate(h, ??)
        while eps > pars["convergence_eps"] && counter < pars["max_iters"]
            counter += 1
            old_??A, old_??B = ??A, ??B

            # TODO Create some fancy criterion that determines when we need
            # to recanonicalize.
            for i in 1:pars["inner_iters"]
                ??A, ??A, ??B, ??B = itebd_step(??A, ??A, ??B, ??B, U, pars)
            end
            ??A, ??A, ??B, ??B = double_canonicalize(??A, ??A, ??B, ??B)
            eps = vecnorm(old_??A - ??A)/vecnorm(??A) + vecnorm(old_??B - ??B)/vecnorm(??B)

            @printf("In iTEBD, eps = %.3e, counter = %i", eps, counter)
            if evalfunc != nothing
                evstr = evalfunc(??A, ??A, ??B, ??B)
                print(evstr)
            end
            println()
        end
        ?? *= pars["??_step"]
    end
    return ??A, ??A, ??B, ??B
end


###############################################################################
#                           running iTEBD
###############################################################################
# let
#     # A bunch of checks that confirm that double_canonicalize works.
#     D = 10
#     d = 2
#     ??A, ??B = randn(D, d, D), randn(D, d, D)
#     ??A, ??B = randn(D), randn(D)
#     ??A, ??A, ??B, ??B = double_canonicalize(??A, ??A, ??B, ??B)
#     @tensor should_be_id_Ar[x,y] := ??A[x,i,a] * ((diagm(??A)[a,b] * conj(diagm(??A))[b,c]) * conj(??A)[y,i,c])
#     @tensor should_be_id_Br[x,y] := ??B[x,i,a] * ((diagm(??B)[a,b] * conj(diagm(??B))[b,c]) * conj(??B)[y,i,c])
#     @tensor should_be_id_Al[x,y] := ??A[a,i,x] * ((diagm(??B)[a,b] * conj(diagm(??B))[b,c]) * conj(??A)[c,i,y])
#     @tensor should_be_id_Bl[x,y] := ??B[a,i,x] * ((diagm(??A)[a,b] * conj(diagm(??A))[b,c]) * conj(??B)[c,i,y])
#     @show vecnorm(should_be_id_Ar - eye(D,D))
#     @show vecnorm(should_be_id_Br - eye(D,D))
#     @show vecnorm(should_be_id_Al - eye(D,D))
#     @show vecnorm(should_be_id_Bl - eye(D,D))
# end


function build_ising_ham(h=1.0)
    X = [0 1; 1 0]
    Z = [1 0; 0 -1]
    I2 = eye(2)
    XX = kron(X, X)
    ZI = kron(Z, I2)
    IZ = kron(I2, Z)
    H = -(XX + h/2*(ZI+IZ))
    return H
end


# Functions for evaluating the ground state energy per site
# (or the expectation of any other two-site operator.)
# dcan stands for "double canonical", meaning the canonical
# form with two-site translation symmetry.

function expect_twositelocal_dcan_AB(??A, ??A, ??B, ??B, O)
    D = size(??A, 1)
    A = reshape(??B, (D,1,1)) .* ??A .* reshape(??A, (1,1,D))
    B = ??B .* reshape(??B, (1,1,D))
    @tensor AB[x,i,j,y] := A[x,i,a] * B[a,j,y]
    @tensor expectAB[] := AB[a,i,j,b] * O[i,j,m,n] * conj(AB)[a,m,n,b]
    return expectAB[1]
end

function expect_twositelocal_dcan(??A, ??A, ??B, ??B, O)
    expectAB = expect_twositelocal_dcan_AB(??A, ??A, ??B, ??B, O)
    expectBA = expect_twositelocal_dcan_AB(??B, ??B, ??A, ??A, O)
    expectation = (expectAB + expectBA) / 2.
    return expectation
end


# magfield = 1.0
# exact_energy = -4/??
# h = build_ising_ham(magfield)
# h = reshape(h, (2,2,2,2))
# pars = Dict(
#     "??_min"  => 5e-4,
#     "??_step" => 1/2,
#     "D"      => 70,
#     "max_iters"       => 150,
#     "convergence_eps" => 1e-6,
#     "inner_iters"     => 30
# )

# Print energy is a function that takes in ??A, ??A, ??B, ??B,
# evaluates the ground-state energy for h, compares to the
# exact value, and returns a string with this information.

# print_energy = (??A, ??A, ??B, ??B) -> begin
#     energy = expect_twositelocal_dcan(??A, ??A, ??B, ??B, h)
#     abs(imag(energy)) > 1e-12 && warn("Imaginary energy value: ", energy)
#     energy = real(energy)
#     error = abs(energy - exact_energy)/abs(exact_energy)
#     str = @sprintf(", energy = %.12e, off by %.3e", energy, error)
# end

# @time ??A, ??A, ??B, ??B = itebd_optimize(h, pars; evalfunc=print_energy)
;
