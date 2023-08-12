using TensorOperations
using Base.Threads
include(string(@__DIR__,"/svd_slow.jl"))

function prepareGL(mps,Dmax,tol=0)
    N = length(mps)

    trafo = false
    if length(size(mps[1]))==4
        MPS.makeCanonical(mps,0)
        d1 = size(mps[1])[2]
        d2 = size(mps[1])[3]
        mps = mpo_to_mps(mps)
        trafo=true
    else
        MPS.makeCanonical_old(mps,0)
    end
    l = Array{Array{Complex128,2}}(N+1)
    g = Array{Array{Complex128,3}}(N)
    l[1] = spdiagm(ones(Complex128,size(mps[1])[1]))
    for k = 1:N-1
        st = size(mps[k])
        tensor = reshape(mps[k],st[1]*st[2],st[3])
        U,S,V = svd(tensor)
        V=V'
        U,S,V,D,err = truncate_svd(U, S, V, Dmax,tol)
        U = reshape(U,st[1],st[2],D)
        @tensor g[k][:] := inv(l[k])[-1,1]*U[1,-2,-3]
        l[k+1] = spdiagm(S)
        @tensor mps[k+1][:] := l[k+1][-1,1]*V[1,2]*mps[k+1][2,-2,-3]
    end
    st = size(mps[N])
    Q,R = qr(reshape(mps[N],st[1]*st[2],st[3]))
    @tensor g[N][:] := inv(l[N])[-1,1]*reshape(Q,st[1],st[2],size(Q)[2])[1,-2,-3]
    l[N+1] = sparse(R)
    l[1] = spdiagm(ones(Complex128,size(g[1])[1]))
    # l[N+1] = eye(Complex128,size(g[N])[3])
    if trafo
        g2 = Array{Array{Complex128,4}}(N)
        for k=1:N
            g2[k] = reshape(g[k],size(g[k])[1],d1,d2,size(g[k])[3])
        end
        g=g2
    end

    return g,l
end

function gl_to_mps(g,l)
    N = length(g)
    mps = Array{Array{Complex128,3}}(N)
    for k = 1:N
        @tensor mps[k][:] := l[k][-1,1]*g[k][1,-2,-3]
    end
    @tensor mps[N][:] := mps[N][-1,-2,3]*l[N+1][3,-3]
    return mps
end
function gl_to_mpo(g,l)
    N = length(g)
    mpo = Array{Array{Complex128,4}}(N)
    for k = 1:N
        @tensor mpo[k][:] := l[k][-1,1]*g[k][1,-2,-3,-4]
    end
    @tensor mpo[N][:] := mpo[N][-1,-2,-3,3]*l[N+1][3,-4]
    return mpo
end

function mpo_to_mps(mpo)
    N = length(mpo)
    mps = Array{Array{Complex128,3}}(N)
    for k = 1:N
        s = size(mpo[k])
        mps[k] = reshape(mpo[k],s[1],s[2]*s[3],s[4])
    end
    return mps
end
function mps_to_mpo(mps)
    N = length(mps)
    s = size(mps)
    d = Int(sqrt(s[2]))
    mpo = Array{Array{Complex128,4}}(N)
    for k = 1:N
        mpo[k] = reshape(mps,s[1],d,d,s[3])
    end
    return mpo
end

function sparse_l(g,l,dir=:left)
    sg=size(g)
    l2=sparse(l)
    if dir == :left
        if length(sg)==3
            g3 = reshape(g,sg[1],sg[2]*sg[3])
            A = l2*g3
        else
            g4 = reshape(g,sg[1],sg[2]*sg[3]*sg[4])
            A = l2*g4
        end
    elseif dir==:right
        if length(sg)==3
            g3 = reshape(g,sg[1]*sg[2],sg[3])
            A = g3*l2
        else
            g4 = reshape(g,sg[1]*sg[2]*sg[3],sg[4])
            A = g4*l2
        end
    end
    return reshape(A,sg)
end

function updateBlocktsvd(lL,lM,lR,gL,gR,block,Dmax,tol)
    gL = sparse_l(gL,lL,:left)

    if length(size(gL))==4
        sgl = size(gL)
        initU = reshape(permutedims(gL,[1 3 2 4]),sgl[1]*sgl[2]*sgl[3],sgl[4])
    else
        sgl = size(gL)
        initU = reshape(permutedims(gL),sgl[1]*sgl[2],sgl[3])
    end
    gL = sparse_l(gL,lM,:right)
    gR = sparse_l(gR,lR,:right)
    sr = size(gR)
    sl = size(gL)
    sb = size(block)
    if length(sl)==4
        function theta(vec)
            vec = reshape(vec,sb[1],sr[3],sr[4])
            @tensor out[:] := gR[-1,-2,7,6]*vec[-3,7,6]
            @tensor out[:] := gL[-1,4,-3,1]*block[3,-2,2,4]*out[1,2,3]
            # @tensoropt (5,6) out[:] := gL[-1,2,-3,5]*gR[5,3,7,6]*block[4,-2,3,2]*vec[4,7,6]
            return reshape(out,sl[1]*sb[2]*sl[3])
        end
        function thetaconj(vec)
            vec = reshape(vec,sl[1],sb[2],sl[3])
            @tensoropt (1,5) out[:] := gL[1,2,4,5]*gR[5,3,-2,-3]*block[-1,8,3,2]*conj(vec[1,8,4])
            return reshape(conj(out),sb[1]*sr[3]*sr[4])
        end
        thetalin = LinearMap{Complex128}(theta,thetaconj,sb[2]*sl[3]*sl[1], sb[1]*sr[3]*sr[4])
    # elseif length(sl)==3
    #     function theta(vec)
    #         reshape(vec,sb[1],sr[3])
    #         @tensor out[:] = gL[-2,2,5]*gR[5,3,4]*block[6,-1,3,2]*vec[6,4]
    #         return reshape(out,sb[2],sl[1])
    #     end
    #     function thetaconj(vec)
    #         println("ASDSAD")
    #         reshape(vec,sb[2],sl[1])
    #         @tensor out[:] = gL[4,2,5]*gR[5,3,-2]*block[-1,6,3,2]*vec[6,4]
    #         return reshape(out,sb[1],sr[3])
    #     end
    #     thetalin = LinearMap{Complex128}(theta, sb[2]*sl[1], sb[1]*sr[3])
    # end
    end

    if min(size(thetalin)...)<2*Dmax
        U,S,V = svd(Base.full(thetalin), thin=true)
    else
        println("ASD")
        U,S,V = tsvd(thetalin, maxiter=1000,min(size(thetalin)[1],size(thetalin)[2],Dmax),tolconv=tol*100,tolreorth=tol*100)
    end
    V = V'
    D1 = size(S)[1] # number of singular values
    U,S,V,D1,err = truncate_svd(U,S,V,Dmax,tol)
    if length(size(gL))==4
        U=reshape(U,sl[1],sl[2],sl[3],D1)
        V=reshape(V,D1,sr[2],sr[3],sr[4])
        ilL = spdiagm(1./diag(lL))
        ilR = spdiagm(1./diag(lR))
        U = sparse_l(U,ilL,:left)
        V = sparse_l(V,ilR,:right)
        # @tensor U[:] := inv(lL)[-1,1]*U[1,-3,-2,-4]
        # @tensor V[:] := V[-1,-2,-3,3]*inv(lR)[3,-4]
    else
        U=reshape(U,sl[1],sl[2],D1)
        V=reshape(V,D1,sl[2],sr[3])
        ilL = spdiagm(1./diag(lL))
        ilR = spdiagm(1./diag(lR))
        U = sparse_l(U,ilL,:left)
        V = sparse_l(V,ilR,:right)
        # @tensor U[:] := inv(lL)[-1,1]*U[1,-2,-3]
        # @tensor V[:] := V[-1,-2,3]*inv(lR)[3,-3]
    end

    return U,spdiagm(S),V, err
end

function updateBlock(lL,lM,lR,gL,gR,block,Dmax,tol;counter_evo=false)
    gL = sparse_l(gL,lL,:left)
    gL = sparse_l(gL,lM,:right)
    gR = sparse_l(gR,lR,:right)
    if length(size(gL))==4
        if counter_evo
            # @tensor blob[-1,-2,-3,-4,-5,-6] := W[2,6,-2,-4]*Tl[-1,2,3,4]*conj(W[3,5,-3,-5])*Tr[4,6,5,-6]
            @tensor theta[:] := gL[-1,2,7,5]*gR[5,3,8,-6]*block[-4,-3,3,2]*conj(block[-5,-2,8,7])
        else
            # @tensor blob[-1,-2,-3,-4,-5,-6] := Tl[-1,2,-3,4]*block[2,6,-2,-4]*Tr[4,6,-5,-6]
            @tensor theta[:] := gL[-1,2,-2,5]*gR[5,3,-5,-6]*block[-4,-3,3,2]
        end

        # @tensor theta[:] := gL[-1,2,-2,5]*gR[5,3,-5,-6]*block[-4,-3,3,2]

        #slow @tensor theta[:] := lL[-1,1]*gL[1,-2,-4,4]*lM[4,-3]
        #slow @tensor theta[:] := theta[-1,2,5,-2]*gR[5,3,-5,6]*lR[6,-6]*block[-4,-3,3,2]
        st = size(theta)
        theta = reshape(theta,st[1]*st[2],st[3],st[4],st[5]*st[6])

    else
        @tensor theta[:] := gL[-1,2,5]*gR[5,3,-4]*block[-3,-2,3,2]
        #slow  @tensor theta[:] := lL[-1,1]*gL[1,-2,4]*lM[4,-3]
        #slow @tensor theta[:] := theta[-1,2,5]*gR[5,3,6]*lR[6,-4]*block[-3,-2,3,2]
        #slowest @time @tensor theta[:] := lL[-1,1]*gL[1,2,4]*lM[4,5]*gR[5,3,6]*lR[6,-4]*block[-2,-3,2,3]
    end

    D1l,d,d,D2r = size(theta)
    theta = reshape(theta, D1l*d,d*D2r)
    U,S,V = svd(theta, thin=true)
    V = V'
    D1 = size(S)[1] # number of singular values
    U,S,V,D1,err = truncate_svd(U,S,V,Dmax,tol)
    if length(size(gL))==4
        U=reshape(U,st[1],st[2],d,D1)
        V=reshape(V,D1,d,st[5],st[6])
        ilL = spdiagm(1./diag(lL))
        ilR = spdiagm(1./diag(lR))
        U = permutedims(sparse_l(U,ilL,:left),[1 3 2 4])
        V = sparse_l(V,ilR,:right)
        # @tensor U[:] := inv(lL)[-1,1]*U[1,-3,-2,-4]
        # @tensor V[:] := V[-1,-2,-3,3]*inv(lR)[3,-4]
    else
        U=reshape(U,D1l,d,D1)
        V=reshape(V,D1,d,D2r)
        ilL = spdiagm(1./diag(lL))
        ilR = spdiagm(1./diag(lR))
        U = sparse_l(U,ilL,:left)
        V = sparse_l(V,ilR,:right)
        # @tensor U[:] := inv(lL)[-1,1]*U[1,-2,-3]
        # @tensor V[:] := V[-1,-2,3]*inv(lR)[3,-3]
    end

    return U,spdiagm(S),V, err
end

function localOpExp(g,l,op,site)
    theta = sparse_l(g[site],l[site],:left)
    theta = sparse_l(theta,l[site+1],:right)
    if length(size(g[1]))==3
        # @tensor theta[:] := l[site][-1,1]*g[site][1,-2,3]*l[site+1][3,-3]
        @tensor r[:] :=theta[1,2,3]*op[4,2]*conj(theta[1,4,3])
    elseif length(size(g[1]))==4
        # @tensor theta[:] := l[site][-1,1]*g[site][1,-2,-4,3]*l[site+1][3,-3]
        @tensor r[:] :=theta[1,2,3,5]*op[4,2]*conj(theta[1,4,3,5])
    end
    return r[1]
end

function gl_mpoExp(g,l,mpo)
    N = length(g)
    F = Array{Complex128,3}
    # F[1][1,1,1]=1
    @tensor F[-1,-2,-3] :=ones(1)[-2]*l[1][a,-3]*conj(l[1][a,-1])
    for k = 1:N
        if length(size(g[1]))==3
            @tensor F[-1,-2,-3] := F[1,2,3]*g[k][3,5,-3]*mpo[k][2,4,5,-2]*conj(g[k][1,4,-1])
        elseif length(size(g[1]))==4
            # @tensor F[:] := F[-1,2,3]*g[k][3,5,-6,-3]*mpo[k][2,-4,5,-2]
            # @tensor F[:] := F[1,-2,-3,4,6]*conj(g[k][1,4,6,-1])
            @tensor F[-1,-2,-3] := F[1,2,3]*g[k][3,5,6,-3]*mpo[k][2,4,5,-2]*conj(g[k][1,4,6,-1])
        end
        F=sparse_l(F,l[k+1],:right)
        F=sparse_l(F,l[k+1],:left)
        # @tensor F[:] := F[1,-2,3]*l[k+1][3,-3]*conj(l[k+1][1,-1])
    end
    @tensor F[:] := F[1,-1,1]
    return F[1]
end

function gl_scalarprod(gA,lA,gB,lB;flipB=false)
    N = length(gA)
    F = Array{Complex128,3}
    # F[1][1,1,1]=1
    @tensor F[-1,-2] := lA[1][a,-2]*conj(lB[1][a,-1])
    for k = 1:N
        if length(size(gA[1]))==3
            @tensor F[-1,-2] := F[1,2]*gA[k][2,5,-2]*conj(gB[k][1,5,-1])
        elseif length(size(gA[1]))==4
            # @tensor F[:] := F[-1,2,3]*g[k][3,5,-6,-3]*mpo[k][2,-4,5,-2]
            # @tensor F[:] := F[1,-2,-3,4,6]*conj(g[k][1,4,6,-1])
            if flipB
                @tensor F[-1,-2] := F[1,2]*gA[k][2,5,6,-2]*conj(gB[k][1,5,6,-1])
            else
                @tensor F[-1,-2] := F[1,2]*gA[k][2,5,6,-2]*conj(gB[k][1,6,5,-1])
            end
        end
        @tensor F[:] := F[1,2]*lA[k+1][2,-2]*conj(lB[k+1][1,-1])
    end
    @tensor F[:] := F[1,1]
    return F[1]
end

function isingHamBlocks(L,J,h,g)
    blocks = Array{Array{Complex128,2},1}(L)
    for i=1:L
        if i==1
            blocks[i] = J*ZZ + h/2*(2XI+IX) + g/2*(2*ZI+IZ)
        elseif i==L-1
            blocks[i] = J*ZZ + h/2*(XI+2IX) + g/2*(ZI+2*IZ)
        else
            blocks[i] = J*ZZ + h/2*(XI+IX) + g/2*(ZI+IZ)
        end
    end
    return blocks
end

function ops_on_gl_dummy(g,l,ops)
    # here assuming second index as dummy to represent MPS as MPO
    N = length(g)
    for k=1:N
        @tensor g[k][:] := g[k][-1,-2,3,-4]*ops[k][-3,3]
    end
    return g
end

function ops_on_gl(g,l,ops)
    N = length(g)
    fourlegs=false
    if length(size(g[1]))==4
        fourlegs=true
    end
    for k=1:N
        if fourlegs
            @tensor g[k][:] := g[k][-1,2,-3,-4]*ops[k][-2,2]
        else
            @tensor g[k][:] := g[k][-1,2,-3]*ops[k][-2,2]
        end
    end
    return g
end

function twositeop_on_gl(g,l,op,m,D,tol)
    # m = chain position (m,m+1)
    g[m], l[m+1], g[m+1], error = updateBlock(l[m],l[m+1],l[m+2],g[m],g[m+1],op, D, tol)
    return g, l
end

#UNFINISHED
function mpo_on_gl(g,l,mpo)
    N = length(g)
    if length(size(g[1]))==3
        glmps = gl_to_mps(g,l)
        for i=1:N
            glmps[i] = MPS.absorb_mpo_in_mps(glmps[i],mpo[i])
        end
        return glmps
    elseif length(size(g[1]))==4
        glmpo = gl_to_mpo(g,l)
        return multiplyMPOs(mpo,glmpo)
    end
end

function check_canon(g,l)
    N = length(g)
    for k=1:N
        @tensor D[:]:= l[k][1,2]*g[k][2,3,-2]*conj(l[k][1,4])*conj(g[k][4,3,-1])
        println("L: ",real(det(D)),"_",real(trace(D))/size(D)[1])
        @tensor D[:]:= l[k+1][2,1]*g[k][-2,3,2]*conj(l[k+1][4,1])*conj(g[k][-1,3,4])
        println("R: ",real(det(D)),"_",real(trace(D))/size(D)[1])
    end
end

function gl_quench(N,time,steps,maxD,tol,inc::Int64)
    J0=1
    h0=1
    g0=0
    q = 2*pi*(3/(N-1))
    hamblocksTH(time) = isingHamBlocks(N,J0,h0,g0)
    hamblocks(time) = isingHamBlocks(N,J0,h0,g0)
    opEmpo = MPS.IsingMPO(N,J0,h0,g0)
    opE(time,g,l) = gl_mpoExp(g,l,opEmpo)
    opmag(time,g,l) = localOpExp(g,l,sx,Int(floor(N/2)))
    opnorm(time,g,l) = gl_mpoExp(g,l,MPS.IdentityMPO(N,2))
    ops = [opE opmag opnorm]
    mpo = MPS.IdentityMPO(N,2)
    # mps = mpo_to_mps(mpo)
    mps = MPS.randomMPS(N,2,5)
    g,l = prepareGL(mpo,maxD)
    # check_canon(g,l)
    # pert_ops = fill(expm(1e-3*im*sx),N)
    pert_ops = [expm(1e-3*sx*im*x) for x in sin.(q*(-1+(1:N)))]
    @time opvals, err = gl_tebd(g,l,hamblocksTH,-2*im,10,maxD,ops,tol=tol,increment=inc,st2=true)
    # check_canon(g,l)
    # println(l[5])
    ops_on_gl(g,l,pert_ops)
    @time opvals, err = gl_tebd(g,l,hamblocks,time,steps,maxD,ops,tol=tol,increment=inc,st2=true)
    # check_canon(g,l)
    return opvals, err, size(g[Int(floor(N/2))])
end

function gl_tebd(g,l, hamblocks, total_time, steps, D, operators; tol=0, increment::Int64=1, st2::Bool=false)
    ### block = hamiltonian
    ### use -im*total_time for imaginary time evolution
    ### assumption: start with rightcanonical mps
	### eth = (true,E1,hamiltonian) --> do ETH calcs if true for excited energy E1 wrt hamiltonian
    stepsize = total_time/steps
    nop = length(operators)
    opvalues = Array{Complex128,2}(1+Int(floor(steps/increment)),nop+1)
    err = Array{Complex128,1}(1+Int(floor(steps/increment)))
    datacount=1
    opvalues[datacount,1] = 0 # =t(0)
    err[datacount] = 0
    for k = 1:nop
        opvalues[datacount,k+1] = operators[k](0.0,g,l)
    end

    for counter = 1:steps
        time = counter*total_time/steps
        if !st2
            err_tmp = gl_tebd_step(g,l,hamblocks(time),stepsize,D,tol=tol)
        elseif st2
            err_tmp =  gl_tebd_step_st2(g,l,hamblocks(time),stepsize,D,tol=tol)
        end
        if counter % increment == 0
            datacount+=1
            println("step ",counter," / ",steps)
            opvalues[datacount,1] = time
            for k = 1:nop
                opvalues[datacount,k+1] = operators[k](time,g,l)
            end
            err[datacount] = err_tmp
        end
    end
    return opvalues, err
end


""" calculation of 2-Renyi entropy density in time evolution """
function gl_tebd_renyi(g,l, hamblocks, total_time, steps, D; tol=0, increment::Int64=1, st2::Bool=false)
    stepsize = total_time/steps
    times = Array{Complex128,1}(1+Int(floor(steps/increment)))
    err   = Array{Complex128,1}(1+Int(floor(steps/increment)))
    renyi = Array{Complex128,1}(1+Int(floor(steps/increment)))
    datacount=1
    times[datacount] = err[datacount] = 0

    N = length(g)
    if isodd(N) println("WARNING: choose integer N and subsystem sizes"); return 0 end
    Nmid = Int(N/2)
    # Nstart = Int.(Nmid - (subsizes/2-1))
    # Nend = Nstart + subsizes - 1
    # @threads for k = 1:length(subsizes)
    renyi[1] = -log(trace_rho_squared_block(g, l, Nmid, N))
    println("s2_i(0): ",renyi[1])
    # end

    for counter = 1:steps
        time = counter*total_time/steps
        if !st2
            err_tmp = gl_tebd_step(g,l,hamblocks(time),stepsize,D,tol=tol)
        elseif st2
            err_tmp =  gl_tebd_step_st2(g,l,hamblocks(time),stepsize,D,tol=tol)
        end
        if counter % increment == 0
            datacount+=1
            println("step ",counter," / ",steps)
            println("err: ",err_tmp)
            times[datacount] = time
            err[datacount] = err_tmp
            # @threads for k = 1:length(subsizes)
            renyi[datacount] = -log(trace_rho_squared_block(g, l, Nmid, N))
            println("s2_i: ",renyi[datacount])
            # end
        end
    end
    return times, err, renyi
end


function gl_ct!(g)
    N = length(g)
    for k=1:N
        g[k]=permutedims(conj(g[k]),[1 3 2 4])
    end
end
function gl_tebd_c(gA,lA,gB,lB, hamblocks, total_time, steps, D; tol=0, increment::Int64=1, st2::Bool=false, legflip::Bool=true, err_max=Inf)
    ### block = hamiltonian
    ### use -im*total_time for imaginary time evolution
    ### assumption: start with rightcanonical mps
	### eth = (true,E1,hamiltonian) --> do ETH calcs if true for excited energy E1 wrt hamiltonian

    stepsize = total_time/steps
    opvalues = Array{Complex128,1}(1+Int(floor(steps/increment)))
    t = Array{Float64,1}(1+Int(floor(steps/increment)))
    errA = Array{Complex128,1}(1+Int(floor(steps/increment)))
    errB = Array{Complex128,1}(1+Int(floor(steps/increment)))

    datacount=1
    t[datacount] = 0
    errA[datacount] = 0
    errB[datacount] = 0
    opvalues[datacount] = gl_scalarprod(gA,lA,gB,lB,flipB=legflip)

    for counter = 1:steps
        time = counter*total_time/steps
        if !st2
            errA_tmp = gl_tebd_step(gA,lA,hamblocks(time),stepsize,D,tol=tol, counter_evo=true) # should be gA,lA ?
            errB_tmp = gl_tebd_step(gB,lB,hamblocks(time),-stepsize,D,tol=tol, counter_evo=true)
        elseif st2
            errA_tmp =  gl_tebd_step_st2(gA,lA,hamblocks(time),stepsize,D,tol=tol, counter_evo=true)
            errB_tmp =  gl_tebd_step_st2(gB,lB,hamblocks(time),-stepsize,D,tol=tol, counter_evo=true)
        end
        ## stop loop at max error:
        if (errA_tmp > err_max || errB_tmp > err_max) break end

        if counter % increment == 0
            datacount+=1
            t[datacount] = 2*time
            errA[datacount] = errA_tmp
            errB[datacount] = errB_tmp
            opvalues[datacount] = gl_scalarprod(gA,lA,gB,lB,flipB=legflip)
            println("step ",counter," / ",steps)
            println("errA, errB: ",errA_tmp,", ",errB_tmp)
        end
    end
    return opvalues, errA,errB, t
end

""" coarsegrained two-site version: """
function gl_tebd_cg(gTH,lTH,gP,lP, hamblocks, op, total_time, steps, D; tol=0, increment::Int64=1, st2::Bool=true, legflip::Bool=true)
    ### op = 2-site coarsegrained operator

    stepsize = total_time/steps
    opvalues = Array{Complex128,1}(1+Int(floor(steps/increment)))
    t = Array{Float64,1}(1+Int(floor(steps/increment)))
    errP = Array{Complex128,1}(1+Int(floor(steps/increment)))

    datacount=1
    t[datacount] = 0
    errP[datacount] = 0
    opvalues[datacount] = gl_scalarprod_cg(gTH,lTH,gP,lP,op,D,tol)

    for counter = 1:steps
        time = counter*total_time/steps
        if !st2
            errP_tmp = gl_tebd_step(gP,lP,hamblocks(time),stepsize,D,tol=tol, counter_evo=true)
        elseif st2
            errP_tmp =  gl_tebd_step_st2(gP,lP,hamblocks(time),stepsize,D,tol=tol, counter_evo=true)
        end
        if counter % increment == 0
            datacount+=1
            t[datacount] = time
            errP[datacount] = errP_tmp
            opvalues[datacount] = gl_scalarprod_cg(gTH,lTH,gP,lP,op,D,tol)
            println("step ",counter," / ",steps)
        end
    end
    return opvalues, errP, t
end

function gl_scalarprod_cg(gTH,lTH,gP,lP,op,D,tol)
    N = length(gTH)
    trace_Re = Atomic{Float64}(0.0)
    trace_Im = Atomic{Float64}(0.0)

    ## trace over full state:
    @threads for k = 1:N-1 # Threads.@threads causes problems for parallel sum of complex numbers
        g_P,l_P = deepcopy(gP), deepcopy(lP)
        g_P,l_P = twositeop_on_gl(g_P,l_P,op,k,D,tol) # op is absorbed into perturbed state
        twosite_trace = gl_scalarprod(g_P,l_P,gTH,lTH,flipB=true)
        atomic_add!(trace_Re, real(twosite_trace))
        atomic_add!(trace_Im, imag(twosite_trace))
    end
    total_trace = trace_Re[] + im*trace_Im[]

    return total_trace
end


""" calculates Tr(rho) assuming M := Ma,Mb,(Mc) represent sqrt{rho} """
function trace_rho_average(M, L)
    if length(M)==2
        Ma,Mb = M
        la,lb = L
        norm_rho_AB = trace_rho_AB(Ma,Mb, lb)
        norm_rho_BA = trace_rho_AB(Mb,Ma, la)
        return (norm_rho_AB+norm_rho_BA)/2.0
    elseif length(M)==3
        Ma,Mb,Mc = M
        la,lb,lc = L
        norm_rho_ABC = trace_rho_ABC(Ma,Mb,Mc, lc)
        norm_rho_BCA = trace_rho_ABC(Mb,Mc,Ma, la)
        norm_rho_CAB = trace_rho_ABC(Mc,Ma,Mb, lb)
        return (norm_rho_ABC+norm_rho_BCA+norm_rho_CAB)/3.0
    end
end
function trace_rho_AB(Ma,Mb, lb)
    @tensor norm_AB[-1,-2] := lb[1,2]*conj(lb[1,3])*Ma[2,4,5,-1]*conj(Ma[3,4,5,-2])
    @tensor norm_AB[-1,-2] := norm_AB[1,2]*Mb[1,3,4,-1]*conj(Mb[2,3,4,-2])
    @tensor norm_AB[] := norm_AB[1,1]
    return norm_AB[1]
end
function trace_rho_ABC(Ma,Mb,Mc, lc)
    # @tensor norm_ABC[] := lc[1,2]*Ma[2,3,4,5]*Mb[5,6,7,8]*Mc[8,9,10,11]*conj(lc[1,12])*conj(Ma[12,3,4,13])*conj(Mb[13,6,7,14])*conj(Mc[14,9,10,11])
    @tensor norm_ABC[-1,-2] := lc[1,2]*conj(lc[1,3])*Ma[2,4,5,-1]*conj(Ma[3,4,5,-2])
    @tensor norm_ABC[-1,-2] := norm_ABC[1,2]*Mb[1,3,4,-1]*conj(Mb[2,3,4,-2])
    @tensor norm_ABC[-1,-2] := norm_ABC[1,2]*Mc[1,3,4,-1]*conj(Mc[2,3,4,-2])
    @tensor norm_ABC[] := norm_ABC[1,1]
    return norm_ABC[1]
end

""" calculates Tr(rho^2) assuming Ma,Mb,(Mc) represent sqrt{rho} """
function trace_rho_squared_average(M, L)
    if length(M)==2
        Ma,Mb = M
        la,lb = L
        norm_rho_AB = trace_rho_squared_AB(Ma,Mb, lb)
        norm_rho_BA = trace_rho_squared_AB(Mb,Ma, la)
        return (norm_rho_AB+norm_rho_BA)/2.0
    elseif length(M)==3
        Ma,Mb,Mc = M
        la,lb,lc = L
        norm_rho_ABC = trace_rho_squared_ABC(Ma,Mb,Mc, lc)
        norm_rho_BCA = trace_rho_squared_ABC(Mb,Mc,Ma, la)
        norm_rho_CAB = trace_rho_squared_ABC(Mc,Ma,Mb, lb)
        return (norm_rho_ABC+norm_rho_BCA+norm_rho_CAB)/3.0
    end
end
function trace_rho_squared_AB(Ma,Mb, lb)
    @tensor norm2_AB[-1,-2,-3,-4] := lb[1,3]*conj(lb[1,6])*conj(lb[2,4])*lb[2,5]*Ma[3,7,10,-1]*conj(Ma[4,7,8,-2])*Ma[5,9,8,-3]*conj(Ma[6,9,10,-4])
    @tensor norm2_AB[-1,-2,-3,-4] := norm2_AB[1,2,3,4]*Mb[1,5,6,-1]*conj(Mb[2,5,7,-2])*Mb[3,8,7,-3]*conj(Mb[4,8,6,-4])
    @tensor norm2_AB[] := norm2_AB[1,2,2,1]
    return norm2_AB[1]
end
function trace_rho_squared_ABC(Ma,Mb,Mc, lc)
    @tensor norm2_ABC[-1,-2,-3,-4] := lc[1,3]*conj(lc[1,6])*conj(lc[2,4])*lc[2,5]*Ma[3,7,10,-1]*conj(Ma[4,7,8,-2])*Ma[5,9,8,-3]*conj(Ma[6,9,10,-4])
    @tensor norm2_ABC[-1,-2,-3,-4] := norm2_ABC[1,2,3,4]*Mb[1,5,6,-1]*conj(Mb[2,5,7,-2])*Mb[3,8,7,-3]*conj(Mb[4,8,6,-4])
    @tensor norm2_ABC[-1,-2,-3,-4] := norm2_ABC[1,2,3,4]*Mc[1,5,6,-1]*conj(Mc[2,5,7,-2])*Mc[3,8,7,-3]*conj(Mc[4,8,6,-4])
    @tensor norm2_ABC[] := norm2_ABC[1,2,2,1]
    return norm2_ABC[1]
end
""" calculates Tr(rho^2) for block of finite size; assuming G,L are double canonical """
function trace_rho_squared_block(G, L, ind_min, ind_max)
    @tensor M1[-1,-2,-3,-4] := G[ind_min][-1,-2,-3,1]*L[ind_min+1][1,-4]
    @tensoropt (1,2,3,4,-1,-2,-3,-4) norm2[-1,-2,-3,-4] := (L[ind_min].^2)[1,2]*(L[ind_min].^2)[3,4]*M1[1,7,10,-1]*conj(M1[2,7,8,-2])*M1[3,9,8,-3]*conj(M1[4,9,10,-4])

    for i = ind_min+1:ind_max
        @tensor M[-1,-2,-3,-4] := G[i][-1,-2,-3,1]*L[i+1][1,-4]
        @tensoropt (1,2,3,4,-1,-2,-3,-4) norm2[-1,-2,-3,-4] := norm2[1,2,3,4]*M[1,5,6,-1]*conj(M[2,5,7,-2])*M[3,8,7,-3]*conj(M[4,8,6,-4])
    end
    @tensor norm2[] := norm2[1,1,2,2]
    return norm2[1]
end


""" calculate rate functions from mixed transfer matrix (assuming MPS) """
function calculate_rate_functions(M1a,M1b, M2a,M2b, num_levels)
    D1, D2 = size(M1a,1), size(M2a,1)
    yR_func(vec) = reshape(tmat_r(M1a,M1b, M2a,M2b, reshape(vec, D1,D2)), D1*D2)
    yR_linmap = LinearMap{Complex128}(yR_func, D1*D2)
    domR = eigs(yR_linmap, nev=num_levels, which=:LM, ritzvec=false)[1] # dominant eigenvalues
    r_i = -2*log.(abs.(domR))
    return r_i
end
""" action of mixed MPS transfer matrix on vector """
function tmat_r(M1a,M1b, M2a,M2b, xR)
    @tensor yR[-1,-2] := M1b[-1,1,2]*conj(M2b[-2,1,3])*xR[2,3]
    @tensor yR[-1,-2] := M1a[-1,3,1]*conj(M2a[-2,3,2])*yR[1,2]
    return yR
end

""" calculates 2-Renyi entropy density s2=-log(Tr{rho^2})/N from right-dominant eigenvalue of transferoperator of rho^2 """
function calculate_2Renyi_entropy(M)
    D = size(M[1])[1]
    if D==1
        T = top_full(M)
        T = reshape(T, D^4,D^4)
        F = eigfact(T)
        domR = F[:values] # dominant eigenvalue
        s2 = -log(domR[1])
    else
        yR_func(vec) = reshape(top_r(M, reshape(vec, D,D,D,D)), D^4)
        yR_linmap = LinearMap{Complex128}(yR_func, D^4)
        domR = eigs(yR_linmap, nev=1, which=:LM, ritzvec=false)[1] # dominant eigenvalue
        s2 = -log(domR[1])
    end
    return s2
end
""" transferoperator of rho^2 of a 2-site or 3-site unit cell acting on right dominant eigenvector """
function top_r(M, xR)
    if length(M)==2
        Ma,Mb = M
        @tensor yR[-1,-2,-3,-4] := Mb[-1,5,6,1]*conj(Mb[-2,5,7,2])*Mb[-3,8,7,3]*conj(Mb[-4,8,6,4])*xR[1,2,3,4] # = B*xR
        @tensor yR[-1,-2,-3,-4] := Ma[-1,5,6,1]*conj(Ma[-2,5,7,2])*Ma[-3,8,7,3]*conj(Ma[-4,8,6,4])*yR[1,2,3,4] # = A*B*xR
    elseif length(M)==3
        Ma,Mb,Mc = M
        # @tensor AB[-1,-2,-3,-4,-5,-6] := Ma[-1,-2,-4,1]*Mb[1,-3,-5,-6]
        # @tensor ABC[-1,-2,-3,-4,-5,-6,-7,-8] := AB[-1,-2,-3,-5,-6,1]*Mc[1,-4,-7,-8]
        # @tensor ABC2[-1,-2,-3,-4,-5,-6,-7,-8,-9,-10] := ABC[-1,1,2,3,-6,-7,-8,-9]*conj(ABC[-2,1,2,3,-3,-4,-5,-10])
        # @tensor ABC2xR[-1,-2,-3,-4,-5,-6,-7,-8,-9,-10] := ABC2[-1,-2,-3,-4,-5,-6,-7,-8,1,2]*xR[-9,-10,1,2]
        # @tensor yR[-1,-2,-3,-4] := ABC2[-1,-2,3,4,5,6,7,8,1,2]*ABC2xR[-3,-4,6,7,8,3,4,5,1,2]

        @tensor yR[-1,-2,-3,-4] := Mc[-1,5,6,1]*conj(Mc[-2,5,7,2])*Mc[-3,8,7,3]*conj(Mc[-4,8,6,4])*xR[1,2,3,4] # = C*xR
        @tensor yR[-1,-2,-3,-4] := Mb[-1,5,6,1]*conj(Mb[-2,5,7,2])*Mb[-3,8,7,3]*conj(Mb[-4,8,6,4])*yR[1,2,3,4] # = B*C*xR
        @tensor yR[-1,-2,-3,-4] := Ma[-1,5,6,1]*conj(Ma[-2,5,7,2])*Ma[-3,8,7,3]*conj(Ma[-4,8,6,4])*yR[1,2,3,4] # = A*B*C*xR

        # @tensor M2[-1,-2,-3,-4,-5,-6] := Mc[-1,1,-4,-5]*conj(Mc[-2,1,-3,-6])
        # @tensor M2x[-1,-2,-3,-4,-5,-6] := M2[-1,-2,-3,-4,1,2]*xR[-5,-6,1,2]
        # @tensor yR[-1,-2,-3,-4] := M2[-1,-2,2,1,3,4]*M2x[-3,-4,1,2,3,4]
        # @tensor M2[-1,-2,-3,-4,-5,-6] := Mb[-1,1,-4,-5]*conj(Mb[-2,1,-3,-6])
        # @tensor yR[-1,-2,-3,-4] := yR[1,2,3,4]*M2[-1,-2,6,5,1,2]*M2[-3,-4,5,6,3,4]
        # @tensor M2[-1,-2,-3,-4,-5,-6] := Ma[-1,1,-4,-5]*conj(Ma[-2,1,-3,-6])
        # @tensor yR[-1,-2,-3,-4] := yR[1,2,3,4]*M2[-1,-2,6,5,1,2]*M2[-3,-4,5,6,3,4]
    end
    return yR
end
""" calculate transferoperator explicitly """
function top_full(M)
    if length(M)==2
        Ma,Mb = M
        @tensor top[-1,-2,-3,-4,-5,-6,-7,-8] := Ma[-1,1,2,-5]*conj(Ma[-2,1,3,-6])*Ma[-3,4,3,-7]*conj(Ma[-4,4,2,-8]) # = A
        @tensor top[-1,-2,-3,-4,-5,-6,-7,-8] := top[-1,-2,-3,-4,1,2,3,4]*Mb[1,5,6,-5]*conj(Mb[2,5,7,-6])*Mb[3,8,7,-7]*conj(Mb[4,8,6,-8]) # = A*B
    elseif length(M)==3
        Ma,Mb,Mc = M
        @tensor top[-1,-2,-3,-4,-5,-6,-7,-8] := Ma[-1,1,2,-5]*conj(Ma[-2,1,3,-6])*Ma[-3,4,3,-7]*conj(Ma[-4,4,2,-8]) # = A
        @tensor top[-1,-2,-3,-4,-5,-6,-7,-8] := top[-1,-2,-3,-4,1,2,3,4]*Mb[1,5,6,-5]*conj(Mb[2,5,7,-6])*Mb[3,8,7,-7]*conj(Mb[4,8,6,-8]) # = A*B
        @tensor top[-1,-2,-3,-4,-5,-6,-7,-8] := top[-1,-2,-3,-4,1,2,3,4]*Mc[1,5,6,-5]*conj(Mc[2,5,7,-6])*Mc[3,8,7,-7]*conj(Mc[4,8,6,-8]) # = A*B*C
    end
    return top
end

""" calculates left and right-dominant eigenvalues and eigenvectors of two sites M1,M2 """
function tm_eigs_dominant(M1a,M1b,M1c, M2a=M1a,M2b=M1b,M2c=M1c)
    D1l = size(M1c)[4]; D2l = size(M2c)[4]
    D1r = size(M1a)[1]; D2r = size(M2a)[1]

    yL_func(vec) = reshape(tm_l(M1a,M1b,M1c, M2a,M2b,M2c, reshape(vec, D1l,D2l)), D1l*D2l)
    yL_linmap = LinearMap{Complex128}(yL_func, D1l*D2l)
    eval_L, evec_L = eigs(yL_linmap, nev=1, which=:LM)

    yR_func(vec) = reshape(tm_r(M1a,M1b,M1c, M2a,M2b,M2c, reshape(vec, D1r,D2r)), D1r*D2r)
    yR_linmap = LinearMap{Complex128}(yR_func, D1r*D2r)
    eval_R, evec_R = eigs(yR_linmap, nev=1, which=:LM)

    # T = tm_full(M1a,M1b,M1c, M2a,M2b,M2c)
    # T = reshape(T, D1r*D2r,D1l*D2l)
    # eval_R, evec_R = eigs(T, nev=1, which=:LM)
    # eval_L, evec_L = eigs(T', nev=1, which=:LM)
    return eval_L[1], reshape(evec_L[:,1],D1l,D2l), eval_R[1], reshape(evec_R[:,1],D1r,D2r)
end
""" transfermatrix between two 3-site pairs (unit cells) acting on left dominant eigenvector """
function tm_l(M1a,M1b,M1c, M2a,M2b,M2c, xL)
    # @tensor yL[-1,-2] := xL[1,2]*M1a[1,3,4,5]*M1b[5,6,7,8]*M1c[8,9,10,-1]*conj(M2a[2,3,4,11])*conj(M2b[11,6,7,12])*conj(M2c[12,9,10,-2]) # too slow!
    @tensor yL[-1,-2]   := xL[1,2]*M1a[1,3,4,-1]*conj(M2a[2,3,4,-2]) # = xL*A
    @tensor yL[-1,-2] := yL[1,2]*M1b[1,3,4,-1]*conj(M2b[2,3,4,-2]) # = xL*A*B
    @tensor yL[-1,-2]  := yL[1,2]*M1c[1,3,4,-1]*conj(M2c[2,3,4,-2]) # = xL*A*B*C
    return yL
end
""" transfermatrix between two 3-site pairs (unit cells) acting on right dominant eigenvector """
function tm_r(M1a,M1b,M1c, M2a,M2b,M2c, xR)
    # @tensor yR[-1,-2] := M1a[-1,1,2,3]*M1b[3,4,5,6]*M1c[6,7,8,9]*conj(M2a[-2,1,2,11])*conj(M2b[11,4,5,12])*conj(M2c[12,7,8,10])*xR[9,10] # too slow!
    @tensor yR[-1,-2] := M1c[-1,3,4,1]*conj(M2c[-2,3,4,2])*xR[1,2] # = C*xR
    @tensor yR[-1,-2] := M1b[-1,3,4,1]*conj(M2b[-2,3,4,2])*yR[1,2] # = B*C*xR
    @tensor yR[-1,-2] := M1a[-1,3,4,1]*conj(M2a[-2,3,4,2])*yR[1,2] # = A*B*C*xR
    return yR
end
""" calculate transfermatrix explicitly"""
function tm_full(M1a,M1b,M1c, M2a=M1a,M2b=M1b,M2c=M1c)
    # @tensor tm[-1,-2,-3,-4] := M1a[-1,1,2,3]*M1b[3,4,5,6]*M1c[6,7,8,-3]*conj(M2a[-2,1,2,9])*conj(M2b[9,4,5,10])*conj(M2c[10,7,8,-4]) # too slow!
    @tensor tm[-1,-2,-3,-4] := M1a[-1,1,2,-3]*conj(M2a[-2,1,2,-4]) # = A*A'
    @tensor tm[-1,-2,-3,-4] := tm[-1,-2,1,2]*M1b[1,3,4,-3]*conj(M2b[2,3,4,-4]) # = A*A'*B*B'
    @tensor tm[-1,-2,-3,-4] := tm[-1,-2,1,2]*M1c[1,3,4,-3]*conj(M2c[2,3,4,-4]) # = A*A'*B*B'*C*C'
    return tm
end

""" check double canonical structure of Γ,λ in A,B unit cell """
function check_gl_canonical(G, L)
    if length(G)==2
        ΓA, ΓB = G
        λA, λB = L
        # site A:
        Γ,λ = ΓA,λA
        @tensor IdrA[:]:=conj(Γ[-1,4,1])*Γ[-2,4,2]*λ[1,3]*λ[2,3]
        println("A rcan = ",IdrA≈eye(size(IdrA,1)),"\t",vecnorm(IdrA-eye(size(IdrA,1))))
        Γ,λ = ΓA,λB
        @tensor IdlA[:]:=conj(Γ[1,4,-1])*Γ[2,4,-2]*λ[3,1]*λ[3,2]
        println("A lcan = ",IdlA≈eye(size(IdlA,1)),"\t",vecnorm(IdlA-eye(size(IdlA,1))))
        # site B
        Γ,λ = ΓB,λB
        @tensor IdrB[:]:=conj(Γ[-1,4,1])*Γ[-2,4,2]*λ[1,3]*λ[2,3]
        println("B rcan = ",IdrB≈eye(size(IdrB,1)),"\t",vecnorm(IdrB-eye(size(IdrB,1))))
        Γ,λ = ΓB,λA
        @tensor IdlB[:]:=conj(Γ[1,4,-1])*Γ[2,4,-2]*λ[3,1]*λ[3,2]
        println("B lcan = ",IdlB≈eye(size(IdlB,1)),"\t",vecnorm(IdlB-eye(size(IdlB,1))))
    end
end

""" check if all 3 sites A,B,C in unit cell are left- and rightcanonical """
function check_triple_canonical(M, L)
    if length(M)==1
        Ma = M[1]
        la = L[1]
        ## rightcanonical:
        @tensor Id[-1,-2] := Ma[-1,1,2,3]*conj(Ma[-2,1,2,3])
        println("A rcan = ",Id≈eye(size(Ma)[1]),"\t",vecnorm(Id-eye(size(Ma)[1])))
        ## leftcanonical:
        @tensor gA[-1,-2,-3,-4] := Ma[-1,-2,-3,1]*inv(la)[1,-4]
        @tensor Id[-1,-2] := la[1,2]*conj(la[1,5])*gA[2,3,4,-1]*conj(gA[5,3,4,-2])
        println("A lcan = ",Id≈eye(size(gA)[4]),"\t",vecnorm(Id-eye(size(gA)[4])))
    elseif length(M)==2
        Ma,Mb = M
        la,lb = L
        ## rightcanonical:
        @tensor Id[-1,-2] := Ma[-1,1,2,3]*conj(Ma[-2,1,2,3])
        println("A rcan = ",Id≈eye(size(Ma)[1]),"\t",vecnorm(Id-eye(size(Ma)[1])))
        @tensor Id[-1,-2] := Mb[-1,1,2,3]*conj(Mb[-2,1,2,3])
        println("B rcan = ",Id≈eye(size(Mb)[1]),"\t",vecnorm(Id-eye(size(Mb)[1])))
        ## leftcanonical:
        @tensor gA[-1,-2,-3,-4] := Ma[-1,-2,-3,1]*inv(la)[1,-4]
        @tensor Id[-1,-2] := lb[1,2]*conj(lb[1,5])*gA[2,3,4,-1]*conj(gA[5,3,4,-2])
        println("A lcan = ",Id≈eye(size(gA)[4]),"\t",vecnorm(Id-eye(size(gA)[4])))
        @tensor gB[-1,-2,-3,-4] := Mb[-1,-2,-3,1]*inv(lb)[1,-4]
        @tensor Id[-1,-2] := la[1,2]*conj(la[1,5])*gB[2,3,4,-1]*conj(gB[5,3,4,-2])
        println("B lcan = ",Id≈eye(size(gB)[4]),"\t",vecnorm(Id-eye(size(gB)[4])))
    elseif length(M)==3
        Ma,Mb,Mc = M
        la,lb,lc = L
        ## rightcanonical:
        @tensor Id[-1,-2] := Ma[-1,1,2,3]*conj(Ma[-2,1,2,3])
        println("A rcan = ",Id≈eye(size(Ma)[1]),"\t",vecnorm(Id-eye(size(Ma)[1])))
        @tensor Id[-1,-2] := Mb[-1,1,2,3]*conj(Mb[-2,1,2,3])
        println("B rcan = ",Id≈eye(size(Mb)[1]),"\t",vecnorm(Id-eye(size(Mb)[1])))
        @tensor Id[-1,-2] := Mc[-1,1,2,3]*conj(Mc[-2,1,2,3])
        println("C rcan = ",Id≈eye(size(Mc)[1]),"\t",vecnorm(Id-eye(size(Mc)[1])))
        ## leftcanonical:
        @tensor gA[-1,-2,-3,-4] := Ma[-1,-2,-3,1]*inv(la)[1,-4]
        @tensor Id[-1,-2] := lc[1,2]*conj(lc[1,5])*gA[2,3,4,-1]*conj(gA[5,3,4,-2])
        println("A lcan = ",Id≈eye(size(gA)[4]),"\t",vecnorm(Id-eye(size(gA)[4])))
        @tensor gB[-1,-2,-3,-4] := Mb[-1,-2,-3,1]*inv(lb)[1,-4]
        @tensor Id[-1,-2] := la[1,2]*conj(la[1,5])*gB[2,3,4,-1]*conj(gB[5,3,4,-2])
        println("B lcan = ",Id≈eye(size(gB)[4]),"\t",vecnorm(Id-eye(size(gB)[4])))
        @tensor gC[-1,-2,-3,-4] := Mc[-1,-2,-3,1]*inv(lc)[1,-4]
        @tensor Id[-1,-2] := lb[1,2]*conj(lb[1,5])*gC[2,3,4,-1]*conj(gC[5,3,4,-2])
        println("C lcan = ",Id≈eye(size(gC)[4]),"\t",vecnorm(Id-eye(size(gC)[4])))
    end
end

""" triple canonicalize the 3-site unit cell """
function triple_canonicalize(Ma,Mb,Mc, Dmax; tol=0)
    ## normalize sites A,B,C and l and r dominant eigenvectors such that <l|r>=1:
    domL,l, domR,r = tm_eigs_dominant(Ma,Mb,Mc)
    println("domL, domR = ", domL,", ",domR)
    Ma ./= domL^(-1/6); Mb ./= domL^(-1/6); Mc ./= domL^(-1/6);
    r_tr = trace(r)
    phase_r = r_tr/abs(r_tr)
    r ./= phase_r
    l_tr = trace(l)
    phase_l = l_tr/abs(l_tr)
    l ./= phase_l
    @tensor n[] := l[1,2]*r[1,2]
    n = n[1]
    abs_n = abs(n)
    phase_n = n/abs_n
    (phase_n ≉ 1) && warn("In triple_canonicalize phase_n = ", phase_n, " ≉ 1")
    sfac = sqrt(abs_n)
    l ./= sfac
    r ./= sfac

    ## canonicalize the unit cell blob ABC:
    sA=size(Ma); sB=size(Mb); sC=size(Mc);
    Ma=reshape(Ma, sA[1],sA[2]*sA[3],sA[4]); Mb=reshape(Mb, sB[1],sB[2]*sB[3],sB[4]); Mc=reshape(Mc, sC[1],sC[2]*sC[3],sC[4]);
    @tensor theta[-1,-2,-3,-4,-5] := Ma[-1,-2,1]*Mb[1,-3,2]*Mc[2,-4,-5]
    st = size(theta)
    theta = reshape(theta, st[1],st[2]*st[3]*st[4],st[5])
    Gamma, lC = iTEBD2.canonical_form(theta, l, r)
    lambda_C = diagm(lC)

    # l_H = 0.5*(l + l')
    # r_H = 0.5*(r + r')
    # X = sqrt.(l_H)
    # YT = transpose(sqrt.(r_H))
    # U,S,V = svd(YT*X, thin=true)
    # S = diagm(S)
    # @tensor theta[-1,-2,-3,-4,-5,-6,-7,-8] := inv(S)[-1,1]*U'[1,2]*YT[2,3]*Ma[3,-2,-3,4]*Mb[4,-4,-5,5]*Mc[5,-6,-7,6]*X[6,7]*V[7,-8]

    ## contract entire unit cell including lambda_C:
    @tensor X_ABC[-1,-2,-3] := lambda_C[-1,1]*Gamma[1,-2,2]*lambda_C[2,-3]
    sXabc = size(X_ABC)
    X_ABC = reshape(X_ABC, sXabc[1],2,2,2,2,2,2,sXabc[3])

    ## split into 3 sites as in gl_iTEBD3_thirdstep:
    sXabc = size(X_ABC)
    X_ABC = reshape(X_ABC, sXabc[1]*sXabc[2]*sXabc[3]*sXabc[4]*sXabc[5], sXabc[6]*sXabc[7]*sXabc[8])
    ## 1st svd (right side)
    U,S,V = svd(X_ABC, thin=true)
    V = V'
    U,S,V,Dr,errR = truncate_svd(U,S,V,Dmax,tol)
    ## new tensors:
    lambda_B = diagm(S)
    M_C = reshape(V, Dr,sXabc[6],sXabc[7],sXabc[8])
    ## 2nd svd (left side)
    X_AB = U*lambda_B
    X_AB = reshape(X_AB, sXabc[1]*sXabc[2]*sXabc[3], sXabc[4]*sXabc[5]*Dr)
    U,S,V = svd(X_AB, thin=true)
    V = V'
    U,S,V,Dl,errL = truncate_svd(U,S,V,Dmax,tol)
    ## new tensors:
    lambda_A = diagm(S)
    M_B = reshape(V, Dl,sXabc[4],sXabc[5],Dr)
    ## reconstruct M_A
    Uab_la = U*lambda_A
    M_A = reshape(inv(lambda_C)*reshape(Uab_la, sXabc[1],sXabc[2]*sXabc[3]*Dl), sXabc[8],sXabc[2],sXabc[3],Dl) # explicit inverse numerically unstable?

    return l,r, Gamma, M_A,M_B,M_C, lambda_A,lambda_B,lambda_C
end


""" thermal correlator Tr(MPO1*op*MPO2') for 3-site unit cell; the sum over all op positions is returned;
    M1,M2 represent the states """
function expect_correlator_sum(M1a,M1b,M1c, M2a,M2b,M2c, op)
    expect_ABC = expect_correlator_ABC(M1a,M1b,M1c, M2a,M2b,M2c, op)
    expect_BCA = expect_correlator_ABC(M1b,M1c,M1a, M2b,M2c,M2a, op)
    expect_CAB = expect_correlator_ABC(M1c,M1a,M1b, M2c,M2a,M2b, op)
    return expect_ABC+expect_BCA+expect_CAB
end
function expect_correlator_ABC(M1a,M1b,M1c, M2a,M2b,M2c, op; printmessage=false)
    ## calculate left- and right-dominant eigenvalues/-vectors of mixed transfer matrix:
    domL,yL, domR,yR = tm_eigs_dominant(M1a,M1b,M1c, M2a,M2b,M2c)
    if printmessage println("dom eval L,R = ", domL,", ",domR) end
    # l_tr = trace(yL)
    # phase_l = l_tr/abs(l_tr)
    # yL ./= phase_l
    # r_tr = trace(yR)
    # phase_r = r_tr/abs(r_tr)
    # yR ./= phase_r
    @tensor norm[] := yL[1,2]*yR[1,2]
    phase = norm[1]/abs(norm[1])

    @tensor expect[] := yL[1,2]*M1a[1,3,4,5]*M1b[5,6,7,8]*M1c[8,9,10,11]*op[3,6,9,13,15,17]*conj(M2a[2,13,4,14])*conj(M2b[14,15,7,16])*conj(M2c[16,17,10,12])*yR[11,12]

    # ## calculate norms of individual MPOs: Tr(M1), Tr(M2)
    # domL1,yL1, domR1,yR1 = tm_eigs_dominant(M1a,M1b,M1c)
    # l_tr1 = trace(yL1)
    # phase_l1 = l_tr1/abs(l_tr1)
    # # yL1 ./= phase_l1
    # r_tr1 = trace(yR1)
    # phase_r1 = r_tr1/abs(r_tr1)
    # # yR1 ./= phase_r1
    # @tensor norm1[] := yL1[1,2]*yR1[1,2]
    #
    # domL2,yL2, domR2,yR2 = tm_eigs_dominant(M2a,M2b,M2c)
    # l_tr2 = trace(yL2)
    # phase_l2 = l_tr2/abs(l_tr2)
    # # yL2 ./= phase_l2
    # r_tr2 = trace(yR2)
    # phase_r2 = r_tr2/abs(r_tr2)
    # # yR2 ./= phase_r2
    # @tensor norm2[] := yL2[1,2]*yR2[1,2]

    # return expect[1]*domR / ( (phase) * (domL1*phase_r1*phase_l1*abs(norm1[1])) * (domL2*phase_r2*phase_l2*abs(norm2[1])) )
    return expect[1] / ( domR^3 * norm[1] )
end


""" thermal expectation value Tr(rho*op) for 2-site or 3-site unit cell;
    M := Ma,Mb,(Mc) represent sqrt{rho} """
function expect_operator_average(M, L, op, power=1)
    if length(M)==2
        Ma,Mb = M
        la,lb = L
        expect_AB = expect_operator_AB(Ma,Mb, lb, op, power)
        expect_BA = expect_operator_AB(Mb,Ma, la, op, power)
        return (expect_AB+expect_BA)/2.0
    elseif length(M)==3
        Ma,Mb,Mc = M
        la,lb,lc = L
        expect_ABC = expect_operator_ABC(Ma,Mb,Mc, lc, op)
        expect_BCA = expect_operator_ABC(Mb,Mc,Ma, la, op)
        expect_CAB = expect_operator_ABC(Mc,Ma,Mb, lb, op)
        return (expect_ABC+expect_BCA+expect_CAB)/3.0
    end
end
function expect_operator_AB(Ma,Mb, lb, op, power=1)
    if power==2
        @tensor expect[] := lb[1,2]*Ma[2,3,4,5]*Mb[5,6,7,8]*op[3,6,9,10]*conj(op[11,12,9,10])*conj(lb[1,13])*conj(Ma[13,11,4,14])*conj(Mb[14,12,7,8])
    else
        @tensor expect[] := lb[1,2]*Ma[2,3,4,5]*Mb[5,6,7,8]*op[3,6,10,12]*conj(lb[1,9])*conj(Ma[9,10,4,11])*conj(Mb[11,12,7,8])
    end
    return expect[1]
end
function expect_operator_ABC(Ma,Mb,Mc, lc, op)
    @tensor expect[] := lc[1,2]*Ma[2,3,4,5]*Mb[5,6,7,8]*Mc[8,9,10,11]*op[3,6,9,13,15,17]*conj(lc[1,12])*conj(Ma[12,13,4,14])*conj(Mb[14,15,7,16])*conj(Mc[16,17,10,11])
    return expect[1]
end


""" iTEBD time evolution for 3-site state M1 and calculation of (retarded) correlator with M2;
    default=real-time  """
function gl_iTEBD3_correlatorevolution(M1a,M1b,M1c,l1a,l1b,l1c, M2a,M2b,M2c,l2a,l2b,l2c, hamblocks, total_time, steps, d, Dmax, operators; tol=0, increment::Int64=1, constant_hamiltonian=true, counter_evo=true)
    ### M1 is time-evolved; retarded correlator with M2 is calculated
    dt = total_time/steps
    errL = Array{Complex128,1}(1+Int(floor(steps/increment)))
    errR = Array{Complex128,1}(1+Int(floor(steps/increment)))
    time = Array{Complex128,1}(1+Int(floor(steps/increment)))
    nop = length(operators)
    opvalues = Array{Complex128,2}(1+Int(floor(steps/increment)),nop)

    datacount=1
    errL[1] = errR[1] = time[1] = 0.0
    for k = 1:nop opvalues[1,k] = expect_correlator_sum(M1a,M1b,M1c, M2a,M2b,M2c, operators[k]) end

    ## unique time evolution gate for no quench: default=real-time
    if constant_hamiltonian # control parameter
        W = expm(-1im*dt*hamblocks)
        W = reshape(W, (d,d,d,d,d,d))
    end

    ## time increment loop:
    println("\n time loop correlator")
    for counter = 1:steps
        ## time evolution: default=real-time
        t = counter*dt
        if !constant_hamiltonian # only in case of time-dependent Hamiltonian
            W = expm(-1im*dt*hamblocks(t))
            W = reshape(W, (d,d,d,d,d,d))
        end
        M1a,M1b,M1c, l1a,l1b,l1c, errL_tmp,errR_tmp = gl_iTEBD3_fullstep(M1a,M1b,M1c, l1a,l1b,l1c, W, Dmax, tol=tol, counter_evo=counter_evo)

        if counter % increment == 0
            datacount+=1
            time[datacount] = t
            errL[datacount],errR[datacount] = errL_tmp,errR_tmp
            ## calculate operators:
            for k = 1:nop
                opvalues[datacount,k] = expect_correlator_sum(M1a,M1b,M1c, M2a,M2b,M2c, operators[k])
                println("corr: ",opvalues[datacount,k])
            end
        end
        println("step ",counter," / ",steps)
    end

    return M1a,M1b,M1c, l1a,l1b,l1c, errL,errR, time, opvalues
end


""" iTEBD time evolution for 3-site gate ABC; default=real-time  """
function gl_iTEBD3_timeevolution(Ma,Mb,Mc, la,lb,lc, hamblocks, total_time, steps, d, Dmax, operators; tol=0, increment::Int64=1, conv_thresh=0.0, constant_hamiltonian::Bool=true, counter_evo::Bool=false, calculate_2Renyi::Bool=false)
    dt = total_time/steps
    errL = Array{Complex128,1}(1+Int(floor(steps/increment)))
    errR = Array{Complex128,1}(1+Int(floor(steps/increment)))
    time = Array{Complex128,1}(1+Int(floor(steps/increment)))
    nop = length(operators)
    opvalues = Array{Complex128,2}(1+Int(floor(steps/increment)),nop)
    renyi = Array{Complex128,2}(1+Int(floor(steps/increment)),2)

    datacount=1
    errL[1] = errR[1] = time[1] = 0.0
    for k = 1:nop opvalues[1,k] = expect_operator_average([Ma,Mb,Mc], [la,lb,lc], operators[k]) end
    if calculate_2Renyi
        renyi[1,1] = calculate_2Renyi_entropy([Ma,Mb,Mc])/3 # density per site
        # renyi[1,2] = -log(trace_rho_squared_average([Ma,Mb,Mc], [la,lb,lc]))/3 # unprecise?
        println("s2(0) = ",renyi[1,1])
    end

    ## unique time evolution gate for no quench: default=real-time
    if constant_hamiltonian # control parameter
        W = expm(-1im*dt*hamblocks)
        W = reshape(W, (d,d,d,d,d,d))
    end

    ## time increment loop:
    println("\n time loop")
    for counter = 1:steps
        ## time evolution: default=real-time
        t = counter*dt
        if !constant_hamiltonian # only in case of time-dependent Hamiltonian
            W = expm(-1im*dt*hamblocks(t))
            W = reshape(W, (d,d,d,d,d,d))
        end
        Ma,Mb,Mc, la,lb,lc, errL_tmp,errR_tmp = gl_iTEBD3_fullstep(Ma,Mb,Mc, la,lb,lc, W, Dmax, tol=tol, counter_evo=counter_evo)

        if counter % increment == 0
            datacount+=1
            time[datacount] = t
            errL[datacount],errR[datacount] = errL_tmp,errR_tmp

            ## calculate operators:
            for k = 1:nop
                opvalues[datacount,k] = expect_operator_average([Ma,Mb,Mc], [la,lb,lc], operators[k])
                println("E = ",real(opvalues[datacount,1])," , E_reldiff = ",abs(opvalues[datacount,1]-opvalues[datacount-1,1])/abs(opvalues[datacount,1]))
            end
            ## calculate 2-Renyi entropy density:
            if calculate_2Renyi
                s2 = calculate_2Renyi_entropy([Ma,Mb,Mc])/3 # density per site
                renyi[datacount,1] = s2
                # renyi[datacount,2] = -log(trace_rho_squared_average([Ma,Mb,Mc], [la,lb,lc]))/3
                println("s2 = ",s2)
            end
            println("Tr_rho(t) = ",trace_rho_average([Ma,Mb,Mc], [la,lb,lc]))
        end
        println("step ",counter," / ",steps)

        ## break at convergence precision (for ground state search in imaginary time evolution):
        if nop>=1 && abs(opvalues[datacount,1]-opvalues[datacount-1,1])/abs(opvalues[datacount,1])<=conv_thresh break end
    end

    if calculate_2Renyi
        return Ma,Mb,Mc, la,lb,lc, errL,errR, time, opvalues, renyi
    else
        return Ma,Mb,Mc, la,lb,lc, errL,errR, time, opvalues
    end
end

""" apply one full time evolution layer into 3-site translationinvariant infinite unit cell
    by all permutations of ABC and determine new tensors """
function gl_iTEBD3_fullstep(Ma,Mb,Mc, la,lb,lc, block, Dmax; tol=0, counter_evo=false)
    ## cycle through permutations of third steps:
    Ma,Mb,Mc, la,lb,lc, errL1,errR1 = MPS.gl_iTEBD3_thirdstep(Ma,Mb,Mc, lc, block, Dmax, tol=tol, counter_evo=counter_evo)
    Mb,Mc,Ma, lb,lc,la, errL2,errR2 = MPS.gl_iTEBD3_thirdstep(Mb,Mc,Ma, la, block, Dmax, tol=tol, counter_evo=counter_evo)
    Mc,Ma,Mb, lc,la,lb, errL3,errR3 = MPS.gl_iTEBD3_thirdstep(Mc,Ma,Mb, lb, block, Dmax, tol=tol, printmessage=true, counter_evo=counter_evo)
    errL = errL1+errL2+errL3
    errR = errR1+errR2+errR3
    println("errL, errR: ",errL," , ",errR)
    return Ma,Mb,Mc, la,lb,lc, errL,errR
end

""" apply one third time evolution layer into 3-site translationinvariant infinite unit cell
    and determine new tensors """
function gl_iTEBD3_thirdstep(M1,M2,M3, l3, block, Dmax; tol=0, printmessage=false, counter_evo=false)
    ### input: mpo = [M1,M2,M3] contain [l1,l2,l3]; explicit l3
    ###        <=> state = -l3-M1-M2-M3-
    ###        ATTENTION: l3 is at array position l[4]
    ### output: [M_A,M_B,M_C] , [lambda_A,lambda_B,lambda_C=l3]

    ## contract full state
    if counter_evo
        @tensor X_ABC[-1,-2,-3,-4,-5,-6,-7,-8] := l3[-1,1]*M1[1,2,3,4]*M2[4,5,6,7]*M3[7,8,9,-8]*block[2,5,8,-2,-4,-6]*conj(block[3,6,9,-3,-5,-7])
    else
        @tensor X_ABC[-1,-2,-3,-4,-5,-6,-7,-8] := l3[-1,1]*M1[1,2,-3,3]*M2[3,4,-5,5]*M3[5,6,-7,-8]*block[2,4,6,-2,-4,-6]
    end
    sXabc = size(X_ABC)
    X_ABC = reshape(X_ABC, sXabc[1]*sXabc[2]*sXabc[3]*sXabc[4]*sXabc[5], sXabc[6]*sXabc[7]*sXabc[8])

    ## 1st svd (right side)
    U,S,V = svd(X_ABC, thin=true)
    V = V'
    U,S,V,Dr,errR = truncate_svd(U,S,V,Dmax,tol)

    ## new tensors:
    lambda_B = diagm(S)
    M_C = reshape(V, Dr,sXabc[6],sXabc[7],sXabc[8])

    ## 2nd svd (left side)
    X_AB = U*lambda_B
    X_AB = reshape(X_AB, sXabc[1]*sXabc[2]*sXabc[3], sXabc[4]*sXabc[5]*Dr)
    U,S,V = svd(X_AB, thin=true)
    V = V'
    U,S,V,Dl,errL = truncate_svd(U,S,V,Dmax,tol)

    ## new tensors:
    lambda_A = diagm(S)
    M_B = reshape(V, Dl,sXabc[4],sXabc[5],Dr)

    ## reconstruct M_A
    Uab_la = U*lambda_A
    M_A = reshape(inv(l3)*reshape(Uab_la, sXabc[1],sXabc[2]*sXabc[3]*Dl), sXabc[8],sXabc[2],sXabc[3],Dl) # explicit inverse numerically unstable?
    # @tensor M_A[-1,-2,-3,-4] := M1[-1,-2,-3,1]*M2[1,2,3,4]*M3[4,5,6,7]*conj(M_C[8,5,6,7])*conj(M_B[-4,2,3,8]) # fancy rewritten but unprecise?
    if printmessage println("Dl,Dr = ",Dl,", ",Dr) end

    return M_A,M_B,M_C, lambda_A,lambda_B,l3, errL,errR
end

""" double canonicalize a unit cell Ma,Mb """
function double_canonicalize_and_normalize(Ma,Mb, la,lb, d; do_normalization=true)
    ## work with MPS or MPO:
    is_mps = false; is_mpo = false
    if length(size(Ma))==3
        is_mps = true
    elseif length(size(Ma))==4
        is_mpo = true
    end

    if is_mps
        @tensor ga[-1,-2,-3] := Ma[-1,-2,1]*inv(la)[1,-3]
        @tensor gb[-1,-2,-3] := Mb[-1,-2,1]*inv(lb)[1,-3]
        ga, la, gb, lb = iTEBD2.double_canonicalize(ga, diag(la), gb, diag(lb), do_normalization=do_normalization)
        la=diagm(la); lb=diagm(lb)
        @tensor Ma[-1,-2,-3] := ga[-1,-2,1]*la[1,-3]
        @tensor Mb[-1,-2,-3] := gb[-1,-2,1]*lb[1,-3]
    elseif is_mpo
        @tensor ga[-1,-2,-3,-4] := Ma[-1,-2,-3,1]*inv(la)[1,-4]
        @tensor gb[-1,-2,-3,-4] := Mb[-1,-2,-3,1]*inv(lb)[1,-4]
        sa=size(ga); sb=size(gb)
        fakeIndex = false
        if sa[3]==1
            fakeIndex = true
            ga = reshape(ga, sa[1],d,sa[4]) # MPS form for fake index
            gb = reshape(gb, sb[1],d,sb[4])
        else
            ga = reshape(ga, sa[1],d^2,sa[4]) # MPS form for MPO
            gb = reshape(gb, sb[1],d^2,sb[4])
        end
        # println("ga: ",size(ga),", la: ",size(la))
        # println("gb: ",size(gb),", lb: ",size(lb))

        ga, la, gb, lb = iTEBD2.double_canonicalize(ga, diag(la), gb, diag(lb), do_normalization=do_normalization)

        la=diagm(la); lb=diagm(lb)
        @tensor Ma[-1,-2,-3] := ga[-1,-2,1]*la[1,-3]
        @tensor Mb[-1,-2,-3] := gb[-1,-2,1]*lb[1,-3]
        sa=size(Ma); sb=size(Mb)
        if fakeIndex
            Ma = reshape(Ma, sa[1],d,1,sa[3]) # MPO form w/ fake index
            Mb = reshape(Mb, sb[1],d,1,sb[3])
        else
            Ma = reshape(Ma, sa[1],d,d,sa[3]) # MPO form
            Mb = reshape(Mb, sb[1],d,d,sb[3])
        end
    end

    return Ma,Mb, la,lb
end

""" double canonicalize by Identity circuit """
function double_canonicalize_by_Identity_circuit(Ma,Mb, la,lb, d, Dmax, Nlayers=100; tol=0)
    Id = reshape(eye(d^2), d,d,d,d)
    for i=1:Nlayers
        Ma,Mb, la,lb, err_tmp = gl_iTEBD2_fullstep(Ma,Mb, la,lb, Id, Dmax, tol=tol, printinfo=false)
        # Ma,Mb, la,lb, err_tmp = gl_iTEBD2_halfstep(Ma,Mb, lb, Id, Dmax, tol=tol)
    end
    return Ma,Mb, la,lb
end

""" apply layers of some MPO sites a,b in uniform circuit (2-site translational invariance)
    onto a MPS state ΓA,λA,ΓB,λB """
function apply_2site_mpo_layers(ΓA,λA, ΓB,λB, a,b, d, Dmax, Nlayers=100; tol=0)
    ### see Fig. 13 in https://journals.aps.org/prb/pdf/10.1103/PhysRevB.78.155117
    ### 2-site translational invariance A,B
    ### a,b = (D,d,d,D) = MPO sites
    err = Array{Complex128,1}(Nlayers)
    entropies = Array{Complex128,1}(Nlayers)

    for i=1:Nlayers
        # if i%2==0
        #     Γtmp,λtmp = deepcopy(ΓA),deepcopy(λA)
        #     ΓA,λA = ΓB,λB
        #     ΓB,λB = Γtmp,λtmp
        # end

        println("layer ",i)
        ## right-dominant eigenvector:
        ## (i)
        @tensor Θ1[:] := ΓA[-1,1,2]*λA[2,3]*ΓB[3,4,5]*λB[5,-5]*a[-2,1,-3,6]*b[6,4,-4,-6]
        sΘ1 = size(Θ1)
        Θ1 = reshape(Θ1, sΘ1[1]*sΘ1[2],sΘ1[3]*sΘ1[4],sΘ1[5]*sΘ1[6])

        ## (ii)
        SR, UR = iTEBD2.tm_eigs(Θ1, "R", 1)
        η = SR[1]
        VR = UR[1]
        ## cancel out phase to be Hermitian:
        r_tr = trace(VR)
        phase_r = r_tr/abs(r_tr)
        VR ./= phase_r
        ## split it:
        D,W = eig(VR)
        X = W*diagm(sqrt.(D))
        Xi = inv(X) # more accurate than  diagm(sqrt.(1 ./ D))*W'

        ## left-dominant eigenvector:
        ## (iii)
        @tensor Θ2[:] := λB[-1,1]*ΓA[1,2,3]*λA[3,4]*ΓB[4,5,-5]*a[-2,2,-3,6]*b[6,5,-4,-6]
        sΘ2 = size(Θ2)
        Θ2 = reshape(Θ2, sΘ2[1]*sΘ2[2],sΘ2[3]*sΘ2[4],sΘ2[5]*sΘ2[6])

        ## (iv)
        SL, UL = iTEBD2.tm_eigs(Θ2, "L", 1)
        τ = SL[1]
        VL = UL[1]
        ## cancel out phase to be Hermitian:
        l_tr = trace(VL)
        phase_l = l_tr/abs(l_tr)
        VL ./= phase_l
        ## split it:
        D,W = eig(VL)
        YT = diagm(sqrt.(D)) * W' # VL ≈ YT'*YT
        YTi = inv(YT) # more accurate than  W*diagm(sqrt.(1 ./ D))

        ## (v)
        @tensor Θ[:] := λB[-1,1]*ΓA[1,2,3]*λA[3,4]*ΓB[4,5,6]*λB[6,-5]*a[-2,2,-3,7]*b[7,5,-4,-6]
        sΘ = size(Θ)
        Θ = reshape(Θ, sΘ[1]*sΘ[2],sΘ[3],sΘ[4],sΘ[5]*sΘ[6])

        ## (vi)
        U,λB,V = svd(YT*X)
        V = V'
        ## truncation and normalization of singular values:
        U,λB,V,χ1,err_B = truncate_svd(U,λB,V,Dmax,tol)
        λBi = diagm(1 ./ λB)
        λB = diagm(λB)
        println("χ1, errB = ",χ1,", ",err_B)

        ## (vii)
        @tensor Σ[:] := λB[-1,1]*V[1,2]*Xi[2,3]*Θ[3,-2,-3,4]*YTi[4,5]*U[5,6]*λB[6,-4]
        sΣ = size(Σ)
        Σ = reshape(Σ, sΣ[1]*sΣ[2],sΣ[3]*sΣ[4])

        ## (viii)
        P,λA,Q = svd(Σ)
        Q = Q'
        ## truncation and normalization of singular values:
        P,λA,Q,χ2,err_A = truncate_svd(P,λA,Q,Dmax,tol)
        P = reshape(P, sΣ[1],sΣ[2],χ2)
        λA = diagm(λA)
        Q = reshape(Q, χ2,sΣ[3],sΣ[4])
        println("χ2, errA = ",χ2,", ",err_A)

        ## (ix)
        @tensor ΓA[:] := λBi[-1,1]*P[1,-2,-3]
        @tensor ΓB[:] := Q[-1,-2,1]*λBi[1,-3]

        ## explicit double-canonicalization:  (Why necessary???)
        @tensor MA[:] := ΓA[-1,-2,1]*λA[1,-3];
        @tensor MB[:] := ΓB[-1,-2,1]*λB[1,-3];
        sMA=size(MA); MA=reshape(MA, sMA[1],sMA[2],1,sMA[3])
        sMB=size(MB); MB=reshape(MB, sMB[1],sMB[2],1,sMB[3])
        MA,MB, λA,λB = double_canonicalize_by_Identity_circuit(MA,MB, λA,λB, d, Dmax)
        sMA=size(MA); MA=reshape(MA, sMA[1],sMA[2],sMA[4])
        sMB=size(MB); MB=reshape(MB, sMB[1],sMB[2],sMB[4])
        @tensor ΓA[:] := MA[-1,-2,1]*inv(λA)[1,-3]
        @tensor ΓB[:] := MB[-1,-2,1]*inv(λB)[1,-3]

        ## values and entanglement entropy:
        err[i] = err_A + err_B
        entropies[i] = -dot(diag(λA).^2,log.(diag(λA).^2))
        println("s1_i = ",entropies[i])
    end
    return ΓA,λA, ΓB,λB, err, entropies
end

""" apply layers of some gate G in uniform circuit (1-site translational invariance)
    onto a MPS state Γ,λ """
function apply_gate_layers(Γ,λ, G, d, Dmax, Nlayers=100; tol=0)
    ### see Fig. 6 in https://arxiv.org/pdf/0711.3960.pdf
    ### 1-site translational invariance
    ### G = (D,d,d,D) = MPO site = gate
    iterations = collect(1:Nlayers)
    err = Array{Complex128,1}(Nlayers)

    for i=1:Nlayers
        ## absorb MPO site G into state Γ,λ -> "tilded quantities t":
        @tensor Γt[:] := Γ[-1,1,-4]*G[-2,1,-3,-5]
        sΓt = size(Γt)
        Γt = reshape(Γt, sΓt[1]*sΓt[2],sΓt[3],sΓt[4]*sΓt[5])
        λt = kron(eye(size(G,1)),λ)

        ## canonicalize:
        @tensor M[:] := Γt[-1,-2,1]*λt[1,-3]
        l, r = iTEBD2.normalize_it(M)
        Γt, λt = iTEBD2.canonical_form(M, l, r)

        ## truncate:
        if length(λt) > Dmax
            println("err: ", sum(λt[Dmax+1:end].^2))
            λt = λt[1:Dmax]
            λt = diagm(λt)
            Γt = Γt[1:Dmax,:,1:Dmax]
        else
            λt = diagm(λt)
        end

        Γ = Γt
        λ = λt

        ## absorb gate in 2-site unit cell:
        # @tensor C[-1,-2,-3,-4,-5,-6,-7,-8] := lb[-1,1]*Ma[1,2,-5,3]*Mb[3,4,-6,-7]*G[-2,-3,2,5]*G[5,-4,4,-8]
        # sC = size(C)
        # C = reshape(C, sC[1]*sC[2],sC[3]*sC[4]*sC[5]*sC[6],sC[7]*sC[8]) # MPS form
        #
        # ## perform splitting into 2 sites as in double canonicalization:
        # l, r = iTEBD2.normalize_it(C)
        # Γ, λB = iTEBD2.canonical_form(C, l, r)
        # @tensor Γ[x,i,y] := diagm(λB)[x,a] * Γ[a,i,b] * diagm(λB)[b,y]
        # Γ = reshape(Γ, (sC[1]*sC[2]*d^2, d^2*sC[1]*sC[2]))
        # ΓA, λA, ΓB = svd(Γ)
        # ΓA, λA, ΓB = iTEBD2.truncate_svd_static(ΓA, λA, ΓB, sC[7]*sC[8])
        # ΓA = reshape(ΓA, (sC[1]*sC[2], d^2, sC[7]*sC[8]))
        # ΓB = reshape(ΓB', (sC[7]*sC[8], d^2, sC[1]*sC[2]))
        # λBinv = 1. ./ λB
        # @tensor ΓA[x,i,y] := diagm(λBinv)[x,a] * ΓA[a,i,y]
        # @tensor ΓB[x,i,y] := ΓB[x,i,a] * diagm(λBinv)[a,y]
        #
        # ## MPO forms:
        # la = diagm(λA); lb = diagm(λB)
        # @tensor Gala[:] := ΓA[-1,-2,1]*la[1,-3]
        # @tensor Gblb[:] := ΓB[-1,-2,1]*lb[1,-3]
        # Ma = reshape(Gala, sC[1]*sC[2],d,d,sC[1]*sC[2])
        # Mb = reshape(Gblb, sC[1]*sC[2],d,d,sC[1]*sC[2])
    end
    return Γ,λ, iterations, err
end

""" iTEBD time evolution for 2-site gate AB; default=real-time  """
function gl_iTEBD2_timeevolution(Ma,Mb, la,lb, hamblocks, total_time, steps, d, Dmax, operators;
                                 tol=0, increment::Int64=1, conv_thresh=0.0, constant_hamiltonian::Bool=true, counter_evo::Bool=false,
                                 calculate_2Renyi::Bool=false, collect_spectrum::Bool=false, num_rate_levels::Int64=0, onesite_ops=[],
                                 do_recanonicalization::Bool=true, do_normalization::Bool=true, err_max=Inf)
    ## variables:
    dt = total_time/steps
    err = Array{Complex128,1}(1+Int(floor(steps/increment)))
    time = Array{Complex128,1}(1+Int(floor(steps/increment)))
    nop = length(operators)
    opvalues = Array{Complex128,2}(1+Int(floor(steps/increment)),nop)
    n_onesite_ops = length(onesite_ops)
    onesite_opvals = Array{Complex128,2}(1+Int(floor(steps/increment)),n_onesite_ops)
    renyi = Array{Complex128,2}(1+Int(floor(steps/increment)),2)
    Dstart = size(la,1)
    spectrum = Array{Complex128,2}(1+Int(floor(steps/increment)),Dstart)
    rates = Array{Complex128,2}(1+Int(floor(steps/increment)),num_rate_levels)
    Ma0,Mb0, la0,lb0 = deepcopy(Ma), deepcopy(Mb), deepcopy(la), deepcopy(lb)

    ## initializations:
    datacount=1
    n_error=0
    err[1] = time[1] = 0.0
    for k = 1:nop # 2-site ops, assumes MPO
        opvalues[1,k] = expect_operator_average([Ma,Mb], [la,lb], operators[k])
    end
    for k = 1:n_onesite_ops # 1-site ops, assumes MPS
        @tensor ga[-1,-2,-3] := Ma[-1,-2,1]*inv(la)[1,-3]; @tensor gb[-1,-2,-3] := Mb[-1,-2,1]*inv(lb)[1,-3]
        onesite_opvals[1,k] = iTEBD2.expect_local_dcan(ga, diag(la), gb, diag(lb), onesite_ops[k])
    end
    if calculate_2Renyi
        s1 = -dot(diag(la).^2,log.(diag(la).^2))
        s2 = -log(sum(diag(la).^4))
        # s2 = calculate_2Renyi_entropy([Ma,Mb])/2 # global density per site
        # s2 = -log(trace_rho_squared_average([Ma,Mb], [la,lb]))/2 # unprecise?
        renyi[1,1] = s1
        renyi[1,2] = s2
        println("s1(0), s2(0) = ",s1,", ",s2)
    end
    if collect_spectrum
        spectrum[1,:] = diag(la).^2
    end
    rate_error = false
    if num_rate_levels>0
        rates_result = calculate_rate_functions(Ma0,Mb0, Ma0,Mb0, num_rate_levels)
        num_rates = length(rates_result) # the actual number of eigenvalues might be smaller for small bond dimensions
        rates[1,1:num_rates] = rates_result
    end

    ## unique time evolution gate for no quench: default=real-time
    if constant_hamiltonian # control parameter
        W = expm(-1im*dt*hamblocks)
        W = reshape(W, (d,d,d,d))
    end

    ## time increment loop:
    println("\n time loop")
    for counter = 1:steps
        println("step ",counter," / ",steps)
        if n_error > 100
            println("ERROR ABORTION")
            break
        end

        ## time evolution: default=real-time
        t = counter*dt
        if !constant_hamiltonian # only in case of time-dependent Hamiltonian
            W = expm(-1im*dt*hamblocks(t))
            W = reshape(W, (d,d,d,d))
        end

        err_tmp = 0.0 # dummy
        old_la, old_lb = la, lb
        try
            Ma,Mb, la,lb, err_tmp = gl_iTEBD2_fullstep(Ma,Mb, la,lb, W, Dmax, tol=tol, counter_evo=counter_evo)
            # Ma,Mb, la,lb = double_canonicalize_and_normalize(Ma,Mb, la,lb, d, do_normalization=do_normalization)
        catch
            println("ERROR: skipped step ",counter)
            n_error += 1
            continue
        end

        ## stop loop at max error:
        if err_tmp > err_max break end

        if counter % increment == 0
            if do_recanonicalization
                try
                    Ma,Mb, la,lb = double_canonicalize_and_normalize(Ma,Mb, la,lb, d, do_normalization=do_normalization)
                catch
                    println("canonicalization ERROR in step ",counter)
                end
            end
            datacount+=1
            time[datacount] = t
            err[datacount] = err_tmp

            ## norm difference in singular values:
            Dcommon = minimum([size(old_la,1), size(la,1), size(old_lb,1), size(lb,1)]) # dynamical truncation might change size of Schmidt values during time steps
            eps = vecnorm(diag(old_la)[1:Dcommon] - diag(la)[1:Dcommon])/vecnorm(diag(la)[1:Dcommon]) + vecnorm(diag(old_lb)[1:Dcommon] - diag(lb)[1:Dcommon])/vecnorm(diag(lb)[1:Dcommon])
            println("eps: ",eps)

            ## calculate operators:
            for k = 1:nop
                opvalues[datacount,k] = expect_operator_average([Ma,Mb], [la,lb], operators[k])
                println("E = ",real(opvalues[datacount,1])," , E_reldiff = ",abs(opvalues[datacount,1]-opvalues[datacount-1,1])/abs(opvalues[datacount,1]))
            end
            for k = 1:n_onesite_ops
                @tensor ga[-1,-2,-3] := Ma[-1,-2,1]*inv(la)[1,-3]; @tensor gb[-1,-2,-3] := Mb[-1,-2,1]*inv(lb)[1,-3]
                onesite_opvals[datacount,k] = iTEBD2.expect_local_dcan(ga, diag(la), gb, diag(lb), onesite_ops[k])
            end

            ## calculate 2-Renyi entropy density:
            if calculate_2Renyi
                s1 = -dot(diag(la).^2,log.(diag(la).^2))
                s2 = -log(sum(diag(la).^4))
                # s2 = calculate_2Renyi_entropy([Ma,Mb])/2 # global density per site
                # s2 = -log(trace_rho_squared_average([Ma,Mb], [la,lb]))/2 # unprecise?
                renyi[datacount,1] = s1
                renyi[datacount,2] = s2
                println("s1, s2 = ",s1,", ",s2)
            end

            ## collect entanglement spectrum:
            if collect_spectrum
                Dtmp = size(la,1)
                if Dtmp < Dstart
                    spectrum[datacount,:] = cat(1, diag(la)[1:Dtmp].^2, zeros(Dstart-Dtmp))
                else
                    spectrum[datacount,:] = diag(la)[1:Dstart].^2
                end
            end

            ## calculate rate functions:
            if num_rate_levels>0
                try
                    rates_result = calculate_rate_functions(Ma,Mb, Ma0,Mb0, num_rate_levels)
                    num_rates = length(rates_result)
                    rates[datacount,1:num_rates] = rates_result
                catch
                    println("ERROR in rate level ",counter)
                    rate_error = true
                end
            end

            if length(size(Ma))==4 println("Tr_rho(t) = ",trace_rho_average([Ma,Mb], [la,lb])) end

            ## break at convergence precision (for ground state search in imaginary time evolution):
            if nop>=1 && abs(opvalues[datacount,1]-opvalues[datacount-1,1])/abs(opvalues[datacount,1])<=conv_thresh break end
        end
    end
    println("rate_error: ",rate_error)

    if collect_spectrum
        return Ma,Mb, la,lb, err, time, opvalues, onesite_opvals, renyi, spectrum, rates
    elseif calculate_2Renyi
        return Ma,Mb, la,lb, err, time, opvalues, renyi
    else
        return Ma,Mb, la,lb, err, time, opvalues
    end
end

""" apply one full time evolution layer into 2-site translationinvariant infinite unit cell
    by all permutations of AB and determine new tensors """
function gl_iTEBD2_fullstep(Ma,Mb, la,lb, block, Dmax; tol=0, counter_evo=false, printinfo=true)
    ## cycle through permutations of half steps:
    Ma,Mb, la,lb, err1 = MPS.gl_iTEBD2_halfstep(Ma,Mb, lb, block, Dmax, tol=tol, counter_evo=counter_evo)
    Mb,Ma, lb,la, err2 = MPS.gl_iTEBD2_halfstep(Mb,Ma, la, block, Dmax, tol=tol, printmessage=printinfo, counter_evo=counter_evo)
    err = err1+err2
    if printinfo println("err: ",err) end
    return Ma,Mb, la,lb, err
end

""" apply one half time evolution layer into 2-site translationinvariant infinite unit cell
    and determine new tensors """
function gl_iTEBD2_halfstep(M1,M2, l2, block, Dmax; tol=0, printmessage=false, counter_evo=false)
    ### input: mpo = [M1,M2] contain [l1,l2]; explicit l2
    ###        <=> state = -l2-M1-M2-
    ###        ATTENTION: l2 is at array position l[3]
    ### output: [M_A,M_B] , [lambda_A,lambda_B=l2]

    ## work with MPS or MPO:
    is_mps = false; is_mpo = false
    if length(size(M1))==3
        is_mps = true
    elseif length(size(M1))==4
        is_mpo = true
    end

    ## contract full state
    if counter_evo
        @tensor X_AB[-1,-2,-3,-4,-5,-6] := l2[-1,1]*M1[1,2,3,4]*M2[4,5,6,-6]*block[2,5,-2,-4]*conj(block[3,6,-3,-5])
    else
        if is_mps
            @tensor X_AB[-1,-2,-3,-4] := l2[-1,1]*M1[1,2,3]*M2[3,4,-4]*block[2,4,-2,-3]
        elseif is_mpo
            @tensor X_AB[-1,-2,-3,-4,-5,-6] := l2[-1,1]*M1[1,2,-3,3]*M2[3,4,-5,-6]*block[2,4,-2,-4]
        end
    end
    sXab = size(X_AB)
    if is_mps
        X_AB = reshape(X_AB, sXab[1]*sXab[2], sXab[3]*sXab[4])
    elseif is_mpo
        X_AB = reshape(X_AB, sXab[1]*sXab[2]*sXab[3], sXab[4]*sXab[5]*sXab[6])
    end

    ## svd (mid side)
    F = try
            svd(X_AB, thin=true)
        catch y
            svd_slow(X_AB)
        end
    U,S,V = F # svd(X_AB, thin=true)
    V = V'
    U,S,V,D,err = truncate_svd(U,S,V,Dmax,tol)

    ## new tensors:
    lambda_A = diagm(S)
    if is_mps
        X = reshape(U, sXab[1],sXab[2],D)
        Y = reshape(V, D,sXab[3],sXab[4])
        @tensor M_A[-1,-2,-3] := inv(l2)[-1,1]*X[1,-2,2]*lambda_A[2,-3]
    elseif is_mpo
        X = reshape(U, sXab[1],sXab[2],sXab[3],D)
        Y = reshape(V, D,sXab[4],sXab[5],sXab[6])
        @tensor M_A[-1,-2,-3,-4] := inv(l2)[-1,1]*X[1,-2,-3,2]*lambda_A[2,-4]
    end

    M_B = Y
    if printmessage println("D = ",D) end

    return M_A,M_B, lambda_A,l2, err
end


""" coarsegrain a given MPO, defined by Γ,λ, by N layers of MERA tensors w,u
    monitor the singular values lambda """
function gl_coarsegraining_mpo(Γa,λa,Ma, Γb,λb,Mb, w,u, N, d, Dmax; tol=0, do_normalization=true, num_singvals=1)
    sing_vals = Array{Float64}(num_singvals,N)

    for i=1:N
        println("cg step ",i," / ",N)
        Γa,λa,Ma, Γb,λb,Mb, λ = gl_coarsegraining_mpo_step(Γa,λa,Ma, Γb,λb,Mb, w,u, d, Dmax, tol=tol, do_normalization=do_normalization)
        sing_vals[:,i] = diag(λ)[1:num_singvals]
        println("λ1 = ",sing_vals[1,i])
    end

    return Γa,λa,Ma, Γb,λb,Mb, sing_vals
end

""" coarsegrain step for a given uniform MPO site by one layer of MERA tensors w,u """
function gl_coarsegraining_mpo_step(Γ1a,λ1a,M1a, Γ1b,λ1b,M1b, w,u, d, Dmax; tol=0, do_normalization=true)
    ## absorb disentangler u:
    MA,MB, λA,λB, err = gl_iTEBD2_halfstep(M1a,M1b, λ1b, u, Dmax, tol=tol, printmessage=true, counter_evo=true)
    println("err: ",err)
    @tensor ΓA[-1,-2,-3,-4] := MA[-1,-2,-3,1]*inv(λA)[1,-4]

    ## absorb isometry w:
    @tensor Γ2[-1,-2,-3,-4] := MB[-1,1,2,3]*w[-2,1,4]*conj(w[-3,2,5])*ΓA[3,4,5,-4]

    ## resulting tensors: {Γ2, λA, M2}
    @tensor M2[-1,-2,-3,-4] := Γ2[-1,-2,-3,1]*λA[1,-4]

    ## canonicalization:
    ## by Identity-iTEBD2, this gives pair -> {Γ3a,λ3a,M3a; Γ3b,λ3b,M3b}:
    M3a,M3b, λ3a,λ3b = double_canonicalize_by_Identity_circuit(M2,M2, λA,λA, d, Dmax, 40, tol=tol)
    @tensor Γ3a[:] := M3a[-1,-2,-3,1]*inv(λ3a)[1,-4]
    @tensor Γ3b[:] := M3b[-1,-2,-3,1]*inv(λ3b)[1,-4]

    ## by double canonicalize:
    # M3a,M3b, λ3a,λ3b = MPS.double_canonicalize_and_normalize(M2,M2, λA,λA, d, do_normalization=do_normalization)
    # @tensor Γ3a[:] := M3a[-1,-2,-3,1]*inv(λ3a)[1,-4]
    # @tensor Γ3b[:] := M3b[-1,-2,-3,1]*inv(λ3b)[1,-4]

    ## by single canonicalize might fail:
    # Γ,λ,M = single_canonicalize_and_normalize(M2,d, do_normalization=do_normalization)

    return Γ3a,λ3a,M3a, Γ3b,λ3b,M3b, λA
end

""" canonicalize a single MPO site M; as default also normalize it """
function single_canonicalize_and_normalize(M, d; do_normalization=true)
    ## reshape into MPS form:
    sM = size(M)
    M = reshape(M, sM[1],d^2,sM[4])

    ## canonicalize:
    l, r = iTEBD2.normalize_it(M, do_normalization=do_normalization)
    # Γ, λ = iTEBD2.canonical_form(M, l, r, 1e-11)
    Γ, λ = iTEBD2.canonical_form_chol(M, l, r)

    ## reshape into MPO form again:
    λ = diagm(λ)
    @tensor M[-1,-2,-3] := Γ[-1,-2,1]*λ[1,-3]
    sM = size(M)
    M = reshape(M, sM[1],d,d,sM[3]) # MPO form
    Γ = reshape(Γ, sM[1],d,d,sM[3])

    return Γ,λ,M
end


function gl_tebd_step(g,l, hamblocks, dt, D; tol=0,counter_evo=false)
    d = size(g[1])[2]
    N = length(g)
    total_error = 0
    W = reshape.(expm.(-1im*dt*hamblocks),d,d,d,d)
    # function local_update(k)
    #     return (k,updateBlock(l[k],l[k+1],l[k+2],g[k],g[k+1],W[k], D, tol)...)
    # end
    if imag(dt)==0
        # results = pmap(local_update,1:2:N-1)
        # for r in results
        #     g[r[1]]=r[2]
        #     l[r[1]+1]=r[3]
        #     g[r[1]+1]=r[4]
        #     total_error+=r[5]
        # end
        # s = isodd(N) ? N-1 : N-2
        # results = pmap(local_update,s:-2:1)
        # for r in results
        #     g[r[1]]=r[2]
        #     l[r[1]+1]=r[3]
        #     g[r[1]+1]=r[4]
        #     total_error+=r[5]
        # end
        Threads.@threads for k = 1:2:N-1
        # W = expm(-1im*dt*hamblocks[k])
        # W = reshape(Ws[k], (d,d,d,d))
        g[k], l[k+1], g[k+1], error = updateBlock(l[k],l[k+1],l[k+2],g[k],g[k+1],W[k], D, tol,counter_evo=counter_evo)
        total_error += error
        end
        s = isodd(N) ? N-1 : N-2
        Threads.@threads for k = s:-2:1
            # W = expm(-1im*dt*hamblocks[k])
            # W = reshape(Ws[k], (d,d,d,d))
        g[k], l[k+1], g[k+1], error = updateBlock(l[k],l[k+1],l[k+2],g[k],g[k+1],W[k], D, tol,counter_evo=counter_evo)

            total_error += error
        end
    else
        WI = reshape(II, (d,d,d,d))
        for k = 1:N-1
            # W = expm(-1im*dt*hamblocks[k])
            # W = reshape(Ws[k], (d,d,d,d))
            g[k], l[k+1], g[k+1], error = updateBlock(l[k],l[k+1],l[k+2],g[k],g[k+1],W[k], D, tol, counter_evo=counter_evo)
            total_error += error
        end
        for k = N-1:-1:1
            g[k], l[k+1], g[k+1], error = updateBlock(l[k],l[k+1],l[k+2],g[k],g[k+1],WI, D, tol,counter_evo=counter_evo)
            total_error += error
        end
    end
    return total_error
end

 function gl_tebd_step_st2(g,l, hamblocks, dt, D; tol=0, counter_evo=false)
    d = size(g[1])[2]
    N = length(g)
    total_error = 0
    W = reshape.(expm.(-1im*dt*hamblocks),d,d,d,d)
    W2 = reshape.(expm.(-1/2*im*dt*hamblocks),d,d,d,d)
    if imag(dt)==0
        Threads.@threads for k = 1:2:N-1
            g[k], l[k+1], g[k+1], error = updateBlock(l[k],l[k+1],l[k+2],g[k],g[k+1],W2[k], D, tol, counter_evo=counter_evo)
            total_error += error
        end
        s = isodd(N) ? N-1 : N-2
        Threads.@threads for k = s:-2:1
            g[k], l[k+1], g[k+1], error = updateBlock(l[k],l[k+1],l[k+2],g[k],g[k+1],W[k], D, tol, counter_evo=counter_evo)
            total_error += error
        end
        Threads.@threads for k = 1:2:N-1
            g[k], l[k+1], g[k+1], error = updateBlock(l[k],l[k+1],l[k+2],g[k],g[k+1],W2[k], D, tol, counter_evo=counter_evo)
            total_error += error
        end
    else
        WI = reshape(II, (d,d,d,d))
        s = isodd(N) ? N-1 : N-2
        for k=1:N
            W[k] = isodd(k) ? W[k] : WI
            W2[k] = iseven(k) ? W2[k] : WI
        end
        for k = 1:N-1
            g[k], l[k+1], g[k+1], error = updateBlock(l[k],l[k+1],l[k+2],g[k],g[k+1],W2[k], D, tol, counter_evo=counter_evo)
            total_error += error
        end
        for k = N-1:-1:1
            g[k], l[k+1], g[k+1], error = updateBlock(l[k],l[k+1],l[k+2],g[k],g[k+1],W[k], D, tol, counter_evo=counter_evo)
            total_error += error
        end
        for k = 1:N-1
            g[k], l[k+1], g[k+1], error = updateBlock(l[k],l[k+1],l[k+2],g[k],g[k+1],W2[k], D, tol, counter_evo=counter_evo)
            total_error += error
        end
        for k = N-1:-1:1
            g[k], l[k+1], g[k+1], error = updateBlock(l[k],l[k+1],l[k+2],g[k],g[k+1],WI, D, tol, counter_evo=counter_evo)
            total_error += error
        end
    end
    return total_error
end

function truncate_svd(U, S, V, D,tol=0)
    ## assume M = U*S*V, i.e. V=V' was set before
    Dtol = 0
    tot = sum(S.^2)
    while (Dtol+1 <= length(S)) && sum(S[Dtol+1:end].^2)/tot>=tol
        Dtol+=1
    end
    D = min(D,Dtol)
    err = sum(S[D+1:end].^2)
    U = U[:, 1:D]
    S = S[1:D]
    S = S/sqrt(sum(S.^2))
    V = V[1:D, :]
    return U, S, V, D,err
end
