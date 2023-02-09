module TEBD
using TensorOperations
using MPS

function isingHamBlocks(L,J,h,g)
    blocks = Array{Any,1}(L)
    for i=1:L
        if i==1
            blocks[i] = J*ZZ + h/2*(XI+2IX) + g/2*(ZI+2IZ)
        elseif i==L-1
            blocks[i] = J*ZZ + h/2*(2*XI+IX) + g/2*(2*ZI+IZ)
        else
            blocks[i] = J*ZZ + h/2*(XI+IX) + g/2*(ZI+IZ)
        end
    end
    return blocks
end

function heisenbergHamblocks(L,Jx, Jy, Jz, hx)
    blocks = Array{Any,1}(L)
    for i=1:L
        if i==1
            blocks[i] = Jx*XX + Jy*YY + Jz*ZZ + hx/2*(XI+2*IX)
        elseif i==L-1
            blocks[i] = Jx*XX + Jy*YY + Jz*ZZ + hx/2*(2*XI+IX)
        else
            blocks[i] = Jx*XX + Jy*YY + Jz*ZZ + hx/2*(XI+IX)
        end
    end
    return blocks
end

function truncate_svd(U, S, V, D,tol=0)
    Dtol = 0
    tot = sum(S.^2)
    while (Dtol+1 <= length(S)) && sum(S[Dtol+1:end].^2)/tot>=tol
        Dtol+=1
    end
    D = min(D,Dtol)
    err = sum(S[D+1:end].^2)
    U = U[:, 1:D]
    S = S[1:D]
    V = V[1:D, :]
    return U, S, V, D, err
end

""" block_decimation(W, Tl, Tr, Dmax,dir)
Apply two-site operator W (4 indexes) to mps tensors Tl (left) and Tr (right)
and performs a block decimation (one TEBD step)
```block_decimation(W,TL,TR,Dmax,dir) -> Tl, Tr"""
function block_decimation(W, Tl, Tr, Dmax, dir=0; tol=0)
    ### input:
    ###     W:      time evolution op W=exp(-tau h) of size (d,d,d,d)
    ###     Tl, Tr: mps sites mps[i] and mps[i+1] of size (D1l,d,D1r) and (D2l,d,D2r)
    ###     Dmax:   maximal bond dimension
    ###     dir:    direction (-1 is leftcanonical, +1 is rightcanonical) for preference where to put singular value matrix during sweep
    ### output:
    ###     Tl, Tr after one time evolution step specified by W

    stl = size(Tl)
    str = size(Tr)
    if length(stl)==4
        Tl = reshape(permutedims(Tl, [1,3,2,4]), stl[1]*stl[3],stl[2],stl[4])
        Tr = reshape(Tr, str[1],str[2],str[3]*str[4])
    end
    D1l,d,D1r = size(Tl)
    D2l,d,D2r = size(Tr)

    # absorb time evolution gate W into Tl and Tr
    @tensor theta[-1,-2,-3,-4] := Tl[-1,2,3]*W[2,4,-2,-3]*Tr[3,4,-4] # = (D1l,d,d,D2r)
    theta = reshape(theta, D1l*d,d*D2r)
    U,S,V = svd(theta, thin=true)
    V = V'
    # SVD = svds(theta, nsv=min(Dmax,D1l*d,d*D2r)-1)[1] # takes (way) longer !?
    # U = SVD[:U]
    # S = SVD[:S]
    # V = SVD[:Vt]
    D1 = size(S)[1] # number of singular values

    # if D1 <= Dmax
    #     err = 0
    #     if dir == -1
    #         Tl = reshape(U, D1l,d,D1)
    #         Tr = reshape(diagm(S)*V, D1,d,D2r)
    #     elseif dir == 1
    #         Tl = reshape(U*diagm(S), D1l,d,D1)
    #         Tr = reshape(V, D1,d,D2r)
    #     else
    #         rS = sqrt.(S)
    #         Tl = reshape(U*diagm(rS), D1l,d,D1)
    #         Tr = reshape(diagm(rS)*V, D1,d,D2r)
    #     end
    # else
        U,S,V,D,err = truncate_svd(U,S,V,Dmax,tol)
        if dir == -1
            Tl = reshape(U, D1l,d,D)
            Tr = reshape(diagm(S)*V, D,d,D2r)
        elseif dir==1
            Tl = reshape(U*diagm(S), D1l,d,D)
            Tr = reshape(V, D,d,D2r)
        else
            rS = sqrt.(S)
            Tl = reshape(U*diagm(rS), D1l,d,D)
            Tr = reshape(diagm(rS)*V, D,d,D2r)
        end
    # end

    if length(stl)==4
        Tl = permutedims(reshape(Tl, stl[1],stl[3],stl[2],D), [1,3,2,4])
        Tr = reshape(Tr, D,str[2],str[3],str[4])
    end

    return Tl, Tr, err
end

function time_evolve_mpoham(mps, block, total_time, steps, D, increment, entropy_cut, params, eth, mpo=nothing)
    ### block = hamiltonian
    ### use -im*total_time for imaginary time evolution
    ### assumption: start with rightcanonical mps
	### eth = (true,E1,hamiltonian) --> do ETH calcs if true for excited energy E1 wrt hamiltonian
    stepsize = total_time/steps
    d = size(mps[1])[2]
    L = length(mps)
    if isodd(L)
        even_start = L-1
        odd_length = true
    else
        even_start = L-2
        odd_length = false
    end
    mpo_to_mps_trafo = false
    if length(size(mps[1])) == 4 # control variable to make mps out of mpo
        mpo_to_mps_trafo = true
    end
    if !mpo_to_mps_trafo && MPS.check_LRcanonical(mps[1],-1) # use rightcanonical mps as default input for sweeping direction
        MPS.makeCanonical(mps)
    end
    datalength = Int(ceil(steps/increment))
    expect        = Array{Any}(datalength,2)
    entropy       = Array{Any}(datalength,2)
	magnetization = Array{Any}(datalength,2)
    correlation   = Array{Any}(datalength,2)
    corr_length   = Array{Any}(datalength,2)

    for counter = 1:steps
        time = counter*total_time/steps

        ## ************ right sweep over odd sites
        for i = 1:2:L-1
            W = expm(-1im*stepsize*block(i,time,params))
            W = reshape(W, (d,d,d,d))
            mps[i], mps[i+1] = block_decimation(W, mps[i], mps[i+1], D, -1)
			# preserve canonical structure:
            if mpo_to_mps_trafo
                smpo = MPS.mpo_to_mps(mps)
                mps[i+1],R,DB = MPS.LRcanonical(mps[i+1],-1) # leftcanonicalize current sites
                if i < L-1 || odd_length
                    @tensor mps[i+2][-1,-2,-3] := R[-1,1]*mps[i+2][1,-2,-3]
                end
                MPS.mps_to_mpo(mps,smpo)
            else
                mps[i+1],R,DB = MPS.LRcanonical(mps[i+1],-1)
                if i < L-1 || odd_length
                    @tensor mps[i+2][-1,-2,-3] := R[-1,1]*mps[i+2][1,-2,-3]
                end
            end
        end

        ## ************ left sweep over even sites
        if mpo_to_mps_trafo
            smpo = MPS.mpo_to_mps(mps)
            mps[L],R,DB = MPS.LRcanonical(mps[L],1) # rightcanonicalize at right end
            @tensor mps[L-1][:] := mps[L-1][-1,-2,1]*R[1,-3]
            MPS.mps_to_mpo(mps,smpo)
        else
            mps[L],R,DB = MPS.LRcanonical(mps[L],1)
            @tensor mps[L-1][:] := mps[L-1][-1,-2,1]*R[1,-3]
        end
        for i = even_start:-2:2
            W = expm(-1im*stepsize*block(i,time,params))
            W = reshape(W, (d,d,d,d))
            mps[i], mps[i+1] = block_decimation(W, mps[i], mps[i+1], D, 1)
			# preserve canonical structure:
            if mpo_to_mps_trafo
                smpo = MPS.mpo_to_mps(mps)
                mps[i],R,DB = MPS.LRcanonical(mps[i],1) # rightcanonicalize current sites
                @tensor mps[i-1][:] := mps[i-1][-1,-2,1]*R[1,-3]
                MPS.mps_to_mpo(mps,smpo)
            else
                mps[i],R,DB = MPS.LRcanonical(mps[i],1)
                @tensor mps[i-1][:] := mps[i-1][-1,-2,1]*R[1,-3]
            end
        end
        if mpo_to_mps_trafo
            smpo = MPS.mpo_to_mps(mps)
            mps[1],R,DB = MPS.LRcanonical(mps[1],1) # rightcanonicalize at left end
            MPS.mps_to_mpo(mps,smpo)
        else
            mps[1],R,DB = MPS.LRcanonical(mps[1],1)
        end

        ## expectation values:
        if mpo != nothing
            if mpo == "Ising"
                J0, h0, g0 = params
                J, h, g = evolveIsingParams(J0, h0, g0, time)
                hamiltonian = MPS.IsingMPO(L, J, h, g)
                expect[counter,:] = [time MPS.mpoExpectation(mps,hamiltonian)]
			elseif mpo == "Isingthermal"
                if counter==1 || counter % increment == 0
                    println("step ",counter," / ",steps)
    				J0, h0, g0 = params
                    J, h, g = evolveIsingParams(J0, h0, g0, time)
                    hamiltonian = MPS.IsingMPO(L, J, h, g)
    				# rho = MPS.multiplyMPOs(mps,mps)
                    tr_rho = real(MPS.traceMPO(mps,2))
                    # expect[Int(floor(counter/increment))+1,:] = [time real(MPS.traceMPOprod(rho,hamiltonian)/tr_rho)] # = E
                    expect[Int(ceil(counter/increment)),:] = [time real(MPS.traceMPOprod(mps,hamiltonian,2)/tr_rho)] # = E ## seems to work
    				magnet_pos = Int(floor(L/2)) # position for magnetization op in spin chain
    				magnetization[Int(ceil(counter/increment)),:] = [time MPS.traceMPOprod(mps,MPS.MpoFromOperators([[sx,magnet_pos]],L),2)/tr_rho]
                    spin_pos = [[sz,Int(floor(L/4))], [sz,Int(floor(3/4*L))]] # position of spins in chain for correlation fct
                    correlation[Int(ceil(counter/increment)),:] = [time MPS.traceMPOprod(mps,MPS.MpoFromOperators(spin_pos,L),2)/tr_rho]
                    # corr_length[Int(floor(counter/increment))+1,:] = [time MPS.correlation_length(rho,d)[2]]
                end
            elseif mpo == "Heisenberg"
                Jx0, Jy0, Jz0, hx0 = params
                Jx, Jy, Jz, hx = evolveHeisenbergParams(Jx0, Jy0, Jz0, hx0, time)
                hamiltonian = MPS.HeisenbergMPO(L, Jx, Jy, Jz, hx)
                expect[counter,:] = [time MPS.mpoExpectation(mps,hamiltonian)]
            else
                expect[counter,:] = [time MPS.mpoExpectation(mps,mpo)]
            end
        end

        ## entanglement entropy:
        if entropy_cut > 0
            entropy[counter,:] = [time MPS.entropy(mps,entropy_cut)]
        end

		## ETH calculations:
		if eth[1] == true
			E1, hamiltonian = real(eth[2]), eth[3]
			rho = MPS.multiplyMPOs(mps,mps)
			E_thermal = real(MPS.traceMPOprod(rho,hamiltonian))
			if E_thermal <= E1
				return E_thermal, real(time*1im) # im*time = beta/2
			end
		end
    end

    return expect, entropy, magnetization, correlation, corr_length
end

function tebd_step(mps, hamblocks, dt, D; tol=0.0)
    d = size(mps[1])[2]
    L = length(mps)
    if isodd(L)
        even_start = L-1
        odd_length = true
    else
        even_start = L-2
        odd_length = false
    end
    mpo_to_mps_trafo = false
    if length(size(mps[1])) == 4 # control variable to make mps out of mpo
        mpo_to_mps_trafo = true
    end
    if !mpo_to_mps_trafo && MPS.check_LRcanonical(mps[1],-1) # use rightcanonical mps as default input for sweeping direction
        MPS.makeCanonical(mps)
    end
    total_error = 0
    ## ************ right sweep over odd sites
    for i = 1:2:L-1
        W = expm(-1im*dt*hamblocks[i])
        W = reshape(W, (d,d,d,d))
        mps[i], mps[i+1], error = block_decimation(W, mps[i], mps[i+1], D, -1, tol=tol)
        total_error += error
        # preserve canonical structure:
        if i < L-2
            if mpo_to_mps_trafo smpo = MPS.mpo_to_mps(mps) end
            mps[i+1],R,DB = MPS.LRcanonical(mps[i+1],-1)
            @tensor mps[i+2][-1,-2,-3] := R[-1,1]*mps[i+2][1,-2,-3]
            if mpo_to_mps_trafo MPS.mps_to_mpo(mps,smpo) end
        end
    end

    ## ************ left sweep over even sites
    if mpo_to_mps_trafo smpo = MPS.mpo_to_mps(mps) end
        mps[L],R,DB = MPS.LRcanonical(mps[L],1) # rightcanonicalize at right end
        @tensor mps[L-1][:] := mps[L-1][-1,-2,1]*R[1,-3]
    if mpo_to_mps_trafo MPS.mps_to_mpo(mps,smpo) end

    for i = even_start:-2:2
        W = expm(-1im*dt*hamblocks[i])
        W = reshape(W, (d,d,d,d))
        mps[i], mps[i+1], error = block_decimation(W, mps[i], mps[i+1], D, 1, tol=tol)
        total_error += error
        # preserve canonical structure:
        if mpo_to_mps_trafo smpo = MPS.mpo_to_mps(mps) end
        mps[i],R,DB = MPS.LRcanonical(mps[i],1) # rightcanonicalize current sites
        @tensor mps[i-1][:] := mps[i-1][-1,-2,1]*R[1,-3]
        if mpo_to_mps_trafo MPS.mps_to_mpo(mps,smpo) end
    end

    if mpo_to_mps_trafo smpo = MPS.mpo_to_mps(mps) end
        mps[1],R,DB = MPS.LRcanonical(mps[1],1) # rightcanonicalize at left end
    if mpo_to_mps_trafo MPS.mps_to_mpo(mps,smpo) end

    return total_error
end

function tebd_simplified(mps, hamblocks, total_time, steps, D, operators, eth, entropy_cut=0; increment=1, tol=0.0)
    ### block = hamiltonian
    ### use -im*total_time for imaginary time evolution
    ### assumption: start with rightcanonical mps
	### eth = (true,E1,hamiltonian) --> do ETH calcs if true for excited energy E1 wrt hamiltonian

    mpo_to_mps_trafo = false
    if length(size(mps[1])) == 4 # control variable to make mps out of mpo
        mpo_to_mps_trafo = true
    end

    corr_spreading = false
    if length(operators)>0 && operators[1]=="corr_fct" # correlator spreading
        corr_spreading = true
    end

    L = length(mps)
    stepsize = total_time/steps
    if corr_spreading
        Lhalf = operators[2]
        dist_interval = operators[3]
        nop = length(dist_interval)
    else
        nop = length(operators)
    end
    datalength = Int(ceil(steps/increment))
    opvalues = Array{Any,2}(datalength,1+nop)
    err = Array{Any,1}(datalength)
    entropy = Array{Any,1}(datalength)

    for counter = 1:steps
        if counter % 10 == 0
            println("step ",counter," / ",steps)
        end

        time = counter*total_time/steps
        err_tmp = tebd_step(mps,hamblocks(time),stepsize,D, tol=tol)

        ## calculate phys quantities after increment steps:
        if counter % increment == 0
            phys_ind = Int(ceil(counter/increment))
            err[phys_ind] = err_tmp
            opvalues[phys_ind,1] = time

            ## expectation values:
            if corr_spreading
                for k = 1:nop
                    m = dist_interval[k]
                    spin_pos = [[sz,Lhalf], [sz,Lhalf+m]]
                    opvalues[phys_ind,k+1] = MPS.traceMPOprod(mps, MPS.MpoFromOperators(spin_pos,L),2) - MPS.traceMPOprod(mps, MPS.MpoFromOperators([spin_pos[1]],L),2)*MPS.traceMPOprod(mps, MPS.MpoFromOperators([spin_pos[2]],L),2)
                    # opvalues[phys_ind,k+1] = MPS.traceMPOprod(mps[1:max(Lhalf,Lhalf+m)],MPS.MpoFromOperators(spin_pos,max(Lhalf,Lhalf+m)),2)
                end
            else
                for k = 1:nop
                    if mpo_to_mps_trafo
                        opvalues[phys_ind,k+1] = MPS.traceMPOprod(mps,operators[k](time),2)
                    else
                        opvalues[phys_ind,k+1] = MPS.mpoExpectation(mps,operators[k](time))
                    end
                end
            end

            ## entanglement entropy:
            if !mpo_to_mps_trafo && entropy_cut > 0
                entropy[phys_ind] = MPS.entropy(mps,entropy_cut)
            end
        end

		## ETH calculations:
		if eth[1] == true
			E1, hamiltonian = real(eth[2]), eth[3]
			E_thermal = real(MPS.traceMPOprod(mps,hamiltonian,2))
			if E_thermal <= E1
				return E_thermal, real(time*1im) # im*time = beta/2
			end
		end
    end

    if !mpo_to_mps_trafo && entropy_cut > 0
        return opvalues, err, entropy
    else
        return opvalues, err
    end
end


###### module end
end
