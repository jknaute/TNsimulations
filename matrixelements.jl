filepath = string(@__DIR__,"/MPSmodule.jl")
include(filepath)
using MPS
using BSON
using PyPlot
Base.BLAS.set_num_threads(1)
using Base.Threads


## chain and evolution parameters:
N=200
maxD_mps=100
prec_DMRG = 1e-8

## Ising parameters:
## nonint1:
J0 = -1.0
h0 = -0.9375
g0 = -0.07457159307550416

## nonint2:
# J0 = -1.0
# h0 = -0.75
# g0 = -0.013757019299574723


## file things:
dict_filename = "linearresponse/continuum_limit_zeroT/statesnonint1newDMRG_D100"
output_filename = "linearresponse/continuum_limit_zeroT/statesnonint1newDMRG_D100"

function sth(N,time,steps,D)
    return string("L= ",N,"  beta= inf","  t_max= ",time,"  steps= ",steps,"  D= ", D,"  prec= ",prec,"  [J0,h0,g0]= ",[J0,h0,g0],"\n")
end

function save_data(data, filename= string(@__DIR__,"/data/quench/opvals.txt"); header="")
    open(filename, "a") do f
        write(f, header)
        writedlm(f, data)
        write(f,"\r\n")
    end
end


## operators to measure:
hamiltonian = MPS.IsingMPO(N, J0, h0, g0)


## ground state and excited states:
# mps = MPS.randomMPS(N,2,maxD_mps) # d=2
# MPS.makeCanonical_old(mps)
# states,energies = MPS.n_lowest_states(mps, hamiltonian, prec_DMRG,21)


## load:
states_data = BSON.load(string(@__DIR__,"/data/"*dict_filename*".bson"))
println(states_data[:info])
states = states_data[:states]
energies = states_data[:energies]

# ## further excited states:
# mps = MPS.randomMPS(N,2,maxD_mps) # d=2
# MPS.makeCanonical_old(mps)
# states,energies = MPS.n_further_states(mps, hamiltonian, prec_DMRG, states, energies, 5)



## save:
# states_data = Dict(:states => states, :energies => energies, :info => "N=$N, maxD=$maxD_mps, prec_DMRG=$prec_DMRG, J0=$J0, h0=$h0, g0=$g0")
# BSON.bson(string(@__DIR__,"/data/"*dict_filename*".bson"), states_data)



## MPO for sum{sx}:
pert_mpo = MPS.qMPO(N,sx,0)


## Data and Plots:
println("\nenergies: ")
for i=1:length(energies)
    println(i-1,", ",energies[i]-energies[1])
end

## magnetization profiles:
figure(1)
magsz = Array{Complex128,2}(N,length(states))
for k=1:length(states)
    println("k = ",k)
    state = states[k]
    @threads for i=1:N
        mposz = MPS.MpoFromOperators([[sz,i]],N)
        magsz[i,k] = MPS.mpoExpectation(state,mposz)
    end
    plot(collect(1:N),real(magsz[:,k]),label=string(k-1))
end
legend(loc = "best", numpoints=3, frameon = 0, fancybox = 0, columnspacing = 1)

println("\nmatrix elements: ")
for i=1:length(states)
      println("0-",i-1,", ",abs(MPS.mpoExpectation(states[1],pert_mpo,states[i])))
end


## local kink density:
figure(2)
nu_i = Array{Complex128,2}(N-1,length(states))
for k=1:length(states)
    println("k = ",k)
    state = states[k]
    @threads for i=1:N-1
        ops_i = [[sz,i],[sz,i+1]]
        mpo_i = MPS.MpoFromOperators(ops_i,N)
        nu_i[i,k] = 0.5(1-MPS.mpoExpectation(state,mpo_i))
    end
    plot(collect(1:N-1),real(nu_i[:,k]),label=string(k-1))
end
legend(loc = "best", numpoints=3, frameon = 0, fancybox = 0, columnspacing = 1)


## kink-kink correlator:
figure(3)
nu_ij = Array{Complex128,3}(N-1,N-1,length(states))
# for k=1:length(states)
#     println("k = ",k)
#     state = states[k]
#     for i=1:N-1
#         println("i = ",i)
#         ops_i = [[sz,i],[sz,i+1]]
#         mpo_i = MPS.MpoFromOperators(ops_i,N)
#         mpo_i = convert(Array{Array{Float64,4},1}, mpo_i)
#         @threads for j=1:N-1
#             ops_j = [[sz,j],[sz,j+1]]
#             mpo_j = MPS.MpoFromOperators(ops_j,N)
#             mpo_j = convert(Array{Array{Float64,4},1}, mpo_j)
#             mpo_ij = MPS.multiplyMPOs(mpo_i,mpo_j)
#             nu_ij[i,j,k] = (1-nu_i[i,k]-nu_i[j,k]+MPS.mpoExpectation(state,mpo_ij))/4 - (nu_i[i,k]*nu_i[j,k])
#         end
#     end
# end
nu_data = Dict(:magsz => magsz, :nu_i => nu_i, :nu_ij => nu_ij)
BSON.bson(string(@__DIR__,"/data/"*dict_filename*"_nu.bson"), nu_data)
nu_data = BSON.load(string(@__DIR__,"/data/"*dict_filename*"_nu.bson"))



println("done: matrixelements.jl")
show()
;
