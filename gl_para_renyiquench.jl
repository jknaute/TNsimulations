### Ising model on a finite chain:
### calculate thermal state under H0 and observe 2-Renyi entropy density
### for finite subsystem in real-time quench under H1


module iTEBD2
include(string(@__DIR__,"/iTEBD2.jl"))
end
module Renyi
include(string(@__DIR__,"/Renyi.jl"))
end

module MPS
using iTEBD2
using Renyi
export sx,sy,sz,si,s0,ZZ,ZI,IZ,XI,IX,II
const sx = [0 1; 1 0]
const sy = [0 1im; -1im 0]
const sz = [1 0; 0 -1]
const si = [1 0; 0 1]
const s0 = [0 0; 0 0]
const ZZ = kron(sz, sz)
const ZI = kron(sz, si)
const IZ = kron(si, sz)
const XI = kron(sx, si)
const IX = kron(si, sx)
const II = kron(si, si)
include(string(@__DIR__,"/MPS.jl"))
include(string(@__DIR__,"/gl.jl"))
end

using iTEBD2
using Renyi
using MPS
using BSON
using PyPlot
using TensorOperations
using Base.Threads
Base.BLAS.set_num_threads(1)


## chain and evolution parameters:
N=10
maxD=20
prec=1e-20

beta_th = 8.0/2
steps_th = 50

total_time_quench = 0.1
steps = 50
inc_t = 5


## file things:
dict_filename = "linearresponse/continuum_limit_renyiquench/test"
output_filename = "linearresponse/continuum_limit_renyiquench/test"
beta_plot = 2*beta_th

## thermal Ising parameters:
J0 = -1.0
h0 = -1.0
g0 = -0.01

## real-time Ising parameters:
J1 = -1.0
h1 = -1.0
g1 = g0*(1+0.05)

# ## subsystem sizes:
# sublengths = [10]


## operators to measure:
hamblocksTH(time) = MPS.isingHamBlocks(N,J0,h0,g0)
hamblocksQuench(time) = MPS.isingHamBlocks(N,J1,h1,g1)


## initializations:
mpo = MPS.IdentityMPO(N,2)
G,L = MPS.prepareGL(mpo,maxD)


function sth(N,beta,time,steps,D)
    return string("L= ",N,"  beta= ",beta,"  t_max= ",time,"  steps= ",steps,"  D= ", D,"  prec= ",prec,"  [J0,h0,g0]= ",[J0,h0,g0],"  [J1,h1,g1]= ",[J1,h1,g1],"\n")
end

function save_data(data, filename= string(@__DIR__,"/data/quench/opvals.txt"); header="")
    open(filename, "a") do f
        write(f, header)
        writedlm(f, data)
        write(f,"\r\n")
    end
end


## thermal state construction:
## calculate:
@time opvalsth, errth = MPS.gl_tebd(G,L,hamblocksTH,-beta_th*im,steps_th,maxD,[],tol=prec,increment=1,st2=true)
## save:
thermal_data = Dict(:g => G, :l => L, :err => errth, :info => "N=$N, maxD=$maxD, prec=$prec, steps_th=$steps_th, beta_th=$beta_plot, J0=$J0, h0=$h0, g0=$g0")
BSON.bson(string(@__DIR__,"/data/"*dict_filename*".bson"), thermal_data)
## load:
thermo = BSON.load(string(@__DIR__,"/data/"*dict_filename*".bson"))
G = thermo[:g]
L = thermo[:l]
G = convert(Array{Array{Complex{Float64},4},1}, G)
L = convert(Array{Array{Complex{Float64},2},1}, L)
println(thermo[:info])





## quench evolution:
@time times, err, renyi = MPS.gl_tebd_renyi(G,L,hamblocksQuench,total_time_quench,steps,maxD,tol=prec,increment=inc_t,st2=true)


## SAVING:
save_data(cat(2,real(times),real(err),real(renyi)), string(@__DIR__,"/data/"*output_filename*".txt"), header=string(sth(N,beta_plot,total_time_quench,steps,maxD), "# t \t err \t s2i\n"))
# finalstate_data = Dict(:gA => gA, :lA => lA, :gB => gB, :lB => lB, :times => times, :errA => errA, :errB => errB, :info => "N=$N, maxD=$maxD, prec=$prec, steps=$steps, beta_th=$beta_plot, tmax/2=$total_time_quench, J0=$J0, h0=$h0, g0=$g0")
# BSON.bson(string(@__DIR__,"/data/"*output_filename*".bson"), finalstate_data)










println("done: gl_para_renyiquench.jl")
show()
;
