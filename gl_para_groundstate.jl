filepath = string(@__DIR__,"/MPSmodule.jl")
include(filepath)
using MPS
using BSON
using PyPlot
Base.BLAS.set_num_threads(1)


## chain and evolution parameters:
N=200
maxD_mps=13
maxD=169
prec_DMRG = 1e-8
prec=1e-20
inc=1
steps = 2500
total_time_quench = 25
q = 2*pi*(3/(N-1))

## file things:
dict_filename = "linearresponse/continuum_limit_zeroT/groundstateproj"
output_filename = "linearresponse/continuum_limit_zeroT/responseGSproj"


## Ising parameters:
J0 = -1.0
h0 = -0.9375
g0 = -0.07457159307550416


## operators to measure:
hamblocksTH(time) = MPS.isingHamBlocks(N,J0,h0,g0)
hamiltonian = MPS.IsingMPO(N, J0, h0, g0)


## ground state:
mps = MPS.randomMPS(N,2,maxD_mps) # d=2
MPS.makeCanonical_old(mps)
states,energies = MPS.n_lowest_states(mps, hamiltonian, prec_DMRG,1)
ground = states[1]
E0 = energies[1]
G,L = MPS.prepareGL(ground,maxD_mps)


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


## save:
groundstate_data = Dict(:g => G, :l => L, :info => "N=$N, maxD=$maxD_mps, prec_DMRG=$prec_DMRG, J0=$J0, h0=$h0, g0=$g0")
BSON.bson(string(@__DIR__,"/data/"*dict_filename*".bson"), groundstate_data)
## load:
groundstate = BSON.load(string(@__DIR__,"/data/"*dict_filename*".bson"))
G = groundstate[:g]
L = groundstate[:l]
G = convert(Array{Array{Complex{Float64},3},1}, G)
L = convert(Array{Array{Complex{Float64},2},1}, L)
println(groundstate[:info])

## projector formalism |0><0| :
ground_mps = MPS.gl_to_mps(G,L)
ground_proj = MPS.pureDensityMatrix(ground_mps)
G,L = MPS.prepareGL(ground_proj,maxD)

gA = copy(G)
lA = copy(L)
gB = copy(G)
lB = copy(L)

## global perturbation:
pert_mpo = MPS.qMPO(N,sx,0)
# pert_mpo = MPS.qMPO(N,sz,0) # longitudinal
pertg = MPS.mpo_on_gl(gA,lA,pert_mpo)
gA, lA = MPS.prepareGL(pertg,maxD,prec)

## local perturbation:
pert_ops = fill(Complex128.(si),N)
pert_ops[Int(floor(N/2))] = sx
# pert_ops[Int(floor(N/2))] = sz # longitudinal
# MPS.ops_on_gl_dummy(gB,lB,pert_ops)
MPS.ops_on_gl(gB,lB,pert_ops)


## quench evolution:
@time opvals2, errA,errB, times = MPS.gl_tebd_c(gA,lA,gB,lB,hamblocksTH,total_time_quench,steps,maxD,tol=prec,increment=inc,st2=true,legflip=true)


## SAVING:
save_data(cat(2,real(times),real(opvals2),imag(opvals2)), string(@__DIR__,"/data/"*output_filename*".txt"), header=string(sth(N,total_time_quench,steps,maxD), "# t \t ReG \t ImG\n"))
# finalstate_data = Dict(:gA => gA, :lA => lA, :gB => gB, :lB => lB, :times => times, :errA => errA, :errB => errB, :info => "N=$N, maxD=$maxD, prec=$prec, steps=$steps, beta_th=$beta_plot, tmax/2=$total_time_quench, J0=$J0, h0=$h0, g0=$g0")
# BSON.bson(string(@__DIR__,"/data/"*output_filename*".bson"), finalstate_data)










println("done: gl_para_groundstate.jl")
show()
;
