### Ising model on an infinite translational invariant 2-site unit cell:
### calculate thermal state under H0 and observe 2-Renyi entropy density in real-time quench under H1


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
# using PyPlot
using TensorOperations
using Base.Threads
# Base.BLAS.set_num_threads(16)


## chain and evolution parameters:
N = 2
d = 2
maxD=500
prec=1e-15
maxErr=1e-6

beta_th = 0.5/2
steps_th = 500
inc_th=1
conv_prec = 0 # relative convergence threshold/precision for imaginary time evolution break w.r.t. to first operator (energy); set to 0 to run full beta

total_time_quench = 10.0
steps = 10000
inc_t = 50

## file things:
dict_filename = "linearresponse/continuum_limit_renyiquench/freefermions/thermalstate_free1_beta05_iTEBD2"
output_filename = "linearresponse/continuum_limit_renyiquench/freefermions/entropiesfreequenchtype1beta05iTEBD2D500"
beta_plot = 2*beta_th


## thermal Ising parameters:
J0 = -1.0
h0 = -0.93 # -0.9375 # -0.93
g0 = -0.0 # -100.0 # -0.07457159307550416

## real-time Ising parameters:
J1 = -1.0
h1 = -0.9375 # -0.9827 # -0.88
g1 = -0.0 # -0.07457159307550416


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


##------------------------------------  quench  ----------------------------------

## Hamiltonian
""" return Ising Hamiltonian on 2 sites """
function H_Ising(J,h,g)
    ham0 = J*ZZ + h*0.5*(XI+IX) + g*0.5*(ZI+IZ)
    # HI = kron(eye(d^2),ham0) # = H*I for MPO in MPS form
    return ham0
end

hamblocksTH = H_Ising(J0,h0,g0)
H0 = reshape(hamblocksTH, d,d,d,d)
hamblocksQuench = H_Ising(J1,h1,g1)
H1 = reshape(hamblocksQuench, d,d,d,d)


## initializations:
mpo = MPS.IdentityMPO(N,d)
gTH,lTH = MPS.prepareGL(mpo,maxD)
MA=mpo[1]; MB=mpo[2]
lA=lTH[2]; lB=lTH[3]



## thermal state construction:
## calculate:
# MA,MB, lA,lB, err_th, betas, ops = MPS.gl_iTEBD2_timeevolution(MA,MB, lA,lB, hamblocksTH, -im*beta_th, steps_th, d, maxD, [], tol=prec, increment=inc_th, conv_thresh=conv_prec, calculate_2Renyi=false)
## save:
# thermal_data = Dict(:M => [MA,MB], :l => [lA,lB], :err => err_th, :info => "maxD=$maxD, prec=$prec, steps_th=$steps_th, beta_th=$beta_plot, conv_prec=$conv_prec, J0=$J0, h0=$h0, g0=$g0")
# BSON.bson(string(@__DIR__,"/data/"*dict_filename*".bson"), thermal_data)

## load:
thermo = BSON.load(string(@__DIR__,"/data/"*dict_filename*".bson"))
## initial thermal state sqrt{rho}:
MA=thermo[:M][1]; MB=thermo[:M][2]
lA=thermo[:l][1]; lB=thermo[:l][2]
println(thermo[:info])

## some data:
E_th_0 = MPS.expect_operator_average([MA,MB], [lA,lB], H0)
E_th_1 = MPS.expect_operator_average([MA,MB], [lA,lB], H1)
E_var1 = MPS.expect_operator_average([MA,MB], [lA,lB], H1, 2) - E_th_1^2 # = <H1^2> - <H1>^2
Tr_rho = MPS.trace_rho_average([MA,MB], [lA,lB])
println("E_th_0 = ",E_th_0)
println("E_th_1 = ",E_th_1)
println("E_var1 = ",E_var1)
println("Tr(rho) = ",Tr_rho)


## quench evolution:
println("\n quench evolution")
MPS.check_triple_canonical([MA,MB],[lA,lB])
MA,MB, lA,lB, err_t, time, ops, renyi = MPS.gl_iTEBD2_timeevolution(MA,MB, lA,lB, hamblocksQuench, total_time_quench, steps, d, maxD, [], tol=prec, increment=inc_t, conv_thresh=conv_prec, do_recanonicalization=true, calculate_2Renyi=true, err_max=maxErr)
MPS.check_triple_canonical([MA,MB],[lA,lB])


## SAVING:
save_data(cat(2, real(time),real(err_t),real(renyi)), string(@__DIR__,"/data/"*output_filename*".txt"), header=string(sth(N,2*beta_th,total_time_quench,steps,maxD), "# t \t err \t s1 \t s2\n"))





println("done: gl_iTEBD2_renyiquench.jl")
# show()
;
