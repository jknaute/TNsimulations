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
# Base.BLAS.set_num_threads(8)


## chain and evolution parameters:
N = 2
d = 2
maxD=200
prec=1e-15
maxErr=1e-5

beta_th = 4.0012/2
steps_th = 4000
inc_th=1
conv_prec = 0.0 # relative convergence threshold/precision for imaginary time evolution break w.r.t. to first operator (energy); set to 0 to run full beta


## file things:
dict_filename = "linearresponse/xxx"
# output_filename = "linearresponse/continuum_limit_renyiquench/freefermions/energy_vs_beta"
# output_filename = "linearresponse/continuum_limit_renyiquench/nonintferro/energy_vs_beta_vs_s2"
beta_plot = 2*beta_th


## thermal Ising parameters:
J0 = -1.0
h0 = -0.99125 # -0.992188 # -0.9825 # -0.984375 # -0.965 # -0.93 # -0.9827 # -0.87 # -0.96875 # -0.9375
g0 = -0.00151105 # -0.00554257 # -0.0203302 # -0.07457159307550416



function sth(N,beta)
    return string("L= ",N,"  beta= ",beta,"  [J,h,g]= ",[J0,h0,g0],"\n")
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


## initializations:
mpo = MPS.IdentityMPO(N,d)
gTH,lTH = MPS.prepareGL(mpo,maxD)
MA=mpo[1]; MB=mpo[2]
lA=lTH[2]; lB=lTH[3]



## thermal state construction:
## calculate:

## with E(β) calculation:
# MA,MB, lA,lB, err_th, beta_vals, ops = MPS.gl_iTEBD2_timeevolution(MA,MB, lA,lB, hamblocksTH, -im*beta_th, steps_th, d, maxD, [H0], tol=prec, increment=inc_th, conv_thresh=conv_prec, calculate_2Renyi=false)

## with E(β) and Renyi calculation:
# MA,MB, lA,lB, err_th, beta_vals, ops, renyi = MPS.gl_iTEBD2_timeevolution(MA,MB, lA,lB, hamblocksTH, -im*beta_th, steps_th, d, maxD, [H0], tol=prec, increment=inc_th, conv_thresh=conv_prec, calculate_2Renyi=true)

## just construction ρ(β):
MA,MB, lA,lB, err_th, beta_vals, ops = MPS.gl_iTEBD2_timeevolution(MA,MB, lA,lB, hamblocksTH, -im*beta_th, steps_th, d, maxD, [], tol=prec, increment=inc_th, conv_thresh=conv_prec, calculate_2Renyi=false)

## with Renyi calculation (look into function to choose how):
# MA,MB, lA,lB, err_th, beta_vals, ops, renyi = MPS.gl_iTEBD2_timeevolution(MA,MB, lA,lB, hamblocksTH, -im*beta_th, steps_th, d, maxD, [], tol=prec, increment=inc_th, conv_thresh=conv_prec, calculate_2Renyi=true)



## beta values:
beta_vals = 2*im*beta_vals


## some data:
E_th_0 = MPS.expect_operator_average([MA,MB], [lA,lB], H0)
Tr_rho = MPS.trace_rho_average([MA,MB], [lA,lB])
println("Tr(rho) = ",Tr_rho)
println("E_th_0 = ",E_th_0)



# E_th_1 = MPS.expect_operator_average([MA,MB], [lA,lB], reshape(H_Ising(-1.0,-0.9375,-0.07457159307550416), d,d,d,d))
# E_th_1 = MPS.expect_operator_average([MA,MB], [lA,lB], reshape(H_Ising(-1.0,-0.9375,-0.0), d,d,d,d))
E_th_1 = MPS.expect_operator_average([MA,MB], [lA,lB], reshape(H_Ising(-1.0,-0.992188,-0.00151105), d,d,d,d))
println("E_th_1 = ",E_th_1)

# save_data(cat(2, real(beta_vals),real(ops[:,1])), string(@__DIR__,"/data/"*output_filename*".txt"), header=string(sth(N,beta_plot), "# beta \t energy\n"))
# save_data(cat(2, real(beta_vals),real(ops[:,1]),real(renyi[:,2])), string(@__DIR__,"/data/"*output_filename*".txt"), header=string(sth(N,beta_plot), "# beta \t energy \t s2\n"))




println("done: gl_iTEBD2_thermal.jl")
# show()
;
