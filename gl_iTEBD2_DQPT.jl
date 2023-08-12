### Ising model on an infinite translational invariant 2-site unit cell:
### calculate groundstate under H0 and observe entanglement spectrum in real-time quench under H1


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
# Base.BLAS.set_num_threads(1)


## chain and evolution parameters:
N = 2
d = 2

beta_th = 40.0
steps_th = 40000
inc_th=10
Dinit = 10
maxD_th = 10
prec_th = 0 # 1e-15
conv_prec = 0 # relative convergence threshold/precision for imaginary time evolution break w.r.t. to first operator (energy); set to 0 to run full beta

total_time_quench = 60.0
steps = 60000
inc_t = 50
maxD = 490
prec = 1e-15
maxErr = 1e-7


## file things:
dict_filename =    "linearresponse/continuum_limit_DQPT/groundstate_ferro"
entropies_filename = "linearresponse/continuum_limit_DQPT/entropies_ferro_E8"
spectrum_filename =   "linearresponse/continuum_limit_DQPT/spectrum_ferro_E8"
rates_filename =         "linearresponse/continuum_limit_DQPT/rates_ferro_E8"


## initial Ising parameters:
J0 = -1.0
h0 = -0.25
g0 = -0.0

## real-time Ising parameters:
J1 = -1.0
h1 = -1.0
g1 = -0.48


function sth(N)
    return string("L= ",N,"  beta_th= ",beta_th,"  steps_th= ",steps_th,"  inc_th= ",inc_th,"  Dinit= ",Dinit,"  prec_th= ",prec_th,"  conv_prec= ",conv_prec,
                  "  t_max= ",total_time_quench,"  steps_t= ",steps,"  inc_t= ",inc_t,"  maxD= ", maxD,"  prec_t= ",prec,"  maxErr= ",maxErr,
                  "  [J0,h0,g0]= ",[J0,h0,g0],"  [J1,h1,g1]= ",[J1,h1,g1],"\n")
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
gA = randn(Dinit,d,Dinit)
lA = diagm(randn(Dinit))
@tensor MA[-1,-2,-3] := gA[-1,-2,1]*lA[1,-3]
MA,MB, lA,lB = MPS.double_canonicalize_and_normalize(MA,MA, lA,lA, d)
@tensor gA[-1,-2,-3] := MA[-1,-2,1]*inv(lA)[1,-3]; @tensor gB[-1,-2,-3] := MB[-1,-2,1]*inv(lB)[1,-3]
MPS.check_gl_canonical([gA,gB], [lA,lB])



## thermal state construction:
## calculate:
# MA,MB, lA,lB, err_th, betas, ops = MPS.gl_iTEBD2_timeevolution(MA,MB, lA,lB, hamblocksTH, -im*beta_th, steps_th, d, maxD_th, [], tol=prec_th, increment=inc_th, conv_thresh=conv_prec, calculate_2Renyi=false)

## save:
# groundstate_data = Dict(:M => [MA,MB], :l => [lA,lB], :err => err_th, :info => "beta_th=$beta_th, steps_th=$steps_th, inc_th=$inc_th, Dinit=$Dinit, maxD_th=$maxD_th, prec_th=$prec_th, conv_prec=$conv_prec, J0=$J0, h0=$h0, g0=$g0")
# BSON.bson(string(@__DIR__,"/data/"*dict_filename*".bson"), groundstate_data)

## load:
groundstate = BSON.load(string(@__DIR__,"/data/"*dict_filename*".bson"))
MA=groundstate[:M][1]; MB=groundstate[:M][2]
lA=groundstate[:l][1]; lB=groundstate[:l][2]
println(groundstate[:info])


## some data:
@tensor gA[-1,-2,-3] := MA[-1,-2,1]*inv(lA)[1,-3]
@tensor gB[-1,-2,-3] := MB[-1,-2,1]*inv(lB)[1,-3]
E_0 = iTEBD2.expect_twositelocal_dcan(gA, diag(lA), gB, diag(lB), H0)
E_1 = iTEBD2.expect_twositelocal_dcan(gA, diag(lA), gB, diag(lB), H1)
println("E_0 = ",E_0)
println("E_1 = ",E_1)
MPS.check_gl_canonical([gA,gB], [lA,lB])


## quench evolution:
println("\n quench evolution")
MA,MB, lA,lB, err_t, time, ops, expect, renyi, spectrum, rates = MPS.gl_iTEBD2_timeevolution(MA,MB, lA,lB, hamblocksQuench, total_time_quench, steps, d, maxD, [], tol=prec, increment=inc_t, conv_thresh=conv_prec, do_recanonicalization=true, calculate_2Renyi=true, collect_spectrum=true, num_rate_levels=5, err_max=maxErr)

@tensor gA[-1,-2,-3] := MA[-1,-2,1]*inv(lA)[1,-3]; @tensor gB[-1,-2,-3] := MB[-1,-2,1]*inv(lB)[1,-3]
MPS.check_gl_canonical([gA,gB], [lA,lB])


## SAVING:
save_data(cat(2, real(time),real(err_t),real(renyi)), string(@__DIR__,"/data/"*entropies_filename*".txt"), header=string(sth(N), "# t \t err \t s1 \t s2\n"))
save_data(cat(2, real(time),real(spectrum)), string(@__DIR__,"/data/"*spectrum_filename*".txt"), header=string(sth(N), "# t \t s_i^2\n"))
save_data(cat(2, real(time),real(rates)), string(@__DIR__,"/data/"*rates_filename*".txt"), header=string(sth(N), "# t \t r_i\n"))



println("done: gl_iTEBD2_DQPT.jl")
# show()
;
