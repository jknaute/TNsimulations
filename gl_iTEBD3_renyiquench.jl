### Ising model on an infinite translational invariant 3-site unit cell:
### calculate thermal state under H0 and observe 2-Renyi entropy density in real-time quench under H1

module MPS
export sx,sy,sz,si,s0,ZZ,ZI,IZ,XI,IX,II,ii
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
const II = kron(eye(4), eye(4)) # kron(eye(8), eye(8)) # kron(si,si)
const ii = kron(si,si)
include(string(@__DIR__,"/MPS.jl"))
include(string(@__DIR__,"/gl.jl"))
end

include(string(@__DIR__,"/coarsegrained.jl"))
using MPS
using cg
using BSON
using PyPlot
using TensorOperations
using Base.Threads
Base.BLAS.set_num_threads(1)


## chain and evolution parameters:
N = 3
d = 2
maxD=80
prec=1e-15

beta_th = 8.0/2
steps_th = 100000
inc_th=1000
conv_prec = 0 # relative convergence threshold/precision for imaginary time evolution break w.r.t. to first operator (energy); set to 0 to run full beta

total_time_quench = 30.0
steps = 10000
inc_t = 50

## file things:
dict_filename = "linearresponse/continuum_limit_renyiquench/thermalstate_E8_beta8_iTEBD3"
output_filename = "linearresponse/continuum_limit_renyiquench/s2E8beta8iTEBD3"
beta_plot = 2*beta_th


## thermal Ising parameters:
J0 = -1.0
h0 = -1.0
g0 = -0.003

## real-time Ising parameters:
J1 = -1.0
h1 = -1.0
g1 = g0*(1+0.05)


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
""" return Ising Hamiltonian on 3 sites """
function H_Ising(J,h,g)
    I = eye(2)
    X = Float64.(sx)
    Z = Float64.(sz)
    ## full symmetrization:
    h0 = ( J*0.5*( kron(I,kron(Z,Z))+kron(Z,kron(Z,I)) ) # J*ZZ = J*(ZZI+IZZ)/2
           +(h/3.0)*( kron(I,kron(I,X))+kron(I,kron(X,I))+kron(X,kron(I,I)) ) # h*X = h*(XII+IXI+IIX)/3
           +(g/3.0)*( kron(I,kron(I,Z))+kron(I,kron(Z,I))+kron(Z,kron(I,I)) ) # g*Z = g*(ZII+IZI+IIZ)/3
         )
    return h0
end

hamblocksTH = H_Ising(J0,h0,g0)
hamblocksQuench = H_Ising(J1,h1,g1)


## initializations:
mpo = MPS.IdentityMPO(N,d)
gTH,lTH = MPS.prepareGL(mpo,maxD)
MA=mpo[1]; MB=mpo[2]; MC=mpo[3]
lA=lTH[2]; lB=lTH[3]; lC=lTH[4]

## thermal state construction:
## calculate:
# MA,MB,MC, lA,lB,lC, errL_th,errR_th, betas, ops = MPS.gl_iTEBD3_timeevolution(MA,MB,MC, lA,lB,lC, hamblocksTH, -im*beta_th, steps_th, d, maxD, [], tol=prec, increment=inc_th, conv_thresh=conv_prec, calculate_2Renyi=false)
## save:
# thermal_data = Dict(:M => [MA,MB,MC], :l => [lA,lB,lC], :err => [errL_th,errR_th], :info => "maxD=$maxD, prec=$prec, steps_th=$steps_th, beta_th=$beta_plot, conv_prec=$conv_prec, J0=$J0, h0=$h0, g0=$g0")
# BSON.bson(string(@__DIR__,"/data/"*dict_filename*".bson"), thermal_data)

## load:
thermo = BSON.load(string(@__DIR__,"/data/"*dict_filename*".bson"))
## initial thermal state sqrt{rho}:
MA=thermo[:M][1]; MB=thermo[:M][2]; MC=thermo[:M][3]
lA=thermo[:l][1]; lB=thermo[:l][2]; lC=thermo[:l][3]
println(thermo[:info])

## some data:
E_th = MPS.expect_operator_average(MA,MB,MC, lA,lB,lC, reshape(hamblocksTH, d,d,d,d,d,d))
Tr_rho = MPS.trace_rho_average(MA,MB,MC, lA,lB,lC)
println("E_th = ",E_th," , Tr(rho) = ",Tr_rho)


## quench evolution:
println("\n quench evolution")
MA,MB,MC, lA,lB,lC, errL_t,errR_t, time, ops, renyi = MPS.gl_iTEBD3_timeevolution(MA,MB,MC, lA,lB,lC, hamblocksQuench, total_time_quench, steps, d, maxD, [], tol=prec, increment=inc_t, conv_thresh=conv_prec, calculate_2Renyi=true)


## SAVING:
save_data(cat(2, real(time),real(errL_t),real(errR_t),real(renyi[:,1])), string(@__DIR__,"/data/"*output_filename*".txt"), header=string(sth(N,2*beta_th,total_time_quench,steps,maxD), "# t \t errL \t errR \t s2\n"))





println("done: gl_iTEBD3_renyiquench.jl")
# show()
;
