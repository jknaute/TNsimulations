### tricritical Ising model on an infinite translational invariant 3-site unit cell

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
inc=100
beta_th = 20.0/2
steps_th = 10000
conv_prec = 0 # relative convergence threshold/precision for imaginary time evolution break w.r.t. to first operator (energy); set to 0 to run full beta



## file things:
dict_filename = "linearresponse/continuum_limit_tricritical/thermalstate_lambda04_beta20_iTEBD3"
output_filename = "linearresponse/continuum_limit_tricritical/s2lambda04beta20iTEBD3"
beta_plot = 2*beta_th


## Ising parameters:
lambda = 0.4 # TCI = 0.428


function sth(N,beta,time,steps,D)
    return string("L= ",N,"  beta= ",beta,"  t_max= ",time,"  steps= ",steps,"  D= ", D,"  prec= ",prec,"  lambda= ",lambda,"\n")
end

function save_data(data, filename= string(@__DIR__,"/data/quench/opvals.txt"); header="")
    open(filename, "a") do f
        write(f, header)
        writedlm(f, data)
        write(f,"\r\n")
    end
end


##------------------------------------  thermal state  ----------------------------------

## Hamiltonian
""" return the OF model Hamiltonian: eq. (4) in [1710.05397]"""
function H_tricritical(coupling)
    I = eye(2)
    X = Float64.(sx)
    Z = Float64.(sz)
    ## full symmetrization:
    h0 = ( -0.5*( kron(I,kron(Z,Z))+kron(Z,kron(Z,I)) ) # -ZZ = -(ZZI+IZZ)/2
           -(1.0/3.0)*( kron(I,kron(I,X))+kron(I,kron(X,I))+kron(X,kron(I,I)) ) # -X = -(XII+IXI+IIX)/3
           + coupling*( kron(Z,kron(Z,X))+kron(X,kron(Z,Z)) ) # + coupling*(XZZ+ZZX)
         )
    return h0
end

hamblocksTH = H_tricritical(lambda)
h = reshape(hamblocksTH, d,d,d,d,d,d)

## initializations:
mpo = MPS.IdentityMPO(N,d)
gTH,lTH = MPS.prepareGL(mpo,maxD)
MA=mpo[1]; MB=mpo[2]; MC=mpo[3]
lA=lTH[2]; lB=lTH[3]; lC=lTH[4]

## thermal state construction:
## calculate:
@time MA,MB,MC, lA,lB,lC, errL,errR, betas, ops, renyi = MPS.gl_iTEBD3_timeevolution(MA,MB,MC, lA,lB,lC, hamblocksTH, -im*beta_th, steps_th, d, maxD, [], tol=prec, increment=inc, conv_thresh=conv_prec, calculate_2Renyi=true)
## save:
thermal_data = Dict(:M => [MA,MB,MC], :l => [lA,lB,lC], :err => [errL,errR], :info => "maxD=$maxD, prec=$prec, steps_th=$steps_th, beta_th=$beta_plot, conv_prec=$conv_prec, lambda=$lambda")
BSON.bson(string(@__DIR__,"/data/"*dict_filename*".bson"), thermal_data)


## some data:
E_th = MPS.expect_operator_average(MA,MB,MC, lA,lB,lC, h)
Tr_rho = MPS.trace_rho_average(MA,MB,MC, lA,lB,lC)
println("E_th = ",E_th," , Tr(rho) = ",Tr_rho)




## SAVING:
save_data(cat(2, 2*real(im*betas),real(errL),real(errR),real(renyi[:,1])), string(@__DIR__,"/data/"*output_filename*".txt"), header=string(sth(N,beta_plot,0,steps_th,maxD), "# beta \t errL \t errR \t s2\n"))





println("done: gl_iTEBD3_tricritical.jl")
# show()
;
