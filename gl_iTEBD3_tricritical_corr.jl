### tricritical Ising model on an infinite translational invariant 3-site unit cell
### calculate eps'(t)eps'(0) correlation function in real-time evolution

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
Base.BLAS.set_num_threads(8)


## chain and evolution parameters:
N = 3
d = 2
maxD=200
prec=1e-15

beta_th = 8.0/2
steps_th = 50000
inc_th = 1000
conv_prec = 0 # relative convergence threshold/precision for imaginary time evolution break w.r.t. to first operator (energy); set to 0 to run full beta

total_time_quench = 20
steps = 20000 # real-time steps
inc_t = 20




## file things:
dict_filename = "linearresponse/continuum_limit_tricritical/thermalstate_crit_beta8_iTEBD3"
output_filename = "linearresponse/continuum_limit_tricritical/responsecritbeta8iTEBD3"
beta_plot = 2*beta_th


## Ising parameters:
lambda = 0.0 # TCI = 0.428


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
I = eye(2)
X = Float64.(sx)
Z = Float64.(sz)
function H_tricritical(coupling)
    ## full symmetrization:
    h0 = ( -0.5*( kron(I,kron(Z,Z))+kron(Z,kron(Z,I)) ) # -ZZ = -(ZZI+IZZ)/2
           -(1.0/3.0)*( kron(I,kron(I,X))+kron(I,kron(X,I))+kron(X,kron(I,I)) ) # -X = -(XII+IXI+IIX)/3
           + coupling*( kron(Z,kron(Z,X))+kron(X,kron(Z,Z)) ) # + coupling*(XZZ+ZZX)
         )
    return h0
end

hamblocksTH = H_tricritical(lambda)
ham = reshape(hamblocksTH, d,d,d,d,d,d)


## initializations:
mpo = MPS.IdentityMPO(N,d)
gTH,lTH = MPS.prepareGL(mpo,maxD)
MA_th=mpo[1]; MB_th=mpo[2]; MC_th=mpo[3]
lA_th=lTH[2]; lB_th=lTH[3]; lC_th=lTH[4]



## thermal state construction:
## calculate:
MA_th,MB_th,MC_th, lA_th,lB_th,lC_th, errL_th,errR_th, betas, ops = MPS.gl_iTEBD3_timeevolution(MA_th,MB_th,MC_th, lA_th,lB_th,lC_th, hamblocksTH, -im*beta_th, steps_th, d, maxD, [], tol=prec, increment=inc_th, conv_thresh=conv_prec)
## save:
thermal_data = Dict(:M => [MA_th,MB_th,MC_th], :l => [lA_th,lB_th,lC_th], :err => [errL_th,errR_th], :info => "maxD=$maxD, prec=$prec, steps_th=$steps_th, beta_th=$beta_plot, conv_prec=$conv_prec, lambda=$lambda")
BSON.bson(string(@__DIR__,"/data/"*dict_filename*".bson"), thermal_data)

## load:
thermo = BSON.load(string(@__DIR__,"/data/"*dict_filename*".bson"))
## initial thermal state sqrt{rho}:
MA_th=thermo[:M][1]; MB_th=thermo[:M][2]; MC_th=thermo[:M][3]
lA_th=thermo[:l][1]; lB_th=thermo[:l][2]; lC_th=thermo[:l][3]
println(thermo[:info])

## some data:
E_th = MPS.expect_operator_average([MA_th,MB_th,MC_th], [lA_th,lB_th,lC_th], ham)
Tr_rho = MPS.trace_rho_average([MA_th,MB_th,MC_th], [lA_th,lB_th,lC_th])
println("E_th = ",E_th," , Tr(rho) = ",Tr_rho)


## canonicalize the thermal state by applying U(0) several times:
println("\n canonicalization of thermal state")
MA_th,MB_th,MC_th, lA_th,lB_th,lC_th, errLdummy,errRdummy, tdummy, opsdummy = MPS.gl_iTEBD3_timeevolution(MA_th,MB_th,MC_th, lA_th,lB_th,lC_th, hamblocksTH, 0.0, 50, d, maxD, [], tol=prec)
println("Tr_th = ",MPS.trace_rho_average([MA_th,MB_th,MC_th], [lA_th,lB_th,lC_th]))
MPS.check_triple_canonical([MA_th,MB_th,MC_th], [lA_th,lB_th,lC_th])



## perturbed state:
MA_p = deepcopy(MA_th); MB_p = deepcopy(MB_th); MC_p = deepcopy(MC_th)
lA_p = deepcopy(lA_th); lB_p = deepcopy(lB_th); lC_p = deepcopy(lC_th)

## global translationally invariant perturbation with epsilon:
println("\n global perturbation")
eps_prime = kron(Z,kron(Z,X))+kron(X,kron(Z,Z))
eps_prime = reshape(eps_prime, d,d,d,d,d,d)
MA_p,MB_p,MC_p, lA_p,lB_p,lC_p, errL_p,errR_p = MPS.gl_iTEBD3_thirdstep(MA_p,MB_p,MC_p, lC_p, eps_prime, maxD, printmessage=true)
println("Tr_p = ",MPS.trace_rho_average([MA_p,MB_p,MC_p], [lA_p,lB_p,lC_p]))

## canonicalize the perturbed state by applying U(0) several times:
println("\n canonicalization of perturbed state")
MA_p,MB_p,MC_p, lA_p,lB_p,lC_p, errL_p,errR_p, tdummy, opsdummy = MPS.gl_iTEBD3_timeevolution(MA_p,MB_p,MC_p, lA_p,lB_p,lC_p, hamblocksTH, 0.0, 50, d, maxD, [], tol=prec)
println("Tr_p = ",MPS.trace_rho_average([MA_p,MB_p,MC_p], [lA_p,lB_p,lC_p]))
MPS.check_triple_canonical([MA_p,MB_p,MC_p], [lA_p,lB_p,lC_p])



## quench evolution:
println("\n quench evolution")
MA_p,MB_p,MC_p, lA_p,lB_p,lC_p, errL_p,errR_p, time, corr = MPS.gl_iTEBD3_correlatorevolution(MA_p,MB_p,MC_p,lA_p,lB_p,lC_p, MA_th,MB_th,MC_th,lA_th,lB_th,lC_th, hamblocksTH, total_time_quench, steps, d, maxD, [eps_prime], tol=prec, increment=inc_t)


## SAVING:
save_data(cat(2,real(time),real(corr[:,1]),imag(corr[:,1]),real(errL_p),real(errR_p)), string(@__DIR__,"/data/"*output_filename*".txt"), header=string(sth(N,beta_plot,total_time_quench,steps,maxD), "# t \t ReG \t ImG \t errL \t errR\n"))



println("done: gl_iTEBD3_tricritical_corr.jl")
# show()
;
