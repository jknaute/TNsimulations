### tricritical Ising model on a finite chain with symmetrized 2-site gates of physical dimension = 4

### export JULIA_NUM_THREADS=17

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
N_bare = 40
N = Int(N_bare/2) # symmetrized system size for 2-site tensors
maxD=200
prec=1e-20
d = 4

beta_th = 8.0/2
steps_th = 200

total_time_quench = 10
steps = 500
inc=2


## file things:
dict_filename = "linearresponse/continuum_limit_tricritical/thermalstate_tricrit_beta8_N40"
output_filename = "linearresponse/continuum_limit_tricritical/responsetricritbeta8N40"
beta_plot = 2*beta_th



## Ising parameters:
lambda = 0.428 # TCI = 0.428




##------------------------------------  quench  ----------------------------------
## Hamiltonian
""" return the OF model Hamiltonian: eq. (4) in [1710.05397]"""
I = eye(2)
X = Float64.(sx)
Z = Float64.(sz)
## full symmetrization:
h0 = ( -0.5*( kron(I,kron(Z,Z))+kron(Z,kron(Z,I)) ) # -ZZ = -(ZZI+IZZ)/2
       -(1.0/3.0)*( kron(I,kron(I,X))+kron(I,kron(X,I))+kron(X,kron(I,I)) ) # -X = -(XII+IXI+IIX)/3
       + lambda*( kron(Z,kron(Z,X))+kron(X,kron(Z,Z)) ) # + coupling*(XZZ+ZZX)
     )
h0 = reshape(cg.symmetrize_vector(reshape(h0,64)), 16,16)
blocks = Array{Array{Complex128,2},1}(N)
for i=1:N
    blocks[i] = h0
end
hamblocksTH(time) = blocks

## initializations:
mpo = MPS.IdentityMPO(N,d)
gTH,lTH = MPS.prepareGL(mpo,maxD)



## operators to measure:
eps_prime = kron(Z,kron(Z,X))+kron(X,kron(Z,Z))
eps_prime_symm  = reshape(cg.symmetrize_vector(reshape(eps_prime,64)), d,d,d,d)



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


## thermal state construction:
## calculate:
# @time opvalsth, errth = MPS.gl_tebd(gTH,lTH,hamblocksTH,-beta_th*im,steps_th,maxD,[],tol=prec,increment=1,st2=true)
## save:
# thermal_data = Dict(:g => gTH, :l => lTH, :err => errth, :info => "N=$N, maxD=$maxD, prec=$prec, steps_th=$steps_th, beta_th=$beta_plot, lambda=$lambda")
# BSON.bson(string(@__DIR__,"/data/"*dict_filename*".bson"), thermal_data)
## load:
thermo = BSON.load(string(@__DIR__,"/data/"*dict_filename*".bson"))
gTH = thermo[:g] # initial thermal state sqrt{rho}
lTH = thermo[:l]
gTH = convert(Array{Array{Complex{Float64},4},1}, gTH)
lTH = convert(Array{Array{Complex{Float64},2},1}, lTH)
println(thermo[:info])


gP = deepcopy(gTH) # perturbed state
lP = deepcopy(lTH)


## local perturbation:
gP,lP = MPS.twositeop_on_gl(gP,lP,eps_prime_symm,Int(floor(length(gP)/2)),maxD,prec) # local pert w/ eps_prime on mid chain

## quench evolution:
@time opvals2, errP, times = MPS.gl_tebd_cg(gTH,lTH,gP,lP,hamblocksTH,eps_prime_symm,total_time_quench,steps,maxD,tol=prec,increment=inc,st2=true)


## SAVING:
save_data(cat(2,real(times),real(opvals2),imag(opvals2),real(errP)), string(@__DIR__,"/data/"*output_filename*".txt"), header=string(sth(N,2*beta_th,total_time_quench,steps,maxD), "# t \t ReG \t ImG \t errP\n"))
# finalstate_data = Dict(:gA => gA, :lA => lA, :gB => gB, :lB => lB, :times => times, :errA => errA, :errB => errB, :info => "N=$N, maxD=$maxD, prec=$prec, steps=$steps, beta_th=$beta_plot, tmax/2=$total_time_quench, J0=$J0, h0=$h0, g0=$g0")
# BSON.bson(string(@__DIR__,"/data/"*output_filename*".bson"), finalstate_data)






println("done: gl_para_tricritical.jl")
# show()
;
