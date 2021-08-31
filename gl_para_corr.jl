filepath = string(@__DIR__,"/MPSmodule.jl")
include(filepath)
using MPS
using BSON
using PyPlot
Base.BLAS.set_num_threads(1)


## chain and evolution parameters:
N=200
maxD=200
prec=1e-20
maxErr=1e-5

beta_th = 0.5/2
steps_th = 5000

total_time_quench = 25
steps = 15000
inc=5


## file things:
dict_filename = "linearresponse/continuum_limit_nonint/thermalstate_nonint1_beta05"
output_filename = "linearresponse/continuum_limit_nonint/responsenonint1beta05"
beta_plot = 2*beta_th

## constants in non-int cont limit:
c1 = 0.0  # = beta*M_h = beta*2(1-h)
c2 = 0.0  # ~ beta*M_g/4.4 = beta*g^(8/15) / 4.4
q = 2*pi*(3/(N-1))

## Ising parameters:
J0 = -1.0
h0 = -0.9375 # -(1+c1/(2*beta_plot)) ## ATTENTION: choose ferro (-) or para (+)
g0 = -0.07457159307550416 # -(c2/beta_plot)^(15/8)


## operators to measure:
hamblocksTH(time) = MPS.isingHamBlocks(N,J0,h0,g0)


## initializations:
mpo = MPS.IdentityMPO(N,2)
G,L = MPS.prepareGL(mpo,maxD)


function sth(N,beta,time,steps,D)
    return string("L= ",N,"  beta= ",beta,"  t_max= ",time,"  steps= ",steps,"  D= ", D,"  prec= ",prec,"  [J0,h0,g0]= ",[J0,h0,g0],"\n")
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
# @time opvalsth, errth = MPS.gl_tebd(G,L,hamblocksTH,-beta_th*im,steps_th,maxD,[],tol=prec,increment=1,st2=true)
## save:
# thermal_data = Dict(:g => G, :l => L, :err => errth, :info => "N=$N, maxD=$maxD, prec=$prec, steps_th=$steps_th, beta_th=$beta_plot, J0=$J0, h0=$h0, g0=$g0")
# BSON.bson(string(@__DIR__,"/data/"*dict_filename*".bson"), thermal_data)
## load:
thermo = BSON.load(string(@__DIR__,"/data/"*dict_filename*".bson"))
G = thermo[:g]
L = thermo[:l]
G = convert(Array{Array{Complex{Float64},4},1}, G)
L = convert(Array{Array{Complex{Float64},2},1}, L)
println(thermo[:info])

gA = copy(G)
lA = copy(L)
gB = copy(G)
lB = copy(L)

# global perturbation:
# pert_ops = fill(expm(1e-4*im*sx),N)
# MPS.ops_on_gl(gA,lA,pert_ops)

pert_mpo = MPS.qMPO(N,sx,0)
# pert_mpo = MPS.qMPO(N,sz,0) # longitudinal
pertg = MPS.mpo_on_gl(gA,lA,pert_mpo)
gA, lA = MPS.prepareGL(pertg,maxD,prec)

# local perturbation:
pert_ops = fill(Complex128.(si),N)
pert_ops[Int(floor(N/2))] = sx
# pert_ops[Int(floor(N/2))] = sz # longitudinal
MPS.ops_on_gl(gB,lB,pert_ops)


## quench evolution:
@time opvals2, errA,errB, times = MPS.gl_tebd_c(gA,lA,gB,lB,hamblocksTH,total_time_quench,steps,maxD,tol=prec,increment=inc,st2=true, err_max=maxErr)


## SAVING:
save_data(cat(2,real(times),real(opvals2),imag(opvals2),real(errA),real(errB)), string(@__DIR__,"/data/"*output_filename*".txt"), header=string(sth(N,2*beta_th,total_time_quench,steps,maxD), "# t \t ReG \t ImG \t errA \t errB\n"))
# finalstate_data = Dict(:gA => gA, :lA => lA, :gB => gB, :lB => lB, :times => times, :errA => errA, :errB => errB, :info => "N=$N, maxD=$maxD, prec=$prec, steps=$steps, beta_th=$beta_plot, tmax/2=$total_time_quench, J0=$J0, h0=$h0, g0=$g0")
# BSON.bson(string(@__DIR__,"/data/"*output_filename*".bson"), finalstate_data)










println("done: gl_para_corr.jl")
show()
;
