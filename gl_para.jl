# using Plots
# nbr_of_workers = 5
filepath = string(@__DIR__,"/MPSmodule.jl")
# addedprocs=addprocs(nbr_of_workers- length(workers()))
include(filepath)
# @sync @parallel for p in addedprocs
#     remotecall_wait(include,p,filepath)
# end
using MPS
using BSON
using PyPlot
# using Traceur
Base.BLAS.set_num_threads(1)


## chain and evolution parameters:
N=50 # 80
maxD=300 # 350
prec=1e-20
inc=5
steps_th = 10000 # 200
steps = 2000 # 1500
total_time_quench = 10 # 15
q = 2*pi*(3/(N-1))
beta_th = 8.0/2

## file things:
dict_filename = "thermalstate_crit_beta3"
beta_plot = 2*beta_th

## Ising parameters:
J0 = 1.0
h0 = 1.0 # 2.0 # -0.525 # 1.0
g0 = 0.0 # 0.25 # 0.0

## quench parameters:
J(time) = J0
delta = 1.0
rate = "nan"
# h(time) = h0 + exp(-3(time-2)^2)                  # large Gaussian quench
h(time) = h0 + delta*exp(-100(time-0.35)^2)        # small transverse Gaussian quench
# h(time) = time < 0.1 ? h0 : h0+delta                # instantaneous quench
# h(time) = 0.1<=time<=0.6 ? h0+delta : h0                # quench bump
# h(time) = h0 + delta*(1 + tanh(5*(time-0.5)))/2  # continuous quench
# rate = 0.01
# h(time) = h0 + delta*time*exp(-rate*(time-0.5)^2) / (((sqrt(rate)+sqrt(8+rate))*exp(1/8*(-4-rate+sqrt(rate)*sqrt(8+rate))))/(4*sqrt(rate)))
g(time) = g0 #+ delta*exp(-20(time-0.5)^2)          # small longitudinal Gaussian quench

## operators to measure:
hamblocksTH(time) = MPS.isingHamBlocks(N,J0,h0,g0)
hamblocks(time) = MPS.isingHamBlocks(N,J(time),h(time),g(time))
opEmpo(time) = MPS.IsingMPO(N,J(time),h(time),g(time))
opE(time,G,L) = MPS.gl_mpoExp(G,L,opEmpo(time))
opmag_x(time,G,L) = MPS.localOpExp(G,L,sx,Int(floor(N/2)))
opmag_z(time,G,L) = MPS.localOpExp(G,L,sz,Int(floor(N/2)))
pert_mpo = MPS.translationMPO(N,sx)
opmag_x_tot(time,G,L) = MPS.gl_mpoExp(G,L,pert_mpo)
opnorm(time,G,L) = MPS.gl_mpoExp(G,L,MPS.IdentityMPO(N,2))
ops = [opE opmag_x opmag_z opmag_x_tot opnorm]

## initializations:
mpo = MPS.IdentityMPO(N,2)
# mps = mpo_to_mps(mpo)
# mps = MPS.randomMPS(N,2,5)
G,L = MPS.prepareGL(mpo,maxD)


function sth(N,beta,time,steps,D,c="nan")
    return string("L= ",N,"  beta= ",beta,"  t_max= ",time,"  steps= ",steps,"  D= ", D,"  prec= ",prec,"  delta= ",delta,"  [J0,h0,g0]= ",[J0,h0,g0],"  rate= ",c,"\n")
end

function sth2(N,beta,steps,D)
    return string("L= ",N,"  beta= ",beta,"  steps= ",steps,"  D= ", D,"  prec= ",prec,"\n")
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
# @time opvalsth, errth = MPS.gl_tebd(G,L,hamblocksTH,-beta_th*im,steps_th,maxD,[],tol=prec,increment=inc,st2=true)
## save:
# thermal_data = Dict(:g => G, :l => L, :info => "N=$N, maxD=$maxD, prec=$prec, steps_th=$steps_th, beta_th=$beta_plot, J0=$J0, h0=$h0, g0=$g0")
# BSON.bson(string(@__DIR__,"/data/quench/"*dict_filename*".bson"), thermal_data)
## load:
thermo = BSON.load(string(@__DIR__,"/data/quench/"*dict_filename*".bson"))
G = thermo[:g]
L = thermo[:l]
println(thermo[:info])
println("delta = ", delta)
# gA = copy(G)
# lA = copy(L)
# gB = copy(g)
# lB = copy(l)

# pert_ops = [expm(1e-3*sx*im*x) for x in sin.(q*(-1+(1:N)))]
# pert_ops = fill(expm(1e-3*im*sx),N)
# pert_ops = fill(Complex128.(si),N)
# pert_ops = fill(expm(1e-4*im*sx),N)
# pert_ops[Int(floor(N/2))] = expm(1e-2*im*sx)
# pert_ops[Int(floor(N/2))] = sx
# MPS.ops_on_gl(G,L,pert_ops)
# pert_ops[Int(floor(N/2))] = sx
# pert_ops = fill(expm(1e-3*im*sx),N)
# pert_mpo = MPS.translationMPO(N,sx)
# pertg = MPS.mpo_on_gl(gA,lA,pert_mpo)
# gA, lA = MPS.prepareGL(pertg,maxD,prec)

# MPS.ops_on_gl(gA,lA,pert_ops)
# pert_ops = fill(Complex128.(si),N)
# pert_ops[Int(floor(N/2))] = sx
# MPS.gl_ct!(gB)
# MPS.ops_on_gl(gA,lA,pert_ops)


## quench evolution:
@time opvals, err = MPS.gl_tebd(G,L,hamblocks,total_time_quench,steps,maxD,ops,tol=prec,increment=inc,st2=true)
# @time opvals2, errA,errB, times = MPS.gl_tebd_c(gA,lA,gB,lB,hamblocks,total_time_quench,steps,maxD,tol=prec,increment=inc,st2=true)


## SAVING:
opvals = real.(opvals)
err = real.(err)
save_data(cat(2,opvals,err), header=string(sth(N,2*beta_th,total_time_quench,steps,maxD,rate), "# t \t E \t mag_x \t mag_z \t mag_x_tot \t norm \t err\n"))
# open(string(@__DIR__,"/opvals3.txt"), "a") do f
#     write(f, "beta/2 = 2, A = global sx, B = local sx \r")
#     writedlm(f, [opvals])
#     writedlm(f, [err])
#     write(f,"\r\n")
# end
# open(string(@__DIR__,"/opvals3.txt"), "a") do f
#     write(f, "beta/2 = 2, A = global sx, B = local sx \r")
#     writedlm(f, [times opvals2])
#     writedlm(f, [errA errB])
#     write(f,"\r\n")
# end

# BSON.bson(string(@__DIR__,"/thermalstate.bson"), thermalstate=[G,L])
# BSON.parse(string(@__DIR__,"/thermalstate.bson"))
# thermo = BSON.load(string(@__DIR__,"/thermalstate.bson"))


## PLOTTING:
figure(1)
plot(opvals[:,1], opvals[:,2], label="\$\\beta_{th} = $beta_plot, D = $maxD\$") # E
figure(2)
plot(opvals[:,1], opvals[:,3]) # <s_x>
figure(3)
plot(opvals[:,1], opvals[:,4]) # <s_z>
figure(4)
plot(opvals[:,1], opvals[:,5]) # <s_x_tot>
figure(5)
plot(opvals[:,1], err) # error

subfolder = ""

figure(1)
xlabel("\$t\\, /\\, J \$")
ylabel("\$E(t)\$")
title("energy")
legend(loc = "best", numpoints=3, frameon = 0, fancybox = 0, columnspacing = 1)
savefig(string(@__DIR__,"/figures/"*subfolder*"/energy.pdf"))

figure(2)
xlabel("\$t\\, /\\, J \$")
ylabel("\$\\langle \\sigma_x(L/2) \\rangle\$")
title("transverse magnetization")
savefig(string(@__DIR__,"/figures/"*subfolder*"/magnetization_trans.pdf"))

figure(3)
xlabel("\$t\\, /\\, J \$")
ylabel("\$\\langle \\sigma_z(L/2) \\rangle\$")
title("longitudinal magnetization")
savefig(string(@__DIR__,"/figures/"*subfolder*"/magnetization_long.pdf"))

figure(4)
xlabel("\$t\\, /\\, J \$")
ylabel("\$\\langle \\sigma_x(L/2)_{total} \\rangle\$")
title("total transverse magnetization")
savefig(string(@__DIR__,"/figures/"*subfolder*"/magnetization_trans_tot.pdf"))

figure(5)
xlabel("\$t\\, /\\, J \$")
title("error")
savefig(string(@__DIR__,"/figures/"*subfolder*"/error.pdf"))













println("done: gl_para.jl")
show()
;
