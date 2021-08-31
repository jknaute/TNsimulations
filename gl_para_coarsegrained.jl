### coarsegrained Ising model on a finite chain with symmetrized 2-site gates of physical dimension = 4

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
const II = kron(eye(64), eye(64)) # kron(eye(4), eye(4)) # kron(eye(8), eye(8)) # kron(si,si)
const ii = kron(si,si)
include(string(@__DIR__,"/MPS.jl"))
include(string(@__DIR__,"/gl.jl"))
end

include(string(@__DIR__,"/coarsegrained.jl"))
using MPS
using cg
using BSON
using JSON
# using Optim   # somehow not compatible with PyPlot
# using PyPlot
using Base.Threads
# Base.BLAS.set_num_threads(1)


## chain and evolution parameters:
# N_bare = 60
N = 4 # Int(N_bare/2) # coarsegrained system size for binary MERA tensors
d = 64
maxD=50
prec=1e-15 # 1e-20

beta_th = 8.0/2
steps_th = 200

total_time_quench = 20
steps = 1000
inc=1


## file things:
dict_MERA_dim8 = "linearresponse/continuum_limit_coarsegrained/binary/MERA_ops_dim8"
dict_filename = "linearresponse/continuum_limit_coarsegrained/binary/thermalstate_crit_dim8_beta8_size4"
output_filename = "linearresponse/continuum_limit_coarsegrained/binary/responsecritdim8beta8size4"
beta_plot = 2*beta_th



## Ising parameters:
J0 = 43.7801 # 1.0/(0.134 + 3.33/N_bare^2.58) # rescaling at criticality
h0 = 0.0    # transverse perturbation w/ epsilon
g0 = 0.0    # longitudinal perturbation w/ sigma

##--------------------------  analytic constructions from MERA  ------------------
## MERA tensors:
# w,u = cg.constr_wu_dim8()

dict_MERA_ops = Dict()
open(string(@__DIR__,"/mera_operators.json"),"r") do io
    global dict_MERA_ops
    dict_MERA_ops = JSON.parse(io)
end

# function project(op,c,p)
#     Nd = Int(log(2,length(op))/2)
#     mat = reshape(op,2^Nd,2^Nd)
#     mat = mat + mat'
#     mat = mat + c*conj(mat)
#     tens = reshape(mat,repeat([2],2*Nd)...)
#     tens = tens + p*permutedims(tens,vcat(Nd:-1:1,2*Nd:-1:Nd+1))
#     return vec(tens)/vecnorm(tens)
# end

## 3-site operators for binary MERA:
# dims, ops = cg.get_approximate_scaling_dims(w,u,8)
#
# inds_Delta1 = find((x)->x≈1.0,dims)
# eps = ops[:,inds_Delta1]
eps_proj = dict_MERA_ops["eps8"] # project(eps,1,1)
eps_proj = convert(Array{Complex{Float64},1},eps_proj)
eps_symm  = 2*cg.symmetrize_vector(eps_proj) # = eps*I+I*eps
println("eps8")

# inds_Delta18 = find((x)->abs(x-1/8)<0.1,dims)
# sigma = ops[:,inds_Delta18]
sigma_proj = dict_MERA_ops["sig8"] # project(sigma,1,1)
sigma_proj = convert(Array{Complex{Float64},1},sigma_proj)
sigma_symm  = 2*cg.symmetrize_vector(sigma_proj)
println("sig8")

# inds_Delta2 = find((x)->x≈2.0,dims)
# ham = ops[:,inds_Delta2]
ham_proj = dict_MERA_ops["ham8"] # project(ham[:,:]*rand(length(inds_Delta2)),1,1);
ham_proj = convert(Array{Complex{Float64},1},ham_proj)
ham_symm  = 2*cg.symmetrize_vector(ham_proj)
println("ham8")

# ## save:
# MERA_dim8_data = Dict(:eps_symm => eps_symm, :sigma_symm => sigma_symm, :ham_symm => ham_symm)
# BSON.bson(string(@__DIR__,"/data/"*dict_MERA_dim8*".bson"), MERA_dim8_data)
# ## load:
# MERA_dim8_ops = BSON.load(string(@__DIR__,"/data/"*dict_MERA_dim8*".bson"))
# eps_symm = MERA_dim8_ops[:eps_symm]
# sigma_symm = MERA_dim8_ops[:sigma_symm]
# ham_symm = MERA_dim8_ops[:ham_symm]



##------------------------  scaling operators of given scaling dimension  ------------------
## energy density eps with Delta=1:
# eps,Deltas = cg.get_scaling_ops(S3,1.0)
# eps_symm  = reshape(cg.epsilon, d,d,d,d) # cg.symmetrize_vector(eps[1])
# eps_symm  = permutedims(eps_symm, [1,2,4,3])
# eps=reshape(eps[1], 8,8)

# ## Delta=2 ops and its hermitian eigenvectors k1...k4:
# ham = cg.get_scaling_ops(S3,2.0)[1]
# k1 = 0.5(reshape(ham[1], 8,8)+reshape(ham[1],8,8)')
# k2 = 0.5(reshape(ham[2], 8,8)+reshape(ham[2],8,8)')
# k3 = 0.5(reshape(ham[3], 8,8)+reshape(ham[3],8,8)')
# k4 = 0.5(reshape(ham[4], 8,8)+reshape(ham[4],8,8)')


# ##--------------------  coarsegraining / renormalizing the Hamiltonian  ------------------
# h0 = reshape(cg.constr_h0(), 8,8)
# projectorNorm(x) = Float64(trace( (h0-x[1]*k1-x[2]*k2-x[3]*k3-x[4]*k4) * (h0-x[1]*k1-x[2]*k2-x[3]*k3-x[4]*k4)' ))
# x0 = [1.0,1.0,1.0,1.0]
# min_h = optimize(projectorNorm,x0,LBFGS(),Optim.Options(g_tol=1e-12))
# params_h,minval_h = Optim.minimizer(min_h),Optim.minimum(min_h)
# hs = params_h[1]*k1 + params_h[2]*k2 + params_h[3]*k3 + params_h[4]*k4
#
#
# ##-----------------  identification of H and P from commutation relations  -----------------
# ##   dt(eps)=i[H,eps]
# ##   dx(eps)=i[P,eps]
# ## here for densities h & p
#
#
# ##----- 3-site minimization
# ## find parameters such that d(eps)=i[X,eps] holds for:
# ## d(eps) = k1+x[1]*k2+x[2]*k3+x[3]*k4
# ## X = x[4]*k1+x[5]*k2+x[6]*k3+x[7]*k4
# ## d can be dt or dx; X can be h or p
# ## norm = Tr(M*M') for M=d(eps)-i[X,eps]:
# Mnorm(x) = Float64(trace( (k1+x[1]*k2+x[2]*k3+x[3]*k4 - im*((x[4]*k1+x[5]*k2+x[6]*k3+x[7]*k4)*eps-eps*(x[4]*k1+x[5]*k2+x[6]*k3+x[7]*k4))) *
#                           (k1+x[1]*k2+x[2]*k3+x[3]*k4 - im*((x[4]*k1+x[5]*k2+x[6]*k3+x[7]*k4)*eps-eps*(x[4]*k1+x[5]*k2+x[6]*k3+x[7]*k4)))' ))
# x0 = [1.0,1.0,1.0,1.0,1.0,1.0,1.0]
# min3 = optimize(Mnorm,x0,Optim.Options(g_tol=1e-12))
# params3=Optim.minimizer(min3)
# minval3 = Optim.minimum(min3)
#
#
# ##----- 4-site minimization
# ## first order derivative approximation:
# ## norm = Tr(M*M') for M=dx(eps)-i[p,eps] with p & eps symmetrized:
# eps_deriv = reshape(cg.derivative_vector(reshape(eps,64)), 16,16) # = dx(eps) = I*eps-eps*I
# eps_symm  = reshape(cg.symmetrize_vector(reshape(eps,64)), 16,16)
# p_symm(x) = reshape(cg.symmetrize_vector(reshape(x[1]*k1+x[2]*k2+x[3]*k3+x[4]*k4,64)), 16,16)
# diffNorm(x) = Float64(trace( (eps_deriv - im*(p_symm(x)*eps_symm-eps_symm*p_symm(x)))*
#                              (eps_deriv - im*(p_symm(x)*eps_symm-eps_symm*p_symm(x)))' ))
# x0 = [1.0,1.0,1.0,1.0]
# min4 = optimize(diffNorm,x0) # min4 = optimize(diffNorm,x0,Optim.Options(g_tol=1e-12))
# params4=Optim.minimizer(min4)
# minval4 = Optim.minimum(min4)
#
#
# ##----- 7-site minimization: P=sum{p} over all leg contributions
# ## RHS: commutator [P,eps]:
# comm_Peps(x) = cg.commutator_vectors(reshape(x[1]*k1+x[2]*k2+x[3]*k3+x[4]*k4,64),reshape(eps,64))
# x0 = [1.0,1.0,1.0,1.0]
#
# ##--- 1st order derivative:
# ## LHS: dx(eps) :
# eps_diff1 = cg.finitediff_order1(reshape(eps,64))
# ## norm = Tr(M*M') for M=dx(eps)-i[p,eps] :
# finitediff1Norm(x) = Float64(trace( (eps_diff1-im*comm_Peps(x)) * (eps_diff1-im*comm_Peps(x))' ))
# minO1 = optimize(finitediff1Norm,x0,LBFGS(),Optim.Options(g_tol=1e-12))
# paramsO1,minvalO1 = Optim.minimizer(minO1),Optim.minimum(minO1)
#
# ##--- 2nd order derivative:
# ## LHS: dx(eps) :
# eps_diff2 = cg.finitediff_order2(reshape(eps,64))
# ## norm = Tr(M*M') for M=dx(eps)-i[P,eps] :
# finitediff2Norm(x) = Float64(trace( (eps_diff2-im*comm_Peps(x)) * (eps_diff2-im*comm_Peps(x))' ))
# minO2 = optimize(finitediff2Norm,x0,LBFGS(),Optim.Options(g_tol=1e-12))
# paramsO2,minvalO2 = Optim.minimizer(minO2),Optim.minimum(minO2)
#
# ##--- check accuracy condition:
# println("2nd order:")
# println("sqrt{Tr[(lhs-rhs)^2]}           = ",sqrt(minvalO2))
# lhs = eps_diff2
# rhs = im*comm_Peps(paramsO2)
# println("sqrt{Tr[lhs^2]}+sqrt{Tr[rhs^2]} = ", sqrt(trace(lhs*lhs'))+sqrt(trace(rhs*rhs')) )
#
#
# ##--- check if momentum p results are in commuting subspace spanned by e1,e2:
# # p for fit params:
# p_O1 = paramsO1[1]*k1 + paramsO1[2]*k2 + paramsO1[3]*k3 + paramsO1[4]*k4
# p_O2 = paramsO2[1]*k1 + paramsO2[2]*k2 + paramsO2[3]*k3 + paramsO2[4]*k4
# x0 = [1.0,1.0]
# basisNorm(x) = Float64(trace( (p-x[1]*e1-x[2]*e2) * (p-x[1]*e1-x[2]*e2)' ))     # commuting subspace
# nonbasisNorm(x) = Float64(trace( (p-x[1]*e3-x[2]*e4) * (p-x[1]*e3-x[2]*e4)' ))  # remaining basis vectors
# p = p_O1
# subminO1 = optimize(basisNorm,x0,LBFGS(),Optim.Options(g_tol=1e-12))
# nonsubminO1 = optimize(nonbasisNorm,x0,Optim.Options(g_tol=1e-12))
# p = p_O2
# subminO2 = optimize(basisNorm,x0,Optim.Options(g_tol=1e-12))
# nonsubminO2 = optimize(nonbasisNorm,x0,LBFGS(),Optim.Options(g_tol=1e-12))
#
#
# ##--- identify h by conservation eq:  -dx(h) = i[H,p]
# h_diff2(x) = cg.finitediff_order2(reshape(x[1]*k1+x[2]*k2+x[3]*k3+x[4]*k4,64))
# comm_Hp(x) = cg.commutator_vectors(reshape(x[1]*k1+x[2]*k2+x[3]*k3+x[4]*k4,64),reshape(p_O2,64))
# x0 = [1.0,1.0,1.0,1.0]
# conservationNorm(x) = Float64(trace( (-h_diff2(x)-im*comm_Hp(x)) * (-h_diff2(x)-im*comm_Hp(x))' ))
# min_conservation = optimize(conservationNorm,x0,LBFGS(),Optim.Options(g_tol=1e-12)) # PROBLEM: finds 0 as a trivial solution
#
#
# # ##--- identify h+p by:  [p,X]=0  for X = x1*k1+...+x4*k4 ~ h+p
# # comm_pX(x) = cg.sum_vectors(reshape(x[1]*k1+x[2]*k2+x[3]*k3+x[4]*k4,64),reshape(p_O2,64)) - cg.sum_vectors(reshape(p_O2,64),reshape(x[1]*k1+x[2]*k2+x[3]*k3+x[4]*k4,64))
# # commNorm(x) = Float64(trace( comm_pX(x) * comm_pX(x)' ))
# # min_comm = optimize(commNorm,x0,Optim.Options(g_tol=1e-12))





##------------------------------------  quench  ----------------------------------
## Hamiltonian:
function Hcg_binary_hamBlocks(L,J,h,g)
    ## J is the rescaled normalization for the coarsegrained system
    hs = (J*reshape(ham_symm,d^2,d^2) + h*reshape(eps_symm,d^2,d^2) + g*reshape(sigma_symm,d^2,d^2))
    blocks = Array{Array{Complex128,2},1}(L)
    for i=1:L
        blocks[i] = hs
    end
    return blocks
end
hamblocksTH(time) = Hcg_binary_hamBlocks(N,J0,h0,g0) # cg.Hs_binary_hamBlocks(N,J0,h0,g0) # cg.H0_hamBlocks(N)


## initializations:
mpo = MPS.IdentityMPO(N,d) # MPS.IdentityMPO(N,8) # MPS.IdentityMPO(N,2)
gTH,lTH = MPS.prepareGL(mpo,maxD)


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
println("thermal state:")
## calculate:
@time opvalsth, errth = MPS.gl_tebd(gTH,lTH,hamblocksTH,-beta_th*im,steps_th,maxD,[],tol=prec,st2=true)
## save:
thermal_data = Dict(:g => gTH, :l => lTH, :err => errth, :info => "N=$N, maxD=$maxD, prec=$prec, steps_th=$steps_th, beta_th=$beta_plot, J0=$J0, h0=$h0, g0=$g0")
BSON.bson(string(@__DIR__,"/data/"*dict_filename*".bson"), thermal_data)
## load:
thermo = BSON.load(string(@__DIR__,"/data/"*dict_filename*".bson"))
gTH = thermo[:g] # initial thermal state sqrt{rho}
lTH = thermo[:l]
gTH = convert(Array{Array{Complex{Float64},4},1}, gTH)
lTH = convert(Array{Array{Complex{Float64},2},1}, lTH)
println(thermo[:info])

# # gTH = copy(G)
# # lTH = copy(L)
# gP = deepcopy(gTH) # perturbed state
# lP = deepcopy(lTH)
#
#
# ## global perturbation:
# # pert_mpo_l0 = MPS.qMPO(N,sx,0) # level 0 = bare
# # #pert_mpo_l0 = MPS.qMPO(N,sz,0) # longitudinal
# # pert_mpo_l1 = cg.coarsegrained_MPO(pert_mpo_l0[Int(floor(N/2))],N,w1,u1)
# # pert_mpo_l2 = cg.coarsegrained_MPO(pert_mpo_l1[Int(floor(N/2))],N,w2,u2)
# # pert_mpo_ls = cg.coarsegrained_MPO(pert_mpo_l2[Int(floor(N/2))],N,ws,us)
#
# ## choose coarsegrained level assuming p=0:
# # pert_mpo = pert_mpo_l0
# # pertg = MPS.mpo_on_gl(gA,lA,pert_mpo)
# # gA, lA = MPS.prepareGL(pertg,maxD,prec)
#
#
# ## local perturbation:
# # pert_ops_l0 = fill(Complex128.(si),N)
# # pert_ops_l0[Int(floor(N/2))] = sx
# # #pert_ops_l0[Int(floor(N/2))] = sz # longitudinal
# # pert_ops_l1 = pert_ops_l2 = pert_ops_ls = fill(Complex128.(eye(8)),N) # fill(Complex128.(cg.coarsegrained_op(si,w1)),N)
# # pert_ops_l1[Int(floor(N/2))] = cg.coarsegrained_op(sx,w1)
# # pert_ops_l2[Int(floor(N/2))] = cg.coarsegrained_op(pert_ops_l1[Int(floor(N/2))],w2)
# # pert_ops_ls[Int(floor(N/2))] = cg.coarsegrained_op(pert_ops_l2[Int(floor(N/2))],ws)
#
# ## choose coarsegrained level:
# # pert_ops = pert_ops_l0
# # MPS.ops_on_gl(gB,lB,pert_ops)
# ## analytical tensors:
# gP,lP = MPS.twositeop_on_gl(gP,lP,eps_symm,Int(floor(length(gP)/2)),maxD,prec) # local pert w/ eps on mid chain
#
# ## quench evolution:
# ## one-site version:
# # @time opvals2, errA,errB, times = MPS.gl_tebd_c(gA,lA,gB,lB,hamblocksTH,total_time_quench,steps,maxD,tol=prec,increment=inc,st2=true)
# ## two-site coarsegrained adjustments:
# @time opvals2, errP, times = MPS.gl_tebd_cg(gTH,lTH,gP,lP,hamblocksTH,eps_symm,total_time_quench,steps,maxD,tol=prec,increment=inc,st2=true)
#
#
# ## SAVING:
# save_data(cat(2,real(times),real(opvals2),imag(opvals2),real(errP)), string(@__DIR__,"/data/"*output_filename*".txt"), header=string(sth(N,2*beta_th,total_time_quench,steps,maxD), "# t \t ReG \t ImG \t errP\n"))
# # finalstate_data = Dict(:gA => gA, :lA => lA, :gB => gB, :lB => lB, :times => times, :errA => errA, :errB => errB, :info => "N=$N, maxD=$maxD, prec=$prec, steps=$steps, beta_th=$beta_plot, tmax/2=$total_time_quench, J0=$J0, h0=$h0, g0=$g0")
# # BSON.bson(string(@__DIR__,"/data/"*output_filename*".bson"), finalstate_data)
#
#
#







println("done: gl_para_coarsegrained.jl")
# show()
;
