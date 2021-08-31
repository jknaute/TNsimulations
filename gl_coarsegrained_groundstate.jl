### find the groundstate of the parent Hamiltonian RR'=exp[-W]
### through iTEBD2 (as imaginary time evolution)
### and calculate its entanglement entropy


module iTEBD2
include(string(@__DIR__,"/iTEBD2.jl"))
end

module MPS
using iTEBD2
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

include(string(@__DIR__,"/coarsegrained.jl")) # module cg

using iTEBD2
using MPS
using cg
using BSON
# using PyPlot
using TensorOperations
using Base.Threads
# Base.BLAS.set_num_threads(1)


## chain and evolution parameters:
N = 2
N_finite = 200
d = 2
maxD = [10] # collect(56:4:64)
prec=0 # 1e-15

beta_th = 10.0
steps_th = 4000
N_layers = 1000 # 20 ## no of RR applications
conv_prec = 0 # 1e-6 # relative convergence threshold/precision for imaginary time evolution break w.r.t. to first operator (energy); set to 0 to run full beta


## file things:
dict_filename = "linearresponse/continuum_limit_coarsegrained/groundstate/groundstate_data_parentHamiltonian_part3"
output_filename = "linearresponse/continuum_limit_coarsegrained/groundstate/xxx"


## thermal Ising parameters:
J0 = -1.0
h0 = -1.0
g0 = -0.0


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

err_vals = Array{Any}(N_layers,length(maxD))
s1_vals  = Array{Any}(N_layers,length(maxD))



# ##-------------------------------- find ground state projector |0><0| via iTEBD2 ----------------------
# ## MERA tensors
# w, u = cg.constr_wu_dim2()
#
# ## T ... translation MPO
# I = eye(d)
# @tensor Tp[-1,-2,-3,-4] := I[-2,-1]*I[-4,-3] # = T_+
# @tensor Tm[-1,-2,-3,-4] := I[-1,-3]*I[-2,-4] # = T_-
# Tp *= (1/sqrt(d)); Tm *= (1/sqrt(d)) # normalization for canonical form
# # l_init = eye(2)/sqrt(2)
# # @tensor Tpl[:] := Tp[-1,-2,-3,1]*l_init[1,-4] # = Tp*l_init =: M
#
# ## define RR'
# println("\n RR' : ")
# @tensor R[-1,-2,-3,-4,-5,-6,-7,-8] := w[-4,-1,1]*u[3,4,1,-6]*conj(u[5,6,7,-8])*Tp[-2,3,5,2]*Tp[2,4,6,-7]*conj(w[-5,-3,7])
# sR = size(R)
# R = reshape(R, sR[1]*sR[2]*sR[3],sR[4],sR[5],sR[6]*sR[7]*sR[8])
# @tensor RR[-1,-2,-3,-4,-5,-6] := R[-1,-3,1,-5]*conj(R[-2,-4,1,-6])
# sRR = size(RR)
# RR = reshape(RR, sRR[1]*sRR[2],sRR[3],sRR[4],sRR[5]*sRR[6])
#
# MPS.check_triple_canonical([RR], [eye(sRR[5]*sRR[6])])
# # RRa,RRb, la,lb = MPS.double_canonicalize_by_Identity_circuit(RR,RR, eye(sRR[5]*sRR[6]),eye(sRR[5]*sRR[6]), d, 200, tol=1e-15)
# # MPS.check_triple_canonical([RRa,RRb], [la,lb])
#
#
# ###-----------------------------------------------------------------------------
# ### Parent Hamiltonian:
# for i=1:length(maxD)
#     D_i = maxD[i]
#     println("\nDmax = ",D_i)
#
#     ## initializations:
#     Γ_A = Γ_B = reshape(diag(eye(d)), 1,d,1)
#     λ_A = λ_B = eye(1)/sqrt(2)
#
#     Γ_A,λ_A, Γ_B,λ_B, err_th, s1 = MPS.apply_2site_mpo_layers(Γ_A,λ_A, Γ_B,λ_B, RR,RR, d, D_i, N_layers, tol=prec) # MPS.apply_gate_layers(Γ,λ, RR, d, D, N_layers, tol=prec)
#
#     ## right-/left canonical checks: observed preference => A leftcan, B rightcan
#     MPS.check_gl_canonical([Γ_A,Γ_B], [λ_A,λ_B])
#
#     err_vals[:,i] = real(err_th)
#     s1_vals[:,i]  = real(s1)
# end
#
# ## Save Data:
# groundstate_data = Dict(:err => err_vals, :s1 => s1_vals, :D => maxD, :info => "N_layers=$N_layers, prec=$prec")
# BSON.bson(string(@__DIR__,"/data/"*dict_filename*".bson"), groundstate_data)
# ## load:
# groundstate_data = BSON.load(string(@__DIR__,"/data/"*dict_filename*".bson"))




###-----------------------------------------------------------------------------
## Hamiltonian
""" return Ising Hamiltonian on 2 sites """
function H_Ising(J,h,g)
    ham = J*ZZ + h*0.5*(XI+IX) + g*0.5*(ZI+IZ)
    return ham
end
hamblocksTH = H_Ising(J0,h0,g0)
H0 = reshape(hamblocksTH, d,d,d,d)


function constr_Euclideons(W2)
    ## W2 = exp[-dt H]

    # vertical svd of mid gate into W2 = E_l2*E_r2:
    W_mid2 = reshape(permutedims(W2, [1,3,2,4]), 2*2,2*2)
    U,S,V = svd(W_mid2)
    V = V'
    D = size(S)[1]
    E_l2 = reshape(U*diagm(sqrt.(S)), 2,2,D)
    E_r2 = reshape(diagm(sqrt.(S))*V, D,2,2)

    # horizontal svd of mid gate into W2 = E_u2*E_d2:
    W_mid2 = reshape(W2, 2*2,2*2)
    U,S,V = svd(W_mid2)
    V = V'
    D = size(S)[1]
    E_u2 = reshape(U*diagm(sqrt.(S)), 2,2,D)
    E_d2 = reshape(diagm(sqrt.(S))*V, D,2,2)

    # collection into one Euclideon:
    @tensor E2[-1,-2,-3,-4] := E_r2[-1,1,2]*E_d2[-2,1,3]*E_u2[2,4,-3]*E_l2[3,4,-4]
    @tensor A[:] := E_r2[-1,1,-3]*E_l2[-2,1,-4]
    @tensor B[:] := E_r2[-1,-2,1]*E_l2[1,-3,-4]

    return E2, A, B
end

###-----------------------------------------------------------------------------
### test with normal Hamiltonian as MPO:
W = reshape(expm(-(beta_th/N_layers)*hamblocksTH), (d,d,d,d))
E,A,B = constr_Euclideons(W)
Γ_A=λ_A=Γ_B=λ_B = []

for i=1:length(maxD)
    D_i = maxD[i]
    println("\nDmax = ",D_i)

    ## initializations:
    Γ_A = Γ_B = reshape(diag(eye(d))/sqrt(d), 1,d,1)
    λ_A = λ_B = eye(1)
    # Γ,λ = randn(D_i,d,D_i), randn(D_i)
    # Γ_A,λ_A, Γ_B,λ_B = iTEBD2.double_canonicalize(Γ, λ, Γ, λ)
    # λ_A,λ_B = diagm(λ_A),diagm(λ_B)

    MPS.check_gl_canonical([Γ_A,Γ_B], [λ_A,λ_B])
    Γ_A,λ_A, Γ_B,λ_B, err_th, s1 = MPS.apply_2site_mpo_layers(Γ_A,λ_A, Γ_B,λ_B, A,B, d, D_i, N_layers, tol=prec) # MPS.apply_gate_layers(Γ,λ, RR, d, D, N_layers, tol=prec)

    ## right-/left canonical checks: observed preference => A leftcan, B rightcan
    MPS.check_gl_canonical([Γ_A,Γ_B], [λ_A,λ_B])

    err_vals[:,i] = real(err_th)
    s1_vals[:,i]  = real(s1)
end

S_finite_entanglement_scaling = 1/(sqrt(24)+1)*log.(maxD) # cf. (10) in https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.102.255701
println("expected entropies: ",S_finite_entanglement_scaling)

@tensor M_A[:] := Γ_A[-1,-2,1]*λ_A[1,-3];
@tensor M_B[:] := Γ_B[-1,-2,1]*λ_B[1,-3];
sMA=size(M_A); M_A=reshape(M_A, sMA[1],sMA[2],1,sMA[3])
sMB=size(M_B); M_B=reshape(M_B, sMB[1],sMB[2],1,sMB[3])
M_A,M_B, λ_A,λ_B = MPS.double_canonicalize_by_Identity_circuit(M_A,M_B, λ_A,λ_B, d, maxD[1])
MPS.check_triple_canonical([M_A,M_B], [λ_A,λ_B])
E_th_0 = MPS.expect_operator_average([M_A,M_B], [λ_A,λ_B], H0)
E0 = iTEBD2.expect_twositelocal_dcan(Γ_A, diag(λ_A), Γ_B, diag(λ_B), H0)


###-----------------------------------------------------------------------------
### normal iTEBD Hamiltonian:
# for i=1:length(maxD)
#     D = maxD[i]
#     ## initializations:
#     mpo = MPS.IdentityMPO(N,d)
#     gTH,lTH = MPS.prepareGL(mpo,D)
#     MA=mpo[1]; MB=mpo[2]
#     lA=lTH[2]; lB=lTH[3]
#
#     # Γ = reshape(diag(eye(d)), 1,d,1)#randn(maxD, d, maxD)
#     # λ = diag(eye(1)) # randn(maxD)
#     # ΓA, λA, ΓB, λB = iTEBD2.double_canonicalize(Γ, λ, Γ, λ)
#     # lA=diagm(λA); lB=diagm(λB)
#     # @tensor MA[:] := ΓA[-1,-2,1]*lA[1,-3]
#     # @tensor MB[:] := ΓB[-1,-2,1]*lB[1,-3]
#     # MA = reshape(MA, 1,d,1,1); MB = reshape(MB, 1,d,1,1) # fake MPS
#
#     ## thermal state construction:
#     ## calculate:
#     MA,MB, lA,lB, err_th, betas, ops = MPS.gl_iTEBD2_timeevolution(MA,MB, lA,lB, hamblocksTH, -im*beta_th, steps_th, d, D, [], tol=prec, increment=steps_th, conv_thresh=conv_prec, counter_evo=false)
#     # MA,MB, lA,lB, err_th, betas, ops = MPS.gl_iTEBD2_timeevolution(MA,MB, lA,lB, hamblocksTH, -im*beta_th, steps_th, d, D, [H0], tol=prec, increment=1, conv_thresh=conv_prec, counter_evo=false)
#     ## save:
#     # thermal_data = Dict(:M => [MA,MB], :l => [lA,lB], :err => err_th, :info => "maxD=$maxD, prec=$prec, steps_th=$steps_th, beta_th=$beta_plot, conv_prec=$conv_prec, J0=$J0, h0=$h0, g0=$g0")
#     # BSON.bson(string(@__DIR__,"/data/"*dict_filename*".bson"), thermal_data)
#
#     ## load:
#     # thermo = BSON.load(string(@__DIR__,"/data/"*dict_filename*".bson"))
#     # ## initial thermal state sqrt{rho}:
#     # MA=thermo[:M][1]; MB=thermo[:M][2]
#     # lA=thermo[:l][1]; lB=thermo[:l][2]
#     # println(thermo[:info])
#
#     ## some data:
#     E_th_0 = MPS.expect_operator_average([MA,MB], [lA,lB], H0)
#     Tr_rho = MPS.trace_rho_average([MA,MB], [lA,lB])
#     println("\nE_th_0 = ",E_th_0)
#     println("Tr(rho) = ",Tr_rho)
#     MPS.check_triple_canonical([MA,MB], [lA,lB])
#
#     # ## SAVING:
#     # save_data(cat(2, 2*real(im*betas),real(errL),real(errR),real(renyi[:,1])), string(@__DIR__,"/data/"*output_filename*".txt"), header=string(sth(N,beta_plot,0,steps_th,maxD), "# beta \t errL \t errR \t s2\n"))
#
#
#     ##--------- calculate central charge c from entanglement scaling in finite chain -------------------
#     # ## fill AB unit cells:
#     # ground_proj = Array{Any}(N_finite)
#     # for i=1:2:N_finite
#     #     ground_proj[i] = MA
#     #     ground_proj[i+1] = MB
#     # end
#     # @tensor M1[:] := lB[-1,1]*MA[1,-2,-3,-4]
#     # ground_proj[1] = M1
#     #
#     # ## multiply with random MPS |0><0|r> = c*|0> and canonicalize it:
#     # mps_rand = fill(reshape([1 1],1,2,1), N_finite)
#     # mps_ground = Array{Any}(N_finite)
#     # for i=1:N_finite
#     #     mps_ground[i] = MPS.absorb_mpo_in_mps(mps_rand[i], ground_proj[i])
#     # end
#     #
#     # ## make OBC:
#     # mps_ground[1] = reshape(mps_ground[1][1,:,:], 1,d,size(mps_ground[1],3))
#     # mps_ground[end] = reshape(mps_ground[end][:,:,1], size(mps_ground[end],1),d,1)
#     # MPS.makeCanonical_old(mps_ground)
#     # Isingham = MPS.IsingMPO(N_finite,J0,h0,g0)
#     # E0 = MPS.mpoExpectation(mps_ground,Isingham)
#     # println("E0/(N-1) = ",E0/(N_finite-1))
#     #
#     # ## calculate entanglement entropies at every cut:
#     # entropy = Array{Any}(N_finite,2)
#     # for i = 1:N_finite
#     #     entropy[i,1] = i # subsystem size
#     #     entropy[i,2] = MPS.entropy(mps_ground,i-1)
#     # end
#     #
#     # figure(1)
#     # plot(entropy[:,1], entropy[:,2])
#     # figure(2)
#     # plot(log.(entropy[:,1]), entropy[:,2])
#
#
#     ##--------- calculate central charge c from entanglement scaling in iMPS bond dimension -------------------
#     ## multiply with random MPS |0><0|r> = c*|0> and canonicalize it:
#     @tensor ga[-1,-2,-3,-4] := MA[-1,-2,-3,1]*inv(lA)[1,-4]
#     @tensor gb[-1,-2,-3,-4] := MB[-1,-2,-3,1]*inv(lB)[1,-4]
#     # ga = reshape(ga, D,d,D); gb = reshape(gb, D,d,D)
#     # la = diag(lA); lb = diag(lB)
#
#     r = reshape([1 1],1,2,1)
#     ga = MPS.absorb_mpo_in_mps(r,ga)
#     gb = MPS.absorb_mpo_in_mps(r,gb)
#     ga, la, gb, lb = iTEBD2.double_canonicalize(ga, diag(lA), gb, diag(lB))
#
#     energy = iTEBD2.expect_twositelocal_dcan(ga, la, gb, lb, H0)
#     E0 = real(energy)
#     println("E0 = ",E0)
#     s1 = -dot(la.^2,log.(la.^2))
#     s1_vals[i] = s1
# end


###-----------------------------------------------------------------------------
### Plots:
# figure(1)
# plot(log.(maxD),s1_vals, ls="",marker="s")
# x = linspace(0.9*maxD[1],1.1*maxD[end],50)
# plot(log.(x),log.(x)/(1+sqrt(24)))






println("done: gl_coarsegrained_groundstate.jl")
# show()
;
