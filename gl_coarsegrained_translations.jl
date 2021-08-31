### coarsegrained operators on an infinite translational invariant 2-site unit cell:
### based on critical Ising model wavelet tensors


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
using PyPlot
using TensorOperations
using Base.Threads
Base.BLAS.set_num_threads(1)


## chain and evolution parameters:
N = 40 # number of coarsegraining layers here!
num_singvals = 8
d = 2
maxD=100
prec=1e-15


## file things:
dict_filename = "linearresponse/continuum_limit_coarsegrained/translations/xxx"
output_filename = "linearresponse/continuum_limit_coarsegrained/translations/xxx"





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

## MERA tensors
w, u = cg.constr_wu_dim2()

## T ... translation MPO
I = eye(d)
@tensor Tp[-1,-2,-3,-4] := I[-2,-1]*I[-4,-3] # = T_+
@tensor Tm[-1,-2,-3,-4] := I[-1,-3]*I[-2,-4] # = T_-
# Tp *= (1/sqrt(d)); Tm *= (1/sqrt(d)) # normalization for canonical form
l_init = eye(2)/sqrt(2)
@tensor Tpl[:] := Tp[-1,-2,-3,1]*l_init[1,-4] # = Tp*l_init =: M

# coarsegrain T
Ga,La,Ma, Gb,Lb,Mb, lambdas = MPS.gl_coarsegraining_mpo(Tp,l_init,Tpl, Tp,l_init,Tpl, w,u, N, d, maxD, tol=prec, do_normalization=true, num_singvals=num_singvals)
MPS.check_triple_canonical([Ma,Mb], [La,Lb])



# # define RR'
# println("\n RR' : ")
# @tensor R[-1,-2,-3,-4,-5,-6,-7,-8] := w[-4,-1,1]*u[3,4,1,-6]*conj(u[5,6,7,-8])*Tp[-2,3,5,2]*Tp[2,4,6,-7]*conj(w[-5,-3,7])
# sR = size(R)
# R = reshape(R, sR[1]*sR[2]*sR[3],sR[4],sR[5],sR[6]*sR[7]*sR[8])
# @tensor RR[-1,-2,-3,-4,-5,-6] := R[-1,-3,1,-5]*conj(R[-2,-4,1,-6])
# sRR = size(RR)
# RR = reshape(RR, sRR[1]*sRR[2],sRR[3],sRR[4],sRR[5]*sRR[6])
#
# # canonicalize it initially
# # MPS.check_triple_canonical([RR], [eye(sRR[5]*sRR[6])])
# # RRa,RRb, la,lb = MPS.double_canonicalize_and_normalize(RR,RR, eye(sRR[5]*sRR[6]),eye(sRR[5]*sRR[6]), d, do_normalization=true)
# # MPS.check_triple_canonical([RRa,RRb], [la,lb])
#
# MPS.check_triple_canonical([RR], [eye(sRR[5]*sRR[6])])
# RRa,RRb, la,lb = MPS.double_canonicalize_by_Identity_circuit(RR,RR, eye(sRR[5]*sRR[6]),eye(sRR[5]*sRR[6]), d, maxD, tol=prec)
# MPS.check_triple_canonical([RRa,RRb], [la,lb])
#
#
#
# ## coarsegrain RR':
# @tensor ga[:] := RRa[-1,-2,-3,1]*inv(la)[1,-4]
# @tensor gb[:] := RRb[-1,-2,-3,1]*inv(lb)[1,-4]
# Ga,La,Ma, Gb,Lb,Mb, lambdas = MPS.gl_coarsegraining_mpo(ga,la,RRa, gb,lb,RRb, w,u, N, d, maxD, tol=prec, do_normalization=true, num_singvals=num_singvals)
# MPS.check_triple_canonical([Ma,Mb], [La,Lb])



############################################   SCALING   #######################
## scaling under coarsegraining:
data = log2.(lambdas)
figure(1)
for i=1:num_singvals
    plot(data[i,:])
end


## finite difference derivatives:
dlambdas = copy(lambdas)
for i=1:num_singvals
    for j=2:N-1
        dlambdas[i,j] = (data[i,j+1]-data[i,j-1])/2
    end
    dlambdas[i,1]   = data[i,2]-data[i,1]
    dlambdas[i,end] = data[i,end]-data[i,end-1]
end
figure(2)
for i=1:num_singvals
    plot(dlambdas[i,:])
end





## SAVING:
# save_data(cat(2, real(time),real(err_t),real(renyi[:,1])), string(@__DIR__,"/data/"*output_filename*".txt"), header=string(sth(N,2*beta_th,total_time_quench,steps,maxD), "# t \t err \t s2\n"))


println("done: gl_coarsegrained_translations.jl")
# show()
;
