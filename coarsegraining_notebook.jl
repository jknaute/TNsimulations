using TN
using LinearAlgebra
using JSON
# %% Small mera
w,u = TN.constr_wu_dim2()
S3 = TN.S3_binary(w,u)

eigenoperators = eigen(S3)

conformal_dims = -log2.(eigenoperators.values)

candidate_hamiltonians=eigenoperators.vectors[:,findall((x)->abs(x-2)<0.01,conformal_dims)]
eps=eigenoperators.vectors[:,findall((x)->abs(x-1)<0.01,conformal_dims)]
sig=eigenoperators.vectors[:,findall((x)->abs(x-1/8)<0.1,conformal_dims)]
id=eigenoperators.vectors[:,findall((x)->abs(x-0)<0.01,conformal_dims)]

function project(op,c,p)
    Nd = Int(log(2,length(op))/2)
    mat = reshape(op,2^Nd,2^Nd)
    mat = mat + mat'
    mat = mat + c*conj(mat)
    tens = reshape(mat,repeat([2],2*Nd)...)
    tens = tens + p*permutedims(tens,vcat(Nd:-1:1,2*Nd:-1:Nd+1))
    return vec(tens)/norm(tens)
end
eps2=project(eps,1,1);
sig2=project(sig,1,1);
h2 = project(candidate_hamiltonians[:,:]*rand(4),1,1);
dxeps2 = project(candidate_hamiltonians[:,:]*rand(4),1,-1);
dteps2 = project(candidate_hamiltonians[:,:]*rand(4),-1,1);
p2 = project(candidate_hamiltonians[:,:]*rand(4),-1,-1);

#@time vec(project(kron(candidate_hamiltonians[:,1],candidate_hamiltonians[:,1],candidate_hamiltonians[:,1]),1,1));

m2dict = Dict([("eps2",vec(eps2)), ("sig2",vec(sig2)), ("ham2",vec(h2)), ("dxeps2",vec(dxeps2)), ("dteps2",vec(dteps2)), ("mom2",vec(p2)) ])

# %% Large mera
w8,u8 = TN.constr_wu_dim8();
dims8, ops8 = TN.get_approximate_scaling_dims(w8,u8,nev=8);

#h_vec8 = vec(project(ops8[:,6],1,1));

candidate_hamiltonians8 = ops8[:,findall(x->abs(x-2)<0.01,dims8)];
eps8 = ops8[:,findall(x->abs(x-1)<0.01,dims8)];
sig8 = ops8[:,findall(x->abs(x-1/8)<0.1,dims8)];

eps_vec8=project(eps8,1,1);
sig_vec8=project(sig8,1,1);
h_vec8 = project(candidate_hamiltonians8[:,:]*rand(4),1,1);
dxeps_vec8 = project(candidate_hamiltonians8[:,:]*rand(4),1,-1);
dteps_vec8 = project(candidate_hamiltonians8[:,:]*rand(4),-1,1);
p_vec8 = project(candidate_hamiltonians8[:,:]*rand(4),-1,-1);


dict = Dict([("eps2",real(eps2)), ("sig2",real(sig2)), ("ham2",real(h2)), ("dxeps2",real(dxeps2)), ("dteps2",imag(dteps2)), ("mom2",imag(p2)), ("eps8",real(eps_vec8)), ("sig8",real(sig_vec8)), ("ham8",real(h_vec8)), ("dxeps8",real(dxeps_vec8)), ("dteps8",imag(dteps_vec8)), ("mom8",imag(p_vec8)) ])

# %% Save operators
filename = string(@__DIR__,"\\mera_operators.json")
open(filename,"w") do io
    JSON.print(io,dict)
end

# %% Energies
@time vals8 = TN.local_ham_eigs(reshape(-h_vec8,repeat([2],18)...),12,nev=12)[1];

@time vals8 = TN.local_ham_eigs(reshape(-h_vec8, repeat([8],6)...),3,nev=12)[1];

print(real.((vals8 .- vals8[1])/(vals8[2]-vals8[1])*1/8))

# %% Eigenvalues
hIsing = TN.isingHamGates(5,1,1,0)[2]
valsIsing = []
sIsing = []
sMera2 = []
sMera8 = []
valsMera2 = []
valsMera8 = []
valsMera8large = []
normalize_energies(es) = real.((es .- es[1])/(es[2]-es[1])*1/8)
function v_and_s(ham,n)
    vv = TN.local_ham_eigs(ham, n, nev=12);
    s = Array{Float64,1}(undef,n)
    for k in 1:8
        v = reshape(vv[2][:,1],2^k,2^(n-k))
        s[k] = sum(map(x->-x*log(abs.(x)), eigen(v*v').values))
    end
    return vv,s
end
for n in [9, 12]
    vv, s = v_and_s(hIsing,n)
    push!(sIsing, s)
    @time push!(valsIsing, normalize_energies(vv[1]))

    vv, s = v_and_s(h_tens,n)
    push!(sMera2, s)
    @time push!(valsMera2, normalize_energies(vv[1]))

    vv, s = v_and_s(reshape( -h_vec8, repeat([2],18)...),n)
    push!(sMera8, s)
    @time push!(valsMera8, normalize_energies(vv[1]))

end

# %% Plotting
exact_energies= [0,1/8,1,1+1/8,1+1/8,2,2,2,2,2+1/8,2+1/8,2+1/8]
 plot(valsIsing, seriestype = :scatter, marker=:x )
 plot!(valsMera2, seriestype = :scatter, marker=:square )
 plot!(valsMera8, seriestype = :scatter, marker=:circle)

# %% Error plot
datamap(vals) = map(x->abs.((x .- exact_energies) ./exact_energies)[3:end],vals)
xs= 3:12
colors = [:blue :red]

plot(xs,datamap(valsIsing), seriestype = :scatter, markercolors = colors, yaxis=:log,
marker=:x, label = ["9 Ising" "15 Ising"],legend = :bottomright)

plot!(xs,datamap(valsMera2), seriestype = :scatter, markercolors = colors, yaxis=:log,
marker=:square, label = ["9 Mera2" "15 Mera2"])

plot!(xs,datamap(valsMera8), seriestype = :scatter, markercolors = colors, yaxis=:log,
marker=:circle, label = ["9 Mera8" "15 Mera8"])
