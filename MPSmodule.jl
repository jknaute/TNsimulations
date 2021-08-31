module MPS

export sx,sy,sz,si,s0,ZZ,ZI,IZ,XI,IX,II

# define Pauli matrices
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
const II = kron(si, si)


include("MPS.jl")
include("gl.jl")
end
