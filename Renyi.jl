using TensorOperations
using LinearMaps

# This function gives the transfer matrix for a single site which acts on the right.
function transfer_matrix_rho_squared(A)
    sA=size(A)
    function contract(R)
        temp = reshape(R,sA[4],sA[4],sA[4],sA[4])
        @tensoropt (r,-2,-3,-4) begin
            temp[:] := temp[r,-2,-3,-4]*conj(A[-1,-5,-6,r])
            temp[:] := temp[-1,r,-3,-4,c,-6]*A[-2,c,-5,r]
            temp[:] := temp[-1,-2,r,-4,c,-6]*conj(A[-3,-5,c,r])
            temp[:] := temp[-1,-2,-3,r,c,-6]*A[-4,c,-5,r]
            temp[:] := temp[-1,-2,-3,-4,c,c]
        end
        st = size(temp)
        return reshape(temp,st[1]*st[2]*st[3]*st[4])
    end
    T = LinearMap{Complex{Float64}}(contract,sA[1]^4,sA[4]^4)
    return T
end

#The transfer matrix which acts on the left. Singular values should have been absorbed on the left side of A. This is actually slightly more memory efficient than the left transfer matrix, as contracting the first index of A is optimal because that is how it's laid out in the memory.
function transfer_matrix_right_rho_squared(A)
    sA=size(A)
    function contract(R)
        temp = reshape(R,sA[4],sA[4],sA[4],sA[4])
        @tensoropt (r,-2,-3,-4) begin
            temp[:] := temp[r,-2,-3,-4]*conj(A[r,-5,-6,-1])
            temp[:] := temp[-1,r,-3,-4,c,-6]*A[r,c,-5,-2]
            temp[:] := temp[-1,-2,r,-4,c,-6]*conj(A[r,-5,c,-3])
            temp[:] := temp[-1,-2,-3,r,c,-6]*A[r,c,-5,-4]
            temp[:] := temp[-1,-2,-3,-4,c,c]
        end
        st = size(temp)
        return reshape(temp,st[1]*st[2]*st[3]*st[4])
    end
    T = LinearMap{Complex{Float64}}(contract,sA[1]^4,sA[4]^4)
    return T
end

# Test the function with bond dimension D and up to subsystem size n
function test_renyi_iMPS(D,n)
    A = rand(D,2,2,D) # The sqrt of the density matrix (with singular values already contracted from the right)
    transfer_matrix = transfer_matrix_rho_squared(A)
    rightVec = reshape(Matrix(1.0I,D^2,D^2),D^4)
    lambdasquared = kron(lambda,lambda)
    leftVec = transpose(rightVec)

    vals = []
    for k in 1:n
        rightVec = transfer_matrix*rightVec
        val = leftVec*rightVec
        push!(vals, val)
    end
    return vals
end

# Test the function with bond dimension D and up to subsystem size n
function test_renyi_finite(D,n)
    A = rand(D,2,2,D) # The sqrt of the density matrix (with singular values already contracted from the right)

    mps = [A for k in 1:n];
    transfer_matrices = transfer_matrix_rho_squared.(mps)
    Dr = size(mps[n],4)
    rightVec = reshape(Matrix(1.0I,Dr^2,Dr^2),Dr^4)

    vals = []
    for k in n:-1:1
        rightVec = transfer_matrices[k]*rightVec
        Dl = size(mps[k],1)
        leftVec = transpose(reshape(Matrix(1.0I,Dl^2,Dl^2),Dl^4))
        val = leftVec*rightVec
        push!(vals, val)
    end
    return vals
end
