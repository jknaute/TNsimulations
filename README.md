# TNsimulations
Tensor Network algorithms for MPO/MPS and some MERA
implemented for Julia 0.6.2
___________________________________________________


main algorithms:
- DMRG
- TEBD
- iTEBD

main implementation files:
- MPS.jl: implementation of MPS & MPO ansatz for Ising and Heisenberg spin chain models, manipulations, DMRG, MERA
- gl.jl: implementation of iMPS functions using "gamma-lambda" notation, iTEBD
- TEBD.jl: implementation of (finite-size) TEBD

projects:
- gl_para_corr.jl: calculation of thermal correlation functions in https://arxiv.org/abs/1912.08836
- gl_iTEBD2_renyiquench.jl: calculation of entanglement quantities after thermal quantum quenches in https://arxiv.org/abs/2206.10528
