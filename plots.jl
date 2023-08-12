include("layout.jl")
using layout # run this module at first as:  include("layout.jl")
using PyPlot

betavals = [0.01, 0.5, 8.0]
elw = 3
cs = 5

function energy_layout(tmax=10)
    xlim(0,tmax)
    xlabel("\$t \\cdot J\$")
    ylabel("\$E(t)\\, / \\,L\$")
    title("\$energy\$")
    legend(loc = "center right", numpoints=3, frameon = 0, fancybox = 0, columnspacing = 1)
    layout.nice_ticks()
end

function energyscaled_layout(tmax=10)
    xlim(0,tmax)
    xlabel("\$t \\cdot J\$")
    ylabel("\$[E(t)-E(t=0)]\\, / \\,L\\, /\\,\\delta\$")
    title("\$energy\$")
    legend(loc = "center right", numpoints=3, frameon = 0, fancybox = 0, columnspacing = 1)
    layout.nice_ticks()
end

function magnetization_trans_layout(tmax=10)
    xlim(0,tmax)
    xlabel("\$t \\cdot J\$")
    ylabel("\$\\langle \\sigma_x(L/2) \\rangle\$")
    title("\$transverse\\, magnetization\$")
    layout.nice_ticks()
end

function magnetization_transscaled_layout(tmax=10)
    xlim(0,tmax)
    xlabel("\$t \\cdot J\$")
    ylabel("\$ [\\langle\\sigma_x^{L/2}\\rangle(t) - \\langle\\sigma_x^{L/2}\\rangle(t=0)]/\\delta \$")
    title("\$transverse\\, magnetization\$")
    layout.nice_ticks()
end

function magnetization_long_layout(tmax=10)
    xlim(0,tmax)
    xlabel("\$t \\cdot J\$")
    ylabel("\$\\langle \\sigma_z(L/2) \\rangle\$")
    title("\$longitudinal\\, magnetization\$")
    layout.nice_ticks()
end

function magnetization_trans_tot_layout(tmax=10)
    xlim(0,tmax)
    xlabel("\$t \\cdot J\$")
    ylabel("\$\\langle \\sigma_x(L/2) \\rangle_{tot}\$")
    title("\$total\\, transverse\\, magnetization\$")
    layout.nice_ticks()
end

function corr_fct_layout(tmax=10)
    xlim(0,tmax)
    xlabel("\$t \\cdot J\$")
    ylabel("\$\\langle \\sigma_z(L/4) \\, \\sigma_z(3/4 L) \\rangle\$")
    title("\$correlation\\, function\$")
    layout.nice_ticks()
end

function error_layout(tmax=10)
    xlim(0,tmax)
    xlabel("\$t \\cdot J\$")
    title("\$error\$")
    layout.nice_ticks()
end

function read_and_plot_Tdependence(fig_num)
    f = open(string(@__DIR__,"/data/quench/"*subfolder*"/opvals.txt"))
    lines = readlines(f)
    close(f)
    sep_inds = findin(lines, [""])

    for i = 1:length(sep_inds)
        counter = 1

        if i==1
            steps = sep_inds[i]-3
            t = Array{Float64}(steps)
            E = Array{Float64}(steps)
            mag_x = Array{Float64}(steps)
            mag_z = Array{Float64}(steps)
            mag_x_tot = Array{Float64}(steps)
            norm = Array{Float64}(steps)
            error = Array{Float64}(steps)
            beta = include_string(split(lines[1])[4])
            L = include_string(split(lines[1])[2])
            delta = include_string(split(lines[1])[14])
            for l = 3 : sep_inds[1]-1
                line = parse.(split(lines[l]))
                t[counter] = line[1]
                E[counter] = line[2]
                mag_x[counter] = line[3]
                mag_z[counter,:] = line[4]
                mag_x_tot[counter] = line[5]
                norm[counter] = line[6]
                error[counter] = line[7]
                counter += 1
            end
        else
            steps = sep_inds[i]-sep_inds[i-1]-3
            t = Array{Float64}(steps)
            E = Array{Float64}(steps)
            mag_x = Array{Float64}(steps)
            mag_z = Array{Float64}(steps)
            mag_x_tot = Array{Float64}(steps)
            norm = Array{Float64}(steps)
            error = Array{Float64}(steps)
            beta = include_string(split(lines[sep_inds[i-1]+1])[4])
            L = include_string(split(lines[sep_inds[i-1]+1])[2])
            delta = include_string(split(lines[sep_inds[i-1]+1])[14])
            for l = sep_inds[i-1]+3 : sep_inds[i]-1
                line = parse.(split(lines[l]))
                t[counter] = line[1]
                E[counter] = line[2]
                mag_x[counter] = line[3]
                mag_z[counter,:] = line[4]
                mag_x_tot[counter] = line[5]
                norm[counter] = line[6]
                error[counter] = line[7]
                counter += 1
            end
        end

        figure(fig_num)
        plot(t, E/(L-1), label="\$\\beta = $beta\\, \\delta = $delta\$")
        figure(fig_num+1)
        plot(t, mag_x)
        figure(fig_num+2)
        plot(t, mag_z)
        figure(fig_num+3)
        plot(t, mag_x_tot)
        figure(fig_num+4)
        plot(t, error)
    end
end

function read_and_plot_quenchstudies(fig_num,tmax_b1,tmax_b2,tmax_b3,lstyle="-"; plot_leg=0)
    f = open(string(@__DIR__,"/data/quench/"*subfolder*"/opvals.txt"))
    lines = readlines(f)
    close(f)
    sep_inds = findin(lines, [""])
    minvals = Array{Float64}(3)

    for i = 1:length(sep_inds)
        counter = 1

        if i==1
            steps = sep_inds[i]-3
            t = Array{Float64}(steps)
            E = Array{Float64}(steps)
            mag_x = Array{Float64}(steps)
            mag_z = Array{Float64}(steps)
            mag_x_tot = Array{Float64}(steps)
            norm = Array{Float64}(steps)
            error = Array{Float64}(steps)
            beta = include_string(split(lines[1])[4])
            L = include_string(split(lines[1])[2])
            delta = include_string(split(lines[1])[14])
            for l = 3 : sep_inds[1]-1
                line = parse.(split(lines[l]))
                t[counter] = line[1]
                E[counter] = line[2]
                mag_x[counter] = line[3]
                mag_z[counter,:] = line[4]
                mag_x_tot[counter] = line[5]
                norm[counter] = line[6]
                error[counter] = line[7]
                counter += 1
            end
        else
            steps = sep_inds[i]-sep_inds[i-1]-3
            t = Array{Float64}(steps)
            E = Array{Float64}(steps)
            mag_x = Array{Float64}(steps)
            mag_z = Array{Float64}(steps)
            mag_x_tot = Array{Float64}(steps)
            norm = Array{Float64}(steps)
            error = Array{Float64}(steps)
            beta = include_string(split(lines[sep_inds[i-1]+1])[4])
            L = include_string(split(lines[sep_inds[i-1]+1])[2])
            delta = include_string(split(lines[sep_inds[i-1]+1])[14])
            for l = sep_inds[i-1]+3 : sep_inds[i]-1
                line = parse.(split(lines[l]))
                t[counter] = line[1]
                E[counter] = line[2]
                mag_x[counter] = line[3]
                mag_z[counter,:] = line[4]
                mag_x_tot[counter] = line[5]
                norm[counter] = line[6]
                error[counter] = line[7]
                counter += 1
            end
        end

        # colors and plot params:
        if beta==0.01 fig_no=fig_num elseif beta==0.5 fig_no=fig_num+6 elseif beta==8.0 fig_no=fig_num+12 end
        if beta==0.01 t_max=tmax_b1 elseif beta==0.5 t_max=tmax_b2 elseif beta==8.0 t_max=tmax_b3 end
        ind_max = minimum(find(t .>= t_max))
        if delta==0.01
            col="C0"
            if beta==0.01 minvals[1]=minimum((mag_x[1:ind_max]-mag_x[1])/delta) elseif beta==0.5 minvals[2]=minimum((mag_x[1:ind_max]-mag_x[1])/delta) elseif beta==8.0 minvals[3]=minimum((mag_x[1:ind_max]-mag_x[1])/delta) end
        elseif delta==0.1 col="C1" elseif delta==1.0 col="C2" end

        figure(fig_no)
        if plot_leg==0
            plot(t[1:ind_max], (E[1:ind_max]-E[1])/delta/(L-1), c=col, ls=lstyle)
        elseif plot_leg==1
            plot(t[1:ind_max], (E[1:ind_max]-E[1])/delta/(L-1), label="\$\\delta = $delta\$", c=col, ls=lstyle)
        elseif plot_leg==2
            plot(t[1:ind_max], (E[1:ind_max]-E[1])/delta/(L-1), label="\$\\beta = $beta\\, \\delta = $delta\$", c=col, ls=lstyle)
        end
        figure(fig_no+1)
        plot(t[1:ind_max], (mag_x[1:ind_max]-mag_x[1])/delta, c=col, ls=lstyle)
        figure(fig_no+2)
        plot(t[1:ind_max], mag_x[1:ind_max], c=col, ls=lstyle)
        figure(fig_no+3)
        plot(t[1:ind_max], mag_z[1:ind_max], c=col, ls=lstyle)
        figure(fig_no+4)
        plot(t[1:ind_max], mag_x_tot[1:ind_max], c=col, ls=lstyle)
        figure(fig_no+5)
        plot(t, error, c=col, ls=lstyle)
    end
    return minvals
end

##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


###-----------------------------------------------------------------------------
### instantaneous quench at criticality:

subfolder = "thermal/instantaneous_quench"
read_and_plot_Tdependence(1)

figure(1)
energy_layout()
savefig(string(@__DIR__,"/figures/"*subfolder*"/energy.pdf"))

figure(2)
magnetization_trans_layout()
savefig(string(@__DIR__,"/figures/"*subfolder*"/magnetization_trans.pdf"))

figure(3)
magnetization_long_layout()
savefig(string(@__DIR__,"/figures/"*subfolder*"/magnetization_long.pdf"))

figure(4)
magnetization_trans_tot_layout()
savefig(string(@__DIR__,"/figures/"*subfolder*"/magnetization_trans_tot.pdf"))

figure(5)
error_layout()
savefig(string(@__DIR__,"/figures/"*subfolder*"/error.pdf"))



###-----------------------------------------------------------------------------
### quench bump at criticality:

subfolder = "thermal/quench_bump"
read_and_plot_Tdependence(6)

figure(6)
energy_layout()
savefig(string(@__DIR__,"/figures/"*subfolder*"/energy.pdf"))

figure(7)
magnetization_trans_layout()
savefig(string(@__DIR__,"/figures/"*subfolder*"/magnetization_trans.pdf"))

figure(8)
magnetization_long_layout()
savefig(string(@__DIR__,"/figures/"*subfolder*"/magnetization_long.pdf"))

figure(9)
magnetization_trans_tot_layout()
savefig(string(@__DIR__,"/figures/"*subfolder*"/magnetization_trans_tot.pdf"))

figure(10)
error_layout()
savefig(string(@__DIR__,"/figures/"*subfolder*"/error.pdf"))



### ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
### linear response functions
## quench profiles:
function h_instantaneous(time)
    h_out=Array{Float64}(length(time))
    for i=1:length(time)
        if time[i]<0.1 h_out[i]=0.0 else h_out[i]=1.0 end
    end
    return h_out
end
h_continuous(time) = (1 + tanh.(5*(time-0.5)))/2
function h_bump(time)
    h_out=Array{Float64}(length(time))
    for i=1:length(time)
        if 0.1<=time[i]<=0.6 h_out[i]=1.0 else h_out[i]=0.0 end
    end
    return h_out
end
h_Gauss(time) = exp.(-100(time-0.35).^2)

function get_G_and_signals(param_type, figno=2, normalizeit=false)
    f = open(string(@__DIR__,"/data/linearresponse/quenchstudies/response_"*param_type*".txt"))
    lines = readlines(f)
    close(f)
    sep_inds = findin(lines, [""])
    sig_beta1 = sig_beta2 = sig_beta3 = []

    for i = 1:length(sep_inds)
        counter = 1

        if i==1
            steps = sep_inds[i]-3
            t = Array{Float64}(steps)
            ReG = Array{Float64}(steps)
            ImG = Array{Float64}(steps)
            header = lines[1]
            beta = include_string(split(lines[1])[4])
            for l = 3 : sep_inds[1]-1
                line = parse.(split(lines[l]))
                t[counter] = line[1]
                ReG[counter] = line[2]
                ImG[counter] = line[3]
                counter += 1
            end
        else
            steps = sep_inds[i]-sep_inds[i-1]-3
            t = Array{Float64}(steps)
            ReG = Array{Float64}(steps)
            ImG = Array{Float64}(steps)
            header = lines[sep_inds[i-1]+1]
            beta = include_string(split(lines[sep_inds[i-1]+1])[4])
            for l = sep_inds[i-1]+3 : sep_inds[i]-1
                line = parse.(split(lines[l]))
                t[counter] = line[1]
                ReG[counter] = line[2]
                ImG[counter] = line[3] # imag(eval.(parse.(join([split(lines[l])[5],split(lines[l])[6]]))))
                counter += 1
            end
        end
        if ImG[2]<0 ImG=-ImG end # equal sign normalization

        if normalizeit
            if beta==0.01 col="b";zo=-3 elseif beta==0.5 col="g";zo=-2 elseif beta==8.0 col="r";zo=-1 end
            figure(figno)
            plot(t, ImG/minimum(-ImG), label="\$\\beta J = $beta\$", c=col, zorder=zo)
            legend(loc = "best", numpoints=3, frameon = 0, fancybox = 0, columnspacing = 1)
        else
            figure(1)
            plot(t, ReG)

            figure(figno)
            if beta==0.01 col="b";zo=-3 elseif beta==0.5 col="g";zo=-2 elseif beta==8.0 col="r";zo=-1 end
            plot(t, -ImG, label="\$\\beta J = $beta\$", c=col, zorder=zo)

            xlim(0,10)
            xlabel("\$t \\cdot J\$")
            ylabel("\$G(t)\$")
            ax = subplot(111)
            ax[:ticklabel_format](axis="y", style="scientific", scilimits=(-2,4), useOffset=true)
            if param_type=="crit" legend(loc = "best", numpoints=3, frameon = 0, fancybox = 0, columnspacing = 1) end
            layout.nice_ticks()
            savefig(string(@__DIR__,"/figures/thermal/quenchstudies/"*param_type*"/response_"*param_type*".pdf"))

            ## convolution:
            sig_insta = conv(-ImG,h_instantaneous(t))[1:length(t)]
            sig_cont = conv(-ImG,h_continuous(t))[1:length(t)]
            sig_bump = conv(-ImG,h_bump(t))[1:length(t)]
            sig_Gauss = conv(-ImG,h_Gauss(t))[1:length(t)]

            if beta==0.01
                sig_beta1=[t,sig_insta,sig_cont,sig_bump,sig_Gauss]
            elseif beta==0.5
                sig_beta2=[t,sig_insta,sig_cont,sig_bump,sig_Gauss]
            elseif beta==8.0
                sig_beta3=[t,sig_insta,sig_cont,sig_bump,sig_Gauss]
            end

            figure(3)
            plot(t, sig_insta)
            plot(t, sig_cont)
            plot(t, sig_bump)
            plot(t, sig_Gauss)

            ## write into separate files:
            open(string(@__DIR__,"/data/linearresponse/quenchstudies/G"*param_type*"beta",findin(betavals,beta)[1],".txt"), "w") do f
                write(f, string("# ",header,"\n"))
                write(f, "t \t ImG\n")
                writedlm(f, cat(2,t,-ImG))
                write(f,"\r\n")
            end
        end
    end

    return sig_beta1, sig_beta2, sig_beta3
end

## plot response functions:
magx_crit = get_G_and_signals("crit")
figure(2)
clf()
magx_nonint1 = get_G_and_signals("nonint1")
figure(2)
clf()
magx_nonint2 = get_G_and_signals("nonint2")

## plot quench profiles:
figure(4)
t = linspace(0,2,1000)
plot(t, h_instantaneous(t), c="b")
plot(t, h_continuous(t), ls="-.", c="b")
plot(t, h_bump(t), c="g",ls="-.")
plot(t, h_Gauss(t), ls=":", c="g")
ax = subplot(111)
ax[:set_xticks]([0,0.5,1,1.5,2])
ax[:set_yticks]([0,0.5,1])
xlim(0,t[end])
xlabel("\$t \\cdot J\$")
ylabel("\$[h(t) - h_0]/\\delta\$")
layout.nice_ticks()
savefig(string(@__DIR__,"/figures/thermal/quenchstudies/quenchprofiles.pdf"))

## plot analytical vs MPS result:
function plot_analyt_response(file)
    tG = readdlm(string(@__DIR__,file))
    figure(5)
    plot(tG[:,1], -tG[:,2]/minimum(tG[:,2]), ls=":", c="k")
end

figure(5)
get_G_and_signals("crit",5,true)
plot_analyt_response("/data/linearresponse/quenchstudies/Gcritbeta1_analyt.txt")
plot_analyt_response("/data/linearresponse/quenchstudies/Gcritbeta2_analyt.txt")
plot_analyt_response("/data/linearresponse/quenchstudies/Gcritbeta3_analyt.txt")
xlim(0,10)
xlabel("\$t \\cdot J\$")
ylabel("\$G_R(t) \\, / \\, G_R(t_{min})\$")
layout.nice_ticks()
savefig(string(@__DIR__,"/data/linearresponse/quenchstudies/figures/analyt_vs_MPS.pdf"))







### +++++++++++++++++++++++++++++++++++++++++++++++++++++  quenchstudies  ++++++++++++++++++++++++++++++++++++++++++++++++

###-----------------------------------------------------------------------------
### crit insta-cont:
figs = [11,17,23]
tmax = [4.5, 4.5, 10.0]

subfolder = "thermal/crit_instantaneous"
min_vals_insta = read_and_plot_quenchstudies(11,tmax[1],tmax[2],tmax[3], plot_leg=1)
subfolder = "thermal/crit_continuous"
min_vals_cont = read_and_plot_quenchstudies(11,tmax[1],tmax[2],tmax[3],"--")

for i=1:length(figs) ## E(t)
    betaplot = betavals[i]
    figure(figs[i])
    energyscaled_layout(tmax[i])
    legend(loc = "center right", numpoints=3, frameon=0, fancybox=0, columnspacing=1, title="\$ \\beta=$betaplot \$")
    if i==1
        ax = subplot(111)
        ax[:ticklabel_format](axis="y", style="scientific", scilimits=(-2,4), useOffset=true)
    end
    savefig(string(@__DIR__,"/figures/thermal/quenchstudies/crit/E_crit_instacont_beta",i,".pdf"))
end

for i=1:length(figs) ## s_x/delta
    figure(figs[i]+1)
    magnetization_transscaled_layout(tmax[i])
    if i==1 || i==2
        ax = subplot(111)
        ax[:ticklabel_format](axis="y", style="scientific", scilimits=(-2,4), useOffset=true)
    end
    plot(magx_crit[i][1], magx_crit[i][2]/(minimum(magx_crit[i][2])/min_vals_insta[i]),c="k",ls="-",zorder=-3,lw=4)
    plot(magx_crit[i][1], magx_crit[i][3]/(minimum(magx_crit[i][3])/min_vals_cont[i]), c="k",ls="-.",zorder=-3,lw=3,label="\$linear\\, response\$")
    legend(loc = "best", numpoints=3, frameon=0, fancybox=0, columnspacing=1)
    savefig(string(@__DIR__,"/figures/thermal/quenchstudies/crit/magtrans_crit_instacont_beta",i,".pdf"))
end

for i=1:length(figs) ## s_x(t)
    figure(figs[i]+2)
    magnetization_trans_layout(tmax[i])
end

for i=1:length(figs)
    figure(figs[i]+3)
    magnetization_long_layout(tmax[i])
    savefig(string(@__DIR__,"/figures/thermal/quenchstudies/crit/maglong_crit_instacont_beta",i,".pdf"))
end

for i=1:length(figs)
    figure(figs[i]+4)
    magnetization_trans_tot_layout(tmax[i])
end

for i=1:length(figs)
    figure(figs[i]+5)
    error_layout()
end


### crit bump-Gauss:
figs = [11,17,23]
tmax = [4.5, 4.5, 5.0]

subfolder = "thermal/crit_bump"
min_vals_bump = read_and_plot_quenchstudies(11,tmax[1],tmax[2],tmax[3], plot_leg=1)
subfolder = "thermal/crit_Gaussian"
min_vals_Gauss = read_and_plot_quenchstudies(11,tmax[1],tmax[2],tmax[3],"--")

for i=1:length(figs)
    betaplot = betavals[i]
    figure(figs[i])
    energyscaled_layout(tmax[i])
    legend(loc = "center right", numpoints=3, frameon=0, fancybox=0, columnspacing=1, title="\$ \\beta=$betaplot \$")
    if i==1
        ax = subplot(111)
        ax[:ticklabel_format](axis="y", style="scientific", scilimits=(-2,4), useOffset=true)
    end
    savefig(string(@__DIR__,"/figures/thermal/quenchstudies/crit/E_crit_bumpGauss_beta",i,".pdf"))
end

for i=1:length(figs)
    figure(figs[i]+1)
    magnetization_transscaled_layout(tmax[i])
    if i==1
        ax = subplot(111)
        ax[:ticklabel_format](axis="y", style="scientific", scilimits=(-2,4), useOffset=true)
    end
    plot(magx_crit[i][1], magx_crit[i][4]/(minimum(magx_crit[i][4])/min_vals_bump[i]),c="k",ls="-",zorder=-3,lw=4)
    plot(magx_crit[i][1], magx_crit[i][5]/(minimum(magx_crit[i][5])/min_vals_Gauss[i]), c="k",ls="-.",zorder=-3,lw=3,label="\$linear\\, response\$")
    legend(loc = "lower right", numpoints=3, frameon=0, fancybox=0, columnspacing=1)
    savefig(string(@__DIR__,"/figures/thermal/quenchstudies/crit/magtrans_crit_bumpGauss_beta",i,".pdf"))
end

for i=1:length(figs)
    figure(figs[i]+2)
    magnetization_trans_layout(tmax[i])
end

for i=1:length(figs)
    figure(figs[i]+3)
    magnetization_long_layout(tmax[i])
    savefig(string(@__DIR__,"/figures/thermal/quenchstudies/crit/maglong_crit_bumpGauss_beta",i,".pdf"))
end

for i=1:length(figs)
    figure(figs[i]+4)
    magnetization_trans_tot_layout(tmax[i])
end

for i=1:length(figs)
    figure(figs[i]+5)
    error_layout()
end




###-----------------------------------------------------------------------------
### nonint1 insta-cont:
figs = [11,17,23]
tmax = [7.0, 8.0, 10.0]

subfolder = "thermal/nonint1_instantaneous"
min_vals_insta = read_and_plot_quenchstudies(11,tmax[1],tmax[2],tmax[3], plot_leg=1)
subfolder = "thermal/nonint1_continuous"
min_vals_cont = read_and_plot_quenchstudies(11,tmax[1],tmax[2],tmax[3],"--")

for i=1:length(figs)
    betaplot = betavals[i]
    figure(figs[i])
    xlim(0,tmax[i])
    layout.nice_ticks()
    # if i==1
    #     ax = subplot(111)
    #     ax[:ticklabel_format](axis="y", style="scientific", scilimits=(-2,4), useOffset=true)
    # end
    savefig(string(@__DIR__,"/figures/thermal/quenchstudies/nonint1/E_nonint1_instacont_beta",i,".pdf"))
end

for i=1:length(figs)
    figure(figs[i]+1)
    xlim(0,tmax[i])
    layout.nice_ticks()
    # if i==1 || i==2
    #     ax = subplot(111)
    #     ax[:ticklabel_format](axis="y", style="scientific", scilimits=(-2,4), useOffset=true)
    # end
    plot(magx_nonint1[i][1], magx_nonint1[i][2]/(minimum(magx_nonint1[i][2])/min_vals_insta[i]),c="k",ls="-",zorder=-3,lw=4)
    plot(magx_nonint1[i][1], magx_nonint1[i][3]/(minimum(magx_nonint1[i][3])/min_vals_cont[i]), c="k",ls="-.",zorder=-3,lw=3,label="\$linear\\, response\$")
    savefig(string(@__DIR__,"/figures/thermal/quenchstudies/nonint1/magtrans_nonint1_instacont_beta",i,".pdf"))
end

for i=1:length(figs)
    figure(figs[i]+2)
    magnetization_trans_layout(tmax[i])
end

for i=1:length(figs)
    figure(figs[i]+3)
    magnetization_long_layout(tmax[i])
    savefig(string(@__DIR__,"/figures/thermal/quenchstudies/nonint1/maglong_nonint1_instacont_beta",i,".pdf"))
end

for i=1:length(figs)
    figure(figs[i]+4)
    magnetization_trans_tot_layout(tmax[i])
end

for i=1:length(figs)
    figure(figs[i]+5)
    error_layout()
end


### nonint1 bump-Gauss:
figs = [11,17,23]
tmax = [7.0, 8.0, 10.0]

subfolder = "thermal/nonint1_bump"
min_vals_bump = read_and_plot_quenchstudies(11,tmax[1],tmax[2],tmax[3], plot_leg=1)
subfolder = "thermal/nonint1_Gaussian"
min_vals_Gauss = read_and_plot_quenchstudies(11,tmax[1],tmax[2],tmax[3],"--")

for i=1:length(figs)
    betaplot = betavals[i]
    figure(figs[i])
    xlim(0,tmax[i])
    layout.nice_ticks()
    # if i==1
    #     ax = subplot(111)
    #     ax[:ticklabel_format](axis="y", style="scientific", scilimits=(-2,4), useOffset=true)
    # end
    savefig(string(@__DIR__,"/figures/thermal/quenchstudies/nonint1/E_nonint1_bumpGauss_beta",i,".pdf"))
end

for i=1:length(figs)
    figure(figs[i]+1)
    xlim(0,tmax[i])
    layout.nice_ticks()
    # if i==1
    #     ax = subplot(111)
    #     ax[:ticklabel_format](axis="y", style="scientific", scilimits=(-2,4), useOffset=true)
    # end
    plot(magx_nonint1[i][1], magx_nonint1[i][4]/(minimum(magx_nonint1[i][4])/min_vals_bump[i]),c="k",ls="-",zorder=-3,lw=4)
    plot(magx_nonint1[i][1], magx_nonint1[i][5]/(minimum(magx_nonint1[i][5])/min_vals_Gauss[i]), c="k",ls="-.",zorder=-3,lw=3,label="\$linear\\, response\$")
    savefig(string(@__DIR__,"/figures/thermal/quenchstudies/nonint1/magtrans_nonint1_bumpGauss_beta",i,".pdf"))
end

for i=1:length(figs)
    figure(figs[i]+2)
    magnetization_trans_layout(tmax[i])
end

for i=1:length(figs)
    figure(figs[i]+3)
    magnetization_long_layout(tmax[i])
    savefig(string(@__DIR__,"/figures/thermal/quenchstudies/nonint1/maglong_nonint1_bumpGauss_beta",i,".pdf"))
end

for i=1:length(figs)
    figure(figs[i]+4)
    magnetization_trans_tot_layout(tmax[i])
end

for i=1:length(figs)
    figure(figs[i]+5)
    error_layout()
end



###-----------------------------------------------------------------------------
### nonint2 insta-cont:
figs = [11,17,23]
tmax = [4.0, 4.5, 10.0]

subfolder = "thermal/nonint2_instantaneous"
min_vals_insta = read_and_plot_quenchstudies(11,tmax[1],tmax[2],tmax[3], plot_leg=1)
subfolder = "thermal/nonint2_continuous"
min_vals_cont = read_and_plot_quenchstudies(11,tmax[1],tmax[2],tmax[3],"--")

for i=1:length(figs)
    betaplot = betavals[i]
    figure(figs[i])
    xlim(0,tmax[i])
    layout.nice_ticks()
    # if i==1
    #     ax = subplot(111)
    #     ax[:ticklabel_format](axis="y", style="scientific", scilimits=(-2,4), useOffset=true)
    # end
    savefig(string(@__DIR__,"/figures/thermal/quenchstudies/nonint2/E_nonint2_instacont_beta",i,".pdf"))
end

for i=1:length(figs)
    figure(figs[i]+1)
    xlim(0,tmax[i])
    layout.nice_ticks()
    # if i==1 || i==2
    #     ax = subplot(111)
    #     ax[:ticklabel_format](axis="y", style="scientific", scilimits=(-2,4), useOffset=true)
    # end
    plot(magx_nonint2[i][1], magx_nonint2[i][2]/(minimum(magx_nonint2[i][2])/min_vals_insta[i]),c="k",ls="-",zorder=-3,lw=4)
    plot(magx_nonint2[i][1], magx_nonint2[i][3]/(minimum(magx_nonint2[i][3][1:minimum(find(magx_nonint2[i][1].>=tmax[i]))])/min_vals_cont[i]), c="k",ls="-.",zorder=-3,lw=3,label="\$linear\\, response\$")
    savefig(string(@__DIR__,"/figures/thermal/quenchstudies/nonint2/magtrans_nonint2_instacont_beta",i,".pdf"))
end

for i=1:length(figs)
    figure(figs[i]+2)
    magnetization_trans_layout(tmax[i])
end

for i=1:length(figs)
    figure(figs[i]+3)
    magnetization_long_layout(tmax[i])
    savefig(string(@__DIR__,"/figures/thermal/quenchstudies/nonint2/maglong_nonint2_instacont_beta",i,".pdf"))
end

for i=1:length(figs)
    figure(figs[i]+4)
    magnetization_trans_tot_layout(tmax[i])
end

for i=1:length(figs)
    figure(figs[i]+5)
    error_layout()
end


### nonint2 bump-Gauss:
figs = [11,17,23]
tmax = [4.0, 4.5, 10.0]

subfolder = "thermal/nonint2_bump"
min_vals_bump = read_and_plot_quenchstudies(11,tmax[1],tmax[2],tmax[3], plot_leg=1)
subfolder = "thermal/nonint2_Gaussian"
min_vals_Gauss = read_and_plot_quenchstudies(11,tmax[1],tmax[2],tmax[3],"--")

for i=1:length(figs)
    betaplot = betavals[i]
    figure(figs[i])
    xlim(0,tmax[i])
    layout.nice_ticks()
    # if i==1
    #     ax = subplot(111)
    #     ax[:ticklabel_format](axis="y", style="scientific", scilimits=(-2,4), useOffset=true)
    # end
    savefig(string(@__DIR__,"/figures/thermal/quenchstudies/nonint2/E_nonint2_bumpGauss_beta",i,".pdf"))
end

for i=1:length(figs)
    figure(figs[i]+1)
    xlim(0,tmax[i])
    layout.nice_ticks()
    # if i==1
    #     ax = subplot(111)
    #     ax[:ticklabel_format](axis="y", style="scientific", scilimits=(-2,4), useOffset=true)
    # end
    plot(magx_nonint2[i][1], magx_nonint2[i][4]/(minimum(magx_nonint2[i][4])/min_vals_bump[i]),c="k",ls="-",zorder=-3,lw=4)
    plot(magx_nonint2[i][1], magx_nonint2[i][5]/(minimum(magx_nonint2[i][5])/min_vals_Gauss[i]), c="k",ls="-.",zorder=-3,lw=3,label="\$linear\\, response\$")
    savefig(string(@__DIR__,"/figures/thermal/quenchstudies/nonint2/magtrans_nonint2_bumpGauss_beta",i,".pdf"))
end

for i=1:length(figs)
    figure(figs[i]+2)
    magnetization_trans_layout(tmax[i])
end

for i=1:length(figs)
    figure(figs[i]+3)
    magnetization_long_layout(tmax[i])
    savefig(string(@__DIR__,"/figures/thermal/quenchstudies/nonint2/maglong_nonint2_bumpGauss_beta",i,".pdf"))
end

for i=1:length(figs)
    figure(figs[i]+4)
    magnetization_trans_tot_layout(tmax[i])
end

for i=1:length(figs)
    figure(figs[i]+5)
    error_layout()
end







### +++++++++++++++++++++++++++++++++++++++++++++++++++++  continuum limit  ++++++++++++++++++++++++++++++++++++++++++++++++

###-----------------------------------------------------------------------------
### integrable case h=g=0 at criticality:

beta_series = [2.0, 4.0, 8.0, 12.0, 16.0, 32.0]
mps_factor = 2*100^1.5/((2*pi)^3)*8

##--- residues:
r1_mid_analyt = [-1.09178,   -0.513658,   -0.252301,    -0.167337, -0.125643,   -0.0631746]
r1_err_analyt = [0.0024654,  0.000233572, 0.0000989486, 0.0027714, 0.000168189, 0.00126018]
r1_mid_MPS = [-0.0173433,  -0.0079513, -0.00387489, -0.00256288,   -0.00195542, -0.00109166]
r1_err_MPS = [0.000048316,  4.44264e-6, 1.51001e-6,  0.0000573406,  7.65183e-6,  0.0000219158]

figure(1)
errorbar(beta_series, beta_series.*r1_mid_analyt, yerr=beta_series.*r1_err_analyt, ls="", marker="s", c="b", ms=0,mew=3,elinewidth=elw,capsize=cs,zorder=1, label="free fermions")
ebMPS = errorbar(beta_series, mps_factor*beta_series.*r1_mid_MPS, yerr=mps_factor*beta_series.*r1_err_MPS, ls="", marker="s", c="g", ms=0,mew=3,elinewidth=elw,capsize=cs,zorder=1, label="MPO (scaled)")
ebMPS[end][1][:set_linestyle]("-.")
axhline(-2, ls="--", c="grey",zorder=-1)

ylim(-2.6,-1.92)
xlabel("\$ \\beta \\cdot J \$")
ylabel("\$r_1\\cdot \\beta / 2\\pi\$")
legend(loc = "lower center", numpoints=3, frameon=0, fancybox=0, columnspacing=1, title="\$ \\beta\\cdot M_h = \\beta\\cdot M_g = 0.0 \$")
layout.nice_ticks()
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_crit/figures/residues_analyt_vs_MPS_crit.pdf"))

figure(2)
plot(beta_series,r1_mid_analyt./r1_mid_MPS, ls="", marker="s")

###-----------------------------------------------------------------------------
### integrable ferro case g=0:

beta_series = [2.0, 4.0, 8.0, 12.0, 16.0, 32.0]
M_series = 2*0.1./beta_series # M*beta/2 = 0.1 = const here

##--- analytical results:
mode1_mid = [1.00088   , 1.00121    , 1.00085   , 1.01126  ,  1.04047  , 1.03849]
mode1_err = [0.00977542, 0.000685282, 0.00127872, 0.00782611, 0.0732301, 0.0508261]

mode2_mid = [NaN, NaN, 3.12043,   3.36046 , 3.64673 , 3.74263]
mode2_err = [NaN, NaN, 0.0899117, 0.223482, 0.463418, 0.357266]

figure(1)
errorbar(beta_series, mode1_mid, yerr=mode1_err, ls="", marker="s", c="b", ms=0,mew=3,elinewidth=elw,capsize=cs,zorder=1, label="free fermions")
errorbar(beta_series, mode2_mid, yerr=mode2_err, ls="", marker="s", c="b", ms=0,mew=3,elinewidth=elw,capsize=cs,zorder=1)
# xlim(0,0.105)
ylim(0,4.5)
ax = subplot(111)
ax[:set_yticks]([0,1,2,3,4])

axhline(1, ls="--", c="grey",zorder=-1)
axhline(3, ls="--", c="grey",zorder=-1)

xlabel("\$ \\beta \\cdot J \$")
ylabel("\$-\\frac{\\beta}{2\\pi}\\, \\operatorname{Im}(\\omega)\$")
legend(loc = "best", numpoints=3, frameon=0, fancybox=0, columnspacing=1, title="\$ \\beta \\cdot M_h = 0.2, \\,\\beta\\cdot M_g = 0.0 \$")
layout.nice_ticks()
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_g0/figures/wTD_mass_dependence_analyt.pdf"))


##--- MPS results:
mode1_mid = [1.00331  ,  1.00032   , 1.00095   , 1.01742  , 1.04062  , 1.13919]
mode1_err = [0.00785346, 0.00230426, 0.00114561, 0.0350506, 0.0751278, 0.176555]

mode2_mid = [NaN, NaN, 3.14907 , 3.34312 , 3.82096 , NaN]
mode2_err = [NaN, NaN, 0.170424, 0.0894716, 0.586168, NaN]

figure(2)
errorbar(beta_series, mode1_mid, yerr=mode1_err, ls="", marker="s", c="g", ms=0,mew=3,elinewidth=elw,capsize=cs,zorder=1, label="MPO")
errorbar(beta_series, mode2_mid, yerr=mode2_err, ls="", marker="s", c="g", ms=0,mew=3,elinewidth=elw,capsize=cs,zorder=1)
# xlim(0,0.105)
ylim(0,4.5)
ax = subplot(111)
ax[:set_yticks]([0,1,2,3,4])

axhline(1, ls="--", c="grey",zorder=-1)
axhline(3, ls="--", c="grey",zorder=-1)

xlabel("\$ \\beta \\cdot J \$")
ylabel("\$-\\frac{\\beta}{2\\pi}\\, \\operatorname{Im}(\\omega)\$")
legend(loc = "lower left", numpoints=3, frameon=0, fancybox=0, columnspacing=1)
layout.nice_ticks()
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_g0/figures/wTD_mass_dependence_MPS.pdf"))


##--- residues:
mps_factor = 2*100^1.5/((2*pi)^3)*8
r1_mid_analyt = [-1.18634,   -0.535039,    -0.258054,    -0.17231,   -0.130521,   -0.0663562]
r1_err_analyt = [ 0.00518984, 0.000134059,  0.000542524,  0.00179033, 0.0012182, 0.00267279]
r1_mid_MPS = [-0.0197979, -0.00853947,  -0.00404075, -0.00266644,  -0.00202661, -0.00113292]
r1_err_MPS = [0.000149223, 0.0000312253, 7.68654e-6,  0.0000134254, 6.89548e-6, 0.0000264101]

figure(3)
errorbar(beta_series, beta_series.*r1_mid_analyt, yerr=beta_series.*r1_err_analyt, ls="", marker="s", c="b", ms=0,mew=3,elinewidth=elw,capsize=cs,zorder=1, label="free fermions")
ebMPS = errorbar(beta_series, mps_factor*beta_series.*r1_mid_MPS, yerr=mps_factor*beta_series.*r1_err_MPS, ls="", marker="s", c="g", ms=0,mew=3,elinewidth=elw,capsize=cs,zorder=1, label="MPO (scaled)")
ebMPS[end][1][:set_linestyle]("-.")
axhline(-2.10851, ls="--", c="grey",zorder=-1)

ylim(-2.6,-1.92)
xlabel("\$ \\beta \\cdot J \$")
ylabel("\$r_1\\cdot \\beta / 2\\pi\$")
legend(loc = "lower center", numpoints=3, frameon=0, fancybox=0, columnspacing=1, title="\$ \\beta \\cdot M_h = 0.2, \\,\\beta\\cdot M_g = 0.0 \$")
layout.nice_ticks()
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_g0/figures/residues_analyt_vs_MPS_intferro.pdf"))

figure(4)
plot(beta_series,r1_mid_analyt./r1_mid_MPS, ls="", marker="s")


###-----------------------------------------------------------------------------
### nonintegrable ferro:

betaseries = [6.0, 8.0, 10.0]
c2 = [0.05, 0.1, 0.2, 0.3]
betaMg = 4.40490857*c2*1.2294108119931473
mode1_mid_beta6 = [1.00088, 1.00036, 0.991828, 0.981914]
mode1_err_beta6 = [0.0229064, 0.022966, 0.030745, 0.0677737]
mode1_mid_beta8 = [0.997507, 0.995283, 1.00043, 0.994195]
mode1_err_beta8 = [0.0182211, 0.0202573, 0.0618318, 0.0573455]
mode1_mid_beta10 = [1.00043, 1.0298, 0.98814, 0.982315]
mode1_err_beta10 = [0.0143078, 0.0448265, 0.0529464, 0.133968]

figure(1)
errorbar(betaMg, mode1_mid_beta6, yerr=mode1_err_beta6, label="\$\\beta J = 6\$", ls="", marker="s", c="b", ms=0,mew=3,elinewidth=elw,capsize=cs,zorder=1)
eb8 = errorbar(betaMg, mode1_mid_beta8, yerr=mode1_err_beta8, label="\$\\beta J = 8\$", ls="", marker="s", c="g", ms=0,mew=3,elinewidth=elw,capsize=cs,zorder=1)
eb8[end][1][:set_linestyle]("--")
eb10 = errorbar(betaMg, mode1_mid_beta10, yerr=mode1_err_beta10, label="\$\\beta J = 10\$", ls="", marker="s", c="r", ms=0,mew=3,elinewidth=elw,capsize=cs,zorder=1)
eb10[end][1][:set_linestyle](":")

axhline(1, ls="--", c="grey",zorder=-1)

ylim(0.835,1.13)
xlabel("\$\\beta \\cdot M_g\$")
ylabel("\$-\\frac{\\beta}{2\\pi}\\, \\operatorname{Im}(\\omega)\$")
legend(loc = "best", numpoints=3, frameon=0, fancybox=0, columnspacing=1, title="\$ \\beta \\cdot M_h = 0.5 \$")
layout.nice_ticks()
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_ferro/figures/wTD_nonintferro.pdf"))


##--- residues:
mps_factor = 2*100^1.5/((2*pi)^3)*8
r1_mid_pert0 = [-0.00588296, -0.0042871, -0.00336868]
r1_err_pert0 = [0.000154241, 0.000207639, 0.000141669]
r1_mid_pert1 = [-0.00588554, -0.00431816, -0.00339452]  # residues for smallest perturbation pert1 for increasing beta=6,8,10
r1_err_pert1 = [0.000143524, 0.000141413, 0.0000993675] # ...corresponding uncertainties
r1_mid_pert2 = [-0.00587769, -0.00429564, -0.00347675]
r1_err_pert2 = [0.000146985, 0.000169837, 0.000136351]
r1_mid_pert3 = [-0.00581744, -0.00411631, -0.00327047]
r1_err_pert3 = [0.00010691, 0.000320023, 0.000302202]
r1_mid_pert4 = [-0.00545036, -0.00400333, -0.00298583]
r1_err_pert4 = [0.000381936, 0.000151642, 0.000375457]

figure(2)
pert1=round(betaMg[1],2); pert2=round(betaMg[2],2); pert3=round(betaMg[3],2); pert4=round(betaMg[4],2)
errorbar(betaseries-0.1, abs.(mps_factor.*betaseries.*r1_mid_pert0), yerr=mps_factor.*betaseries.*r1_err_pert0, ls="", marker="s", c="k",        ms=0,mew=3,elinewidth=elw,capsize=cs,zorder=1, label="\$\\beta\\cdot M_g = 0.0\$")
eb1 = errorbar(betaseries-0.05, abs.(mps_factor.*betaseries.*r1_mid_pert1), yerr=mps_factor.*betaseries.*r1_err_pert1, ls="", marker="s", c="C0", ms=0,mew=3,elinewidth=elw,capsize=cs,zorder=1, label="\$\\beta\\cdot M_g\\approx $pert1\$")
eb1[end][1][:set_linestyle]("--")
eb2 = errorbar(betaseries, abs.(mps_factor.*betaseries.*r1_mid_pert2), yerr=mps_factor.*betaseries.*r1_err_pert2, ls="", marker="s", c="C1", ms=0,mew=3,elinewidth=elw,capsize=cs,zorder=1, label="\$\\beta\\cdot M_g\\approx $pert2\$")
eb2[end][1][:set_linestyle]("--")
eb3 = errorbar(betaseries+0.05, abs.(mps_factor.*betaseries.*r1_mid_pert3), yerr=mps_factor.*betaseries.*r1_err_pert3, ls="", marker="s", c="C2", ms=0,mew=3,elinewidth=elw,capsize=cs,zorder=1, label="\$\\beta\\cdot M_g\\approx $pert3\$")
eb3[end][1][:set_linestyle]("-.")
eb4 = errorbar(betaseries+0.1, abs.(mps_factor.*betaseries.*r1_mid_pert4), yerr=mps_factor.*betaseries.*r1_err_pert4, ls="", marker="s", c="C3", ms=0,mew=3,elinewidth=elw,capsize=cs,zorder=1, label="\$\\beta\\cdot M_g\\approx $pert4\$")
eb4[end][1][:set_linestyle](":")
ax = subplot(111)
ax[:set_xticks]([6,8,10])
ax[:ticklabel_format](axis="y", style="scientific", scilimits=(-2,4), useOffset=true)
axhline(2.30699, ls="--", c="grey",zorder=-1)

xlabel("\$ \\beta \\cdot J \$")
ylabel("\$\\vert r_1\\vert \\cdot \\beta / 2\\pi\$")
# ylabel("\$ \\vert\\beta \\cdot r_1\\vert\$")
legend(loc="best", numpoints=3, frameon=0, fancybox=0, columnspacing=1, ncol=2, title="\$ \\beta \\cdot M_h = 0.5 \$")
layout.nice_ticks()
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_ferro/figures/residues_nonint_ferro.pdf"))


###-----------------------------------------------------------------------------
### nonintegrable para:

betaseries = [4.0, 6.0, 8.0, 10.0]
c2 = [0.05, 0.1, 0.2, 0.3]
betaMg = 4.40490857*c2*1.2294108119931473
mode1_mid_beta4 = [1.00437, 1.00518, 1.00592, 0.999933]
mode1_err_beta4 = [0.00772613, 0.0050964, 0.00796787, 0.0528455]
mode1_mid_beta6 = [1.00074, 1.0004, 0.99988, 0.996389]
mode1_err_beta6 = [0.0100175, 0.0517487, 0.0184397, 0.0284463]
mode1_mid_beta8 = [0.998108, 0.999151, 0.999361, 1.01292]
mode1_err_beta8 = [0.0214159, 0.0220955, 0.042681, 0.0146277]
mode1_mid_beta10 = [1.00333, 1.03643, 0.999979, 1.01971]
mode1_err_beta10 = [0.0108629, 0.0526172, 0.0229829, 0.0192727]

figure(1)
errorbar(betaMg, mode1_mid_beta6, yerr=mode1_err_beta6, label="\$\\beta J = 6\$", ls="", marker="s", c="b", ms=0,mew=3,elinewidth=elw,capsize=cs,zorder=1)
eb8 = errorbar(betaMg, mode1_mid_beta8, yerr=mode1_err_beta8, label="\$\\beta J = 8\$", ls="", marker="s", c="g", ms=0,mew=3,elinewidth=elw,capsize=cs,zorder=1)
eb8[end][1][:set_linestyle]("--")
eb10 = errorbar(betaMg, mode1_mid_beta10, yerr=mode1_err_beta10, label="\$\\beta J = 10\$", ls="", marker="s", c="r", ms=0,mew=3,elinewidth=elw,capsize=cs,zorder=1)
eb10[end][1][:set_linestyle](":")

axhline(1, ls="--", c="grey",zorder=-1)

ylim(0.835,1.13)
xlabel("\$\\beta \\cdot M_g\$")
ylabel("\$-\\frac{\\beta}{2\\pi}\\, \\operatorname{Im}(\\omega)\$")
# legend(loc = "best", numpoints=3, frameon = 0, fancybox = 0, columnspacing = 1)
layout.nice_ticks()
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_para/figures/wTD_nonintpara.pdf"))


##--- residues:
mps_factor = 2*100^1.5/((2*pi)^3)*8
r1_mid_pert0 = [-0.00462908, -0.00350411, -0.00291749]
r1_err_pert0 = [0.000114917,  0.00021743,  0.0000440214]
r1_mid_pert1 = [-0.00466635, -0.00352241, -0.00293158] # residues for smallest perturbation pert1 for increasing beta=6,8,10
r1_err_pert1 = [0.0000444308, 0.000177103, 0.0000556761] # ...corresponding uncertainties
r1_mid_pert2 = [-0.00448397, -0.00353202, -0.00301048]
r1_err_pert2 = [0.000412691, 0.000172201, 0.000176575]
r1_mid_pert3 = [-0.00463453, -0.00354144, -0.00290431]
r1_err_pert3 = [0.0000762117, 0.000288356, 0.0000868237]
r1_mid_pert4 = [-0.00456059, -0.0035357, -0.00285262]
r1_err_pert4 = [0.000147898, 0.0000965075, 0.00015433]

figure(2)
betaseries = [6.0, 8.0, 10.0]
pert1=round(betaMg[1],2); pert2=round(betaMg[2],2); pert3=round(betaMg[3],2); pert4=round(betaMg[4],2)
errorbar(betaseries-0.1, abs.(mps_factor.*betaseries.*r1_mid_pert0), yerr=mps_factor.*betaseries.*r1_err_pert0, ls="", marker="s", c="k",        ms=0,mew=3,elinewidth=elw,capsize=cs,zorder=1, label="\$\\beta\\cdot M_g = 0.0\$")
eb1 = errorbar(betaseries-0.05, abs.(mps_factor.*betaseries.*r1_mid_pert1), yerr=mps_factor.*betaseries.*r1_err_pert1, ls="", marker="s", c="C0", ms=0,mew=3,elinewidth=elw,capsize=cs,zorder=1, label="\$\\beta\\cdot M_g\\approx $pert1\$")
eb1[end][1][:set_linestyle]("--")
eb2 = errorbar(betaseries, abs.(mps_factor.*betaseries.*r1_mid_pert2), yerr=mps_factor.*betaseries.*r1_err_pert2, ls="", marker="s", c="C1", ms=0,mew=3,elinewidth=elw,capsize=cs,zorder=1, label="\$\\beta\\cdot M_g\\approx $pert2\$")
eb2[end][1][:set_linestyle]("--")
eb3 = errorbar(betaseries+0.05, abs.(mps_factor.*betaseries.*r1_mid_pert3), yerr=mps_factor.*betaseries.*r1_err_pert3, ls="", marker="s", c="C2", ms=0,mew=3,elinewidth=elw,capsize=cs,zorder=1, label="\$\\beta\\cdot M_g\\approx $pert3\$")
eb3[end][1][:set_linestyle]("-.")
eb4 = errorbar(betaseries+0.1, abs.(mps_factor.*betaseries.*r1_mid_pert4), yerr=mps_factor.*betaseries.*r1_err_pert4, ls="", marker="s", c="C3", ms=0,mew=3,elinewidth=elw,capsize=cs,zorder=1, label="\$\\beta\\cdot M_g\\approx $pert4\$")
eb4[end][1][:set_linestyle](":")
ax = subplot(111)
ax[:set_xticks]([6,8,10])
ax[:ticklabel_format](axis="y", style="scientific", scilimits=(-2,4), useOffset=true)
axhline(1.79433, ls="--", c="grey",zorder=-1)

xlabel("\$ \\beta \\cdot J \$")
ylabel("\$\\vert r_1\\vert \\cdot \\beta / 2\\pi\$")
# ylabel("\$ \\vert\\beta \\cdot r_1\\vert\$")
# legend(loc = "best", numpoints=3, frameon=0, fancybox=0, columnspacing=1, title="\$ \\beta \\cdot M_h = 0.5 \$")
layout.nice_ticks()
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_para/figures/residues_nonint_para.pdf"))


###-----------------------------------------------------------------------------
### nonintegrable ferro constant 4.4*M_h/M_g = 2.5:

betaseries = [6.0, 8.0, 10.0]
betaMh = [0.4, 0.5, 0.6]
mode1_mid_beta6 = [0.999283, 0.991828, 1.00019]
mode1_err_beta6 = [0.024097, 0.030745, 0.0396174]
mode1_mid_beta8 = [0.994243, 0.998566, 1.00714]
mode1_err_beta8 = [0.0316912, 0.0731761, 0.056819]
mode1_mid_beta10 = [0.99873, 0.98814, 0.982847]
mode1_err_beta10 = [0.0154962, 0.0529464, 0.0902507]

figure(1)
errorbar(betaMh, mode1_mid_beta6, yerr=mode1_err_beta6, label="\$\\beta J = 6\$", ls="", marker="s", c="b", ms=0,mew=3,elinewidth=3,capsize=5,zorder=1)
eb8 = errorbar(betaMh, mode1_mid_beta8, yerr=mode1_err_beta8, label="\$\\beta J = 8\$", ls="", marker="s", c="g", ms=0,mew=3,elinewidth=3,capsize=5,zorder=1)
eb8[end][1][:set_linestyle]("--")
eb10 = errorbar(betaMh, mode1_mid_beta10, yerr=mode1_err_beta10, label="\$\\beta J = 10\$", ls="", marker="s", c="r", ms=0,mew=3,elinewidth=3,capsize=5,zorder=1)
eb10[end][1][:set_linestyle](":")

axhline(1, ls="--", c="grey",zorder=-1)

ylim(0.835,1.13)
xlabel("\$\\beta \\cdot M_h\$")
ylabel("\$-\\frac{\\beta}{2\\pi}\\, \\operatorname{Im}(\\omega)\$")
legend(loc = "lower left", numpoints=3, frameon=0, fancybox=0, columnspacing=1, title="\$ M_h/M_g \\approx 0.4616 \$")
ax = subplot(111)
ax[:set_xticks]([0.4, 0.5, 0.6])
layout.nice_ticks()
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_ferro/figures/wTD_nonintferro_MhMg.pdf"))



### +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++   longitudinal vs transverse response function

function plot_response(file, plotlabel=false, lstyle="-")
    f = open(string(@__DIR__,file))
    lines = readlines(f)
    close(f)
    sep_inds = findin(lines, [""])
    beta = include_string(split(lines[1])[5])
    if beta==4.0 col="C0" else col="C1" end

    for i = 1:length(sep_inds)
        counter = 1

        if i==1
            steps = sep_inds[i]-3
            t = Array{Float64}(steps)
            ReG = Array{Float64}(steps)
            ImG = Array{Float64}(steps)
            for l = 3 : sep_inds[1]-1
                line = parse.(split(lines[l]))
                t[counter] = line[1]
                ReG[counter] = line[2]
                ImG[counter] = line[3]
                counter += 1
            end
        end
        if ImG[2]<0 ImG=-ImG end # equal sign normalization

        figure(1)
        if plotlabel
            plot(t, ImG/minimum(-ImG), c=col, ls=lstyle, label="\$ \\beta J = $beta \$")
        else
            plot(t, ImG/minimum(-ImG), c=col, ls=lstyle)
        end
    end
end

plot_response("/data/linearresponse/continuum_limit_g0/responseM2beta4.txt", true)
plot_response("/data/linearresponse/longitudinal_response/responseM2beta4.txt", false, "-.")
plot_response("/data/linearresponse/continuum_limit_g0/responseM6beta32.txt", true)
plot_response("/data/linearresponse/longitudinal_response/responseM6beta32.txt", false, "-.")

figure(1)
xlabel("\$t \\cdot J\$")
ylabel("\$G_R(t) \\, / \\, G_R(t_{min})\$")
xlim(0,10)
legend(loc = "best", numpoints=3, frameon=0, fancybox=0, columnspacing=1)
layout.nice_ticks()
savefig(string(@__DIR__,"/data/linearresponse/longitudinal_response/trans_vs_long.pdf"))






### +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++   second Renyi entropy density

function plot_renyi(file, plotlabel, col, lstyle="-")
    f = open(string(@__DIR__,file))
    lines = readlines(f)
    close(f)
    sep_inds = findin(lines, [""])

    for i = 1:length(sep_inds)
        counter = 1

        if i==1
            steps = sep_inds[i]-3
            beta = Array{Float64}(steps)
            errL = Array{Float64}(steps)
            errR = Array{Float64}(steps)
            s2 = Array{Float64}(steps)
            for l = 3 : sep_inds[1]-1
                line = parse.(split(lines[l]))
                beta[counter] = line[1]
                errL[counter] = line[2]
                errR[counter] = line[3]
                s2[counter] = line[4]
                counter += 1
            end
        end

        figure(1)
        loglog(beta, s2, c=col, ls=lstyle, label=plotlabel)
    end
end

plot_renyi("/data/linearresponse/continuum_limit_tricritical/s2critbeta20iTEBD3.txt", "critical", "C0")
plot_renyi("/data/linearresponse/continuum_limit_tricritical/s2tricritbeta20iTEBD3.txt", "tricritical", "C1")
plot_renyi("/data/linearresponse/continuum_limit_tricritical/s2lambda04beta20iTEBD3.txt", "\$\\lambda = 0.4\$", "C2")

figure(1)
xlabel("\$\\beta \\, J\$")
ylabel("\$s_2\$")
axvline(1.8, ls="--", c="grey",zorder=-1)
axvline(4, ls="--", c="grey",zorder=-1)
xlim(0,20)
legend(loc = "best", numpoints=3, frameon=0, fancybox=0, columnspacing=1)
layout.nice_ticks()
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_tricritical/s2_tricrit.pdf"))









### ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
### pseudo entanglement entropies after quench

function plot_renyiquench(file, plotlabel, col, select_dataset="all", lstyle="-")
    f = open(string(@__DIR__,file))
    lines = readlines(f)
    close(f)
    sep_inds = findin(lines, [""])

    for i = 1:length(sep_inds)
        counter = 1

        if i==1
            steps = sep_inds[i]-3
            t = Array{Float64}(steps)
            err = Array{Float64}(steps)
            s1 = Array{Float64}(steps)
            s2 = Array{Float64}(steps)
            for l = 3 : sep_inds[1]-1
                line = parse.(split(lines[l]))
                t[counter] = line[1]
                err[counter] = line[2]
                s1[counter] = line[3]
                s2[counter] = line[4]
                counter += 1
            end
        else
            steps = sep_inds[i]-sep_inds[i-1]-3
            t = Array{Float64}(steps)
            err = Array{Float64}(steps)
            s1 = Array{Float64}(steps)
            s2 = Array{Float64}(steps)
            for l = sep_inds[i-1]+3 : sep_inds[i]-1
                line = parse.(split(lines[l]))
                t[counter] = line[1]
                err[counter] = line[2]
                s1[counter] = line[3]
                s2[counter] = line[4]
                counter += 1
            end
        end

        if i==select_dataset || select_dataset=="all"
            figure(1)
            plot(t, s1-s1[1], ls="--", label=plotlabel)

            figure(2)
            plot(t, s2-s2[1], ls=lstyle, label=plotlabel)

            figure(3)
            semilogy(t, err, ls=lstyle, label=plotlabel)

            figure(4)
            plot(t[1:end-1],diff(s1-s1[1]), ls=lstyle, label=plotlabel)

            figure(5)
            plot(t[1:end-1],diff(s2-s2[1]), ls=lstyle, label=plotlabel)
        end
    end
end

plot_renyiquench("/data/linearresponse/continuum_limit_renyiquench/small/entropiessmallquenchtononint1beta16iTEBD2.txt", "(1)", "C0")
plot_renyiquench("/data/linearresponse/continuum_limit_renyiquench/nonintferro/entropiesE8tononint1beta16iTEBD2.txt", "(2)", "C1")
plot_renyiquench("/data/linearresponse/continuum_limit_renyiquench/nonintferro/entropiescrittononint1beta16iTEBD2D300.txt", "(3)", "C2")
plot_renyiquench("/data/linearresponse/continuum_limit_renyiquench/classical/entropiesclassicaltononint1beta16iTEBD2D300.txt", "(4)", "C3")
# plot_renyiquench("/data/linearresponse/continuum_limit_renyiquench/classical/entropiesclassicalh0g0tononint1beta16iTEBD2D500.txt", "(5)", "C4")

figure(1)
xlabel("\$t J\$")
ylabel("\$\\tilde{s}_1(t) - \\tilde{s}_1(0)\$")
xlim(0,50)
legend(loc = "center right", bbox_to_anchor=(1.0, 0.75), numpoints=3, frameon=0, fancybox=0, columnspacing=1)
layout.nice_ticks()
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_renyiquench/figures/s1.pdf"))

figure(2)
xlabel("\$t J\$")
ylabel("\$\\tilde{s}_2(t) - \\tilde{s}_2(0)\$")
xlim(0,50)
# ax = subplot(111)
# ax[:ticklabel_format](axis="y", style="scientific", scilimits=(-2,1), useOffset=true)
# legend(loc = "best", numpoints=3, frameon=0, fancybox=0, columnspacing=1)
layout.nice_ticks()
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_renyiquench/figures/s2.pdf"))

figure(3)
xlabel("\$t J\$")
ylabel("truncation error")
xlim(0,50)
ylim(1e-18,1e-6)
legend(loc = "lower right", numpoints=3, frameon=0, fancybox=0, columnspacing=1)
layout.nice_ticks()
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_renyiquench/figures/err.pdf"))

figure(4)
xlabel("\$t J\$")
ylabel("\$\\Delta\\tilde{s}_1(t)\$")
xlim(0,50)
# legend(loc = "best", numpoints=3, frameon=0, fancybox=0, columnspacing=1)
layout.nice_ticks()
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_renyiquench/figures/s1_diff.pdf"))

figure(5)
xlabel("\$t J\$")
ylabel("\$\\Delta\\tilde{s}_2(t)\$")
xlim(0,50)
# legend(loc = "best", numpoints=3, frameon=0, fancybox=0, columnspacing=1)
layout.nice_ticks()
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_renyiquench/figures/s2_diff.pdf"))









### ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
### βJ = 2 quenches

function plot_entropies(file, plotlabel, col)
    f = open(string(@__DIR__,file))
    lines = readlines(f)
    close(f)
    sep_inds = findin(lines, [""])

    for i = 1:length(sep_inds)
        counter = 1

        if i==1
            steps = sep_inds[i]-3
            t = Array{Float64}(steps)
            err = Array{Float64}(steps)
            s1 = Array{Float64}(steps)
            s2 = Array{Float64}(steps)
            for l = 3 : sep_inds[1]-1
                line = parse.(split(lines[l]))
                t[counter] = line[1]
                err[counter] = line[2]
                s1[counter] = line[3]
                s2[counter] = line[4]
                counter += 1
            end
        end

        figure(1)
        plot(t, s1-s1[1], ls="--", c=col)
        plot(t, s2-s2[1], ls="-", label=plotlabel, c=col)

        figure(2)
        plot(t, err, label=plotlabel, c=col)
    end
end

## profile 1:
plot_entropies("/data/linearresponse/continuum_limit_renyiquench/small/entropiessmallquenchtononint1beta2iTEBD2D400.txt", "\$\\chi=400\$", "C0")
plot_entropies("/data/linearresponse/continuum_limit_renyiquench/small/entropiessmallquenchtononint1beta2iTEBD2D600.txt", "\$\\chi=600\$", "C1")
ytop = 4.3

figure(1)
xlabel("\$t J\$")
ylabel("\$\\tilde{s}_{1,2}(t) - \\tilde{s}_{1,2}(0)\$")
xlim(0,50)
ylim(0,ytop)
text(3,3.9, "\$\\beta^*J = 2.00,\\, \\beta^* M_1 = 3.0\$")
legend(loc = "best", numpoints=3, frameon=0, fancybox=0, columnspacing=1)
layout.nice_ticks()
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_renyiquench/figures/betaJ2/s12_betaJ2_type1.pdf"))
clf()

figure(2)
xlabel("\$t J\$")
ylabel("truncation error")
xlim(0,50)
# ylim(top=1e-7)
layout.nice_ticks()
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_renyiquench/figures/betaJ2/err_betaJ2_type1.pdf"))
clf()


## profile 2:
plot_entropies("/data/linearresponse/continuum_limit_renyiquench/nonintferro/entropiesE8tononint1beta2iTEBD2D400.txt", "\$\\chi=400\$", "C0")
plot_entropies("/data/linearresponse/continuum_limit_renyiquench/nonintferro/entropiesE8tononint1beta2iTEBD2D600.txt", "\$\\chi=600\$", "C1")

figure(1)
xlabel("\$t J\$")
ylabel("\$\\tilde{s}_{1,2}(t) - \\tilde{s}_{1,2}(0)\$")
xlim(0,50)
ylim(0,ytop)
text(3,3.9, "\$\\beta^*J = 1.92,\\, \\beta^* M_1 = 2.9\$")
# legend(loc = "best", numpoints=3, frameon=0, fancybox=0, columnspacing=1)
layout.nice_ticks()
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_renyiquench/figures/betaJ2/s12_betaJ2_type2.pdf"))
clf()

figure(2)
xlabel("\$t J\$")
ylabel("truncation error")
xlim(0,50)
layout.nice_ticks()
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_renyiquench/figures/betaJ2/err_betaJ2_type2.pdf"))
clf()


## profile 3:
plot_entropies("/data/linearresponse/continuum_limit_renyiquench/nonintferro/entropiescrittononint1beta2iTEBD2D400.txt", "\$\\chi=400\$", "C0")
plot_entropies("/data/linearresponse/continuum_limit_renyiquench/nonintferro/entropiescrittononint1beta2iTEBD2D600.txt", "\$\\chi=600\$", "C1")

figure(1)
xlabel("\$t J\$")
ylabel("\$\\tilde{s}_{1,2}(t) - \\tilde{s}_{1,2}(0)\$")
xlim(0,50)
ylim(0,ytop)
text(3,3.9, "\$\\beta^*J = 1.34,\\, \\beta^* M_1 = 2.0\$")
# legend(loc = "best", numpoints=3, frameon=0, fancybox=0, columnspacing=1)
layout.nice_ticks()
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_renyiquench/figures/betaJ2/s12_betaJ2_type3.pdf"))
clf()

figure(2)
xlabel("\$t J\$")
ylabel("truncation error")
xlim(0,50)
layout.nice_ticks()
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_renyiquench/figures/betaJ2/err_betaJ2_type3.pdf"))
clf()


##---------------------- type 1 & 2 from βJ=16:
plot_entropies("/data/linearresponse/continuum_limit_renyiquench/small/entropiessmallquenchtononint1beta16iTEBD2.txt", "(1)", "C0")
plot_entropies("/data/linearresponse/continuum_limit_renyiquench/nonintferro/entropiesE8tononint1beta16iTEBD2.txt", "(2)", "C1")

figure(1)
xlabel("\$t J\$")
ylabel("\$\\tilde{s}_{1,2}(t) - \\tilde{s}_{1,2}(0)\$")
xlim(0,50)
ylim(-0.01,0.08)
# text(1,5.5, "\$\\beta^*J = 0.50,\\, \\beta^* M_1 = 0.8\$")
legend(loc = "upper left", numpoints=3, frameon=0, fancybox=0, columnspacing=1)
layout.nice_ticks()
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_renyiquench/small/plots/s12_betaJ16_type12.pdf"))


##---------------------- type 3 & 4 from βJ=16:
plot_entropies("/data/linearresponse/continuum_limit_renyiquench/nonintferro/entropiescrittononint1beta16iTEBD2D300.txt", "(3)", "C2")
plot_entropies("/data/linearresponse/continuum_limit_renyiquench/classical/entropiesclassicaltononint1beta16iTEBD2D300.txt", "(4)", "C3")

figure(1)
xlabel("\$t J\$")
ylabel("\$\\tilde{s}_{1,2}(t) - \\tilde{s}_{1,2}(0)\$")
xlim(0,50)
# ylim(-0.01,0.08)
# text(1,5.5, "\$\\beta^*J = 0.50,\\, \\beta^* M_1 = 0.8\$")
legend(loc = "upper left", numpoints=3, frameon=0, fancybox=0, columnspacing=1)
layout.nice_ticks()
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_renyiquench/nonintferro/plots/s12_betaJ16_type34.pdf"))


##---------------------- high temp  βJ=0.5 profile 1:
plot_entropies("/data/linearresponse/continuum_limit_renyiquench/small/entropiessmallquenchtononint1beta05iTEBD2D400.txt", "\$\\chi=400\$", "C0")
plot_entropies("/data/linearresponse/continuum_limit_renyiquench/small/entropiessmallquenchtononint1beta05iTEBD2D500.txt", "\$\\chi=500\$", "C1")

figure(1)
x = linspace(0,10,100)
# y1 = -0.02261 + 0.682388*x  # s1 fit
y1 = 0.037469 + 0.665854*x  # s1 fit
y2 = 0.183528 + 0.527892*x  # s2 fit
plot(x, y1, ls="--", c="k",lw=1)
plot(x, y2, ls="-", c="k",lw=1)
xlabel("\$t J\$")
ylabel("\$\\tilde{s}_{1,2}(t) - \\tilde{s}_{1,2}(0)\$")
xlim(0,10)
ylim(0,6)
axvline(7.5, ls="--", c="grey",zorder=-1)
text(1,5.5, "\$\\beta^*J = 0.50,\\, \\beta^* M_1 = 0.8\$")
# legend(loc = "lower right", numpoints=3, frameon=0, fancybox=0, columnspacing=1)
legend(loc = "lower right", numpoints=3, frameon=1, fancybox=1, columnspacing=1, facecolor="white")
layout.nice_ticks()
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_renyiquench/small/plots/s12_betaJ05_type1.pdf"))

figure(2)
xlabel("\$t J\$")
ylabel("truncation error")
xlim(0,10)
# ylim(top=1e-7)
layout.nice_ticks()
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_renyiquench/small/plots/err_betaJ05_type1.pdf"))



##---------------------- comparison high temp  βJ=0.5 profile 1 with free fermion equivalent:
plot_entropies("/data/linearresponse/continuum_limit_renyiquench/small/entropiessmallquenchtononint1beta05iTEBD2D500.txt", "(1)", "C0")
plot_entropies("/data/linearresponse/continuum_limit_renyiquench/freefermions/entropiesfreequenchtype1beta05iTEBD2D500.txt", "free fermions", "C1")

figure(1)
x = linspace(0,10,100)
y1 = 0.037469 + 0.665854*x  # s1 fit (1)
y2 = 0.183528 + 0.527892*x  # s2 fit (1)
y3 = 0.0108503 + 0.68425*x  # s1 fit (ff1)
y4 = 0.0835516 + 0.58525*x  # s2 fit (ff1)
plot(x, y1, ls="--", c="k",lw=1)
plot(x, y2, ls="-", c="k",lw=1)
plot(x, y3, ls="--", c="grey",lw=1)
plot(x, y4, ls="-", c="grey",lw=1)
xlabel("\$t J\$")
ylabel("\$\\tilde{s}_{1,2}(t) - \\tilde{s}_{1,2}(0)\$")
xlim(0,10)
ylim(0,6)
axvline(7.5, ls="--", c="grey",zorder=-1)
# text(1,5.5, "\$\\beta^*J = 0.50,\\, \\beta^* M_1 = 0.8,\\, \\beta^* M_h = 0.06\$")
# legend(loc = "lower right", numpoints=3, frameon=1, fancybox=1, columnspacing=1, facecolor="white")
legend(loc = "upper left", numpoints=3, frameon=0, fancybox=0, columnspacing=1, title="\$\\beta^* M_1 = 0.8,\\, \\beta^* M_h = 0.06\$")
layout.nice_ticks()
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_renyiquench/small/plots/s12_betaJ05_type1_vs_ff1.pdf"))



##---------------------- high temp  βeff J=0.91 profile 4:
plot_entropies("/data/linearresponse/continuum_limit_renyiquench/classical/entropiesclassicaltononint1beta16iTEBD2D300.txt", "\$\\chi=300\$", "C0")
plot_entropies("/data/linearresponse/continuum_limit_renyiquench/classical/entropiesclassicaltononint1beta16iTEBD2D500.txt", "\$\\chi=500\$", "C1")
plot_entropies("/data/linearresponse/continuum_limit_renyiquench/classical/entropiesclassicaltononint1beta16iTEBD2D600.txt", "\$\\chi=600\$", "C2")

figure(1)
xlabel("\$t J\$")
ylabel("\$\\tilde{s}_{1,2}(t) - \\tilde{s}_{1,2}(0)\$")
xlim(0,15)
ylim(0,5)
axvline(9.0, ls="--", c="grey",zorder=-1)
text(1,4.5, "\$\\beta^*J = 0.91,\\, \\beta^* M_1 = 1.4\$")
legend(loc = "lower right", numpoints=3, frameon=0, fancybox=0, columnspacing=1)
layout.nice_ticks()
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_renyiquench/classical/plots/s12_betaJ16_type4.pdf"))

figure(2)
xlabel("\$t J\$")
ylabel("truncation error")
xlim(0,15)
# ylim(top=1e-7)
layout.nice_ticks()
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_renyiquench/classical/plots/err_betaJ16_type4.pdf"))






### ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
### all high T types (4), (7), (3) at βeff ≈ 0.91

plot_entropies("/data/linearresponse/continuum_limit_renyiquench/nonintferro/entropiescrittononint1beta097iTEBD2D500.txt", "(3)", "C0")
plot_entropies("/data/linearresponse/continuum_limit_renyiquench/classical/entropiesclassicaltononint1beta16iTEBD2D500.txt", "(4)", "C1")
plot_entropies("/data/linearresponse/continuum_limit_renyiquench/nonintferro/entropiestype7beta16iTEBD2D500.txt", "(6)", "C2")

figure(1)
x = linspace(0,15,100)
y1 = 0.0880231 + 0.422513*x  # s1 fit
plot(x, y1, ls="--", c="k",lw=1)
xlabel("\$t J\$")
ylabel("\$\\tilde{s}_{1,2}(t) - \\tilde{s}_{1,2}(0)\$")
xlim(0,10)
ylim(0,6)
axvline(9.0, ls="--", c="grey",zorder=-1)
# text(1,5.5, "\$\\beta^* M_1 \\approx 1.4\$")
legend(loc = "upper left", numpoints=3, frameon=0, fancybox=0, columnspacing=1, title="\$\\beta^* M_1 \\approx 1.4\$")
layout.nice_ticks()
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_renyiquench/figures/s12_betaeff09_typecomp.pdf"))




##---------------------- high temp  βJ=0.5 profile 1 for different cont limits:
plot_entropies("/data/linearresponse/continuum_limit_renyiquench/small/entropiessmallquenchtononint1beta05iTEBD2D500.txt", "\$n = 0\$", "C0")
plot_entropies("/data/linearresponse/continuum_limit_renyiquench/small/entropiestype1beta05cont2D500.txt", "\$n = 1\$", "C1")
plot_entropies("/data/linearresponse/continuum_limit_renyiquench/small/entropiestype1beta05cont3D500.txt", "\$n = 2\$", "C2")
plot_entropies("/data/linearresponse/continuum_limit_renyiquench/small/entropiestype1beta05cont4D500.txt", "\$n = 3\$", "C3")

figure(1)
# x = linspace(0,10,100)
# y1 = 0.037469 + 0.665854*x  # s1 fit
# y2 = 0.183528 + 0.527892*x  # s2 fit
# plot(x, y1, ls="--", c="k",lw=1)
# plot(x, y2, ls="-", c="k",lw=1)
xlabel("\$t J\$")
ylabel("\$\\tilde{s}_{1,2}(t) - \\tilde{s}_{1,2}(0)\$")
xlim(0,10)
ylim(0,6)
# text(1,5.5, "\$\\beta^*J = 0.50,\\, \\beta^* M_1 = 0.8\$")
legend(loc = "lower right", numpoints=3, frameon=0, fancybox=0, columnspacing=1)
layout.nice_ticks()
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_renyiquench/small/plots/s12_betaJ05_type1_contlimits.pdf"))

figure(2)
xlabel("\$t J\$")
ylabel("truncation error")
xlim(0,10)
# ylim(top=1e-7)
layout.nice_ticks()
# savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_renyiquench/small/plots/err_betaJ05_type1_contlimits.pdf"))




### ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
### different cont limits at βJ=0.5

function plot_entropies_contlimit(file, plotlabel, col, M1, indmax, ls_s1="--")
    f = open(string(@__DIR__,file))
    lines = readlines(f)
    close(f)
    sep_inds = findin(lines, [""])

    for i = 1:length(sep_inds)
        counter = 1

        if i==1
            steps = sep_inds[i]-3
            t = Array{Float64}(steps)
            err = Array{Float64}(steps)
            s1 = Array{Float64}(steps)
            s2 = Array{Float64}(steps)
            for l = 3 : sep_inds[1]-1
                line = parse.(split(lines[l]))
                t[counter] = line[1]
                err[counter] = line[2]
                s1[counter] = line[3]
                s2[counter] = line[4]
                counter += 1
            end
        end

        figure(1)
        plot(t[1:indmax]*M1, s1[1:indmax]-s1[1], ls=ls_s1, c=col)
        plot(t[1:indmax]*M1, s2[1:indmax]-s2[1], ls="-", label=plotlabel, c=col)

        figure(2)
        plot(t, err, label=plotlabel, c=col)
    end
end


### ---------------------------  cont limits type 1:
plot_entropies_contlimit("/data/linearresponse/continuum_limit_renyiquench/small/entropiessmallquenchtononint1beta05iTEBD2D500.txt", "\$n = 0\$", "C0", 1.5280723187141292, 200)
plot_entropies_contlimit("/data/linearresponse/continuum_limit_renyiquench/small/entropiestype1beta05cont2D500.txt", "\$n = 1\$", "C1", 0.7640361593570648, 150)
plot_entropies_contlimit("/data/linearresponse/continuum_limit_renyiquench/small/entropiestype1beta05cont3D500.txt", "\$n = 2\$", "C2", 0.3820180796785324, 150)
plot_entropies_contlimit("/data/linearresponse/continuum_limit_renyiquench/small/entropiestype1beta05cont4D500.txt", "\$n = 3\$", "C3", 0.19100903983926618, 150)

figure(1)
xlabel("\$t M_1\$")
ylabel("\$\\tilde{s}_{1,2}(t) - \\tilde{s}_{1,2}(0)\$")
xlim(0,8)
ylim(0,5)
legend(loc = "lower right", numpoints=3, frameon=0, fancybox=0, columnspacing=1)
layout.nice_ticks()
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_renyiquench/small/plots/s12_betaJ05_type1_contlimits.pdf"))


### ---------------------------  different slopes:
figure(2)
βeffM = 0.5*[1.5280723187141292, 0.7640361593570648, 0.3820180796785324,0.19100903983926618]
slopesS1 = [0.442649, 0.910587, 1.83865, 3.68663]
slopesS2 = [0.353898, 0.789369, 1.61486, 3.24505]
x = linspace(0,1,100)
s1fit = 2.93485*x
s2fit = 3.59939*x
plot(βeffM, 1 ./ slopesS1, ls="", marker="s", label="\$\\tilde s_1\$")
plot(βeffM, 1 ./ slopesS2, ls="", marker="o", label="\$\\tilde s_2\$")
plot(x, s1fit, ls="-", c="blue", zorder=-1)
plot(x, s2fit, ls="--", c="orange", zorder=-1)
legend(loc = "lower right", numpoints=3, frameon=0, fancybox=0, columnspacing=1)
xlim(0,1)
ylim(0,3)
xlabel("\$\\beta^* M_1\$")
ylabel("\$1 / r\$")
layout.nice_ticks()
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_renyiquench/small/plots/contlimits_slopedependence.pdf"))


### ---------------------------  contlimit at fixed β^* M^post:
plot_entropies_contlimit("/data/linearresponse/continuum_limit_renyiquench/small/entropiessmallquenchtononint1beta05iTEBD2D500.txt", "\$n = 0\$", "C0", 1.5280723187141292, 200)
plot_entropies_contlimit("/data/linearresponse/continuum_limit_renyiquench/small/entropiestype1beta1cont2D500.txt", "\$n = 1\$", "C1", 0.7640361593570648, 150)
plot_entropies_contlimit("/data/linearresponse/continuum_limit_renyiquench/small/entropiestype1beta2cont3D500.txt", "\$n = 2\$", "C2", 0.3820180796785324, 200, "-.")
plot_entropies_contlimit("/data/linearresponse/continuum_limit_renyiquench/small/entropiestype1beta4cont4D500.txt", "\$n = 3\$", "C3", 0.19100903983926618, 400)

figure(1)
xlabel("\$t M_1\$")
ylabel("\$\\tilde{s}_{1,2}(t) - \\tilde{s}_{1,2}(0)\$")
xlim(0,3.8)
ylim(0,2.5)
legend(loc = "lower right", numpoints=3, frameon=0, fancybox=0, columnspacing=1)
layout.nice_ticks()
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_renyiquench/small/plots/s12_type1_contlimits2.pdf"))


### ---------------------------  dependence of slope of s_1 growth on effective temperature

βeff  = [6.1,          3.55,        1.55,     0.91, 2.0,       1.92,      1.34,     0.5]
slope = [0.0000305656, 0.00124843, 0.0471359, 0.42, 0.0467689, 0.0516883, 0.193384, 0.67]
M1    = 1.5

figure(1)
axvline(1.0, ls="--", c="grey",zorder=-1)
plot(βeff*M1, slope, ls="", marker="s")
xlabel("\$\\beta^* M_1\$")
ylabel("linear slope")
layout.nice_ticks()
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_renyiquench/figures/slope_temp_dependence.pdf"))





### ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
### scaling of temperature dependence of s2 in nonintferro regime
M1J = [1.52807, 0.764036, 0.382018, 0.191009]

function plot_s2scaling(file)
    f = open(string(@__DIR__,file))
    lines = readlines(f)
    close(f)
    sep_inds = findin(lines, [""])

    beta_vals=Array{Any}(length(sep_inds)); E_vals=Array{Any}(length(sep_inds)); s2_vals=Array{Any}(length(sep_inds))

    for i = 1:length(sep_inds)
        counter = 1

        if i==1
            steps = sep_inds[i]-3
             beta = Array{Float64}(steps)
             E = Array{Float64}(steps)
             s2 = Array{Float64}(steps)
            for l = 3 : sep_inds[1]-1
                line = parse.(split(lines[l]))
                beta[counter] = line[1]
                E[counter] = line[2]
                s2[counter] = line[3]
                counter += 1
            end
        else
            steps = sep_inds[i]-sep_inds[i-1]-3
             beta = Array{Float64}(steps)
             E = Array{Float64}(steps)
             s2 = Array{Float64}(steps)
            for l = sep_inds[i-1]+3 : sep_inds[i]-1
                line = parse.(split(lines[l]))
                beta[counter] = line[1]
                E[counter] = line[2]
                s2[counter] = line[3]
                counter += 1
            end
        end
        beta_vals[i] = beta; E_vals[i] = E; s2_vals[i] = s2

        figure(1)
        n=i-1
        plot(beta*M1J[i], exp.(beta*M1J[i]) .* s2 .* sqrt.(beta/M1J[i]), label="\$n=$n\$")

        figure(2)
        loglog(beta*M1J[i], s2, label="\$n=$n\$")

        figure(3)
        n=i-1
        semilogy(beta*M1J[i], s2, label="\$n=$n\$")

        figure(4)
        loglog(beta*M1J[i], E-E[end])

        ## s2 linear fit:
        println(i)
        dbeta = beta[2]-beta[1]
        ind_min = 51
        ind_max = find(x->x<=1, beta*M1J[i])[end]
        a1,b1 = linreg(log.(beta[ind_min:ind_max]),log.(s2[ind_min:ind_max]))
        println("s2: ind_max, b1 = ",ind_max,", ",b1)
        a2,b2 = linreg(log.(beta[ind_min:ind_max]),log.((E-E[end])[ind_min:ind_max]))
        println("E: ind_max, b2 = ",ind_max,", ",b2)
        a3,b3 = linreg(log.(beta[ind_min:ind_max-1]), log.(-diff(E[ind_min:ind_max])/dbeta))
        println("dE: ind_max, b3 = ",ind_max,", ",b3)
    end

    return beta_vals, E_vals, s2_vals
end

beta_n,E_n,s2_n = plot_s2scaling("/data/linearresponse/continuum_limit_renyiquench/nonintferro/thermal_Renyi_scaling/energy_vs_beta_vs_s2.txt")

figure(1)
xlabel("\$\\beta M_1\$")
ylabel("\$e^{\\beta M_1}\\sqrt{\\beta/M_1} s_2\$")
xlim(-0.5,14)
ylim(0,0.8)
legend(loc = "best", numpoints=3, frameon=0, fancybox=0, columnspacing=1)
layout.nice_ticks()
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_renyiquench/nonintferro/plots/s2_lowT_scaling(2).pdf"))

figure(2)
axvline(1.0, ls="-", c="grey",zorder=-1)
for i=1:4
    axvline(0.5*M1J[i], ls="--", c="C"*string(i-1),zorder=-1)
end
xlabel("\$\\beta M_1\$")
ylabel("\$s_2\$")
xlim(1.8e-3,5)
ylim(3e-3,1)
legend(loc = "best", numpoints=3, frameon=0, fancybox=0, columnspacing=1)
layout.nice_ticks()
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_renyiquench/nonintferro/plots/s2_highT_scaling.pdf"))

figure(3)
xlabel("\$\\beta M_1\$")
ylabel("\$s_2\$")
xlim(0,14)
ylim(1e-7,1)
legend(loc = "best", numpoints=3, frameon=0, fancybox=0, columnspacing=1)
layout.nice_ticks()
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_renyiquench/nonintferro/plots/s2_lowT_scaling.pdf"))

figure(4)
axvline(1.0, ls="-", c="grey",zorder=-1)
for i=1:4
    axvline(0.5*M1J[i], ls="--", c="C"*string(i-1),zorder=-1)
end
xlabel("\$\\beta M_1\$")
ylabel("\$E-E(\\beta J=16)\$")
xlim(1.8e-3,5)
ylim(1e-4,2)
legend(loc = "best", numpoints=3, frameon=0, fancybox=0, columnspacing=1)
layout.nice_ticks()
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_renyiquench/nonintferro/plots/E_highT_scaling.pdf"))




### ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
###   example meson oscillations of G_ret at Mh/Mg≈1

function plot_response(file, lstyle="-")
    f = open(string(@__DIR__,file))
    lines = readlines(f)
    close(f)
    sep_inds = findin(lines, [""])
    beta = include_string(split(lines[1])[5])
    if beta==16.0 col="b" elseif beta==2.0 col="g" else col="r" end

    for i = 1:length(sep_inds)
        counter = 1

        if i==1
            steps = sep_inds[i]-3
            t = Array{Float64}(steps)
            ReG = Array{Float64}(steps)
            ImG = Array{Float64}(steps)
            for l = 3 : sep_inds[1]-1
                line = parse.(split(lines[l]))
                t[counter] = line[1]
                ReG[counter] = line[2]
                ImG[counter] = line[3]
                counter += 1
            end
        end
        # if ImG[2]<0 ImG=-ImG end # equal sign normalization

        figure(1)
        if beta==2.0
            plot(t[1:2200], ImG[1:2200], ls=lstyle, c=col, label="\$ \\beta J = $beta \$")
        else
            plot(t, ImG, ls=lstyle, c=col, label="\$ \\beta J = $beta \$")
        end
    end
end

plot_response("/data/linearresponse/continuum_limit_nonint/responsenonint2beta16.txt")
plot_response("/data/linearresponse/continuum_limit_nonint/responsenonint2beta2.txt", "-.")
plot_response("/data/linearresponse/continuum_limit_nonint/responsenonint2beta05.txt", "-.")

figure(1)
xlabel("\$t \\cdot J\$")
ylabel("\$G_R(t)\$")
# xlim(0,50)
ylim(-0.0025,0.0025)
ax = subplot(111)
ax[:ticklabel_format](axis="y", style="scientific", scilimits=(-3,1), useOffset=true)
legend(loc = "best", bbox_to_anchor=(0.1, 0., 0.5, 0.5), numpoints=3, frameon=0, fancybox=0, columnspacing=1)
layout.nice_ticks()
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_nonint/G_with_mesons_at_different_T.pdf"))


##--- residues of first meson:
betaM1 = [0.6967592592592593, 1.3935185185185186, 2.787037037037037, 5.574074074074074, 22.296296296296298]
r1_mid_MPS = [2.3845080621099208e-6,1.4882628053948498e-6,0.00005981646784212492,0.00017568671988650068,0.00018653900838201803]
r1_err_MPS = [1.201433988268095e-6,8.298365469459521e-7,0.00003382377223108669,0.00008904512983323913,0.00009825941305345205]

figure(1)
errorbar(betaM1, r1_mid_MPS, yerr=r1_err_MPS, ls="", marker="o", ms=3,mew=3,elinewidth=elw,capsize=cs,zorder=1)
yscale("log")
xlim(0,25)
xlabel("\$ \\beta M_1 \$")
ylabel("\$r_1\$")
layout.nice_ticks()
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_nonint/residues1stMeson.pdf"))





### ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
###   coarsegrained correlator signals:

function plot_cg_response(file, lbl, num, lstyle, col)
    f = open(string(@__DIR__,file))
    lines = readlines(f)
    close(f)
    sep_inds = findin(lines, [""])
    beta = include_string(split(lines[1])[5])
    # if beta==16.0 col="b" elseif beta==2.0 col="g" else col="r" end

    for i = 1:length(sep_inds)
        counter = 1

        if i==1
            steps = sep_inds[i]-3
            t = Array{Float64}(steps)
            ReG = Array{Float64}(steps)
            ImG = Array{Float64}(steps)
            err = Array{Float64}(steps)
            for l = 3 : sep_inds[1]-1
                line = parse.(split(lines[l]))
                t[counter] = line[1]
                ReG[counter] = line[2]
                ImG[counter] = line[3]
                err[counter] = line[4]
                counter += 1
            end
        end
        # if ImG[2]<0 ImG=-ImG end # equal sign normalization

        figure(num)
        plot(t, ImG, label=lbl, c=col, ls=lstyle)
    end
end

function plot_cg_err(file, lbl, lstyle, col)
    f = open(string(@__DIR__,file))
    lines = readlines(f)
    close(f)
    sep_inds = findin(lines, [""])
    beta = include_string(split(lines[1])[5])

    for i = 1:length(sep_inds)
        counter = 1

        if i==1
            steps = sep_inds[i]-3
            t = Array{Float64}(steps)
            ReG = Array{Float64}(steps)
            ImG = Array{Float64}(steps)
            err = Array{Float64}(steps)
            for l = 3 : sep_inds[1]-1
                line = parse.(split(lines[l]))
                t[counter] = line[1]
                ReG[counter] = line[2]
                ImG[counter] = line[3]
                err[counter] = line[4]
                counter += 1
            end
        end

        figure(3)
        semilogy(t, err, label=lbl, c=col, ls=lstyle)
    end
end

## bare responses:
plot_cg_response("/data/linearresponse/continuum_limit_coarsegrained/bare/responsecritbeta8size10bare.txt", "\$N=10\$", 1, ":", "C0")
plot_cg_response("/data/linearresponse/continuum_limit_coarsegrained/bare/responsecritbeta8size20bare.txt", "\$N=20\$", 1, "-.", "C1")
plot_cg_response("/data/linearresponse/continuum_limit_coarsegrained/bare/responsecritbeta8size40bare.txt", "\$N=40\$", 1, "-", "C2")
figure(1)
xlabel("\$t \\cdot J\$")
ylabel("\$G_R(t)\$")
ax = subplot(111)
ax[:ticklabel_format](axis="y", style="scientific", scilimits=(-2,1), useOffset=true)
# legend(loc = "lower right", numpoints=3, frameon=0, fancybox=0, columnspacing=1)
legend(loc = "lower right", ncol=3, numpoints=3, frameon=1, facecolor="white", fancybox=1, columnspacing=1, title="bare")
layout.nice_ticks()
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_coarsegrained/bare/Gsigmax_bare_finite_sizes.pdf"))


## cg binary responses:
# clf()
plot_cg_response("/data/linearresponse/continuum_limit_coarsegrained/binary/responsecritHsbeta8size5cgscaled.txt", "\$N=10\$", 2, ":", "C0")
plot_cg_response("/data/linearresponse/continuum_limit_coarsegrained/binary/responsecritHsbeta8size10cgscaled.txt", "\$N=20\$", 2, "-.", "C1")
plot_cg_response("/data/linearresponse/continuum_limit_coarsegrained/binary/responsecritHsbeta8size20cgscaled.txt", "\$N=40\$", 2, "-", "C2")
figure(2)
# ylim(-1.4,1.2)
xlabel("\$t \\cdot J\$")
ylabel("\$G_R(t)\$")
legend(loc = "lower right", ncol=3, numpoints=3, frameon=1, facecolor="white", fancybox=1, columnspacing=1, title="coarsegrained")
layout.nice_ticks()
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_coarsegrained/binary/Geps_binary_finite_sizes.pdf"))



## slected truncation errors:
plot_cg_err("/data/linearresponse/continuum_limit_coarsegrained/bare/responsecritbeta8size40bare.txt", "bare critical \$(N=40)\$", "-", "C1")
plot_cg_err("/data/linearresponse/continuum_limit_coarsegrained/binary/responsecritHsbeta8size20cgscaled.txt", "coarsegrained critical \$(N=40)\$", "-", "C0")
plot_cg_err("/data/linearresponse/continuum_limit_coarsegrained/binary/responsecritHsbeta8size30cgscaledsigmapert.txt", "coarsegrained E\$_8\$ \$(N=60)\$", "--", "C0")
figure(3)
xlabel("\$t \\cdot J\$")
ylabel("truncation error")
legend(loc = "lower right", numpoints=3, frameon=0, fancybox=0, columnspacing=1)
layout.nice_ticks()
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_coarsegrained/binary/err_binary.pdf"))




# #######
show()

;
