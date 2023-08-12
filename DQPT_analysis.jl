include("layout_subfigs.jl")
using layout
using PyPlot
using PyCall
@pyimport matplotlib.ticker as ticker

spectrum_max_level = 5
rates_max_level = 4

### FUNCTIONS:

function plot_entropies(entropy_vals,ind_max,numfig)
    figure(0+numfig)
    plot(entropy_vals[:,1], entropy_vals[:,2])

    figure(1+numfig)
    plot(entropy_vals[1:ind_max,1], entropy_vals[1:ind_max,3], label="\$\\alpha = 1\$")
    plot(entropy_vals[1:ind_max,1], entropy_vals[1:ind_max,4], label="\$\\alpha = 2\$")
end

function plot_spectrum(lambda_vals,ind_max,numfig,plot_doubleaxis=true)
    figure(2+numfig)

    if plot_doubleaxis
        ax1 = subplot(111)
        r = 0
        plot(lambda_vals[1:ind_max,1], -log.(lambda_vals[1:ind_max,r+2]), ls="--", label="\$r = $r\$")
        ax2 = ax1[:twinx]()
        plot(lambda_vals[1:ind_max,1], -log.(lambda_vals[1:ind_max,r+2]), ls="--", label="\$r = 0\$") # dummy for legend
        for r = 1:min(spectrum_max_level,size(lambda_vals,2)-2)
            plot(lambda_vals[1:ind_max,1], -log.(lambda_vals[1:ind_max,r+2]), label="\$r = $r\$",c="C"*string(r))
            # semilogy(lambda_vals[1:ind_max,1], -log.(lambda_vals[1:ind_max,r+2]), label="\$r = $r\$")
        end
        return ax1, ax2
    else
        for r = 0:min(spectrum_max_level,size(lambda_vals,2)-2)
            plot(lambda_vals[1:ind_max,1], -log.(lambda_vals[1:ind_max,r+2]), label="\$r = $r\$",c="C"*string(r))
        end
    end
end

function g_r(lambda_vals, r, ind_max)
    ## r >= 1
    return log.(lambda_vals[1:ind_max,2]) - log.(lambda_vals[1:ind_max,r+2])
end

function plot_ratios(lambda_vals,ind_max,numfig,plot_log=false)
    figure(3+numfig)
    for r = 2:min(size(lambda_vals,2)-2,spectrum_max_level)
        axhline(r+1, ls="--", c="grey",zorder=-1)
        if plot_log
            semilogy(lambda_vals[1:ind_max,1], g_r(lambda_vals,r,ind_max) ./ g_r(lambda_vals,1,ind_max),c="C"*string(r), label="\$r = $r\$")
        else
            plot(lambda_vals[1:ind_max,1], g_r(lambda_vals,r,ind_max) ./ g_r(lambda_vals,1,ind_max),c="C"*string(r), label="\$r = $r\$")
        end
    end
end

function plot_rates(rate_vals,ind_max,numfig,plot_log=false)
    ## mask empty entries:
    for i=1:size(rate_vals,1)
        for j=1:size(rate_vals,2)
            if abs.(rate_vals[i,j]).<1e-50 || abs.(rate_vals[i,j])>.1e50
                rate_vals[i,j] = NaN
            end
        end
    end

    figure(4+numfig)
    for i = 1:rates_max_level # size(rate_vals,2)-1
        if plot_log
            semilogy(rate_vals[1:ind_max,1], rate_vals[1:ind_max,i+1], label="\$i = $i\$")
        else
            plot(rate_vals[1:ind_max,1], rate_vals[1:ind_max,i+1], label="\$i = $i\$")
        end
    end
end

function plot_gaps(lambda_vals,ind_max,numfig,plot_log=false)
    figure(5+numfig)
    for r = 1:min(size(lambda_vals,2)-2,spectrum_max_level)
        if plot_log
            semilogy(lambda_vals[1:ind_max,1], 1 ./ g_r(lambda_vals,r,ind_max),c="C"*string(r), label="\$r = $r\$")
        else
            plot(lambda_vals[1:ind_max,1], 1 ./ g_r(lambda_vals,r,ind_max),c="C"*string(r), label="\$r = $r\$")
        end
    end
end


### PLOTS:

##==============================================================================
##  ferro --> meson
## data:
file_type = "ferro_meson"
entropies_filename = "linearresponse/continuum_limit_DQPT/entropies_"*file_type
spectrum_filename = "linearresponse/continuum_limit_DQPT/spectrum_"*file_type
rates_filename = "linearresponse/continuum_limit_DQPT/rates_"*file_type

entropies_ferro_meson = readdlm(string(@__DIR__,"/data/"*entropies_filename*".txt"), skipstart=2)
lambdas_ferro_meson = readdlm(string(@__DIR__,"/data/"*spectrum_filename*".txt"), skipstart=2)
rates_ferro_meson = readdlm(string(@__DIR__,"/data/"*rates_filename*".txt"), skipstart=2)

tmax_ferro_meson = entropies_ferro_meson[end,1]
dt_ferro_meson = entropies_ferro_meson[2,1]-entropies_ferro_meson[1,1]
ind_max_ferro_meson = Int(round(tmax_ferro_meson/dt_ferro_meson))

## entropies:
plot_entropies(entropies_ferro_meson,ind_max_ferro_meson,0)
figure(1)
ax = subplot(111)
ax[:ticklabel_format](axis="y",style="scientific",scilimits=(-2,1),useOffset=true,useMathText=true)
formatter = ticker.ScalarFormatter(useMathText=true)
formatter[:set_scientific](true)
formatter[:set_powerlimits]((-1,1))
ax[:yaxis][:set_major_formatter](formatter)
# ax[:xaxis][:set_tick_params](pad=0)
xlim(0,tmax_ferro_meson)
ylim(0,6.5e-4)
xlabel("\$t J\$")
ylabel("\$S_{\\alpha}\$")
text(-0.14,1.02, "(a)", fontsize=25, ha="center",va="center", transform=ax[:transAxes])
layout.nice_ticks()
legend(loc = "upper right", numpoints=3, frameon=0, facecolor="white", fancybox=1, ncol=2)
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_DQPT/figures/paper/Fig2a.pdf"))
# savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_DQPT/figures/entropies_"*file_type*".pdf"))

# double axis example:
# figure(1)
# t = linspace(1,10,200)
# data1 = exp.(t)
# data2 = sin.(2 * pi * t)
# ax1 = subplot(111)
# plot(t,data1,c="r",label="data1")
# # ylabel("y1")
# ax1[:set_ylabel]("y1")
# ax2 = ax1[:twinx]()
# plot(t,data2,label="data2")
# plot(t,data2*2,label="data3")
# # ylabel("y2")
# ax2[:set_ylabel]("y2")
# ax1[:set_ylim](0,30000)
# ax1[:legend](loc="upper right")
# ax2[:legend](loc="lower right")

## spectrum:
ax1,ax2 = plot_spectrum(lambdas_ferro_meson,ind_max_ferro_meson,0)
figure(2)
ax1[:ticklabel_format](axis="y", style="scientific", scilimits=(-3,1), useOffset=true, useMathText=true)
formatter = ticker.ScalarFormatter(useMathText=true)
formatter[:set_scientific](true)
formatter[:set_powerlimits]((-1,1))
ax1[:yaxis][:set_major_formatter](formatter)
xlim(0,tmax_ferro_meson)
ax1[:set_zorder](ax1[:get_zorder]()+1) # plot this curve in front
ax1[:set_frame_on](false)
ax1[:set_ylim](-2.0e-5,6.8e-5)
ax1[:set_yticks]([0,1,2,3,4,5,6]*10.0^(-5))
ax2[:set_ylim](9,40)
ax1[:set_xlabel]("\$t J\$")
ax1[:set_ylabel]("\$-\\ln(\\lambda_0)\$", color="C0")
ax2[:set_ylabel]("\$-\\ln(\\lambda_{r\\geq 1})\$")
layout.nice_ticks_doubleaxis(ax1,ax2)
ax1[:tick_params](axis="y", labelcolor="C0")
# legend(loc = "best", numpoints=3, frameon=0, fancybox=0, columnspacing=1)
# ax1[:legend](loc = "upper left", numpoints=3, frameon=0, fancybox=0, framealpha=0, bbox_to_anchor=(0.11,1))
ax2[:legend](loc = "upper right", numpoints=3, frameon=0, fancybox=1, facecolor="white", ncol=3)
text(-0.14,1.02, "(b)", fontsize=25, ha="center",va="center", transform=ax1[:transAxes])
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_DQPT/figures/paper/Fig2b.pdf"))
# savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_DQPT/figures/spectrum_"*file_type*".pdf"))

## ratios:
plot_ratios(lambdas_ferro_meson,ind_max_ferro_meson,0)
figure(3)
ax = subplot(111)
xlim(0,tmax_ferro_meson)
ylim(1,3.1)
xlabel("\$t J\$")
ylabel("\$g_r / g_1\$")
text(-0.14,1.02, "(c)", fontsize=25, ha="center",va="center", transform=ax[:transAxes])
layout.nice_ticks()
legend(loc = "best", numpoints=3, frameon=0, fancybox=0, ncol=2, bbox_to_anchor=(0.95,0.95))
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_DQPT/figures/paper/Fig2c.pdf"))
# savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_DQPT/figures/ratios_"*file_type*".pdf"))

## rates:
plot_rates(rates_ferro_meson,ind_max_ferro_meson,0,true)
figure(4)
ax = subplot(111)
xlim(0,tmax_ferro_meson)
ylim(bottom=3e-6)
xlabel("\$t J\$")
ylabel("\$r_i\$")
text(-0.14,1.02, "(d)", fontsize=25, ha="center",va="center", transform=ax[:transAxes])
layout.nice_ticks()
legend(loc = "best", numpoints=3, frameon=0, fancybox=0, ncol=2)
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_DQPT/figures/paper/Fig2d.pdf"))
# savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_DQPT/figures/rates_"*file_type*".pdf"))

## inverse gaps:
plot_gaps(lambdas_ferro_meson,ind_max_ferro_meson,0)
figure(5)
xlim(0,tmax_ferro_meson)
xlabel("\$t J\$")
ylabel("\$1 / g_r\$")
layout.nice_ticks()
# savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_DQPT/figures/gaps_"*file_type*".pdf"))




##==============================================================================
##  para --> meson
## data:
file_type = "para_meson"
num_fig = 10
entropies_filename = "linearresponse/continuum_limit_DQPT/entropies_"*file_type
spectrum_filename = "linearresponse/continuum_limit_DQPT/spectrum_"*file_type
rates_filename = "linearresponse/continuum_limit_DQPT/rates_"*file_type

entropies_para_meson = readdlm(string(@__DIR__,"/data/"*entropies_filename*".txt"), skipstart=2)
lambdas_para_meson = readdlm(string(@__DIR__,"/data/"*spectrum_filename*".txt"), skipstart=2)
rates_para_meson = readdlm(string(@__DIR__,"/data/"*rates_filename*".txt"), skipstart=2)

tmax_para_meson = entropies_para_meson[end,1]
dt_para_meson = entropies_para_meson[2,1]-entropies_para_meson[1,1]
ind_max_para_meson = Int(round(tmax_para_meson/dt_para_meson))

## entropies:
plot_entropies(entropies_para_meson,ind_max_para_meson,num_fig)
figure(1+num_fig)
ax = subplot(111)
xlim(0,tmax_para_meson)
ylim(0,4.2)
xlabel("\$t J\$")
ylabel("\$S_{\\alpha}\$")
text(-0.14,1.02, "(e)", fontsize=25, ha="center",va="center", transform=ax[:transAxes])
layout.nice_ticks()
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_DQPT/figures/paper/Fig2e.pdf"))
# savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_DQPT/figures/entropies_"*file_type*".pdf"))

## spectrum:
plot_spectrum(lambdas_para_meson,ind_max_para_meson,num_fig,false)
figure(2+num_fig)
ax = subplot(111)
xlim(0,tmax_para_meson)
ylim(0,12)
xlabel("\$t J\$")
ylabel("\$-\\ln(\\lambda_r)\$")
text(-0.14,1.02, "(f)", fontsize=25, ha="center",va="center", transform=ax[:transAxes])
layout.nice_ticks()
legend(loc = "upper right", numpoints=3, frameon=0, facecolor="white", fancybox=1, ncol=3)
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_DQPT/figures/paper/Fig2f.pdf"))
# savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_DQPT/figures/spectrum_"*file_type*".pdf"))

## ratios:
plot_ratios(lambdas_para_meson,ind_max_para_meson,num_fig,true)
figure(3+num_fig)
ax = subplot(111)
xlim(0,tmax_para_meson)
ylim(1,100)
# ylim(bottom=1)
xlabel("\$t J\$")
ylabel("\$g_r / g_1\$")
text(-0.14,1.02, "(g)", fontsize=25, ha="center",va="center", transform=ax[:transAxes])
layout.nice_ticks()
# legend(loc = "upper right", numpoints=3, frameon=0, fancybox=0, ncol=3, columnspacing=1, bbox_to_anchor=(0.9,0.9))
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_DQPT/figures/paper/Fig2g.pdf"))
# savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_DQPT/figures/ratios_"*file_type*".pdf"))

## rates:
plot_rates(rates_para_meson,ind_max_para_meson,num_fig,false)
figure(4+num_fig)
ax = subplot(111)
xlim(0,tmax_para_meson)
ylim(0,4)
xlabel("\$t J\$")
ylabel("\$r_i\$")
text(-0.14,1.02, "(h)", fontsize=25, ha="center",va="center", transform=ax[:transAxes])
layout.nice_ticks()
# legend(loc = "best", numpoints=3, frameon=0, fancybox=0, ncol=3, columnspacing=1)
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_DQPT/figures/paper/Fig2h.pdf"))
# savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_DQPT/figures/rates_"*file_type*".pdf"))

## inverse gaps:
plot_gaps(lambdas_para_meson,ind_max_para_meson,num_fig)
figure(5+num_fig)
xlim(0,tmax_para_meson)
xlabel("\$t J\$")
ylabel("\$1 / g_r\$")
layout.nice_ticks()
# savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_DQPT/figures/gaps_"*file_type*".pdf"))






##==============================================================================
##  ferro --> E8
## data:
file_type = "ferro_E8"
num_fig = 20
entropies_filename = "linearresponse/continuum_limit_DQPT/entropies_"*file_type
spectrum_filename = "linearresponse/continuum_limit_DQPT/spectrum_"*file_type
rates_filename = "linearresponse/continuum_limit_DQPT/rates_"*file_type

entropies_ferro_E8 = readdlm(string(@__DIR__,"/data/"*entropies_filename*".txt"), skipstart=2)
lambdas_ferro_E8 = readdlm(string(@__DIR__,"/data/"*spectrum_filename*".txt"), skipstart=2)
rates_ferro_E8 = readdlm(string(@__DIR__,"/data/"*rates_filename*".txt"), skipstart=2)

tmax_ferro_E8 = entropies_ferro_E8[end,1]
dt_ferro_E8 = entropies_ferro_E8[2,1]-entropies_ferro_E8[1,1]
ind_max_ferro_E8 = Int(round(tmax_ferro_E8/dt_ferro_E8))

## entropies:
plot_entropies(entropies_ferro_E8,ind_max_ferro_E8,num_fig)
figure(1+num_fig)
ax = subplot(111)
xlim(0,tmax_ferro_E8)
ylim(0,1.5)
xlabel("\$t J\$")
ylabel("\$S_{\\alpha}\$")
text(-0.14,1.02, "(a)", fontsize=25, ha="center",va="center", transform=ax[:transAxes])
layout.nice_ticks()
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_DQPT/figures/paper/Fig4a.pdf"))
# savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_DQPT/figures/entropies_"*file_type*".pdf"))

## spectrum:
ax1,ax2 = plot_spectrum(lambdas_ferro_E8,ind_max_ferro_E8,num_fig)
figure(2+num_fig)
# ax1[:ticklabel_format](axis="y", style="scientific", scilimits=(-3,1), useOffset=true)
xlim(0,tmax_ferro_E8)
ax1[:set_zorder](ax1[:get_zorder]()+1) # plot this curve in front
ax1[:set_frame_on](false)
ax1[:set_ylim](0,0.29)
ax2[:set_ylim](3,8)
ax1[:set_xlabel]("\$t J\$")
ax1[:set_ylabel]("\$-\\ln(\\lambda_0)\$", color="C0")
ax2[:set_ylabel]("\$-\\ln(\\lambda_{r\\geq 1})\$")
layout.nice_ticks_doubleaxis(ax1,ax2)
ax1[:tick_params](axis="y", labelcolor="C0")
text(-0.14,1.02, "(b)", fontsize=25, ha="center",va="center", transform=ax2[:transAxes])
# ax1[:legend](loc = "upper left", numpoints=3, frameon=1, facecolor="white", fancybox=1, columnspacing=1)
# ax2[:legend](loc = "upper right", numpoints=3, frameon=1, facecolor="white", fancybox=1, ncol=3, columnspacing=1)
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_DQPT/figures/paper/Fig4b.pdf"))
# savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_DQPT/figures/spectrum_"*file_type*".pdf"))

## ratios:
plot_ratios(lambdas_ferro_E8,ind_max_ferro_E8,num_fig)
figure(3+num_fig)
ax = subplot(111)
xlim(0,tmax_ferro_E8)
ylim(1,3.1)
xlabel("\$t J\$")
ylabel("\$g_r / g_1\$")
text(-0.14,1.02, "(c)", fontsize=25, ha="center",va="center", transform=ax[:transAxes])
layout.nice_ticks()
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_DQPT/figures/paper/Fig4c.pdf"))
# savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_DQPT/figures/ratios_"*file_type*".pdf"))

## rates:
plot_rates(rates_ferro_E8,ind_max_ferro_E8,num_fig,false)
figure(4+num_fig)
ax = subplot(111)
xlim(0,tmax_ferro_E8)
ylim(0,4)
xlabel("\$t J\$")
ylabel("\$r_i\$")
layout.nice_ticks()
text(-0.14,1.02, "(d)", fontsize=25, ha="center",va="center", transform=ax[:transAxes])
# legend(loc = "best", numpoints=3, frameon=0, fancybox=0, ncol=3, columnspacing=1)
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_DQPT/figures/paper/Fig4d.pdf"))
# savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_DQPT/figures/rates_"*file_type*".pdf"))

## inverse gaps:
plot_gaps(lambdas_ferro_E8,ind_max_ferro_E8,num_fig)
figure(5+num_fig)
xlim(0,tmax_ferro_E8)
xlabel("\$t J\$")
ylabel("\$1 / g_r\$")
layout.nice_ticks()
# savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_DQPT/figures/gaps_"*file_type*".pdf"))





##==============================================================================
##  para --> E8
## data:
file_type = "para_E8"
num_fig = 30
entropies_filename = "linearresponse/continuum_limit_DQPT/entropies_"*file_type
spectrum_filename = "linearresponse/continuum_limit_DQPT/spectrum_"*file_type
rates_filename = "linearresponse/continuum_limit_DQPT/rates_"*file_type

entropies_para_E8 = readdlm(string(@__DIR__,"/data/"*entropies_filename*".txt"), skipstart=2)
lambdas_para_E8 = readdlm(string(@__DIR__,"/data/"*spectrum_filename*".txt"), skipstart=2)
rates_para_E8 = readdlm(string(@__DIR__,"/data/"*rates_filename*".txt"), skipstart=2)

tmax_para_E8 = 30
dt_para_E8 = entropies_para_E8[2,1]-entropies_para_E8[1,1]
ind_max_para_E8 = Int(round(tmax_para_E8/dt_para_E8))

## entropies:
plot_entropies(entropies_para_E8,ind_max_para_E8,num_fig)
figure(1+num_fig)
ax = subplot(111)
xlim(0,tmax_para_E8)
ylim(0,3.7)
xlabel("\$t J\$")
ylabel("\$S_{\\alpha}\$")
text(-0.14,1.02, "(e)", fontsize=25, ha="center",va="center", transform=ax[:transAxes])
layout.nice_ticks()
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_DQPT/figures/paper/Fig4e.pdf"))
# savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_DQPT/figures/entropies_"*file_type*".pdf"))

## spectrum:
plot_spectrum(lambdas_para_E8,ind_max_para_E8,num_fig,false)
figure(2+num_fig)
ax = subplot(111)
xlim(0,tmax_para_E8)
ylim(0,6)
xlabel("\$t J\$")
ylabel("\$-\\ln(\\lambda_r)\$")
text(-0.14,1.02, "(f)", fontsize=25, ha="center",va="center", transform=ax[:transAxes])
layout.nice_ticks()
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_DQPT/figures/paper/Fig4f.pdf"))
# savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_DQPT/figures/spectrum_"*file_type*".pdf"))

## ratios:
plot_ratios(lambdas_para_E8,ind_max_para_E8,num_fig)
figure(3+num_fig)
ax = subplot(111)
xlim(0,tmax_para_E8)
ylim(1,5.1)
xlabel("\$t J\$")
ylabel("\$g_r / g_1\$")
text(-0.14,1.02, "(g)", fontsize=25, ha="center",va="center", transform=ax[:transAxes])
layout.nice_ticks()
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_DQPT/figures/paper/Fig4g.pdf"))
# savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_DQPT/figures/ratios_"*file_type*".pdf"))

## rates:
plot_rates(rates_para_E8,ind_max_para_E8,num_fig,false)
figure(4+num_fig)
ax = subplot(111)
xlim(0,tmax_para_E8)
ylim(0,4)
xlabel("\$t J\$")
ylabel("\$r_i\$")
text(-0.14,1.02, "(h)", fontsize=25, ha="center",va="center", transform=ax[:transAxes])
layout.nice_ticks()
# legend(loc = "best", numpoints=3, frameon=0, fancybox=0, ncol=3, columnspacing=1)
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_DQPT/figures/paper/Fig4h.pdf"))
# savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_DQPT/figures/rates_"*file_type*".pdf"))

## inverse gaps:
plot_gaps(lambdas_para_E8,ind_max_para_E8,num_fig)
figure(5+num_fig)
xlim(0,tmax_para_E8)
xlabel("\$t J\$")
ylabel("\$1 / g_r\$")
layout.nice_ticks()
# savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_DQPT/figures/gaps_"*file_type*".pdf"))





##==============================================================================
##  para --> crit
## data:
file_type = "para_crit"
num_fig = 40
entropies_filename = "linearresponse/continuum_limit_DQPT/entropies_"*file_type
spectrum_filename = "linearresponse/continuum_limit_DQPT/spectrum_"*file_type
rates_filename = "linearresponse/continuum_limit_DQPT/rates_"*file_type

entropies_para_crit = readdlm(string(@__DIR__,"/data/"*entropies_filename*".txt"), skipstart=2)
lambdas_para_crit = readdlm(string(@__DIR__,"/data/"*spectrum_filename*".txt"), skipstart=2)
rates_para_crit = readdlm(string(@__DIR__,"/data/"*rates_filename*".txt"), skipstart=2)

tmax_para_crit = 10
dt_para_crit = entropies_para_crit[2,1]-entropies_para_crit[1,1]
ind_max_para_crit = Int(round(tmax_para_crit/dt_para_crit))

## entropies:
plot_entropies(entropies_para_crit,ind_max_para_crit,num_fig)
figure(1+num_fig)
ax = subplot(111)
xlim(0,tmax_para_crit)
ylim(0,3.5)
xlabel("\$t J\$")
ylabel("\$S_{\\alpha}\$")
layout.nice_ticks()
text(-0.14,1.02, "(a)", fontsize=25, ha="center",va="center", transform=ax[:transAxes])
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_DQPT/figures/entropies_"*file_type*".pdf"))

## spectrum:
plot_spectrum(lambdas_para_crit,ind_max_para_crit,num_fig,false)
figure(2+num_fig)
ax = subplot(111)
xlim(0,tmax_para_crit)
ylim(0,20)
ax = subplot(111)
ax[:set_yticks]([0,5,10,15,20])
xlabel("\$t J\$")
ylabel("\$-\\ln(\\lambda_r)\$")
text(-0.14,1.02, "(b)", fontsize=25, ha="center",va="center", transform=ax[:transAxes])
layout.nice_ticks()
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_DQPT/figures/spectrum_"*file_type*".pdf"))

## ratios:
plot_ratios(lambdas_para_crit,ind_max_para_crit,num_fig)
figure(3+num_fig)
ax = subplot(111)
xlim(0,tmax_para_crit)
# ylim(1,5.1)
xlabel("\$t J\$")
ylabel("\$g_r / g_1\$")
text(-0.14,1.02, "(c)", fontsize=25, ha="center",va="center", transform=ax[:transAxes])
layout.nice_ticks()
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_DQPT/figures/ratios_"*file_type*".pdf"))

## rates:
plot_rates(rates_para_crit,ind_max_para_crit,num_fig,false)
figure(4+num_fig)
ax = subplot(111)
xlim(0,tmax_para_crit)
ylim(0,4)
xlabel("\$t J\$")
ylabel("\$r_i\$")
text(-0.14,1.02, "(d)", fontsize=25, ha="center",va="center", transform=ax[:transAxes])
layout.nice_ticks()
# legend(loc = "best", numpoints=3, frameon=0, fancybox=0, ncol=3, columnspacing=1)
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_DQPT/figures/rates_"*file_type*".pdf"))

## inverse gaps:
plot_gaps(lambdas_para_crit,ind_max_para_crit,num_fig)
figure(5+num_fig)
xlim(0,tmax_para_crit)
xlabel("\$t J\$")
ylabel("\$1 / g_r\$")
layout.nice_ticks()
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_DQPT/figures/gaps_"*file_type*".pdf"))






##==============================================================================
##  para --> ferro
## data:
file_type = "para_ff"
num_fig = 50
entropies_filename = "linearresponse/continuum_limit_DQPT/entropies_"*file_type
spectrum_filename = "linearresponse/continuum_limit_DQPT/spectrum_"*file_type
rates_filename = "linearresponse/continuum_limit_DQPT/rates_"*file_type

entropies_para_ff = readdlm(string(@__DIR__,"/data/"*entropies_filename*".txt"), skipstart=2)
lambdas_para_ff = readdlm(string(@__DIR__,"/data/"*spectrum_filename*".txt"), skipstart=2)
rates_para_ff = readdlm(string(@__DIR__,"/data/"*rates_filename*".txt"), skipstart=2)

tmax_para_ff = 30
dt_para_ff = entropies_para_ff[2,1]-entropies_para_ff[1,1]
ind_max_para_ff = Int(round(tmax_para_ff/dt_para_ff))

## entropies:
plot_entropies(entropies_para_ff,ind_max_para_ff,num_fig)
figure(1+num_fig)
ax = subplot(111)
xlim(0,tmax_para_ff)
ylim(0,4.3)
xlabel("\$t J\$")
ylabel("\$S_{\\alpha}\$")
text(-0.14,1.02, "(e)", fontsize=25, ha="center",va="center", transform=ax[:transAxes])
layout.nice_ticks()
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_DQPT/figures/entropies_"*file_type*".pdf"))

## spectrum:
plot_spectrum(lambdas_para_ff,ind_max_para_ff,num_fig,false)
figure(2+num_fig)
ax = subplot(111)
xlim(0,tmax_para_ff)
ylim(0,20)
ax = subplot(111)
ax[:set_yticks]([0,5,10,15,20])
xlabel("\$t J\$")
ylabel("\$-\\ln(\\lambda_r)\$")
text(-0.14,1.02, "(f)", fontsize=25, ha="center",va="center", transform=ax[:transAxes])
layout.nice_ticks()
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_DQPT/figures/spectrum_"*file_type*".pdf"))

## ratios:
plot_ratios(lambdas_para_ff,ind_max_para_ff,num_fig,true)
figure(3+num_fig)
ax = subplot(111)
xlim(0,tmax_para_ff)
ylim(bottom=1)
xlabel("\$t J\$")
ylabel("\$g_r / g_1\$")
text(-0.14,1.02, "(g)", fontsize=25, ha="center",va="center", transform=ax[:transAxes])
layout.nice_ticks()
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_DQPT/figures/ratios_"*file_type*".pdf"))

## rates:
plot_rates(rates_para_ff,ind_max_para_ff,num_fig,false)
figure(4+num_fig)
ax = subplot(111)
xlim(0,tmax_para_ff)
ylim(0,4)
xlabel("\$t J\$")
ylabel("\$r_i\$")
text(-0.14,1.02, "(h)", fontsize=25, ha="center",va="center", transform=ax[:transAxes])
layout.nice_ticks()
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_DQPT/figures/rates_"*file_type*".pdf"))

## inverse gaps:
plot_gaps(lambdas_para_ff,ind_max_para_ff,num_fig)
figure(5+num_fig)
xlim(0,tmax_para_ff)
xlabel("\$t J\$")
ylabel("\$1 / g_r\$")
layout.nice_ticks()
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_DQPT/figures/gaps_"*file_type*".pdf"))





##==============================================================================
##  compare g_2/g_1 of all meson types
## data:

lambdas_ferro_meson = readdlm(string(@__DIR__,"/data/linearresponse/continuum_limit_DQPT/spectrum_ferro_meson.txt"), skipstart=2)
tmax_ferro_meson = 60
dt_ferro_meson = lambdas_ferro_meson[2,1]-lambdas_ferro_meson[1,1]
ind_max_ferro_meson = Int(round(tmax_ferro_meson/dt_ferro_meson))

lambdas_ferro_E8 = readdlm(string(@__DIR__,"/data/linearresponse/continuum_limit_DQPT/spectrum_ferro_E8.txt"), skipstart=2)
tmax_ferro_E8 = 60
dt_ferro_E8 = lambdas_ferro_E8[2,1]-lambdas_ferro_E8[1,1]
ind_max_ferro_E8 = Int(round(tmax_ferro_E8/dt_ferro_E8))

lambdas_para_meson = readdlm(string(@__DIR__,"/data/linearresponse/continuum_limit_DQPT/spectrum_para_meson.txt"), skipstart=2)
tmax_para_meson = 60
dt_para_meson = lambdas_para_meson[2,1]-lambdas_para_meson[1,1]
ind_max_para_meson = Int(round(tmax_para_meson/dt_para_meson))

lambdas_para_E8 = readdlm(string(@__DIR__,"/data/linearresponse/continuum_limit_DQPT/spectrum_para_E8.txt"), skipstart=2)
tmax_para_E8 = 30
dt_para_E8 = lambdas_para_E8[2,1]-lambdas_para_E8[1,1]
ind_max_para_E8 = Int(round(tmax_para_E8/dt_para_E8))

figure(6)
axhline(3, ls="--", c="grey",zorder=-1)
plot(lambdas_ferro_meson[1:ind_max_ferro_meson,1], g_r(lambdas_ferro_meson,2,ind_max_ferro_meson) ./ g_r(lambdas_ferro_meson,1,ind_max_ferro_meson), label="(1)") # label="ferro \$\\to\$ meson")
plot(lambdas_ferro_E8[1:ind_max_ferro_E8,1], g_r(lambdas_ferro_E8,2,ind_max_ferro_E8) ./ g_r(lambdas_ferro_E8,1,ind_max_ferro_E8), ls="--", label="(3)") # label="ferro \$\\to\$ E\$_8\$")
plot(lambdas_para_meson[1:ind_max_para_meson,1], g_r(lambdas_para_meson,2,ind_max_para_meson) ./ g_r(lambdas_para_meson,1,ind_max_para_meson), ls=":", label="(2)") # label="para \$\\to\$ meson")
plot(lambdas_para_E8[1:ind_max_para_E8,1], g_r(lambdas_para_E8,2,ind_max_para_E8) ./ g_r(lambdas_para_E8,1,ind_max_para_E8), ls="-.", label="(4)") # label="para \$\\to\$ E\$_8\$")
xlim(0,60)
ylim(1,6)
xlabel("\$t J\$")
ylabel("\$g_2 / g_1\$")
legend(loc = "best", numpoints=3, frameon=0, fancybox=0, ncol=2)
layout.nice_ticks_regular()
savefig(string(@__DIR__,"/data/linearresponse/continuum_limit_DQPT/figures/ratios_comparison.pdf"))








println("done: DQPT_analysis.jl")
# show()
;
