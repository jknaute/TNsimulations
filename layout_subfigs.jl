module layout

using PyPlot
using PyCall
@pyimport matplotlib.transforms as mpltrafo



### LAYOUT choices:

## for doubleplot:
function nice_ticks()
    linew = 2
    rc("font", size = 22) #fontsize of axis labels (numbers)
    rc("axes", labelsize = 25, lw = linew) #fontsize of axis labels (symbols)
    rc("lines", mew = 2, lw = linew, markeredgewidth = 2)
    rc("patch", ec = "k")
    rc("xtick.major", pad = 4)
    rc("ytick.major", pad = 4)
    rc("axes", labelpad = 2)
    rc("legend", fontsize=22, columnspacing=0.5)

    PyCall.PyDict(matplotlib["rcParams"])["mathtext.fontset"] = "cm"
    PyCall.PyDict(matplotlib["rcParams"])["mathtext.rm"] = "serif"
    PyCall.PyDict(matplotlib["rcParams"])["figure.figsize"] = [8.0, 6.0]

    ax = subplot(111)
    ax[:get_xaxis]()[:set_tick_params](direction="in", bottom=1, top=1)
    ax[:get_yaxis]()[:set_tick_params](direction="in", left=1, right=1)

    for l in ax[:get_xticklines]()
        l[:set_markersize](8)
        l[:set_markeredgewidth](2.0)
    end
    for l in ax[:get_yticklines]()
        l[:set_markersize](8)
        l[:set_markeredgewidth](2.0)
    end
    for l in ax[:yaxis][:get_minorticklines]()
        l[:set_markersize](4)
        l[:set_markeredgewidth](1.5)
    end
    for l in ax[:xaxis][:get_minorticklines]()
        l[:set_markersize](4)
        l[:set_markeredgewidth](1.5)
    end

    ax[:set_position](mpltrafo.Bbox([[0.14, 0.11], [0.89, 0.945]]))
end

## for normal plot:
function nice_ticks_regular()
    linew = 2
    rc("font", size = 18) #fontsize of axis labels (numbers)
    rc("axes", labelsize = 20, lw = linew) #fontsize of axis labels (symbols)
    rc("lines", mew = 2, lw = linew, markeredgewidth = 2)
    rc("patch", ec = "k")
    rc("xtick.major", pad = 7)
    rc("ytick.major", pad = 7)

    PyCall.PyDict(matplotlib["rcParams"])["mathtext.fontset"] = "cm"
    PyCall.PyDict(matplotlib["rcParams"])["mathtext.rm"] = "serif"
    PyCall.PyDict(matplotlib["rcParams"])["figure.figsize"] = [8.0, 6.0]

    ax = subplot(111)
    ax[:get_xaxis]()[:set_tick_params](direction="in", bottom=1, top=1)
    ax[:get_yaxis]()[:set_tick_params](direction="in", left=1, right=1)

    for l in ax[:get_xticklines]()
        l[:set_markersize](8)
        l[:set_markeredgewidth](2.0)
    end
    for l in ax[:get_yticklines]()
        l[:set_markersize](8)
        l[:set_markeredgewidth](2.0)
    end
    for l in ax[:yaxis][:get_minorticklines]()
        l[:set_markersize](4)
        l[:set_markeredgewidth](1.5)
    end
    for l in ax[:xaxis][:get_minorticklines]()
        l[:set_markersize](4)
        l[:set_markeredgewidth](1.5)
    end

    ax[:set_position](mpltrafo.Bbox([[0.16, 0.12], [0.95, 0.94]]))
end

## for doubleplot:
function nice_ticks_doubleaxis(ax1,ax2)
    linew = 2
    rc("font", size = 22) #fontsize of axis labels (numbers)
    rc("axes", labelsize = 25, lw = linew) #fontsize of axis labels (symbols)
    rc("lines", mew = 2, lw = linew, markeredgewidth = 2)
    rc("patch", ec = "k")
    rc("xtick.major", pad = 4)
    rc("ytick.major", pad = 4)
    rc("axes", labelpad = 2)
    rc("legend", fontsize=22, columnspacing=0.5)

    PyCall.PyDict(matplotlib["rcParams"])["mathtext.fontset"] = "cm"
    PyCall.PyDict(matplotlib["rcParams"])["mathtext.rm"] = "serif"
    PyCall.PyDict(matplotlib["rcParams"])["figure.figsize"] = [8.0, 6.0]

    ax1[:get_xaxis]()[:set_tick_params](direction="in", bottom=1, top=1)
    ax1[:get_yaxis]()[:set_tick_params](direction="in", left=1, right=0)
    ax2[:get_yaxis]()[:set_tick_params](direction="in", left=0, right=1)

    for l in ax1[:get_xticklines]()
        l[:set_markersize](8)
        l[:set_markeredgewidth](2.0)
    end
    for l in ax1[:xaxis][:get_minorticklines]()
        l[:set_markersize](4)
        l[:set_markeredgewidth](1.5)
    end

    for l in ax1[:get_yticklines]()
        l[:set_markersize](8)
        l[:set_markeredgewidth](2.0)
    end
    for l in ax1[:yaxis][:get_minorticklines]()
        l[:set_markersize](4)
        l[:set_markeredgewidth](1.5)
    end

    for l in ax2[:get_yticklines]()
        l[:set_markersize](8)
        l[:set_markeredgewidth](2.0)
    end
    for l in ax2[:yaxis][:get_minorticklines]()
        l[:set_markersize](4)
        l[:set_markeredgewidth](1.5)
    end

    ax1[:set_position](mpltrafo.Bbox([[0.14, 0.11], [0.89, 0.945]]))
    ax2[:set_position](mpltrafo.Bbox([[0.14, 0.11], [0.89, 0.945]]))
end





end
