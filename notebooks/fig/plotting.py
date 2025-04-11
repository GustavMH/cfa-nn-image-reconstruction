#!/usr/bin/env python3

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Patch

# FIGURE 3
def whitelevel_99(save_to=None):
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, axs = plt.subplots(1,3, layout='constrained')
    fig.set_size_inches(4, .8)

    bins = [i*0.02 for i in range(0,50)]
    data = np.load("99th-percent-whitelevel.npz")
    for ax, name in zip(axs, ["CelebA-HQ", "Imagenette", "EuroSAT"]):
        ax.hist(data[name], bins, rwidth=1, histtype="stepfilled")
        ax.set_xticks([0],[name], fontname="monospace")
        ax.set_yticks([])
        ax.set_xlim(0,1)
        plt.setp(ax.xaxis.get_majorticklabels(), ha="left", font="monospace" )

    if save_to:
        plt.savefig(save_to, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()
        plt.close()


# FIGURE 4
def msre_demosaic(save_to=None):
    # WARN This plot renders correctly with .savefig but not .show
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.rcParams['hatch.linewidth'] = .3
    fig, ax = plt.subplots(1,1, figsize=(4,2))

    colors = [mpl.cm.inferno(i) for i in range(0, 256, 256 // 5)][1:5]
    cfa_color, *light_colors = ["#5b99c6","#9579ac","#be87ac","#e29c94","#f3c77d"]

    data = np.load("msre-demosaicing.npz")
    xs, offsets = np.arange(0,3,1), np.arange(0,.8,.2)
    ax.bar(xs + 0.12, data["cfa"], width=0.08, color=cfa_color, hatch="//////")
    ax.bar(np.add.outer(xs, offsets+.28).flat, data["demosaic"].flat, width=0.08, color=colors)
    ax.bar(np.add.outer(xs, offsets+.36).flat, data["unet"].flat, width=0.08, color=light_colors, hatch="//////")

    custom_lines = list(np.array([
        Patch(facecolor=cfa_color,   label='Synthetic CFA'),
        Patch(facecolor=colors[0],   label='Bilinear'),
        Patch(facecolor=colors[1],   label='PPG'),
        Patch(facecolor=colors[2],   label='VNG  '),
        Patch(facecolor=colors[3],   label='AHD  '),
        Patch(facecolor='white',     label=''),
        Patch(facecolor='grey',      label='Demosaic'),
        Patch(facecolor="lightgrey", label="U-net", hatch="//////"),
        Patch(facecolor="white",     label="")
    ]).reshape(3,3).T.flat)

    ax.legend(handles=t_lines, bbox_to_anchor=(0, 1.02, 1, 0.2),
            fontsize=9, loc="lower left",
            mode="expand", borderaxespad=0, ncol=3)

    ax.set_xticks(xs, ["Imagenette", "EuroSAT", "CelebA-HQ"], font="monospace")
    ax.set_xlim(0,3)
    ax.set_ylabel('MSRE')
    ax.set_yscale("log")
    ax.grid(which='minor', alpha=.6)
    plt.grid(which='major', alpha=1)

    if save_to:
        plt.savefig(save_to, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()
        plt.close()

# FIGURE 5
def isp_msre():
    plt.style.use("seaborn-v0_8-darkgrid")

    fig, ax = plt.subplots(1,1, figsize=(4,3), layout="tight")
    colors = [mpl.cm.inferno(i) for i in range(0, 256, 256 // 5)]

    data = {
        "unet": {
            "Synthetic CFA":   [2.48621875e-04, 1.56603595e-04, 9.34193022e-05],
            "Synthetic RGB":   [2.72279273e-04, 1.78603875e-04, 9.64219652e-05],
            "+ Median filter": [0.00047144, 0.0002417 , 0.00012527],
            "+ White balance": [0.00166321, 0.0014517 , 0.00066913],
            "+ White level":   [0.00183745, 0.00309079, 0.00098799]
        },
        "demosaic": {
            "Synthetic RGB":   [0.00104938, 0.00028977, 0.00014757],
            "+ Median filter": [0.00073673, 0.00022674, 0.00010911],
            "+ White balance": [0.00804051, 0.00597073, 0.01455077],
            "+ White level":   [0.00377549, 0.0983795 , 0.00196165]
        }
    }

    xs = np.arange(3)
    for n, (key, color) in enumerate(zip(data["unet"].keys(), colors)):
        unet_pos = xs + n*.2 + .15
        if n == 0:
            ax.scatter(unet_pos, data["unet"][key], s=50, marker="1")
        else:
            ax.scatter(unet_pos, data["unet"][key], s=50, marker="1", color=color)
            x_pos = xs + n*.2 + .05
            ax.scatter(x_pos, data["demosaic"][key], color=color, s=25)

    ax.set_xticks(xs, ["Imagenette", "EuroSAT", "CelebA-HQ"], font="monospace")
    ax.set_xticks(np.arange(0,3,1/5), minor=True)

    plt.setp(ax.xaxis.get_majorticklabels(), ha="left")

    custom_lines = [
        Patch(facecolor=None,      label="Synthetic CFA"),
        Patch(facecolor=colors[1], label="Synthetic RGB"),
        Patch(facecolor=colors[2], label="+ Median filter"),
        Patch(facecolor=colors[3], label="+ White balance"),
        Patch(facecolor=colors[4], label="+ White level"),
        ax.scatter([0],[0], marker="o", color="black", label="Demosaic"),
        ax.scatter([0],[0], marker="1", color="black", label="U-net denoising")
    ]

    plt.legend(
        handles=custom_lines,
        labelspacing=0.05,
        markerscale=1.2,
        frameon=True,
        facecolor="white",
        edgecolor="white",
        framealpha=1,
        fontsize=9,
        loc='upper left'
    )

    ax.set_xlim(0,3)
    ax.set_ylabel('MSRE')
    ax.set_yscale("log", base=10)
    ax.set_ylim(0.00006,0.45)
    ax.grid(which='minor', alpha=.6)

    if save_to:
        plt.savefig(save_to, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()
        plt.close()

# FIGURE 6
def noise():
    colors = [mpl.cm.inferno(i) for i in range(0, 256, 256 // 5)]
    colors = [colors[2], "tab:blue", colors[4]]
    fig, ax = plt.subplots(1,1,sharex=True,sharey=True, figsize=(4.5, 1.5))

    data = {
        ("Gaussian",      "RGB\nBaseline"):            [0.00051499, 0.00034514, 0.00027454, 0.00027881, 0.00026266, 0.00027084],
        ("Gaussian",      "Synthetic\nCFA"):           [0.00070235, 0.00050858, 0.00052158, 0.00048302, 0.00048818, 0.00047803],
        ("Gaussian",      "Processed\nSynthetic RGB"): [0.00208503, 0.0020265 , 0.0020786 , 0.00202847, 0.00201781, 0.00202827],
        ("Speckle",       "RGB\nBaseline"):            [0.00033135, 0.00030073, 0.00026953, 0.0002686 , 0.0002947 , 0.00025103],
        ("Speckle",       "Synthetic\nCFA"):           [0.00052451, 0.0005018 , 0.00048334, 0.00045706, 0.00047914, 0.00045365],
        ("Speckle",       "Processed\nSynthetic RGB"): [0.00203969, 0.00203843, 0.00204255, 0.00201996, 0.00203276, 0.00205119],
        ("Salt & Pepper", "RGB\nBaseline"):            [0.00129703, 0.0011405 , 0.00098233, 0.00074213, 0.00059676, 0.00040755],
        ("Salt & Pepper", "Synthetic\nCFA"):           [0.00150531, 0.00125095, 0.00109906, 0.00089852, 0.00073943, 0.00061555],
        ("Salt & Pepper", "Processed\nSynthetic RGB"): [0.00233864, 0.00236401, 0.00240565, 0.00229095, 0.00219994, 0.00216638]
    }

    formats = ["RGB\nBaseline", "Synthetic\nCFA", "Processed\nSynthetic RGB"]
    noises  = ["Gaussian", "Speckle", "Salt & Pepper"]

    for n, noise in enumerate(noises):
        for color_idx, form in enumerate(formats):
            us = np.array(data[(noise, form)])
            xs = np.ones_like(us)*n + .2 + color_idx*.3
            ax.scatter(xs, us, color=colors[color_idx], alpha=[.83, .67, .5, 0.33, 0.17, 0.])

    ax.set_xticks(range(3), noises, font="monospace")
    plt.setp(ax.xaxis.get_majorticklabels(), ha="left")
    legend = [ax.scatter([],[], label=l, color=c) for c, l in zip(colors, formats)]

    plt.legend(
        handles=legend,
        bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
        fontsize=9,
        mode="expand", borderaxespad=0, ncol=3, handletextpad=0.1
    )
    plt.ylabel('MSRE')
    plt.grid(which='major', alpha=1)
    plt.xlim(0,3)

    if save_to:
        plt.savefig(save_to, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()
        plt.close()

# FIGURE 7
def transfer():
    ds_names = ["Imagenette", "EuroSAT", "CelebA-HQ"]
    means = np.load("transfer.npy")
    µmax, µmin = means.max(), means.min()

    fig, axs = plt.subplots(1,3, sharey=True)
    for ax, µs, form in zip(axs, means, forms_names):
        ax.grid(False)
        img = ax.imshow(µs.T, vmin=0, vmax=µmax, cmap=mpl.colormaps["inferno"])
        ax.set_yticks(np.arange(len(ds)), labels=[f"{s}\nmodel" for s in ds_names])
        ax.set_xticks(np.arange(len(ds)), labels=ds_names)
        ax.set_title(form)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        for i in range(3):
            for j in range(3):
                µ = np.round(µs.T[i, j], 3)
                text = ax.text(j, i, µ,
                            ha="center", va="center", color="w" if µ < 0.02 else "black")

    if save_to:
        plt.savefig(save_to, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()
        plt.close()
