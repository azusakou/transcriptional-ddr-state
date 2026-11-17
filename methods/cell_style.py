"""Shared scientific figure typography and vector exports."""
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

MM = 1 / 25.4
INK = '#202124'
GRAY = '#8B929A'
LIGHT = '#E5E7EB'
NAVY = '#285A79'
TEAL = '#258D8A'
CORAL = '#D35E60'
PURPLE = '#8571A6'
GOLD = '#D3A14A'
DRUGS = ['Gemcitabine', 'Idasanutlin', 'Prexasertib', 'Bortezomib']
COLORS = dict(zip(DRUGS, [NAVY, CORAL, PURPLE, GOLD]))
PRIMARY = 'state_curated_TP53_Gem_pharmacology__cross_entropy'
BIOLOGY = 'curated_TP53_Gem_pharmacology_only__cross_entropy'
GENOTYPE = 'curated_TP53_all_regimens__cross_entropy'

def set_style():
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 8,
        'axes.labelsize': 8,
        'axes.titlesize': 8.5,
        'xtick.labelsize': 7.3,
        'ytick.labelsize': 7.3,
        'legend.fontsize': 7.2,
        'axes.linewidth': .65,
        'axes.edgecolor': INK,
        'axes.labelcolor': INK,
        'text.color': INK,
        'xtick.color': INK,
        'ytick.color': INK,
        'xtick.major.width': .65,
        'ytick.major.width': .65,
        'xtick.major.size': 2.7,
        'ytick.major.size': 2.7,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': False,
        'legend.frameon': False,
        'legend.handlelength': 1.4,
        'legend.handletextpad': .45,
        'legend.borderaxespad': .2,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
        'savefig.facecolor': 'white',
        'figure.facecolor': 'white',
        'figure.dpi': 130,
        'mathtext.fontset': 'dejavusans',
    })

def new_figure(height=165, width=180):
    return plt.figure(figsize=(width*MM, height*MM))

def panel(ax, letter, title=None, x=-.13, y=1.075):
    ax.text(x,y,letter.upper(),transform=ax.transAxes,ha='left',va='bottom',
            fontsize=11,fontweight='bold',color=INK,clip_on=False)
    if title:
        ax.set_title(title,loc='left',pad=7,fontweight='normal')

def source(frame, name, out):
    p=Path(out)/'source_data';p.mkdir(parents=True,exist_ok=True)
    frame.to_csv(p/(name+'.csv'),index=False,float_format='%.17g')

def save(fig, name, out):
    p=Path(out);p.mkdir(parents=True,exist_ok=True)
    for ext,opts in [('pdf',{}),('svg',{}),('png',{'dpi':400})]:
        fig.savefig(p/(name+'.'+ext),facecolor='white',**opts)
    plt.close(fig)

set_style()
