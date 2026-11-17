#!/usr/bin/env python3
"""Render the main and supplementary figures from fixed scientific inputs."""
from pathlib import Path
import argparse
import os
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'ddr_figure_cache'))

import matplotlib as mpl
from cell_style import set_style
from figure12 import prediction_figure, biological_figure
from figure34 import fig3, fig4
import supplementary_figures as supplementary

# Shared numerical API used by the full reproduction entry point.
classifier_ci = supplementary.classifier_ci


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--only',nargs='+',choices=['1','2','3','4','context','imaging','classifier'])
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    chosen=set(args.only or ['1','2','3','4','context','imaging','classifier'])
    # Supplementary graphics retain their original style and scientific content.
    mpl.rcdefaults();supplementary.style()
    for name,draw in [('context',supplementary.context_figure),
                      ('imaging',supplementary.imaging_supplement),
                      ('classifier',supplementary.classifier_figure)]:
        if name in chosen:draw(args.data,args.output)
    set_style()
    for name,draw in [('1',prediction_figure),('2',biological_figure),('3',fig3),('4',fig4)]:
        if name in chosen:draw(args.data,args.output)
    print('Rendered',', '.join(sorted(chosen)),'from supplied scientific analysis records.')


if __name__=='__main__':main()
