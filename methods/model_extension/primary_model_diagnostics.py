#!/usr/bin/env python3
"""Descriptive calibration and count sensitivity on fixed held-out forecasts."""
from pathlib import Path
import os,json
import numpy as np,pandas as pd
from scipy.stats import spearmanr
R=Path(os.environ['DDR_EXTENSION_OUTPUT']).resolve()
d=pd.read_csv(R/'curated_all_outer_predictions.csv',float_precision='round_trip');main='state_curated_TP53_Gem_pharmacology__cross_entropy';g=d[d.model.eq(main)].copy();g['absolute_error']=abs(g.observed-g.predicted)
rows=[]
for drug,a in [('all',g)]+list(g.groupby('drug')):
 slope,intercept=np.polyfit(a.predicted,a.observed,1)
 rows.append(dict(drug=drug,n_observations=len(a),n_lines=a.cell_line.nunique(),treated_n_min=int(a.treated_n_cells.min()),treated_n_max=int(a.treated_n_cells.max()),prediction_min=a.predicted.min(),prediction_max=a.predicted.max(),calibration_intercept=intercept,calibration_slope=slope,unweighted_mae=a.absolute_error.mean(),cell_count_weighted_mae=np.average(a.absolute_error,weights=a.treated_n_cells),count_absolute_error_spearman=spearmanr(a.treated_n_cells,a.absolute_error).statistic))
pd.DataFrame(rows).to_csv(R/'primary_model_diagnostics.csv',index=False)
# Fixed count bins are descriptive; all observations remain in the primary analysis.
g['treated_count_bin']=pd.cut(g.treated_n_cells,[0,49,99,199,399,np.inf],labels=['1-49','50-99','100-199','200-399','400+'])
g.groupby('treated_count_bin',observed=True).agg(n_observations=('observed','size'),n_lines=('cell_line','nunique'),min_treated_n=('treated_n_cells','min'),max_treated_n=('treated_n_cells','max'),mae=('absolute_error','mean')).to_csv(R/'primary_model_count_strata.csv')
selection=pd.read_csv(Path(os.environ['DDR_EXTENSION_RECORDS'])/'curated_biology_selection_by_fold.csv')
selection.groupby(['criterion','chosen_family']).size().rename('n_outer_folds').reset_index().to_csv(R/'primary_selection_counts.csv',index=False)
