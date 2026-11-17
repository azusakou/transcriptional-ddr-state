#!/usr/bin/env python3
"""Draw publication figures from the archived held-out and same-cell assay data.

Inputs are the scientific analysis directories produced by reproduce.py. No
model is fitted or selected by this renderer. Optional forecast results can be
supplied separately after their grouped validation has been completed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'ddr_figure_cache'))

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

MM = 1 / 25.4
NAVY = '#285A76'
TEAL = '#568F81'
WARM = '#BD8350'
LIGHT = '#D6DBDF'
GRAY = '#98A0A8'
INK = '#26313A'
DRUGS = ['Gemcitabine', 'Idasanutlin', 'Prexasertib', 'Bortezomib']
COLORS = dict(zip(DRUGS, ['#537C9B', '#568F78', '#9C7FA8', '#BF8B58']))
PRIMARY = 'state_curated_TP53_Gem_pharmacology__cross_entropy'
BIOLOGY = 'curated_TP53_Gem_pharmacology_only__cross_entropy'
GENOTYPE = 'curated_TP53_all_regimens__cross_entropy'
METHODS = [
    ('drug_only', 'Drug + time'),
    ('baseline_only', 'Vehicle state only'),
    ('cycle_only_plus_drug', 'Cycle + drug/time'),
    ('no_DDR_TP53_plus_drug', 'Cycle/stress + drug/time'),
    ('ddr_fraction_additive', 'DDR fraction + drug/time'),
    ('parsimonious_additive', 'Four-feature model'),
]


def style():
    mpl.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 8,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
        'axes.linewidth': .6, 'xtick.major.width': .6, 'ytick.major.width': .6,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'legend.frameon': False, 'axes.edgecolor': INK, 'text.color': INK,
        'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK,
    })


def load(path):
    path=Path(path)
    if not path.exists() and path.suffix=='.csv' and path.with_suffix('.csv.gz').exists():
        path=path.with_suffix('.csv.gz')
    return pd.read_csv(path, float_precision='round_trip')


def subset(frame, **selection):
    for key, value in selection.items():
        if key in frame:
            frame = frame[frame[key].eq(value)]
    return frame.copy()


def unique(frame, **selection):
    frame = subset(frame, **selection)
    assert len(frame) == 1, (selection, len(frame))
    return frame.iloc[0]


def label(ax, letter, title, x=-.17, y=1.08):
    ax.set_title(title, loc='left', pad=10, weight='normal')
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10,
            fontweight='bold', va='bottom', ha='left')


def save(fig, name, output):
    # The PDF/SVG keep editable vector text. PNG is a review preview only.
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer();width,height=fig.canvas.get_width_height()
    outside_ticks=set()
    for ax in fig.axes:
        for locations,texts,bounds in [(ax.get_xticks(),ax.get_xticklabels(),ax.get_xlim()),
                                       (ax.get_yticks(),ax.get_yticklabels(),ax.get_ylim())]:
            lo,hi=sorted(bounds)
            outside_ticks.update(text for value,text in zip(locations,texts) if value<lo or value>hi)
    outside=[]
    for artist in fig.findobj(mpl.text.Text):
        if not artist.get_visible() or not artist.get_text().strip() or artist in outside_ticks:
            continue
        bb=artist.get_window_extent(renderer)
        if bb.x0 < -.5 or bb.y0 < -.5 or bb.x1 > width+.5 or bb.y1 > height+.5:
            outside.append(artist.get_text().replace('\n',' / '))
    if outside:
        raise ValueError((name,'Text outside the final figure bounds',outside))
    fig.savefig(output / (name + '.pdf'))
    fig.savefig(output / (name + '.svg'))
    fig.savefig(output / (name + '.png'), dpi=300)
    plt.close(fig)


def source(frame, name, output):
    (output / 'source_data').mkdir(exist_ok=True)
    frame.to_csv(output / 'source_data' / (name + '.csv'), index=False,
                 float_format='%.17g')


def figure1(data, output, forecast_root, primary, extension):
    pred=load(extension/'curated_all_outer_predictions.csv')
    perf=load(extension/'curated_all_performance.csv')
    boot=load(extension/'curated_biology_line_bootstrap.csv')
    selected = pred[pred.model.eq(primary)].copy()
    assert len(selected) == 243 and selected.cell_line.nunique() == 84
    methods=[('drug_only','Drug + time'),('ddr_fraction_additive','DDR tail + drug/time'),
             ('parsimonious_additive','Four states + drug/time'),
             (GENOTYPE,'TP53 context + drug/time'),
             (BIOLOGY,'Biology + drug/time'),(primary,'State + known biology')]
    fig = plt.figure(figsize=(183*MM, 183*MM))
    gs = fig.add_gridspec(3, 4, height_ratios=[.66, 2.4, 1.95],
                          hspace=.82, wspace=.62, left=.095, right=.98,
                          bottom=.105, top=.96)
    a = fig.add_subplot(gs[0, :]);a.set_axis_off()
    # A schematic describes the measured populations, not unobserved cell fates.
    boxes = [(.01, .1, .18, .69, 'Matched vehicle cells\nExperiment · time\nCell line'),
             (.225, .1, .255, .69, 'RNA state + known biology\nDDR · p53 · cycle · stress\nTP53 · Gem metabolism'),
             (.515, .1, .24, .69, 'Grouped model evaluation\nTrain/test cell lines separated'),
             (.790, .1, .200, .69, 'Predicted treated\nDDR-high fraction')]
    for i, (x, y, w, h, text) in enumerate(boxes):
        a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.008,rounding_size=.01',
                                  ec=LIGHT, fc='#F4F6F7', lw=.7, clip_on=False))
        a.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=7)
        if i < len(boxes)-1:
            a.add_patch(FancyArrowPatch((x+w+.008,y+h/2),(boxes[i+1][0]-.008,y+h/2),
                                       arrowstyle='-|>',mutation_scale=8,color=GRAY,lw=.7))
    a.text(-.055, 1.04, 'a', transform=a.transAxes, fontweight='bold', fontsize=10)
    a.text(.5,-.22,'243 matched line–drug–time populations · 84 cell lines · nested grouped validation',
           transform=a.transAxes,ha='center',va='top',fontsize=7)
    b = fig.add_subplot(gs[1, :2]);c = fig.add_subplot(gs[1, 2:])
    table = []
    for y,(key,name) in enumerate(methods):
        row = unique(perf,model=key,drug='all')
        ci = unique(boot,model=key,drug='all',reference_model='none',statistic='r2')
        color = NAVY if key == primary else GRAY
        b.plot([ci.ci_low,ci.ci_high],[y,y],color=color,lw=1.15,zorder=2)
        b.scatter(row.r2,y,s=27 if key==primary else 20,color=color,zorder=3)
        b.text(.90,y,f'{row.r2:.3f}',va='center',ha='right',fontsize=7,color=color)
        table.append(dict(model=key,label=name,r2=row.r2,ci_low=ci.ci_low,ci_high=ci.ci_high,mae=row.mae))
    b.axvline(0,color=LIGHT,lw=.7)
    b.set(yticks=np.arange(len(methods)),yticklabels=[v for _,v in methods],
          ylim=(len(methods)-.55,-.55),xlim=(-.04,.92),xlabel='Held-line $R^2$ (95% line-bootstrap CI)')
    pos=b.get_position()
    b.set_position([pos.x0+.075,pos.y0,pos.width-.075,pos.height])
    label(b,'b','Prediction with biological context',x=-.24)
    state_delta=unique(boot,model=primary,drug='all',reference_model=BIOLOGY,statistic='delta_r2')
    b.text(-.02,-.25,f'Added-state Δ$R^2$={state_delta.estimate:.3f}'+chr(10)+f'(95% CI {state_delta.ci_low:.3f} to {state_delta.ci_high:.3f})',
           transform=b.transAxes,fontsize=7,va='top')
    gains=[]
    for y,drug in enumerate(DRUGS):
        for reference,offset,marker,filled in [('drug_only',-.20,'o',True),('ddr_fraction_additive',0,'o',False),(BIOLOGY,.20,'s',True)]:
            ci=unique(boot,model=primary,drug=drug,reference_model=reference,statistic='mae_improvement')
            c.plot([ci.ci_low,ci.ci_high],[y+offset,y+offset],color=COLORS[drug],lw=1.1)
            c.scatter(ci.estimate,y+offset,s=26,facecolor=COLORS[drug] if filled else 'white',
                      edgecolor=COLORS[drug],lw=.9,zorder=3,marker=marker)
            gains.append(dict(drug=drug,reference_model=reference,mae_improvement=ci.estimate,
                              ci_low=ci.ci_low,ci_high=ci.ci_high))
    c.axvline(0,color=GRAY,lw=.7,ls='--')
    c.set(yticks=np.arange(4),yticklabels=DRUGS,ylim=(3.55,-.55),
          xlabel='Paired MAE improvement')
    c.legend(handles=[Line2D([],[],marker='o',color=INK,ls='',label='Versus drug/time'),
                      Line2D([],[],marker='o',mfc='white',mec=INK,ls='',label='Versus DDR tail + drug/time'),
                      Line2D([],[],marker='s',color=INK,ls='',label='Versus known biology + drug/time')],
             loc='upper left',bbox_to_anchor=(-.05,-.23),fontsize=7,handletextpad=.4)
    label(c,'c','Improvement within each drug',x=-.22)
    axes=[]
    for i,drug in enumerate(DRUGS):
        ax=fig.add_subplot(gs[2,i]);axes.append(ax)
        q=selected[selected.drug.eq(drug)]
        lim=1.0 if drug!='Bortezomib' else float(np.ceil(max(q.observed.max(),q.predicted.max())*10)/10)
        lim=max(lim,.1)
        ax.plot([0,lim],[0,lim],color=GRAY,lw=.65,ls='--',zorder=0)
        ax.scatter(q.observed,q.predicted,s=11,color=COLORS[drug],alpha=.78,edgecolors='white',linewidths=.2)
        m=unique(perf,model=primary,drug=drug)
        ax.set(xlim=(-.025*lim,1.025*lim),ylim=(-.025*lim,1.025*lim),
               xlabel='Observed fraction',ylabel='Predicted fraction' if i==0 else '')
        ax.set_aspect('equal',adjustable='box')
        ax.text(.04,.96,f'{len(q)} populations\n{q.cell_line.nunique()} cell lines\n$R^2$={m.r2:.3f}\nMAE={m.mae:.3f}',
                transform=ax.transAxes,va='top',fontsize=7)
        ax.set_xticks([0,lim/2,lim]);ax.set_yticks([0,lim/2,lim])
        label(ax,chr(ord('d')+i),drug,x=-.23,y=1.07)
    fig.text(.5,.035,'Population fractions; bortezomib uses a smaller axis range.',ha='center',fontsize=7)
    source(pd.DataFrame(table),'Fig1_model_comparisons',output)
    source(pd.DataFrame([state_delta]),'Fig1_added_state_increment',output)
    source(pd.DataFrame(gains),'Fig1_paired_drug_gains',output)
    source(selected,'Fig1_held_line_predictions',output)
    save(fig,'Fig1_compendium_and_prediction',output)


def classifier_ci(data, output, recompute=False):
    path=data/'classifier/held_cell_line_uncertainty.csv'
    if path.exists() and not recompute:return load(path)
    # Reproduce the manuscript's fixed-prediction line bootstrap if it is not
    # already present in an analysis output. No classifier is refitted here.
    from sklearn.metrics import average_precision_score,roc_auc_score
    df=load(data/'classifier/held_cell_line_McFarland_predictions.csv.gz')
    groups=[g.index.to_numpy() for _,g in df.groupby('cell_line',sort=True)]
    y=df.observed.to_numpy();keys=['QC','QC_cellcycle','biology_QC_cellcycle']
    est={m:{k:[] for k in keys} for m in ['AP','AUROC']};rng=np.random.default_rng(20261117)
    for _ in range(1000):
        ix=np.concatenate([groups[j] for j in rng.integers(0,len(groups),len(groups))])
        for key in keys:
            pp=df[key].to_numpy()[ix]
            est['AP'][key].append(average_precision_score(y[ix],pp))
            est['AUROC'][key].append(roc_auc_score(y[ix],pp))
    out=[]
    for metric,func in [('AP',average_precision_score),('AUROC',roc_auc_score)]:
        for key in keys:
            lo,hi=np.quantile(est[metric][key],[.025,.975])
            out.append(dict(metric=metric,comparison=key,estimate=func(y,df[key]),
                            lower95=lo,upper95=hi,n_cell_lines=len(groups),bootstrap_resamples=1000))
        for key in keys[:2]:
            vals=np.array(est[metric]['biology_QC_cellcycle'])-np.array(est[metric][key]);lo,hi=np.quantile(vals,[.025,.975])
            out.append(dict(metric=metric,comparison='biology_QC_cellcycle minus '+key,
                            estimate=func(y,df.biology_QC_cellcycle)-func(y,df[key]),
                            lower95=lo,upper95=hi,n_cell_lines=len(groups),bootstrap_resamples=1000))
    return pd.DataFrame(out)


def classifier_figure(data,output):
    fig=plt.figure(figsize=(183*MM,155*MM))
    gs=fig.add_gridspec(2,2,left=.15,right=.97,top=.94,bottom=.11,hspace=.70,wspace=.64)
    a=fig.add_subplot(gs[0,0]);b=fig.add_subplot(gs[0,1]);c=fig.add_subplot(gs[1,0]);d=fig.add_subplot(gs[1,1])
    tasks=load(data/'classifier/leave_task_out_task_metrics.csv')
    tasks['drug']=tasks.task_name.str.split('__').str[-1]
    base=tasks[tasks.feature_set.eq('QC_cellcycle')].set_index('drug')
    full=tasks[tasks.feature_set.eq('biology_QC_cellcycle')].set_index('drug')
    order=DRUGS+['Nutlin','SAHA'];colors={**COLORS,'Nutlin':TEAL,'SAHA':GRAY}
    for i,key in enumerate(order):
        a.plot([base.loc[key,'AP'],full.loc[key,'AP']],[i,i],color=colors[key],lw=1.2)
        a.scatter(base.loc[key,'AP'],i,facecolor='white',edgecolor=colors[key],s=23,lw=.9,zorder=3)
        a.scatter(full.loc[key,'AP'],i,color=colors[key],s=27,zorder=4)
    a.set(yticks=np.arange(6),yticklabels=order,ylim=(5.55,-.55),xlim=(0,.66),xlabel='Held-task average precision')
    a.legend(handles=[Line2D([],[],marker='o',mfc='white',mec=NAVY,ls='',label='QC + context + cycle'),
                      Line2D([],[],marker='o',color=NAVY,ls='',label='+ stress proxies')],
             loc='lower right',bbox_to_anchor=(1.02,-.01),fontsize=7,handletextpad=.4)
    label(a,'a','Stress signal across compound tasks',x=-.31)
    ci=classifier_ci(data,output)
    keys=['QC','QC_cellcycle','biology_QC_cellcycle'];names=['QC + context','+ cycle / E2F','+ stress proxies']
    for ax,metric,xlim,letter,title in [(b,'AP',(.1,.50),'b','Precision across held-out cell lines'),
                                      (d,'AUROC',(.45,.86),'d','Discrimination across cell lines')]:
        for i,key in enumerate(keys):
            z=unique(ci,metric=metric,comparison=key);col=NAVY if i==2 else GRAY
            ax.plot([z.lower95,z.upper95],[i,i],color=col,lw=1.2)
            ax.scatter(z.estimate,i,s=28,color=col,zorder=3)
            ax.text(xlim[1]-.01,i,f'{z.estimate:.3f}',ha='right',va='center',fontsize=7,color=col)
        ax.set(yticks=np.arange(3),yticklabels=names,ylim=(2.7,-.6),xlim=xlim,
               xlabel='Held-cell-line average precision' if metric=='AP' else 'Held-cell-line AUROC')
        delta=unique(ci,metric=metric,comparison='biology_QC_cellcycle minus QC_cellcycle')
        ax.text(.03,.02,f'Paired Δ{metric}={delta.estimate:.3f} (95% CI {delta.lower95:.3f}–{delta.upper95:.3f})'+chr(10)+'93 cell lines; 23,704 cells',
                transform=ax.transAxes,fontsize=7,va='bottom')
        label(ax,letter,title,x=-.31)
    importance=load(data/'classifier/grouped_permutation_importance_by_task.csv')
    groups=[('QC_joint','QC'),('cellcycle_programmes_joint','Cycle / E2F'),('stress_programmes_joint','Stress proxies')]
    for y,(key,_) in enumerate(groups):
        q=importance[importance.group.eq(key)]
        c.scatter(q.AP_drop,np.full(len(q),y)+np.linspace(-.13,.13,len(q)),color=GRAY,s=20,zorder=2)
        c.scatter(q.AP_drop.mean(),y,color=NAVY,s=35,marker='D',edgecolor='white',lw=.4,zorder=3)
    c.axvline(0,color=LIGHT,lw=.7)
    c.set(yticks=np.arange(3),yticklabels=[v for _,v in groups],ylim=(2.65,-.6),xlabel='AP drop after joint permutation')
    c.text(.03,.02,'Small points: six tasks'+chr(10)+'Diamonds: equal-task means',transform=c.transAxes,fontsize=7,va='bottom')
    label(c,'c','Contribution of programme groups',x=-.31)
    source(tasks,'FigS3_task_classification',output)
    source(ci,'FigS3_held_line_classification',output)
    source(importance,'FigS3_joint_permutation',output)
    save(fig,'Supplementary_FigS3',output)


def mechanistic_figure(data,output,extension):
    """Display the same curated biology-informed family across all drug panels."""
    pred=load(extension/'curated_all_outer_predictions.csv')
    perf=load(extension/'curated_all_performance.csv')
    boot=load(extension/'curated_biology_line_bootstrap.csv')
    fig=plt.figure(figsize=(183*MM,172*MM))
    gs=fig.add_gridspec(2,2,left=.15,right=.97,top=.94,bottom=.11,hspace=.75,wspace=.65)
    a=fig.add_subplot(gs[0,0]);b=fig.add_subplot(gs[0,1]);c=fig.add_subplot(gs[1,0]);d=fig.add_subplot(gs[1,1])
    gains=[]
    for y,drug in enumerate(DRUGS):
        for reference,offset,filled in [('parsimonious_additive',-.12,True),(GENOTYPE,.12,False)]:
            z=unique(boot,drug=drug,model=PRIMARY,reference_model=reference,statistic='mae_improvement')
            a.plot([z.ci_low,z.ci_high],[y+offset]*2,color=COLORS[drug],lw=1.1)
            a.scatter(z.estimate,y+offset,s=26,facecolor=COLORS[drug] if filled else 'white',
                      edgecolor=COLORS[drug],lw=.9,zorder=3)
            gains.append(dict(drug=drug,reference_model=reference,mae_improvement=z.estimate,ci_low=z.ci_low,ci_high=z.ci_high))
    a.axvline(0,color=GRAY,lw=.7,ls='--')
    a.set(yticks=np.arange(4),yticklabels=DRUGS,ylim=(3.6,-.6),xlabel='Paired MAE improvement')
    a.legend(handles=[Line2D([],[],marker='o',color=INK,ls='',label='Versus four features'),
                      Line2D([],[],marker='o',mfc='white',mec=INK,ls='',label='Versus TP53 + drug/time')],
             loc='upper left',bbox_to_anchor=(-.05,-.25),fontsize=7,handletextpad=.4)
    label(a,'a','Biological context across all regimens',x=-.31)
    old=subset(pred,model='parsimonious_additive',drug='Idasanutlin')
    new=subset(pred,model=PRIMARY,drug='Idasanutlin')
    joined=new.merge(old[['cell_line','regimen','predicted']],on=['cell_line','regimen'],
                     suffixes=('_context','_original'),validate='one_to_one')
    assert len(joined)==48 and joined.cell_line.nunique()==24 and joined.TP53_curated_qualifying.notna().all()
    for r in joined.itertuples():
        col=WARM if r.TP53_curated_qualifying else TEAL
        b.plot([r.observed]*2,[r.predicted_original,r.predicted_context],color=LIGHT,lw=.6,zorder=1)
        b.scatter(r.observed,r.predicted_original,s=22,facecolor='white',edgecolor=col,lw=.8,zorder=2)
        b.scatter(r.observed,r.predicted_context,s=25,color=col,edgecolor='white',lw=.2,zorder=3)
    b.plot([0,1],[0,1],ls='--',lw=.7,color=GRAY)
    b.set(xlim=(-.025,1.025),ylim=(-.025,1.025),xlabel='Observed DDR-high fraction',ylabel='Held-line predicted fraction')
    b.set_aspect('equal',adjustable='box')
    scores={key:unique(perf,drug='Idasanutlin',model=key).r2 for key in ['parsimonious_additive',PRIMARY,GENOTYPE]}
    b.text(.04,.97,f"$R^2$\nFour features: {scores['parsimonious_additive']:.3f}\nState + biology: {scores[PRIMARY]:.3f}\nTP53 + drug/time: {scores[GENOTYPE]:.3f}",
           transform=b.transAxes,va='top',fontsize=7)
    b.text(.96,.04,'48 populations\n24 cell lines',transform=b.transAxes,va='bottom',ha='right',fontsize=7)
    b.legend(handles=[Line2D([],[],marker='o',mfc='white',mec=GRAY,ls='',label='Four features'),
                      Line2D([],[],marker='o',color=INK,ls='',label='State + known biology')],
             loc='upper left',bbox_to_anchor=(-.05,-.25),fontsize=7,handletextpad=.4)
    label(b,'b','Idasanutlin: matched predictions',x=-.33)
    genotype=subset(load(data/'source_context/idasanutlin_TP53_context.csv'),mode='24h')
    # Re-annotate the actual 24-hour response using the same curated genotype
    # used in the model. Preserve the source annotation in separate columns.
    status=new[['cell_line','TP53_curated_qualifying']].drop_duplicates()
    assert status.cell_line.is_unique
    genotype=genotype.merge(status,on='cell_line',validate='one_to_one')
    assert len(genotype)==24
    rng=np.random.default_rng(20261117)
    for i,(group,col) in enumerate([(0,TEAL),(1,WARM)]):
        values=genotype[genotype.TP53_curated_qualifying.eq(group)].delta_mean_score.to_numpy()
        c.scatter(i+rng.uniform(-.13,.13,len(values)),values,s=25,color=col,edgecolor='white',lw=.3)
        c.plot([i-.24,i+.24],[values.mean()]*2,color=INK,lw=1.2)
        c.text(i,values.max()+.065,f'n={len(values)}',ha='center',fontsize=7)
    c.axhline(0,color=LIGHT,lw=.7)
    c.set(xticks=[0,1],xticklabels=['No qualifying\nTP53 variant','Qualifying\nTP53 variant'],
          xlim=(-.6,1.6),ylim=(-.04,genotype.delta_mean_score.max()+.18),ylabel='Idasanutlin 24 h p53-proxy induction')
    label(c,'c','Transcriptional response and TP53',x=-.31)
    chem=subset(load(data/'source_context/normal_matched_chemical_cell_line_summary.csv'),mode='24h')
    mods=['ddr_composite','TP53_axis','E2F_replication','ATF4_integrated_stress','XBP1_UPR']
    matrix=np.array([[unique(chem,drug=drug,module=m).mean_delta_fraction_z2 for m in mods] for drug in DRUGS])
    bound=float(np.ceil(np.abs(matrix).max()*10)/10)
    im=d.imshow(matrix,cmap='BrBG',norm=TwoSlopeNorm(vmin=-bound,vcenter=0,vmax=bound),aspect='auto')
    d.set(yticks=np.arange(4),yticklabels=DRUGS,xticks=np.arange(5),xticklabels=['DDR','p53','E2F','ATF4','XBP1'])
    d.tick_params(axis='both',length=0)
    for i in range(4):
        for j in range(5):
            d.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',fontsize=7,
                   color='white' if abs(matrix[i,j])>.64*bound else INK)
    cb=fig.colorbar(im,ax=d,orientation='horizontal',fraction=.065,pad=.18)
    cb.set_label('Mean programme-high fraction change at 24 h',fontsize=7)
    label(d,'d','Drug-associated programme patterns',x=-.31)
    source(pd.DataFrame(gains),'Fig2_biological_context_paired_MAE_gains',output)
    source(joined,'Fig2_idasanutlin_matched_predictions',output)
    source(genotype,'Fig2_idasanutlin_TP53_transcription',output)
    source(chem[chem.module.isin(mods)],'Fig2_drug_programme_changes',output)
    save(fig,'Fig2_regulons_and_external_validation',output)


def figure3(data,output):
    damage=data/'direct_damage';repair=data/'direct_repair'
    effect=load(damage/'primary_plate_summary.csv').sort_values('plate')
    stain=load(damage/'within_plate_DDR_label_associations.csv')
    stain=stain[stain.label.eq('DDR_high')].set_index('plate').loc[effect.plate]
    assay=load(repair/'primary_results.csv')
    assert len(effect)==6 and len(assay)==16
    fig=plt.figure(figsize=(183*MM,188*MM))
    gs=fig.add_gridspec(3,2,height_ratios=[.55,1.55,2.3],left=.255,right=.97,
                        top=.965,bottom=.10,hspace=.66,wspace=.54)
    a=fig.add_subplot(gs[0,:]);a.set_axis_off()
    a.text(-.255,1.03,'a',transform=a.transAxes,fontweight='bold',fontsize=10)
    labels=[('RNA + protein, same cell','Fixed DDR score  ↔  γH2AX fluorescence','RPE1 · six treated sorting plates'),
            ('RNA + lesion reporter, same cell','Fixed DDR score  ↔  repair-dependent reporter','19-line source panel · 2 h and 24 h')]
    for x,(title,body,foot) in zip([-.11,.47],labels):
        a.add_patch(FancyBboxPatch((x,.03),.52,.9,boxstyle='round,pad=.008,rounding_size=.01',
                                  ec=LIGHT,fc='#F4F6F7',lw=.7,clip_on=False))
        a.text(x+.260,.76,title,ha='center',va='center',fontsize=8,color=NAVY)
        a.text(x+.260,.44,body,ha='center',va='center',fontsize=7)
        a.text(x+.260,.15,foot,ha='center',va='center',fontsize=7,color=INK)
    b=fig.add_subplot(gs[1,0]);c=fig.add_subplot(gs[1,1],sharey=b)
    yy=np.arange(6)
    b.hlines(yy,effect.rho_adjusted_cycle,effect.rho_adjusted,color=LIGHT,lw=1.2)
    b.scatter(effect.rho_adjusted,yy,s=27,color=NAVY,label='DNA content + RNA QC',zorder=3)
    b.scatter(effect.rho_adjusted_cycle,yy,s=27,facecolor='white',edgecolor=WARM,lw=1,
              label='+ cycle proxy',zorder=4)
    b.set(yticks=yy,yticklabels=[f'{r.plate[-3:]} (n={r.n})' for r in effect.itertuples()],
          xlim=(0,.48),ylim=(5.55,-.55),xlabel='Within-plate partial rank correlation',ylabel='Sorting plate (cells)')
    b.legend(loc='lower left',bbox_to_anchor=(-.04,-.48),fontsize=7,handletextpad=.4)
    label(b,'b','RNA score and γH2AX',x=-.54)
    for y,(_,row) in enumerate(stain.iterrows()):
        color=NAVY
        c.plot([row.gammaH2AX_other_median,row.gammaH2AX_high_median],[y,y],color=color,lw=1)
        c.scatter(row.gammaH2AX_other_median,y,facecolor='white',edgecolor=color,s=26,lw=.9,zorder=3)
        c.scatter(row.gammaH2AX_high_median,y,color=color,s=27,zorder=4)
    c.tick_params(axis='y',labelleft=False)
    c.set(xlim=(0,800),xlabel='Median measured γH2AX (a.u.)')
    c.legend(handles=[Line2D([],[],marker='o',mfc='white',mec=INK,ls='',label='Other cells'),
                      Line2D([],[],marker='o',color=INK,ls='',label='RNA-defined DDR-high')],
             loc='lower left',bbox_to_anchor=(-.04,-.48),fontsize=7,handletextpad=.4)
    label(c,'c','A vehicle-defined expression tail',x=-.21)
    d=fig.add_subplot(gs[2,:])
    keys=['MMR','NER','DR_O6MeG','BER_8oxoG_C','BER_A_8oxoG','BER_UG','BER_Hx','NHEJ']
    names=['Mismatch repair','NER reactivation','O6-methylG reversal','BER: 8-oxoG:C',
           'BER: A:8-oxoG','BER: U:G','BER: hypoxanthine','End-joining reactivation']
    for time,col,off in [('2h',NAVY,-.14),('24h',WARM,.14)]:
        q=assay[assay.timepoint.eq(time)].set_index('endpoint').loc[keys]
        y=np.arange(8)+off
        d.errorbar(q.rho,y,xerr=np.vstack([q.rho-q.ci_low,q.ci_high-q.rho]),
                   fmt='o' if time=='2h' else 's',ms=3.8,lw=1,capsize=2,color=col,
                   mfc=col if time=='2h' else 'white',
                   label='2 h (17 lines)' if time=='2h' else '24 h (12 lines)')
        for i,(_,r) in enumerate(q.iterrows()):
            if r.q_signflip<.05:
                d.text(r.ci_high+.013,y[i]+.03,'*',color=col,va='center',fontsize=12)
    d.axvline(0,color=GRAY,lw=.7,ls='--')
    d.set(yticks=np.arange(8),yticklabels=names,xlim=(-.18,.33),ylim=(7.65,-.65),
          xlabel='Within-line adjusted association (equal-line Fisher-z mean)')
    d.legend(loc='lower right',bbox_to_anchor=(1,.13),fontsize=7)
    label(d,'d','Functional correspondence depends on the lesion readout',x=-.255)
    d.text(.00,-.23,'Bars: 95% line-bootstrap intervals · * q<0.05 across all 16 pathway × time tests',
           transform=d.transAxes,fontsize=7,ha='left',va='top')
    source(effect,'Fig3_RNA_gammaH2AX_associations',output)
    source(stain.reset_index(),'Fig3_RNA_DDRhigh_gammaH2AX_contrasts',output)
    source(assay,'Fig3_all_functional_reporter_associations',output)
    save(fig,'Fig3_lineage_same_cell_evidence',output)


def context_figure(data,output):
    # Used as Supplementary Figure S1 unless it is promoted by the manuscript.
    fig=plt.figure(figsize=(183*MM,112*MM))
    gs=fig.add_gridspec(1,2,left=.15,right=.97,top=.89,bottom=.25,wspace=.68)
    a=fig.add_subplot(gs[0,0]);c=fig.add_subplot(gs[0,1])
    vv=data/'source_context';records=[]
    for filename,offset,marker,kind in [('observed_prism_associations.csv',-.12,'o','Observed fraction'),
                                       ('frozen_forecast_PRISM_associations.csv',.12,'s','Held-out prediction')]:
        frame=load(vv/filename);frame=frame[frame.primary.eq(True)].set_index('drug')
        for y,drug in enumerate(DRUGS):
            r=frame.loc[drug];color=COLORS[drug]
            a.plot([r.ci_low,r.ci_high],[y+offset]*2,color=color,lw=1)
            a.scatter(r.rho,y+offset,s=25,marker=marker,
                      facecolor=color if offset<0 else 'white',edgecolor=color,lw=.9,zorder=3)
            records.append(dict(quantity=kind,drug=drug,**r.to_dict()))
    a.axvline(0,color=GRAY,lw=.7,ls='--')
    a.set(yticks=np.arange(4),yticklabels=DRUGS,ylim=(3.6,-.6),xlim=(-1,.65),
          xlabel='Spearman correlation with exact-drug PRISM AUC')
    a.legend(handles=[Line2D([],[],marker='o',color=INK,ls='',label='Observed DDR-high'),
                      Line2D([],[],marker='s',mfc='white',mec=INK,ls='',label='Held-out prediction')],
             loc='upper left',bbox_to_anchor=(-.05,-.24),fontsize=7,handletextpad=.4)
    label(a,'a','Context of viability associations',x=-.28)
    targets=load(data/'genetic/target_effects.csv');targets=targets[targets.eligible_summary.eq(True)]
    rng=np.random.default_rng(20261117)
    groups=['repair_checkpoint','proteostasis','mitosis','ribosome']
    names=['Repair / checkpoint','Proteostasis','Mitosis','Ribosome']
    cols=[NAVY,TEAL,'#9380A4',WARM]
    records2=[]
    for x,(group,name,col) in enumerate(zip(groups,names,cols)):
        q=targets[targets.predefined_group.eq(group)]
        for off,key,marker in [(-.12,'composite_fraction_delta','o'),(.12,'p53_response_fraction_delta','^')]:
            vals=q[key].to_numpy()
            c.scatter(x+off+rng.uniform(-.035,.035,len(vals)),vals,s=20,marker=marker,
                      facecolor=col if off<0 else 'white',edgecolor=col,lw=.75,alpha=.9)
            c.plot([x+off-.10,x+off+.10],[np.median(vals)]*2,color=INK,lw=1.15)
        records2.append(dict(group=group,n_targets=len(q),n_cells=int(q.n_cells.sum()),
                             median_DDR=q.composite_fraction_delta.median(),median_p53=q.p53_response_fraction_delta.median()))
    c.axhline(0,color=LIGHT,lw=.7)
    c.set(xticks=np.arange(4),xticklabels=[f'Repair /\ncheckpoint\n{len(targets[targets.predefined_group.eq(g)])} targets' if g=='repair_checkpoint' else f'{n}\n{len(targets[targets.predefined_group.eq(g)])} targets' for n,g in zip(names,groups)],
          xlim=(-.6,3.6),ylim=(-.08,1.15),ylabel='Target-wise high-fraction change')
    c.legend(handles=[Line2D([],[],marker='o',color=INK,ls='',label='Composite DDR'),
                      Line2D([],[],marker='^',mfc='white',mec=INK,ls='',label='p53 response')],
             loc='upper right',fontsize=7)
    label(c,'b','Genetic perturbation context',x=-.31)
    source(pd.DataFrame(records),'FigS1_viability_associations',output)
    source(pd.DataFrame(records2),'FigS1_genetic_group_summary',output)
    source(targets[targets.predefined_group.isin(groups)],'FigS1_genetic_target_effects',output)
    save(fig,'Supplementary_FigS1',output)


def imaging_supplement(data,output):
    tracks=load(data/'external/imaging_track_summaries.csv')
    live=load(data/'external/imaging_per_frame.csv.gz')
    links=load(data/'external/imaging_live_endpoint_links.csv')
    associations=load(data/'external/imaging_associations.csv')
    fig=plt.figure(figsize=(183*MM,160*MM))
    gs=fig.add_gridspec(2,2,left=.13,right=.94,top=.94,bottom=.12,hspace=.62,wspace=.62)
    a=fig.add_subplot(gs[0,0]);b=fig.add_subplot(gs[0,1]);c=fig.add_subplot(gs[1,0]);d=fig.add_subplot(gs[1,1])
    q=live[live.track_id.isin(tracks.track_id)]
    z=q.groupby('Time')['53BP1_foci__Counts'].agg(['median',lambda x:x.quantile(.25),lambda x:x.quantile(.75),'count'])
    z.columns=['median','q25','q75','n_tracks']
    a.fill_between(z.index,z.q25,z.q75,color=LIGHT,lw=0)
    a.plot(z.index,z['median'],color=NAVY,lw=1.2)
    a.set(xlabel='Live frame',ylabel='Observed 53BP1 foci')
    a.text(.03,.97,'Median and interquartile range\nAll eligible observed tracks',transform=a.transAxes,fontsize=7,va='top')
    label(a,'a','53BP1 dynamics in one field',x=-.25)
    ids=tracks.sort_values(['max_53BP1','duration_frames'],ascending=False).track_id.tolist()
    first=int(q.Time.min());last=int(q.Time.max());mat=np.full((len(ids),last-first+1),np.nan)
    for i,tid in enumerate(ids):
        g=q[q.track_id.eq(tid)];mat[i,(g.Time-first).astype(int).to_numpy()]=g['53BP1_foci__Counts']
    cmap=plt.colormaps['magma'].copy();cmap.set_bad('white')
    im=b.imshow(mat,aspect='auto',cmap=cmap,interpolation='none',extent=[first-.5,last+.5,len(ids),0],vmin=0,vmax=np.nanmax(mat))
    b.set(xlabel='Live frame',ylabel='All 244 eligible tracks',yticks=[0,244])
    cb=fig.colorbar(im,ax=b,fraction=.045,pad=.03);cb.set_label('53BP1 foci',fontsize=7)
    label(b,'b','Observed histories without interpolation',x=-.25)
    rng=np.random.default_rng(20261117)
    c.scatter(links.max_53BP1+rng.uniform(-.10,.10,len(links)),
              links['gH2AX_foci__Counts']+rng.uniform(-.10,.10,len(links)),
              s=18,color=TEAL,alpha=.78,edgecolors='white',lw=.2)
    c.set(xlabel='Maximum live 53BP1 foci',ylabel='Endpoint γH2AX foci')
    c.text(.96,.96,'95 exactly linked cells\nSpearman ρ=0.526\nObservation-count adjusted: 0.632',
           transform=c.transAxes,ha='right',va='top',fontsize=7)
    label(c,'c','Live-to-endpoint marker correspondence',x=-.25)
    # All reported associations are shown; no cell-level significance is implied.
    xname='rho' if 'rho' in associations else 'spearman_rho'
    aname='rho_adjusted' if 'rho_adjusted' in associations else 'observation_count_adjusted_rank_correlation'
    if xname not in associations or aname not in associations:
        # Column naming is checked explicitly against the supplied source table.
        candidates=[k for k in associations if 'rho' in k]
        raise ValueError(('Unknown imaging correlation columns',candidates))
    names=['PCNA max ↔ 53BP1 max','53BP1 last ↔ γH2AX','53BP1 max ↔ γH2AX',
           'PCNA last ↔ γH2AX','53BP1 mean ↔ γH2AX']
    assert len(associations)==5
    for y,row in enumerate(associations.itertuples()):
        raw=getattr(row,xname);adj=getattr(row,aname)
        d.plot([raw,adj],[y,y],color=GRAY,lw=1)
        d.scatter(raw,y,s=23,facecolor='white',edgecolor=GRAY,lw=.9)
        d.scatter(adj,y,s=25,color=NAVY,zorder=3)
    d.set(yticks=np.arange(5),yticklabels=names,ylim=(4.6,-.6),xlim=(0,.75),xlabel='Rank correlation')
    d.legend(handles=[Line2D([],[],marker='o',mfc='white',mec=GRAY,ls='',label='Unadjusted'),
                      Line2D([],[],marker='o',color=NAVY,ls='',label='Count adjusted')],
             loc='upper left',bbox_to_anchor=(0,-.22),fontsize=7)
    pos=d.get_position();d.set_position([pos.x0+.06,pos.y0,pos.width-.06,pos.height])
    label(d,'d','Dependence on observation count',x=-.25)
    source(z.reset_index(),'FigS2_all_track_temporal_summary',output)
    source(links,'FigS2_exact_live_endpoint_links',output)
    source(associations,'FigS2_all_descriptive_associations',output)
    save(fig,'Supplementary_FigS2',output)



def physical_figure(lesions,output):
    """Draw all frozen conditions and relative physical-localization endpoints."""
    order=['Ctrl','0hr','2hr','6hr','24hr','48hr']
    names=['Ctrl','0 h','2 h','6 h','24 h','48 h']
    assoc=load(lesions/'condition_associations.csv')
    summary=load(lesions/'condition_summaries.csv').set_index('condition').loc[order].reset_index()
    quartiles=load(lesions/'quartile_pseudobulk.csv')
    readmatched=load(lesions/'read_matched_summary.csv')
    assert len(assoc)==18 and set(assoc.condition)==set(order)
    primary=assoc[assoc.endpoint.eq('induced_fraction')].set_index('condition').loc[order]
    conserved=assoc[assoc.endpoint.eq('conserved_fraction')].set_index('condition').loc[order]
    partitions=quartiles.groupby(['Treatment','barcode_tube','capture_library'],as_index=False)[['DNA_auto','induced_reads','conserved_reads','n_cells']].sum()
    partitions['induced_fraction']=partitions.induced_reads/partitions.DNA_auto
    partitions['conserved_fraction']=partitions.conserved_reads/partitions.DNA_auto
    fig=plt.figure(figsize=(183*MM,187*MM))
    gs=fig.add_gridspec(4,6,height_ratios=[.57,1.85,.10,1.25],left=.12,right=.97,
                        top=.965,bottom=.11,hspace=.65,wspace=.68)
    a=fig.add_subplot(gs[0,:]);a.set_axis_off()
    scheme=[(.01,.29,'Paired RNA + lesion-associated DNA','HeLa S3 · 64,794 source cells'),
            (.35,.29,'H$_2$O$_2$ exposure and recovery','Ctrl · 0 · 2 · 6 · 24 · 48 h'),
            (.69,.30,'Frozen induced-region localization','2,148 regions / autosomal fragments')]
    for i,(x,w,title,body) in enumerate(scheme):
        a.add_patch(FancyBboxPatch((x,.05),w,.84,boxstyle='round,pad=.006,rounding_size=.01',
                                  ec=LIGHT,fc='#F4F6F7',lw=.7,clip_on=False))
        a.text(x+w/2,.62,title,ha='center',va='center',fontsize=7,color=NAVY)
        a.text(x+w/2,.29,body,ha='center',va='center',fontsize=7)
        if i<2:a.add_patch(FancyArrowPatch((x+w+.01,.47),(scheme[i+1][0]-.01,.47),
                                          arrowstyle='-|>',mutation_scale=8,color=GRAY,lw=.7))
    a.text(-.08,1.04,'a',transform=a.transAxes,fontsize=10,weight='bold')
    b=fig.add_subplot(gs[1,:3]);c=fig.add_subplot(gs[1,3:])
    rng=np.random.default_rng(20261117)
    for i,condition in enumerate(order):
        z=partitions[partitions.Treatment.eq(condition)]
        b.scatter(i+rng.uniform(-.13,.13,len(z)),100*z.induced_fraction,s=14,
                  facecolors='white',edgecolors=GRAY,lw=.55,alpha=.72,zorder=2)
        point=summary.loc[summary.condition.eq(condition),'pseudobulk_induced_fraction'].iloc[0]
        b.scatter(i,100*point,s=31,color=NAVY,edgecolors='white',lw=.5,zorder=3)
    b.set(xticks=np.arange(6),xticklabels=names,xlim=(-.5,5.5),ylim=(0,100*partitions.induced_fraction.max()*1.1),
          xlabel='Recovery time after H$_2$O$_2$',ylabel='DNA fragments in induced regions (%)')
    b.legend(handles=[Line2D([],[],marker='o',mfc='white',mec=GRAY,ls='',label='Technical partitions'),
                      Line2D([],[],marker='o',color=NAVY,ls='',label='All cells pooled')],
             loc='upper left',bbox_to_anchor=(-.03,-.26),fontsize=7,handletextpad=.4)
    label(b,'b','Relative physical lesion localization',x=-.22)
    yy=np.arange(6)
    c.scatter(primary.rho_QC,yy-.14,s=26,color=NAVY,label='Induced regions: QC',zorder=3)
    c.scatter(primary.rho_cycle_QC,yy,s=26,facecolor='white',edgecolor=WARM,lw=1,
              label='Induced regions: QC + cycle',zorder=4)
    c.scatter(conserved.rho_cycle_QC,yy+.14,s=25,facecolor='white',edgecolor=GRAY,marker='^',lw=.9,
              label='Conserved regions: QC + cycle',zorder=2)
    for i in range(6):c.plot([primary.rho_QC.iloc[i],primary.rho_cycle_QC.iloc[i]],[i-.14,i],color=LIGHT,lw=.8,zorder=1)
    c.axvline(0,color=GRAY,lw=.7,ls='--')
    extent=np.abs(np.r_[primary.rho_QC,primary.rho_cycle_QC,conserved.rho_cycle_QC]).max()
    c.set(yticks=yy,yticklabels=[f'{name} (n={int(n):,})' for name,n in zip(names,primary.n_cells)],
          ylim=(5.55,-.55),xlim=(-max(.08,extent*1.2),max(.08,extent*1.2)),
          xlabel='Within-condition partial rank correlation')
    c.legend(loc='upper left',bbox_to_anchor=(-.12,-.26),fontsize=7,handletextpad=.4)
    pos=c.get_position();c.set_position([pos.x0+.065,pos.y0,pos.width-.065,pos.height])
    label(c,'c','RNA state and physical localization',x=-.30)
    ratios=[];facets=[]
    for i,condition in enumerate(order):
        ax=fig.add_subplot(gs[3,i]);facets.append(ax)
        g=quartiles[quartiles.Treatment.eq(condition)]
        for (tube,library),z in g.groupby(['barcode_tube','capture_library']):
            z=z.set_index('DDR_quartile').reindex([1,2,3,4])
            if not np.isfinite(z.induced_fraction).all() or z.loc[1,'induced_fraction']<=0:continue
            ratio=z.induced_fraction/z.loc[1,'induced_fraction']
            ax.plot([1,2,3,4],ratio,color=GRAY,alpha=.25,lw=.5,zorder=1)
            ratios.extend(dict(condition=condition,barcode_tube=tube,capture_library=library,
                               DDR_quartile=int(q),induced_fraction=float(z.loc[q,'induced_fraction']),
                               relative_to_partition_Q1=float(r)) for q,r in ratio.items())
        pooled=g.groupby('DDR_quartile')[['induced_reads','DNA_auto']].sum().reindex([1,2,3,4])
        value=pooled.induced_reads/pooled.DNA_auto;value=value/value.iloc[0]
        ax.plot([1,2,3,4],value,color=NAVY,lw=1.25,marker='o',ms=3,zorder=3)
        matched=unique(readmatched,condition=condition,endpoint='induced')
        ax.scatter(4,matched.ratio_median,s=29,marker='D',facecolor='white',edgecolor=WARM,lw=.9,zorder=4)
        ax.axhline(1,color=LIGHT,lw=.6,ls='--',zorder=0)
        ax.set(xticks=[1,4],xticklabels=['Q1','Q4'],xlim=(.8,4.2))
        ax.set_title(names[i],fontsize=7,pad=5)
        if i==0:ax.set_ylabel('Induced localization / Q1')
        else:ax.tick_params(axis='y',labelleft=False)
    allvals=np.r_[pd.DataFrame(ratios).relative_to_partition_Q1.to_numpy(),1]
    low=min(.98,float(np.min(allvals)));high=max(1.02,float(np.max(allvals)));pad=(high-low)*.08
    for ax in facets:ax.set_ylim(low-pad,high+pad)
    facets[0].text(-.75,1.25,'d',transform=facets[0].transAxes,fontsize=10,weight='bold')
    facets[0].text(-.05,1.28,'DDR quartiles within matched technical partitions',
                   transform=facets[0].transAxes,fontsize=8,ha='left',va='bottom')
    fig.text(.55,.045,'Gray: technical partitions · Blue: pooled quartiles · Orange: read-matched Q4/Q1 median',
             fontsize=7,ha='center')
    source(summary,'Fig4_all_condition_physical_summaries',output)
    source(partitions,'Fig4_physical_technical_partitions',output)
    source(assoc,'Fig4_all_condition_physical_associations',output)
    source(quartiles,'Fig4_DDR_quartile_pseudobulk',output)
    source(readmatched,'Fig4_read_matched_quartile_sensitivity',output)
    source(pd.DataFrame(ratios),'Fig4_within_partition_quartile_localization',output)
    save(fig,'Fig4_Direct_Validation',output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--forecast-root',type=Path)
    parser.add_argument('--extension-root',type=Path)
    parser.add_argument('--lesion-root',type=Path)
    parser.add_argument('--primary',default=PRIMARY)
    parser.add_argument('--only',nargs='+',choices=['1','2','3','4','context','imaging','classifier'])
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True);style()
    chosen=set(args.only or ['1','2','3','context','imaging','classifier'])
    if '1' in chosen:figure1(args.data,args.output,args.forecast_root or args.data/'forecast',args.primary,args.extension_root or args.data/'model_extension')
    if '2' in chosen:mechanistic_figure(args.data,args.output,args.extension_root or args.data/'model_extension')
    if '3' in chosen:figure3(args.data,args.output)
    if '4' in chosen:physical_figure(args.lesion_root or args.data/'direct_lesions',args.output)
    if 'context' in chosen:context_figure(args.data,args.output)
    if 'imaging' in chosen:imaging_supplement(args.data,args.output)
    if 'classifier' in chosen:classifier_figure(args.data,args.output)
    print('Rendered', ', '.join(sorted(chosen)), 'from supplied scientific analysis records.')


if __name__=='__main__':main()
