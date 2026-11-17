"""Supplementary scientific figures from fixed archived analysis records."""

from __future__ import annotations

import os

from pathlib import Path

import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'ddr_figure_cache'))

import matplotlib as mpl

import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

import numpy as np

import pandas as pd

MM = 1 / 25.4

NAVY = '#285A76'

TEAL = '#568F81'

WARM = '#BD8350'

LIGHT = '#D6DBDF'

GRAY = '#98A0A8'

INK = '#26313A'

DRUGS = ['Gemcitabine', 'Idasanutlin', 'Prexasertib', 'Bortezomib']

COLORS = dict(zip(DRUGS, ['#537C9B', '#568F78', '#9C7FA8', '#BF8B58']))

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
