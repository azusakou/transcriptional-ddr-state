"""Cell-style biological-context and held-line prediction figures.

The renderer consumes archived scientific records and performs no model fitting.
All observations are retained; ordering affects display only.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, Rectangle, Ellipse

from cell_style import (
    INK,GRAY,LIGHT,NAVY,TEAL,CORAL,PURPLE,GOLD,DRUGS,COLORS,
    PRIMARY,BIOLOGY,GENOTYPE,new_figure,panel,source,save,set_style,
)

def load(path):
    path=Path(path)
    if not path.exists() and path.suffix=='.csv':path=path.with_suffix('.csv.gz')
    return pd.read_csv(path,float_precision='round_trip')

def one(frame,**selection):
    for key,value in selection.items():frame=frame[frame[key].eq(value)]
    assert len(frame)==1,(selection,len(frame))
    return frame.iloc[0]

def heading(fig,x,y,letter,title):
    fig.text(x,y,letter,fontsize=11,fontweight='bold',va='baseline')
    fig.text(x+.035,y,title,fontsize=8.5,va='baseline')

def export(fig,name,out):
    fig.canvas.draw()
    r=fig.canvas.get_renderer();w,h=fig.canvas.get_width_height()
    outside=[]
    for artist in fig.findobj(mpl.text.Text):
        if not artist.get_visible() or not artist.get_text().strip():continue
        if artist in [t for a in fig.axes for t in a.get_xticklabels()+a.get_yticklabels()]:
            continue
        bb=artist.get_window_extent(r)
        if bb.x0<-.5 or bb.y0<-.5 or bb.x1>w+.5 or bb.y1>h+.5:
            outside.append(artist.get_text().replace('\n',' / '))
    assert not outside,(name,outside)
    save(fig,name,out)

def prediction_figure(data,out,extension=None):
    extension=Path(extension) if extension else Path(data)/'model_extension'
    pred=load(extension/'curated_all_outer_predictions.csv')
    perf=load(extension/'curated_all_performance.csv')
    boot=load(extension/'curated_biology_line_bootstrap.csv')
    selected=pred[pred.model.eq(PRIMARY)].copy()
    assert len(selected)==243 and selected.cell_line.nunique()==84
    assert selected.groupby('cell_line').fold.nunique().max()==1
    fig=new_figure(height=176)
    # The prediction grid is the figure's largest visual element.
    benchmark=fig.add_axes([.285,.825,.235,.12])
    gains_ax=fig.add_axes([.695,.825,.275,.12])
    heading(fig,.025,.975,'A','Prediction across cell lines')
    heading(fig,.53,.975,'B','Improvement within each drug')
    methods=[('drug_only','Regimen only'),
             ('ddr_fraction_additive','Baseline DDR fraction'),
             ('parsimonious_additive','Four RNA states'),
             (GENOTYPE,'TP53 context'),
             (BIOLOGY,'TP53 + Gem metabolism'),
             (PRIMARY,'RNA + biological context')]
    records=[]
    for i,(key,title) in enumerate(methods):
        m=one(perf,model=key,drug='all')
        ci=one(boot,model=key,drug='all',reference_model='none',statistic='r2')
        col=NAVY if key==PRIMARY else GRAY
        benchmark.plot([ci.ci_low,ci.ci_high],[i,i],lw=1.1,color=col)
        benchmark.scatter(m.r2,i,s=20 if key==PRIMARY else 14,color=col,zorder=3)
        benchmark.text(1.04,i,f'{m.r2:.3f}',transform=benchmark.get_yaxis_transform(),
                       va='center',ha='left',fontsize=7.2,fontweight='bold' if key==PRIMARY else 'normal',color=INK)
        records.append(dict(model=key,label=title,r2=m.r2,ci_low=ci.ci_low,ci_high=ci.ci_high,mae=m.mae))
    benchmark.set(yticks=range(6),yticklabels=[x[1] for x in methods],
                  ylim=(5.6,-.6),xlim=(.10,.75),xticks=[.2,.4,.6],xlabel='Held-line $R^2$')
    benchmark.tick_params(axis='y',length=0,pad=5,labelsize=7.3)
    benchmark.spines['left'].set_visible(False)
    delta=one(boot,drug='all',model=PRIMARY,reference_model=BIOLOGY,statistic='delta_r2')
    fig.text(.045,.757,f'Added RNA state: Δ$R^2$ = {delta.estimate:.3f}  (95% CI {delta.ci_low:.3f} to {delta.ci_high:.3f})',fontsize=7)
    gains=[]
    references=[('drug_only',-.19,'o',True),('ddr_fraction_additive',0,'o',False),(BIOLOGY,.19,'s',True)]
    for y,drug in enumerate(DRUGS):
        for ref,offset,marker,filled in references:
            row=one(boot,drug=drug,model=PRIMARY,reference_model=ref,statistic='mae_improvement')
            color=COLORS[drug]
            gains_ax.plot([row.ci_low,row.ci_high],[y+offset]*2,color=color,lw=.9,zorder=1)
            gains_ax.scatter(row.estimate,y+offset,s=16,marker=marker,
                           facecolor=color if filled else 'white',edgecolor=color,lw=.8,zorder=3)
            gains.append(dict(drug=drug,reference_model=ref,estimate=row.estimate,ci_low=row.ci_low,ci_high=row.ci_high))
    gains_ax.axvline(0,color=GRAY,lw=.7,ls=(0,(2,2)),zorder=0)
    gains_ax.set(yticks=range(4),yticklabels=DRUGS,ylim=(3.55,-.55),
                 xlim=(-.025,.14),xticks=[0,.05,.1],xlabel='Paired MAE improvement')
    gains_ax.tick_params(axis='y',length=0,pad=5)
    gains_ax.spines['left'].set_visible(False)
    handles=[Line2D([],[],ls='',marker='o',ms=3,mfc=INK,mec=INK,label='Regimen'),
             Line2D([],[],ls='',marker='o',ms=3,mfc='white',mec=INK,label='DDR fraction'),
             Line2D([],[],ls='',marker='s',ms=3,mfc=INK,mec=INK,label='Biology')]
    fig.legend(handles=handles,loc='center',bbox_to_anchor=(.77,.765),ncol=3,
               columnspacing=.7,handletextpad=.35,fontsize=7)
    axes=[]
    positions=[(.12,.425,.28,.28),(.625,.425,.28,.28),(.12,.06,.28,.28),(.625,.06,.28,.28)]
    for idx,(drug,pos) in enumerate(zip(DRUGS,positions)):
        ax=fig.add_axes(pos);axes.append(ax)
        q=selected[selected.drug.eq(drug)]
        m=one(perf,model=PRIMARY,drug=drug)
        upper=.4 if drug=='Bortezomib' else 1
        ax.plot([0,upper],[0,upper],color=GRAY,lw=.75,ls=(0,(3,2)),zorder=0)
        ax.scatter(q.observed,q.predicted,s=19,color=COLORS[drug],alpha=.8,
                   edgecolor='white',lw=.35,zorder=3)
        ax.set(xlim=(-.025*upper,1.025*upper),ylim=(-.025*upper,1.025*upper),
               xticks=[0,upper/2,upper],yticks=[0,upper/2,upper],
               xlabel='Observed fraction' if idx>=2 else '',
               ylabel='Predicted fraction' if idx%2==0 else '')
        ax.set_aspect('equal',adjustable='box')
        ax.text(.04,.96,f'$R^2$ = {m.r2:.3f}\nMAE = {m.mae:.3f}',transform=ax.transAxes,
                fontsize=7.3,va='top',ha='left',linespacing=1.35)
        panel(ax,chr(ord('C')+idx),x=-.34,y=1.09)
        ax.text(0,1.105,drug,transform=ax.transAxes,fontsize=8.5,ha='left',va='bottom')
        ax.text(0,1.025,f'{len(q)} populations · {q.cell_line.nunique()} cell lines',
                transform=ax.transAxes,fontsize=7,ha='left',va='bottom')
    source(pd.DataFrame(records),'Fig1_model_comparisons',out)
    source(pd.DataFrame([delta]),'Fig1_added_state_increment',out)
    source(pd.DataFrame(gains),'Fig1_paired_drug_gains',out)
    source(selected,'Fig1_held_line_predictions',out)
    export(fig,'Fig1_compendium_and_prediction',out)

def arrow(ax,start,end,color=INK,style='-|>',lw=.8,scale=8):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle=style,mutation_scale=scale,color=color,lw=lw))

def biological_figure(data,out,extension=None):
    data=Path(data);extension=Path(extension) if extension else data/'model_extension'
    pred=load(extension/'curated_all_outer_predictions.csv')
    new=pred[pred.model.eq(PRIMARY)&pred.drug.eq('Idasanutlin')].copy()
    old=pred[pred.model.eq('parsimonious_additive')&pred.drug.eq('Idasanutlin')].copy()
    assert len(new)==len(old)==48 and new.cell_line.nunique()==24
    fig=new_figure(height=159)
    # Panel A is an explanatory drug-context diagram, not an experimental image.
    a=fig.add_axes([.045,.777,.91,.17]);a.set_axis_off();a.set(xlim=(0,1),ylim=(0,1))
    heading(fig,.025,.975,'A','Drug-specific biological context')
    a.text(.20,.92,'MDM2–p53 context',ha='center',fontweight='bold',fontsize=8,color=CORAL)
    a.text(.20,.69,'Idasanutlin',ha='center',fontsize=8)
    arrow(a,(.20,.61),(.20,.47),CORAL)
    a.text(.20,.34,'MDM2 inhibition',ha='center',fontsize=7.7)
    arrow(a,(.20,.27),(.20,.12),CORAL)
    a.text(.20,.00,'p53-response transcription',ha='center',fontsize=7.5)
    a.text(.41,.43,'Genetic context\nTP53 annotation',ha='center',fontsize=7.1,color=GRAY,linespacing=1.35)
    # Gemcitabine's fixed three covariate groups are displayed as biological inputs.
    a.text(.75,.92,'Gemcitabine pharmacology',ha='center',fontweight='bold',fontsize=8,color=NAVY)
    labels=[('SLC29A1 / DCK','Transport / activation'),('CDA','Deamination'),('RRM1 / RRM2','Ribonucleotide reductase')]
    for i,(genes,desc) in enumerate(labels):
        y=.66-i*.27
        a.plot([.535,.56],[y+.02]*2,color=NAVY,lw=2.1)
        a.text(.58,y+.02,genes,fontsize=7.7,fontweight='bold',va='center',ha='left')
        a.text(.995,y+.02,desc,fontsize=7.2,va='center',ha='right',color=INK)
    # A dense line-resolved heatmap retains every eligible 24-hour population.
    b=fig.add_axes([.12,.505,.815,.174])
    heading(fig,.025,.728,'B','Transcriptional response across individual cell lines')
    effects=load(data/'source_context/normal_matched_chemical_cell_line_effects.csv')
    effects=effects[effects['mode'].eq('24h')].copy()
    modules=['ddr_composite','TP53_axis','E2F_replication','ATF4_integrated_stress','XBP1_UPR']
    effects=effects[effects.module.isin(modules)]
    arrays=[];ordered=[];offset=0;bounds=[]
    for drug,n in zip(DRUGS,[69,24,79,23]):
        g=effects[effects.drug.eq(drug)]
        pivot=g.pivot(index='module',columns='cell_line',values='delta_fraction_z2').reindex(modules)
        lines=pivot.loc['ddr_composite'].sort_values(kind='stable').index
        pivot=pivot.loc[:,lines]
        assert pivot.shape==(5,n) and not pivot.isna().any().any(),(drug,pivot.shape)
        arrays.append(pivot.to_numpy());bounds.append((offset,offset+n,drug,n))
        for j,line in enumerate(lines):
            row=g[g.cell_line.eq(line)].copy();row['display_column']=offset+j;ordered.append(row)
        offset+=n
    matrix=np.concatenate(arrays,axis=1)
    cmap=LinearSegmentedColormap.from_list('fraction_change',['#376C8B','#F7F7F7','#CB565B'])
    im=b.pcolormesh(np.arange(196)-.5,np.arange(6)-.5,matrix,cmap=cmap,
                   norm=TwoSlopeNorm(vmin=-1,vcenter=0,vmax=1),shading='flat',rasterized=False)
    b.set_xlim(-.5,194.5);b.set_ylim(4.5,-.5)
    b.set(yticks=range(5),yticklabels=['DDR','p53','E2F','ATF4','XBP1'],xticks=[])
    b.tick_params(axis='both',length=0)
    for spine in b.spines.values():spine.set_visible(False)
    for start,end,drug,n in bounds:
        if start:b.axvline(start-.5,color='white',lw=1)
        b.plot([start-.5,end-.5],[-.79,-.79],color=COLORS[drug],lw=2.3,clip_on=False)
        b.text((start+end-1)/2,-1.22,f'{drug}  ({n})',ha='center',va='center',fontsize=7.3,clip_on=False,color=INK)
    b.text(.50,-.09,'Cell lines ordered within each drug',transform=b.transAxes,ha='center',va='top',fontsize=7)
    cax=fig.add_axes([.746,.441,.186,.009])
    cb=fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[-1,0,1]);cb.ax.tick_params(labelsize=7,pad=2,length=2)
    cb.set_label('Δ programme-high fraction',fontsize=7,labelpad=1)
    # Actual transcriptional induction is independent of the OOF prediction errors.
    c=fig.add_axes([.12,.10,.30,.25])
    panel(c,'C','p53 response to idasanutlin',x=-.32,y=1.085)
    gen=load(data/'source_context/idasanutlin_TP53_context.csv')
    gen=gen[gen['mode'].eq('24h')].copy()
    statuses=new[['cell_line','TP53_curated_qualifying']].drop_duplicates()
    assert statuses.cell_line.is_unique
    gen=gen.merge(statuses,on='cell_line',validate='one_to_one')
    rng=np.random.default_rng(20261117)
    for i,(status,color) in enumerate([(0,TEAL),(1,CORAL)]):
        vals=gen[gen.TP53_curated_qualifying.eq(status)].delta_mean_score.to_numpy()
        c.scatter(i+rng.uniform(-.16,.16,len(vals)),vals,s=23,color=color,alpha=.85,edgecolor='white',lw=.4,zorder=3)
        c.plot([i-.25,i+.25],[vals.mean()]*2,color=INK,lw=1.2,zorder=4)
        c.text(i,.94,f'n = {len(vals)}',transform=c.get_xaxis_transform(),ha='center',va='top',fontsize=7)
    c.axhline(0,color=LIGHT,lw=.8,zorder=0)
    c.set(xlim=(-.65,1.65),ylim=(-.075,gen.delta_mean_score.max()+.14),
          xticks=[0,1],xticklabels=['No qualifying\nTP53 annotation','Qualifying\nTP53 annotation'],
          ylabel='p53-score change at 24 h')
    c.tick_params(axis='x',length=0,pad=5,labelsize=7)
    # Each of the 24 cell lines contributes one time-averaged absolute OOF error.
    d=fig.add_axes([.605,.10,.32,.25])
    panel(d,'D','Prediction error by TP53 context',x=-.234,y=1.085)
    joined=new.merge(old[['cell_line','regimen','predicted']],on=['cell_line','regimen'],
                     suffixes=('_biology','_RNA'),validate='one_to_one')
    assert joined.groupby('cell_line').size().eq(2).all()
    joined['error_RNA']=np.abs(joined.observed-joined.predicted_RNA)
    joined['error_biology']=np.abs(joined.observed-joined.predicted_biology)
    errors=joined.groupby(['cell_line','TP53_curated_qualifying'],as_index=False)[['error_RNA','error_biology']].mean()
    assert len(errors)==24 and sorted(errors.groupby('TP53_curated_qualifying').size().tolist())==[7,17]
    for status,positions,color in [(0,[0,1],TEAL),(1,[2.55,3.55],CORAL)]:
        g=errors[errors.TP53_curated_qualifying.eq(status)]
        for row in g.itertuples():
            d.plot(positions,[row.error_RNA,row.error_biology],color=color,alpha=.35,lw=.8,zorder=1)
            d.scatter(positions,[row.error_RNA,row.error_biology],color=color,s=13,edgecolor='white',lw=.3,zorder=2)
        means=g[['error_RNA','error_biology']].mean()
        d.plot(positions,means,color=INK,lw=1.3,zorder=4)
        d.scatter(positions,means,color=INK,s=18,marker='D',zorder=5)
    upper=float(np.ceil(errors[['error_RNA','error_biology']].to_numpy().max()*10)/10)+.02
    d.set(xlim=(-.48,4.03),ylim=(-.007,upper),yticks=np.arange(0,upper,.1),
          xticks=[0,1,2.55,3.55],xticklabels=['RNA','+ biology','RNA','+ biology'],
          ylabel='Mean absolute prediction error')
    d.tick_params(axis='x',length=0,pad=5,labelsize=7)
    d.text(.5,-.20,'No qualifying TP53\nannotation (n = 7)',transform=d.get_xaxis_transform(),ha='center',va='top',fontsize=7.1,color=TEAL)
    d.text(3.05,-.20,'Qualifying TP53\nannotation (n = 17)',transform=d.get_xaxis_transform(),ha='center',va='top',fontsize=7.1,color=CORAL)
    source(pd.concat(ordered,ignore_index=True),'Fig2_individual_line_programme_changes',out)
    source(gen,'Fig2_idasanutlin_TP53_transcription',out)
    source(joined,'Fig2_idasanutlin_population_prediction_errors',out)
    source(errors,'Fig2_idasanutlin_line_prediction_errors',out)
    export(fig,'Fig2_regulons_and_external_validation',out)

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--extension-root',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True);set_style()
    prediction_figure(args.data,args.output,args.extension_root)
    biological_figure(args.data,args.output,args.extension_root)
    print('Rendered complete Cell-style Figures 1 and 2 from archived results.')

if __name__=='__main__':main()
