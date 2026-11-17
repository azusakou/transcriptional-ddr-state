"""Same-cell experimental endpoint figures from fixed, archived measurements."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch
from matplotlib.lines import Line2D
from cell_style import (INK,GRAY,LIGHT,NAVY,TEAL,CORAL,PURPLE,GOLD,
                        new_figure,panel,source,save,set_style)

ORDER=['Ctrl','0hr','2hr','6hr','24hr','48hr']
NAMES=['Ctrl','0 h','2 h','6 h','24 h','48 h']
ENDPOINTS=['MMR','NER','DR_O6MeG','BER_8oxoG_C','BER_A_8oxoG','BER_UG','BER_Hx','NHEJ']
ENAMES=['MMR','NER','O6-methylG','BER · 8-oxoG:C','BER · A:8-oxoG','BER · U:G','BER · hypoxanthine','End joining']
BLUE=LinearSegmentedColormap.from_list('quantity',['#FFFFFF','#D9E7EF',NAVY])
DIVERGE=LinearSegmentedColormap.from_list('signed',[NAVY,'#FFFFFF',CORAL])

def load(path):
    path=Path(path)
    if not path.exists() and path.suffix=='.csv':path=path.with_suffix('.csv.gz')
    return pd.read_csv(path,float_precision='round_trip')

def arrow(ax,a,b,color=INK,lw=.8):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=8,
                               lw=lw,color=color,clip_on=False))

def nucleus(ax,x,y,r=.075):
    ax.add_patch(Circle((x,y),r,fc='#F1F6F8',ec=NAVY,lw=.9))
    ax.add_patch(Circle((x,y),r*.55,fc='white',ec=NAVY,lw=.7))
    for dx,dy in [(-.018,.012),(.013,.022),(.012,-.017),(-.014,-.020)]:
        ax.add_patch(Circle((x+dx,y+dy),r*.075,fc=CORAL,ec='none'))

def rna(ax,x,y,w=.12,col=TEAL):
    t=np.linspace(0,1,60)
    ax.plot(x+w*t,y+.014*np.sin(5*np.pi*t),color=col,lw=1.2)
    for k in [.18,.38,.58,.78]:
        ax.plot([x+w*k,x+w*k],[y-.006,y+.028],color=col,lw=.7)

def plasmid(ax,x,y,r=.04):
    t=np.linspace(0,2*np.pi,120)
    ax.plot(x+r*np.cos(t),y+.6*r*np.sin(t),color=CORAL,lw=1.1)
    ax.plot([x-r*.4,x+r*.4],[y+r*.25,y+r*.25],color=CORAL,lw=2)
    ax.scatter([x+r*.38],[y+r*.25],s=8,marker='x',color=INK,lw=.7)

def rank_residuals(d):
    rows=[]
    for key,g in d.groupby('experimentid',sort=True):
        cov=g[['DAPI_fluorescence','log_depth','log_genes']].rank().to_numpy()
        z=np.column_stack([np.ones(len(g)),cov])
        a=g.DDR.rank().to_numpy();b=g.yH2AX_fluorescence.rank().to_numpy()
        a=a-z@np.linalg.lstsq(z,a,rcond=None)[0]
        b=b-z@np.linalg.lstsq(z,b,rcond=None)[0]
        rho=np.corrcoef(a,b)[0,1]
        a=a/np.std(a);b=b/np.std(b)
        rows.extend(dict(cell_id=cell,plate=key,hras=hras,DDR_adjusted_rank=x,
                         gammaH2AX_adjusted_rank=y,plate_partial_rho=rho)
                    for cell,hras,x,y in zip(g.cell_id,g.hras,a,b))
    return pd.DataFrame(rows)

def fig3(data,out):
    damage=Path(data)/'direct_damage';repair=Path(data)/'direct_repair'
    effect=load(damage/'primary_plate_summary.csv').sort_values('plate')
    stain=load(damage/'within_plate_DDR_label_associations.csv')
    stain=stain[stain.label.eq('DDR_high')].set_index('plate').loc[effect.plate]
    sensitivity=load(damage/'H2AFX_shared_gene_sensitivity.csv').set_index('plate').loc[effect.plate]
    cells=load(damage/'scored_cells.csv.gz');cells=cells[cells.treatment.eq('GP')]
    adjusted=rank_residuals(cells)
    assert len(adjusted)==1514
    exact=adjusted.groupby('plate').plate_partial_rho.first().loc[effect.plate]
    assert np.allclose(exact,effect.rho_adjusted,atol=1e-12)
    assay=load(repair/'primary_results.csv');assert len(assay)==16
    fig=new_figure(height=169)
    # A deliberately small biological schematic leaves the plate to measurements.
    a=fig.add_axes([.065,.865,.90,.115]);a.set(xlim=(0,1),ylim=(0,.28));a.axis('off')
    panel(a,'A',x=-.045,y=.94)
    nucleus(a,.057,.145,.050);arrow(a,(.112,.145),(.164,.145))
    rna(a,.185,.185,.12);a.text(.245,.220,'Endogenous RNA',ha='center',fontsize=7.5)
    a.scatter([.208,.235,.263,.291],[.078,.052,.085,.057],s=[9,14,8,12],color=CORAL)
    a.text(.245,.015,r'$\gamma$H2AX fluorescence',ha='center',fontsize=7.5)
    a.text(.057,.015,'RPE1',ha='center',fontsize=7.5)
    a.text(.365,.145,'Same-cell\nindex sorting',ha='center',va='center',fontsize=7.5)
    a.plot([.47,.47],[.015,.26],color=LIGHT,lw=.8)
    plasmid(a,.545,.145,.044);arrow(a,(.60,.145),(.65,.145))
    nucleus(a,.705,.145,.047);arrow(a,(.755,.145),(.807,.145))
    rna(a,.825,.195,.12);a.text(.888,.235,'Endogenous RNA',ha='center',fontsize=7.5)
    a.text(.888,.070,'Lesion-specific\nrepair reporters',ha='center',va='center',fontsize=7.5)
    a.text(.705,.015,'19-line panel',ha='center',fontsize=7.5)
    a.text(.545,.015,'HCR',ha='center',fontsize=7.5)
    # B: actual paired cells, conditional only on within-plate rank adjustment.
    b=fig.add_axes([.095,.475,.365,.300])
    limit=float(np.ceil(np.abs(adjusted[['DDR_adjusted_rank','gammaH2AX_adjusted_rank']]).to_numpy().max()*1.04*2)/2)
    hist=b.hexbin(adjusted.DDR_adjusted_rank,adjusted.gammaH2AX_adjusted_rank,
                  gridsize=32,mincnt=1,cmap=BLUE,linewidths=0,extent=(-limit,limit,-limit,limit))
    # Hexbin's white low end can hide singleton cells; explicitly retain those observations.
    b.scatter(adjusted.DDR_adjusted_rank,adjusted.gammaH2AX_adjusted_rank,
              s=2.3,c=NAVY,alpha=.22,lw=0,rasterized=False,zorder=1)
    b.axhline(0,color=LIGHT,lw=.65,zorder=0);b.axvline(0,color=LIGHT,lw=.65,zorder=0)
    b.set(xlim=(-limit,limit),ylim=(-limit,limit),xticks=[-3,0,3],yticks=[-3,0,3],
          xlabel='Adjusted DDR rank',ylabel=r'Adjusted $\gamma$H2AX rank')
    b.text(.04,.96,'1,514 cells · 6 plates',transform=b.transAxes,ha='left',va='top',fontsize=7.5)
    panel(b,'B','RNA–damage signalling',x=-.17)
    cbax=fig.add_axes([.463,.475,.010,.300]);cb=fig.colorbar(hist,cax=cbax)
    cb.ax.tick_params(labelsize=7,length=2);cb.set_label('Cells / bin',fontsize=7,labelpad=4)
    cb.outline.set_visible(False)
    # C: every plate and three transparent control definitions.
    c=fig.add_axes([.650,.475,.285,.300])
    values=np.column_stack([effect.rho_adjusted,effect.rho_adjusted_cycle,sensitivity.without_H2AFX_partial_rho])
    im=c.imshow(values,cmap=BLUE,vmin=0,vmax=.45,aspect='auto')
    c.set(yticks=np.arange(6),yticklabels=[f'{p[-3:]}  ({n})' for p,n in zip(effect.plate,effect.n)],
          xticks=np.arange(3),xticklabels=['RNA QC\n+ DAPI','+ cycle','− H2AFX'])
    c.tick_params(axis='both',length=0,pad=5);c.set_ylabel('Sorting plate (cells)',labelpad=4)
    for iy in range(6):
        for ix in range(3):c.text(ix,iy,f'{values[iy,ix]:.2f}',ha='center',va='center',fontsize=7.4,color='white' if values[iy,ix]>.27 else INK)
    for sp in c.spines.values():sp.set_visible(False)
    c.text(.5,-.25,'Within-plate partial ρ',transform=c.transAxes,ha='center',fontsize=7.5)
    panel(c,'C','Association across sorting plates',x=-.27)
    # D: fluorescence is shown in the original units, without invented per-cell microscopy.
    d=fig.add_axes([.095,.115,.300,.235])
    for i,(_,r) in enumerate(stain.iterrows()):
        d.plot([r.gammaH2AX_other_median,r.gammaH2AX_high_median],[i,i],color=GRAY,lw=.9)
        d.scatter(r.gammaH2AX_other_median,i,s=20,fc='white',ec=GRAY,lw=.9,zorder=3)
        d.scatter(r.gammaH2AX_high_median,i,s=25,fc=CORAL,ec='white',lw=.4,zorder=4)
    d.set(yticks=np.arange(6),yticklabels=[p[-3:] for p in effect.plate],ylim=(5.5,-.5),
          xlim=(200,700),xticks=[200,400,600],xlabel=r'Median $\gamma$H2AX (a.u.)',ylabel='Sorting plate')
    d.legend(handles=[Line2D([],[],marker='o',mfc='white',mec=GRAY,ls='',label='Other cells'),
                      Line2D([],[],marker='o',mfc=CORAL,mec='white',ls='',label='DDR-high')],
             loc='upper left',bbox_to_anchor=(-.03,-.29),ncol=2,fontsize=7,handletextpad=.35,columnspacing=.9)
    panel(d,'D','Vehicle-defined expression state',x=-.21)
    # E: full primary comparison family, not a display of only significant pathways.
    e=fig.add_axes([.655,.115,.285,.235])
    e.axhspan(-.48,.48,fc='#F5F8FA',zorder=0)
    for time,col,off,marker in [('2h',NAVY,-.14,'o'),('24h',CORAL,.14,'s')]:
        q=assay[assay.timepoint.eq(time)].set_index('endpoint').loc[ENDPOINTS]
        e.errorbar(q.rho,np.arange(8)+off,xerr=np.vstack([q.rho-q.ci_low,q.ci_high-q.rho]),
                   fmt=marker,color=col,mfc=col if time=='2h' else 'white',mec=col,
                   ms=3.3,lw=.8,capsize=1.8,elinewidth=.8,zorder=3,label='2 h' if time=='2h' else '24 h')
        for iy,(_,r) in enumerate(q.iterrows()):
            if r.q_signflip<.05:e.text(r.ci_high+.014,iy+off,'*',color=col,va='center',fontsize=11,weight='bold')
    e.axvline(0,color=GRAY,lw=.65,ls=(0,(3,2)),zorder=0)
    e.set(yticks=np.arange(8),yticklabels=ENAMES,ylim=(7.6,-.6),xlim=(-.16,.30),xticks=[-.1,0,.1,.2,.3],
          xlabel='Adjusted reporter association (ρ)')
    e.tick_params(axis='y',length=0,pad=5)
    e.legend(loc='upper left',bbox_to_anchor=(-.02,-.29),ncol=2,fontsize=7.1,columnspacing=1.0)
    panel(e,'E','Functional repair readouts',x=-.50)
    source(adjusted,'Fig3_adjusted_same_cell_display',out)
    source(effect,'Fig3_all_plate_primary_associations',out)
    source(sensitivity.reset_index(),'Fig3_shared_gene_controls',out)
    source(stain.reset_index(),'Fig3_measured_fluorescence_medians',out)
    source(assay,'Fig3_all16_primary_functional_associations',out)
    save(fig,'Fig3_lineage_same_cell_evidence',out)

def fig4(data,out):
    lesions=Path(data)/'direct_lesions'
    assoc=load(lesions/'condition_associations.csv')
    summary=load(lesions/'condition_summaries.csv').set_index('condition').loc[ORDER]
    quartiles=load(lesions/'quartile_pseudobulk.csv')
    matched=load(lesions/'read_matched_summary.csv')
    assert summary.n_cells.sum()==64794
    primary=assoc[assoc.endpoint.eq('induced_fraction')].set_index('condition').loc[ORDER]
    conserved=assoc[assoc.endpoint.eq('conserved_fraction')].set_index('condition').loc[ORDER]
    parts=quartiles.groupby(['Treatment','barcode_tube','capture_library'],as_index=False)[['DNA_auto','induced_reads','conserved_reads','n_cells']].sum()
    parts['induced_percent']=100*parts.induced_reads/parts.DNA_auto
    fig=new_figure(height=161)
    a=fig.add_axes([.06,.865,.90,.115]);a.axis('off');a.set(xlim=(0,1),ylim=(0,.28))
    panel(a,'A',x=-.04,y=.94)
    nucleus(a,.050,.15,.05)
    a.text(.05,.015,'HeLa S3',ha='center',fontsize=7.5)
    arrow(a,(.107,.15),(.160,.15));rna(a,.185,.205,.12)
    a.text(.245,.255,'RNA',ha='center',fontsize=7.5)
    a.plot([.185,.305],[.095,.095],color=NAVY,lw=.8)
    for j in range(7):
        a.plot([.185+j*.017,.194+j*.017],[.075,.112],color=NAVY,lw=.75)
    for j in [1,4]:a.scatter(.19+j*.017,.113,s=12,marker='x',color=CORAL,lw=.8)
    a.text(.245,.025,'Lesion-associated DNA',ha='center',fontsize=7.5)
    a.text(.40,.15,'64,794\npaired nuclei',ha='center',va='center',fontsize=8)
    a.plot([.505,.505],[.02,.26],color=LIGHT,lw=.8)
    arrow(a,(.57,.15),(.97,.15),color=GRAY)
    for x,txt in zip(np.linspace(.59,.95,5),['0','2','6','24','48 h']):
        a.plot([x,x],[.125,.175],color=INK,lw=.7)
        a.text(x,.075,txt,ha='center',fontsize=7.4)
    a.text(.77,.235,r'200 μM H$_2$O$_2$ · 2 h exposure',ha='center',fontsize=7.6)
    a.text(.77,.015,'Recovery',ha='center',fontsize=7.5)
    a.text(.55,.078,'Ctrl',ha='center',fontsize=7.2,color=GRAY)
    # B: all technical partition measurements with condition-level captured fractions.
    b=fig.add_axes([.10,.475,.36,.285]);rng=np.random.default_rng(20261117)
    for i,condition in enumerate(ORDER):
        z=parts[parts.Treatment.eq(condition)]
        b.scatter(i+rng.uniform(-.11,.11,len(z)),z.induced_percent,s=8,fc='white',ec=GRAY,alpha=.6,lw=.5,zorder=2)
    y=summary.pseudobulk_induced_fraction.to_numpy()*100
    b.plot(np.arange(1,6),y[1:],color=NAVY,lw=1.2,zorder=3)
    b.scatter(np.arange(6),y,s=30,color=NAVY,ec='white',lw=.5,zorder=4)
    b.axvline(.5,color=LIGHT,lw=.7,ls=(0,(3,2)))
    b.set(xticks=np.arange(6),xticklabels=NAMES,xlim=(-.5,5.5),ylim=(0,.115),yticks=[0,.04,.08,.12],
          xlabel='Recovery after exposure',ylabel='Induced-region fragments (%)')
    b.text(.05,.05,'2,148 fixed damage regions',transform=b.transAxes,fontsize=7.3)
    panel(b,'B','Captured lesion localization',x=-.17)
    # C: all conditions and the two fixed genomic targets, no biological P values.
    c=fig.add_axes([.655,.475,.28,.285])
    values=np.column_stack([primary.rho_QC,primary.rho_cycle_QC,conserved.rho_cycle_QC])
    c.imshow(values,cmap=DIVERGE,norm=TwoSlopeNorm(vmin=-.4,vcenter=0,vmax=.4),aspect='auto')
    for iy in range(6):
        for ix in range(3):c.text(ix,iy,f'{values[iy,ix]:.3f}',ha='center',va='center',fontsize=7.2,color=INK)
    c.set(yticks=np.arange(6),yticklabels=NAMES,xticks=np.arange(3),
          xticklabels=['Induced\nQC','Induced\n+ cycle','Conserved\n+ cycle'])
    c.tick_params(axis='both',length=0,pad=5)
    for sp in c.spines.values():sp.set_visible(False)
    c.text(.5,-.29,'Within-condition partial ρ',transform=c.transAxes,fontsize=7.4,ha='center')
    panel(c,'C','RNA–localization associations',x=-.26)
    # D: all 24 pooled quartile signals, avoiding overlapping technical spaghetti traces.
    pooled=quartiles.groupby(['Treatment','DDR_quartile'])[['DNA_auto','induced_reads']].sum()
    pooled['induced_fraction']=pooled.induced_reads/pooled.DNA_auto
    qmat=pooled.induced_fraction.unstack('DDR_quartile').loc[ORDER,[1,2,3,4]]
    ratios=qmat.divide(qmat[1],axis=0)
    d=fig.add_axes([.125,.115,.32,.235])
    di=d.imshow(ratios,cmap=DIVERGE,norm=TwoSlopeNorm(vmin=.90,vcenter=1,vmax=1.10),aspect='auto')
    for iy in range(6):
        for ix in range(4):d.text(ix,iy,f'{ratios.iloc[iy,ix]:.2f}',ha='center',va='center',fontsize=7.4,color=INK)
    d.set(yticks=np.arange(6),yticklabels=NAMES,xticks=np.arange(4),xticklabels=['Q1','Q2','Q3','Q4'],
          xlabel='DDR expression quartile')
    d.tick_params(axis='both',length=0,pad=5)
    for sp in d.spines.values():sp.set_visible(False)
    cbax=fig.add_axes([.464,.115,.010,.235]);cb=fig.colorbar(di,cax=cbax,ticks=[.9,1,1.1]);cb.outline.set_visible(False)
    cb.set_label('Localization / Q1',fontsize=7,labelpad=3);cb.ax.tick_params(labelsize=7,length=2)
    panel(d,'D','Localization across RNA states',x=-.25)
    # E: read sampling intervals describe thinning variation, not independent cultures.
    e=fig.add_axes([.70,.115,.24,.235])
    m=matched[matched.endpoint.eq('induced')].set_index('condition').loc[ORDER]
    raw=ratios[4]
    for i,condition in enumerate(ORDER):
        e.plot([raw.loc[condition],m.loc[condition,'ratio_median']],[i,i],color=GRAY,lw=.8)
    e.scatter(raw,np.arange(6),s=21,fc='white',ec=GRAY,lw=.8,zorder=3,label='Pooled quartiles')
    e.errorbar(m.ratio_median,np.arange(6),xerr=np.vstack([m.ratio_median-m.sampling_min,m.sampling_max-m.ratio_median]),
               fmt='s',color=CORAL,ms=3.5,lw=.8,capsize=1.8,zorder=4,label='Read matched')
    e.axvline(1,color=GRAY,lw=.65,ls=(0,(3,2)),zorder=0)
    e.set(yticks=np.arange(6),yticklabels=NAMES,ylim=(5.5,-.5),xlim=(.955,1.08),xticks=[.96,1,1.04,1.08],
          xlabel='Q4 / Q1 localization')
    e.tick_params(axis='y',length=0,pad=5)
    e.legend(loc='upper left',bbox_to_anchor=(-.21,-.29),fontsize=7,ncol=1)
    panel(e,'E','DNA-depth matching',x=-.31)
    source(summary.reset_index(),'Fig4_condition_summaries',out)
    source(parts,'Fig4_all_technical_partition_fractions',out)
    source(assoc,'Fig4_all18_fixed_genomic_associations',out)
    source(ratios.reset_index(),'Fig4_pooled_RNA_quartile_localization',out)
    source(m.reset_index(),'Fig4_read_matched_Q4_Q1',out)
    save(fig,'Fig4_Direct_Validation',out)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--only',choices=['3','4'],nargs='+')
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True);set_style()
    if not a.only or '3' in a.only:fig3(a.data,a.output)
    if not a.only or '4' in a.only:fig4(a.data,a.output)
    print('Rendered same-cell and physical-localization figures from fixed scientific records.')

if __name__=='__main__':main()
