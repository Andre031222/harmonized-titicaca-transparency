"""
==============================================================================
24_dataflow_graphical_abstract.py
Pulido editorial para Remote Sensing (MDPI):
 (A) fig_data_flow.png      : diagrama de flujo de datos (funnel de n) que
     homogeneiza las cifras (881 -> 734 con Secchi -> 1002 match-ups -> 812).
 (B) graphical_abstract.png : resumen visual del estudio (para el slot de
     Graphical Abstract de MDPI; se sube por separado).
==============================================================================
"""
import numpy as np, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT=Path(__file__).resolve().parent.parent
FIG=ROOT/"results/figures"; FIG.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({"font.family":"DejaVu Sans","savefig.dpi":600})

BLUE="#2e86ab"; AMBER="#edae49"; RED="#d1495b"; GREY="#4a4a4a"; LBLUE="#dce9f2"

def box(ax,x,y,w,h,text,fc,ec=GREY,fs=9,tc="black",bold=False):
    p=FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle="round,pad=0.02,rounding_size=0.04",
                     linewidth=1.1,edgecolor=ec,facecolor=fc,zorder=2)
    ax.add_patch(p)
    ax.text(x,y,text,ha="center",va="center",fontsize=fs,color=tc,zorder=3,
            fontweight="bold" if bold else "normal")

def arrow(ax,x1,y1,x2,y2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=14,
                 lw=1.3,color=GREY,zorder=1))

# ---------- (A) data-flow funnel ----------
fig,ax=plt.subplots(figsize=(8.6,4.4)); ax.set_xlim(0,10); ax.set_ylim(0,6); ax.axis("off")
# fila in-situ
box(ax,1.7,5.0,2.7,1.0,"881 IMARPE in-situ\nmeasurements (2011–2024)\n156 stations",LBLUE,fs=8.5)
box(ax,1.7,2.7,2.7,1.0,"734 with a valid\nSecchi reading",LBLUE,fs=8.5)
arrow(ax,1.7,4.5,1.7,3.2)
# satelite
box(ax,5.0,5.0,2.7,1.0,"Sentinel-2 MSI +\nLandsat 8/9 OLI (SR)\nharmonized (Roy 2016)",("#eaf3da"),fs=8.5)
arrow(ax,3.05,5.0,3.65,5.0)
box(ax,5.0,2.7,2.7,1.0,"Match-up extraction\n30 m buffer, ±10 days,\ncloud-masked median",("#eaf3da"),fs=8.5)
arrow(ax,5.0,4.5,5.0,3.2)
arrow(ax,3.05,2.7,3.65,2.7)
# salida
box(ax,8.3,3.85,2.7,1.05,"1,002 match-ups\n(340 S2 + 662 Landsat)",BLUE,fs=9,tc="white",bold=True)
box(ax,8.3,1.7,2.7,1.05,"812 with Secchi\n143 stations · 2013–2024",BLUE,fs=9,tc="white",bold=True)
arrow(ax,6.35,2.9,6.95,3.6)
arrow(ax,8.3,3.3,8.3,2.25)
ax.text(5,5.75,"Data flow: from field campaigns to satellite–field match-ups",
        ha="center",fontsize=11,fontweight="bold")
fig.tight_layout(); fig.savefig(FIG/"fig_data_flow.png",bbox_inches="tight"); plt.close(fig)
print("[A] fig_data_flow.png")

# ---------- (B) graphical abstract ----------
fig,ax=plt.subplots(figsize=(9,4.6)); ax.set_xlim(0,12); ax.set_ylim(0,6); ax.axis("off")
ax.text(6,5.7,"Harmonized S2/Landsat retrieval of water transparency in Lake Titicaca",
        ha="center",fontsize=12,fontweight="bold")
ax.text(6,5.25,"A transferable validation & retrievability protocol for high-altitude lakes",
        ha="center",fontsize=9.5,color=GREY,style="italic")
# step 1 inputs
box(ax,1.6,3.7,2.6,1.4,"Sentinel-2 + Landsat 8/9\nharmonized reflectance\n\n812 real in-situ\nSecchi match-ups",LBLUE,fs=8.5)
arrow(ax,2.95,3.7,3.75,3.7)
# step 2 model
box(ax,5.1,3.7,2.7,1.4,"Random Forest\n12 spectral features\n\nSHAP: green/blue ratios\n(water-clarity optics)",("#eaf3da"),fs=8.5)
arrow(ax,6.5,3.7,7.3,3.7)
# step 3 result
box(ax,8.9,3.7,2.7,1.4,"$Z_{SD}$ retrieval\n$R^2$=0.64, RMSE 1.8 m\n\n+ conformal 90%\nuncertainty (cov. 90%)",BLUE,fs=8.5,tc="white",bold=False)
# bottom band: validation hierarchy mini-bars
bx=[2.0,3.1,4.2,5.3,6.4]; bh=[0.60,0.60,0.51,0.25,-0.05]
bc=[BLUE,BLUE,AMBER,RED,RED]; bl=["rand","stat","temp","zone\nMenor","zone\nMayor"]
base=1.1
for x,h,c,l in zip(bx,bh,bc,bl):
    ax.add_patch(plt.Rectangle((x-0.42,base),0.84,max(h,0)*1.4 if h>0 else h*1.4,
                 facecolor=c,edgecolor=GREY,lw=0.8,zorder=2))
    ax.text(x,base-0.25,l,ha="center",va="top",fontsize=7.2)
    ax.text(x,base+max(h,0)*1.4+0.06,f"{h:.2f}",ha="center",va="bottom",fontsize=7)
ax.plot([1.4,6.9],[base,base],color=GREY,lw=0.8)
ax.text(4.15,2.55,"Validation hierarchy: random ≈ station, but zone-out collapses",
        ha="center",fontsize=8.3,fontweight="bold")
ax.text(9.6,1.55,"Honest envelope:\nSecchi retrievable;\nChl-a / TSS / temp.\nNOT retrievable\nin clear water",
        ha="center",va="center",fontsize=8,
        bbox=dict(boxstyle="round,pad=0.4",fc="#fbeae0",ec=RED,lw=1.1))
fig.tight_layout(); fig.savefig(FIG/"graphical_abstract.png",bbox_inches="tight"); plt.close(fig)
print("[B] graphical_abstract.png")
