"""
==============================================================================
19_map_and_harmonization.py
Fig area de estudio (lago + estaciones reales por zona) y
Fig firma espectral del agua por sensor (consistencia S2 vs Landsat armonizado).
==============================================================================
"""
import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import geopandas as gpd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent.parent
PROC=ROOT/"data/processed"; FIG=ROOT/"results/figures"; FIG.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({"font.size":10,"savefig.dpi":600,"font.family":"DejaVu Sans"})
Zc={"BAHIA PUNO":"#d1495b","LAGO MENOR":"#edae49","LAGO MAYOR":"#2e86ab"}

def add_cartography(ax,bar_km=25):
    """Add a north arrow and a metric scale bar to a lon/lat map (north up)."""
    x0,x1=ax.get_xlim(); y0,y1=ax.get_ylim()
    latc=(y0+y1)/2.0
    bar=bar_km/(111.320*np.cos(np.radians(latc)))   # km -> degrees of longitude
    xs=x0+0.06*(x1-x0); ys=y0+0.06*(y1-y0); tick=0.012*(y1-y0)
    ax.plot([xs,xs+bar],[ys,ys],color="#222",lw=2.5,solid_capstyle="butt",zorder=5)
    ax.plot([xs,xs],[ys,ys+tick],color="#222",lw=2.5,zorder=5)
    ax.plot([xs+bar,xs+bar],[ys,ys+tick],color="#222",lw=2.5,zorder=5)
    ax.text(xs+bar/2,ys+1.6*tick,f"{bar_km} km",ha="center",va="bottom",fontsize=8,zorder=5)
    xn=x1-0.08*(x1-x0); yn=y1-0.17*(y1-y0)
    ax.annotate("N",xy=(xn,yn+0.09*(y1-y0)),xytext=(xn,yn),
                arrowprops=dict(arrowstyle="-|>",color="#222",lw=2),
                ha="center",va="center",fontsize=11,fontweight="bold",zorder=5)

def field():
    j=pd.read_csv(ROOT/"data/external_validation/jairo_titicaca_wq.csv")
    j.columns=[c.replace("\xf1","n").replace("�","n") for c in j.columns]
    return j.rename(columns={"Latitud":"lat","Longitud":"lon","Zona":"zona","Estacion":"station"})

# ===== FIG 1: Mapa de area de estudio =====
def study_map():
    lake=gpd.read_file(ROOT/"data/lake_boundary/titicaca.gpkg")
    j=field()
    j=j.rename(columns={"Transparencia_m":"secchi"})
    st=j.dropna(subset=["secchi"]).groupby("station").agg(
        lat=("lat","mean"),lon=("lon","mean"),secchi=("secchi","mean")).reset_index()
    fig,ax=plt.subplots(figsize=(6.8,6.8))
    lake.plot(ax=ax,color="#eaf4f9",edgecolor="#3b6e8f",linewidth=0.8,zorder=1)
    sccoll=ax.scatter(st.lon,st.lat,c=st.secchi,s=38,cmap="YlGnBu",vmin=2,vmax=14,
                      edgecolor="#333",linewidth=0.3,zorder=3)
    cb=fig.colorbar(sccoll,ax=ax,fraction=0.040,pad=0.02)
    cb.set_label("Mean in-situ Secchi depth (m)")
    ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")
    ax.set_title("Study area — Lake Titicaca\n%d in-situ monitoring stations (IMARPE/ALT), coloured by transparency"%len(st))
    ax.annotate("Bahía de Puno",(-69.92,-15.74),fontsize=8,color="#8a2436",ha="center",
                bbox=dict(boxstyle="round,pad=0.2",fc="white",ec="none",alpha=0.7))
    ax.annotate("Puno",(-70.02,-15.84),fontsize=7,color="#444",ha="right")
    ax.set_aspect("equal"); ax.grid(True,alpha=0.25)
    add_cartography(ax,bar_km=25)
    fig.tight_layout(); fig.savefig(FIG/"fig_study_area.png",bbox_inches="tight"); plt.close(fig)
    print(f"  mapa estudio: {len(st)} estaciones (coloreadas por Secchi medio)")

# ===== FIG 2: Firma espectral del agua por sensor =====
def spectral_signature():
    fr=[]
    for f,s in [("matchups_s2.csv","Sentinel-2"),("matchups_ls.csv","Landsat (harmonized)")]:
        d=pd.read_csv(PROC/f); d=d[d.matched==1].copy(); d["sensor"]=s; fr.append(d)
    df=pd.concat(fr,ignore_index=True)
    bands=["B2","B3","B4","B8","B11","B12"]; wl=[490,560,665,842,1610,2190]
    fig,ax=plt.subplots(figsize=(6.4,4.4))
    for s,c in [("Sentinel-2","#2e86ab"),("Landsat (harmonized)","#d1495b")]:
        sub=df[df.sensor==s]
        mean=[sub[b].mean() for b in bands]; sd=[sub[b].std() for b in bands]
        ax.errorbar(wl,mean,yerr=sd,marker="o",color=c,capsize=3,label=f"{s} (n={len(sub)})",lw=1.8,ms=5)
    ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Surface reflectance")
    ax.set_title("Mean water-leaving reflectance by sensor\n(cross-sensor consistency over Lake Titicaca)")
    ax.legend(frameon=False,fontsize=8); ax.grid(True,alpha=0.25)
    fig.tight_layout(); fig.savefig(FIG/"fig_spectral_signature.png",bbox_inches="tight"); plt.close(fig)
    print("  firma espectral por sensor lista")

if __name__=="__main__":
    study_map(); spectral_signature()
    print("OK")
