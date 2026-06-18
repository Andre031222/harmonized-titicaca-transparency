"""
==============================================================================
20_transparency_map.py
Mapa de transparencia (Secchi) del Lago Titicaca: aplica el modelo RF de
produccion a un composite Sentinel-2 de estacion seca (2022), pixel a pixel.
Demuestra la utilidad operacional del modelo (de validacion a herramienta).
==============================================================================
"""
import ee, numpy as np, requests, io, json, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import rasterio, joblib
import geopandas as gpd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent.parent
FIG=ROOT/"results/figures"; MOD=ROOT/"results/models"
ee.Initialize(project="black-display-445217-p3")

AOI=ee.Geometry.Rectangle([-70.05,-16.62,-68.56,-15.22])
BANDS=["B2","B3","B4","B8","B11","B12"]; SCALE=500

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

def mask_s2(img):
    scl=img.select("SCL")
    good=(scl.neq(1).And(scl.neq(3)).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11)))
    return img.updateMask(good.And(img.select("MSK_CLDPRB").lt(30)))

def main():
    print("[map] componiendo S2 estacion seca 2022...")
    comp=(ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(AOI)
          .filterDate("2022-05-01","2022-10-31").map(mask_s2).median()
          .select(BANDS).divide(10000))
    water=ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("recurrence").gt(50)
    comp=comp.updateMask(water).clip(AOI)

    url=comp.getDownloadURL({"scale":SCALE,"region":AOI,"format":"GEO_TIFF","bands":BANDS})
    print("[map] descargando raster...")
    r=requests.get(url,timeout=300); r.raise_for_status()
    with rasterio.open(io.BytesIO(r.content)) as ds:
        arr=ds.read().astype("float32")            # (6,H,W)
        bounds=ds.bounds; nod=ds.nodata
    arr=np.where(arr==nod,np.nan,arr) if nod is not None else arr
    B={b:arr[i] for i,b in enumerate(BANDS)}
    H,W=arr.shape[1],arr.shape[2]; print(f"[map] raster {H}x{W}")

    # features (mismas del modelo)
    e=1e-9
    feat={**B}
    feat["NDWI"]=(B["B3"]-B["B8"])/(B["B3"]+B["B8"]+e)
    feat["NDTI"]=(B["B4"]-B["B3"])/(B["B4"]+B["B3"]+e)
    feat["B2_B3"]=B["B2"]/(B["B3"]+e); feat["B4_B3"]=B["B4"]/(B["B3"]+e)
    feat["B3_B2"]=B["B3"]/(B["B2"]+e); feat["B2_B4"]=B["B2"]/(B["B4"]+e)
    P=joblib.load(MOD/"secchi_rf_production.joblib"); F=P["features"]
    stack=np.stack([feat[f] for f in F],axis=-1).reshape(-1,len(F))
    valid=~np.isnan(stack).any(axis=1) & (feat["NDWI"].reshape(-1)>0)  # solo agua valida
    pred=np.full(stack.shape[0],np.nan)
    if valid.sum()>0:
        Xs=P["scaler"].transform(stack[valid])
        pred[valid]=np.clip(P["model"].predict(Xs),0.05,18)
    secchi=pred.reshape(H,W)
    print(f"[map] pixeles de agua predichos: {valid.sum()} | Secchi {np.nanmin(secchi):.1f}-{np.nanmax(secchi):.1f}m")

    # plot
    lake=gpd.read_file(ROOT/"data/lake_boundary/titicaca.gpkg")
    ext=[bounds.left,bounds.right,bounds.bottom,bounds.top]
    fig,ax=plt.subplots(figsize=(7,7))
    lake.boundary.plot(ax=ax,color="#3b6e8f",linewidth=0.7,zorder=2)
    im=ax.imshow(secchi,extent=ext,origin="upper",cmap="YlGnBu",vmin=2,vmax=14,zorder=1)
    cb=fig.colorbar(im,ax=ax,fraction=0.040,pad=0.02); cb.set_label("Predicted Secchi depth (m)")
    ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")
    ax.set_title("Modelled water transparency — Lake Titicaca\nSentinel-2 dry-season composite (2022), Random Forest")
    ax.set_xlim(ext[0],ext[1]); ax.set_ylim(ext[2],ext[3]); ax.set_aspect("equal"); ax.grid(True,alpha=0.2)
    add_cartography(ax,bar_km=25)
    fig.tight_layout(); fig.savefig(FIG/"fig_transparency_map.png",bbox_inches="tight",dpi=600); plt.close(fig)
    print(f"[map] guardado: {FIG/'fig_transparency_map.png'}")

if __name__=="__main__":
    main()
