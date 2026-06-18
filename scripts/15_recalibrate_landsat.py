"""
==============================================================================
15_recalibrate_landsat.py
Recalibracion radiometrica LOCAL de Landsat -> Sentinel-2 sobre AGUA del
Lago Titicaca por emparejamiento de distribuciones (CDF/quantile matching),
en lugar de los coeficientes Roy et al. (2016) calibrados sobre vegetacion.

Justificacion fisica: el offset NIR de Roy infla el NIR de Landsat sobre agua
oscura (NDWI Landsat = -0.37, fisicamente imposible para agua). El CDF matching
alinea cada banda de Landsat a la distribucion de S2 sobre el mismo lago.
NO usa las etiquetas Secchi -> es correccion del sensor, no leakage del target.

Mide el efecto con la misma validacion GroupKFold por estacion.
==============================================================================
"""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

ROOT=Path(__file__).resolve().parent.parent; PROC=ROOT/"data/processed"
COMMON=["B2","B3","B4","B8","B11","B12"]

def cdf_match(src, ref):
    """Mapea valores de src a la distribucion de ref (quantile mapping)."""
    src=np.asarray(src,float)
    order=np.argsort(np.argsort(src))            # rango de cada valor
    q=(order+0.5)/len(src)
    ref_sorted=np.sort(np.asarray(ref,float))
    return np.interp(q, (np.arange(len(ref_sorted))+0.5)/len(ref_sorted), ref_sorted)

def add_idx(d):
    e=1e-9
    d["NDWI"]=(d.B3-d.B8)/(d.B3+d.B8+e); d["NDTI"]=(d.B4-d.B3)/(d.B4+d.B3+e)
    d["B2_B3"]=d.B2/(d.B3+e); d["B4_B3"]=d.B4/(d.B3+e); d["B3_B2"]=d.B3/(d.B2+e)
    d["B2_B4"]=d.B2/(d.B4+e)
    return d
FEATS=COMMON+["NDWI","NDTI","B2_B3","B4_B3","B3_B2","B2_B4"]

def load():
    s2=pd.read_csv(PROC/"matchups_s2.csv"); s2=s2[s2.matched==1].copy(); s2["sensor"]="S2"
    ls=pd.read_csv(PROC/"matchups_ls.csv"); ls=ls[ls.matched==1].copy(); ls["sensor"]="LS"
    return s2,ls

def cv_rf(df,feats):
    s=df[df.secchi.notna()].dropna(subset=feats)
    X=s[feats].values; y=s.secchi.values; g=s.station.values
    gkf=GroupKFold(5); yt,yp=[],[]
    for tr,te in gkf.split(X,y,g):
        sc=RobustScaler().fit(X[tr])
        m=RandomForestRegressor(n_estimators=500,max_depth=12,min_samples_leaf=3,random_state=42,n_jobs=-1)
        m.fit(sc.transform(X[tr]),y[tr]); yp.append(np.clip(m.predict(sc.transform(X[te])),0.05,None)); yt.append(y[te])
    yt=np.concatenate(yt); yp=np.concatenate(yp)
    return r2_score(yt,yp), float(np.sqrt(mean_squared_error(yt,yp))), len(yt), s.station.nunique()

def main():
    s2,ls=load()
    # referencia S2 por banda (solo agua: NDWI_S2>0 ya se cumple casi siempre)
    s2i=add_idx(s2.copy())
    print("ANTES de recalibrar:")
    print(f"  Landsat NDWI mediana={add_idx(ls.copy()).NDWI.median():+.3f} (deberia ser >0)")

    ls_cal=ls.copy()
    for b in COMMON:
        ls_cal[b]=cdf_match(ls[b].values, s2[b].values)   # alinea LS a distribucion S2
    ls_cal=add_idx(ls_cal)
    print(f"  Landsat NDWI mediana recalibrado={ls_cal.NDWI.median():+.3f}")
    print(f"  Landsat con NDWI>0 ahora: {(ls_cal.NDWI>0).sum()}/{len(ls_cal)} "
          f"({100*(ls_cal.NDWI>0).mean():.0f}%)\n")

    # guardar dataset combinado recalibrado
    comb=pd.concat([s2i,ls_cal],ignore_index=True)
    comb.to_csv(PROC/"matchups_combined_calibrated.csv",index=False)

    print("=== R2 (RF, GroupKFold por estacion) ===")
    r=cv_rf(s2i,FEATS);          print(f"  S2 solo                       R2={r[0]:+.3f} RMSE={r[1]:.2f} n={r[2]} est={r[3]}")
    r=cv_rf(add_idx(ls.copy()),FEATS); print(f"  LS (Roy, sin recalibrar)      R2={r[0]:+.3f} RMSE={r[1]:.2f} n={r[2]} est={r[3]}")
    r=cv_rf(ls_cal,FEATS);       print(f"  LS recalibrado                R2={r[0]:+.3f} RMSE={r[1]:.2f} n={r[2]} est={r[3]}")
    r=cv_rf(pd.concat([s2i,add_idx(ls.copy())],ignore_index=True),FEATS)
    print(f"  Combinado (Roy)               R2={r[0]:+.3f} RMSE={r[1]:.2f} n={r[2]} est={r[3]}")
    r=cv_rf(comb,FEATS);         print(f"  Combinado RECALIBRADO          R2={r[0]:+.3f} RMSE={r[1]:.2f} n={r[2]} est={r[3]}")

if __name__=="__main__":
    main()
