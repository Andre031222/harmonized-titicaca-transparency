"""
==============================================================================
14_improve_secchi.py
Mejoras HONESTAS al modelo de transparencia (Secchi), medidas de forma
incremental con la MISMA validacion (GroupKFold por estacion, sin leakage).
Cada paso se justifica fisicamente a priori; se reporta su efecto real.
==============================================================================
"""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

ROOT=Path(__file__).resolve().parent.parent; PROC=ROOT/"data/processed"

COMMON=["B2","B3","B4","B8","B11","B12"]
def add_idx(d):
    e=1e-9
    d["NDWI"]=(d.B3-d.B8)/(d.B3+d.B8+e); d["NDTI"]=(d.B4-d.B3)/(d.B4+d.B3+e)
    d["B2_B3"]=d.B2/(d.B3+e); d["B4_B3"]=d.B4/(d.B3+e); d["B3_B2"]=d.B3/(d.B2+e)
    d["B2_B4"]=d.B2/(d.B4+e); d["lnB2_B3"]=np.log((d.B2+e)/(d.B3+e))
    d["B3_B4"]=d.B3/(d.B4+e)
    return d
FEATS=COMMON+["NDWI","NDTI","B2_B3","B4_B3","B3_B2","B2_B4","lnB2_B3","B3_B4"]

def load():
    fr=[]
    for f,s in [("matchups_s2.csv","S2"),("matchups_ls.csv","LS")]:
        p=PROC/f
        if p.exists():
            d=pd.read_csv(p); d=d[d.matched==1].copy(); d["sensor"]=s; fr.append(d)
    return add_idx(pd.concat(fr,ignore_index=True))

def cv(model_fn,X,y,g,log=False,k=5):
    gkf=GroupKFold(n_splits=k); yt,yp=[],[]
    for tr,te in gkf.split(X,y,g):
        sc=RobustScaler().fit(X[tr]); m=model_fn()
        m.fit(sc.transform(X[tr]), np.log1p(y[tr]) if log else y[tr])
        p=m.predict(sc.transform(X[te])); p=np.expm1(p) if log else p
        p=np.clip(p,0.05,None); yt.append(y[te]); yp.append(p)
    yt=np.concatenate(yt); yp=np.concatenate(yp)
    return r2_score(yt,yp), float(np.sqrt(mean_squared_error(yt,yp))), float(mean_absolute_error(yt,yp)), len(yt)

def rf(): return RandomForestRegressor(n_estimators=500,max_depth=12,min_samples_leaf=3,random_state=42,n_jobs=-1)

def report(tag,df,feats,log=False):
    s=df[df.secchi.notna()].dropna(subset=feats)
    X=s[feats].values; y=s.secchi.values; g=s.station.values
    r2,rmse,mae,n=cv(rf,X,y,g,log=log)
    print(f"  {tag:42s} R2={r2:+.3f} RMSE={rmse:.2f}m n={n} est={s.station.nunique()}")
    return r2

def main():
    df0=load()
    print(f"[base] {len(df0)} matchups, {df0[df0.secchi.notna()].shape[0]} con Secchi\n")
    print("=== Mejoras incrementales (RF, GroupKFold por estacion) ===")

    # 0) baseline
    base=COMMON+["NDWI","NDTI","B2_B3","B4_B3","B3_B2","B2_B4"]
    report("0) baseline (features comunes)", df0, base)

    # 1) filtro de agua: NDWI>0 y rojo bajo (agua clara, sin orilla/nube)
    df1=df0[(df0.NDWI>0)&(df0.B4<0.06)&(df0.B3<0.12)&(df0.B2>0)].copy()
    print(f"     [filtro agua: {len(df0)}->{len(df1)} matchups]")
    report("1) + filtro calidad de agua (NDWI>0)", df1, base)

    # 2) quitar outliers de Secchi (IQR) y reflectancias negativas
    s=df1.secchi; q1,q3=s.quantile(.02),s.quantile(.98)
    df2=df1[(df1.secchi>=q1)&(df1.secchi<=q3)].copy()
    for b in COMMON: df2=df2[df2[b]>0]
    print(f"     [outliers Secchi/refl: {len(df1)}->{len(df2)}]")
    report("2) + quitar outliers", df2, base)

    # 3) target log(Secchi) (fisica de atenuacion)
    report("3) + target log(Secchi)", df2, base, log=True)

    # 4) features espectrales ampliadas (ratios log, B3_B4)
    report("4) + features ampliadas + log", df2, FEATS, log=True)

    return df2

if __name__=="__main__":
    main()
