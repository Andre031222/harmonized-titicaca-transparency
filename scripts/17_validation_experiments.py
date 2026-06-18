"""
==============================================================================
17_validation_experiments.py
Demuestra EMPIRICAMENTE (con nuestros datos) dos efectos sobre el R2 reportado:

  (A) Diseno de validacion: split ALEATORIO vs GroupKFold por ESTACION. En ESTE
      dataset ambos dan un R2 IDENTICO -> NO hay fuga por autocorrelacion
      espacial a nivel estacion (el modelo no memoriza la ubicacion). La prueba
      de extrapolacion mas dura (leave-one-zone-out) esta en 21_validation_q1.py.
  (B) Tipo/rango de agua: el R2 depende del rango de transparencia muestreado
      (range restriction). Aguas homogeneas (solo claras) -> R2 mas bajo;
      rango amplio (turbio+claro) -> R2 mas alto, aunque el error absoluto sea
      similar -> comparar RMSE, no solo R2.

NOTA: la suite de validacion completa y honesta (jerarquia random/estacion/
temporal/zona + incertidumbre conformal) esta en 21_validation_q1.py y
22_uncertainty_benchmark.py; este script se conserva por trazabilidad.

Salida: results/metrics/validation_experiments.csv
==============================================================================
"""
import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

ROOT=Path(__file__).resolve().parent.parent; PROC=ROOT/"data/processed"; MET=ROOT/"results/metrics"

def prep(d):
    e=1e-9
    d["NDWI"]=(d.B3-d.B8)/(d.B3+d.B8+e); d["NDTI"]=(d.B4-d.B3)/(d.B4+d.B3+e)
    d["B2_B3"]=d.B2/(d.B3+e); d["B4_B3"]=d.B4/(d.B3+e); d["B3_B2"]=d.B3/(d.B2+e); d["B2_B4"]=d.B2/(d.B4+e)
    return d
F=["B2","B3","B4","B8","B11","B12","NDWI","NDTI","B2_B3","B4_B3","B3_B2","B2_B4"]

def load():
    fr=[]
    for f,s in [("matchups_s2.csv","S2"),("matchups_ls.csv","LS")]:
        d=pd.read_csv(PROC/f); d=d[d.matched==1].copy(); d["sensor"]=s; fr.append(d)
    return prep(pd.concat(fr,ignore_index=True))

def rf(): return RandomForestRegressor(n_estimators=500,max_depth=12,min_samples_leaf=3,random_state=42,n_jobs=-1)

def run(splitter, X, y, groups=None):
    yt,yp=[],[]
    it = splitter.split(X,y,groups) if groups is not None else splitter.split(X,y)
    for tr,te in it:
        sc=RobustScaler().fit(X[tr]); m=rf(); m.fit(sc.transform(X[tr]),y[tr])
        yp.append(np.clip(m.predict(sc.transform(X[te])),0.05,None)); yt.append(y[te])
    yt=np.concatenate(yt); yp=np.concatenate(yp)
    return r2_score(yt,yp), float(np.sqrt(mean_squared_error(yt,yp))), float(mean_absolute_error(yt,yp))

def main():
    df=load(); s=df[df.secchi.notna()].dropna(subset=F)
    X=s[F].values; y=s.secchi.values; g=s.station.values
    rows=[]
    print("="*64)
    print("EXPERIMENTO A: diseno de validacion (mismo modelo, mismos datos)")
    print("="*64)
    r2r,rmser,maer=run(KFold(5,shuffle=True,random_state=42),X,y)
    r2g,rmseg,maeg=run(GroupKFold(5),X,y,g)
    print(f"  Split ALEATORIO                   R2={r2r:+.3f} RMSE={rmser:.2f}m MAE={maer:.2f}")
    print(f"  GroupKFold por ESTACION (honesto) R2={r2g:+.3f} RMSE={rmseg:.2f}m MAE={maeg:.2f}")
    print(f"  --> Diferencia random vs estacion: {r2r-r2g:+.3f} R2 -> SIN fuga espacial a nivel estacion")
    rows+=[{"exp":"validation","case":"random_kfold","R2":r2r,"RMSE":rmser,"MAE":maer,"n":len(y)},
           {"exp":"validation","case":"groupkfold_station","R2":r2g,"RMSE":rmseg,"MAE":maeg,"n":len(y)}]

    print("\n"+"="*64)
    print("EXPERIMENTO B: rango/tipo de agua (GroupKFold por estacion en cada estrato)")
    print("="*64)
    # estratos por transparencia (mas turbio = Secchi bajo; mas claro = Secchi alto)
    med=np.median(y)
    for tag,mask in [("Rango COMPLETO (turbio+claro)", np.ones(len(y),bool)),
                     (f"Mas TURBIO (Secchi<{med:.1f}m)", y<med),
                     (f"Mas CLARO (Secchi>={med:.1f}m)", y>=med)]:
        Xs,ys,gs=X[mask],y[mask],g[mask]
        if len(np.unique(gs))<5:
            print(f"  {tag:34s} (estaciones insuficientes)"); continue
        r2,rmse,mae=run(GroupKFold(5),Xs,ys,gs)
        print(f"  {tag:34s} R2={r2:+.3f} RMSE={rmse:.2f}m  rango_y={ys.min():.1f}-{ys.max():.1f}m  sd={ys.std():.2f}  n={len(ys)}")
        rows.append({"exp":"water_range","case":tag,"R2":r2,"RMSE":rmse,"MAE":mae,"n":len(ys),"sd_y":float(ys.std())})

    pd.DataFrame(rows).to_csv(MET/"validation_experiments.csv",index=False)
    print(f"\nGuardado: {MET/'validation_experiments.csv'}")
    print("\nLECTURA: (A) random ~ estacion -> NO hay fuga espacial a nivel estacion;")
    print("la extrapolacion dura (leave-one-zone-out) si colapsa el R2 (ver script 21).")
    print("(B) el R2 sube con el rango/desviacion de la transparencia (range")
    print("restriction): aguas homogeneas dan R2 menor aunque el RMSE sea similar")
    print("-> comparar RMSE, no solo R2.")

if __name__=="__main__":
    main()
