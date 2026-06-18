"""
==============================================================================
22_uncertainty_benchmark.py
Refuerzos de nivel Q1 sobre el retrieval de transparencia (Secchi):

 (A) CUANTIFICACION DE INCERTIDUMBRE por prediccion:
     - Intervalos de prediccion de Random Forest via la dispersion entre
       arboles (quantile-RF empirico).
     - Conformal prediction (split conformal, cobertura garantizada).
     Se reporta la COBERTURA EMPIRICA out-of-fold (un PI al 90% debe cubrir
     ~90% de las observaciones) y el ancho medio del intervalo.

 (B) BENCHMARK numerico cabeza-a-cabeza (mismo GroupKFold por estacion):
     algoritmo clasico azul/verde, lineal multibanda, y Random Forest;
     RMSE, MAE, bias y sesgo por zona trofica. Materializa el argumento
     "RMSE es la metrica justa" con numeros, no retorica.

Salidas: results/metrics/uncertainty_metrics.json, benchmark_models.csv
==============================================================================
"""
import pandas as pd, numpy as np, json, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data/processed"; MET = ROOT / "results/metrics"
MET.mkdir(parents=True, exist_ok=True)

F = ["B2","B3","B4","B8","B11","B12","NDWI","NDTI","B2_B3","B4_B3","B3_B2","B2_B4"]
ZONES = ["BAHIA PUNO","LAGO MENOR","LAGO MAYOR"]

def prep(d):
    e=1e-9
    d["NDWI"]=(d.B3-d.B8)/(d.B3+d.B8+e); d["NDTI"]=(d.B4-d.B3)/(d.B4+d.B3+e)
    d["B2_B3"]=d.B2/(d.B3+e); d["B4_B3"]=d.B4/(d.B3+e); d["B3_B2"]=d.B3/(d.B2+e); d["B2_B4"]=d.B2/(d.B4+e)
    return d

def load():
    fr=[]
    for f,s in [("matchups_s2.csv","S2"),("matchups_ls.csv","LS")]:
        d=pd.read_csv(PROC/f); d=d[d.matched==1].copy(); d["sensor"]=s; fr.append(d)
    df=prep(pd.concat(fr,ignore_index=True))
    return df[df.secchi.notna()].dropna(subset=F).reset_index(drop=True)

def rf():
    return RandomForestRegressor(n_estimators=500, max_depth=12, min_samples_leaf=3,
                                 random_state=42, n_jobs=-1)

def main():
    s=load(); X=s[F].values; y=s.secchi.values; g=s.station.values
    gkf=GroupKFold(5)
    n=len(y)
    oof=np.full(n,np.nan); lo=np.full(n,np.nan); hi=np.full(n,np.nan)
    conf_lo=np.full(n,np.nan); conf_hi=np.full(n,np.nan)
    ALPHA=0.10  # PI al 90%

    # ---- (A) Incertidumbre: quantile-RF entre arboles + split-conformal ----
    for tr,te in gkf.split(X,y,g):
        sc=RobustScaler().fit(X[tr]); Xtr=sc.transform(X[tr]); Xte=sc.transform(X[te])
        # sub-split del train para conformal (80/20)
        rng=np.random.RandomState(42); perm=rng.permutation(len(tr))
        cal=perm[:max(20,len(tr)//5)]; fit=perm[len(cal):]
        m=rf(); m.fit(Xtr[fit], y[tr][fit])
        # prediccion puntual y dispersion entre arboles
        per_tree_te=np.stack([t.predict(Xte) for t in m.estimators_],axis=0)  # (T, n_te)
        pred_te=per_tree_te.mean(0)
        oof[te]=np.clip(pred_te,0.05,None)
        lo[te]=np.clip(np.percentile(per_tree_te,100*ALPHA/2,axis=0),0.05,None)
        hi[te]=np.percentile(per_tree_te,100*(1-ALPHA/2),axis=0)
        # split-conformal: residuos absolutos en calibracion -> cuantil
        pred_cal=m.predict(Xtr[cal]); resid=np.abs(y[tr][cal]-pred_cal)
        q=np.quantile(resid, 1-ALPHA)
        conf_lo[te]=np.clip(pred_te-q,0.05,None); conf_hi[te]=pred_te+q

    R2=r2_score(y,oof); RMSE=float(np.sqrt(mean_squared_error(y,oof))); MAE=float(mean_absolute_error(y,oof))
    cov_qrf=float(np.mean((y>=lo)&(y<=hi))); width_qrf=float(np.mean(hi-lo))
    cov_conf=float(np.mean((y>=conf_lo)&(y<=conf_hi))); width_conf=float(np.mean(conf_hi-conf_lo))
    print("=== (A) Incertidumbre (out-of-fold, PI nominal 90%) ===")
    print(f"  OOF (cal-split): R2={R2:.3f} RMSE={RMSE:.2f} MAE={MAE:.2f} n={n}")
    print(f"  Quantile-RF :  cobertura={cov_qrf*100:.1f}%  ancho medio={width_qrf:.2f} m")
    print(f"  Conformal   :  cobertura={cov_conf*100:.1f}%  ancho medio={width_conf:.2f} m")
    unc={"PI_nominal":1-ALPHA,"n":n,"R2_oof_calsplit":round(R2,3),"RMSE":round(RMSE,3),
         "qrf_coverage":round(cov_qrf,3),"qrf_mean_width_m":round(width_qrf,3),
         "conformal_coverage":round(cov_conf,3),"conformal_mean_width_m":round(width_conf,3)}
    json.dump(unc,open(MET/"uncertainty_metrics.json","w"),indent=2)

    # ---- (B) Benchmark numerico cabeza-a-cabeza ----
    print("\n=== (B) Benchmark numerico (GroupKFold por estacion) ===")
    def cv(Xm,log=False,model="rf"):
        yt,yp=[],[]
        for tr,te in gkf.split(Xm,y,g):
            sc=RobustScaler().fit(Xm[tr])
            mdl = LinearRegression() if model=="lin" else rf()
            mdl.fit(sc.transform(Xm[tr]), np.log1p(y[tr]) if log else y[tr])
            p=mdl.predict(sc.transform(Xm[te])); p=np.expm1(p) if log else p
            yp.append(np.clip(p,0.05,None)); yt.append(y[te])
        return np.concatenate(yt),np.concatenate(yp)
    rows=[]
    configs=[("Classical blue/green (log-linear)", s[["B2_B3"]].values, True, "lin"),
             ("Linear multiband",                  X,                   True, "lin"),
             ("Random Forest (this study)",        X,                   False,"rf")]
    for name,Xm,log,mdl in configs:
        yt,yp=cv(Xm,log=log,model=mdl)
        bias=float(np.mean(yp-yt))
        rows.append({"model":name,"R2":round(r2_score(yt,yp),3),
                     "RMSE":round(float(np.sqrt(mean_squared_error(yt,yp))),3),
                     "MAE":round(float(mean_absolute_error(yt,yp)),3),"bias_m":round(bias,3),"n":len(yt)})
        print(f"  {name:34s} R2={rows[-1]['R2']:+.3f} RMSE={rows[-1]['RMSE']:.2f} bias={bias:+.2f}")
    pd.DataFrame(rows).to_csv(MET/"benchmark_models.csv",index=False)

    # bias por zona del RF (sobre OOF puntual)
    print("\n  RF bias por zona trofica (OOF):")
    zrows=[]
    for z in ZONES:
        mk=s.zona.values==z
        b=float(np.mean(oof[mk]-y[mk])); rm=float(np.sqrt(mean_squared_error(y[mk],oof[mk])))
        zrows.append({"zona":z,"RMSE":round(rm,3),"bias_m":round(b,3),"n":int(mk.sum())})
        print(f"    {z:11s} RMSE={rm:.2f} bias={b:+.2f} n={int(mk.sum())}")
    pd.DataFrame(zrows).to_csv(MET/"benchmark_by_zone.csv",index=False)
    print(f"\nGuardado: uncertainty_metrics.json, benchmark_models.csv, benchmark_by_zone.csv")

if __name__=="__main__":
    main()
