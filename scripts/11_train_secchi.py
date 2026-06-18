"""
==============================================================================
11_train_secchi.py
Modelo HONESTO de recuperacion de transparencia (Secchi) en el Lago Titicaca
a partir de reflectancia Sentinel-2 real (match-ups satelite-campo).

Salvaguardas (lo que fallo en el estudio anterior):
 - Features ESPECTRALES (no coordenadas) -> recupera de la optica, no memoriza estacion
 - GroupKFold por estacion -> la misma estacion NO aparece en train y test
 - Scaler ajustado SOLO en train de cada fold (sin leakage)
 - Comparacion justa vs algoritmo clasico de Secchi (mismo conjunto)

Salida: results/metrics/secchi_cv_metrics.csv, results/figures/secchi_scatter.png
==============================================================================
"""
import pandas as pd, numpy as np, json
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
try:
    from xgboost import XGBRegressor
    HAS_XGB=True
except Exception:
    HAS_XGB=False

ROOT=Path(__file__).resolve().parent.parent
MET=ROOT/"results/metrics"; FIG=ROOT/"results/figures"
MET.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True)

S2=["B1","B2","B3","B4","B5","B6","B7","B8","B8A","B11","B12"]

def add_indices(d):
    e=1e-9
    d["NDCI"]=(d.B5-d.B4)/(d.B5+d.B4+e)
    d["NDTI"]=(d.B4-d.B3)/(d.B4+d.B3+e)
    d["NDWI"]=(d.B3-d.B8)/(d.B3+d.B8+e)
    d["B2_B3"]=d.B2/(d.B3+e)      # azul/verde (clasico Secchi)
    d["B4_B3"]=d.B4/(d.B3+e)
    d["B3_B2"]=d.B3/(d.B2+e)
    d["B5_B4"]=d.B5/(d.B4+e)
    return d

FEATURES=S2+["NDCI","NDTI","NDWI","B2_B3","B4_B3","B3_B2","B5_B4"]

def metrics(y,p):
    return dict(R2=r2_score(y,p),
                RMSE=float(np.sqrt(mean_squared_error(y,p))),
                MAE=float(mean_absolute_error(y,p)),
                n=len(y))

def cv_eval(model_fn, X, y, groups, log=False):
    gkf=GroupKFold(n_splits=5); yt,yp=[],[]
    for tr,te in gkf.split(X,y,groups):
        sc=RobustScaler().fit(X[tr])           # fit SOLO en train
        m=model_fn();
        ytr = np.log1p(y[tr]) if log else y[tr]
        m.fit(sc.transform(X[tr]), ytr)
        pr=m.predict(sc.transform(X[te]))
        pr = np.expm1(pr) if log else pr
        yt.append(y[te]); yp.append(pr)
    yt=np.concatenate(yt); yp=np.concatenate(yp)
    return metrics(yt,yp), yt, yp

def main():
    mp=ROOT/"data/processed/matchups_s2.csv"
    df=pd.read_csv(mp)
    df=df[(df["matched"]==1)&(df["secchi"].notna())].copy()
    df=add_indices(df).dropna(subset=FEATURES)
    print(f"[secchi] {len(df)} match-ups con Secchi | estaciones={df['station'].nunique()}")
    print(f"  Secchi rango {df.secchi.min():.1f}-{df.secchi.max():.1f} m, mediana {df.secchi.median():.1f}")
    X=df[FEATURES].values; y=df["secchi"].values; g=df["station"].values

    results={}
    # baseline clasico: Secchi ~ ratio azul/verde (regresion log-lineal 1 feature)
    Xc=df[["B2_B3"]].values
    mc,_,_=cv_eval(lambda:LinearRegression(), Xc, y, g, log=True)
    results["Classical_B2B3_loglin"]=mc
    # ML
    results["LinearReg_full"],_,_=cv_eval(lambda:LinearRegression(),X,y,g,log=True)
    results["RandomForest"],yt,yp=cv_eval(
        lambda:RandomForestRegressor(n_estimators=400,max_depth=12,
                                     min_samples_leaf=3,random_state=42,n_jobs=-1),X,y,g)
    if HAS_XGB:
        results["XGBoost"],yt2,yp2=cv_eval(
            lambda:XGBRegressor(n_estimators=500,max_depth=5,learning_rate=0.03,
                                subsample=0.8,colsample_bytree=0.8,random_state=42),X,y,g)

    print("\n=== Secchi retrieval (GroupKFold por estacion, sin leakage) ===")
    for k,v in results.items():
        print(f"  {k:26s} R2={v['R2']:+.3f}  RMSE={v['RMSE']:.2f} m  MAE={v['MAE']:.2f}  n={v['n']}")
    pd.DataFrame(results).T.to_csv(MET/"secchi_cv_metrics.csv")
    print(f"\n  guardado: {MET/'secchi_cv_metrics.csv'}")

    # scatter del mejor (RF)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(5,5))
        ax.scatter(yt,yp,s=18,alpha=0.5,edgecolor="none")
        lim=[0,max(yt.max(),yp.max())*1.05]; ax.plot(lim,lim,"k--",lw=1)
        ax.set_xlabel("Secchi in-situ (m)"); ax.set_ylabel("Secchi predicho (m)")
        ax.set_title(f"RF  R2={results['RandomForest']['R2']:.2f}")
        ax.set_xlim(lim); ax.set_ylim(lim); fig.tight_layout()
        fig.savefig(FIG/"secchi_scatter.png",dpi=150)
        print(f"  figura: {FIG/'secchi_scatter.png'}")
    except Exception as e:
        print("  (sin figura:",repr(e)[:60],")")

if __name__=="__main__":
    main()
