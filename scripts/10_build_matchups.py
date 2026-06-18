"""
==============================================================================
10_build_matchups.py
Construye el dataset REAL de pares satelite-campo (match-ups) para el
Lago Titicaca, emparejando cada medicion in-situ (IMARPE/ALT - Jairo et al.)
con la reflectancia Sentinel-2 limpia mas cercana en +-WINDOW dias.

Sin datos sinteticos. Reflectancia extraida en la coordenada real de cada
medicion, con enmascaramiento de nubes (SCL + MSK_CLDPRB).

Salida: data/processed/matchups_s2.csv
==============================================================================
"""
import ee, pandas as pd, numpy as np, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/processed"
OUT.mkdir(parents=True, exist_ok=True)

ee.Initialize(project="black-display-445217-p3")

WINDOW = 10          # +- dias
BUFFER = 30          # m radio de extraccion
CLDPRB = 30          # umbral MSK_CLDPRB
YEAR_MIN = 2016      # era Sentinel-2

S2_BANDS = ["B1","B2","B3","B4","B5","B6","B7","B8","B8A","B11","B12"]

def load_field():
    j = pd.read_csv(ROOT/"data/external_validation/jairo_titicaca_wq.csv")
    j.columns = [c.replace("\xf1","n").replace("�","n") for c in j.columns]
    M={"ENE":1,"FEB":2,"MAR":3,"ABR":4,"MAY":5,"JUN":6,"JUL":7,"AGO":8,
       "SET":9,"SEP":9,"OCT":10,"NOV":11,"DIC":12}
    j["mon"]=j["Mes"].astype(str).str.upper().str[:3].map(M)
    j["date"]=pd.to_datetime(dict(year=j["Ano"],month=j["mon"],day=j["Dia"]),errors="coerce")
    j=j.rename(columns={"Latitud":"lat","Longitud":"lon","Clorofila-A":"chl",
                        "Transparencia_m":"secchi","SST":"tss","Temperatura":"temp_insitu",
                        "Ph":"ph","Oxigeno_disuelto":"do","Zona":"zona","Estacion":"station"})
    j=j.dropna(subset=["date","lat","lon"])
    j=j[j["date"].dt.year>=YEAR_MIN].reset_index(drop=True)
    return j

def s2_at(lat, lon, date):
    pt=ee.Geometry.Point([lon,lat]); d0=ee.Date(date.strftime("%Y-%m-%d"))
    col=(ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(pt)
         .filterDate(d0.advance(-WINDOW,"day"), d0.advance(WINDOW,"day")))
    n=col.size().getInfo()
    if n==0: return None
    def mask(img):
        scl=img.select("SCL")
        good=(scl.neq(1).And(scl.neq(3)).And(scl.neq(8)).And(scl.neq(9))
              .And(scl.neq(10)).And(scl.neq(11)))
        return img.updateMask(good.And(img.select("MSK_CLDPRB").lt(CLDPRB)))
    comp=col.map(mask).median().select(S2_BANDS).divide(10000)
    v=comp.reduceRegion(ee.Reducer.mean(), pt.buffer(BUFFER), 20).getInfo()
    if not v or v.get("B4") is None: return None
    v["n_img"]=n
    return v

COLS=["row_id","station","zona","date","lat","lon","secchi","chl","tss",
      "temp_insitu","ph","do","matched","n_img"]+S2_BANDS

def main():
    j=load_field()
    print(f"[matchups] {len(j)} mediciones in-situ era-S2 (>= {YEAR_MIN})")
    out_path=OUT/"matchups_s2.csv"
    done=set()
    if out_path.exists():
        prev=pd.read_csv(out_path); done=set(prev["row_id"].tolist())
        print(f"[matchups] reanudando: {len(done)} ya hechos")
    rows=[]; ok=0; t0=time.time()
    for i,r in j.iterrows():
        if i in done: continue
        try: v=s2_at(r["lat"],r["lon"],r["date"])
        except Exception as e:
            print(f"  [{i}] ERR {repr(e)[:70]}"); v=None; time.sleep(1)
        rec={"row_id":i,"station":r["station"],"zona":r["zona"],
             "date":r["date"].strftime("%Y-%m-%d"),"lat":r["lat"],"lon":r["lon"],
             "secchi":r["secchi"],"chl":r["chl"],"tss":r["tss"],
             "temp_insitu":r["temp_insitu"],"ph":r["ph"],"do":r["do"],"matched":0}
        if v:
            ok+=1; rec["matched"]=1; rec["n_img"]=v["n_img"]
            for b in S2_BANDS: rec[b]=round(v[b],5)
        rows.append(rec)
        if len(rows)%25==0:
            pd.DataFrame(rows).reindex(columns=COLS).to_csv(out_path, mode="a", header=not out_path.exists(), index=False)
            print(f"  {i+1}/{len(j)} | matched={ok} | {time.time()-t0:.0f}s"); rows=[]
    if rows:
        pd.DataFrame(rows).reindex(columns=COLS).to_csv(out_path, mode="a", header=not out_path.exists(), index=False)
    final=pd.read_csv(out_path)
    m=final[final["matched"]==1]
    print(f"\n=== LISTO: {len(final)} intentos, {len(m)} matchups ({100*len(m)/len(final):.0f}%) ===")
    print(f"  con Secchi: {m['secchi'].notna().sum()} | con Chl: {m['chl'].notna().sum()}")
    print(f"  guardado: {out_path}")

if __name__=="__main__":
    main()
