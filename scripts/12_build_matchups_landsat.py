"""
==============================================================================
12_build_matchups_landsat.py
Match-ups Landsat 8/9 (Collection 2, Level-2) <-> mediciones in-situ reales.
Reflectancia SR armonizada a equivalente Sentinel-2 (coeficientes Roy et al. 2016)
+ temperatura superficial (banda termica ST_B10).

Bandas comunes con S2 (armonizadas): B2,B3,B4,B8(NIR),B11,B12
Salida: data/processed/matchups_ls.csv
==============================================================================
"""
import ee, pandas as pd, numpy as np, time
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/"data/processed"; OUT.mkdir(parents=True,exist_ok=True)
ee.Initialize(project="black-display-445217-p3")

WINDOW=10; BUFFER=30; YEAR_MIN=2013   # era Landsat-8

# Roy et al. 2016 OLI->MSI: S2 = a + b*OLI  (banda comun -> nombre S2)
ROY={"B2":("SR_B2",0.0183,0.8850),"B3":("SR_B3",0.0123,0.9317),
     "B4":("SR_B4",0.0123,0.9372),"B8":("SR_B5",0.0448,0.8339),
     "B11":("SR_B6",0.0306,0.8639),"B12":("SR_B7",0.0116,0.9165)}

def load_field():
    j=pd.read_csv(ROOT/"data/external_validation/jairo_titicaca_wq.csv")
    j.columns=[c.replace("\xf1","n").replace("�","n") for c in j.columns]
    M={"ENE":1,"FEB":2,"MAR":3,"ABR":4,"MAY":5,"JUN":6,"JUL":7,"AGO":8,
       "SET":9,"SEP":9,"OCT":10,"NOV":11,"DIC":12}
    j["mon"]=j["Mes"].astype(str).str.upper().str[:3].map(M)
    j["date"]=pd.to_datetime(dict(year=j["Ano"],month=j["mon"],day=j["Dia"]),errors="coerce")
    j=j.rename(columns={"Latitud":"lat","Longitud":"lon","Clorofila-A":"chl",
                        "Transparencia_m":"secchi","SST":"tss","Temperatura":"temp_insitu",
                        "Ph":"ph","Oxigeno_disuelto":"do","Zona":"zona","Estacion":"station"})
    j=j.dropna(subset=["date","lat","lon"])
    return j[j["date"].dt.year>=YEAR_MIN].reset_index(drop=True)

def mask_ls(img):
    qa=img.select("QA_PIXEL")
    # bits: 1 dilated, 3 cloud, 4 shadow. NO se filtra cirrus (bit 2):
    # a 3810 m el flag cirrus esta casi siempre activo y enmascararia todo.
    clear=(qa.bitwiseAnd(1<<1).eq(0).And(qa.bitwiseAnd(1<<3).eq(0))
           .And(qa.bitwiseAnd(1<<4).eq(0)))
    return img.updateMask(clear)

def ls_at(lat,lon,date):
    pt=ee.Geometry.Point([lon,lat]); d0=ee.Date(date.strftime("%Y-%m-%d"))
    col=(ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
         .merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
         .filterBounds(pt).filterDate(d0.advance(-WINDOW,"day"),d0.advance(WINDOW,"day")))
    n=col.size().getInfo()
    if n==0: return None
    comp=col.map(mask_ls).median()
    srbands=[v[0] for v in ROY.values()]
    v=comp.select(srbands+["ST_B10"]).reduceRegion(ee.Reducer.mean(),pt.buffer(BUFFER),30).getInfo()
    if not v or v.get("SR_B4") is None: return None
    out={"n_img":n}
    for s2b,(srb,a,b) in ROY.items():
        sr=v[srb]*0.0000275-0.2          # escala SR L2
        out[s2b]=round(a+b*sr,5)          # armonizado a S2
    if v.get("ST_B10") is not None:
        out["lst_C"]=round(v["ST_B10"]*0.00341802+149.0-273.15,2)  # Kelvin->C
    return out

COLS=["row_id","station","zona","date","lat","lon","secchi","chl","tss",
      "temp_insitu","sensor","matched","n_img","B2","B3","B4","B8","B11","B12","lst_C"]

def main():
    j=load_field(); print(f"[ls] {len(j)} mediciones era-Landsat (>= {YEAR_MIN})")
    out_path=OUT/"matchups_ls.csv"; rows=[]; ok=0; t0=time.time()
    for i,r in j.iterrows():
        try: v=ls_at(r["lat"],r["lon"],r["date"])
        except Exception as e: print(f"  [{i}] ERR {repr(e)[:60]}"); v=None; time.sleep(1)
        rec={"row_id":i,"station":r["station"],"zona":r["zona"],
             "date":r["date"].strftime("%Y-%m-%d"),"lat":r["lat"],"lon":r["lon"],
             "secchi":r["secchi"],"chl":r["chl"],"tss":r["tss"],
             "temp_insitu":r["temp_insitu"],"sensor":"LS","matched":0}
        if v:
            ok+=1; rec["matched"]=1; rec.update(v)
        rows.append(rec)
        if len(rows)%25==0:
            pd.DataFrame(rows).reindex(columns=COLS).to_csv(out_path,mode="a",header=not out_path.exists(),index=False)
            print(f"  {i+1}/{len(j)} | matched={ok} | {time.time()-t0:.0f}s"); rows=[]
    if rows: pd.DataFrame(rows).reindex(columns=COLS).to_csv(out_path,mode="a",header=not out_path.exists(),index=False)
    final=pd.read_csv(out_path); m=final[final["matched"]==1]
    print(f"\n=== LISTO: {len(final)} intentos, {len(m)} matchups ===")
    print(f"  con Secchi: {m['secchi'].notna().sum()} | con temp_insitu: {m['temp_insitu'].notna().sum()} | con LST: {m['lst_C'].notna().sum() if 'lst_C' in m else 0}")

if __name__=="__main__":
    main()
