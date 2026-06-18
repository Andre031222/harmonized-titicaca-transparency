"""
==============================================================================
16_extract_nearest.py
Re-extraccion de match-ups usando la IMAGEN LIMPIA MAS CERCANA en el tiempo a
cada medicion (no la mediana de la ventana). Reduce el desajuste temporal,
la mayor fuente de ruido del match-up.

Usa getRegion (una llamada por medicion -> todas las imagenes en el punto) y
selecciona en local la mas cercana sin nube.
Sensor por argumento: S2 (default) o LS.
Salida: data/processed/matchups_<sensor>_nearest.csv
==============================================================================
"""
import ee, pandas as pd, numpy as np, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent; OUT=ROOT/"data/processed"
ee.Initialize(project="black-display-445217-p3")

SENSOR=sys.argv[1] if len(sys.argv)>1 else "S2"
WINDOW=7
S2_BANDS=["B1","B2","B3","B4","B5","B6","B7","B8","B8A","B11","B12"]
ROY={"B2":("SR_B2",0.0183,0.8850),"B3":("SR_B3",0.0123,0.9317),
     "B4":("SR_B4",0.0123,0.9372),"B8":("SR_B5",0.0448,0.8339),
     "B11":("SR_B6",0.0306,0.8639),"B12":("SR_B7",0.0116,0.9165)}

def load_field(year_min):
    j=pd.read_csv(ROOT/"data/external_validation/jairo_titicaca_wq.csv")
    j.columns=[c.replace("\xf1","n").replace("�","n") for c in j.columns]
    M={"ENE":1,"FEB":2,"MAR":3,"ABR":4,"MAY":5,"JUN":6,"JUL":7,"AGO":8,
       "SET":9,"SEP":9,"OCT":10,"NOV":11,"DIC":12}
    j["mon"]=j["Mes"].astype(str).str.upper().str[:3].map(M)
    j["date"]=pd.to_datetime(dict(year=j["Ano"],month=j["mon"],day=j["Dia"]),errors="coerce")
    j=j.rename(columns={"Latitud":"lat","Longitud":"lon","Clorofila-A":"chl",
                        "Transparencia_m":"secchi","SST":"tss","Temperatura":"temp_insitu",
                        "Zona":"zona","Estacion":"station"})
    j=j.dropna(subset=["date","lat","lon"])
    return j[j["date"].dt.year>=year_min].reset_index(drop=True)

def nearest_s2(lat,lon,d):
    pt=ee.Geometry.Point([lon,lat]); d0=ee.Date(d.strftime("%Y-%m-%d"))
    col=(ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(pt)
         .filterDate(d0.advance(-WINDOW,"day"),d0.advance(WINDOW,"day")))
    try: arr=col.select(S2_BANDS+["SCL","MSK_CLDPRB"]).getRegion(pt,20).getInfo()
    except Exception: return None
    if len(arr)<2: return None
    df=pd.DataFrame(arr[1:],columns=arr[0])
    tgt=d.timestamp()/86400
    df["tdiff"]=(df["time"]/86400000-tgt).abs()
    df=df[(~df["SCL"].isin([1,3,8,9,10,11]))&(df["MSK_CLDPRB"]<30)].dropna(subset=["B4"])
    if df.empty: return None
    b=df.loc[df["tdiff"].idxmin()]
    out={k:round(float(b[k])/10000,5) for k in S2_BANDS}
    out["tdiff"]=round(float(b["tdiff"]),1); out["sensor"]="S2"
    return out

def nearest_ls(lat,lon,d):
    pt=ee.Geometry.Point([lon,lat]); d0=ee.Date(d.strftime("%Y-%m-%d"))
    col=(ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
         .filterBounds(pt).filterDate(d0.advance(-WINDOW,"day"),d0.advance(WINDOW,"day")))
    srb=[v[0] for v in ROY.values()]
    try: arr=col.select(srb+["ST_B10","QA_PIXEL"]).getRegion(pt,30).getInfo()
    except Exception: return None
    if len(arr)<2: return None
    df=pd.DataFrame(arr[1:],columns=arr[0])
    tgt=d.timestamp()/86400; df["tdiff"]=(df["time"]/86400000-tgt).abs()
    qa=df["QA_PIXEL"].astype("Int64")
    clear=((qa.values>>1&1)==0)&((qa.values>>3&1)==0)&((qa.values>>4&1)==0)
    df=df[clear].dropna(subset=["SR_B4"])
    if df.empty: return None
    b=df.loc[df["tdiff"].idxmin()]
    out={"sensor":"LS","tdiff":round(float(b["tdiff"]),1)}
    for s2b,(srb_,a,bb) in ROY.items():
        out[s2b]=round(a+bb*(float(b[srb_])*0.0000275-0.2),5)
    out["lst_C"]=round(float(b["ST_B10"])*0.00341802+149.0-273.15,2)
    return out

def main():
    ymin=2016 if SENSOR=="S2" else 2013
    j=load_field(ymin); fn=nearest_s2 if SENSOR=="S2" else nearest_ls
    print(f"[{SENSOR}-nearest] {len(j)} mediciones, ventana +-{WINDOW}d")
    out_path=OUT/f"matchups_{SENSOR}_nearest.csv"
    COLS=["row_id","station","zona","date","lat","lon","secchi","chl","tss",
          "temp_insitu","sensor","matched","tdiff"]+S2_BANDS+["lst_C"]
    rows=[]; ok=0; t0=time.time()
    for i,r in j.iterrows():
        try: v=fn(r["lat"],r["lon"],r["date"])
        except Exception: v=None
        rec={"row_id":i,"station":r["station"],"zona":r["zona"],"date":r["date"].strftime("%Y-%m-%d"),
             "lat":r["lat"],"lon":r["lon"],"secchi":r["secchi"],"chl":r["chl"],"tss":r["tss"],
             "temp_insitu":r["temp_insitu"],"sensor":SENSOR,"matched":0}
        if v: ok+=1; rec["matched"]=1; rec.update(v)
        rows.append(rec)
        if len(rows)%25==0:
            pd.DataFrame(rows).reindex(columns=COLS).to_csv(out_path,mode="a",header=not out_path.exists(),index=False)
            print(f"  {i+1}/{len(j)} | matched={ok} | {time.time()-t0:.0f}s"); rows=[]
    if rows: pd.DataFrame(rows).reindex(columns=COLS).to_csv(out_path,mode="a",header=not out_path.exists(),index=False)
    f=pd.read_csv(out_path); m=f[f.matched==1]
    print(f"\n=== {SENSOR}: {len(m)} matchups | tdiff mediana={m.tdiff.median():.1f}d | con Secchi={m.secchi.notna().sum()} ===")

if __name__=="__main__":
    main()
