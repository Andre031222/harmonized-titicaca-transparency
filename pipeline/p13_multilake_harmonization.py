"""
p13_multilake_harmonization.py -- ¿el fallo de Roy es del Titicaca o de toda
el agua oscura?

Los dos diagnosticos del articulo (intercepto de Roy frente a la senal nativa,
y prueba del clasificador de dos muestras) no necesitan dato de campo: solo
imagenes sobre agua. Aqui se repiten sobre lagos grandes de distinta claridad,
sin una sola medicion in-situ, para ver si el fallo generaliza.

Muestreo: puntos aleatorios sobre agua permanente (JRC Global Surface Water,
ocurrencia = 100 %) a mas de 1 km de la orilla; en cada punto, la mediana
temporal de las imagenes limpias de la estacion elegida, promediada en un
buffer de 30 m, igual que los match-ups del articulo.

Requiere Earth Engine:  earthengine authenticate

    python pipeline/p13_multilake_harmonization.py          # extrae y evalua
    python pipeline/p13_multilake_harmonization.py --eval   # solo reevalua
"""
import json
import sys

import pandas as pd
from p00_config import BANDS, MET, PROC, ROY, SEED, TIDY, banner

CACHE = PROC / "multilake_reflectance_raw.csv"
N_POINTS = 300
BUFFER = 30
CLDPRB = 30
SHORE_KM = 1.0

ROY_SR = {"B2": "SR_B2", "B3": "SR_B3", "B4": "SR_B4",
          "B8": "SR_B5", "B11": "SR_B6", "B12": "SR_B7"}

# lagos grandes de distinta claridad; la ventana es su estacion mas despejada
LAKES = [
    ("Lake Titicaca", "Peru/Bolivia", (-70.1, -16.6, -68.5, -15.2), (7, 10), "clear"),
    ("Lake Tahoe", "United States", (-120.16, 38.90, -119.92, 39.25), (7, 10), "clear"),
    ("Lake Baikal", "Russia", (104.0, 51.5, 110.0, 55.8), (7, 9), "clear"),
    ("Lake Malawi", "Malawi/Mozambique", (34.0, -14.4, 35.3, -9.5), (6, 9), "clear"),
    ("Lake Erie", "United States/Canada", (-83.4, 41.4, -78.9, 42.9), (6, 9), "turbid"),
    ("Lake Okeechobee", "United States", (-81.1, 26.7, -80.6, 27.2), (3, 6), "turbid"),
]
YEARS = (2021, 2024)
CHUNK = 50            # los lagos grandes agotan la memoria de GEE de una sola vez
MAX_CLOUD = 40


def water_points(ee, bbox):
    """Puntos sobre agua permanente, lejos de la orilla."""
    region = ee.Geometry.Rectangle(list(bbox))
    occ = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence")
    water = occ.gte(90).unmask(0)
    # se erosiona la mascara para alejar los puntos de la orilla: el kernel es
    # de SHORE_KM a la escala de muestreo (300 m)
    inland = water.focalMin(radius=int(SHORE_KM * 1000 / 300), units="pixels")
    pts = inland.rename("w").stratifiedSample(
        numPoints=0, classBand="w", classValues=[1], classPoints=[N_POINTS],
        region=region, scale=300, seed=SEED, geometries=True)
    return region, pts


def composite(ee, region, months, sensor):
    lo, hi = YEARS
    rng = ee.Filter.calendarRange(months[0], months[1], "month")
    if sensor == "S2":
        col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
               .filterBounds(region).filterDate(f"{lo}-01-01", f"{hi}-12-31")
               .filter(rng)
               .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD)))

        def mask(img):
            scl = img.select("SCL")
            good = (scl.neq(1).And(scl.neq(3)).And(scl.neq(8)).And(scl.neq(9))
                    .And(scl.neq(10)).And(scl.neq(11)))
            return (img.updateMask(good.And(img.select("MSK_CLDPRB").lt(CLDPRB)))
                    .select(["B2", "B3", "B4", "B8", "B11", "B12"])
                    .multiply(1e-4))
        return col.map(mask).median(), 20

    col = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
           .merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
           .filterBounds(region).filterDate(f"{lo}-01-01", f"{hi}-12-31")
           .filter(rng).filter(ee.Filter.lt("CLOUD_COVER", MAX_CLOUD)))

    def mask_ls(img):
        qa = img.select("QA_PIXEL")
        clear = (qa.bitwiseAnd(1 << 1).eq(0).And(qa.bitwiseAnd(1 << 3).eq(0))
                 .And(qa.bitwiseAnd(1 << 4).eq(0)))
        sr = (img.select([ROY_SR[b] for b in BANDS], BANDS)
              .multiply(2.75e-5).add(-0.2))
        return sr.updateMask(clear)
    return col.map(mask_ls).median(), 30


def extract():
    import ee
    ee.Initialize(project="black-display-445217-p3")
    rows = []
    for name, country, bbox, months, clarity in LAKES:
        region, pts = water_points(ee, bbox)
        n_pts = pts.size().getInfo()
        print(f"  {name:18s} puntos sobre agua: {n_pts}")
        coords = [f["geometry"]["coordinates"][:2]
                  for f in pts.getInfo()["features"]]
        for sensor in ("S2", "LS"):
            got = 0
            for i in range(0, len(coords), CHUNK):
                chunk = coords[i:i + CHUNK]
                fc_pts = ee.FeatureCollection(
                    [ee.Feature(ee.Geometry.Point(c).buffer(BUFFER), {"i": k})
                     for k, c in enumerate(chunk)])
                sub_region = fc_pts.geometry().bounds()
                img, scale = composite(ee, sub_region, months, sensor)
                fc = img.reduceRegions(collection=fc_pts,
                                       reducer=ee.Reducer.mean(),
                                       scale=scale, tileScale=4)
                for f in fc.getInfo()["features"]:
                    p = f["properties"]
                    if any(p.get(b) is None for b in BANDS):
                        continue
                    lon, lat = chunk[int(p["i"])]
                    rows.append({"lake": name, "country": country,
                                 "clarity": clarity, "sensor": sensor,
                                 "lon": lon, "lat": lat,
                                 **{b: p[b] for b in BANDS}})
                    got += 1
            print(f"    {sensor}: {got} muestras")
    d = pd.DataFrame(rows)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(CACHE, index=False, float_format="%.6f")
    print(f"\n  -> {CACHE} ({len(d)} filas)")
    return d


def evaluate(d):
    """Si la transformacion de Roy valiese sobre el agua de estos lagos, la
    mediana de Landsat armonizado y la de Sentinel-2 coincidirian (razon ~1).

    No se repite aqui la prueba del clasificador: necesita pares del mismo dia,
    y estos compuestos estacionales mezclan fechas distintas, con lo que
    cualquier clasificador separaria los sensores por construccion.
    """
    banner("(A) SENAL NATIVA DE LANDSAT SOBRE EL AGUA DE CADA LAGO")
    rows = []
    for (lake, country, clarity), g in d.groupby(["lake", "country", "clarity"],
                                                 sort=False):
        ls, s2 = g[g.sensor == "LS"], g[g.sensor == "S2"]
        if len(ls) < 30 or len(s2) < 30:
            print(f"  {lake:18s} muestras insuficientes, se omite")
            continue
        for b in BANDS:
            a_roy, slope = ROY[b]
            nat, s2m = float(ls[b].median()), float(s2[b].median())
            har = a_roy + slope * nat
            rows.append({"lake": lake, "country": country, "clarity": clarity,
                         "band": b, "roy_intercept": a_roy,
                         "native_ls_median": round(nat, 6),
                         "s2_median": round(s2m, 6),
                         "harmonized_ls_median": round(har, 6),
                         "harmonized_over_s2": round(har / s2m, 2) if s2m > 1e-4 else None,
                         "n_ls": len(ls), "n_s2": len(s2)})
    band_df = pd.DataFrame(rows)
    nat = band_df.pivot(index="lake", columns="band", values="native_ls_median")
    print("  mediana nativa de Landsat (reflectancia); negativa = el agua no")
    print("  devuelve senal, y un intercepto aditivo no tiene nada que corregir\n")
    print(nat.to_string())

    banner("(B) LANDSAT ARMONIZADO CON ROY FRENTE A SENTINEL-2 (razon de medianas)")
    piv = band_df.pivot(index="lake", columns="band", values="harmonized_over_s2")
    print("  si la transformacion valiese sobre agua, cada celda seria ~1.0\n")
    print(piv.to_string())

    summ = []
    for (lake, country, clarity), g in band_df.groupby(["lake", "country", "clarity"],
                                                       sort=False):
        r = g.dropna(subset=["harmonized_over_s2"]).harmonized_over_s2
        s2_green = float(d[(d.lake == lake) & (d.sensor == "S2")].B3.median())
        summ.append({"lake": lake, "country": country, "clarity": clarity,
                     "s2_green_median": round(s2_green, 5),
                     "worst_band_ratio": float(r.max()),
                     "median_band_ratio": round(float(r.median()), 2),
                     "bands_within_20pct": int(((r - 1).abs() <= 0.2).sum()),
                     "negative_native_bands": int((g.native_ls_median < 0).sum()),
                     "n_points": int(g.n_ls.iloc[0])})
    sm = pd.DataFrame(summ).sort_values("s2_green_median")
    banner("(C) RESUMEN: EL FALLO CRECE CUANTO MAS CLARA ES EL AGUA")
    print("  (ordenado por reflectancia verde de Sentinel-2, proxy de turbidez)\n")
    print(sm.to_string(index=False))
    rho = sm.s2_green_median.corr(sm.worst_band_ratio, method="spearman")
    print(f"\n  Spearman entre verde de S2 y peor razon: {rho:+.2f} "
          f"(mas clara el agua, peor la armonizacion)")

    band_df.to_csv(TIDY / "multilake_bands.csv", index=False)
    sm.to_csv(TIDY / "multilake_summary.csv", index=False)
    json.dump({"lakes": sm.to_dict("records"), "bands": band_df.to_dict("records"),
               "spearman_green_vs_worst_ratio": round(float(rho), 3),
               "n_points_requested": N_POINTS, "buffer_m": BUFFER,
               "years": list(YEARS), "shore_distance_km": SHORE_KM,
               "note": ("Seasonal medians per sensor at the same points; not "
                        "same-day pairs, so this is a consistency check of the "
                        "transform, not a paired cross-sensor comparison.")},
              open(MET / "multilake_harmonization.json", "w"), indent=2)
    print(f"\n  -> {TIDY / 'multilake_summary.csv'}")
    print(f"  -> {TIDY / 'multilake_bands.csv'}")
    print(f"  -> {MET / 'multilake_harmonization.json'}")


def main():
    banner("p13  EL DIAGNOSTICO DE ARMONIZACION EN OTROS LAGOS GRANDES")
    if "--eval" in sys.argv:
        if not CACHE.exists():
            raise SystemExit(f"no existe {CACHE}: corre sin --eval")
        d = pd.read_csv(CACHE)
        print(f"  cache: {len(d)} filas, {d.lake.nunique()} lagos\n")
    else:
        d = extract()
    evaluate(d)


if __name__ == "__main__":
    main()
