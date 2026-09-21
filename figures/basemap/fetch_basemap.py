"""
fetch_basemap.py -- capas cartograficas de las figuras de mapa.

Descarga y recorta a la region lo que la cartografia necesita y el analisis
no: fronteras y departamentos (Natural Earth 1:10m), la cuenca endorreica y
sus rios (HydroSHEDS) y un modelo de elevacion (SRTM), estos dos via Earth
Engine. Las salidas se versionan en data/basemap/, asi que las figuras se
regeneran sin red ni credenciales.

    python figures/basemap/fetch_basemap.py            # todo
    python figures/basemap/fetch_basemap.py --no-dem   # sin Earth Engine
"""
import sys
import urllib.request
import zipfile
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "basemap"
CACHE = Path("/tmp/naturalearth")
NE = "https://naciscdn.org/naturalearth"

REGION = (-71.6, -17.9, -67.4, -14.1)
SOUTH_AMERICA = (-95.0, -60.0, -30.0, 15.0)
DEM_SCALE_M = 300


def natural_earth(res, kind, name):
    zp = CACHE / f"{name}.zip"
    if not zp.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{NE}/{res}/{kind}/{name}.zip", zp)
    d = CACHE / name
    if not d.exists():
        zipfile.ZipFile(zp).extractall(d)
    return gpd.read_file(next(d.glob("*.shp")))


def clip(gdf, bbox):
    return gpd.clip(gdf, box(*bbox)).reset_index(drop=True)


def vectors():
    OUT.mkdir(parents=True, exist_ok=True)
    countries = natural_earth("10m", "cultural", "ne_10m_admin_0_countries")
    keep = ["ADM0_A3", "NAME", "CONTINENT", "geometry"]
    sa = countries[countries.CONTINENT == "South America"][keep]
    sa.to_file(OUT / "south_america.gpkg", driver="GPKG")
    clip(countries[keep], REGION).to_file(OUT / "countries_region.gpkg", driver="GPKG")

    adm1 = natural_earth("10m", "cultural", "ne_10m_admin_1_states_provinces")
    adm1 = adm1[adm1.adm0_a3.isin(["PER", "BOL", "CHL"])][["adm0_a3", "name", "geometry"]]
    clip(adm1, REGION).to_file(OUT / "departments_region.gpkg", driver="GPKG")

    print(f"  vectores -> {OUT}")


def hydrology():
    """Cuenca endorreica y sus rios (HydroSHEDS). Natural Earth no trae ningun
    afluente del Titicaca a 1:10m."""
    import ee
    ee.Initialize(project="black-display-445217-p3")
    region = ee.Geometry.Rectangle(list(REGION))
    basin = (ee.FeatureCollection("WWF/HydroSHEDS/v1/Basins/hybas_5")
             .filterBounds(ee.Geometry.Point([-69.4, -15.8])).first().geometry())
    rivers = (ee.FeatureCollection("WWF/HydroSHEDS/v1/FreeFlowingRivers")
              .filterBounds(basin.intersection(region, 1))
              .filter(ee.Filter.gte("UPLAND_SKM", 1500))
              .select(["UPLAND_SKM", "DIS_AV_CMS", "RIV_ORD"]))
    outline = gpd.GeoDataFrame.from_features(
        [ee.Feature(basin.intersection(region, 1)).simplify(500).getInfo()], crs=4326)
    outline.to_file(OUT / "basin_region.gpkg", driver="GPKG")

    # HydroSHEDS enruta el flujo a traves del lago y filterBounds admite rios que
    # solo tocan el borde: se recorta a la cuenca y se quita lo que cae en el agua
    lake = gpd.read_file(ROOT / "data" / "lake_boundary" / "titicaca.gpkg").to_crs(4326)
    riv = gpd.GeoDataFrame.from_features(rivers.getInfo()["features"], crs=4326)
    riv = gpd.clip(riv, outline).overlay(lake[["geometry"]], how="difference")
    riv[~riv.is_empty].to_file(OUT / "rivers_region.gpkg", driver="GPKG")
    print(f"  cuenca y rios HydroSHEDS -> {OUT}")


def dem():
    import ee
    ee.Initialize(project="black-display-445217-p3")
    region = ee.Geometry.Rectangle(list(REGION))
    img = ee.Image("USGS/SRTMGL1_003").select("elevation").clip(region).toInt16()
    url = img.getDownloadURL({"region": region, "scale": DEM_SCALE_M,
                              "crs": "EPSG:4326", "format": "GEO_TIFF"})
    out = OUT / "srtm_region.tif"
    urllib.request.urlretrieve(url, out)
    print(f"  DEM SRTM {DEM_SCALE_M} m -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    vectors()
    if "--no-dem" not in sys.argv:
        hydrology()
        dem()
