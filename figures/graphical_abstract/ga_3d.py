"""
ga_3d.py -- resumen grafico: el lago en 3D con las estaciones de monitoreo.

Relieve SRTM (data/basemap/srtm_region.tif) con exageracion vertical, lago a
la cota del espejo de agua y estaciones coloreadas por Secchi medio medido.
Formato Elsevier: apaisado, minimo 1328 x 531 px.

    python figures/graphical_abstract/ga_3d.py
"""
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import LightSource, LinearSegmentedColormap, Normalize
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "figures_r" / "graphical_abstract.png"

BOUNDS = (-70.25, -16.75, -68.50, -15.15)   # al este empiezan los Yungas   # lon0, lat0, lon1, lat1
LAKE_LEVEL = 3810.0                          # m s. n. m., espejo del Titicaca
EXAG = 8.0
FLOOR = 1500.0                               # piso para los valles hacia los Yungas
STEP = 2                                     # 300 m -> ~600 m en la malla 3D

# mismo tinte del Altiplano que los mapas en R (figures/R/map_base.R)
ALTI = LinearSegmentedColormap.from_list("alti", [
    (0.00, "#6E9F5C"), (0.105, "#A9C27A"), (0.237, "#E3D99A"),
    (0.395, "#D2AE78"), (0.579, "#A67C55"), (0.763, "#8A6A55"),
    (1.00, "#F2EEE8")])
SECCHI = LinearSegmentedColormap.from_list("secchi", [
    "#7A3E1D", "#C2582C", "#E8B04A", "#8FC1B5", "#2E6E8E", "#16324F"])


def load_dem():
    with rasterio.open(ROOT / "data" / "basemap" / "srtm_region.tif") as src:
        win = from_bounds(*BOUNDS, transform=src.transform)
        z = src.read(1, window=win).astype(float)
        tr = src.window_transform(win)
    lake = gpd.read_file(ROOT / "data" / "lake_boundary" / "titicaca.gpkg").to_crs(4326)
    water = ~geometry_mask(lake.geometry, out_shape=z.shape, transform=tr)
    z = z[::STEP, ::STEP]
    water = water[::STEP, ::STEP]
    ny, nx = z.shape
    lon = tr.c + tr.a * STEP * (np.arange(nx) + 0.5)
    lat = tr.f + tr.e * STEP * (np.arange(ny) + 0.5)
    return z, water, lon, lat


def to_km(lon, lat, lat0):
    return ((lon - BOUNDS[0]) * 111.32 * np.cos(np.radians(lat0)),
            (lat - BOUNDS[1]) * 110.57)


def stations():
    d = pd.read_csv(ROOT / "results" / "tidy" / "insitu_distribution.csv")
    return d.groupby("station").agg(lon=("lon", "mean"), lat=("lat", "mean"),
                                    secchi=("secchi", "mean")).reset_index()


def main():
    z, water, lon, lat = load_dem()
    z = np.where(np.isfinite(z), z, LAKE_LEVEL)
    z = np.where(water, LAKE_LEVEL, np.maximum(z, FLOOR))
    lat0 = (BOUNDS[1] + BOUNDS[3]) / 2
    LON, LAT = np.meshgrid(lon, lat)
    X, Y = to_km(LON, LAT, lat0)
    Z = (z - LAKE_LEVEL) / 1000 * EXAG          # km sobre el espejo de agua

    ls = LightSource(azdeg=315, altdeg=40)
    norm = Normalize(3700, 5600)
    rgb = ls.shade(z, cmap=ALTI, norm=norm, blend_mode="soft", vert_exag=0.05,
                   dx=0.6, dy=0.6)
    rgb[water] = (0.62, 0.80, 0.90, 1.0)

    st = stations()
    sx, sy = to_km(st.lon.values, st.lat.values, lat0)

    fig = plt.figure(figsize=(13.28, 5.31), dpi=150)
    # computed_zorder=False: si no, la superficie tapa las estaciones
    ax = fig.add_axes([-0.10, -0.30, 0.86, 1.6], projection="3d",
                      computed_zorder=False)
    ax.plot_surface(X, Y, Z, facecolors=rgb, rstride=1, cstride=1,
                    linewidth=0, antialiased=False, shade=False)
    sc = ax.scatter(sx, sy, np.full(len(sx), 0.05), c=st.secchi, cmap=SECCHI,
                    vmin=2, vmax=14, s=22, edgecolors="white", linewidths=0.5,
                    depthshade=False, zorder=10)
    ax.set_box_aspect((X.max(), Y.max(), Z.max() - Z.min()))
    ax.view_init(elev=32, azim=-62)
    ax.set_axis_off()

    # leyenda y mensajes, a la derecha
    cax = fig.add_axes([0.70, 0.60, 0.012, 0.28])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("Mean Secchi depth (m)", fontsize=10)
    cb.ax.tick_params(labelsize=9)
    txt = [("Harmonization", "Land-derived Landsat→Sentinel-2\ncoefficients fail over clear water"),
           ("Validation", "Campaign-date blocking: R² = 0.574,\nRMSE 1.86 m (n = 812)"),
           ("Limits", "90% intervals under-cover the\nclearest and most turbid water")]
    y = 0.50
    for head, body in txt:
        fig.text(0.70, y, head, fontsize=11, fontweight="bold", color="#1A1A1A")
        fig.text(0.70, y - 0.085, body, fontsize=9.5, color="#4A4A4A",
                 linespacing=1.25)
        y -= 0.155
    fig.text(0.70, 0.93, "Lake Titicaca, 3810 m a.s.l.", fontsize=12.5,
             fontweight="bold", color="#1A1A1A")
    fig.text(0.02, 0.04, f"SRTM relief, vertical exaggeration ×{EXAG:.0f}",
             fontsize=8, color="#6A6A6A")
    fig.savefig(OUT, dpi=300, facecolor="white")
    print(f"  -> {OUT}")


if __name__ == "__main__":
    main()
