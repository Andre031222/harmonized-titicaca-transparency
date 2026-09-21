"""Mide sobre el PNG de fig02 los recuadros rojos y los marcos de panel, y
escribe fig02_zoom.csv con las lineas de zoom en coordenadas de hoja (0-1).

    FIG02_ZOOM=FALSE Rscript figures/R/fig02_study_area.R
    python figures/R/fig02_measure_zoom.py
    Rscript figures/R/fig02_study_area.R
"""
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
PNG = ROOT / "results" / "figures_r" / "fig02_study_area.png"
OUT = Path(__file__).with_name("fig02_zoom.csv")

img = np.asarray(Image.open(PNG).convert("RGB")).astype(int)
H, W, _ = img.shape
r, g, b = img[..., 0], img[..., 1], img[..., 2]


def boxes(mask, min_side):
    lab, n = ndimage.label(mask)
    out = []
    for sl in ndimage.find_objects(lab):
        y, x = sl
        if x.stop - x.start >= min_side and y.stop - y.start >= min_side:
            out.append((x.start, y.start, x.stop, y.stop))
    return sorted(out, key=lambda t: (t[1], t[0]))


def frame(x_lo, x_hi, y_lo, y_hi):
    """Marco de un panel: filas y columnas casi enteramente oscuras dentro de
    una franja que se sabe interior al panel."""
    dark = (r < 120) & (g < 120) & (b < 120)
    xs, xe = x_lo + (x_hi - x_lo) // 5, x_hi - (x_hi - x_lo) // 5
    ys, ye = y_lo + (y_hi - y_lo) // 5, y_hi - (y_hi - y_lo) // 5
    rows = [y for y in range(y_lo, y_hi) if dark[y, xs:xe].mean() > 0.9]
    # el recuadro (d) y la leyenda tapan parte de los bordes laterales
    cols = [x for x in range(x_lo, x_hi) if dark[ys:ye, x].mean() > 0.45]
    return min(cols), min(rows), max(cols), max(rows)


red = (abs(r - 194) < 30) & (g < 70) & (abs(b - 43) < 40)
found = boxes(red, 8)
for bx in found:
    print("rojo", bx)

fx = lambda px: px / W
fy = lambda py: 1 - py / H

# a) recuadro del globo, b) recuadro del lago en (b), c) recuadro de la bahia
# en (c), d) marco rojo del recuadro ampliado (el mas grande dentro de c)
globe_box = min(found, key=lambda t: (t[2] - t[0]) * (t[3] - t[1]))
reg_box = [t for t in found if t[0] < W * 0.42 and t[1] > H * 0.25 and t != globe_box]
reg_box = max(reg_box, key=lambda t: (t[2] - t[0]) * (t[3] - t[1]))
right = [t for t in found if t[0] > W * 0.43]
d_frame = max(right, key=lambda t: (t[2] - t[0]) * (t[3] - t[1]))
bay_box = min(right, key=lambda t: (t[2] - t[0]) * (t[3] - t[1]))

b_frame = frame(0, int(W * 0.42), int(H * 0.24), int(H * 0.62))
c_frame = frame(int(W * 0.43), W, 0, int(H * 0.43))
print("marco b", b_frame, "| marco c", c_frame)

lines = [
    # globo -> (b): esquinas inferiores del recuadro a las superiores del panel
    (globe_box[0], globe_box[3], b_frame[0], b_frame[1]),
    (globe_box[2], globe_box[3], b_frame[2], b_frame[1]),
    # recuadro del lago en (b) -> (c): esquinas derechas a las izquierdas de c
    (reg_box[2], reg_box[1], c_frame[0], c_frame[1]),
    (reg_box[2], reg_box[3], c_frame[0], c_frame[3]),
    # recuadro de la bahia -> (d)
    (bay_box[0], bay_box[3], d_frame[0], d_frame[1]),
    (bay_box[2], bay_box[3], d_frame[2], d_frame[1]),
]
with open(OUT, "w") as f:
    f.write("x0,y0,x1,y1\n")
    for x0, y0, x1, y1 in lines:
        f.write(f"{fx(x0):.4f},{fy(y0):.4f},{fx(x1):.4f},{fy(y1):.4f}\n")
print(f"-> {OUT}")
