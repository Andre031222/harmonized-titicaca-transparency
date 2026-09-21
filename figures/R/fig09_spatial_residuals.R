# ============================================================================
# fig09_spatial_residuals.R -- Autocorrelacion espacial del residuo (p12)
#
# (a) Clusters LISA del residuo medio por estacion sobre el lago
# (b) Ampliacion de la Bahia de Puno
# (c) Diagrama de dispersion de Moran: la pendiente es la I global
# (d) Correlograma: I de Moran por anillo de distancia con la envolvente del 95%
#     de las permutaciones
# ============================================================================

source(file.path(local({a <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(a)) dirname(normalizePath(sub("^--file=", "", a[1]))) else getwd()}),
  "theme_titicaca.R"))
source(file.path(ROOT, "figures", "R", "map_base.R"))
suppressPackageStartupMessages({library(sf); library(ggspatial); library(jsonlite)})
sf_use_s2(FALSE)

st   <- read_tidy("lisa_stations.csv")
ring <- read_tidy("moran_correlogram.csv")
sa   <- fromJSON(file.path(ROOT, "results/metrics/spatial_autocorrelation.json"))
lake <- st_read(file.path(ROOT, "data/lake_boundary/titicaca.gpkg"), quiet = TRUE) |>
  st_transform(32719)

CLUSTER <- c(HH = "High–high", LL = "Low–low", HL = "High–low",
             LH = "Low–high", ns = "Not significant")
PAL_LISA <- c(HH = "#B2182B", LL = "#2166AC", HL = "#F4A582", LH = "#92C5DE",
              ns = "#D0D0D0")
counts <- table(factor(st$cluster, levels = names(CLUSTER)))
lab_cl <- sprintf("%s (%d)", CLUSTER, as.integer(counts))
names(lab_cl) <- names(CLUSTER)

st_sf <- st |>
  mutate(cluster = factor(cluster, levels = names(CLUSTER))) |>
  arrange(cluster != "ns") |>
  st_as_sf(coords = c("lon", "lat"), crs = 4326) |> st_transform(32719)

# recuadro de la bahia: las estaciones de la zona con un margen de 2.5 km
bay <- st_sf |> filter(zone_label == "Bahia de Puno") |> st_bbox()
stopifnot(all(is.finite(bay)))
bay <- bay + c(-2500, -2500, 2500, 2500)
# recuadro cuadrado: se ensancha el lado corto alrededor del centro
half <- max(bay[["xmax"]] - bay[["xmin"]], bay[["ymax"]] - bay[["ymin"]]) / 2
cx <- (bay[["xmin"]] + bay[["xmax"]]) / 2; cy <- (bay[["ymin"]] + bay[["ymax"]]) / 2
bay[c("xmin", "ymin", "xmax", "ymax")] <- c(cx - half, cy - half, cx + half, cy + half)
bay_poly <- st_as_sfc(bay)

lisa_layer <- function(size) {
  list(geom_sf(data = st_sf, aes(fill = cluster), shape = 21, colour = "white",
               stroke = 0.25, size = size),
       scale_fill_manual(values = PAL_LISA, labels = lab_cl, drop = TRUE,
                         name = NULL))
}

# El recuadro y el marco de (b) se dibujan primero en un color marcador para
# ubicarlos en el PNG y trazar la flecha curva entre ellos (segunda pasada).
MARK <- "#FF00FE"
panel_a <- function(box_col) {
  ggplot() +
    relief_layers() +
    lisa_layer(1.9) +
    geom_sf(data = bay_poly, fill = NA, colour = box_col, linewidth = 0.7) +
    north("tl") + scale_bar("bl", 0.3) +
    lake_coord() +
    scale_x_continuous(breaks = c(-70, -69.5, -69, -68.5)) +
    scale_y_continuous(breaks = c(-16.5, -16, -15.5)) +
    map_theme + legend_box(c(0.015, 0.075), c(0, 0)) +
    theme(legend.key.size = unit(8, "pt"), legend.text = element_text(size = 6.5),
          plot.margin = margin(4, 2, 2, 2))
}
panel_b <- function(box_col) {
  ggplot() +
    relief_layers() +
    lisa_layer(3) +
    scale_bar("bl", 0.35) +
    coord_sf(crs = UTM, datum = 4326, xlim = bay[c("xmin", "xmax")],
             ylim = bay[c("ymin", "ymax")], expand = FALSE) +
    scale_x_continuous(breaks = c(-70, -69.9, -69.8)) +
    scale_y_continuous(breaks = c(-15.9, -15.8, -15.7)) +
    map_theme +
    theme(legend.position = "none",
          panel.border = element_rect(fill = NA, colour = box_col, linewidth = 1.1),
          plot.margin = margin(4, 2, 2, 10))
}

# ------------------------------------------------------------------ (c) --
# con pesos estandarizados por fila la pendiente de Wz sobre z es la I global
pc <- ggplot(st |> mutate(cluster = factor(cluster, levels = names(CLUSTER))) |>
               arrange(cluster != "ns"), aes(z, lag)) +
  geom_hline(yintercept = 0, colour = INK_2, linewidth = 0.3) +
  geom_vline(xintercept = 0, colour = INK_2, linewidth = 0.3) +
  geom_point(aes(fill = cluster), shape = 21, colour = "white", stroke = 0.25,
             size = 1.8) +
  geom_smooth(method = "lm", formula = y ~ x, se = TRUE, colour = INK,
              fill = "grey75", linewidth = 0.5) +
  annotate("text", x = -Inf, y = Inf, hjust = -0.08, vjust = 1.3, size = 2.4,
           colour = INK, parse = TRUE,
           label = sprintf('"Moran\'s "*italic(I)*" = %.3f, "*italic(P)*" = %.2f"',
                           sa$moran_I, sa$moran_p)) +
  scale_fill_manual(values = PAL_LISA, guide = "none") +
  labs(x = "Station mean residual (z-score)",
       y = "Spatial lag of neighbours (z-score)")

# ------------------------------------------------------------------ (d) --
e_i <- sa$moran_expected
pd <- ggplot(ring, aes(mid_km, moran_I)) +
  geom_ribbon(aes(ymin = env_lo, ymax = env_hi), fill = "grey85") +
  geom_hline(yintercept = e_i, colour = INK_2, linetype = "22", linewidth = 0.35) +
  geom_line(colour = ACCENT, linewidth = 0.6) +
  geom_point(aes(shape = p < 0.05), colour = ACCENT, fill = "white", size = 2,
             stroke = 0.7) +
  # P de cada anillo en una fila al pie, lejos de la curva
  geom_text(aes(y = -Inf, label = sprintf("P = %.2f", p)), vjust = -0.6,
            size = 2, colour = INK_2) +
  scale_shape_manual(values = c(`TRUE` = 16, `FALSE` = 21), guide = "none") +
  scale_x_log10(breaks = ring$mid_km,
                labels = sprintf("%d–%d", ring$lo_km, ring$hi_km)) +
  scale_y_continuous(expand = expansion(mult = c(0.14, 0.05))) +
  labs(x = "Distance band (km, log scale)", y = expression("Moran's "*italic(I))) +
  coord_cartesian(clip = "off") +
  theme(axis.text.x = element_text(size = 6))

# ---------------------------------------------------------------- montaje --
build <- function(box_col) {
  (panel_a(box_col) | panel_b(box_col)) / (pc | pd) +
    plot_layout(widths = c(1.45, 1), heights = c(1.35, 1)) + tags_abc()
}
W_MM <- W2; H_MM <- 165

# pasada 1: ubicar recuadro (izquierda) y marco de (b) (derecha) en el PNG
tmp <- tempfile(fileext = ".png")
ggsave(tmp, build(MARK), width = W_MM, height = H_MM, units = "mm", dpi = 150,
       device = ragg::agg_png, bg = "white")
img <- png::readPNG(tmp)
mask <- img[, , 1] > 0.9 & img[, , 2] < 0.15 & img[, , 3] > 0.9
H <- nrow(mask); Wd <- ncol(mask)
bbox_of <- function(cols) {
  m <- mask; m[, -cols] <- FALSE
  ij <- which(m, arr.ind = TRUE)
  stopifnot(nrow(ij) > 20)
  c(x0 = min(ij[, 2]) / Wd, x1 = max(ij[, 2]) / Wd,
    y0 = 1 - max(ij[, 1]) / H, y1 = 1 - min(ij[, 1]) / H)
}
box_a <- bbox_of(seq_len(floor(Wd * 0.55)))
frm_b <- bbox_of(seq(floor(Wd * 0.55) + 1, Wd))
stopifnot(box_a[["x1"]] < frm_b[["x0"]])

# pasada 2: figura final con la flecha curva del recuadro al panel ampliado
arrow_df <- data.frame(x = box_a[["x1"]], y = box_a[["y1"]],
                       xend = frm_b[["x0"]] - 0.004,
                       yend = frm_b[["y0"]] + 0.88 * (frm_b[["y1"]] - frm_b[["y0"]]))
fig <- cowplot::ggdraw() + cowplot::draw_plot(build(BOX)) +
  geom_curve(data = arrow_df, aes(x = x, y = y, xend = xend, yend = yend),
             curvature = -0.35, colour = BOX, linewidth = 0.6,
             arrow = arrow(length = unit(5, "pt"), type = "closed"))
save_fig(fig, "fig09_spatial_residuals", W_MM, H_MM)
cat("fig09 lista\n")
