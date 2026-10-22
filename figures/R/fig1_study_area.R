# ============================================================================
# fig1_study_area.R -- Area de estudio y red de muestreo (diseno HLS)
#
# (a) Sudamerica en proyeccion ortografica, con Peru y Bolivia
# (b) Altiplano norte: relieve SRTM, cuenca endorreica, rios y frontera
# (c) El lago con las 212 LOCALIDADES de los match-ups (coordenadas agrupadas
#     a 500 m), coloreadas por Secchi medio medido y con tamano segun numero
#     de match-ups; contornos tenues de los 8 bloques espaciales (B1..B8)
# (d) Ampliacion de Bahia de Puno
# (e) Esfuerzo de muestreo: localidades segun numero de campanas
# (f) Por que no se usan los codigos de estacion de IMARPE: dispersion maxima
#     de las coordenadas que comparten un mismo codigo
#
# Entradas: results/tidy/location_units.csv, results/tidy/station_id_spread.csv
# Capas base en data/basemap/ (figures/basemap/fetch_basemap.py).
# Disenyo heredado de fig02_study_area.R (rama main).
#
# Lineas de zoom: dependen de la posicion de los recuadros en la hoja. Se
# midieron sobre el PNG renderizado sin ellas (FIG1_ZOOM=FALSE) y se fijan en
# ZOOM mas abajo; son constantes de maquetacion, no resultados.
# ============================================================================

source(file.path(local({a <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(a)) dirname(normalizePath(sub("^--file=", "", a[1]))) else getwd()}),
  "theme_titicaca.R"))

suppressPackageStartupMessages({
  library(sf); library(terra); library(tidyterra)
  library(ggspatial); library(ggnewscale); library(cowplot)
})
sf_use_s2(FALSE)

UTM   <- 32719
BM    <- file.path(ROOT, "data", "basemap")
WATER <- "#CFE3EE"
LAND  <- "#F4F2EE"
SHORE <- "#2B7BB9"
BOX   <- "#C2182B"

rd  <- function(f) st_read(file.path(BM, f), quiet = TRUE)
utm <- function(x) st_transform(x, UTM)
ext <- function(x0, x1, y0, y1) {
  st_as_sfc(st_bbox(c(xmin = x0, xmax = x1, ymin = y0, ymax = y1), crs = 4326))
}
lims <- function(e) {
  b <- st_bbox(utm(e))
  list(x = unname(b[c(1, 3)]), y = unname(b[c(2, 4)]))
}

E_REG  <- ext(-71.25, -67.70, -17.45, -14.35)
E_LAKE <- ext(-70.14, -68.40, -16.70, -15.18)
E_BAY  <- ext(-70.10, -69.72, -15.93, -15.56)

sa     <- rd("south_america.gpkg")
ctry   <- rd("countries_region.gpkg")
dept   <- rd("departments_region.gpkg")
basin  <- rd("basin_region.gpkg")
rivers <- rd("rivers_region.gpkg")
lake   <- st_transform(st_read(file.path(ROOT, "data/lake_boundary/titicaca.gpkg"),
                               quiet = TRUE), 4326)

loc <- read_tidy("location_units.csv") |>
  mutate(zone_label = zone_factor(zona))
locs <- st_as_sf(loc, coords = c("lon", "lat"), crs = 4326, remove = FALSE)
spread <- read_tidy("station_id_spread.csv")

# limites de bloque: los bloques salen de k-medias sobre las localidades
# (p02), asi que la teselacion de Voronoi de las localidades, disuelta por
# bloque, reproduce la particion sin solaparse. Solo se dibujan los limites
# interiores, recortados al lago.
lake_utm <- utm(lake) |> st_union() |> st_make_valid()
loc_utm  <- utm(locs)
vor <- st_voronoi(st_union(loc_utm), st_as_sfc(st_bbox(st_buffer(lake_utm, 20000)))) |>
  st_collection_extract("POLYGON") |> st_as_sf() |> st_set_geometry("geometry") |>
  st_join(loc_utm["spatial_block"]) |>
  group_by(spatial_block) |> summarise(.groups = "drop")
block_lines <- st_boundary(vor) |> st_union() |>
  st_intersection(st_buffer(lake_utm, -400)) |> st_as_sf()
block_lab <- loc_utm |>
  group_by(spatial_block) |>
  summarise(geometry = st_union(geometry), .groups = "drop") |>
  st_centroid()

cities <- st_as_sf(data.frame(name = c("Puno", "Juliaca", "La Paz"),
                              lon = c(-70.02, -70.13, -68.15),
                              lat = c(-15.84, -15.50, -16.50)),
                   coords = c("lon", "lat"), crs = 4326)

map_theme <- theme(
  panel.grid.major = element_line(colour = "#C9C9C9", linewidth = 0.2,
                                  linetype = "22"),
  panel.background = element_rect(fill = NA, colour = NA),
  panel.border = element_rect(fill = NA, colour = INK, linewidth = 0.5),
  axis.text = element_text(size = rel(0.82), colour = INK_2),
  axis.title = element_blank(), plot.margin = margin(1, 1, 1, 1),
  legend.title = element_text(size = rel(0.85)),
  legend.text = element_text(size = rel(0.8)))

north <- function(loc = "tr", w = 0.9, h = 1.15) {
  annotation_north_arrow(location = loc, which_north = "true",
                         height = unit(h, "cm"), width = unit(w, "cm"),
                         pad_x = unit(0.12, "cm"), pad_y = unit(0.12, "cm"),
                         style = north_arrow_fancy_orienteering(
                           text_size = 6, line_width = 0.6))
}
scale_bar <- function(loc = "bl", w = 0.3, px = 0.15) {
  annotation_scale(location = loc, width_hint = w, text_cex = 0.55,
                   height = unit(0.12, "cm"), bar_cols = c(INK, "white"),
                   line_width = 0.4, pad_x = unit(px, "cm"),
                   pad_y = unit(0.15, "cm"))
}
legend_box <- function(pos, just) {
  theme(legend.position = "inside", legend.position.inside = pos,
        legend.justification = just, legend.box = "vertical",
        legend.spacing.y = unit(2, "pt"),
        legend.background = element_rect(fill = alpha("white", 0.85), colour = NA))
}

# ------------------------------------------------------------------ (a) --
ORTHO <- "+proj=ortho +lat_0=-15 +lon_0=-62 +x_0=0 +y_0=0 +a=6371000 +b=6371000"
globe <- st_sfc(st_buffer(st_point(c(0, 0)), 6371000), crs = ORTHO)
grat  <- st_graticule(lat = seq(-60, 30, 30), lon = seq(-120, 0, 30)) |>
  st_transform(ORTHO) |> st_intersection(globe)

pa <- ggplot() +
  geom_sf(data = globe, fill = WATER, colour = INK_2, linewidth = 0.3) +
  geom_sf(data = grat, colour = "white", linewidth = 0.2) +
  geom_sf(data = st_transform(sa, ORTHO), fill = LAND, colour = "#9A9A9A",
          linewidth = 0.12) +
  geom_sf(data = st_transform(sa[sa$ADM0_A3 %in% c("PER", "BOL"), ], ORTHO),
          aes(fill = ADM0_A3), colour = INK_2, linewidth = 0.2,
          show.legend = FALSE) +
  scale_fill_manual(values = c(PER = "#E7C8B5", BOL = "#D7D1E6")) +
  geom_sf(data = st_transform(E_REG, ORTHO), fill = NA, colour = BOX,
          linewidth = 0.6) +
  coord_sf(crs = ORTHO, expand = FALSE) +
  theme_void() + theme(plot.margin = margin(0, 0, 0, 0))

# ------------------------------------------------------------------ (b) --
dem <- rast(file.path(BM, "srtm_region.tif")) |> aggregate(2) |>
  project(paste0("EPSG:", UTM))
hill <- shade(terrain(dem, "slope", unit = "radians"),
              terrain(dem, "aspect", unit = "radians"), 40, 315)
names(dem) <- "elev"; names(hill) <- "hill"
L_REG <- lims(E_REG)

pb <- ggplot() +
  geom_spatraster(data = hill, show.legend = FALSE, maxcell = 6e5) +
  scale_fill_gradient(low = "#3C3C3C", high = "#FFFFFF", na.value = NA) +
  new_scale_fill() +
  geom_spatraster(data = dem, alpha = 0.6, maxcell = 6e5) +
  scale_fill_hypso_tint_c(palette = "dem_poster", limits = c(1000, 6000),
                          breaks = c(2000, 3800, 5500), na.value = NA,
                          name = "Elevation\n(m a.s.l.)",
                          guide = guide_colourbar(barwidth = unit(4, "pt"),
                                                  barheight = unit(30, "pt"))) +
  geom_sf(data = utm(basin), fill = NA, colour = "#1F5F8B", linewidth = 0.35,
          linetype = "42") +
  geom_sf(data = utm(rivers), aes(linewidth = UPLAND_SKM), colour = SHORE,
          show.legend = FALSE) +
  scale_linewidth(range = c(0.12, 0.55), transform = "log10") +
  geom_sf(data = utm(lake), fill = WATER, colour = SHORE, linewidth = 0.25) +
  geom_sf(data = utm(dept), fill = NA, colour = "#6E6E6E", linewidth = 0.18,
          linetype = "13") +
  geom_sf(data = utm(ctry), fill = NA, colour = INK, linewidth = 0.45) +
  geom_sf(data = utm(cities), shape = 22, size = 1.4, fill = "white",
          colour = INK, stroke = 0.4) +
  geom_sf_label(data = utm(cities), aes(label = name), size = 2.3, colour = INK,
                fontface = "bold", fill = alpha("white", 0.75), linewidth = 0,
                label.padding = unit(0.8, "pt"),
                nudge_x = c(-30000, -27000, 2000),
                nudge_y = c(-6000, 5000, 12000)) +
  annotate("text", x = L_REG$x[1] + 0.14 * diff(L_REG$x),
           y = L_REG$y[1] + 0.33 * diff(L_REG$y), label = "PERU",
           size = 2.5, colour = INK_2, fontface = "bold") +
  annotate("text", x = L_REG$x[1] + 0.82 * diff(L_REG$x),
           y = L_REG$y[1] + 0.15 * diff(L_REG$y), label = "BOLIVIA",
           size = 2.5, colour = INK_2, fontface = "bold") +
  geom_sf(data = utm(E_LAKE), fill = NA, colour = BOX, linewidth = 0.6) +
  north("tr", 0.7, 0.9) + scale_bar("bl", 0.28) +
  coord_sf(crs = UTM, datum = 4326, xlim = L_REG$x, ylim = L_REG$y,
           expand = FALSE) +
  scale_x_continuous(breaks = c(-71, -70, -69, -68)) +
  scale_y_continuous(breaks = c(-17, -16, -15)) +
  map_theme + legend_box(c(0.015, 0.985), c(0, 1))

# ------------------------------------------------------------------ (c) --
L_LAKE <- lims(E_LAKE)
zone_lab <- st_as_sf(data.frame(
  txt = c("Lago Mayor", "Lago Menor\n(Wiñaymarca)"),
  lon = c(-69.36, -68.66), lat = c(-15.62, -16.62)),
  coords = c("lon", "lat"), crs = 4326)

sec_lim <- c(floor(min(loc$secchi_mean)), ceiling(max(loc$secchi_mean)))
n_brk   <- unique(c(1, round(median(loc$n[loc$n > 1])), max(loc$n)))

loc_layers <- function(size_rng = c(0.9, 3.6), legend = TRUE) {
  list(
    geom_sf(data = utm(locs) |> arrange(desc(n)),
            aes(fill = secchi_mean, size = n),
            shape = 21, colour = "white", stroke = 0.25, alpha = 0.95,
            show.legend = legend),
    scale_fill_viridis_c(option = "mako", direction = -1,
                         name = "Mean measured\nSecchi (m)", limits = sec_lim,
                         breaks = scales::breaks_pretty(4),
                         guide = guide_colourbar(order = 1, barwidth = unit(4, "pt"),
                                                 barheight = unit(38, "pt"))),
    scale_size_continuous(range = size_rng, name = "Match-ups\nper location",
                          limits = c(1, max(loc$n)), breaks = n_brk,
                          guide = guide_legend(order = 2, override.aes = list(
                            fill = "#5B7F95", colour = "white"))))
}

pc <- ggplot() +
  geom_sf(data = utm(ctry), fill = LAND, colour = NA) +
  geom_sf(data = utm(lake), fill = WATER, colour = SHORE, linewidth = 0.3) +
  geom_sf(data = utm(ctry), fill = NA, colour = INK_2, linewidth = 0.35,
          linetype = "42") +
  geom_sf(data = block_lines, colour = "#6A6A6A", linewidth = 0.3,
          linetype = "21") +
  loc_layers() +
  geom_sf_label(data = block_lab, aes(label = spatial_block), size = 2.3,
                colour = INK_2, fontface = "italic", fill = alpha("white", 0.7),
                linewidth = 0, label.padding = unit(0.8, "pt")) +
  geom_sf_label(data = utm(zone_lab), aes(label = txt), size = 2.3,
                fontface = "bold", colour = INK, lineheight = 0.95,
                fill = alpha("white", 0.8), linewidth = 0,
                label.padding = unit(1.4, "pt")) +
  geom_sf(data = utm(E_BAY), fill = NA, colour = BOX, linewidth = 0.6) +
  # la barra se aparta de las lineas de zoom que salen del recuadro de la bahia
  north("tr") + scale_bar("bl", 0.2, px = 3.0) +
  coord_sf(crs = UTM, datum = 4326, xlim = L_LAKE$x, ylim = L_LAKE$y,
           expand = FALSE) +
  scale_x_continuous(breaks = seq(-70, -68.5, 0.5)) +
  scale_y_continuous(breaks = seq(-16.5, -15.25, 0.25)) +
  map_theme + theme(legend.position = "none")

# ------------------------------------------------------------------ (d) --
L_BAY <- lims(E_BAY)
pd <- ggplot() +
  geom_sf(data = utm(ctry), fill = LAND, colour = NA) +
  geom_sf(data = utm(lake), fill = WATER, colour = SHORE, linewidth = 0.35) +
  loc_layers(size_rng = c(1.5, 4.8), legend = FALSE) +
  geom_sf(data = utm(cities[1, ]), shape = 22, size = 1.6, fill = "white",
          colour = INK, stroke = 0.4) +
  geom_sf_label(data = utm(cities[1, ]), label = "Puno", size = 2.3,
                fontface = "bold", colour = INK, fill = alpha("white", 0.8),
                linewidth = 0, label.padding = unit(0.8, "pt"),
                nudge_x = -600, nudge_y = 2900) +
  annotate("label", x = L_BAY$x[1] + 0.04 * diff(L_BAY$x),
           y = L_BAY$y[2] - 0.03 * diff(L_BAY$y),
           label = "Bahía de Puno", hjust = 0, vjust = 1, size = 2.3,
           fontface = "bold", colour = INK, fill = alpha("white", 0.85),
           linewidth = 0, label.padding = unit(1.5, "pt")) +
  scale_bar("br", 0.4) +
  coord_sf(crs = UTM, datum = 4326, xlim = L_BAY$x, ylim = L_BAY$y,
           expand = FALSE) +
  theme_void() +
  theme(panel.border = element_rect(fill = NA, colour = BOX, linewidth = 1.1),
        plot.background = element_rect(fill = "white", colour = NA),
        plot.margin = margin(0, 0, 0, 0))

# ------------------------------------------------------------ leyendas --
leg_st <- get_legend(pc + theme(legend.position = "right", legend.box = "horizontal",
                                legend.title = element_text(size = 7),
                                legend.text = element_text(size = 6.5)))
k <- data.frame(y = 6:1, lab = c("Lake Titicaca", "River (HydroSHEDS)",
                                 "Endorheic basin", "International border",
                                 "Department boundary",
                                 "Spatial block (B1–B8), panel c"))
pk <- ggplot(k) +
  geom_rect(data = k[1, ], aes(xmin = 0, xmax = 0.9, ymin = y - 0.28, ymax = y + 0.28),
            fill = WATER, colour = SHORE, linewidth = 0.3) +
  geom_segment(data = k[2, ], aes(x = 0, xend = 0.9, y = y, yend = y),
               colour = SHORE, linewidth = 0.5) +
  geom_segment(data = k[3, ], aes(x = 0, xend = 0.9, y = y, yend = y),
               colour = "#1F5F8B", linewidth = 0.4, linetype = "42") +
  geom_segment(data = k[4, ], aes(x = 0, xend = 0.9, y = y, yend = y),
               colour = INK, linewidth = 0.5) +
  geom_segment(data = k[5, ], aes(x = 0, xend = 0.9, y = y, yend = y),
               colour = "#6E6E6E", linewidth = 0.3, linetype = "13") +
  geom_segment(data = k[6, ], aes(x = 0, xend = 0.9, y = y, yend = y),
               colour = "#7A7A7A", linewidth = 0.3, linetype = "21") +
  geom_text(aes(x = 1.2, y = y, label = lab), hjust = 0, size = 2.3, colour = INK) +
  coord_cartesian(xlim = c(0, 6.5), ylim = c(0.5, 6.5), expand = FALSE) +
  theme_void()

# ------------------------------------------------------------------ (e) --
eff <- loc |> count(n_campaigns, zone_label)
eff_tot <- loc |> count(n_campaigns)
pe <- ggplot(eff, aes(factor(n_campaigns, levels = seq_len(max(loc$n_campaigns))),
                      n, fill = zone_label)) +
  geom_col(width = 0.72, colour = "white", linewidth = 0.2) +
  geom_text(data = eff_tot, aes(x = factor(n_campaigns), y = n, label = n),
            inherit.aes = FALSE, vjust = -0.4, size = 2.3, colour = INK_2) +
  scale_fill_manual(values = PAL_ZONE, name = NULL) +
  scale_x_discrete(drop = FALSE) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.1))) +
  labs(x = "Campaigns in which the location was sampled",
       y = "Locations") +
  theme(legend.position = "inside", legend.position.inside = c(0.98, 0.98),
        legend.justification = c(1, 1), legend.key.size = unit(7, "pt"),
        panel.grid.major.y = element_blank(), plot.margin = margin(4, 6, 2, 4))

# ------------------------------------------------------------------ (f) --
# ECDF de la dispersion maxima de las posiciones que comparten un codigo.
n_code <- nrow(spread)
pf <- ggplot(spread, aes(max_spread_km)) +
  geom_vline(xintercept = 0.5, colour = NEUTRAL, linewidth = 0.3,
             linetype = "22") +
  annotate("text", x = 0.5, y = 1.0, label = " 500 m location threshold",
           hjust = 0, vjust = 1, size = 2.3, colour = INK_2) +
  annotate("text", x = 1.2, y = 0.12, hjust = 0, vjust = 0, size = 2.3,
           colour = INK, lineheight = 1.05,
           label = sprintf("%.0f%% of codes span > 1 km\n%.0f%% span > 10 km",
                           100 * mean(spread$max_spread_km > 1),
                           100 * mean(spread$max_spread_km > 10))) +
  stat_ecdf(geom = "step", colour = ACCENT, linewidth = 0.6, pad = FALSE) +
  scale_x_continuous(trans = scales::pseudo_log_trans(sigma = 0.5, base = 10),
                     breaks = c(0, 1, 5, 20, 50),
                     limits = c(0, max(spread$max_spread_km) * 1.05)) +
  scale_y_continuous(labels = scales::percent, limits = c(0, 1.02),
                     expand = expansion(mult = c(0, 0))) +
  labs(x = "Maximum spread of one station code (km)",
       y = sprintf("Station codes (n = %d)", n_code)) +
  theme(plot.margin = margin(4, 8, 2, 4))

# ------------------------------------------------------------- montaje --
tag <- function(l, x, y) draw_label(l, x = x, y = y, hjust = 0, vjust = 1,
                                    fontface = "bold", size = 10, colour = INK)
zoom <- function(x, y) draw_line(x = x, y = y, colour = BOX, linewidth = 0.35,
                                 linetype = "22")

fig <- ggdraw() +
  draw_plot(pa, x = 0.050, y = 0.670, width = 0.30, height = 0.325) +
  draw_plot(pb, x = 0.000, y = 0.255, width = 0.415, height = 0.405) +
  draw_plot(pc, x = 0.425, y = 0.505, width = 0.575, height = 0.490) +
  draw_plot(pd, x = 0.450, y = 0.255, width = 0.150, height = 0.235) +
  draw_plot(leg_st, x = 0.625, y = 0.385, width = 0.36, height = 0.115) +
  draw_plot(pk, x = 0.625, y = 0.255, width = 0.37, height = 0.120) +
  draw_plot(pe, x = 0.020, y = 0.000, width = 0.560, height = 0.235) +
  draw_plot(pf, x = 0.600, y = 0.000, width = 0.400, height = 0.235) +
  tag("a", 0.005, 0.995) + tag("b", 0.005, 0.660) + tag("c", 0.430, 0.995) +
  tag("d", 0.425, 0.447) + tag("e", 0.005, 0.240) + tag("f", 0.590, 0.240)

# medido con figures/R/measure_zoom.py sobre el render FIG1_ZOOM=FALSE
# (globo->b, recuadro del lago->c, recuadro de la bahia->d)
ZOOM <- read.csv(text = "x0,y0,x1,y1
0.1756,0.8247,0.0365,0.6270
0.1867,0.8247,0.4127,0.6270
0.3400,0.5432,0.4795,0.9809
0.3400,0.3836,0.4795,0.5377
0.4915,0.7608,0.4501,0.4441
0.6070,0.7608,0.5998,0.4441", header = TRUE)
if (as.logical(Sys.getenv("FIG1_ZOOM", "TRUE"))) {
  for (i in seq_len(nrow(ZOOM))) {
    fig <- fig + zoom(c(ZOOM$x0[i], ZOOM$x1[i]), c(ZOOM$y0[i], ZOOM$y1[i]))
  }
}

save_fig(fig, "fig1_study_area", W2, 200)
cat("fig1 lista\n")
