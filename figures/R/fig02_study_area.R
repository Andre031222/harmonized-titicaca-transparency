# ============================================================================
# fig02_study_area.R -- Area de estudio en tres escalas y red de monitoreo
#
# (a) Sudamerica en proyeccion ortografica, con Peru y Bolivia
# (b) El Altiplano: relieve SRTM, fronteras, cuenca endorreica y rios
# (c) El lago con las estaciones, coloreadas por Secchi medio medido
# (d) Ampliacion de Bahia de Puno, dentro de (c)
# (e) Esfuerzo de muestreo por estacion
#
# Capas base en data/basemap/ (figures/basemap/fetch_basemap.py).
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

rd <- function(f) st_read(file.path(BM, f), quiet = TRUE)
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
E_BAY  <- ext(-70.02, -69.72, -15.93, -15.56)

sa     <- rd("south_america.gpkg")
ctry   <- rd("countries_region.gpkg")
dept   <- rd("departments_region.gpkg")
basin  <- rd("basin_region.gpkg")
rivers <- rd("rivers_region.gpkg")
lake   <- st_transform(st_read(file.path(ROOT, "data/lake_boundary/titicaca.gpkg"),
                               quiet = TRUE), 4326)

dist <- read_tidy("insitu_distribution.csv")
st_summary <- dist |>
  group_by(station) |>
  summarise(zona = first(zona), lat = mean(lat), lon = mean(lon),
            secchi_mean = mean(secchi), n_visits = n(),
            n_campaigns = n_distinct(campaign_date), .groups = "drop") |>
  mutate(zone_label = zone_factor(zona))
stations <- st_as_sf(st_summary, coords = c("lon", "lat"), crs = 4326)

cities <- st_as_sf(data.frame(name = c("Puno", "Juliaca", "La Paz"),
                              lon = c(-70.02, -70.13, -68.15),
                              lat = c(-15.84, -15.50, -16.50)),
                   coords = c("lon", "lat"), crs = 4326)

map_theme <- theme(
  panel.grid.major = element_line(colour = "#C9C9C9", linewidth = 0.2,
                                  linetype = "22"),
  panel.background = element_rect(fill = NA, colour = NA),
  panel.border = element_rect(fill = NA, colour = INK, linewidth = 0.5),
  axis.text = element_text(size = rel(0.62), colour = INK_2),
  axis.title = element_blank(), plot.margin = margin(1, 1, 1, 1),
  legend.title = element_text(size = rel(0.64)),
  legend.text = element_text(size = rel(0.58)))

north <- function(loc = "tr", w = 0.9, h = 1.15) {
  annotation_north_arrow(location = loc, which_north = "true",
                         height = unit(h, "cm"), width = unit(w, "cm"),
                         pad_x = unit(0.12, "cm"), pad_y = unit(0.12, "cm"),
                         style = north_arrow_fancy_orienteering(
                           text_size = 6, line_width = 0.6))
}
scale_bar <- function(loc = "bl", w = 0.3) {
  annotation_scale(location = loc, width_hint = w, text_cex = 0.55,
                   height = unit(0.12, "cm"), bar_cols = c(INK, "white"),
                   line_width = 0.4, pad_x = unit(0.15, "cm"),
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
  geom_sf_text(data = utm(cities), aes(label = name), size = 2.1, colour = INK,
               fontface = "bold", nudge_x = c(-15000, -24000, -19000),
               nudge_y = c(-9000, 5000, 9000)) +
  annotate("text", x = L_REG$x[1] + 0.14 * diff(L_REG$x),
           y = L_REG$y[1] + 0.33 * diff(L_REG$y), label = "PERU",
           size = 2.5, colour = INK_2, fontface = "bold") +
  annotate("text", x = L_REG$x[1] + 0.80 * diff(L_REG$x),
           y = L_REG$y[1] + 0.22 * diff(L_REG$y), label = "BOLIVIA",
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
  lon = c(-69.36, -68.66), lat = c(-15.62, -16.58)),
  coords = c("lon", "lat"), crs = 4326)

station_layers <- function(size_rng = c(1.1, 3.8), legend = TRUE) {
  list(
    geom_sf(data = utm(stations), aes(fill = secchi_mean, size = n_visits),
            shape = 21, colour = "white", stroke = 0.3, alpha = 0.95,
            show.legend = legend),
    scale_fill_viridis_c(option = "mako", direction = -1,
                         name = "Mean Secchi\n(m)", limits = c(1.5, 16),
                         breaks = c(4, 8, 12, 16),
                         guide = guide_colourbar(order = 1, barwidth = unit(4, "pt"),
                                                 barheight = unit(38, "pt"))),
    scale_size_continuous(range = size_rng, name = "Match-ups",
                          breaks = c(2, 6, 12),
                          guide = guide_legend(order = 2, override.aes = list(
                            fill = "#5B7F95", colour = "white"))))
}

pc <- ggplot() +
  geom_sf(data = utm(ctry), fill = LAND, colour = NA) +
  geom_sf(data = utm(lake), fill = WATER, colour = SHORE, linewidth = 0.3) +
  geom_sf(data = utm(ctry), fill = NA, colour = INK_2, linewidth = 0.35,
          linetype = "42") +
  station_layers() +
  geom_sf_label(data = utm(zone_lab), aes(label = txt), size = 2.3,
                fontface = "bold", colour = INK, lineheight = 0.95,
                fill = alpha("white", 0.8), linewidth = 0,
                label.padding = unit(1.4, "pt")) +
  geom_sf(data = utm(E_BAY), fill = NA, colour = BOX, linewidth = 0.6) +
  north("tr") + scale_bar("br", 0.22) +
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
  station_layers(size_rng = c(1.8, 5), legend = FALSE) +
  geom_sf(data = utm(cities[1, ]), shape = 22, size = 1.6, fill = "white",
          colour = INK, stroke = 0.4) +
  geom_sf_text(data = utm(cities[1, ]), label = "Puno", size = 2.2,
               fontface = "bold", colour = INK, nudge_x = -3200) +
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
k <- data.frame(y = 5:1, lab = c("Lake Titicaca", "River (HydroSHEDS)",
                                 "Endorheic basin", "International border",
                                 "Department boundary"))
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
  geom_text(aes(x = 1.2, y = y, label = lab), hjust = 0, size = 2.3, colour = INK) +
  coord_cartesian(xlim = c(0, 6.5), ylim = c(0.5, 5.5), expand = FALSE) +
  theme_void()

# ------------------------------------------------------------------ (e) --
pe <- ggplot(st_summary, aes(n_campaigns, secchi_mean, colour = zone_label)) +
  geom_point(aes(size = n_visits), alpha = 0.65) +
  scale_colour_manual(values = PAL_ZONE, name = NULL) +
  scale_size_continuous(range = c(0.9, 3.2), guide = "none") +
  scale_x_continuous(breaks = scales::pretty_breaks(5)) +
  labs(x = "Campaigns in which the station was visited",
       y = "Mean measured Secchi (m)") +
  theme(legend.position = "bottom", legend.margin = margin(t = -6),
        axis.title = element_text(size = rel(0.72)),
        axis.text = element_text(size = rel(0.66)),
        legend.text = element_text(size = rel(0.66)),
        plot.margin = margin(4, 6, 2, 4))

# ------------------------------------------------------------- montaje --
# Posiciones en la hoja (0-1). Las lineas de zoom unen las esquinas de cada
# recuadro rojo con el panel que lo amplia; se midieron sobre el render.
tag <- function(l, x, y) draw_label(l, x = x, y = y, hjust = 0, vjust = 1,
                                    fontface = "bold", size = 10, colour = INK)
zoom <- function(x, y) draw_line(x = x, y = y, colour = BOX, linewidth = 0.35,
                                 linetype = "22")

fig <- ggdraw() +
  draw_plot(pa, x = 0.035, y = 0.755, width = 0.30, height = 0.240) +
  draw_plot(pb, x = 0.000, y = 0.395, width = 0.415, height = 0.350) +
  draw_plot(pc, x = 0.425, y = 0.585, width = 0.575, height = 0.410) +
  draw_plot(pd, x = 0.450, y = 0.395, width = 0.150, height = 0.175) +
  draw_plot(leg_st, x = 0.625, y = 0.475, width = 0.36, height = 0.100) +
  draw_plot(pk, x = 0.640, y = 0.395, width = 0.36, height = 0.085) +
  draw_plot(pe, x = 0.080, y = 0.000, width = 0.840, height = 0.370) +
  tag("a", 0.005, 0.995) + tag("b", 0.005, 0.745) + tag("c", 0.430, 0.995) +
  tag("d", 0.428, 0.572) + tag("e", 0.085, 0.372)

ZOOM <- read.csv(file.path(ROOT, "figures", "R", "fig02_zoom.csv"))
if (as.logical(Sys.getenv("FIG02_ZOOM", "TRUE"))) {
  for (i in seq_len(nrow(ZOOM))) {
    fig <- fig + zoom(c(ZOOM$x0[i], ZOOM$x1[i]), c(ZOOM$y0[i], ZOOM$y1[i]))
  }
}

save_fig(fig, "fig02_study_area", W2, 205)
cat("fig02 lista\n")
