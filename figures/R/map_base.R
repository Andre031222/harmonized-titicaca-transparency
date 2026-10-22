# ============================================================================
# map_base.R -- Fondo cartografico comun de los mapas del lago (fig04, fig09)
#
# Relieve SRTM sombreado con tinte hipsometrico, rios, frontera, orilla, marco
# con reticula en grados, flecha de norte y barra de escala. Mismas capas que
# fig02 (data/basemap/, figures/basemap/fetch_basemap.py). fig02 conserva su
# propio codigo: no se toca.
# ============================================================================

suppressPackageStartupMessages({
  library(sf); library(terra); library(tidyterra)
  library(ggspatial); library(ggnewscale)
})
sf_use_s2(FALSE)

UTM   <- 32719
BM    <- file.path(ROOT, "data", "basemap")
WATER <- "#D6E9F3"
SHORE <- "#2B7BB9"
BOX   <- "#C2182B"

utm <- function(x) st_transform(x, UTM)
ext_ll <- function(x0, x1, y0, y1) {
  st_as_sfc(st_bbox(c(xmin = x0, xmax = x1, ymin = y0, ymax = y1), crs = 4326))
}
lims_utm <- function(e) {
  b <- st_bbox(utm(e))
  list(x = unname(b[c(1, 3)]), y = unname(b[c(2, 4)]))
}

E_LAKE <- ext_ll(-70.14, -68.40, -16.70, -15.18)

.rd <- function(f) st_read(file.path(BM, f), quiet = TRUE)
lake_ll <- st_transform(st_read(file.path(ROOT, "data/lake_boundary/titicaca.gpkg"),
                                quiet = TRUE), 4326)
ctry_ll   <- .rd("countries_region.gpkg")
rivers_ll <- .rd("rivers_region.gpkg")

# relieve recortado a la extension del lago con un margen, en UTM
.dem_cache <- NULL
relief <- function() {
  if (is.null(.dem_cache)) {
    dem <- rast(file.path(BM, "srtm_region.tif")) |>
      crop(ext(-70.4, -68.2, -16.9, -15.0)) |>
      project(paste0("EPSG:", UTM))
    hill <- shade(terrain(dem, "slope", unit = "radians"),
                  terrain(dem, "aspect", unit = "radians"), 40, 315)
    names(dem) <- "elev"; names(hill) <- "hill"
    .dem_cache <<- list(dem = dem, hill = hill)
  }
  .dem_cache
}

# capas de fondo; el agua va encima del relieve para que los puntos se lean
relief_layers <- function(elev_legend = FALSE) {
  r <- relief()
  list(
    geom_spatraster(data = r$hill, show.legend = FALSE, maxcell = 8e5),
    scale_fill_gradient(low = "#4A4A4A", high = "#FFFFFF", na.value = NA),
    new_scale_fill(),
    geom_spatraster(data = r$dem, alpha = 0.72, maxcell = 8e5,
                    show.legend = elev_legend),
    # verde del altiplano bajo a pardo y blanco en las cumbres, como los mapas
    # de relieve de referencia; lo que cae fuera se satura, no se borra
    # (las paletas hypso de tidyterra van ancladas a cotas absolutas: sobre
    # 3700 m todo salia blanco)
    scale_fill_gradientn(colours = c("#6E9F5C", "#A9C27A", "#E3D99A", "#D2AE78",
                                     "#A67C55", "#8A6A55", "#F2EEE8"),
                         values = scales::rescale(c(3700, 3900, 4150, 4450, 4800,
                                            5150, 5600), from = c(3700, 5600)),
                         limits = c(3700, 5600), oob = squish, na.value = NA,
                         name = "Elevation\n(m a.s.l.)",
                         breaks = c(4000, 4800, 5600)),
    new_scale_fill(),
    geom_sf(data = utm(rivers_ll), aes(linewidth = UPLAND_SKM), colour = SHORE,
            show.legend = FALSE),
    scale_linewidth(range = c(0.15, 0.6), transform = "log10"),
    geom_sf(data = utm(lake_ll), fill = WATER, colour = SHORE, linewidth = 0.35),
    geom_sf(data = utm(ctry_ll), fill = NA, colour = INK, linewidth = 0.35,
            linetype = "42"))
}

map_theme <- theme(
  panel.grid.major = element_line(colour = alpha("white", 0.55), linewidth = 0.2,
                                  linetype = "22"),
  panel.background = element_rect(fill = NA, colour = NA),
  panel.border = element_rect(fill = NA, colour = INK, linewidth = 0.5),
  panel.ontop = FALSE,
  axis.text = element_text(size = rel(0.78), colour = INK_2),
  axis.title = element_blank(), axis.line = element_blank(),
  axis.ticks = element_line(colour = INK, linewidth = 0.3),
  legend.title = element_text(size = rel(0.82)),
  legend.text = element_text(size = rel(0.78)))

north <- function(loc = "tr", w = 0.75, h = 0.95) {
  annotation_north_arrow(location = loc, which_north = "true",
                         height = unit(h, "cm"), width = unit(w, "cm"),
                         pad_x = unit(0.12, "cm"), pad_y = unit(0.12, "cm"),
                         style = north_arrow_fancy_orienteering(
                           text_size = 6, line_width = 0.6,
                           fill = c("white", INK)))
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
        legend.background = element_rect(fill = alpha("white", 0.88), colour = NA),
        legend.margin = margin(3, 4, 3, 4))
}
lake_coord <- function(e = E_LAKE) {
  L <- lims_utm(e)
  coord_sf(crs = UTM, datum = 4326, xlim = L$x, ylim = L$y, expand = FALSE)
}
