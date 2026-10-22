# ============================================================================
# fig8_lake_maps.R -- Mapas de Secchi de todo el lago, temporada seca 2024 (p16)
#
# (a) RF conjunto aplicado al compuesto HLSL30 (Landsat 8/9)
# (b) el mismo RF aplicado al compuesto HLSS30 (Sentinel-2)
# (c) diferencia S2 - LS (consecuencia de que los sensores no sean
#     intercambiables), escala divergente centrada en 0
# (d) prediccion frente a lectura de campo 2024 (RF reajustado sin la campana
#     2024, prueba fuera de campana)
# (e) medias por zona de los mapas y de las lecturas de campo
# (f) distribucion por pixel de la diferencia S2 - LS frente al semi-ancho
#     del intervalo conformal del 90 %
# ============================================================================

source(file.path(local({a <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(a)) dirname(normalizePath(sub("^--file=", "", a[1]))) else getwd()}),
  "theme_titicaca.R"))
source(file.path(ROOT, "figures", "R", "map_base.R"))
suppressPackageStartupMessages(library(jsonlite))

mj    <- fromJSON(file.path(ROOT, "results/metrics/lake_maps.json"))
field <- read_tidy("lake_map_field.csv")
zones <- read_tidy("lake_map_zones.csv")
hist  <- read_tidy("lake_map_diff_hist.csv")
maps  <- rast(file.path(ROOT, "data/processed/lake_composites/lake_secchi_maps_2024.tif"))
names(maps) <- c("LS_pooled", "S2_pooled", "diff_pooled", "LS_single", "S2_single",
                 "diff_single", "outside_range_LS", "outside_range_S2",
                 "n_clear_LS", "n_clear_S2")

LAND   <- "#F4F2EE"
NODATA <- "#C9C9C9"
HW     <- mj$halfwidth90_m$marginal_split

# --------------------------------------------------------------- mapas --
sec_rng <- range(c(global(maps$LS_pooled, "range", na.rm = TRUE),
                   global(maps$S2_pooled, "range", na.rm = TRUE)))
SEC_LIM <- c(floor(sec_rng[1]), ceiling(sec_rng[2]))
d_q     <- quantile(values(maps$diff_pooled, na.rm = TRUE), c(0.01, 0.99))
DLIM    <- ceiling(max(abs(d_q)) * 2) / 2
RES_PAL <- c("#2166AC", "#67A9CF", "#F7F7F7", "#EF8A62", "#B2182B")

L <- lims_utm(E_LAKE)
base_map <- function() {
  list(geom_sf(data = utm(ctry_ll), fill = LAND, colour = NA),
       geom_sf(data = utm(lake_ll), fill = NODATA, colour = NA))
}
top_map <- function() {
  list(geom_sf(data = utm(lake_ll), fill = NA, colour = SHORE, linewidth = 0.3),
       geom_sf(data = utm(ctry_ll), fill = NA, colour = INK_2, linewidth = 0.3,
               linetype = "42"),
       coord_sf(crs = UTM, datum = 4326, xlim = L$x, ylim = L$y, expand = FALSE),
       scale_x_continuous(breaks = seq(-70, -68.5, 0.5)),
       scale_y_continuous(breaks = seq(-16.5, -15.25, 0.5)),
       map_theme,
       theme(panel.background = element_rect(fill = LAND, colour = NA),
             panel.grid.major = element_blank(),
             axis.text = element_text(size = 6.5)))
}
sec_scale <- scale_fill_viridis_c(
  option = "mako", direction = -1, limits = SEC_LIM, oob = squish,
  na.value = NA, name = "Secchi (m)", breaks = breaks_pretty(4),
  guide = guide_colourbar(barwidth = unit(4, "pt"), barheight = unit(36, "pt")))

sec_map <- function(lyr, legend = TRUE) {
  ggplot() + base_map() +
    geom_spatraster(data = maps[[lyr]], maxcell = 1.2e6) + sec_scale +
    top_map() +
    (if (legend) legend_box(c(0.985, 0.985), c(1, 1))
     else theme(legend.position = "none"))
}

pa <- sec_map("LS_pooled", legend = FALSE) + scale_bar("bl", 0.25)
pb <- sec_map("S2_pooled", legend = TRUE)
pc <- ggplot() + base_map() +
  geom_spatraster(data = maps$diff_pooled, maxcell = 1.2e6) +
  scale_fill_gradientn(colours = RES_PAL, limits = c(-DLIM, DLIM), oob = squish,
                       na.value = NA, name = "S2 − LS (m)",
                       breaks = breaks_pretty(3),
                       guide = guide_colourbar(barwidth = unit(4, "pt"),
                                               barheight = unit(36, "pt"))) +
  top_map() + legend_box(c(0.985, 0.985), c(1, 1)) + north("bl", 0.5, 0.65)

# ------------------------------------------------------------ (d) campo --
fd <- field |>
  filter(model == "pooled", training == "without_2024", !is.na(predicted)) |>
  mutate(sensor = factor(composite_label, levels = names(PAL_SENSOR)))
fm <- mj$field_2024$metrics$without_2024
lab_d <- tibble(
  sensor = factor(c("Landsat 8/9", "Sentinel-2"), levels = names(PAL_SENSOR)),
  txt = c(sprintf("Landsat 8/9: R² = %.2f, RMSE = %.2f m", fm$LS_pooled$R2, fm$LS_pooled$RMSE),
          sprintf("Sentinel-2: R² = %.2f, RMSE = %.2f m", fm$S2_pooled$R2, fm$S2_pooled$RMSE)))
AX <- c(0, ceiling(max(c(fd$secchi, fd$predicted)) / 2) * 2)
pd <- ggplot(fd, aes(secchi, predicted, colour = sensor)) +
  geom_abline(slope = 1, intercept = 0, colour = NEUTRAL, linewidth = 0.3,
              linetype = "22") +
  geom_line(aes(group = paste(location, sampling_date)), colour = "#BDBDBD",
            linewidth = 0.25) +
  geom_point(size = 1.1, alpha = 0.9, shape = 16) +
  geom_label(data = lab_d, aes(x = AX[1] + 0.3, y = AX[2] - 0.2 - c(0, 1.3), label = txt),
             hjust = 0, vjust = 1, size = 2.3, show.legend = FALSE, fill = "white",
             linewidth = 0, label.padding = unit(0.6, "pt")) +
  scale_colour_manual(values = PAL_SENSOR, guide = "none") +
  coord_equal(xlim = AX, ylim = AX, expand = FALSE) +
  scale_x_continuous(breaks = breaks_pretty(4)) +
  scale_y_continuous(breaks = breaks_pretty(4)) +
  labs(x = "Measured Secchi, 2024 (m)", y = "Map Secchi (m)")

# ----------------------------------------------------------- (e) zonas --
MAP_LAB <- c(LS_pooled = "Landsat 8/9 map", S2_pooled = "Sentinel-2 map",
             field_2024 = "Field 2024")
PAL_E <- c("Landsat 8/9 map" = unname(PAL_SENSOR["Landsat 8/9"]),
           "Sentinel-2 map"  = unname(PAL_SENSOR["Sentinel-2"]),
           "Field 2024"      = INK)
ze <- zones |>
  filter(map %in% names(MAP_LAB)) |>
  mutate(src = factor(map_strict(map, MAP_LAB, "mapa"), levels = unname(MAP_LAB)),
         zl = zone_factor(zona))
pe <- ggplot(ze, aes(zl, median, colour = src)) +
  geom_linerange(aes(ymin = q25, ymax = q75), linewidth = 0.5,
                 position = position_dodge(width = 0.6)) +
  geom_point(aes(shape = src), size = 1.5, fill = "white", stroke = 0.6,
             position = position_dodge(width = 0.6)) +
  scale_colour_manual(values = PAL_E, name = NULL) +
  scale_shape_manual(values = c(16, 16, 21), name = NULL) +
  scale_y_continuous(limits = c(0, NA), breaks = breaks_pretty(4),
                     expand = expansion(mult = c(0, 0.05))) +
  labs(x = NULL, y = "Secchi, median and IQR (m)") +
  theme(legend.position = "inside", legend.position.inside = c(0.02, 1),
        legend.justification = c(0, 1), legend.key.size = unit(7, "pt"),
        axis.text.x = element_text(size = 6.8))

# ------------------------------------------------------ (f) diferencia --
DIFF_LAB <- c(pooled = "Pooled model", single = "Single-sensor models")
PAL_F <- c("Pooled model" = INK, "Single-sensor models" = "#7B5EA7")
hf <- hist |> mutate(m = factor(map_strict(model, DIFF_LAB, "modelo"),
                                levels = unname(DIFF_LAB)))
XF <- c(-1, 1) * ceiling(max(HW, max(abs(c(hf$lo[hf$frac > 1e-4],
                                         hf$hi[hf$frac > 1e-4])))) + 0.25)
pf <- ggplot(hf, aes(mid, frac * 100, colour = m)) +
  annotate("rect", xmin = -HW, xmax = HW, ymin = -Inf, ymax = Inf,
           fill = "#EDEDED", colour = NA) +
  geom_vline(xintercept = 0, colour = NEUTRAL, linewidth = 0.3) +
  geom_step(direction = "mid", linewidth = 0.45) +
  scale_colour_manual(values = PAL_F, name = NULL) +
  scale_x_continuous(limits = XF, breaks = breaks_pretty(5)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.25))) +
  labs(x = "S2 − LS map difference (m)", y = "Pixels (%)") +
  theme(legend.position = "inside", legend.position.inside = c(0.02, 1),
        legend.justification = c(0, 1), legend.key.width = unit(10, "pt"))

# ------------------------------------------------------------ montaje --
top <- (pa | pb | pc)
bot <- (pd | pe | pf) + plot_layout(widths = c(1, 1.15, 1.15))
fig <- (top / bot) + plot_layout(heights = c(1, 0.85)) + tags_abc()
save_fig(fig, "fig8_lake_maps", W2, 116)
