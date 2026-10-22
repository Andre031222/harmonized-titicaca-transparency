# ============================================================================
# fig6_spatial.R -- Estructura espacial del error fuera de fold (p09)
#
# Unidad espacial = LOCALIZACION (coordenadas agrupadas a 500 m), no el codigo
# de estacion IMARPE, que cambia de sitio entre anios.
#
# (a) Mapa del lago: residuo medio fuera de fold por localizacion (prediccion
#     - medida), tamano = n match-ups; anillo = LISA P < 0.05 SIN ajustar
#     (ninguna sobrevive a FDR)
# (b) Ampliacion de la Bahia de Puno
# (c) Correlograma de Moran por anillo de distancia, residuo bruto y residuo
#     sin la media de zona, con la envolvente del 95 % de las permutaciones
# (d) Semivariograma empirico y modelo esferico ajustado
# (e) Residuo frente a distancia a la orilla, por zona, con Spearman
# ============================================================================

source(file.path(local({a <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(a)) dirname(normalizePath(sub("^--file=", "", a[1]))) else getwd()}),
  "theme_titicaca.R"))
source(file.path(ROOT, "figures", "R", "map_base.R"))
suppressPackageStartupMessages(library(jsonlite))

loc  <- read_tidy("lisa_locations.csv")
ring <- read_tidy("moran_correlogram.csv")
vemp <- read_tidy("variogram_empirical.csv")
vmod <- read_tidy("variogram_model.csv") |> filter(best)
shore <- read_tidy("shore_distance.csv")
sj   <- fromJSON(file.path(ROOT, "results/metrics/spatial.json"))

stopifnot(nrow(loc) == sj$n_locations, sj$lisa$n_fdr_significant == 0,
          all(loc$lisa_class == "ns"))

VAR_LAB <- c(residual = "Raw residual",
             residual_zone_demeaned = "Zone mean removed")
PAL_VAR <- c(residual = INK, residual_zone_demeaned = "#7B5EA7")
var_factor <- function(x) factor(map_strict(x, VAR_LAB, "variable"),
                                 levels = unname(VAR_LAB))
PAL_V2 <- setNames(unname(PAL_VAR), unname(VAR_LAB))

# --------------------------------------------------------------- mapa --
# escala divergente simetrica, centrada en 0; los extremos se saturan
LIM <- ceiling(quantile(abs(loc$residual), 0.98) * 2) / 2
RES_PAL <- c("#2166AC", "#67A9CF", "#F7F7F7", "#EF8A62", "#B2182B")

loc_sf <- loc |>
  mutate(zl = zone_factor(zona),
         lisa_u = lisa_class_unadjusted != "ns") |>
  arrange(n) |>
  st_as_sf(coords = c("lon", "lat"), crs = 4326) |> st_transform(UTM)
sig_sf <- filter(loc_sf, lisa_u)
n_sig <- nrow(sig_sf)
stopifnot(n_sig == sj$lisa$n_unadjusted_significant)
SIG_LAB <- sprintf("LISA P < 0.05, unadjusted (%d)", n_sig)
# localizaciones reales que caen fuera del contorno del lago (HydroLAKES)
n_off <- sum(!loc$inside_polygon)
stopifnot(n_off == sj$shore_distance$n_outside_polygon)
OFF_LAB <- sprintf("Outside lake outline (%d)", n_off)
IN_LAB  <- "Inside outline"
loc_sf <- loc_sf |> mutate(outline = ifelse(inside_polygon, IN_LAB, OFF_LAB))
sig_sf <- filter(loc_sf, lisa_u)
SHAPE_OUT <- setNames(c(21, 23), c(IN_LAB, OFF_LAB))

# recuadro de la bahia: sus localizaciones con 2.5 km de margen, cuadrado
bay <- loc_sf |> filter(zl == "Bahía de Puno") |> st_bbox()
bay <- bay + c(-2500, -2500, 2500, 2500)
half <- max(bay[["xmax"]] - bay[["xmin"]], bay[["ymax"]] - bay[["ymin"]]) / 2
cx <- (bay[["xmin"]] + bay[["xmax"]]) / 2; cy <- (bay[["ymin"]] + bay[["ymax"]]) / 2
bay[c("xmin", "ymin", "xmax", "ymax")] <- c(cx - half, cy - half, cx + half, cy + half)
bay_poly <- st_as_sfc(bay)

N_BRK <- c(1, 5, 10)
res_layers <- function(size_rng) {
  list(
    geom_sf(data = filter(loc_sf, !lisa_u), aes(fill = residual, size = n,
                                                shape = outline),
            colour = INK_2, stroke = 0.2),
    # las localizaciones con LISA P < 0.05 sin ajustar llevan borde grueso
    geom_sf(data = sig_sf, aes(fill = residual, size = n, colour = SIG_LAB,
                               shape = outline),
            stroke = 0.75),
    scale_shape_manual(values = SHAPE_OUT, breaks = OFF_LAB, name = NULL,
                       guide = guide_legend(order = 4, override.aes = list(
                         size = 2.4, fill = "white", stroke = 0.4))),
    scale_colour_manual(values = INK, name = NULL,
                        guide = guide_legend(order = 3, override.aes = list(
                          size = 2.6, fill = "white", stroke = 0.75, shape = 21))),
    scale_fill_gradientn(colours = RES_PAL, limits = c(-LIM, LIM), oob = squish,
                         name = "Mean residual (m)\npredicted − measured",
                         guide = guide_colourbar(direction = "horizontal",
                                                 title.position = "top",
                                                 barwidth = unit(62, "pt"),
                                                 barheight = unit(5, "pt"),
                                                 order = 1)),
    scale_size(range = size_rng, breaks = N_BRK, limits = c(1, max(loc$n)),
               name = "Match-ups",
               guide = guide_legend(direction = "horizontal",
                                    title.position = "left", order = 2,
                                    override.aes = list(fill = "grey80"))))
}

pa <- ggplot() +
  relief_layers() +
  res_layers(c(0.9, 3.4)) +
  geom_sf(data = bay_poly, fill = NA, colour = BOX, linewidth = 0.6) +
  annotate("text", x = bay[["xmin"]] + 1200, y = bay[["ymax"]] - 1200,
           label = "b", hjust = 0, vjust = 1, colour = BOX, fontface = "bold",
           size = 3) +
  north("tr") + scale_bar("br", 0.25) +
  lake_coord() +
  scale_x_continuous(breaks = c(-70, -69.5, -69, -68.5)) +
  scale_y_continuous(breaks = c(-16.5, -16, -15.5)) +
  map_theme + legend_box(c(0.012, 0.012), c(0, 0)) +
  theme(legend.key.size = unit(8, "pt"), legend.text = element_text(size = 6.5),
        legend.title = element_text(size = 6.8),
        plot.margin = margin(4, 2, 2, 2))

pb <- ggplot() +
  relief_layers() +
  res_layers(c(1.6, 6)) +
  scale_bar("bl", 0.35) +
  coord_sf(crs = UTM, datum = 4326, xlim = bay[c("xmin", "xmax")],
           ylim = bay[c("ymin", "ymax")], expand = FALSE) +
  scale_x_continuous(breaks = c(-70, -69.9, -69.8)) +
  scale_y_continuous(breaks = c(-15.9, -15.8, -15.7)) +
  map_theme +
  theme(legend.position = "none",
        panel.border = element_rect(fill = NA, colour = BOX, linewidth = 1),
        plot.margin = margin(4, 2, 2, 8))

# ------------------------------------------------------------------ (c) --
# las dos series se separan un poco en el eje log para que no se tapen
DX <- c(residual = 0.94, residual_zone_demeaned = 1.06)
rc <- ring |>
  mutate(x = mid_km * DX[variable], v = var_factor(variable),
         sig = p_two_sided < 0.05)
e_i <- unique(round(ring$E_I, 6)); stopifnot(length(e_i) == 1)
pc <- ggplot(rc, aes(x, I, colour = v)) +
  geom_hline(yintercept = e_i, colour = INK_2, linetype = "22", linewidth = 0.3) +
  geom_linerange(aes(ymin = env_lo, ymax = env_hi), linewidth = 2.2,
                 alpha = 0.22) +
  geom_line(linewidth = 0.5) +
  geom_point(aes(shape = sig), fill = "white", size = 1.7, stroke = 0.6) +
  scale_colour_manual(values = PAL_V2, name = NULL) +
  scale_shape_manual(values = c(`TRUE` = 16, `FALSE` = 21),
                     labels = c(`TRUE` = "P < 0.05", `FALSE` = "P ≥ 0.05"),
                     name = NULL) +
  scale_x_log10(breaks = unique(ring$mid_km),
                labels = sprintf("%g–%g", unique(ring$lo_km), unique(ring$hi_km))) +
  labs(x = "Distance band (km)", y = expression("Moran's "*italic(I))) +
  guides(colour = guide_legend(order = 1, override.aes = list(linewidth = 0.5)),
         shape = guide_legend(order = 2)) +
  theme(axis.text.x = element_text(size = 6.5, angle = 35, hjust = 1),
        legend.position = "inside", legend.position.inside = c(1, 1),
        legend.justification = c(1, 1), legend.key.size = unit(7, "pt"),
        legend.text = element_text(size = 6.5), legend.spacing.y = unit(0, "pt"))

# ------------------------------------------------------------------ (d) --
vg <- sj$variogram
vpar <- tibble(variable = c("residual", "residual_zone_demeaned"),
               nugget = c(vg$residual$nugget, vg$zone_demeaned$nugget),
               sill = c(vg$residual$sill, vg$zone_demeaned$sill),
               range = c(vg$residual$range_km, vg$zone_demeaned$range_km)) |>
  mutate(v = var_factor(variable),
         txt = sprintf("nugget %.2f, sill %.2f\nrange %.1f km",
                       nugget, sill, range))
stopifnot(all(vmod$model == "spherical"))
h_max <- max(vemp$hi_km)
ve <- vemp |> mutate(v = var_factor(variable))
vm <- vmod |> filter(h_km <= h_max) |> mutate(v = var_factor(variable))
y_rng <- range(c(ve$gamma, vm$gamma))
y_lab <- y_rng[1] + c(0.19, 0.03) * diff(y_rng)
pd <- ggplot(ve, aes(h_km, gamma, colour = v)) +
  geom_vline(data = vpar, aes(xintercept = range, colour = v), linetype = "23",
             linewidth = 0.35, show.legend = FALSE) +
  geom_line(data = vm, linewidth = 0.6) +
  geom_point(aes(size = pairs), shape = 21, fill = "white", stroke = 0.55) +
  geom_text(data = vpar, aes(x = h_max, y = y_lab, label = txt), hjust = 1,
            vjust = 0.5, size = 2.3, lineheight = 0.95, show.legend = FALSE) +
  scale_colour_manual(values = PAL_V2, guide = "none") +
  scale_size(range = c(0.8, 2.2), guide = "none") +
  scale_x_continuous(limits = c(0, h_max), expand = expansion(add = c(1, 2))) +
  scale_y_continuous(expand = expansion(mult = c(0.06, 0.06))) +
  labs(x = "Lag distance (km)",
       y = expression("Semivariance "*gamma*" (m"^2*")"))

# ------------------------------------------------------------------ (e) --
# estadisticos de orilla tal como los escribe p09 (shore_distance.csv); p09
# asigna shore_km = 0 a las localizaciones fuera del contorno, que aqui se
# dibujan en 0 con un eje pseudo-log y como rombo
SH_ZONE <- shore |> filter(subset %in% names(ZONE_NAME)) |>
  mutate(zl = zone_factor(subset)) |> arrange(zl) |>
  mutate(ptxt = ifelse(p_signed < 0.001, "P < 0.001",
                       sprintf("P = %.3f", p_signed)),
         txt = sprintf("%s: ρ = %+.2f, %s (n = %d)", zl, rho_signed, ptxt, n))
stopifnot(nrow(SH_ZONE) == 3, sum(SH_ZONE$n) == nrow(loc),
          all(loc$shore_km[!loc$inside_polygon] == 0))
PAL_ZONE_TXT <- setNames(colorspace::darken(PAL_ZONE, 0.25), names(PAL_ZONE))
le <- loc |> mutate(zl = zone_factor(zona),
                    outline = ifelse(inside_polygon, IN_LAB, OFF_LAB))
y_top <- max(le$residual)
pe <- ggplot(le, aes(shore_km, residual)) +
  geom_hline(yintercept = 0, colour = INK_2, linetype = "22", linewidth = 0.3) +
  geom_point(aes(fill = zl, shape = outline), colour = "white", stroke = 0.2,
             size = 1.4, alpha = 0.9) +
  geom_text(data = SH_ZONE, aes(x = 0, y = y_top + (3:1 - 0.5) * 0.11 *
                                  diff(range(le$residual)) + 0.3,
                                label = txt, colour = zl),
            hjust = 0, size = 2.3, show.legend = FALSE) +
  scale_fill_manual(values = PAL_ZONE, guide = "none") +
  scale_colour_manual(values = PAL_ZONE_TXT, guide = "none") +
  scale_shape_manual(values = SHAPE_OUT, guide = "none") +
  scale_x_continuous(transform = pseudo_log_trans(sigma = 0.01, base = 10),
                     breaks = c(0, 0.1, 1, 10), labels = c("0", "0.1", "1", "10")) +
  scale_y_continuous(expand = expansion(mult = c(0.04, 0.02))) +
  labs(x = "Distance to shore (km, log)", y = "Mean residual (m)")

# ---------------------------------------------------------------- montaje --
top <- (pa | pb) + plot_layout(widths = c(1.45, 1))
bot <- (pc | pd | pe) + plot_layout(widths = c(1, 1, 1))
fig <- top / bot + plot_layout(heights = c(1.3, 1)) + tags_abc()
save_fig(fig, "fig6_spatial_errors", W2, 178)
cat("fig6 lista\n")
