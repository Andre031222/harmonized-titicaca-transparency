# ============================================================================
# figS4_paired_scatter.R -- Reflectancia pareada Landsat 8/9 vs Sentinel-2 del
# producto NASA HLS v2.0 en las bandas visibles.
#
# Filas: eventos pareados (ambos sensores dentro de la ventana de +-10 dias) y
# pares casi simultaneos (+-3 dias, una sola imagen por sensor). Columnas:
# azul, verde, rojo. Linea 1:1 discontinua; r de Pearson y n de
# harmonization_bands.csv (se comprueba contra los puntos dibujados).
#
# Entradas: results/tidy/harmonization_paired.csv, harmonization_bands.csv.
# Ningun numero escrito a mano.
# ============================================================================

source(file.path(local({a <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(a)) dirname(normalizePath(sub("^--file=", "", a[1]))) else getwd()}),
  "theme_titicaca.R"))

pair  <- read_tidy("harmonization_paired.csv")
bands <- read_tidy("harmonization_bands.csv")

ROUTE   <- "hls"
BANDS   <- c("Blue", "Green", "Red")
SAMPLES <- c(paired_w10  = "Paired events (±10 d)",
             pm3d_single = "Pairs ±3 d, single image")

pd <- pair |> filter(route == ROUTE, sample %in% names(SAMPLES),
                     band_label %in% BANDS) |>
  mutate(band_f = factor(band_label, levels = BANDS),
         samp_f = factor(map_strict(sample, SAMPLES, "muestra"), levels = SAMPLES))
st <- bands |> filter(route == ROUTE, sample %in% names(SAMPLES),
                      band_label %in% BANDS) |>
  select(sample, band_label, n, pearson_r, median_abs_diff)
chk <- pd |> group_by(sample, band_label) |>
  summarise(n_pts = n(), r_pts = cor(s2, ls), .groups = "drop") |>
  left_join(st, by = c("sample", "band_label"))
stopifnot(nrow(chk) == length(BANDS) * length(SAMPLES),
          all(chk$n_pts == chk$n), all(abs(chk$r_pts - chk$pearson_r) < 0.002))

# ejes iguales por banda (comunes a las dos filas) para que la 1:1 sea la
# diagonal de cada panel
lims <- pd |> group_by(band_f) |>
  summarise(lo = min(c(s2, ls)), hi = max(c(s2, ls)), .groups = "drop")
blank <- tidyr::crossing(lims, samp_f = factor(SAMPLES, levels = SAMPLES)) |>
  tidyr::pivot_longer(c(lo, hi), values_to = "v") |> mutate(s2 = v, ls = v)
lab <- st |>
  mutate(band_f = factor(band_label, levels = BANDS),
         samp_f = factor(map_strict(sample, SAMPLES, "muestra"), levels = SAMPLES),
         txt = sprintf("r = %.2f\nn = %d", pearson_r, n)) |>
  left_join(lims, by = "band_f") |>
  mutate(x = hi - 0.03 * (hi - lo), y = lo + 0.03 * (hi - lo))

refl_lab <- function(x) sub("^(-?)0\\.", "\\1.", format(x, drop0trailing = TRUE,
                                                        scientific = FALSE, trim = TRUE))

p <- ggplot(pd, aes(s2, ls)) +
  geom_blank(data = blank) +
  geom_hline(yintercept = 0, colour = GRID, linewidth = 0.3) +
  geom_vline(xintercept = 0, colour = GRID, linewidth = 0.3) +
  geom_abline(slope = 1, intercept = 0, colour = INK, linetype = "22",
              linewidth = 0.35) +
  geom_point(size = 0.8, shape = 16, alpha = 0.45,
             colour = PAL_SENSOR[["Sentinel-2"]]) +
  geom_label(data = lab, aes(x, y, label = txt), hjust = 1, vjust = 0,
             size = 2.3, colour = INK, lineheight = 0.95, fill = "white",
             label.size = 0, label.padding = unit(1.2, "pt")) +
  facet_wrap(vars(samp_f, band_f), nrow = 2, scales = "free",
             labeller = function(l) label_value(l, multi_line = FALSE)) +
  scale_x_continuous(breaks = breaks_pretty(n = 3), labels = refl_lab) +
  scale_y_continuous(breaks = breaks_pretty(n = 3), labels = refl_lab) +
  labs(x = "Sentinel-2 surface reflectance (HLS S30)",
       y = "Landsat 8/9 surface reflectance (HLS L30)") +
  theme(aspect.ratio = 1, panel.grid.major.y = element_blank(),
        strip.text = element_text(size = 7, face = "plain"))

save_fig(p, "figS4_harmonization_scatter", W15, 105)
print(as.data.frame(chk))
cat("figS4 lista\n")
