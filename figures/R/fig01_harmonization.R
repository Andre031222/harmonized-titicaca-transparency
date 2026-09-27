# ============================================================================
# fig01_harmonization.R -- FIGURA PRINCIPAL DEL ARTICULO
#
# Por que el ajuste de banda de HLS (Landsat 8/9 <-> Sentinel-2) no vuelve
# intercambiables a los sensores sobre agua oligotrofica clara:
#   (a) cuanto mueve el ajuste a Sentinel-2 sobre el agua, frente al 2 % para
#       el que esta disenado sobre tierra
#   (b) prueba de dos muestras con sus controles: los sensores siguen siendo
#       distinguibles, y dos mitades del mismo sensor no
#   (c) acuerdo banda a banda en los eventos vistos por ambos sensores
#   (d) la firma espectral de cada sensor en cada etapa
# ============================================================================

source(file.path(local({a <- grep("^--file=", commandArgs(FALSE), value = TRUE); if (length(a)) dirname(normalizePath(sub("^--file=", "", a[1]))) else getwd()}), "theme_titicaca.R"))

bands <- read_tidy("harmonization_bands.csv")
sep   <- read_tidy("sensor_separability.csv")
pair  <- read_tidy("harmonization_paired.csv")
sig   <- read_tidy("spectral_signature.csv")

BAND_ORDER <- c("Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2")
ord <- function(x) factor(x, levels = BAND_ORDER)
BLUE <- "#2E6E8E"

# ---------------------------------------------------------------- panel (a) --
# Cambio relativo de la mediana de Sentinel-2 al aplicar HLS. Sobre tierra el
# ajuste corrige diferencias de <2 % (Claverie et al. 2018); sobre agua oscura
# los mismos interceptos pesan mucho mas.
pa_df <- bands |> mutate(band_label = ord(band_label),
                         dir = ifelse(hls_shift_pct < 0, "down", "up"))
pa <- ggplot(pa_df, aes(band_label, hls_shift_pct, fill = dir)) +
  geom_blank() +
  annotate("rect", xmin = -Inf, xmax = Inf, ymin = -2, ymax = 2,
           fill = NEUTRAL, alpha = 0.25) +
  geom_hline(yintercept = 0, colour = INK, linewidth = 0.35) +
  geom_col(width = 0.62) +
  geom_text(aes(label = sprintf("%+.0f%%", hls_shift_pct),
                vjust = ifelse(hls_shift_pct < 0, 1.4, -0.5)),
            size = 2.5, colour = INK, fontface = "bold") +
  scale_fill_manual(values = c(down = "#B35806", up = BLUE), guide = "none") +
  scale_y_continuous(labels = function(x) paste0(x, "%"),
                     limits = c(-37, 24), breaks = seq(-30, 20, 10)) +
  scale_x_discrete(labels = function(x) sub("SWIR", "SWIR\n", x)) +
  labs(x = NULL, y = "Change in Sentinel-2 over water\nfrom the HLS adjustment") +
  theme(axis.ticks.x = element_blank(), panel.grid.major.y = element_blank(),
        axis.text.x = element_text(lineheight = 0.9))

# ---------------------------------------------------------------- panel (b) --
# Exactitud balanceada (azar = 50 %) con GroupKFold por fecha de muestreo.
TEST_LAB <- c(all_raw = "All match-ups, no adjustment",
              all_hls = "All match-ups, HLS",
              paired_hls = "Paired events only, HLS",
              pm3d_hls = "Pairs within ±3 days, HLS",
              paired_recal = "Paired events, local recalibration",
              pm3d_recal = "Pairs within ±3 days, local recal.",
              negative_ls = "Control: Landsat vs Landsat",
              negative_s2 = "Control: Sentinel-2 vs Sentinel-2")
GRP <- c(all_raw = "Sensors", all_hls = "Sensors", paired_hls = "Sensors",
         pm3d_hls = "Sensors", paired_recal = "Sensors", pm3d_recal = "Sensors",
         negative_ls = "Same sensor", negative_s2 = "Same sensor")
wilson <- function(p, n, z = 1.96) {
  c0 <- (p + z^2 / (2 * n)) / (1 + z^2 / n)
  h  <- z * sqrt(p * (1 - p) / n + z^2 / (4 * n^2)) / (1 + z^2 / n)
  list(lo = c0 - h, hi = c0 + h)
}
pb_df <- sep |>
  filter(test %in% names(TEST_LAB)) |>
  mutate(label = factor(map_strict(test, TEST_LAB, "test"),
                        levels = rev(unname(TEST_LAB))),
         grp = map_strict(test, GRP, "grupo"),
         lo = wilson(balanced_accuracy, n)$lo, hi = wilson(balanced_accuracy, n)$hi)
stopifnot(nrow(pb_df) == length(TEST_LAB))

pb <- ggplot(pb_df, aes(y = label)) +
  geom_rect(aes(xmin = 0.5, xmax = balanced_accuracy,
                ymin = as.numeric(label) - 0.3, ymax = as.numeric(label) + 0.3,
                fill = grp), alpha = 0.9) +
  geom_errorbar(aes(xmin = lo, xmax = hi), width = 0.22, colour = INK,
                linewidth = 0.4, orientation = "y") +
  geom_vline(xintercept = 0.5, colour = INK, linetype = "22", linewidth = 0.45) +
  geom_text(aes(x = pmax(hi, 0.56), label = sprintf("%.0f%%  AUC %.2f",
                                                   100 * balanced_accuracy, auc)),
            hjust = -0.08, size = 2.3, colour = INK) +
  annotate("text", x = 0.505, y = 8.6, label = "chance", hjust = 0, size = 2.2,
           colour = INK_2, fontface = "italic") +
  scale_fill_manual(values = c(Sensors = ACCENT, `Same sensor` = NEUTRAL),
                    guide = "none") +
  scale_x_continuous(labels = percent_format(accuracy = 1),
                     limits = c(0.45, 1.32), breaks = seq(0.5, 1, 0.1),
                     expand = expansion(mult = c(0, 0))) +
  scale_y_discrete(expand = expansion(add = c(0.6, 0.9))) +
  coord_cartesian(clip = "off") +
  labs(x = "Balanced accuracy of the sensor classifier", y = NULL) +
  theme(panel.grid.major.y = element_blank())

# ---------------------------------------------------------------- panel (c) --
# Landsat nativo frente a Sentinel-2 con HLS en los eventos pareados. Cada
# panel se acerca al percentil 1-99 de cada sensor; r y sesgo usan todos.
stats_c <- pair |> mutate(band_label = ord(band_label)) |>
  group_by(band_label) |>
  summarise(r = cor(s2_hls, ls_native), bias = mean(ls_native - s2_hls),
            .groups = "drop") |>
  mutate(strip = sprintf("%s\nr = %.2f, bias %+.4f", band_label, r, bias))
pc_df <- pair |> mutate(band_label = ord(band_label)) |> group_by(band_label) |>
  filter(between(s2_hls, quantile(s2_hls, 0.01), quantile(s2_hls, 0.99)),
         between(ls_native, quantile(ls_native, 0.01), quantile(ls_native, 0.99))) |>
  ungroup() |>
  left_join(select(stats_c, band_label, strip), by = "band_label") |>
  mutate(strip = factor(strip, levels = stats_c$strip[order(stats_c$band_label)]))
dens_at <- function(x, y) {
  k <- MASS::kde2d(x, y, n = 120)
  d <- k$z[cbind(findInterval(x, k$x, all.inside = TRUE),
                 findInterval(y, k$y, all.inside = TRUE))]
  d / max(d)
}
pc_df <- pc_df |> group_by(band_label) |> mutate(dens = dens_at(s2_hls, ls_native)) |>
  ungroup() |> arrange(dens)

pc <- ggplot(pc_df, aes(s2_hls, ls_native)) +
  geom_hline(yintercept = 0, colour = INK_2, linewidth = 0.25) +
  geom_point(aes(colour = dens), size = 0.9, shape = 16) +
  geom_abline(slope = 1, intercept = 0, colour = INK, linetype = "22",
              linewidth = 0.35) +
  geom_smooth(method = "lm", formula = y ~ x, se = FALSE, colour = ACCENT,
              linewidth = 0.6) +
  facet_wrap(~strip, nrow = 1, scales = "free") +
  scale_colour_viridis_c(option = "viridis", name = "Relative\ndensity",
                         breaks = c(0.2, 0.6, 1),
                         guide = guide_colourbar(barwidth = unit(4, "pt"),
                                                 barheight = unit(34, "pt"))) +
  scale_x_continuous(breaks = breaks_pretty(n = 2),
                     labels = label_number(accuracy = 0.001)) +
  scale_y_continuous(breaks = breaks_pretty(n = 3),
                     labels = label_number(accuracy = 0.001)) +
  labs(x = "Sentinel-2 reflectance after the HLS adjustment",
       y = "Landsat 8/9\nreflectance") +
  theme(aspect.ratio = 0.95, panel.grid.major.y = element_blank(),
        axis.text = element_text(size = 5.5), legend.position = "right",
        strip.clip = "off",
        strip.text = element_text(size = 5.8, hjust = 0, face = "plain",
                                  lineheight = 1.1),
        legend.title = element_text(size = 6.5), legend.text = element_text(size = 6))

# ---------------------------------------------------------------- panel (d) --
# Mediana de cada sensor por banda: Sentinel-2 sin ajustar -> con HLS -> con
# recalibracion local, frente a Landsat nativo. El NIR de Landsat es negativo:
# sobre el agua no hay senal que armonizar.
STAGE <- c("Sentinel-2 unadjusted" = "raw", "HLS bandpass adjustment" = "hls",
           "Local water recalibration" = "recal")
s2 <- sig |> filter(sensor == "Sentinel-2") |>
  mutate(stage = map_strict(stage, STAGE, "etapa"), band_label = ord(band_label)) |>
  select(band_label, stage, median) |>
  pivot_wider(names_from = stage, values_from = median)
ls <- sig |> filter(sensor == "Landsat 8/9", stage == "HLS bandpass adjustment") |>
  mutate(band_label = ord(band_label)) |> select(band_label, ls = median)
pd_df <- s2 |> left_join(ls, by = "band_label") |>
  mutate(x = as.numeric(band_label), ratio = sprintf("%.2f", ls / hls))
DX <- 0.17
SERIES <- c(ls = "Landsat 8/9, native", raw = "Sentinel-2, unadjusted",
            hls = "Sentinel-2, HLS adjusted", recal = "Sentinel-2, local recalibration")
pts <- pd_df |>
  transmute(band_label, x, ls, raw, hls, recal) |>
  pivot_longer(c(ls, raw, hls, recal), names_to = "series", values_to = "v") |>
  mutate(xp = x + ifelse(series == "ls", -DX, DX),
         series = factor(series, levels = names(SERIES)))

pd <- ggplot() +
  geom_hline(yintercept = 0, colour = INK, linewidth = 0.3) +
  geom_segment(data = pd_df, aes(x = x + DX, xend = x + DX, y = raw, yend = hls),
               colour = BLUE, linewidth = 0.5,
               arrow = arrow(length = unit(3.5, "pt"), type = "closed")) +
  geom_segment(data = pd_df, aes(x = x + DX, xend = x + DX, y = hls, yend = recal),
               colour = "#6B8E5A", linewidth = 0.5, linetype = "22",
               arrow = arrow(length = unit(3.5, "pt"), type = "closed")) +
  geom_point(data = pts, aes(xp, v, shape = series, fill = series, colour = series),
             size = 2.1, stroke = 0.7) +
  geom_text(data = pd_df, aes(x = x, y = 0.0172, label = ratio), size = 2.3,
            colour = INK_2) +
  annotate("text", x = 0.55, y = 0.0172, label = "LS / S2 (HLS)", hjust = 1,
           size = 2.1, colour = INK_2, fontface = "italic") +
  scale_shape_manual(values = c(ls = 23, raw = 21, hls = 21, recal = 24),
                     labels = SERIES, name = NULL) +
  scale_fill_manual(values = c(ls = INK, raw = "white", hls = BLUE, recal = "#6B8E5A"),
                    labels = SERIES, name = NULL) +
  scale_colour_manual(values = c(ls = INK, raw = BLUE, hls = BLUE, recal = "#6B8E5A"),
                      labels = SERIES, name = NULL) +
  scale_x_continuous(breaks = 1:6, labels = BAND_ORDER,
                     expand = expansion(add = c(0.75, 0.4))) +
  scale_y_continuous(labels = label_number(accuracy = 0.005),
                     breaks = seq(0, 0.015, 0.005), limits = c(-0.0012, 0.018)) +
  coord_cartesian(clip = "off") +
  labs(x = NULL, y = "Median reflectance over water") +
  theme(legend.position = "inside", legend.position.inside = c(0.99, 0.82),
        legend.justification = c(1, 1), legend.key.size = unit(8, "pt"),
        legend.text = element_text(size = 6.5), legend.background = element_blank(),
        axis.ticks.x = element_blank())

# ---------------------------------------------------------------- montaje ----
fig <- ((pa | pb) + plot_layout(widths = c(0.85, 1.25))) / pc / pd +
  plot_layout(heights = c(1, 0.66, 1)) + tags_abc()
save_fig(fig, "fig01_harmonization_failure", W2, 180)
cat("fig01 lista\n")
