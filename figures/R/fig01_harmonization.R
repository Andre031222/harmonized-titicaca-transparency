# ============================================================================
# fig01_harmonization.R -- FIGURA PRINCIPAL DEL ARTICULO
#
# Por que la armonizacion Landsat->Sentinel-2 de Roy et al. (2016) no es
# valida sobre agua oligotrofica clara, en cuatro paneles:
#   (a) el intercepto terrestre frente a la senal real sobre el agua
#   (b) los dos sensores siguen siendo distinguibles despues de armonizar
#   (c) acuerdo banda a banda en los eventos vistos por ambos el mismo dia
#   (d) la firma espectral antes y despues de recalibrar localmente
# ============================================================================

source(file.path(local({a <- grep("^--file=", commandArgs(FALSE), value = TRUE); if (length(a)) dirname(normalizePath(sub("^--file=", "", a[1]))) else getwd()}), "theme_titicaca.R"))

bands <- read_tidy("harmonization_bands.csv")
sep   <- read_tidy("sensor_separability.csv")
coefs <- read_tidy("harmonization_local_coefficients.csv")
pair  <- read_tidy("harmonization_paired.csv")
sig   <- read_tidy("spectral_signature.csv")

BAND_ORDER <- c("Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2")
VERDICT_LAB <- c(bueno = "Good", aceptable = "Acceptable", debil = "Weak",
                 roto = "Broken")

ord <- function(x) factor(x, levels = BAND_ORDER)

# ---------------------------------------------------------------- panel (a) --
# El intercepto de Roy comparado con la senal nativa de Landsat sobre el agua,
# sobre franjas que marcan las cuatro clases. Escala log: va de 1.4x a 72x.
CLASS_BREAKS <- tibble(severity = c("bueno", "aceptable", "debil", "roto"),
                       lo = c(0.85, 2, 5, 20), hi = c(2, 5, 20, 130))
pa_df <- bands |>
  mutate(band_label = ord(band_label),
         severity = case_when(intercept_over_signal >= 20 ~ "roto",
                              intercept_over_signal >= 5  ~ "debil",
                              intercept_over_signal >= 2  ~ "aceptable",
                              TRUE ~ "bueno"))

pa <- ggplot(pa_df, aes(band_label, intercept_over_signal, fill = severity)) +
  geom_col(width = 0.66) +
  geom_hline(yintercept = 1, linewidth = 0.45, colour = INK, linetype = "22") +
  geom_text(aes(label = sprintf("%.1f×", intercept_over_signal)),
            vjust = -0.45, size = 2.55, colour = INK, fontface = "bold") +
  scale_fill_manual(values = PAL_VERDICT, labels = VERDICT_LAB,
                    breaks = names(VERDICT_LAB), name = NULL, drop = FALSE) +
  scale_y_log10(breaks = c(1, 3, 10, 30, 100),
                labels = c("1×", "3×", "10×", "30×", "100×"),
                expand = expansion(mult = c(0, 0.08))) +
  coord_cartesian(ylim = c(0.9, 110)) +
  labs(x = NULL, y = "Intercept / native signal") +
  theme(legend.position = "inside", legend.position.inside = c(0.02, 0.99),
        legend.justification = c(0, 1), legend.key.size = unit(6, "pt"),
        legend.background = element_blank(), axis.ticks.x = element_blank())

# ---------------------------------------------------------------- panel (b) --
# Si la armonizacion funcionara, un clasificador no deberia superar el azar.
# Intervalo de Wilson al 95 % sobre los n registros clasificados; el azar es
# la clase mayoritaria, asi que n sale de los conteos por sensor (p02).
chance <- sep$chance[1]
n_sep  <- sig |> filter(harmonization == first(harmonization),
                        band == first(band)) |> pull(n) |> sum()
stopifnot(abs(max(sig$n[1:2]) / n_sep - chance) < 1e-3)
wilson <- function(p, n, z = 1.96) {
  c0 <- (p + z^2 / (2 * n)) / (1 + z^2 / n)
  h  <- z * sqrt(p * (1 - p) / n + z^2 / (4 * n^2)) / (1 + z^2 / n)
  tibble(lo = c0 - h, hi = c0 + h)
}
pb_df <- sep |>
  mutate(label = recode(feature_set,
                        all_12_features = "All 12 features",
                        visible_only    = "Visible only",
                        ratios_only     = "Ratios only",
                        nir_swir_only   = "NIR + SWIR only",
                        green_blue_only = "Green and blue only"),
         label = fct_reorder(label, accuracy)) |>
  bind_cols(wilson(sep$accuracy, n_sep))

# barras flotantes desde el azar: su largo es lo que la armonizacion deja
# de identidad del sensor
pb <- ggplot(pb_df, aes(y = label)) +
  geom_rect(aes(xmin = chance, xmax = accuracy,
                ymin = as.numeric(label) - 0.3, ymax = as.numeric(label) + 0.3),
            fill = ACCENT, alpha = 0.85) +
  geom_errorbar(aes(xmin = lo, xmax = hi), width = 0.22, colour = INK,
                linewidth = 0.4, orientation = "y") +
  geom_vline(xintercept = chance, colour = INK, linetype = "22",
             linewidth = 0.45) +
  geom_text(aes(x = hi, label = sprintf("%.1f%%", 100 * accuracy)),
            hjust = -0.25, size = 2.5, colour = INK, fontface = "bold") +
  geom_text(aes(x = (chance + accuracy) / 2,
                label = sprintf("+%.1f pts", 100 * (accuracy - chance))),
            size = 2.2, colour = "white", fontface = "bold") +
  annotate("text", x = chance + 0.006, y = 0.5,
           label = sprintf("chance = %.1f%%", 100 * chance), hjust = 0,
           size = 2.2, colour = INK_2, fontface = "italic") +
  scale_x_continuous(labels = percent_format(accuracy = 1),
                     limits = c(0.6, 1.1), breaks = seq(0.6, 1, 0.1),
                     expand = expansion(mult = c(0, 0))) +
  scale_y_discrete(expand = expansion(add = c(0.75, 0.5))) +
  coord_cartesian(clip = "off") +
  labs(x = "Classifier accuracy (95% Wilson interval)", y = NULL) +
  theme(panel.grid.major.y = element_blank())

# ---------------------------------------------------------------- panel (c) --
# Densidad (hexbin) en los 270 eventos vistos por ambos sensores. Cada panel
# se acerca al percentil 1-99 de cada sensor; las metricas usan todos los
# eventos. La 1:1 solo aparece donde cae dentro del rango.
stats_c <- pair |> mutate(band_label = ord(band_label)) |>
  group_by(band_label) |>
  summarise(r = cor(ls_roy, s2), rmse = sqrt(mean((ls_roy - s2)^2)),
            bias = mean(ls_roy - s2), .groups = "drop") |>
  # las metricas van en la tira del panel: dentro tapaban los hexagonos
  mutate(strip = sprintf("%s\nr = %.2f, bias %+.3f", band_label, r, bias))
chk <- stats_c |> left_join(coefs |> mutate(band_label = ord(band_label)),
                            by = "band_label")
stopifnot(all(abs(chk$r - chk$pearson_r) < 0.005))

pc_df <- pair |> mutate(band_label = ord(band_label)) |> group_by(band_label) |>
  filter(between(ls_roy, quantile(ls_roy, 0.01), quantile(ls_roy, 0.99)),
         between(s2, quantile(s2, 0.01), quantile(s2, 0.99))) |>
  ungroup()

pc_df <- pc_df |> left_join(select(stats_c, band_label, strip), by = "band_label") |>
  mutate(strip = factor(strip, levels = stats_c$strip[order(stats_c$band_label)]))

# densidad KDE 2D por banda, evaluada en cada evento y reescalada a 0-1:
# con 270 eventos los hexagonos salian en bloques
dens_at <- function(x, y) {
  k <- MASS::kde2d(x, y, n = 120)
  d <- k$z[cbind(findInterval(x, k$x, all.inside = TRUE),
                 findInterval(y, k$y, all.inside = TRUE))]
  d / max(d)
}
pc_df <- pc_df |> group_by(band_label) |> mutate(dens = dens_at(ls_roy, s2)) |>
  ungroup() |> arrange(dens)

pc <- ggplot(pc_df, aes(ls_roy, s2)) +
  geom_point(aes(colour = dens), size = 0.9, shape = 16) +
  geom_abline(slope = 1, intercept = 0, colour = INK, linetype = "22",
              linewidth = 0.35) +
  geom_smooth(method = "lm", formula = y ~ x, se = FALSE,
              colour = ACCENT, linewidth = 0.6) +
  facet_wrap(~strip, nrow = 1, scales = "free") +
  scale_colour_viridis_c(option = "viridis", name = "Relative\ndensity",
                         breaks = c(0.2, 0.6, 1),
                       guide = guide_colourbar(barwidth = unit(4, "pt"),
                                               barheight = unit(34, "pt"))) +
  scale_x_continuous(breaks = breaks_pretty(n = 2),
                     labels = label_number(accuracy = 0.01)) +
  scale_y_continuous(breaks = breaks_pretty(n = 3),
                     labels = label_number(accuracy = 0.001)) +
  labs(x = "Landsat 8/9 reflectance, harmonized with Roy (2016)",
       y = "Sentinel-2\nreflectance") +
  theme(aspect.ratio = 0.95, panel.grid.major.y = element_blank(),
        axis.text = element_text(size = 5.5), legend.position = "right",
        strip.clip = "off",
        strip.text = element_text(size = 5.8, hjust = 0, face = "plain",
                                  lineheight = 1.1),
        legend.title = element_text(size = 6.5), legend.text = element_text(size = 6))

# ---------------------------------------------------------------- panel (d) --
# Firma espectral (mediana y rango intercuartil) antes y despues de la
# recalibracion local, y debajo la diferencia relativa Landsat - Sentinel-2.
HARM <- c("Roy (land-derived)" = "Roy (2016), land-derived",
          "Local water recalibration" = "Local water recalibration")
pd_df <- sig |>
  mutate(band_label = ord(band_label),
         harmonization = factor(map_strict(harmonization, HARM, "armonizacion"),
                                levels = HARM))

ratio_df <- pd_df |>
  select(harmonization, band_label, sensor, median) |>
  pivot_wider(names_from = sensor, values_from = median) |>
  mutate(ratio = `Landsat 8/9` / `Sentinel-2`,
         rel = 100 * (ratio - 1),
         txt = sprintf("%.1f×", ratio))

# Sentinel-2 es identico bajo ambas armonizaciones, asi que cabe en un panel:
# cada flecha lleva la mediana de Landsat de Roy a la recalibracion local, que
# cae sobre Sentinel-2. Escala log: la longitud de la flecha es la razon.
# Barras = rango intercuartil.
stopifnot(pd_df |> filter(sensor == "Sentinel-2") |>
            group_by(band_label) |> summarise(k = n_distinct(median)) |>
            pull(k) |> max() == 1)
SERIES <- c(roy = "Landsat 8/9, Roy (2016)",
            loc = "Landsat 8/9, local recalibration",
            s2  = "Sentinel-2")
DX <- 0.13
pd_pts <- pd_df |>
  mutate(series = case_when(sensor == "Sentinel-2" ~ "s2",
                            grepl("^Roy", harmonization) ~ "roy",
                            TRUE ~ "loc"),
         x = as.numeric(band_label) + ifelse(series == "s2", DX, -DX)) |>
  distinct(series, band_label, .keep_all = TRUE)
arrows_df <- pd_pts |> filter(series != "s2") |>
  select(band_label, x, series, median) |>
  pivot_wider(names_from = series, values_from = median)
lab_d <- ratio_df |> select(harmonization, band_label, txt) |>
  mutate(h = ifelse(grepl("^Roy", harmonization), "roy", "loc")) |>
  select(-harmonization) |> pivot_wider(names_from = h, values_from = txt) |>
  left_join(arrows_df |> select(band_label, x, y_roy = roy), by = "band_label") |>
  left_join(pd_pts |> filter(series != "roy") |> group_by(band_label) |>
              summarise(y_lo = min(q25)), by = "band_label")

pd <- ggplot(pd_pts, aes(x, median)) +
  geom_line(data = filter(pd_pts, series == "s2"), colour = PAL_SENSOR[["Sentinel-2"]],
            linewidth = 0.4, alpha = 0.6) +
  geom_segment(data = arrows_df, aes(x = x, xend = x, y = roy * 0.9,
                                     yend = loc * 1.18), inherit.aes = FALSE,
               colour = ACCENT, linewidth = 0.55,
               arrow = arrow(length = unit(3.5, "pt"), type = "closed")) +
  geom_errorbar(aes(ymin = q25, ymax = q75, colour = series), width = 0,
                linewidth = 0.45) +
  geom_point(aes(colour = series, fill = series, shape = series), size = 2.2,
             stroke = 0.7) +
  geom_text(data = lab_d, aes(x = x, y = y_roy * 1.32, label = roy),
            inherit.aes = FALSE, size = 2.4, fontface = "bold", colour = ACCENT) +
  geom_text(data = lab_d, aes(x = x + DX, y = y_lo * 0.72, label = loc),
            inherit.aes = FALSE, size = 2.2, colour = INK_2) +
  scale_colour_manual(values = c(roy = ACCENT, loc = ACCENT,
                                 s2 = PAL_SENSOR[["Sentinel-2"]]),
                      labels = SERIES, breaks = names(SERIES), name = NULL) +
  scale_fill_manual(values = c(roy = "white", loc = ACCENT,
                               s2 = PAL_SENSOR[["Sentinel-2"]]),
                    labels = SERIES, breaks = names(SERIES), name = NULL) +
  scale_shape_manual(values = c(roy = 21, loc = 21, s2 = 21), labels = SERIES,
                     breaks = names(SERIES), name = NULL) +
  scale_x_continuous(breaks = 1:6, labels = BAND_ORDER,
                     expand = expansion(add = 0.4)) +
  scale_y_log10(breaks = c(0.002, 0.005, 0.01, 0.02, 0.05),
                labels = c("0.002", "0.005", "0.01", "0.02", "0.05")) +
  coord_cartesian(ylim = c(0.0011, 0.075)) +
  labs(x = NULL, y = "Median reflectance (log scale)") +
  theme(legend.position = "inside", legend.position.inside = c(0.995, 0.99),
        legend.justification = c(1, 1), legend.key.size = unit(8, "pt"),
        legend.text = element_text(size = 6.8), legend.background = element_blank(),
        axis.ticks.x = element_blank())

# ---------------------------------------------------------------- montaje ----
fig <- ((pa | pb) + plot_layout(widths = c(1, 1.15))) / pc / pd +
  plot_layout(heights = c(1, 0.66, 1)) + tags_abc()

save_fig(fig, "fig01_harmonization_failure", W2, 180)
cat("fig01 lista\n")
