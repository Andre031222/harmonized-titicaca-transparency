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

pa <- ggplot(pa_df, aes(band_label, intercept_over_signal)) +
  geom_rect(data = CLASS_BREAKS, aes(xmin = -Inf, xmax = Inf, ymin = lo,
                                     ymax = hi, fill = severity),
            inherit.aes = FALSE, alpha = 0.16) +
  geom_hline(yintercept = 1, linewidth = 0.45, colour = INK, linetype = "22") +
  geom_segment(aes(xend = band_label, y = 1, yend = intercept_over_signal,
                   colour = severity), linewidth = 0.8) +
  geom_point(aes(colour = severity), size = 3) +
  geom_text(aes(label = sprintf("%.1f×", intercept_over_signal)),
            vjust = -1.1, size = 2.5, colour = INK, fontface = "bold") +
  scale_fill_manual(values = PAL_VERDICT, labels = VERDICT_LAB,
                    breaks = names(VERDICT_LAB), name = NULL) +
  scale_colour_manual(values = PAL_VERDICT, guide = "none") +
  scale_y_log10(breaks = c(1, 2, 5, 20, 100),
                labels = c("1×", "2×", "5×", "20×", "100×"),
                expand = expansion(mult = c(0, 0))) +
  coord_cartesian(ylim = c(0.85, 130)) +
  guides(fill = guide_legend(override.aes = list(alpha = 0.6))) +
  labs(x = NULL, y = "Intercept / native signal") +
  theme(legend.position = "inside", legend.position.inside = c(0.02, 0.99),
        legend.justification = c(0, 1), legend.key.size = unit(6, "pt"),
        legend.background = element_blank(), axis.ticks.x = element_blank(),
        panel.grid.major.y = element_blank())

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

pb <- ggplot(pb_df, aes(accuracy, label)) +
  geom_vline(xintercept = chance, colour = INK, linetype = "22",
             linewidth = 0.45) +
  geom_errorbar(aes(xmin = lo, xmax = hi), width = 0.28, colour = ACCENT,
                linewidth = 0.5, orientation = "y") +
  geom_point(size = 2.4, colour = ACCENT) +
  geom_text(aes(x = hi, label = sprintf("%.1f%%", 100 * accuracy)),
            hjust = -0.3, size = 2.5, colour = INK, fontface = "bold") +
  annotate("text", x = chance + 0.008, y = 0.62,
           label = sprintf("chance = %.1f%%", 100 * chance),
           hjust = 0, size = 2.3, colour = INK_2, fontface = "italic") +
  scale_x_continuous(labels = percent_format(accuracy = 1),
                     limits = c(0.6, 1.1), breaks = seq(0.6, 1, 0.1),
                     expand = expansion(mult = c(0, 0))) +
  labs(x = "Classifier accuracy (95% Wilson interval)", y = NULL)

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

# Mancuernas en escala log: la distancia vertical entre los dos sensores es
# directamente la razon Landsat/Sentinel-2, y NIR/SWIR (reflectancias de 0.002)
# dejan de quedar aplastadas contra cero. Barras = rango intercuartil.
dumb <- ratio_df |> rename(ls = `Landsat 8/9`, s2 = `Sentinel-2`)
pd <- ggplot(pd_df, aes(band_label, median, colour = sensor)) +
  geom_blank() +
  # une los dos puntos esquivados (Landsat a la izquierda, Sentinel-2 a la derecha)
  geom_segment(data = dumb, aes(x = as.numeric(band_label) - 0.08,
                                xend = as.numeric(band_label) + 0.08,
                                y = ls, yend = s2), inherit.aes = FALSE,
               colour = "grey55", linewidth = 0.5) +
  geom_errorbar(aes(ymin = q25, ymax = q75), width = 0, linewidth = 0.45,
                position = position_dodge(width = 0.32)) +
  geom_point(size = 2.1, position = position_dodge(width = 0.32)) +
  geom_text(data = dumb, aes(x = band_label, y = pmax(ls, s2) * 1.55,
                             label = txt), inherit.aes = FALSE, size = 2.35,
            fontface = "bold", colour = INK) +
  facet_wrap(~harmonization, nrow = 1) +
  scale_colour_manual(values = PAL_SENSOR, name = NULL) +
  scale_y_log10(breaks = c(0.002, 0.005, 0.01, 0.02, 0.05),
                labels = c("0.002", "0.005", "0.01", "0.02", "0.05")) +
  coord_cartesian(ylim = c(0.0012, 0.08)) +
  labs(x = NULL, y = "Median reflectance (log scale)") +
  theme(legend.position = "inside", legend.position.inside = c(0.99, 0.99),
        legend.justification = c(1, 1), legend.key.size = unit(8, "pt"),
        legend.background = element_blank(),
        strip.text = element_text(face = "bold", hjust = 0),
        panel.spacing = unit(10, "pt"))

# ---------------------------------------------------------------- montaje ----
fig <- (pa | pb) / pc / pd +
  plot_layout(heights = c(1, 0.66, 1)) + tags_abc()

save_fig(fig, "fig01_harmonization_failure", W2, 180)
cat("fig01 lista\n")
