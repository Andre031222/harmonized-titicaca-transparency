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
ord <- function(x) factor(x, levels = BAND_ORDER)

# ---------------------------------------------------------------- panel (a) --
# El intercepto de Roy comparado con la senal nativa de Landsat sobre el agua.
# Escala log porque el rango va de 1.4x a 72x.
pa_df <- bands |>
  mutate(band_label = ord(band_label),
         severity = case_when(intercept_over_signal >= 20 ~ "roto",
                              intercept_over_signal >= 5  ~ "debil",
                              intercept_over_signal >= 2  ~ "aceptable",
                              TRUE ~ "bueno"))

pa <- ggplot(pa_df, aes(band_label, intercept_over_signal, fill = severity)) +
  geom_hline(yintercept = 1, linewidth = 0.45, colour = INK, linetype = "22") +
  geom_col(width = 0.62) +
  geom_text(aes(label = sprintf("%.1f×", intercept_over_signal)),
            vjust = -0.45, size = 2.55, colour = INK, fontface = "bold") +
  annotate("text", x = 0.62, y = 1, label = "intercepto = señal", hjust = 0,
           vjust = -0.55, size = 2.3, colour = INK_2, fontface = "italic") +
  scale_fill_manual(values = PAL_VERDICT, guide = "none") +
  scale_y_log10(breaks = c(1, 3, 10, 30, 100),
                labels = c("1×", "3×", "10×", "30×", "100×"),
                expand = expansion(mult = c(0.02, 0.16))) +
  labs(title = "(a)  El intercepto terrestre supera a la señal",
       subtitle = paste("Coeficientes de Roy et al. (2016), ajustados sobre",
                        "vegetación\ny suelo, frente a la reflectancia nativa",
                        "de Landsat\nsobre el agua del lago"),
       x = NULL, y = "Intercepto / señal nativa")

# ---------------------------------------------------------------- panel (b) --
# Si la armonizacion funcionara, un clasificador no deberia superar el azar.
chance <- sep$chance[1]
pb_df <- sep |>
  mutate(label = recode(feature_set,
                        all_12_features = "Las 12 features",
                        visible_only    = "Solo visible",
                        ratios_only     = "Solo ratios",
                        nir_swir_only   = "Solo NIR + SWIR",
                        green_blue_only = "Solo verde y azul"),
         label = fct_reorder(label, accuracy))

pb <- ggplot(pb_df, aes(accuracy, label)) +
  annotate("rect", xmin = 0, xmax = chance, ymin = -Inf, ymax = Inf,
           fill = NEUTRAL, alpha = 0.13) +
  geom_vline(xintercept = chance, colour = INK, linetype = "22",
             linewidth = 0.45) +
  geom_segment(aes(x = chance, xend = accuracy, yend = label),
               colour = ACCENT, linewidth = 0.9) +
  geom_point(size = 2.4, colour = ACCENT) +
  geom_text(aes(label = sprintf("%.1f%%", 100 * accuracy)),
            hjust = -0.30, size = 2.5, colour = INK, fontface = "bold") +
  annotate("text", x = chance - 0.012, y = 0.62,
           label = sprintf("azar = %.1f%%", 100 * chance),
           hjust = 1, size = 2.3, colour = INK_2, fontface = "italic") +
  scale_x_continuous(labels = percent_format(accuracy = 1),
                     limits = c(0.5, 1.12),
                     breaks = seq(0.5, 1, 0.1),
                     expand = expansion(mult = c(0, 0))) +
  labs(title = "(b)  Los sensores siguen distinguiéndose",
       subtitle = paste("Exactitud de un Random Forest que predice el sensor",
                        "de\norigen a partir de la reflectancia ya armonizada.",
                        "Si la\narmonización funcionara, estaría en el azar"),
       x = "Exactitud del clasificador", y = NULL)

# ---------------------------------------------------------------- panel (c) --
# Acuerdo banda a banda en los 270 eventos vistos por ambos sensores.
lab_df <- coefs |>
  mutate(band_label = ord(band_label),
         txt = sprintf("r = %.2f", pearson_r),
         severity = case_when(pearson_r < 0.35 ~ "roto",
                              pearson_r < 0.55 ~ "debil",
                              pearson_r < 0.70 ~ "aceptable",
                              TRUE ~ "bueno"))

pc <- pair |>
  mutate(band_label = ord(band_label)) |>
  ggplot(aes(ls_roy, s2)) +
  geom_abline(slope = 1, intercept = 0, colour = INK, linetype = "22",
              linewidth = 0.4) +
  geom_point(aes(colour = band_label), size = 0.55, alpha = 0.30,
             show.legend = FALSE) +
  geom_smooth(method = "lm", formula = y ~ x, se = FALSE,
              colour = ACCENT, linewidth = 0.6) +
  geom_text(data = lab_df, aes(x = -Inf, y = Inf, label = txt, colour = severity),
            hjust = -0.18, vjust = 1.45, size = 2.5, fontface = "bold",
            inherit.aes = FALSE, show.legend = FALSE) +
  facet_wrap(~band_label, nrow = 2, scales = "free") +
  scale_colour_manual(values = c(PAL_VERDICT,
                                 setNames(rep(INK_2, 6), BAND_ORDER))) +
  scale_x_continuous(n.breaks = 3, labels = label_number(accuracy = 0.01)) +
  scale_y_continuous(n.breaks = 3, labels = label_number(accuracy = 0.01)) +
  labs(title = "(c)  El azul es la banda que peor concuerda entre sensores",
       subtitle = paste0("Reflectancia de Sentinel-2 frente a Landsat armonizado",
                         " en los ", coefs$n_paired[1], " eventos observados\n",
                         "por ambos sensores en la misma estación y fecha.",
                         " La línea discontinua es 1:1"),
       x = "Landsat 8/9 armonizado con Roy (2016)", y = "Sentinel-2") +
  theme(panel.grid.minor = element_blank())

# ---------------------------------------------------------------- panel (d) --
# Firma espectral antes y despues de la recalibracion local sobre agua.
pd_df <- sig |>
  mutate(band_label = ord(band_label),
         harmonization = factor(
           harmonization,
           levels = c("Roy (land-derived)", "Local water recalibration"),
           labels = c("Roy (2016), derivado sobre tierra",
                      "Recalibración local sobre agua")))

ratio_df <- pd_df |>
  select(harmonization, band_label, sensor, median) |>
  pivot_wider(names_from = sensor, values_from = median) |>
  mutate(ratio = `Landsat 8/9` / `Sentinel-2`,
         txt = sprintf("%.1f×", ratio))

pd <- ggplot(pd_df, aes(band_label, median, colour = sensor, group = sensor)) +
  geom_line(linewidth = 0.65) +
  geom_point(size = 1.7) +
  geom_text(data = ratio_df, aes(x = band_label, y = pmax(`Landsat 8/9`,
                                                          `Sentinel-2`),
                                 label = txt),
            inherit.aes = FALSE, vjust = -0.85, size = 2.35, colour = INK_2) +
  facet_wrap(~harmonization, nrow = 1) +
  scale_colour_manual(values = PAL_SENSOR, name = NULL) +
  scale_y_continuous(labels = label_number(accuracy = 0.01),
                     expand = expansion(mult = c(0.06, 0.20))) +
  labs(title = "(d)  La recalibración local sobre agua sí armoniza las bandas",
       subtitle = paste("Reflectancia mediana sobre el agua por sensor.",
                        "Las etiquetas son el cociente Landsat / Sentinel-2:\n",
                        "pasa de 15.4× en el NIR a ~1.0× en todas las bandas"),
       x = NULL, y = "Reflectancia mediana") +
  theme(legend.position = "bottom",
        legend.margin = margin(t = -4))

# ---------------------------------------------------------------- montaje ----
fig <- (pa | pb) / pc / pd +
  plot_layout(heights = c(1, 1.18, 1.02))

save_fig(fig, "fig01_harmonization_failure", W2, 235)
cat("fig01 lista\n")
