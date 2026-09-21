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
  scale_fill_manual(values = PAL_VERDICT, labels = VERDICT_LAB,
                    breaks = names(VERDICT_LAB), name = NULL, drop = FALSE) +
  scale_y_log10(breaks = c(1, 3, 10, 30, 100),
                labels = c("1×", "3×", "10×", "30×", "100×"),
                expand = expansion(mult = c(0.02, 0.16))) +
  labs(x = NULL, y = "Intercept / native signal") +
  theme(legend.position = "inside", legend.position.inside = c(0.02, 0.98),
        legend.justification = c(0, 1), legend.key.size = unit(6, "pt"),
        axis.ticks.x = element_blank())

# ---------------------------------------------------------------- panel (b) --
# Si la armonizacion funcionara, un clasificador no deberia superar el chance.
chance <- sep$chance[1]
pb_df <- sep |>
  mutate(label = recode(feature_set,
                        all_12_features = "All 12 features",
                        visible_only    = "Visible only",
                        ratios_only     = "Ratios only",
                        nir_swir_only   = "NIR + SWIR only",
                        green_blue_only = "Green and blue only"),
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
  annotate("text", x = chance + 0.014, y = 0.62,
           label = sprintf("chance = %.1f%%", 100 * chance),
           hjust = 0, size = 2.3, colour = INK_2, fontface = "italic") +
  scale_x_continuous(labels = percent_format(accuracy = 1),
                     limits = c(0.5, 1.12),
                     breaks = seq(0.5, 1, 0.1),
                     expand = expansion(mult = c(0, 0))) +
  labs(x = "Classifier accuracy", y = NULL)

# ---------------------------------------------------------------- panel (c) --
# Acuerdo banda a banda en los 270 eventos vistos por ambos sensores.
lab_df <- coefs |>
  mutate(band_label = ord(band_label),
         txt = sprintf("r = %.2f", pearson_r),
         severity = case_when(pearson_r < 0.35 ~ "roto",
                              pearson_r < 0.55 ~ "debil",
                              pearson_r < 0.70 ~ "aceptable",
                              TRUE ~ "bueno"))

# Con ejes iguales la nube de NIR/SWIR quedaba en una esquina (el desfase es
# de un orden de magnitud y ya lo muestra d). Cada panel se acerca al
# percentil 1-99 de cada sensor; la linea 1:1 solo aparece donde cae dentro.
pc_df <- pair |> mutate(band_label = ord(band_label)) |> group_by(band_label) |>
  mutate(inside = between(ls_roy, quantile(ls_roy, 0.01), quantile(ls_roy, 0.99)) &
                  between(s2, quantile(s2, 0.01), quantile(s2, 0.99))) |>
  ungroup()
out_n <- lab_df |> mutate(band_label = ord(band_label))

pc <- ggplot(filter(pc_df, inside), aes(ls_roy, s2)) +
  geom_abline(slope = 1, intercept = 0, colour = INK, linetype = "22",
              linewidth = 0.35) +
  geom_point(size = 0.5, alpha = 0.35, colour = "#4A4A4A", shape = 16) +
  geom_smooth(method = "lm", formula = y ~ x, se = FALSE,
              colour = ACCENT, linewidth = 0.55, fullrange = FALSE) +
  geom_label(data = out_n, aes(x = Inf, y = -Inf, label = txt, colour = severity),
             hjust = 1.05, vjust = -0.3, size = 2.4, fontface = "bold",
             fill = "white", label.size = 0, label.padding = unit(1, "pt"),
             inherit.aes = FALSE, show.legend = FALSE) +
  facet_wrap(~band_label, nrow = 1, scales = "free") +
  scale_colour_manual(values = PAL_VERDICT) +
  scale_x_continuous(n.breaks = 3, labels = label_number(accuracy = 0.001)) +
  scale_y_continuous(n.breaks = 3, labels = label_number(accuracy = 0.001)) +
  labs(x = "Landsat 8/9 reflectance, harmonized with Roy (2016)",
       y = "Sentinel-2 reflectance") +
  theme(aspect.ratio = 0.9, panel.grid.major.y = element_blank(),
        axis.text = element_text(size = 5.5))

# ---------------------------------------------------------------- panel (d) --
# Firma espectral antes y despues de la recalibracion local sobre agua.
pd_df <- sig |>
  mutate(band_label = ord(band_label),
         harmonization = factor(
           harmonization,
           levels = c("Roy (land-derived)", "Local water recalibration"),
           labels = c("Roy (2016), land-derived",
                      "Local water recalibration")))

ratio_df <- pd_df |>
  select(harmonization, band_label, sensor, median) |>
  pivot_wider(names_from = sensor, values_from = median) |>
  mutate(ratio = `Landsat 8/9` / `Sentinel-2`,
         txt = sprintf("%.1f×", ratio))

pd <- ggplot(pd_df, aes(band_label, median, colour = sensor, group = sensor)) +
  geom_line(linewidth = 0.65) +
  geom_point(size = 1.7) +
  # las razones van en una fila fija encima de las curvas: sobre cada punto
  # se montaban en la linea descendente
  geom_text(data = ratio_df, aes(x = band_label, y = 0.054, label = txt),
            inherit.aes = FALSE, size = 2.3, colour = INK_2) +
  annotate("text", x = 0.62, y = 0.054, label = "LS/S2", hjust = 1, size = 2.1,
           colour = INK_2, fontface = "italic") +
  facet_wrap(~harmonization, nrow = 1) +
  scale_colour_manual(values = PAL_SENSOR, name = NULL) +
  scale_y_continuous(labels = label_number(accuracy = 0.01),
                     breaks = seq(0, 0.04, 0.01), limits = c(0, 0.056), expand = expansion(mult = c(0.02, 0))) +
  coord_cartesian(clip = "off") +
  labs(x = NULL, y = "Median reflectance") +
  theme(legend.position = "bottom",
        legend.margin = margin(t = -4))

# ---------------------------------------------------------------- montaje ----
fig <- (pa | pb) / pc / pd +
  plot_layout(heights = c(1, 0.62, 1)) + tags_abc()

save_fig(fig, "fig01_harmonization_failure", W2, 175)
cat("fig01 lista\n")
