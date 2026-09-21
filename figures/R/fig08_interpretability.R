# ============================================================================
# fig08_interpretability.R -- SHAP y sobre operativo
#
# (a) Importancia SHAP fuera de fold. El mensaje incomodo que el articulo
#     tiene que afrontar: los ratios verde/azul dominan la atribucion, y el
#     azul es la banda con peor acuerdo entre sensores (fig01c).
# (b) Dependencia SHAP de la feature dominante, por zona trofica
# (c) Que se puede y que NO se puede recuperar en esta agua
# ============================================================================

source(file.path(local({a <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(a)) dirname(normalizePath(sub("^--file=", "", a[1]))) else getwd()}),
  "theme_titicaca.R"))

imp  <- read_tidy("shap_importance.csv")
dep  <- read_tidy("shap_dependence.csv")
retr <- read_tidy("retrievability.csv")

# ---------------------------------------------------------------- panel (a) --
BAND_GROUP <- c("Verde/azul" = "Green/blue", "NIR/SWIR" = "NIR/SWIR",
                "Otras" = "Other")
band_group_factor <- function(x) {
  out <- unname(BAND_GROUP[as.character(x)])
  bad <- unique(as.character(x)[is.na(out)])
  if (length(bad)) stop("band_group no reconocido: ", paste(bad, collapse = ", "))
  factor(out, levels = unname(BAND_GROUP))
}

pa_df <- imp |>
  mutate(label = fct_reorder(label, mean_abs_shap),
         grupo = band_group_factor(band_group))

PAL_GROUP <- c("Green/blue" = "#2E6E8E", "NIR/SWIR" = ACCENT, "Other" = NEUTRAL)
pa <- ggplot(pa_df, aes(mean_abs_shap, label, colour = grupo)) +
  geom_segment(aes(x = 0, xend = mean_abs_shap, yend = label), linewidth = 1) +
  geom_point(size = 2.4) +
  geom_text(aes(label = sprintf("%.0f%%", 100 * share)), hjust = -0.45,
            size = 2.35, colour = INK_2) +
  scale_colour_manual(values = PAL_GROUP, name = NULL) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.14))) +
  labs(x = "Mean |SHAP| (m)", y = NULL) +
  theme(legend.position = "inside", legend.position.inside = c(0.98, 0.04),
        legend.justification = c(1, 0), legend.key.size = unit(7, "pt"),
        legend.background = element_blank(), panel.grid.major.y = element_blank(),
        axis.line.y = element_blank(), axis.ticks.y = element_blank())

# ---------------------------------------------------------------- panel (b) --
top_feat <- imp$feature[1]
pb_all <- dep |> filter(feature == top_feat)
# Recortar al 1-99 percentil: unos pocos valores extremos de la ratio hacen
# que cualquier suavizado extrapole fuera del rango con datos y sugiera una
# forma que no esta soportada por la evidencia.
qlim <- quantile(pb_all$value, c(0.01, 0.99), na.rm = TRUE)
n_drop <- sum(pb_all$value < qlim[1] | pb_all$value > qlim[2])
pb_df <- pb_all |>
  filter(value >= qlim[1], value <= qlim[2]) |>
  mutate(zl = zone_factor(zona))

# lectura fisica del eje: mas azul = agua mas clara, mas verde = mas turbia
y_top <- max(pb_df$shap) + 0.55
x_mid <- median(pb_df$value)
pb <- ggplot(pb_df, aes(value, shap)) +
  geom_hline(yintercept = 0, colour = INK, linewidth = 0.35, linetype = "22") +
  geom_point(aes(fill = zl), shape = 21, colour = "white", stroke = 0.15,
             size = 1.2, alpha = 0.8) +
  geom_smooth(method = "loess", formula = y ~ x, se = TRUE, span = 0.9,
              linewidth = 0.7, colour = INK, fill = NEUTRAL, alpha = 0.25) +
  annotate("segment", x = x_mid - 0.03, xend = min(pb_df$value), y = y_top,
           yend = y_top, colour = "#2E6E8E", linewidth = 0.5,
           arrow = arrow(length = unit(4, "pt"), type = "closed")) +
  annotate("segment", x = x_mid + 0.03, xend = max(pb_df$value), y = y_top,
           yend = y_top, colour = "#6E8B3D", linewidth = 0.5,
           arrow = arrow(length = unit(4, "pt"), type = "closed")) +
  annotate("text", x = min(pb_df$value), y = y_top + 0.28, hjust = 0,
           label = "bluer, clearer water", size = 2.3, colour = "#2E6E8E",
           fontface = "italic") +
  annotate("text", x = max(pb_df$value), y = y_top + 0.28, hjust = 1,
           label = "greener, more turbid", size = 2.3, colour = "#6E8B3D",
           fontface = "italic") +
  scale_fill_manual(values = PAL_ZONE, name = NULL) +
  guides(fill = guide_legend(override.aes = list(size = 2.2, alpha = 1))) +
  labs(x = imp$label[1], y = "SHAP contribution (m)") +
  theme(legend.position = "inside", legend.position.inside = c(0.98, 0.72),
        legend.justification = c(1, 1), legend.key.size = unit(7, "pt"),
        legend.background = element_blank())

# ---------------------------------------------------------------- panel (c) --
pc_df <- retr |>
  mutate(label_es = recode(variable,
                           secchi      = "Transparency
(Secchi)",
                           chl         = "Chlorophyll-a",
                           tss         = "Suspended
solids",
                           temp_insitu = "Water
temperature",
                           lst_thermal = "Temperature
(thermal ST_B10)"),
         label_es = fct_reorder(label_es, R2),
         estado = ifelse(retrievable, "Retrievable", "Not retrievable"))

pc <- ggplot(pc_df, aes(R2, label_es, colour = estado)) +
  annotate("rect", xmin = 0.30, xmax = Inf, ymin = -Inf, ymax = Inf,
           fill = "#3D7A57", alpha = 0.07) +
  geom_vline(xintercept = 0.30, colour = INK, linetype = "22",
             linewidth = 0.45) +
  geom_segment(aes(x = 0, xend = R2, yend = label_es), linewidth = 1.1) +
  geom_point(size = 2.8) +
  geom_text(aes(label = sprintf("%.3f  (n = %d)", R2, n)), hjust = -0.18,
            size = 2.35, colour = INK_2) +
  annotate("text", x = 0.31, y = 0.55, label = "utility threshold, R² = 0.30",
           hjust = 0, size = 2.2, colour = INK_2, fontface = "italic") +
  scale_colour_manual(values = c("Retrievable" = "#3D7A57",
                                 "Not retrievable" = NEUTRAL), name = NULL) +
  scale_x_continuous(limits = c(0, 0.92), expand = expansion(mult = c(0, 0))) +
  labs(x = expression("Out-of-fold "*italic(R)^2), y = NULL) +
  theme(legend.position = "inside", legend.position.inside = c(0.99, 0.35),
        legend.justification = c(1, 0.5), legend.key.size = unit(7, "pt"),
        legend.background = element_blank(), panel.grid.major.y = element_blank(),
        axis.line.y = element_blank(), axis.ticks.y = element_blank())

fig <- (pa | pb) / pc + plot_layout(heights = c(1.2, 0.72)) + tags_abc()
save_fig(fig, "fig08_shap_and_retrievability", W2, 128)
cat("fig08 lista\n")
