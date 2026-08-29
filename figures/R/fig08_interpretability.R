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

ZLAB <- c("BAHIA PUNO" = "Bahía de Puno", "LAGO MENOR" = "Lago Menor",
          "LAGO MAYOR" = "Lago Mayor")
PAL2 <- setNames(unname(PAL_ZONE), unname(ZLAB))

# ---------------------------------------------------------------- panel (a) --
pa_df <- imp |>
  mutate(label = fct_reorder(label, mean_abs_shap),
         grupo = factor(band_group,
                        levels = c("Verde/azul", "NIR/SWIR", "Otras")))

pa <- ggplot(pa_df, aes(mean_abs_shap, label, fill = grupo)) +
  geom_col(width = 0.62) +
  geom_text(aes(label = sprintf("%.0f%%", 100 * share)), hjust = -0.22,
            size = 2.35, colour = INK_2) +
  scale_fill_manual(values = c("Verde/azul" = "#2E6E8E",
                               "NIR/SWIR"   = ACCENT,
                               "Otras"      = NEUTRAL), name = NULL) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.16))) +
  labs(title = "(a)  El verde y el azul dominan la atribución",
       subtitle = paste("|SHAP| medio calculado FUERA DE FOLD: cada observación",
                        "se explica con el\nmodelo que no la vio. Las etiquetas",
                        "son la cuota sobre el total"),
       x = "|SHAP| medio (m)", y = NULL) +
  theme(legend.position = "bottom", legend.margin = margin(t = -4))

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
  mutate(zl = factor(ZLAB[zona], levels = unname(ZLAB)))

pb <- ggplot(pb_df, aes(value, shap)) +
  geom_hline(yintercept = 0, colour = INK, linewidth = 0.35, linetype = "22") +
  geom_point(aes(colour = zl), size = 0.7, alpha = 0.45) +
  geom_smooth(method = "loess", formula = y ~ x, se = TRUE, span = 0.9,
              linewidth = 0.6, colour = INK, fill = NEUTRAL, alpha = 0.20) +
  scale_colour_manual(values = PAL2, name = NULL) +
  labs(title = sprintf("(b)  Cómo actúa %s sobre la predicción", imp$label[1]),
       subtitle = sprintf(paste("Valor SHAP frente al valor de la feature.",
                                "Más verde respecto al azul empuja\nla",
                                "predicción hacia agua menos transparente.",
                                "Recortado al 1–99 percentil\n(%d puntos",
                                "extremos fuera)"), n_drop),
       x = imp$label[1], y = "Contribución SHAP (m)") +
  theme(legend.position = "bottom", legend.margin = margin(t = -4))

# ---------------------------------------------------------------- panel (c) --
pc_df <- retr |>
  mutate(label_es = recode(variable,
                           secchi      = "Transparencia\n(Secchi)",
                           chl         = "Clorofila-a",
                           tss         = "Sólidos\nsuspendidos",
                           temp_insitu = "Temperatura\ndel agua",
                           lst_thermal = "Temperatura\n(térmico ST_B10)"),
         label_es = fct_reorder(label_es, R2),
         estado = ifelse(retrievable, "Recuperable", "NO recuperable"))

pc <- ggplot(pc_df, aes(R2, label_es, fill = estado)) +
  geom_vline(xintercept = 0.30, colour = INK, linetype = "22",
             linewidth = 0.45) +
  geom_col(width = 0.58) +
  geom_text(aes(label = sprintf("R² %+.3f   (n = %d)", R2, n),
                hjust = ifelse(R2 > 0.05, -0.06, -0.30)),
            size = 2.35, colour = INK_2) +
  annotate("text", x = 0.305, y = 0.6, label = "umbral de utilidad", hjust = 0,
           size = 2.2, colour = INK_2, fontface = "italic") +
  scale_fill_manual(values = c("Recuperable" = "#3D7A57",
                               "NO recuperable" = NEUTRAL), name = NULL) +
  scale_x_continuous(limits = c(-0.05, 0.92),
                     expand = expansion(mult = c(0.01, 0))) +
  labs(title = "(c)  Sólo la transparencia es recuperable en esta agua",
       subtitle = paste("Random Forest con el mismo diseño honesto (bloqueo por",
                        "campaña) para cada variable.\nLa afirmación negativa se",
                        "somete al mismo rigor que la positiva"),
       x = "R² fuera de fold", y = NULL) +
  theme(legend.position = "bottom", legend.margin = margin(t = -4))

fig <- (pa | pb) / pc + plot_layout(heights = c(1.1, 1))
save_fig(fig, "fig08_shap_and_retrievability", W2, 168)
cat("fig08 lista\n")
