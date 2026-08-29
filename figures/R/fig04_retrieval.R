# ============================================================================
# fig04_retrieval.R -- El retrieval en si, bajo el diseno honesto
#
# (a) Predicho vs observado fuera de fold, por zona trofica, con 1:1
# (b) Residuos por zona: el sesgo positivo en Bahia de Puno, la zona mas
#     eutrofica, es la limitacion operativa clave
# (c) Comparacion con los baselines, incluyendo la demostracion del artefacto
#     que producia el R2 = -2.46 del manuscrito anterior
# ============================================================================

source(file.path(local({a <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(a)) dirname(normalizePath(sub("^--file=", "", a[1]))) else getwd()}),
  "theme_titicaca.R"))

oof   <- read_tidy("oof_predictions.csv")
zone  <- read_tidy("error_by_zone.csv")
bench <- read_tidy("benchmark_models.csv")
art   <- read_tidy("baseline_artefact.csv")

# El pipeline exporta las zonas sin tildes; se acentuan aqui, en la capa de
# presentacion, para no tocar las claves de los datos.
ZLAB <- c("Bahia de Puno" = "Bahía de Puno", "Lago Menor" = "Lago Menor",
          "Lago Mayor" = "Lago Mayor")
ZORD <- unname(ZLAB)
PAL2 <- setNames(unname(PAL_ZONE), ZORD)
oof  <- oof  |> mutate(zone_label = factor(ZLAB[zone_label], levels = ZORD))
zone <- zone |> mutate(zone_label = factor(ZLAB[zone_label], levels = ZORD))

rf <- bench |> filter(family == "rf")
lims <- range(c(oof$secchi, oof$predicted))

# ---------------------------------------------------------------- panel (a) --
pa <- ggplot(oof, aes(secchi, predicted, colour = zone_label)) +
  geom_abline(slope = 1, intercept = 0, colour = INK, linetype = "22",
              linewidth = 0.45) +
  geom_point(size = 1.05, alpha = 0.5) +
  annotate("label", x = lims[1], y = lims[2], hjust = 0, vjust = 1,
           label = metric_label(rf$R2, rf$RMSE, n = rf$n, mae = rf$MAE),
           size = 2.5, colour = INK, fill = "white", label.size = 0,
           label.padding = unit(3, "pt"), lineheight = 1.05) +
  scale_colour_manual(values = PAL2, name = NULL) +
  coord_equal(xlim = lims, ylim = lims) +
  labs(title = "(a)  Predicho frente a medido",
       subtitle = paste("Random Forest, validación bloqueada\npor campaña.",
                        "La discontinua es 1:1"),
       x = "Secchi medido in situ (m)", y = "Secchi predicho (m)") +
  theme(legend.position = "bottom", legend.margin = margin(t = -4))

# ---------------------------------------------------------------- panel (b) --
zlab <- zone |>
  mutate(txt = sprintf("sesgo %+.2f m\nRMSE %.2f m\nn = %d", bias, RMSE, n))

pb <- ggplot(oof, aes(zone_label, residual, fill = zone_label)) +
  geom_hline(yintercept = 0, colour = INK, linewidth = 0.45, linetype = "22") +
  geom_violin(colour = NA, alpha = 0.24, width = 0.95) +
  geom_boxplot(width = 0.19, outlier.size = 0.4, outlier.alpha = 0.35,
               colour = INK_2, linewidth = 0.35, alpha = 0.9) +
  stat_summary(fun = mean, geom = "point", shape = 23, size = 2.1,
               fill = "white", colour = INK, stroke = 0.5) +
  geom_text(data = zlab, aes(x = zone_label, y = 7.4, label = txt),
            inherit.aes = FALSE, size = 2.3, colour = INK_2, vjust = 1,
            lineheight = 1.05) +
  scale_fill_manual(values = PAL2, guide = "none") +
  scale_x_discrete(labels = function(x) sub(" de ", "\nde ", sub("Lago ", "Lago\n", x))) +
  scale_y_continuous(limits = c(-8, 7.8), breaks = seq(-6, 6, 3)) +
  labs(title = "(b)  Sobreestima donde más importa",
       subtitle = paste("Residuos (predicho − medido) por zona.\nEl rombo es la",
                        "media. Bahía de Puno, la\nzona más eutrófica, +1.0 m"),
       x = NULL, y = "Residuo (m)")

# ---------------------------------------------------------------- panel (c) --
pc_df <- bench |>
  mutate(name = case_when(
    family == "artefact"  ~ "Ratio azul/verde\nSIN acotar (como se publicó)",
    family == "classical" ~ "Ratio azul/verde\nacotado (Kloiber 2002)",
    family == "linear"    ~ "Lineal multibanda",
    TRUE                  ~ "Random Forest\n(este estudio)"),
    kind = ifelse(family == "artefact", "artefacto", "correcto"),
    name = fct_reorder(name, R2))

pc <- ggplot(pc_df, aes(R2, name, fill = kind)) +
  geom_vline(xintercept = 0, colour = INK, linewidth = 0.4) +
  geom_col(width = 0.55) +
  geom_text(aes(label = sprintf("%+.3f", R2),
                hjust = ifelse(R2 > 0, -0.22, 1.22)),
            size = 2.7, colour = INK, fontface = "bold") +
  scale_fill_manual(values = c(artefacto = NEUTRAL, correcto = "#2E6E8E"),
                    guide = "none") +
  scale_x_continuous(limits = c(-3.3, 1.25), breaks = seq(-3, 1, 1),
                     expand = expansion(mult = c(0.01, 0))) +
  labs(title = "(c)  El baseline estaba mal implementado",
       subtitle = paste("Mismo diseño de validación para todos.\nEl baseline",
                        "«publicado» deshacía el log sin\nacotar; una única",
                        "predicción de 143 m hundía su R²"),
       x = "R² fuera de fold", y = NULL)

# ---------------------------------------------------------------- panel (d) --
pd <- art |>
  select(observed, predicted_unbounded, predicted_bounded) |>
  pivot_longer(-observed, names_to = "version", values_to = "pred") |>
  mutate(version = recode(version,
                          predicted_unbounded = "Sin acotar",
                          predicted_bounded   = "Acotado")) |>
  ggplot(aes(observed, pred, colour = version)) +
  geom_abline(slope = 1, intercept = 0, colour = INK, linetype = "22",
              linewidth = 0.4) +
  geom_hline(yintercept = max(art$observed), colour = ACCENT,
             linetype = "12", linewidth = 0.4) +
  annotate("text", x = 1.6, y = max(art$observed), vjust = -0.6, hjust = 0,
           label = sprintf("máximo real = %.1f m", max(art$observed)),
           size = 2.25, colour = ACCENT, fontface = "italic") +
  geom_point(size = 0.85, alpha = 0.45) +
  scale_colour_manual(values = c("Sin acotar" = NEUTRAL,
                                 "Acotado" = "#2E6E8E"), name = NULL) +
  scale_y_log10(breaks = c(1, 3, 10, 30, 100), labels = c(1, 3, 10, 30, 100)) +
  labs(title = "(d)  Una predicción imposible basta",
       subtitle = paste("Ratio clásico en escala logarítmica.\nEl punto de 143 m",
                        "es el que producía\nel R² = −2.46"),
       x = "Secchi medido in situ (m)", y = "Secchi predicho (m, log)") +
  theme(legend.position = "bottom", legend.margin = margin(t = -4))

fig <- (pa | pb) / (pc | pd) + plot_layout(heights = c(1.12, 1))
save_fig(fig, "fig04_retrieval_and_benchmark", W2, 185)
cat("fig04 lista\n")
