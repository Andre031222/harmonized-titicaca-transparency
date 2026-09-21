# ============================================================================
# fig07_uncertainty.R -- Incertidumbre conformal por prediccion
#
# Esta es la parte del analisis original que SI aguanto la auditoria. La
# figura tiene que dejar claras dos cosas: que los intervalos estan
# calibrados en todo el rango nominal, y que su anchura real (~6 m al 90%)
# marca lo que el producto puede y no puede decidir.
#
# (a) cobertura empirica frente a nominal
# (b) predicciones con banda al 90%, ordenadas por transparencia medida
# (c) anchura del intervalo segun el nivel de transparencia
# ============================================================================

source(file.path(local({a <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(a)) dirname(normalizePath(sub("^--file=", "", a[1]))) else getwd()}),
  "theme_titicaca.R"))

cov <- read_tidy("conformal_coverage.csv")
iv  <- read_tidy("conformal_intervals.csv")
wb  <- read_tidy("interval_width_by_secchi.csv")

iv <- iv |> mutate(zl = zone_factor(zone_label))

# ---------------------------------------------------------------- panel (a) --
pa <- ggplot(cov, aes(nominal, empirical_coverage)) +
  geom_abline(slope = 1, intercept = 0, colour = INK, linetype = "22",
              linewidth = 0.45) +
  geom_ribbon(aes(ymin = nominal - 0.03, ymax = nominal + 0.03),
              fill = NEUTRAL, alpha = 0.15) +
  geom_line(colour = "#2E6E8E", linewidth = 0.7) +
  geom_point(size = 2.6, colour = "#2E6E8E") +
  geom_text(aes(label = sprintf("%.1f%%", 100 * empirical_coverage)),
            hjust = -0.22, vjust = 1.3, size = 2.4, colour = INK) +
  # hasta 1.13, no 1.01: la etiqueta "94.3%" va a la derecha de su punto y
  # quedaba cortada por el borde del panel
  scale_x_continuous(labels = percent_format(accuracy = 1),
                     breaks = c(0.6, 0.8, 1.0), limits = c(0.44, 1.13)) +
  scale_y_continuous(labels = percent_format(accuracy = 1),
                     limits = c(0.44, 1.01)) +
  coord_equal() +
  labs(x = "Nominal level", y = "Empirical coverage")

# ---------------------------------------------------------------- panel (b) --
# Medido frente a predicho, una barra por intervalo. Los puntos que quedan
# fuera (naranja) se acumulan en los extremos del rango: el fallo condicional.
set.seed(1)
iv_pl <- iv |> mutate(xj = secchi + runif(n(), -0.18, 0.18)) |> arrange(covered)
n_out <- sum(!iv$covered)
lim7  <- c(0, max(c(iv$upper, iv$secchi)) + 0.5)

pb <- ggplot(iv_pl, aes(xj, predicted, colour = covered)) +
  geom_abline(slope = 1, intercept = 0, colour = INK, linetype = "22",
              linewidth = 0.3) +
  geom_linerange(aes(ymin = lower, ymax = upper), linewidth = 0.18, alpha = 0.35) +
  geom_point(size = 0.6, shape = 16, alpha = 0.8) +
  annotate("text", x = 0.3, y = lim7[2], hjust = 0, vjust = 1, size = 2.3,
           colour = INK_2, label = sprintf("%d of %d outside the 90%% interval",
                                           n_out, nrow(iv))) +
  scale_colour_manual(values = c(`TRUE` = "#6F97AD", `FALSE` = ACCENT),
                      labels = c(`TRUE` = "Inside interval", `FALSE` = "Outside"),
                      name = NULL) +
  coord_cartesian(xlim = c(0, 17.2), ylim = lim7, expand = FALSE) +
  labs(x = "Measured Secchi (m)", y = "Predicted Secchi with 90% interval (m)") +
  theme(legend.position = "inside", legend.position.inside = c(1, 0.02),
        legend.justification = c(1, 0), legend.key.size = unit(6, "pt"))

# ---------------------------------------------------------------- panel (c) --
wb2 <- wb |> mutate(bin = factor(bin, levels = wb$bin))

# El conformal split usa UN cuantil global de residuos, asi que la anchura es
# constante por construccion. Esa es precisamente la razon de que la cobertura
# condicional falle en los extremos: el intervalo no puede adaptarse.
pc <- ggplot(wb2, aes(bin)) +
  geom_hline(yintercept = 90, colour = INK, linetype = "22", linewidth = 0.45) +
  geom_col(aes(y = 100 * coverage, fill = coverage < 0.85), width = 0.62) +
  geom_text(aes(y = 100 * coverage, label = sprintf("%.0f%%", 100 * coverage)),
            vjust = -0.5, size = 2.6, colour = INK, fontface = "bold") +
  geom_text(aes(y = 8, label = sprintf("width\n%.1f m\nn = %d",
                                       mean_width, n)),
            vjust = 0, size = 2.1, colour = "white", lineheight = 1.05) +
  annotate("text", x = 0.6, y = 90, label = "nominal 90%", hjust = 0,
           vjust = -0.5, size = 2.25, colour = INK_2, fontface = "italic") +
  scale_fill_manual(values = c(`TRUE` = ACCENT, `FALSE` = "#2E6E8E"),
                    guide = "none") +
  scale_y_continuous(limits = c(0, 104),
                     expand = expansion(mult = c(0, 0.02))) +
  labs(x = "Measured Secchi (m)", y = "Empirical coverage (%)")

fig <- (pa | pc) / pb + plot_layout(heights = c(1, 1.05)) + tags_abc()
save_fig(fig, "fig07_conformal_uncertainty", W2, 165)
cat("fig07 lista\n")
