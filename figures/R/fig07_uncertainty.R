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

wilson_ci <- function(p, n, z = 1.96) {
  c0 <- (p + z^2 / (2 * n)) / (1 + z^2 / n)
  h  <- z * sqrt(p * (1 - p) / n + z^2 / (4 * n^2)) / (1 + z^2 / n)
  list(lo = c0 - h, hi = c0 + h)
}
N_IV <- nrow(iv)

# ---------------------------------------------------------------- panel (a) --
# banda gris = variacion muestral esperada de una cobertura perfectamente
# calibrada con n = 812 (binomial al 95%); barras = Wilson de lo observado
band <- tibble(nominal = seq(0.45, 0.97, 0.005)) |>
  mutate(lo = nominal - 1.96 * sqrt(nominal * (1 - nominal) / N_IV),
         hi = nominal + 1.96 * sqrt(nominal * (1 - nominal) / N_IV))
cov2 <- cov |> mutate(lo = wilson_ci(empirical_coverage, N_IV)$lo,
                      hi = wilson_ci(empirical_coverage, N_IV)$hi)

pa <- ggplot(cov2, aes(nominal, empirical_coverage)) +
  geom_ribbon(data = band, aes(nominal, ymin = lo, ymax = hi), inherit.aes = FALSE,
              fill = NEUTRAL, alpha = 0.25) +
  geom_abline(slope = 1, intercept = 0, colour = INK, linetype = "22",
              linewidth = 0.45) +
  geom_line(colour = "#2E6E8E", linewidth = 0.7) +
  geom_errorbar(aes(ymin = lo, ymax = hi), width = 0.012, colour = "#2E6E8E",
                linewidth = 0.45) +
  geom_point(size = 2.4, colour = "#2E6E8E") +
  geom_text(aes(label = sprintf("%.1f%%", 100 * empirical_coverage)),
            hjust = -0.3, vjust = 1.4, size = 2.4, colour = INK) +
  scale_x_continuous(labels = percent_format(accuracy = 1),
                     breaks = c(0.5, 0.6, 0.7, 0.8, 0.9), limits = c(0.44, 1.05)) +
  scale_y_continuous(labels = percent_format(accuracy = 1),
                     breaks = c(0.5, 0.6, 0.7, 0.8, 0.9), limits = c(0.44, 1.01)) +
  coord_equal() +
  labs(x = "Nominal level", y = "Empirical coverage")

# ---------------------------------------------------------------- panel (c) --
# Oruga: cada match-up es una barra (su intervalo del 90%), ordenadas por el
# Secchi medido; el punto es el valor medido. Los que caen fuera (naranja) se
# acumulan en los dos extremos: el fallo condicional de b, visto uno a uno.
n_out <- sum(!iv$covered)
iv_pl <- iv |> arrange(secchi, predicted) |> mutate(rank = row_number())

pb <- ggplot(iv_pl, aes(rank)) +
  geom_linerange(aes(ymin = lower, ymax = upper, colour = covered),
                 linewidth = 0.35, alpha = 0.55) +
  geom_point(aes(y = predicted), size = 0.25, colour = INK_2, alpha = 0.6) +
  geom_point(aes(y = secchi, fill = covered), shape = 21, colour = "white",
             stroke = 0.15, size = 1.3) +
  annotate("text", x = 5, y = max(iv_pl$upper) + 0.3, hjust = 0, vjust = 1,
           size = 2.3, colour = INK_2,
           label = sprintf("%d of %d measurements outside their 90%% interval",
                           n_out, N_IV)) +
  scale_colour_manual(values = c(`TRUE` = "#9DBCCD", `FALSE` = "#E3A98C"),
                      guide = "none") +
  scale_fill_manual(values = c(`TRUE` = "#2E6E8E", `FALSE` = ACCENT),
                    labels = c(`TRUE` = "Measured, inside interval",
                               `FALSE` = "Measured, outside"), name = NULL) +
  scale_x_continuous(expand = expansion(add = 4)) +
  labs(x = "Match-ups ranked by measured Secchi",
       y = "Secchi and 90% interval (m)") +
  guides(fill = guide_legend(override.aes = list(size = 2.2))) +
  theme(legend.position = "inside", legend.position.inside = c(0.99, 0.03),
        legend.justification = c(1, 0), legend.key.size = unit(7, "pt"),
        legend.background = element_blank())

# ---------------------------------------------------------------- panel (b) --
wb2 <- wb |> mutate(bin = factor(bin, levels = wb$bin),
                    lo = wilson_ci(coverage, n)$lo, hi = wilson_ci(coverage, n)$hi)

# El conformal split usa UN cuantil global de residuos, asi que la anchura es
# constante por construccion. Esa es precisamente la razon de que la cobertura
# condicional falle en los extremos: el intervalo no puede adaptarse.
pc <- ggplot(wb2, aes(bin)) +
  geom_hline(yintercept = 90, colour = INK, linetype = "22", linewidth = 0.45) +
  geom_col(aes(y = 100 * coverage, fill = coverage < 0.85), width = 0.62) +
  geom_errorbar(aes(ymin = 100 * lo, ymax = 100 * hi), width = 0.18,
                colour = INK, linewidth = 0.4) +
  geom_text(aes(y = 45, label = sprintf("%.0f%%", 100 * coverage)),
            vjust = 0.5, size = 3, colour = "white", fontface = "bold") +
  geom_text(aes(y = 8, label = sprintf("width\n%.1f m\nn = %d",
                                       mean_width, n)),
            vjust = 0, size = 2.1, colour = "white", lineheight = 1.05) +
  annotate("text", x = 5.5, y = 101, label = "nominal 90%", hjust = 1,
           vjust = 0, size = 2.25, colour = INK_2, fontface = "italic") +
  scale_fill_manual(values = c(`TRUE` = ACCENT, `FALSE` = "#2E6E8E"),
                    guide = "none") +
  scale_y_continuous(limits = c(0, 110), breaks = seq(0, 100, 25),
                     expand = expansion(mult = c(0, 0.02))) +
  labs(x = "Measured Secchi (m)", y = "Empirical coverage (%)")

fig <- (pa | pc) / pb + plot_layout(heights = c(1, 0.9)) + tags_abc()
save_fig(fig, "fig07_conformal_uncertainty", W2, 155)
cat("fig07 lista\n")
