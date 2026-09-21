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

# ---------------------------------------------------------------- panel (b) --
# Cobertura condicional en forma de forest plot: por Secchi medido (los mismos
# cortes que p07), por zona y por sensor, con Wilson al 95%. Rojo si el
# intervalo de Wilson no contiene el 90% nominal.
BREAKS <- c(0, 5, 7.5, 10, 12.5, 20)
BIN_LAB <- c("< 5 m", "5–7.5 m", "7.5–10 m", "10–12.5 m", "> 12.5 m")
iv <- iv |> mutate(bin = cut(secchi, BREAKS, labels = BIN_LAB, right = TRUE))
stopifnot(!any(is.na(iv$bin)),
          all(as.integer(table(iv$bin)) == wb$n))
SENS <- c(LS = "Landsat 8/9", S2 = "Sentinel-2")
cond <- bind_rows(
  iv |> mutate(grp = " ", lvl = "All match-ups"),
  iv |> mutate(grp = "Measured Secchi", lvl = as.character(bin)),
  iv |> mutate(grp = "Trophic zone", lvl = as.character(zl)),
  iv |> mutate(grp = "Sensor", lvl = map_strict(sensor, SENS, "sensor"))) |>
  group_by(grp, lvl) |>
  summarise(n = n(), cov = mean(covered), w = mean(width), .groups = "drop") |>
  mutate(lo = wilson_ci(cov, n)$lo, hi = wilson_ci(cov, n)$hi,
         off = hi < 0.90 | lo > 0.90,
         grp = factor(grp, levels = c(" ", "Measured Secchi", "Trophic zone",
                                      "Sensor")))
ord <- c("All match-ups", rev(BIN_LAB), rev(levels(iv$zl)), rev(unname(SENS)))
cond <- cond |> mutate(lvl = factor(lvl, levels = rev(ord)))
stopifnot(nrow(cond) == 11)

pc <- ggplot(cond, aes(cov, lvl, colour = off)) +
  geom_vline(xintercept = 0.90, colour = INK, linetype = "22", linewidth = 0.4) +
  geom_errorbar(aes(xmin = lo, xmax = hi), width = 0.3, linewidth = 0.55,
                orientation = "y") +
  geom_point(size = 2.3) +
  geom_text(aes(x = 1.005, label = sprintf("%.0f%%", 100 * cov)), hjust = 0,
            size = 2.3, fontface = "bold") +
  geom_text(aes(x = 1.095, label = sprintf("%d", n)), hjust = 1, size = 2.1,
            colour = INK_2) +
  geom_text(aes(x = 1.155, label = sprintf("%.1f m", w)), hjust = 1, size = 2.1,
            colour = INK_2) +
  geom_text(data = tibble(grp = factor(" ", levels = levels(cond$grp)),
                          lvl = factor("All match-ups", levels = levels(cond$lvl)),
                          x = c(1.095, 1.155), lab = c("n", "width")),
            aes(x = x, y = Inf, label = lab), inherit.aes = FALSE, hjust = 1,
            vjust = -0.7, size = 2.1, colour = INK_2, fontface = "italic") +
  facet_grid(grp ~ ., scales = "free_y", space = "free_y", switch = "y") +
  scale_colour_manual(values = c(`TRUE` = ACCENT, `FALSE` = "#2E6E8E"),
                      guide = "none") +
  scale_x_continuous(limits = c(0.58, 1.16), breaks = seq(0.6, 1, 0.1),
                     labels = percent_format(accuracy = 1),
                     expand = expansion(mult = c(0, 0))) +
  coord_cartesian(clip = "off") +
  labs(x = "Coverage of the 90% interval (95% Wilson)", y = NULL) +
  theme(strip.placement = "outside",
        strip.text.y.left = element_text(angle = 0, face = "bold", hjust = 1,
                                         size = 6.5),
        panel.grid.major.y = element_blank(), axis.line.y = element_blank(),
        axis.ticks.y = element_blank(), panel.spacing.y = unit(5, "pt"),
        plot.margin = margin(12, 4, 2, 2))

# ---------------------------------------------------------------- panel (c) --
# Oruga: cada match-up es una barra (su intervalo del 90%), ordenadas por el
# Secchi medido; el punto es el valor medido. Franjas = los cortes de (b), con
# su cobertura arriba.
n_out <- sum(!iv$covered)
iv_pl <- iv |> arrange(secchi, predicted) |> mutate(rank = row_number())
bands <- iv_pl |> group_by(bin) |>
  summarise(x0 = min(rank) - 0.5, x1 = max(rank) + 0.5, cov = mean(covered),
            .groups = "drop") |>
  mutate(shade = row_number() %% 2 == 0, xm = (x0 + x1) / 2)
y_top <- max(iv_pl$upper) + 1.2

pb <- ggplot(iv_pl, aes(rank)) +
  geom_rect(data = filter(bands, shade), aes(xmin = x0, xmax = x1, ymin = -Inf,
                                             ymax = Inf),
            inherit.aes = FALSE, fill = "grey94") +
  geom_linerange(aes(ymin = lower, ymax = upper, colour = covered),
                 linewidth = 0.35, alpha = 0.55) +
  geom_point(aes(y = predicted), size = 0.25, colour = INK_2, alpha = 0.6) +
  geom_point(aes(y = secchi, fill = covered), shape = 21, colour = "white",
             stroke = 0.15, size = 1.3) +
  # color fijo, fuera de la escala: compartirla con las barras invertia colores
  geom_text(data = bands, aes(x = xm, y = y_top, label = sprintf("%s\n%.0f%%",
                                                                 bin, 100 * cov)),
            inherit.aes = FALSE, size = 2.2, lineheight = 0.95, fontface = "bold",
            colour = ifelse(bands$cov < 0.85, ACCENT, "#2E6E8E")) +
  scale_colour_manual(values = c(`TRUE` = "#9DBCCD", `FALSE` = "#E3A98C"),
                      guide = "none") +
  scale_fill_manual(values = c(`TRUE` = "#2E6E8E", `FALSE` = ACCENT),
                    labels = c(`TRUE` = "Measured, inside interval",
                               `FALSE` = "Measured, outside"), name = NULL) +
  scale_x_continuous(expand = expansion(add = 4)) +
  scale_y_continuous(limits = c(NA, y_top + 1.2), breaks = seq(0, 15, 5)) +
  labs(x = "Match-ups ranked by measured Secchi",
       y = "Secchi and 90% interval (m)") +
  guides(fill = guide_legend(override.aes = list(size = 2.2))) +
  theme(legend.position = "inside", legend.position.inside = c(0.99, 0.03),
        legend.justification = c(1, 0), legend.key.size = unit(7, "pt"),
        legend.background = element_blank())

design <- "
AB
CC
"
fig <- wrap_plots(A = pa, B = pc, C = pb, design = design) +
  plot_layout(widths = c(1, 1.25), heights = c(1, 0.85)) + tags_abc()
save_fig(fig, "fig07_conformal_uncertainty", W2, 165)
cat("fig07 lista\n")
