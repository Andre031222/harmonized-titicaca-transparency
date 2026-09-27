# ============================================================================
# fig07_uncertainty.R -- Intervalos conformales (p07)
#
# (a) Calibracion marginal de los tres metodos a 50, 80, 90 y 95 %
# (b) Cobertura del intervalo al 90 % por tramos del Secchi MEDIDO y del
#     PREDICHO: los tres fallan en los extremos de lo medido y cumplen en lo
#     predicho, que es lo que ve el usuario
# (c) Cada intervalo al 90 % (Mondrian), ordenado por el Secchi medido
# ============================================================================

source(file.path(local({a <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(a)) dirname(normalizePath(sub("^--file=", "", a[1]))) else getwd()}),
  "theme_titicaca.R"))

cov  <- read_tidy("conformal_coverage.csv")
iv   <- read_tidy("conformal_intervals.csv")
cond <- read_tidy("interval_width_by_secchi.csv")

METHOD <- c(split = "Split (constant width)", mondrian = "Mondrian (by prediction)",
            cqr = "Conformalized quantile regression")
PAL_M <- c(split = "#8A8A8A", mondrian = "#2E6E8E", cqr = ACCENT)
BIN_LAB <- c("<5", "5–7.5", "7.5–10", "10–12.5", ">12.5")
wilson <- function(p, n, z = 1.96) {
  c0 <- (p + z^2 / (2 * n)) / (1 + z^2 / n)
  h  <- z * sqrt(p * (1 - p) / n + z^2 / (4 * n^2)) / (1 + z^2 / n)
  list(lo = c0 - h, hi = c0 + h)
}
N_IV <- nrow(filter(iv, method == "split"))

# ------------------------------------------------------------------ (a) --
band <- tibble(nominal = seq(0.45, 0.97, 0.005)) |>
  mutate(lo = nominal - 1.96 * sqrt(nominal * (1 - nominal) / N_IV),
         hi = nominal + 1.96 * sqrt(nominal * (1 - nominal) / N_IV))
pa <- ggplot(cov, aes(nominal, empirical_coverage, colour = method)) +
  geom_ribbon(data = band, aes(nominal, ymin = lo, ymax = hi), inherit.aes = FALSE,
              fill = NEUTRAL, alpha = 0.25) +
  geom_abline(slope = 1, intercept = 0, colour = INK, linetype = "22",
              linewidth = 0.4) +
  geom_line(linewidth = 0.6) +
  geom_point(size = 1.9) +
  scale_colour_manual(values = PAL_M, labels = METHOD, name = NULL) +
  scale_x_continuous(labels = percent_format(accuracy = 1),
                     breaks = seq(0.5, 0.9, 0.1), limits = c(0.45, 1)) +
  scale_y_continuous(labels = percent_format(accuracy = 1),
                     breaks = c(0.5, 0.6, 0.7, 0.8, 0.9, 1), limits = c(0.45, 1)) +
  coord_equal() +
  labs(x = "Nominal level", y = "Empirical coverage") +
  theme(legend.position = "inside", legend.position.inside = c(0.02, 0.98),
        legend.justification = c(0, 1), legend.key.size = unit(7, "pt"),
        legend.text = element_text(size = 6), legend.background = element_blank())

# ------------------------------------------------------------------ (b) --
pb_df <- cond |>
  mutate(bin = factor(bin, levels = BIN_LAB),
         side = factor(ifelse(conditioned_on == "measured", "By measured Secchi",
                              "By predicted Secchi"),
                       levels = c("By measured Secchi", "By predicted Secchi")),
         lo = wilson(coverage, n)$lo, hi = wilson(coverage, n)$hi,
         method = factor(method, levels = names(METHOD)))
DODGE <- position_dodge(width = 0.55)
pb <- ggplot(pb_df, aes(bin, coverage, colour = method)) +
  geom_hline(yintercept = 0.9, colour = INK, linetype = "22", linewidth = 0.4) +
  geom_errorbar(aes(ymin = lo, ymax = hi), width = 0, linewidth = 0.45,
                position = DODGE) +
  geom_point(size = 1.8, position = DODGE) +
  facet_wrap(~side, nrow = 1) +
  scale_colour_manual(values = PAL_M, labels = METHOD, guide = "none") +
  scale_y_continuous(labels = percent_format(accuracy = 1),
                     limits = c(0.45, 1.01), breaks = seq(0.5, 1, 0.1)) +
  labs(x = "Secchi depth (m)", y = "Coverage of the 90% interval") +
  theme(strip.text = element_text(face = "bold", hjust = 0),
        panel.spacing = unit(8, "pt"))

# ------------------------------------------------------------------ (c) --
ivm <- iv |> filter(method == "mondrian") |>
  mutate(covered = as.logical(covered)) |>
  arrange(secchi, predicted) |> mutate(rank = row_number())
n_out <- sum(!ivm$covered)
pc <- ggplot(ivm, aes(rank)) +
  geom_linerange(aes(ymin = lower, ymax = upper, colour = covered),
                 linewidth = 0.35, alpha = 0.55) +
  geom_point(aes(y = predicted), size = 0.25, colour = INK_2, alpha = 0.6) +
  geom_point(aes(y = secchi, fill = covered), shape = 21, colour = "white",
             stroke = 0.15, size = 1.3) +
  annotate("text", x = 5, y = max(ivm$upper) + 0.4, hjust = 0, vjust = 1,
           size = 2.3, colour = INK_2,
           label = sprintf("Mondrian, 90%%: %d of %d measurements outside their interval",
                           n_out, nrow(ivm))) +
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

fig <- ((pa | pb) + plot_layout(widths = c(0.8, 1.4))) / pc +
  plot_layout(heights = c(1, 0.85)) + tags_abc()
save_fig(fig, "fig07_conformal_uncertainty", W2, 160)
cat("fig07 lista\n")
