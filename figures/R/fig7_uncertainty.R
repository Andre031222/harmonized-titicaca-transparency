# ============================================================================
# fig7_uncertainty.R -- Intervalos conformales bajo leave-one-campaign-out (p07)
#
# (a) Cobertura empirica frente a nominal (50/80/90/95 %) de los tres metodos,
#     con IC 95 % por bootstrap de campanas
# (b) Cobertura del intervalo al 90 % condicionada al Secchi MEDIDO y al
#     PREDICHO, con n por tramo
# (c) Cada intervalo Mondrian al 90 %, ordenado por el Secchi medido
# (d) Cobertura al 90 % por campana (2022 es la debil)
# ============================================================================

source(file.path(local({a <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(a)) dirname(normalizePath(sub("^--file=", "", a[1]))) else getwd()}),
  "theme_titicaca.R"))
suppressPackageStartupMessages(library(jsonlite))

cov <- read_tidy("conformal_coverage.csv")
iv  <- read_tidy("conformal_intervals.csv")
# el JSON de p07 escribe NaN (no es JSON valido): se lee como null
uj  <- fromJSON(gsub("\\bNaN\\b", "null", paste(readLines(
  file.path(ROOT, "results/metrics/uncertainty.json"), warn = FALSE), collapse = "\n")))

METHOD <- c(split = "Split (constant width)",
            mondrian = "Mondrian (by predicted Secchi)",
            cqr = "Conformalized quantile regression")
PAL_M <- c(split = "#8A8A8A", mondrian = "#2E6E8E", cqr = ACCENT)
BIN_KEY <- c("<5", "5-7.5", "7.5-10", "10-12.5", ">12.5")
BIN_LAB <- setNames(c("<5", "5–7.5", "7.5–10", "10–12.5", ">12.5"), BIN_KEY)
cov <- cov |> mutate(method = factor(method, levels = names(METHOD)))
stopifnot(!anyNA(cov$method))
DODGE <- position_dodge(width = 0.6)

# ------------------------------------------------------------------ (a) --
ma <- cov |> filter(conditioned_on == "marginal")
stopifnot(nrow(ma) == 12)
DA <- position_dodge(width = 0.035)
pa <- ggplot(ma, aes(nominal, coverage, colour = method)) +
  geom_abline(slope = 1, intercept = 0, colour = INK, linetype = "22",
              linewidth = 0.35) +
  geom_errorbar(aes(ymin = coverage_lo, ymax = coverage_hi), width = 0,
                linewidth = 0.45, position = DA) +
  geom_point(size = 1.6, position = DA) +
  scale_colour_manual(values = PAL_M, labels = METHOD, name = NULL) +
  scale_x_continuous(labels = function(x) sprintf("%.0f", 100 * x),
                     breaks = sort(unique(ma$nominal))) +
  scale_y_continuous(labels = percent_format(accuracy = 1),
                     breaks = seq(0.4, 1, 0.1)) +
  coord_cartesian(xlim = c(0.45, 0.99), ylim = c(0.4, 1)) +
  labs(x = "Nominal level (%)", y = "Empirical coverage") +
  theme(legend.position = "inside", legend.position.inside = c(1, 0.02),
        legend.justification = c(1, 0), legend.key.size = unit(7, "pt"),
        legend.text = element_text(size = 6.5))

# ------------------------------------------------------------------ (b) --
SIDE <- c(measured_secchi = "By measured Secchi",
          predicted_secchi = "By predicted Secchi")
mb <- cov |> filter(conditioned_on %in% names(SIDE)) |>
  mutate(bin = factor(map_strict(group, BIN_LAB, "tramo"), levels = BIN_LAB),
         side = factor(map_strict(conditioned_on, SIDE, "condicion"),
                       levels = SIDE))
nb <- mb |> distinct(side, bin, n)
stopifnot(nrow(nb) == 10)
Y0 <- 0.2
pb <- ggplot(mb, aes(bin, coverage, colour = method)) +
  geom_hline(yintercept = 0.9, colour = INK, linetype = "22", linewidth = 0.35) +
  geom_errorbar(aes(ymin = coverage_lo, ymax = coverage_hi), width = 0,
                linewidth = 0.45, position = DODGE) +
  geom_point(size = 1.6, position = DODGE) +
  geom_text(data = nb, aes(bin, Y0, label = paste0("n = ", n)),
            inherit.aes = FALSE, size = 2.3, colour = INK_2, vjust = 0) +
  facet_wrap(~side, nrow = 1) +
  scale_colour_manual(values = PAL_M, guide = "none") +
  scale_y_continuous(labels = percent_format(accuracy = 1),
                     breaks = seq(0.3, 1, 0.1)) +
  coord_cartesian(ylim = c(Y0, 1.01)) +
  labs(x = "Secchi depth bin (m)", y = "Coverage of the 90% interval")

# ------------------------------------------------------------------ (c) --
ivm <- iv |> filter(method == "mondrian", abs(nominal - 0.9) < 1e-9) |>
  mutate(covered = as.logical(covered)) |>
  arrange(secchi, predicted) |> mutate(rank = row_number())
stopifnot(nrow(ivm) == uj$n)
n_out <- sum(!ivm$covered)
pc <- ggplot(ivm, aes(rank)) +
  geom_linerange(aes(ymin = lower, ymax = upper, colour = covered),
                 linewidth = 0.3, alpha = 0.6) +
  geom_point(aes(y = predicted, shape = "Predicted"), size = 0.35,
             colour = INK_2, alpha = 0.8) +
  geom_point(data = filter(ivm, covered), aes(y = secchi, fill = covered),
             shape = 21, colour = "white", stroke = 0.12, size = 1.1) +
  geom_point(data = filter(ivm, !covered), aes(y = secchi, fill = covered),
             shape = 21, colour = "white", stroke = 0.12, size = 1.3) +
  annotate("text", x = 1, y = Inf, hjust = 0, vjust = 1.2, size = 2.3,
           colour = INK_2,
           label = sprintf("%d of %d measurements (%.1f%%) outside their 90%% interval",
                           n_out, nrow(ivm), 100 * n_out / nrow(ivm))) +
  scale_colour_manual(values = c(`TRUE` = "#A9C6D6", `FALSE` = "#E6A483"),
                      guide = "none") +
  scale_fill_manual(values = c(`TRUE` = "#2E6E8E", `FALSE` = ACCENT),
                    labels = c(`TRUE` = "Measured, inside",
                               `FALSE` = "Measured, outside"), name = NULL,
                    breaks = c("TRUE", "FALSE")) +
  scale_shape_manual(values = 16, name = NULL) +
  scale_x_continuous(expand = expansion(add = 6)) +
  scale_y_continuous(expand = expansion(mult = c(0.03, 0.1))) +
  labs(x = "Match-ups ranked by measured Secchi",
       y = "Secchi and 90% interval (m)") +
  guides(fill = guide_legend(order = 1, override.aes = list(size = 2)),
         shape = guide_legend(order = 2, override.aes = list(size = 1, alpha = 1))) +
  theme(legend.position = "inside", legend.position.inside = c(1, 0.02),
        legend.justification = c(1, 0), legend.key.size = unit(7, "pt"),
        legend.text = element_text(size = 6.5), legend.spacing.y = unit(0, "pt"),
        legend.box.just = "left")

# ------------------------------------------------------------------ (d) --
md <- cov |> filter(conditioned_on == "campaign") |>
  mutate(campaign = factor(group))
nd <- md |> distinct(campaign, n)
stopifnot(nrow(nd) == uj$n_campaigns)
Y0d <- 0.46  # fila de n por campana al pie del panel
pd <- ggplot(md, aes(campaign, coverage, colour = method)) +
  geom_hline(yintercept = 0.9, colour = INK, linetype = "22", linewidth = 0.35) +
  geom_point(size = 1.5, position = position_dodge(width = 0.6)) +
  geom_text(data = nd, aes(campaign, Y0d, label = n), inherit.aes = FALSE,
            size = 2.3, colour = INK_2, vjust = 0) +
  scale_colour_manual(values = PAL_M, guide = "none") +
  scale_y_continuous(labels = percent_format(accuracy = 1),
                     breaks = seq(0.5, 1, 0.1)) +
  coord_cartesian(ylim = c(Y0d - 0.02, 1.01)) +
  labs(x = "Held-out campaign", y = "Coverage of the 90% interval") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# ---------------------------------------------------------------- montaje --
top <- (pa | pb) + plot_layout(widths = c(0.75, 1.5))
bot <- (pc | pd) + plot_layout(widths = c(1.6, 0.75))
fig <- top / bot + plot_layout(heights = c(1, 0.95)) + tags_abc()
save_fig(fig, "fig7_conformal_uncertainty", W2, 160)
cat("fig7 lista\n")
