# ============================================================================
# fig5_retrieval.R -- Recuperacion bajo el diseno primario (dejar fuera una
# campana) con el producto HLS
#
# (a) Predicho vs medido fuera de fold, 1:1 y ajuste por minimos cuadrados
# (b) Residuos por zona (nube de lluvia), sesgo medio y n
# (c) Error por campana: sesgo y RMSE, con n y match-ups Sentinel-2
# (d) Benchmark bajo el mismo diseno: Kloiber 2002, razon de dos bandas,
#     lineal multibanda y Random Forest, con IC 95 % (bootstrap por campana)
#
# Entradas: results/tidy/oof_predictions.csv, error_by_campaign.csv,
# error_by_zone.csv, benchmark_models.csv; results/metrics/validation.json.
# Ningun numero escrito a mano. Estilo heredado de fig04_retrieval.R (main).
# ============================================================================

source(file.path(local({a <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(a)) dirname(normalizePath(sub("^--file=", "", a[1]))) else getwd()}),
  "theme_titicaca.R"))

oof   <- read_tidy("oof_predictions.csv") |> mutate(zone_label = zone_factor(zona))
ecamp <- read_tidy("error_by_campaign.csv")
ezone <- read_tidy("error_by_zone.csv") |> mutate(zone_label = zone_factor(zona))
bench <- read_tidy("benchmark_models.csv")
val   <- jsonlite::fromJSON(file.path(ROOT, "results", "metrics", "validation.json"))
hd    <- val$headline

# coherencia entre fuentes: el titular y las predicciones son la misma cosa
stopifnot(nrow(oof) == hd$n,
          abs(sqrt(mean(oof$residual^2)) - hd$RMSE) < 0.01,
          all(abs(oof$pred - oof$secchi - oof$residual) < 1e-3))

minus <- function(s) gsub("-", "−", s)
zone_x <- function(x) sub(" de ", "\nde ", sub("Lago ", "Lago\n", x))

# ------------------------------------------------------------------ (a) --
lims <- range(c(0, oof$secchi, oof$pred))
lims[2] <- ceiling(lims[2])
fit <- lm(pred ~ secchi, data = oof)

pa <- ggplot(oof, aes(secchi, pred)) +
  geom_abline(slope = 1, intercept = 0, colour = INK, linetype = "22",
              linewidth = 0.35) +
  # Secchi se anota a 0.5 m: un temblor horizontal de +-0.12 m (solo visual)
  # separa los puntos apilados; el ajuste usa los valores reales
  geom_point(aes(fill = zone_label), shape = 21, colour = "white", stroke = 0.2,
             size = 1.3, alpha = 0.8,
             position = position_jitter(width = 0.12, height = 0, seed = 1)) +
  geom_smooth(method = "lm", formula = y ~ x, colour = INK, fill = "grey55",
              linewidth = 0.55, alpha = 0.3) +
  annotate("text", x = lims[1] + 0.3, y = lims[2] - 0.2, hjust = 0, vjust = 1,
           size = 2.4, lineheight = 1.1, colour = INK, parse = FALSE,
           label = sprintf("R² = %.3f [%.3f, %.3f]\nRMSE = %.2f m [%.2f, %.2f]\nBias = %s m\nLS slope = %.2f\nn = %d",
                           hd$R2, hd$R2_lo, hd$R2_hi, hd$RMSE, hd$RMSE_lo,
                           hd$RMSE_hi, minus(sprintf("%+.2f", hd$bias)),
                           coef(fit)[2], hd$n)) +
  scale_fill_manual(values = PAL_ZONE, name = NULL) +
  scale_x_continuous(breaks = seq(0, 20, 4)) +
  scale_y_continuous(breaks = seq(0, 20, 4)) +
  coord_equal(xlim = lims, ylim = lims, expand = FALSE) +
  labs(x = "Measured Secchi (m)", y = "Predicted Secchi (m)") +
  guides(fill = guide_legend(override.aes = list(size = 2, alpha = 1))) +
  theme(legend.position = "inside", legend.position.inside = c(1, 0.02),
        legend.justification = c(1, 0), legend.key.size = unit(7, "pt"),
        legend.background = element_blank(), panel.grid.major.y = element_blank())

# ------------------------------------------------------------------ (b) --
kw <- kruskal.test(residual ~ zone_label, data = oof)$p.value
rng <- range(oof$residual)
bias_y <- rng[1] - 1.2
n_y    <- rng[1] - 2.3
ez <- ezone |> mutate(txt = minus(sprintf("%+.2f m", bias)))

pb <- ggplot(oof, aes(zone_label, residual, colour = zone_label)) +
  geom_hline(yintercept = 0, colour = INK, linewidth = 0.3, linetype = "22") +
  raincloud(point_size = 0.45) +
  geom_text(data = ez, aes(zone_label, bias_y, label = txt), inherit.aes = FALSE,
            size = 2.3, colour = INK, fontface = "bold") +
  n_labels(oof, zone_label, n_y + 0.35) +
  annotate("text", x = 0.5, y = rng[2] + 1.6, hjust = 0, vjust = 1, size = 2.3,
           colour = INK_2, parse = TRUE,
           label = paste0('"Kruskal–Wallis, "*', p_fmt(kw))) +
  scale_colour_manual(values = PAL_ZONE, guide = "none") +
  scale_x_discrete(labels = zone_x) +
  scale_y_continuous(limits = c(n_y - 0.6, rng[2] + 1.7),
                     breaks = seq(-9, 9, 3), labels = function(b) minus(b)) +
  labs(x = NULL, y = "Residual, predicted − measured (m)")

# ------------------------------------------------------------------ (c) --
ec <- ecamp |>
  mutate(campaign = factor(campaign),
         lab = sprintf("%d\n%d S2", n, n_s2))
yl <- c(min(-max(ec$RMSE), min(ec$bias)) - 0.9, max(ec$RMSE) + 0.4)

pc <- ggplot(ec, aes(campaign)) +
  geom_hline(yintercept = 0, colour = INK, linewidth = 0.3) +
  geom_col(aes(y = bias, fill = bias > 0), width = 0.62) +
  geom_point(aes(y = RMSE), shape = 23, size = 2, fill = "white", colour = INK,
             stroke = 0.5) +
  geom_text(data = ec |> slice_max(abs(bias), n = 1),
            aes(y = bias, label = minus(sprintf("%+.2f m", bias)),
                vjust = ifelse(bias < 0, 1.4, -0.4)),
            size = 2.3, fontface = "bold", colour = INK) +
  geom_text(aes(y = yl[1], label = lab), vjust = 0, size = 2.3, colour = INK_2,
            lineheight = 0.9) +
  scale_fill_manual(values = c(`TRUE` = "#B2182B", `FALSE` = "#2166AC"),
                    labels = c(`TRUE` = "Bias > 0", `FALSE` = "Bias < 0"),
                    name = NULL, breaks = c("FALSE", "TRUE")) +
  scale_y_continuous(limits = yl, breaks = seq(-3, 3, 1),
                     labels = function(b) minus(b)) +
  annotate("point", x = 0.75, y = yl[2] - 0.05, shape = 23, size = 2,
           fill = "white", colour = INK, stroke = 0.5) +
  annotate("text", x = 0.95, y = yl[2] - 0.05, label = "RMSE", hjust = 0,
           size = 2.3, colour = INK) +
  labs(x = "Campaign (held out)", y = "Error (m)") +
  theme(legend.position = "inside", legend.position.inside = c(0.27, 1),
        legend.justification = c(0, 1), legend.direction = "horizontal",
        legend.key.size = unit(7, "pt"), legend.text = element_text(size = 6.8))

# ------------------------------------------------------------------ (d) --
# Las versiones sin acotar dan aqui lo mismo que las acotadas (ninguna
# prediccion absurda): se muestran solo las acotadas.
FAM <- c(kloiber  = "Kloiber et al. (2002)\nblue/red + blue",
         two_band = "Two-band ratio\nln(blue/green)",
         linear   = "Multiband linear\n(12 features)",
         rf       = "Random Forest\n(12 features)")
bd <- bench |> filter(subset == "pooled", bounded) |>
  mutate(name = factor(map_strict(model, FAM, "modelo"), levels = FAM),
         kind = ifelse(model == "rf", "rf", ifelse(model == "linear", "lin", "cl")))
stopifnot(nrow(bd) == 4)
x_txt <- max(bd$R2_hi) + 0.13

pd <- ggplot(bd, aes(y = name)) +
  geom_vline(xintercept = 0, colour = INK, linewidth = 0.3) +
  geom_linerange(aes(xmin = R2_lo, xmax = R2_hi, colour = kind), linewidth = 0.9) +
  geom_point(aes(x = R2, colour = kind), size = 2.8) +
  geom_label(aes(x = R2, label = minus(sprintf("%.3f", R2))), vjust = -0.7,
             size = 2.3, fontface = "bold", colour = INK, fill = "white",
             linewidth = 0, label.padding = unit(0.6, "pt")) +
  geom_text(aes(x = x_txt, label = sprintf("%.2f m", RMSE)), hjust = 1,
            size = 2.3, colour = INK_2) +
  annotate("text", x = x_txt, y = 4.6, label = "RMSE", hjust = 1, size = 2.3,
           colour = INK_2, fontface = "italic") +
  scale_colour_manual(values = c(cl = "#9E9E9E", lin = "#8FB4C7", rf = "#2E6E8E"),
                      guide = "none") +
  scale_x_continuous(limits = c(min(bd$R2_lo) - 0.03, x_txt + 0.01),
                     breaks = seq(0, 0.6, 0.2),
                     labels = function(b) minus(b)) +
  scale_y_discrete(expand = expansion(add = c(0.5, 0.8))) +
  labs(x = expression("Out-of-fold "*italic(R)^2*" (95% CI)"), y = NULL) +
  theme(panel.grid.major.y = element_blank(), axis.line.y = element_blank(),
        axis.ticks.y = element_blank(),
        axis.text.y = element_text(colour = INK, lineheight = 0.95))

fig <- ((pa | pb) + plot_layout(widths = c(1, 1))) /
       ((pc | pd) + plot_layout(widths = c(1.15, 1))) +
  plot_layout(heights = c(1, 0.8)) + tags_abc()
save_fig(fig, "fig5_retrieval", W2, 160)
cat("fig5 lista\n")
