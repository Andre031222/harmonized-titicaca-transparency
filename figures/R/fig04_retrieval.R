# ============================================================================
# fig04_retrieval.R -- Recuperacion bajo el diseno primario
#
# (a) Predicho vs medido fuera de fold, con 1:1
# (b) Residuos por zona trofica (nube de lluvia), con n y Kruskal-Wallis
# (c) Sesgo medio por estacion sobre el lago
# (d) Benchmark contra los baselines clasico y lineal
# (e) El baseline sin acotar en escala log: una prediccion imposible
# ============================================================================

source(file.path(local({a <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(a)) dirname(normalizePath(sub("^--file=", "", a[1]))) else getwd()}),
  "theme_titicaca.R"))
source(file.path(ROOT, "figures", "R", "map_base.R"))
suppressPackageStartupMessages({library(sf); library(ggspatial)})
sf_use_s2(FALSE)

oof   <- read_tidy("oof_predictions.csv") |> mutate(zone_label = zone_factor(zone_label))
bench <- read_tidy("benchmark_models.csv")
lake  <- st_read(file.path(ROOT, "data/lake_boundary/titicaca.gpkg"), quiet = TRUE) |>
  st_transform(32719)

rf   <- bench |> filter(family == "rf")
lims <- range(c(oof$secchi, oof$predicted))
zone_x <- function(x) sub(" de ", "\nde ", sub("Lago ", "Lago\n", x))

# ------------------------------------------------------------------ (a) --
pa <- ggplot(oof, aes(secchi, predicted)) +
  geom_abline(slope = 1, intercept = 0, colour = INK, linetype = "22",
              linewidth = 0.35) +
  geom_point(aes(fill = zone_label), shape = 21, colour = "white", stroke = 0.2,
             size = 1.35, alpha = 0.85) +
  geom_smooth(method = "lm", formula = y ~ x, colour = INK, fill = "grey60",
              linewidth = 0.55, alpha = 0.3) +
  annotate("text", x = lims[1], y = lims[2], hjust = 0, vjust = 1, size = 2.4,
           lineheight = 1.05, colour = INK,
           label = metric_label(rf$R2, rf$RMSE, n = rf$n, mae = rf$MAE)) +
  scale_fill_manual(values = PAL_ZONE, name = NULL) +
  coord_equal(xlim = lims, ylim = lims) +
  labs(x = "Measured Secchi (m)", y = "Predicted Secchi (m)") +
  guides(fill = guide_legend(override.aes = list(size = 2, alpha = 1))) +
  theme(legend.position = "inside", legend.position.inside = c(1, 0.02),
        legend.justification = c(1, 0), legend.key.size = unit(7, "pt"),
        legend.background = element_blank(), panel.grid.major.y = element_blank())

# ------------------------------------------------------------------ (b) --
kw   <- kruskal.test(residual ~ zone_label, data = oof)$p.value
bias <- oof |> group_by(zone_label) |>
  summarise(b = mean(residual), .groups = "drop") |>
  mutate(txt = sub("-", "−", sprintf("%+.2f m", b)))

# comparaciones por pares (Wilcoxon, BH) en corchetes sobre la nube
brk <- pairwise_bh(oof$residual, oof$zone_label, c(6.6, 8.1, 9.6))

pb <- ggplot(oof, aes(zone_label, residual, colour = zone_label)) +
  geom_hline(yintercept = 0, colour = INK, linewidth = 0.3, linetype = "22") +
  raincloud(point_size = 0.5) +
  geom_segment(data = brk, aes(x = x1, xend = x2, y = y, yend = y),
               inherit.aes = FALSE, linewidth = 0.3, colour = INK) +
  geom_segment(data = brk, aes(x = x1, xend = x1, y = y, yend = y - 0.3),
               inherit.aes = FALSE, linewidth = 0.3, colour = INK) +
  geom_segment(data = brk, aes(x = x2, xend = x2, y = y, yend = y - 0.3),
               inherit.aes = FALSE, linewidth = 0.3, colour = INK) +
  geom_text(data = brk, aes(x = (x1 + x2) / 2, y = y + 0.1, label = lab),
            inherit.aes = FALSE, parse = TRUE, vjust = 0, size = 2.1, colour = INK) +
  geom_text(data = bias, aes(zone_label, -8.2, label = txt), inherit.aes = FALSE,
            size = 2.3, colour = INK, fontface = "bold") +
  n_labels(oof, zone_label, -9.3) +
  annotate("text", x = 0.5, y = 12.9, hjust = 0, size = 2.2, colour = INK_2,
           parse = TRUE, label = paste0('"Kruskal–Wallis, "*', p_fmt(kw))) +
  scale_colour_manual(values = PAL_ZONE, guide = "none") +
  scale_x_discrete(labels = zone_x) +
  scale_y_continuous(limits = c(-9.8, 13.3), breaks = seq(-6, 6, 3)) +
  labs(x = NULL, y = "Residual (m)")

# ------------------------------------------------------------------ (c) --
st_bias <- oof |>
  group_by(station) |>
  summarise(lat = mean(lat), lon = mean(lon), bias = mean(residual), n = n(),
            .groups = "drop") |>
  st_as_sf(coords = c("lon", "lat"), crs = 4326) |> st_transform(32719)
# unos pocos extremos aplastarian la escala: se acota a +-2 m
lim_b <- 2

pc <- ggplot() +
  relief_layers() +
  geom_sf(data = st_bias |> arrange(abs(bias)),
          aes(fill = bias, size = n), shape = 21, colour = "white", stroke = 0.3) +
  scale_fill_gradient2(low = "#2166AC", mid = "#F7F7F7", high = "#B2182B",
                       midpoint = 0, limits = c(-lim_b, lim_b), oob = squish,
                       breaks = c(-2, -1, 0, 1, 2),
                       labels = c("≤−2", "−1", "0", "1", "≥2"),
                       name = "Mean bias (m)",
                       guide = guide_colourbar(order = 1, barwidth = unit(4, "pt"),
                                               barheight = unit(34, "pt"))) +
  scale_size_continuous(range = c(0.8, 3.2), name = "Match-ups",
                        breaks = c(2, 6, 12),
                        guide = guide_legend(order = 2, override.aes = list(
                          fill = "#9E9E9E"))) +
  north("tr") + scale_bar("bl", 0.3) +
  lake_coord() +
  scale_x_continuous(breaks = c(-70, -69.5, -69, -68.5)) +
  scale_y_continuous(breaks = c(-16.5, -16, -15.5)) +
  map_theme +
  theme(legend.position = "right", legend.key.size = unit(7, "pt"),
        plot.margin = margin(2, 2, 2, 2))

# ------------------------------------------------------------------ (d) --
# Benchmark bajo el mismo diseno: los dos algoritmos clasicos de razon de
# bandas no tienen habilidad; el lineal multibanda y el RF si.
pd_df <- bench |>
  filter(family != "artefact") |>
  mutate(name = case_when(
    family == "kloiber"   ~ "Kloiber et al. (2002),\nblue/red + blue",
    family == "classical" ~ "Two-band\nblue/green ratio",
    family == "linear"    ~ "Multiband linear",
    TRUE                  ~ "Random Forest"),
    kind = ifelse(family == "rf", "rf", ifelse(family == "linear", "lin", "cl")),
    name = fct_reorder(name, R2))

pd <- ggplot(pd_df, aes(y = name)) +
  geom_vline(xintercept = 0, colour = INK, linewidth = 0.3) +
  geom_segment(aes(x = 0, xend = R2, colour = kind), linewidth = 1.1) +
  geom_point(aes(x = R2, colour = kind), size = 3) +
  geom_text(aes(x = pmax(R2, 0), label = sub("-", "−", sprintf("%.3f", R2))),
            hjust = -0.35, size = 2.4, fontface = "bold", colour = INK) +
  geom_text(aes(x = 0.86, label = sprintf("%.2f m", RMSE)),
            hjust = 1, size = 2.3, colour = INK_2) +
  annotate("text", x = 0.86, y = 4.55, label = "RMSE", hjust = 1, size = 2.2,
           colour = INK_2, fontface = "italic") +
  scale_colour_manual(values = c(cl = "#BDBDBD", lin = "#8FB4C7", rf = "#2E6E8E"),
                      guide = "none") +
  scale_x_continuous(limits = c(-0.05, 0.88), breaks = seq(0, 0.6, 0.2),
                     expand = expansion(mult = c(0, 0))) +
  scale_y_discrete(limits = levels(pd_df$name),
                   expand = expansion(add = c(0.6, 0.9))) +
  labs(x = expression("Out-of-fold "*italic(R)^2), y = NULL) +
  theme(panel.grid.major.y = element_blank(), axis.line.y = element_blank(),
        axis.ticks.y = element_blank())

# ------------------------------------------------------------------ (e) --
# Residuo medio por estacion frente a la distancia a la orilla (p12): cerca de
# tierra la correccion atmosferica terrestre sufre el efecto de adyacencia.
sa  <- jsonlite::fromJSON(file.path(ROOT, "results/metrics/spatial_autocorrelation.json"))
pe_df <- read_tidy("lisa_stations.csv") |> mutate(zone_label = zone_factor(zone_label))
sh <- sa$shore_distance$all
pe <- ggplot(pe_df, aes(shore_km, residual)) +
  geom_hline(yintercept = 0, colour = INK, linetype = "22", linewidth = 0.3) +
  geom_point(aes(fill = zone_label, size = n), shape = 21, colour = "white",
             stroke = 0.25, alpha = 0.9) +
  geom_smooth(method = "loess", formula = y ~ x, span = 0.9, colour = INK,
              fill = "grey70", linewidth = 0.55, alpha = 0.3) +
  annotate("text", x = Inf, y = Inf, hjust = 1.03, vjust = 1.3, size = 2.3,
           colour = INK, parse = TRUE,
           label = sprintf('"Spearman "*rho*" = %.2f, "*italic(P)*" = %.3f"',
                           sh$rho_residual, sh$p_residual)) +
  scale_fill_manual(values = PAL_ZONE, name = NULL) +
  scale_size_continuous(range = c(0.8, 3.2), guide = "none") +
  scale_x_continuous(trans = "log10", breaks = c(0.2, 0.5, 1, 2, 5, 10, 20),
                     labels = c("0.2", "0.5", "1", "2", "5", "10", "20")) +
  labs(x = "Distance to shore (km, log scale)",
       y = "Station mean residual (m)") +
  guides(fill = guide_legend(override.aes = list(size = 2.2))) +
  theme(legend.position = "inside", legend.position.inside = c(0.99, 0.03),
        legend.justification = c(1, 0), legend.key.size = unit(6, "pt"),
        legend.background = element_blank())

fig <- (pa | pb | pc) / (pd | pe) +
  plot_layout(heights = c(1, 0.78)) + tags_abc()
save_fig(fig, "fig04_retrieval_and_benchmark", W2, 126)
cat("fig04 lista\n")
