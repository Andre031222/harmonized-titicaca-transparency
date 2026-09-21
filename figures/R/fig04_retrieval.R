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
suppressPackageStartupMessages({library(sf); library(ggspatial)})
sf_use_s2(FALSE)

oof   <- read_tidy("oof_predictions.csv") |> mutate(zone_label = zone_factor(zone_label))
bench <- read_tidy("benchmark_models.csv")
art   <- read_tidy("baseline_artefact.csv")
lake  <- st_read(file.path(ROOT, "data/lake_boundary/titicaca.gpkg"), quiet = TRUE) |>
  st_transform(32719)

rf   <- bench |> filter(family == "rf")
lims <- range(c(oof$secchi, oof$predicted))
zone_x <- function(x) sub(" de ", "\nde ", sub("Lago ", "Lago\n", x))

# ------------------------------------------------------------------ (a) --
pa <- ggplot(oof, aes(secchi, predicted, colour = zone_label)) +
  geom_abline(slope = 1, intercept = 0, colour = INK, linetype = "22",
              linewidth = 0.35) +
  geom_point(size = 0.8, alpha = 0.55, shape = 16) +
  annotate("text", x = lims[1], y = lims[2], hjust = 0, vjust = 1, size = 2.4,
           lineheight = 1.05, colour = INK,
           label = metric_label(rf$R2, rf$RMSE, n = rf$n, mae = rf$MAE)) +
  scale_colour_manual(values = PAL_ZONE, name = NULL) +
  coord_equal(xlim = lims, ylim = lims) +
  labs(x = "Measured Secchi (m)", y = "Predicted Secchi (m)") +
  theme(legend.position = "inside", legend.position.inside = c(1, 0.05),
        legend.justification = c(1, 0), legend.key.size = unit(6, "pt"),
        panel.grid.major.y = element_blank())

# ------------------------------------------------------------------ (b) --
kw   <- kruskal.test(residual ~ zone_label, data = oof)$p.value
bias <- oof |> group_by(zone_label) |>
  summarise(b = mean(residual), .groups = "drop") |>
  mutate(txt = sub("-", "−", sprintf("%+.2f m", b)))

pb <- ggplot(oof, aes(zone_label, residual, colour = zone_label)) +
  geom_hline(yintercept = 0, colour = INK, linewidth = 0.3, linetype = "22") +
  raincloud(point_size = 0.5) +
  geom_text(data = bias, aes(zone_label, 7.2, label = txt), inherit.aes = FALSE,
            size = 2.3, colour = INK, fontface = "bold") +
  n_labels(oof, zone_label, -6.9) +
  annotate("text", x = 0.55, y = 8.6, hjust = 0, size = 2.3, colour = INK_2,
           parse = TRUE, label = paste0('"Kruskal–Wallis, "*', p_fmt(kw))) +
  scale_colour_manual(values = PAL_ZONE, guide = "none") +
  scale_x_discrete(labels = zone_x) +
  scale_y_continuous(limits = c(-7.6, 8.8), breaks = seq(-6, 6, 3)) +
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
  geom_sf(data = lake, fill = "#EEF3F6", colour = "#7FA3BA", linewidth = 0.3) +
  geom_sf(data = st_bias |> arrange(abs(bias)),
          aes(fill = bias, size = n), shape = 21, colour = "white", stroke = 0.25) +
  scale_fill_gradient2(low = "#2166AC", mid = "#F7F7F7", high = "#B2182B",
                       midpoint = 0, limits = c(-lim_b, lim_b), oob = squish,
                       breaks = c(-2, -1, 0, 1, 2),
                       labels = c("≤−2", "−1", "0", "1", "≥2"),
                       name = "Mean bias (m)",
                       guide = guide_colourbar(barwidth = unit(4, "pt"),
                                               barheight = unit(34, "pt"))) +
  scale_size_continuous(range = c(0.7, 3), name = "Match-ups",
                        breaks = c(2, 6, 12),
                        guide = guide_legend(override.aes = list(fill = "#9E9E9E"))) +
  annotation_north_arrow(location = "tr", height = unit(0.7, "cm"),
                         width = unit(0.55, "cm"),
                         style = north_arrow_fancy_orienteering(text_size = 5)) +
  annotation_scale(location = "bl", width_hint = 0.3, text_cex = 0.5,
                   height = unit(0.1, "cm"), bar_cols = c(INK, "white")) +
  coord_sf(crs = 32719, datum = NA) +
  theme_void(base_size = 8) +
  theme(legend.position = "right", legend.title = element_text(size = 6.5),
        legend.text = element_text(size = 6), legend.key.size = unit(7, "pt"),
        plot.tag = element_text(size = 10, face = "bold"),
        plot.tag.position = c(0, 1), plot.margin = margin(8, 2, 2, 2))

# ------------------------------------------------------------------ (d) --
pd_df <- bench |>
  mutate(name = case_when(
    family == "artefact"  ~ "Blue/green ratio,\nunbounded",
    family == "classical" ~ "Blue/green ratio,\nbounded",
    family == "linear"    ~ "Multiband linear",
    TRUE                  ~ "Random Forest"),
    kind = ifelse(family == "artefact", "a", ifelse(family == "rf", "rf", "b")),
    name = fct_reorder(name, R2))

pd <- ggplot(pd_df, aes(R2, name, fill = kind)) +
  geom_vline(xintercept = 0, colour = INK, linewidth = 0.3) +
  geom_col(width = 0.6) +
  geom_text(aes(label = sprintf("%.3f", R2), hjust = ifelse(R2 > 0, -0.2, 1.2)),
            size = 2.4, colour = INK) +
  scale_fill_manual(values = c(a = "#BDBDBD", b = "#8FB4C7", rf = "#2E6E8E"),
                    guide = "none") +
  scale_x_continuous(limits = c(-3.2, 1.3), breaks = seq(-3, 1, 1),
                     expand = expansion(mult = c(0.01, 0))) +
  labs(x = expression("Out-of-fold "*italic(R)^2), y = NULL) +
  theme(panel.grid.major.y = element_blank(), axis.line.y = element_blank(),
        axis.ticks.y = element_blank())

# ------------------------------------------------------------------ (e) --
pe <- art |>
  select(observed, predicted_unbounded, predicted_bounded) |>
  pivot_longer(-observed, names_to = "version", values_to = "pred") |>
  mutate(version = recode(version, predicted_unbounded = "Unbounded",
                          predicted_bounded = "Bounded")) |>
  ggplot(aes(observed, pred, colour = version)) +
  geom_abline(slope = 1, intercept = 0, colour = INK, linetype = "22",
              linewidth = 0.3) +
  geom_hline(yintercept = max(art$observed), colour = ACCENT, linetype = "12",
             linewidth = 0.35) +
  annotate("text", x = 1.6, y = max(art$observed) * 1.35, hjust = 0, size = 2.2,
           colour = ACCENT, label = sprintf("observed maximum, %.1f m",
                                            max(art$observed))) +
  geom_point(size = 0.7, alpha = 0.5, shape = 16) +
  scale_colour_manual(values = c(Unbounded = "#9E9E9E", Bounded = "#2E6E8E"),
                      name = NULL) +
  scale_y_log10(breaks = c(1, 3, 10, 30, 100), labels = c(1, 3, 10, 30, 100)) +
  labs(x = "Measured Secchi (m)", y = "Predicted Secchi (m, log scale)") +
  theme(legend.position = "inside", legend.position.inside = c(1, 1),
        legend.justification = c(1, 1), legend.key.size = unit(6, "pt"))

fig <- (pa | pb | pc) / (pd | pe) +
  plot_layout(heights = c(1, 0.8)) + tags_abc()
save_fig(fig, "fig04_retrieval_and_benchmark", W2, 128)
cat("fig04 lista\n")
