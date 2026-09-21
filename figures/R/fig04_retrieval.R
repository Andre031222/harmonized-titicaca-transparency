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
pw <- pairwise.wilcox.test(oof$residual, oof$zone_label, p.adjust.method = "BH")$p.value
lv <- levels(oof$zone_label)
brk <- tibble(g1 = c(lv[1], lv[2], lv[1]), g2 = c(lv[2], lv[3], lv[3]),
              y = c(6.6, 8.1, 9.6)) |>
  mutate(p = unname(mapply(\(a, b) {v <- pw[b, a]; if (is.na(v)) pw[a, b] else v},
                           g1, g2)),
         x1 = match(g1, lv), x2 = match(g2, lv),
         lab = vapply(p, p_fmt, ""))
stopifnot(!any(is.na(brk$p)))

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
  annotate("text", x = 0.5, y = 11.6, hjust = 0, size = 2.2, colour = INK_2,
           parse = TRUE, label = paste0('"Kruskal–Wallis, "*', p_fmt(kw))) +
  scale_colour_manual(values = PAL_ZONE, guide = "none") +
  scale_x_discrete(labels = zone_x) +
  scale_y_continuous(limits = c(-9.8, 12), breaks = seq(-6, 6, 3)) +
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
# el ratio sin acotar (R2 = -2.46) aplastaba el eje: queda como flecha fuera
# de escala y el panel muestra los tres modelos comparables
pd_df <- bench |>
  mutate(name = case_when(
    family == "artefact"  ~ "Blue/green ratio,\nunbounded",
    family == "classical" ~ "Blue/green ratio,\nbounded",
    family == "linear"    ~ "Multiband linear",
    TRUE                  ~ "Random Forest"),
    kind = ifelse(family == "artefact", "a", ifelse(family == "rf", "rf", "b")),
    name = fct_reorder(name, R2),
    shown = pmax(R2, 0))
art_row <- filter(pd_df, kind == "a")

pd <- ggplot(pd_df, aes(y = name)) +
  geom_vline(xintercept = 0, colour = INK, linewidth = 0.3) +
  geom_segment(data = filter(pd_df, kind != "a"),
               aes(x = 0, xend = R2, colour = kind), linewidth = 1.1) +
  geom_point(data = filter(pd_df, kind != "a"), aes(x = R2, colour = kind),
             size = 3) +
  geom_text(data = filter(pd_df, kind != "a"),
            aes(x = R2, label = sprintf("%.3f", R2)), hjust = -0.45, size = 2.4,
            fontface = "bold", colour = INK) +
  geom_text(data = pd_df, aes(x = 0.86, label = sprintf("%.2f m", RMSE)),
            hjust = 1, size = 2.3, colour = INK_2) +
  annotate("text", x = 0.86, y = 4.55, label = "RMSE", hjust = 1, size = 2.2,
           colour = INK_2, fontface = "italic") +
  geom_segment(data = art_row, aes(x = 0.3, xend = 0.02, y = name, yend = name),
               colour = "#9E9E9E", linewidth = 0.6,
               arrow = arrow(length = unit(4, "pt"), type = "closed")) +
  geom_text(data = art_row, aes(x = 0.32, label = sprintf("R² = %.2f, off scale", R2)),
            hjust = 0, size = 2.3, colour = "#7A7A7A", fontface = "italic") +
  scale_colour_manual(values = c(b = "#8FB4C7", rf = "#2E6E8E"), guide = "none") +
  scale_x_continuous(limits = c(-0.02, 0.88), breaks = seq(0, 0.6, 0.2),
                     expand = expansion(mult = c(0, 0))) +
  scale_y_discrete(limits = levels(pd_df$name),
                   expand = expansion(add = c(0.6, 0.9))) +
  labs(x = expression("Out-of-fold "*italic(R)^2), y = NULL) +
  theme(panel.grid.major.y = element_blank(), axis.line.y = element_blank(),
        axis.ticks.y = element_blank())

# ------------------------------------------------------------------ (e) --
pe_df <- art |>
  select(observed, predicted_unbounded, predicted_bounded) |>
  pivot_longer(-observed, names_to = "version", values_to = "pred") |>
  mutate(version = recode(version, predicted_unbounded = "Unbounded",
                          predicted_bounded = "Bounded"))
worst <- art |> slice_max(predicted_unbounded, n = 1)
one_one <- tibble(x = seq(1, 17, 0.1))

pe <- ggplot(pe_df, aes(observed, pred)) +
  geom_line(data = one_one, aes(x, x), colour = INK, linetype = "22",
            linewidth = 0.3) +
  geom_hline(yintercept = max(art$observed), colour = ACCENT, linetype = "12",
             linewidth = 0.35) +
  annotate("text", x = 17, y = max(art$observed) * 1.3, hjust = 1, size = 2.2,
           colour = ACCENT, label = sprintf("observed maximum, %.1f m",
                                            max(art$observed))) +
  geom_point(aes(colour = version, fill = version, shape = version), size = 1,
             alpha = 0.6, stroke = 0.35) +
  annotate("segment", x = worst$observed + 3, xend = worst$observed + 0.35,
           y = worst$predicted_unbounded, yend = worst$predicted_unbounded,
           colour = INK, linewidth = 0.4,
           arrow = arrow(length = unit(4, "pt"), type = "closed")) +
  annotate("text", x = worst$observed + 3.2, y = worst$predicted_unbounded,
           hjust = 0, size = 2.3, colour = INK, lineheight = 0.95,
           label = sprintf("%.1f m predicted for a\n%.1f m measurement",
                           worst$predicted_unbounded, worst$observed)) +
  scale_colour_manual(values = c(Unbounded = "#8A8A8A", Bounded = "#2E6E8E"),
                      name = NULL) +
  scale_fill_manual(values = c(Unbounded = NA, Bounded = "#2E6E8E"), name = NULL) +
  scale_shape_manual(values = c(Unbounded = 21, Bounded = 21), name = NULL) +
  scale_y_log10(breaks = c(0.1, 1, 3, 10, 30, 100),
                labels = c("0.1", "1", "3", "10", "30", "100")) +
  coord_cartesian(xlim = c(1, 17), ylim = c(0.04, 250)) +
  labs(x = "Measured Secchi (m)", y = "Predicted Secchi (m, log scale)") +
  guides(colour = guide_legend(override.aes = list(size = 2, alpha = 1))) +
  theme(legend.position = "inside", legend.position.inside = c(0.02, 0.02),
        legend.justification = c(0, 0), legend.key.size = unit(7, "pt"),
        legend.background = element_blank())

fig <- (pa | pb | pc) / (pd | pe) +
  plot_layout(heights = c(1, 0.78)) + tags_abc()
save_fig(fig, "fig04_retrieval_and_benchmark", W2, 126)
cat("fig04 lista\n")
