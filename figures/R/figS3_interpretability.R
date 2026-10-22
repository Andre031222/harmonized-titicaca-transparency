# ============================================================================
# figS3_interpretability.R -- SHAP fuera de fold y recuperabilidad (p08)
#
# (a) Enjambre SHAP: contribucion de cada feature a cada prediccion fuera de
#     fold (leave-one-campaign-out), color = percentil del valor de la feature
# (b) Dependencia SHAP de la feature dominante (verde) por zona
# (c) R2 fuera de fold con IC 95 % (bootstrap de campanas) para Secchi,
#     clorofila-a, SST y temperatura del agua, mas la r de Pearson de la
#     temperatura de brillo B10
# ============================================================================

source(file.path(local({a <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(a)) dirname(normalizePath(sub("^--file=", "", a[1]))) else getwd()}),
  "theme_titicaca.R"))
suppressPackageStartupMessages({library(jsonlite); library(ggbeeswarm)})

imp  <- read_tidy("shap_importance.csv")
grp  <- read_tidy("shap_groups.csv")
dep  <- read_tidy("shap_dependence.csv")
retr <- read_tidy("retrievability.csv")
allv <- read_tidy("shap_values_all.csv")
ij   <- fromJSON(gsub("\\bNaN\\b", "null", paste(readLines(
  file.path(ROOT, "results/metrics/interpretability.json"), warn = FALSE),
  collapse = "\n")))

# la importancia exportada es la media de |SHAP| de los valores por fila
chk <- allv |> group_by(feature) |> summarise(m = mean(abs(shap)), .groups = "drop") |>
  inner_join(select(imp, feature, mean_abs_shap), by = "feature")
stopifnot(nrow(chk) == nrow(imp), all(abs(chk$m - chk$mean_abs_shap) < 1e-3),
          nrow(allv) == ij$n * nrow(imp),
          abs(sum(grp$share) - 1) < 1e-3)

# ---------------------------------------------------------------- panel (a) --
GROUP_SHORT <- setNames(c("Green/blue", "NIR/SWIR", "Other"), grp$group)
stopifnot(all(imp$group %in% names(GROUP_SHORT)))
PAL_GROUP <- c("Green/blue" = "#2E6E8E", "NIR/SWIR" = ACCENT, "Other" = INK_2)
lev <- imp |> arrange(mean_abs_shap) |> pull(label)
sw <- allv |>
  group_by(feature) |> mutate(pct = percent_rank(value)) |> ungroup() |>
  mutate(label = factor(label, levels = lev))
share_lab <- imp |> mutate(label = factor(label, levels = lev),
                           g = GROUP_SHORT[group],
                           txt = sprintf("%.0f%%", 100 * share))
x_hi <- max(sw$shap) + 0.3

# etiquetas del eje y coloreadas por grupo de bandas
lab_col <- PAL_GROUP[GROUP_SHORT[imp$group[match(lev, imp$label)]]]
grp_txt <- grp |> mutate(g = GROUP_SHORT[group],
                         txt = sprintf("%s %.0f%%", g, 100 * share))

pa <- ggplot(sw, aes(shap, label)) +
  geom_vline(xintercept = 0, colour = INK_2, linewidth = 0.3) +
  geom_quasirandom(aes(colour = pct), orientation = "y", size = 0.3,
                   alpha = 0.8, width = 0.38, method = "quasirandom") +
  geom_text(data = share_lab, aes(x = x_hi, y = label, label = txt),
            hjust = 0, size = 2.3, fontface = "bold", colour = INK_2) +
  annotate("text", x = x_hi, y = length(lev) + 0.75, label = "share",
           hjust = 0, size = 2.3, colour = INK_2, fontface = "italic") +
  scale_colour_gradientn(colours = c("#2166AC", "#9EC5E0", "#F2F2F2", "#F4A582",
                                     "#B2182B"),
                         name = "Feature\nvalue", breaks = c(0, 1),
                         labels = c("Low", "High"),
                         guide = guide_colourbar(barwidth = unit(4, "pt"),
                                                 barheight = unit(36, "pt"))) +
  scale_x_continuous(expand = expansion(mult = c(0.02, 0.12))) +
  scale_y_discrete(expand = expansion(add = c(0.6, 1.1))) +
  coord_cartesian(clip = "off") +
  labs(x = "SHAP contribution to predicted Secchi (m)", y = NULL) +
  theme(legend.position = "inside", legend.position.inside = c(0.02, 0.02),
        legend.justification = c(0, 0), legend.title = element_text(size = 6.6),
        legend.background = element_blank(),
        axis.text.y = element_text(colour = lab_col, size = 7),
        panel.grid.major.y = element_line(colour = "grey93", linewidth = 0.25),
        axis.line.y = element_blank(), axis.ticks.y = element_blank())

# leyenda de grupos como texto bajo el enjambre (color = color del eje y)
grp_lab <- grp_txt |> mutate(i = row_number())
pa <- pa + annotate("text", x = x_hi - 0.2, y = 0.75 + 0.6 * (rev(grp_lab$i) - 1),
                    label = grp_lab$txt, colour = PAL_GROUP[grp_lab$g],
                    hjust = 1, vjust = 0, size = 2.3, fontface = "bold")

# ---------------------------------------------------------------- panel (b) --
top_feat <- ij$leading_feature$feature
stopifnot(top_feat == imp$feature[1], all(dep$feature == top_feat))
# el suavizado se ajusta solo entre los percentiles 1 y 99: unos pocos
# valores extremos no deben dictar la forma de la curva; los puntos se
# dibujan todos
qlim <- quantile(dep$value, c(0.01, 0.99))
pb_df <- dep |> mutate(zl = zone_factor(zona),
                       core = value >= qlim[1] & value <= qlim[2])
n_out <- sum(!pb_df$core)
# cada curva de zona se limita a su propio rango 2-98 %: fuera de el hay
# muy pocos puntos y el loess extrapola una forma que los datos no muestran
sm_df <- pb_df |> filter(core) |> group_by(zl) |>
  filter(value >= quantile(value, 0.02), value <= quantile(value, 0.98)) |>
  ungroup()
# ventana del eje x: 1-99 % con un margen del 4 %. Los puntos fuera de ella no
# se dibujan (se informa cuantos); la expansion por defecto del eje deja sitio
# para que ningun punto del borde quede partido por el marco.
xwin <- qlim + c(-1, 1) * 0.04 * diff(qlim)
n_hidden <- sum(pb_df$value < xwin[1] | pb_df$value > xwin[2])
pb <- ggplot(pb_df, aes(value, shap)) +
  geom_hline(yintercept = 0, colour = INK, linewidth = 0.3, linetype = "22") +
  geom_point(data = filter(pb_df, between(value, xwin[1], xwin[2])),
             aes(fill = zl), shape = 21, colour = "white", stroke = 0.15,
             size = 1.1, alpha = 0.85) +
  geom_smooth(data = sm_df, aes(colour = zl), method = "loess",
              formula = y ~ x, se = FALSE, span = 0.5, linewidth = 0.7) +
  scale_fill_manual(values = PAL_ZONE, name = NULL) +
  scale_colour_manual(values = setNames(colorspace::darken(PAL_ZONE, 0.2),
                                        names(PAL_ZONE)), guide = "none") +
  coord_cartesian(xlim = xwin) +
  guides(fill = guide_legend(override.aes = list(size = 2.2, alpha = 1))) +
  labs(x = sprintf("%s surface reflectance (%s)", ij$leading_feature$label,
                   top_feat),
       y = "SHAP contribution (m)") +
  theme(legend.position = "inside", legend.position.inside = c(0.99, 0.99),
        legend.justification = c(1, 1), legend.key.size = unit(7, "pt"),
        legend.text = element_text(size = 6.6))

# ---------------------------------------------------------------- panel (c) --
VAR_LAB <- c(secchi = "Secchi depth",
             chl = "Chlorophyll-a",
             tss = "Suspended solids",
             temp_insitu = "Water temperature",
             temp_insitu_B10 = "Water temp., B10 only")
pc_df <- retr |>
  mutate(lab0 = map_strict(variable, VAR_LAB, "variable"),
         lab = sprintf("%s (n = %d)", lab0, n),
         lab = factor(lab, levels = lab[match(rev(VAR_LAB), lab0)]),
         is_secchi = variable == "secchi",
         txt = sprintf("%.2f [%.2f, %.2f]", R2, R2_lo, R2_hi))
stopifnot(nrow(pc_df) == length(VAR_LAB))
th <- ij$thermal
r_txt <- sprintf("r = %.2f [%.2f, %.2f]",
                 th$pearson_r, th$pearson_r_ci[1], th$pearson_r_ci[2])
X_LO <- min(pc_df$R2_lo)
X_T  <- max(pc_df$R2_hi) + 0.08
pc <- ggplot(pc_df, aes(R2, lab, colour = is_secchi)) +
  geom_vline(xintercept = 0, colour = INK_2, linewidth = 0.3) +
  geom_errorbar(aes(xmin = R2_lo, xmax = R2_hi), width = 0, linewidth = 0.6,
                orientation = "y") +
  geom_point(size = 2) +
  geom_text(aes(x = X_T, label = txt), hjust = 0, size = 2.3, colour = INK_2) +
  annotate("text", x = X_T, y = 1 - 0.36, label = r_txt, hjust = 0, size = 2.3,
           colour = INK_2, fontface = "italic") +
  scale_colour_manual(values = c(`TRUE` = "#3D7A57", `FALSE` = NEUTRAL),
                      guide = "none") +
  scale_x_continuous(breaks = seq(-1.5, 0.5, 0.5)) +
  coord_cartesian(xlim = c(X_LO, X_T + 0.95), clip = "off") +
  labs(x = expression("Out-of-fold "*italic(R)^2*" (leave-one-campaign-out)"),
       y = NULL) +
  theme(panel.grid.major.y = element_blank(), axis.line.y = element_blank(),
        axis.ticks.y = element_blank(), axis.text.y = element_text(size = 7))

# ---------------------------------------------------------------- montaje --
fig <- (pa | (pb / pc + plot_layout(heights = c(1.35, 1)))) +
  plot_layout(widths = c(1, 1.1)) + tags_abc()
save_fig(fig, "figS3_shap_retrievability", W2, 140)
cat(sprintf("  panel b: %d de %d puntos fuera de la ventana del eje x (no dibujados)\n",
            n_hidden, nrow(pb_df)))
cat(sprintf("figS3 lista (%d valores de %s fuera del 1-99%% sin suavizar)\n",
            n_out, top_feat))
