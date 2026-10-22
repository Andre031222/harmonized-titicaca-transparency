# ============================================================================
# fig4_validation.R -- Diseno de validacion y nulos jerarquicos (HLS)
#
# (a) Matriz nulo x diseno: R2 fuera de fold de los cuatro nulos jerarquicos
#     (zona; zona + localidad; zona + fecha de muestreo; zona + campana) bajo
#     seis disenos de validacion, de menos a mas exigente. Se marcan las
#     celdas donde el diseno retira el grupo del nulo y este se reduce a la
#     media de zona.
# (b) R2 del Random Forest fuera de fold con IC 95 % (bootstrap por campana)
#     por diseno, junto al mejor nulo que sobrevive en ese diseno.
# (c) Descomposicion de varianza dentro de muestra (R2 anadido por paso).
# (d) Pruebas complementarias: zona excluida, prediccion temporal y
#     subconjuntos bajo el diseno primario.
#
# Entradas: results/tidy/null_models_grid.csv, validation_hierarchy.csv,
# variance_decomposition.csv. Ningun numero escrito a mano.
# ============================================================================

source(file.path(local({a <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(a)) dirname(normalizePath(sub("^--file=", "", a[1]))) else getwd()}),
  "theme_titicaca.R"))

grid <- read_tidy("null_models_grid.csv")
hier <- read_tidy("validation_hierarchy.csv")
vdec <- read_tidy("variance_decomposition.csv")

# orden de exigencia creciente, tal como lo escribe el pipeline
DESIGNS <- unique(grid$design)
NULLS   <- unique(grid$null_model)
PRIMARY <- "campaign"
stopifnot(PRIMARY %in% DESIGNS, length(DESIGNS) == 6, length(NULLS) == 4)

des_lab <- grid |> distinct(design, design_label, n_folds) |>
  mutate(short = case_when(
           design == "random"         ~ "Random\nK-fold",
           design == "location"       ~ "Location",
           design == "sampling_date"  ~ "Sampling\ndate",
           design == "spatial_block"  ~ "Spatial\nblock",
           design == "campaign"       ~ "Campaign\n(primary)",
           design == "campaign_block" ~ "Campaign ×\nspatial block",
           TRUE ~ NA_character_),
         lab = sprintf("%s\n%d folds", short, n_folds))
stopifnot(!anyNA(des_lab$short))
DLAB <- setNames(des_lab$lab, des_lab$design)

null_lab <- c("zone"               = "Zone",
              "zone+location"      = "Zone + location",
              "zone+sampling_date" = "Zone + sampling date",
              "zone+campaign"      = "Zone + campaign")

# ------------------------------------------------------------------ (a) --
# Un nulo "se retira" cuando su anomalia de grupo no puede estimarse para las
# filas de test: su R2 coincide entonces con el de la media de zona.
zone_r2 <- grid |> filter(null_model == "zone") |> select(design, R2_zone = R2)
ga <- grid |>
  left_join(zone_r2, by = "design") |>
  mutate(collapsed = null_model != "zone" & abs(R2 - R2_zone) < 1e-9,
         design = factor(design, levels = DESIGNS),
         null = factor(map_strict(null_model, null_lab, "nulo"),
                       levels = rev(null_lab[NULLS])),
         txt = sprintf("%.2f", R2),
         txt_col = ifelse(R2 > 0.42, "white", INK))

pa <- ggplot(ga, aes(design, null)) +
  geom_tile(aes(fill = R2), colour = "white", linewidth = 0.8) +
  geom_tile(data = filter(ga, collapsed), fill = NA, colour = INK_2,
            linewidth = 0.35, linetype = "22", width = 0.9, height = 0.84) +
  geom_text(aes(label = txt, colour = txt_col,
                fontface = ifelse(collapsed, "plain", "bold")), size = 2.6) +
  scale_colour_identity() +
  scale_fill_gradient(low = "#F1F4F6", high = "#2E6E8E",
                      limits = c(0, max(ga$R2)),
                      name = expression("Null out-of-fold "*italic(R)^2),
                      guide = guide_colourbar(barheight = unit(4, "pt"),
                                              barwidth = unit(60, "pt"),
                                              title.position = "top")) +
  scale_x_discrete(labels = DLAB, position = "top") +
  labs(x = NULL, y = NULL) +
  theme(axis.line = element_blank(), axis.ticks = element_blank(),
        panel.grid.major.y = element_blank(),
        axis.text.x.top = element_text(lineheight = 0.95, size = 7,
                                       margin = margin(b = 2)),
        axis.text.y = element_text(size = 7.2, colour = INK),
        legend.position = "none")

# ------------------------------------------------------------------ (b) --
null_lab2 <- c("zone" = "Zone", "zone+location" = "Zone +\nlocation",
               "zone+sampling_date" = "Zone +\nsampl. date",
               "zone+campaign" = "Zone +\ncampaign")
hb <- hier |> filter(setting == "design") |>
  mutate(design = factor(design, levels = DESIGNS),
         x = as.numeric(design),
         primary = design == PRIMARY,
         best_null_txt = map_strict(best_null, null_lab2, "nulo"))
stopifnot(nrow(hb) == length(DESIGNS))
y_lo <- min(hb$best_null_R2) - 0.06
y_hi <- max(hb$R2_hi) + 0.06

pb <- ggplot(hb) +
  geom_rect(data = filter(hb, primary),
            aes(xmin = x - 0.48, xmax = x + 0.48, ymin = -Inf, ymax = Inf),
            fill = "#F6E7DF", colour = NA) +
  # mejor nulo que sobrevive al diseno: raya horizontal con su nombre
  geom_segment(aes(x = x - 0.24, xend = x + 0.24, y = best_null_R2,
                   yend = best_null_R2), colour = INK_2, linewidth = 0.9) +
  geom_text(aes(x = x + 0.27, y = best_null_R2, label = best_null_txt),
            hjust = 0, size = 2.3, colour = INK_2, lineheight = 0.85) +
  geom_linerange(aes(x = x, ymin = R2_lo, ymax = R2_hi, colour = primary),
                 linewidth = 0.7) +
  geom_point(aes(x = x, y = R2, colour = primary), size = 2.3) +
  geom_text(aes(x = x, y = R2_hi, label = sprintf("%.3f", R2),
                fontface = ifelse(primary, "bold", "plain")),
            vjust = -0.6, size = 2.4, colour = INK) +
  # clave
  annotate("point", x = 0.62, y = y_lo + 0.075, size = 2.3, colour = "#2E6E8E") +
  annotate("segment", x = 0.62, xend = 0.62, y = y_lo + 0.06, yend = y_lo + 0.09,
           colour = "#2E6E8E", linewidth = 0.7) +
  annotate("text", x = 0.72, y = y_lo + 0.075, hjust = 0, size = 2.3, colour = INK,
           label = "Random Forest, 95% campaign-bootstrap CI") +
  annotate("segment", x = 0.54, xend = 0.70, y = y_lo + 0.03,
           yend = y_lo + 0.03, colour = INK_2, linewidth = 0.9) +
  annotate("text", x = 0.72, y = y_lo + 0.03, hjust = 0, size = 2.3, colour = INK,
           label = "Best null that survives the design") +
  scale_colour_manual(values = c(`TRUE` = ACCENT, `FALSE` = "#2E6E8E"),
                      guide = "none") +
  scale_x_continuous(breaks = seq_along(DESIGNS), labels = DLAB[DESIGNS],
                     limits = c(0.5, length(DESIGNS) + 0.5),
                     expand = expansion(add = 0)) +
  scale_y_continuous(limits = c(y_lo, y_hi), breaks = seq(0, 1, 0.1),
                     expand = expansion(add = 0)) +
  labs(x = NULL, y = expression("Out-of-fold "*italic(R)^2)) +
  theme(axis.text.x = element_text(lineheight = 0.95, size = 7, colour = INK_2))

# ------------------------------------------------------------------ (c) --
vc <- vdec |>
  mutate(i = row_number(),
         start = lag(R2_cumulative, default = 0),
         lab = map_strict(step, c("zone" = "Zone",
                                  "+ campaign" = "+ Campaign",
                                  "+ zone x campaign" = "+ Zone × campaign",
                                  "+ location" = "+ Location"), "paso"),
         yy = rev(i))
pc <- ggplot(vc, aes(y = yy)) +
  geom_segment(data = vc |> filter(i < max(i)),
               aes(x = R2_cumulative, xend = R2_cumulative,
                   y = yy - 0.3, yend = yy - 0.7),
               colour = NEUTRAL, linewidth = 0.3, linetype = "22") +
  geom_rect(aes(xmin = start, xmax = R2_cumulative,
                ymin = yy - 0.3, ymax = yy + 0.3,
                fill = i == max(i))) +
  geom_text(aes(x = R2_cumulative, label = sprintf("+%.3f", R2_added)),
            hjust = -0.15, size = 2.4, colour = INK) +
  scale_fill_manual(values = c(`FALSE` = "#8FB4C7", `TRUE` = "#BDBDBD"),
                    guide = "none") +
  scale_x_continuous(limits = c(0, 1.12), breaks = c(0, 0.5, 1),
                     labels = c("0", "0.5", "1"),
                     expand = expansion(mult = c(0, 0))) +
  scale_y_continuous(breaks = vc$yy, labels = vc$lab,
                     expand = expansion(add = 0.5)) +
  labs(x = expression("Cumulative in-sample "*italic(R)^2), y = NULL) +
  theme(panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_line(colour = "#EDEDED", linewidth = 0.25),
        axis.text.y = element_text(colour = INK))

# ------------------------------------------------------------------ (d) --
# Encabezados de grupo como filas del eje (en negrita via ggtext): facet_grid
# con tiras a la derecha robaba un tercio del ancho.
prim <- hb |> filter(primary)
GROUPS <- c(zone_out = "Zone held out (training on the other two)",
            temporal = "Temporal hold-out",
            subset   = "Primary design, subsets")
hd <- hier |> filter(setting != "design") |>
  mutate(group = map_strict(setting, GROUPS, "setting"),
         lab = ifelse(setting == "zone_out",
                      paste("Leave out", unname(ZONE_NAME[design])), label),
         lab = str_replace(lab, "<=", "≤"),
         lab = sprintf("%s (n = %d)", lab, n))
stopifnot(!anyNA(hd$lab), !any(grepl("NA", hd$lab)))
rows <- bind_rows(lapply(unname(GROUPS), function(g) {
  bind_rows(tibble(lab = paste0("**", g, "**"), head = TRUE),
            hd |> filter(group == g) |> select(lab) |> mutate(head = FALSE))
})) |> mutate(yy = rev(seq_len(n())))
hd <- hd |> left_join(rows |> filter(!head) |> select(lab, yy), by = "lab")

pd <- ggplot(hd, aes(y = yy)) +
  annotate("rect", xmin = prim$R2_lo, xmax = prim$R2_hi, ymin = -Inf,
           ymax = Inf, fill = "#F6E7DF") +
  geom_vline(xintercept = prim$R2, colour = ACCENT, linewidth = 0.4,
             linetype = "22") +
  geom_vline(xintercept = 0, colour = INK, linewidth = 0.25) +
  annotate("label", x = prim$R2, y = max(rows$yy), label = "Primary design",
           size = 2.3, colour = ACCENT, fill = "#F6E7DF", linewidth = 0,
           label.padding = unit(1, "pt")) +
  geom_linerange(aes(xmin = R2_lo, xmax = R2_hi), colour = "#2E6E8E",
                 linewidth = 0.6, na.rm = TRUE) +
  geom_point(aes(x = R2), size = 1.8, colour = "#2E6E8E") +
  scale_y_continuous(breaks = rows$yy, labels = rows$lab,
                     limits = c(0.5, max(rows$yy) + 0.5),
                     expand = expansion(add = 0)) +
  scale_x_continuous(breaks = seq(0, 0.75, 0.25),
                     labels = c("0", "0.25", "0.5", "0.75"),
                     limits = c(min(-0.15, min(hd$R2) - 0.03),
                                max(0.62, max(hd$R2_hi, na.rm = TRUE) + 0.02))) +
  labs(x = expression("Out-of-fold "*italic(R)^2), y = NULL) +
  theme(panel.grid.major.y = element_blank(),
        axis.ticks.y = element_blank(),
        axis.text.y = ggtext::element_markdown(size = 6.6, colour = INK))

# ------------------------------------------------------------- montaje --
fig <- (pa / pb / ((pc | pd) + plot_layout(widths = c(0.7, 1.3)))) +
  plot_layout(heights = c(0.8, 0.95, 1.05)) + tags_abc()

save_fig(fig, "fig4_validation_design", W2, 180)
cat("fig4 lista\n")
