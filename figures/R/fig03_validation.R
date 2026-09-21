# ============================================================================
# fig03_validation.R -- Diagnostico de la unidad de bloqueo
#
# (a) Los tres modelos nulos bajo cada diseno de bloqueo (matriz 3x3, R2 y
#     RMSE). La diagonal colapsa por construccion; lo que importa es lo que
#     sobrevive fuera de ella.
# (b) Habilidad por encima del nulo: del mejor nulo que sobrevive a cada
#     diseno al Random Forest bajo ese mismo diseno.
# (c) R2 fuera de fold bajo cada diseno, con una zona retenida y con cada
#     conjunto de features o sensor.
# ============================================================================

source(file.path(local({a <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(a)) dirname(normalizePath(sub("^--file=", "", a[1]))) else getwd()}),
  "theme_titicaca.R"))

grid <- read_tidy("null_models_grid.csv")
hier <- read_tidy("validation_hierarchy.csv")

UNIT <- c(station = "Station", campaign_date = "Campaign date",
          year = "Year", zona = "Zone")
BLUE <- "#2E6E8E"

# ------------------------------------------------------------------ (a) --
pa_df <- grid |>
  mutate(fold = factor(map_strict(blocked_by, UNIT, "bloqueo"),
                       levels = c("Station", "Campaign date", "Year")),
         null = factor(map_strict(null_model, UNIT, "nulo"),
                       levels = rev(c("Zone", "Station", "Campaign date"))),
         # la campana esta anidada en el anio: bloquear por anio tambien la retiene
         diag = (blocked_by == null_model) |
                (blocked_by == "year" & null_model == "campaign_date"),
         txt = sub("-", "−", sprintf("%.3f", R2)),
         sub = sprintf("RMSE %.2f m", RMSE))

pa <- ggplot(pa_df, aes(fold, null, fill = R2)) +
  geom_tile(colour = "white", linewidth = 1.2) +
  geom_tile(data = filter(pa_df, diag), fill = NA, colour = INK,
            linewidth = 0.45, linetype = "22", width = 0.94, height = 0.94) +
  geom_text(aes(label = txt, colour = R2 > 0.2), size = 2.9, fontface = "bold",
            vjust = -0.15) +
  geom_text(aes(label = sub, colour = R2 > 0.2), size = 1.95, vjust = 1.6) +
  geom_text(data = filter(pa_df, diag), aes(label = "collapses"), size = 1.9,
            vjust = 4.1, colour = INK_2, fontface = "italic") +
  scale_fill_gradient(low = "#F3F6F8", high = "#1F4E79", limits = c(-0.05, 0.32),
                      oob = squish, name = expression(italic(R)^2),
                      guide = guide_colourbar(barwidth = unit(4, "pt"),
                                              barheight = unit(40, "pt"))) +
  scale_colour_manual(values = c(`TRUE` = "white", `FALSE` = INK), guide = "none") +
  scale_x_discrete(position = "top", expand = c(0, 0)) +
  scale_y_discrete(expand = c(0, 0)) +
  labs(x = "Folds blocked by", y = "Null model (mean of)") +
  coord_fixed() +
  theme(axis.line = element_blank(), axis.ticks = element_blank(),
        panel.grid.major.y = element_blank(),
        axis.title.x.top = element_text(margin = margin(b = 4)))

# ------------------------------------------------------------------ (b) --
# habilidad real = lo que el modelo gana sobre el mejor nulo que el diseno no
# consigue anular
CASE_OF <- c(station = "by_station", campaign_date = "by_campaign_date",
             year = "by_year")
pb_df <- pa_df |>
  filter(!diag) |>
  group_by(blocked_by, fold) |>
  slice_max(R2, n = 1, with_ties = FALSE) |>
  ungroup() |>
  transmute(fold, null_R2 = R2, null_name = as.character(null),
            case = map_strict(blocked_by, CASE_OF, "caso")) |>
  left_join(hier |> select(case, rf_R2 = R2), by = "case") |>
  mutate(gain = rf_R2 - null_R2, primary = case == "by_campaign_date",
         fold = factor(fold, levels = rev(levels(pa_df$fold))))
stopifnot(nrow(pb_df) == 3, !any(is.na(pb_df$rf_R2)))

pb <- ggplot(pb_df, aes(y = fold)) +
  geom_segment(aes(x = null_R2 + 0.012, xend = rf_R2 - 0.014, yend = fold),
               colour = BLUE, linewidth = 0.7,
               arrow = arrow(length = unit(4.5, "pt"), type = "closed")) +
  geom_point(aes(x = null_R2), shape = 21, fill = "white", colour = NEUTRAL,
             size = 2.6, stroke = 0.8) +
  geom_point(aes(x = rf_R2, size = primary), colour = BLUE) +
  geom_text(aes(x = (null_R2 + rf_R2) / 2, label = sprintf("+%.3f", gain)),
            vjust = -0.9, size = 2.5, colour = BLUE, fontface = "bold") +
  geom_text(aes(x = null_R2, label = paste0("mean of\n", tolower(null_name))),
            vjust = 1.6, size = 1.95, colour = INK_2, lineheight = 0.9) +
  geom_text(aes(x = rf_R2, label = "Random\nForest"), vjust = 1.6, size = 1.95,
            colour = INK_2, lineheight = 0.9) +
  scale_size_manual(values = c(`TRUE` = 3.4, `FALSE` = 2.6), guide = "none") +
  scale_x_continuous(limits = c(0.15, 0.68), breaks = seq(0.2, 0.6, 0.1)) +
  scale_y_discrete(expand = expansion(add = c(0.7, 0.55))) +
  labs(x = expression("Out-of-fold "*italic(R)^2), y = "Folds blocked by") +
  theme(panel.grid.major.y = element_blank())

# ------------------------------------------------------------------ (c) --
LAB <- c(random_kfold          = "Random K-fold",
         by_station            = "Station",
         by_campaign_date      = "Campaign date",
         by_year               = "Year",
         one_record_per_event  = "One record per event",
         `LAGO MENOR`          = "Lago Menor",
         `BAHIA PUNO`          = "Bahía de Puno",
         `LAGO MAYOR`          = "Lago Mayor",
         roy_all               = "Roy, all 12 features",
         visible_only          = "Visible bands only",
         ratios_only           = "Band ratios only",
         local_reharm          = "Local water recalibration",
         S2_only               = "Sentinel-2 only",
         LS_only               = "Landsat 8/9 only")
GRP <- c(blocking = "Blocking design", leave_one_zone_out = "Zone withheld",
         harmonization = "Features and sensor")
NOTE <- c(by_station = "143 stations", by_campaign_date = "120 campaigns",
          by_year = "9 years", one_record_per_event = "n = 540",
          S2_only = "n = 274", LS_only = "n = 538")

pc_df <- hier |>
  filter(case %in% names(LAB)) |>
  mutate(name = map_strict(case, LAB, "caso"),
         grp = factor(map_strict(experiment, GRP, "grupo"), levels = unname(GRP)),
         primary = case == "by_campaign_date",
         note = unname(NOTE[case]),
         txt = sub("-", "−", sprintf("%.3f", R2)),
         rm = sprintf("%.2f m", RMSE)) |>
  group_by(grp) |> mutate(name = fct_reorder(name, R2)) |> ungroup()
stopifnot(nrow(pc_df) == length(LAB))

PAL_GRP <- c("Blocking design" = BLUE, "Zone withheld" = ACCENT,
             "Features and sensor" = "#6B8E5A")
pc <- ggplot(pc_df, aes(R2, name, colour = grp)) +
  geom_vline(xintercept = 0, colour = INK, linewidth = 0.3) +
  geom_segment(aes(x = 0, xend = R2, yend = name), linewidth = 0.9) +
  geom_point(aes(size = primary)) +
  geom_text(aes(x = pmax(R2, 0), label = txt), hjust = -0.35, size = 2.35,
            colour = INK, fontface = "bold") +
  geom_text(aes(x = 1.02, label = rm), hjust = 1, size = 2.2, colour = INK_2) +
  geom_text(aes(x = 1.06, label = note), hjust = 0, size = 2, colour = INK_2,
            fontface = "italic", na.rm = TRUE) +
  # cabecera de la columna, solo en el primer grupo
  geom_text(data = tibble(grp = factor(GRP[1], levels = unname(GRP)),
                          name = levels(pc_df$name)[1]),
            aes(x = 1.02, y = Inf, label = "RMSE"), inherit.aes = FALSE,
            hjust = 1, vjust = -0.6, size = 2.2, colour = INK_2,
            fontface = "italic") +
  facet_grid(grp ~ ., scales = "free_y", space = "free_y", switch = "y") +
  scale_colour_manual(values = PAL_GRP, guide = "none") +
  scale_size_manual(values = c(`TRUE` = 3.2, `FALSE` = 2), guide = "none") +
  scale_x_continuous(limits = c(-0.08, 1.42), breaks = seq(0, 0.6, 0.2),
                     expand = expansion(mult = c(0, 0))) +
  coord_cartesian(clip = "off") +
  labs(x = expression("Out-of-fold "*italic(R)^2), y = NULL) +
  theme(strip.placement = "outside",
        strip.text.y.left = element_text(angle = 90, face = "bold", hjust = 0.5,
                                         size = 7),
        panel.grid.major.y = element_blank(), axis.line.y = element_blank(),
        axis.ticks.y = element_blank(), panel.spacing.y = unit(6, "pt"))

design <- "
AC
BC
"
fig <- wrap_plots(A = pa, B = pb, C = pc, design = design) +
  plot_layout(widths = c(1, 1.15), heights = c(1.1, 0.9)) + tags_abc()
save_fig(fig, "fig03_validation_design", W2, 125)
cat("fig03 lista\n")
