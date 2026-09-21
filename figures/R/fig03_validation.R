# ============================================================================
# fig03_validation.R -- Diagnostico de la unidad de bloqueo
#
# (a) Los tres modelos nulos bajo cada diseno de bloqueo (matriz 3x3). La
#     diagonal colapsa por construccion; lo que importa es lo que sobrevive
#     fuera de ella.
# (b) R2 fuera de fold del Random Forest bajo cada diseno y bajo
#     extrapolacion a una zona no vista.
# ============================================================================

source(file.path(local({a <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(a)) dirname(normalizePath(sub("^--file=", "", a[1]))) else getwd()}),
  "theme_titicaca.R"))

grid <- read_tidy("null_models_grid.csv")
hier <- read_tidy("validation_hierarchy.csv")

# ------------------------------------------------------------------ (a) --
UNIT <- c(station = "Station", campaign_date = "Campaign date",
          year = "Year", zona = "Zone")
pa_df <- grid |>
  mutate(fold = factor(map_strict(blocked_by, UNIT, "bloqueo"),
                       levels = c("Station", "Campaign date", "Year")),
         null = factor(map_strict(null_model, UNIT, "nulo"),
                       levels = rev(c("Zone", "Station", "Campaign date"))),
         diag = (blocked_by == null_model) |
                (blocked_by == "year" & null_model == "campaign_date"),
         txt = sub("-", "−", sprintf("%.3f", R2)))

pa <- ggplot(pa_df, aes(fold, null, fill = R2)) +
  geom_tile(colour = "white", linewidth = 1.2) +
  geom_tile(data = filter(pa_df, diag), fill = NA, colour = INK,
            linewidth = 0.45, linetype = "22", width = 0.94, height = 0.94) +
  geom_text(aes(label = txt, colour = R2 > 0.2), size = 2.7, fontface = "bold") +
  scale_fill_gradient(low = "#F3F6F8", high = "#1F4E79", limits = c(-0.05, 0.32),
                      oob = squish, name = expression(italic(R)^2),
                      guide = guide_colourbar(barwidth = unit(4, "pt"),
                                              barheight = unit(40, "pt"))) +
  scale_colour_manual(values = c(`TRUE` = "white", `FALSE` = INK), guide = "none") +
  scale_x_discrete(position = "top", expand = c(0, 0)) +
  scale_y_discrete(expand = c(0, 0)) +
  labs(x = "Folds blocked by", y = "Null model\n(mean of)") +
  coord_fixed() +
  theme(axis.line = element_blank(), axis.ticks = element_blank(),
        panel.grid.major.y = element_blank(),
        axis.title.x.top = element_text(margin = margin(b = 4)))

# ------------------------------------------------------------------ (b) --
LAB <- c(random_kfold          = "Random K-fold",
         by_station            = "Station",
         by_campaign_date      = "Campaign date",
         by_year               = "Year",
         one_record_per_event  = "One record per event",
         `LAGO MENOR`          = "Lago Menor",
         `BAHIA PUNO`          = "Bahía de Puno",
         `LAGO MAYOR`          = "Lago Mayor")
GRP <- c(random_kfold = "Blocking design", by_station = "Blocking design",
         by_campaign_date = "Blocking design", by_year = "Blocking design",
         one_record_per_event = "Blocking design",
         `LAGO MENOR` = "Zone withheld", `BAHIA PUNO` = "Zone withheld",
         `LAGO MAYOR` = "Zone withheld")

pb_df <- hier |>
  filter(case %in% names(LAB)) |>
  mutate(name = map_strict(case, LAB, "caso"),
         grp = factor(map_strict(case, GRP, "grupo"),
                      levels = c("Blocking design", "Zone withheld")),
         primary = case == "by_campaign_date",
         name = fct_reorder(name, R2),
         txt = sub("-", "−", sprintf("%.3f", R2)))

pb <- ggplot(pb_df, aes(R2, name, colour = grp)) +
  geom_vline(xintercept = 0, colour = INK, linewidth = 0.3) +
  geom_segment(aes(x = 0, xend = R2, yend = name), linewidth = 0.8) +
  geom_point(aes(size = primary)) +
  geom_text(aes(label = txt, hjust = ifelse(R2 > 0.1, -0.35, ifelse(R2 > 0, -1.1, 1.35))),
            size = 2.4, colour = INK) +
  scale_colour_manual(values = c("Blocking design" = "#2E6E8E",
                                 "Zone withheld" = ACCENT), guide = "none") +
  scale_size_manual(values = c(`TRUE` = 2.8, `FALSE` = 1.7), guide = "none") +
  scale_x_continuous(limits = c(-0.2, 0.78), breaks = seq(0, 0.6, 0.2)) +
  facet_grid(grp ~ ., scales = "free_y", space = "free_y", switch = "y") +
  labs(x = expression("Out-of-fold "*italic(R)^2), y = NULL) +
  theme(strip.placement = "outside",
        strip.text.y.left = element_text(angle = 90, face = "bold", hjust = 0.5),
        panel.grid.major.y = element_blank(), axis.line.y = element_blank(),
        axis.ticks.y = element_blank())

fig <- (pa | pb) + plot_layout(widths = c(1, 1.25)) + tags_abc()
save_fig(fig, "fig03_validation_design", W2, 78)
cat("fig03 lista\n")
