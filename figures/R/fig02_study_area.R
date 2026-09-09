# ============================================================================
# fig02_study_area.R -- Area de estudio y red de monitoreo
#
# (a) Mapa del lago con las estaciones IMARPE/ALT coloreadas por transparencia
#     media medida, sobre el limite de Natural Earth
# (b) Esfuerzo de muestreo por estacion: cuantas veces se visito cada una
# ============================================================================

source(file.path(local({a <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(a)) dirname(normalizePath(sub("^--file=", "", a[1]))) else getwd()}),
  "theme_titicaca.R"))

suppressPackageStartupMessages({library(sf); library(viridis)})

dist <- read_tidy("insitu_distribution.csv")
lake <- st_read(file.path(ROOT, "data/lake_boundary/titicaca.gpkg"), quiet = TRUE)

ZMAP <- c("BAHIA PUNO" = "Puno Bay", "LAGO MENOR" = "Minor Lake",
          "LAGO MAYOR" = "Major Lake")
PAL2 <- setNames(unname(PAL_ZONE), c("Puno Bay", "Minor Lake", "Major Lake"))

st_summary <- dist |>
  group_by(station, zona, lat, lon) |>
  summarise(secchi_mean = mean(secchi), n_visits = n(),
            n_campaigns = n_distinct(campaign_date), .groups = "drop") |>
  mutate(zone_label = factor(ZMAP[zona], levels = names(PAL2)))

bb <- st_bbox(lake)

# ---------------------------------------------------------------- panel (a) --
pa <- ggplot() +
  geom_sf(data = lake, fill = "#EAF1F5", colour = "#9FB8C6", linewidth = 0.3) +
  geom_point(data = st_summary,
             aes(lon, lat, fill = secchi_mean, size = n_visits),
             shape = 21, colour = "white", stroke = 0.3, alpha = 0.95) +
  scale_fill_viridis_c(option = "mako", direction = -1, name = "Secchi\nmedio (m)",
                       guide = guide_colourbar(barwidth = unit(5, "pt"),
                                               barheight = unit(34, "pt"))) +
  scale_size_continuous(range = c(0.9, 3.6), name = "Match-ups",
                        breaks = c(2, 6, 12)) +
  coord_sf(xlim = c(bb["xmin"], bb["xmax"]),
           ylim = c(bb["ymin"], bb["ymax"]), expand = FALSE) +
  annotate("text", x = -69.95, y = -15.83, label = "Puno\nBay",
           size = 2.4, colour = INK, fontface = "bold", lineheight = 1,
           hjust = 0.5) +
  annotate("text", x = -69.35, y = -15.85, label = "Major Lake",
           size = 2.4, colour = INK, fontface = "bold") +
  annotate("text", x = -68.95, y = -16.30, label = "Minor Lake\n(Wiñaymarca)",
           size = 2.4, colour = INK, fontface = "bold", lineheight = 1) +
  labs(title = "(a)  Limnological monitoring network (IMARPE)",
       subtitle = paste("IMARPE/ALT stations with satellite-field match-ups",
                        "(2013–2024), coloreadas por la\ntransparencia media",
                        "measured. The trophic gradient is visible in the",
                        "field data itself"),
       x = NULL, y = NULL) +
  theme(legend.position = "right", legend.box = "vertical",
        panel.grid.major = element_line(colour = GRID, linewidth = 0.2),
        axis.text = element_text(size = rel(0.78)))

# ---------------------------------------------------------------- panel (b) --
pb <- ggplot(st_summary, aes(n_campaigns, secchi_mean, colour = zone_label)) +
  geom_point(aes(size = n_visits), alpha = 0.6) +
  scale_colour_manual(values = PAL2, name = NULL) +
  scale_size_continuous(range = c(0.9, 3.6), guide = "none") +
  scale_x_continuous(breaks = scales::pretty_breaks(5)) +
  labs(title = "(b)  El esfuerzo de muestreo es muy desigual",
       subtitle = paste("Each point is a station.", nrow(st_summary),
                        "stations, but most were visited\nin very few",
                        "campaigns, limiting the power of any",
                        "station-based analysis"),
       x = "Campaigns where the station was visited",
       y = "Secchi medio medido (m)") +
  theme(legend.position = "bottom", legend.margin = margin(t = -4))

fig <- pa / pb + plot_layout(heights = c(1.55, 1))
save_fig(fig, "fig02_study_area", W15, 185)
cat("fig02 lista\n")
