# ============================================================================
# fig05_data_inventory.R -- Que hay REALMENTE en el dataset
#
# El manuscrito anterior describia "2013-2024" y un hold-out "2022-2024" que
# incluia "El Nino 2023-2024". No hay campanas en 2020, 2021 ni 2023, y todo
# el dataset es epoca seca. Esta figura hace esos huecos imposibles de ignorar.
#
# (a) matriz anio x mes de match-ups: los huecos a la vista
# (b) cobertura estacional frente al ciclo hidrologico del Altiplano
# (c) distribucion del Secchi por zona, con la granularidad de la medida
# ============================================================================

source(file.path(local({a <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(a)) dirname(normalizePath(sub("^--file=", "", a[1]))) else getwd()}),
  "theme_titicaca.R"))

dist <- read_tidy("insitu_distribution.csv")
seas <- read_tidy("seasonality.csv")

MES <- c("Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec")
ZORD <- c("Bahia de Puno", "Minor Lake", "Major Lake")
ZMAP <- c("BAHIA PUNO" = "Bahia de Puno", "LAGO MENOR" = "Minor Lake",
          "LAGO MAYOR" = "Major Lake")

# ---------------------------------------------------------------- panel (a) --
grid_full <- expand.grid(year = seq(min(dist$year), max(dist$year)),
                         month = 1:12)
cov <- dist |>
  count(year, month, name = "n") |>
  right_join(grid_full, by = c("year", "month")) |>
  mutate(n = replace_na(n, 0),
         mes = factor(MES[month], levels = MES),
         has_data = n > 0)

year_gaps <- cov |> group_by(year) |> summarise(tot = sum(n), .groups = "drop") |>
  filter(tot == 0)

pa <- ggplot(cov, aes(mes, factor(year))) +
  geom_tile(aes(fill = ifelse(n > 0, n, NA)), colour = "white", linewidth = 0.7) +
  geom_text(data = filter(cov, n > 0), aes(label = n), size = 2.3,
            colour = "white", fontface = "bold") +
  annotate("rect", xmin = 0.5, xmax = 12.5,
           ymin = match(year_gaps$year, sort(unique(cov$year))) - 0.5,
           ymax = match(year_gaps$year, sort(unique(cov$year))) + 0.5,
           fill = ACCENT, alpha = 0.10) +
  geom_text(data = year_gaps, aes(x = 6.5, y = factor(year),
                                  label = "no campaign"),
            inherit.aes = FALSE, size = 2.5, colour = ACCENT,
            fontface = "italic") +
  scale_fill_gradient(low = "#8FB4C7", high = "#1F4D66", na.value = "grey96",
                      name = "match-ups", breaks = c(25, 60, 100),
                      guide = guide_colourbar(
                        barwidth = unit(60, "pt"), barheight = unit(5, "pt"),
                        title.position = "left", title.vjust = 1)) +
  scale_x_discrete(drop = FALSE) +
  labs(title = "(a)  Three years with no campaigns and no data outside the dry season",
       subtitle = paste("Number of satellite-field match-ups by year and month.",
                        "There is no data for 2020, 2021 or 2023,\nso the",
                        "2022-2024 hold-out only contains 2022 and 2024"),
       x = NULL, y = NULL) +
  theme(legend.position = "bottom", legend.direction = "horizontal",
        legend.margin = margin(t = -2), panel.grid = element_blank())

# ---------------------------------------------------------------- panel (b) --
seas2 <- seas |>
  mutate(mes = factor(month_label, levels = MES),
         season = factor(season, levels = c("Wet season (Dec-Mar)", "Transition",
                                            "Dry season (Jul-Oct)"),
                         labels = c("Wet season (Dec-Mar)", "Transition",
                                    "Dry season (Jul-Oct)")))

pb <- ggplot(seas2, aes(mes, n, fill = season)) +
  geom_col(width = 0.7) +
  geom_text(data = filter(seas2, n > 0), aes(label = n), vjust = -0.4,
            size = 2.35, colour = INK, fontface = "bold") +
  annotate("text", x = 3.1, y = 235, label = "0 match-ups in the\nentire wet season",
           size = 2.5, colour = ACCENT, fontface = "bold", lineheight = 1.1,
           hjust = 0.5) +
  scale_fill_manual(values = c("Wet season (Dec-Mar)" = "#7BA8C4",
                               "Transition" = "grey82",
                               "Dry season (Jul-Oct)" = "#2E6E8E"), name = NULL) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.14))) +
  labs(title = "(b)  Only dry season data exists",
       subtitle = paste("Puno Bay blooms occur\nduring the",
                        "wet season, which the dataset does\nnot cover at all"),
       x = NULL, y = "Match-ups") +
  theme(legend.position = "bottom", legend.margin = margin(t = -4))

# ---------------------------------------------------------------- panel (c) --
pc_df <- dist |> mutate(zone_label = factor(ZMAP[zona], levels = ZORD))
stats <- pc_df |> group_by(zone_label) |>
  summarise(n = n(), med = median(secchi), mn = mean(secchi),
            sd = sd(secchi), .groups = "drop") |>
  mutate(txt = sprintf("n = %d\nmean %.1f ± %.1f m", n, mn, sd))

pc <- ggplot(pc_df, aes(secchi, zone_label, fill = zone_label)) +
  geom_violin(colour = NA, alpha = 0.28, width = 1.0) +
  geom_boxplot(width = 0.16, outlier.size = 0.4, outlier.alpha = 0.4,
               colour = INK_2, linewidth = 0.35, alpha = 0.92) +
  geom_text(data = stats, aes(x = 0.3, y = zone_label, label = txt),
            inherit.aes = FALSE, hjust = 0, vjust = -1.15, size = 2.25,
            colour = INK_2, lineheight = 1.05) +
  scale_fill_manual(values = PAL_ZONE, guide = "none") +
  scale_x_continuous(breaks = seq(0, 16, 4), limits = c(0, 18)) +
  labs(title = "(c)  The trophic gradient dominates Secchi",
       subtitle = paste("Only 73 distinct values in 812 match-ups:\n",
                        "readings are rounded to half a meter, so\nthe RMSE of",
                        "1.86 m borders the granularity of the",
                        "reference measurement itself"),
       x = "Secchi disk depth (m)", y = NULL)

fig <- pa / (pb | pc) + plot_layout(heights = c(1.25, 1))
save_fig(fig, "fig05_data_inventory", W2, 175)
cat("fig05 lista\n")
