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
# El pipeline exporta month_label y season en espanol. El panel (a) indexa MES
# por numero de mes y siempre estuvo bien; el (b) casaba por nombre.
MES_ES    <- setNames(MES, c("Ene","Feb","Mar","Abr","May","Jun",
                             "Jul","Ago","Sep","Oct","Nov","Dic"))
SEASON_ES <- c("Lluvias (Dic-Mar)" = "Wet season (Dec-Mar)",
               "Transicion"        = "Transition",
               "Seca (Jul-Oct)"    = "Dry season (Jul-Oct)")


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
  geom_text(data = filter(cov, n > 0), aes(label = n), size = 2.2,
            colour = "white", fontface = "bold") +
  annotate("rect", xmin = 0.5, xmax = 12.5,
           ymin = match(year_gaps$year, sort(unique(cov$year))) - 0.5,
           ymax = match(year_gaps$year, sort(unique(cov$year))) + 0.5,
           fill = ACCENT, alpha = 0.10) +
  geom_text(data = year_gaps, aes(x = 6.5, y = factor(year),
                                  label = "no campaign"),
            inherit.aes = FALSE, size = 2.3, colour = ACCENT,
            fontface = "italic") +
  scale_fill_gradient(low = "#8FB4C7", high = "#1F4D66", na.value = "grey96",
                      name = "match-ups", breaks = c(25, 60, 100),
                      guide = guide_colourbar(
                        barwidth = unit(60, "pt"), barheight = unit(5, "pt"),
                        title.position = "left", title.vjust = 1)) +
  scale_x_discrete(drop = FALSE) +
  labs(x = NULL, y = NULL) +
  theme(legend.position = "bottom", legend.direction = "horizontal",
        legend.margin = margin(t = -2), panel.grid = element_blank(),
        axis.line = element_blank(), axis.ticks = element_blank())

# ---------------------------------------------------------------- panel (b) --
seas2 <- seas |>
  mutate(mes    = factor(map_strict(month_label, MES_ES, "mes"), levels = MES),
         season = factor(map_strict(season, SEASON_ES, "temporada"),
                         levels = unname(SEASON_ES)))

pb <- ggplot(seas2, aes(mes, n, fill = season)) +
  geom_col(width = 0.7) +
  # doce meses no caben en una fila a media pagina: salian pegados
  scale_x_discrete(guide = guide_axis(n.dodge = 2)) +
  geom_text(data = filter(seas2, n > 0), aes(label = n), vjust = -0.4,
            size = 2.2, colour = INK) +
  annotate("text", x = 3.1, y = 235, label = "No match-ups in the\nwet season",
           size = 2.3, colour = ACCENT, lineheight = 1.05,
           hjust = 0.5) +
  scale_fill_manual(values = c("Wet season (Dec-Mar)" = "#7BA8C4",
                               "Transition" = "grey82",
                               "Dry season (Jul-Oct)" = "#2E6E8E"), name = NULL) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.14))) +
  labs(x = NULL, y = "Match-ups") +
  theme(legend.position = "bottom", legend.margin = margin(t = -4),
        legend.key.size = unit(6, "pt"))

# ---------------------------------------------------------------- panel (c) --
pc_df <- dist |> mutate(zone_label = zone_factor(zona))
kw    <- kruskal.test(secchi ~ zone_label, data = pc_df)$p.value
means <- pc_df |> group_by(zone_label) |>
  summarise(txt = sprintf("%.1f \u00b1 %.1f m", mean(secchi), sd(secchi)),
            .groups = "drop")

pc <- ggplot(pc_df, aes(zone_label, secchi, colour = zone_label)) +
  raincloud(point_size = 0.45) +
  geom_text(data = means, aes(zone_label, 18.2, label = txt), inherit.aes = FALSE,
            size = 2.2, colour = INK) +
  n_labels(pc_df, zone_label, 0.6) +
  annotate("text", x = 0.55, y = 20.2, hjust = 0, size = 2.2, colour = INK_2,
           parse = TRUE, label = paste0('"Kruskal\u2013Wallis, "*', p_fmt(kw))) +
  scale_colour_manual(values = PAL_ZONE, guide = "none") +
  scale_x_discrete(labels = function(x) sub(" de ", "\nde ", sub("Lago ", "Lago\n", x))) +
  scale_y_continuous(limits = c(-0.6, 20.5), breaks = seq(0, 16, 4)) +
  labs(x = NULL, y = "Secchi disk depth (m)")

fig <- pa / (pb | pc) + plot_layout(heights = c(1.2, 1)) + tags_abc()
save_fig(fig, "fig05_data_inventory", W2, 150)
cat("fig05 lista\n")
