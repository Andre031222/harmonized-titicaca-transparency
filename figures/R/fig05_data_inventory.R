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

# la temporada de lluvias no tiene ni un match-up: en vez de una leyenda con
# un color que nunca aparece, se sombrea su franja de meses
wet <- which(levels(seas2$mes) %in% c("Jan", "Feb", "Mar", "Dec"))
stopifnot(length(wet) == 4)
wet_rect <- tibble(x0 = c(0.5, 11.5), x1 = c(3.5, 12.5))
pb <- ggplot(seas2, aes(mes, n)) +
  geom_rect(data = wet_rect, aes(xmin = x0, xmax = x1, ymin = -Inf, ymax = Inf),
            inherit.aes = FALSE, fill = "#7BA8C4", alpha = 0.16) +
  geom_col(aes(fill = season), width = 0.7, show.legend = FALSE) +
  geom_text(data = filter(seas2, n > 0), aes(label = n), vjust = -0.4,
            size = 2.3, colour = INK, fontface = "bold") +
  annotate("text", x = 2, y = 330, label = "wet season:\nno match-ups",
           size = 2.3, colour = "#3F6F8C", lineheight = 1, fontface = "italic") +
  annotate("text", x = 8.5, y = 450, label = "dry season (Jul–Oct)",
           size = 2.3, colour = "#2E6E8E", fontface = "italic") +
  scale_fill_manual(values = c("Wet season (Dec-Mar)" = "#7BA8C4",
                               "Transition" = "grey70",
                               "Dry season (Jul-Oct)" = "#2E6E8E"), name = NULL) +
  scale_x_discrete(labels = function(x) substr(x, 1, 1)) +
  scale_y_continuous(limits = c(0, 470), expand = expansion(mult = c(0, 0))) +
  labs(x = "Month", y = "Match-ups") +
  theme(panel.grid.major.y = element_blank(), axis.ticks.x = element_blank())

# ---------------------------------------------------------------- panel (c) --
pc_df <- dist |> mutate(zone_label = zone_factor(zona))
kw    <- kruskal.test(secchi ~ zone_label, data = pc_df)$p.value
means <- pc_df |> group_by(zone_label) |>
  summarise(txt = sprintf("%.1f \u00b1 %.1f m", mean(secchi), sd(secchi)),
            .groups = "drop")

brk <- pairwise_bh(pc_df$secchi, pc_df$zone_label, c(17.3, 18.7, 20.1))

pc <- ggplot(pc_df, aes(zone_label, secchi, colour = zone_label)) +
  raincloud(point_size = 0.45) +
  geom_segment(data = brk, aes(x = x1, xend = x2, y = y, yend = y),
               inherit.aes = FALSE, linewidth = 0.3, colour = INK) +
  geom_segment(data = brk, aes(x = x1, xend = x1, y = y, yend = y - 0.3),
               inherit.aes = FALSE, linewidth = 0.3, colour = INK) +
  geom_segment(data = brk, aes(x = x2, xend = x2, y = y, yend = y - 0.3),
               inherit.aes = FALSE, linewidth = 0.3, colour = INK) +
  geom_text(data = brk, aes(x = (x1 + x2) / 2, y = y + 0.1, label = lab),
            inherit.aes = FALSE, parse = TRUE, vjust = 0, size = 2.1, colour = INK) +
  geom_text(data = means, aes(zone_label, -0.7, label = txt), inherit.aes = FALSE,
            size = 2.2, colour = INK, fontface = "bold") +
  n_labels(pc_df, zone_label, -2) +
  annotate("text", x = 0.5, y = 21.9, hjust = 0, size = 2.2, colour = INK_2,
           parse = TRUE, label = paste0('"Kruskal\u2013Wallis, "*', p_fmt(kw))) +
  scale_colour_manual(values = PAL_ZONE, guide = "none") +
  scale_x_discrete(labels = function(x) sub(" de ", "\nde ", sub("Lago ", "Lago\n", x))) +
  scale_y_continuous(limits = c(-2.6, 22.3), breaks = seq(0, 16, 4)) +
  labs(x = NULL, y = "Secchi disk depth (m)")

fig <- pa / (pb | pc) + plot_layout(heights = c(1.2, 1)) + tags_abc()
save_fig(fig, "fig05_data_inventory", W2, 150)
cat("fig05 lista\n")
