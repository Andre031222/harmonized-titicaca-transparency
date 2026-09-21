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
# Matriz anio x mes con sus marginales: totales por mes arriba y por anio y
# sensor a la derecha. La temporada de lluvias (dic-mar) va sombreada en ambas.
grid_full <- expand.grid(year = seq(min(dist$year), max(dist$year)),
                         month = 1:12)
cov <- dist |>
  count(year, month, name = "n") |>
  right_join(grid_full, by = c("year", "month")) |>
  mutate(n = replace_na(n, 0),
         mes = factor(MES[month], levels = MES))
years <- sort(unique(cov$year))
year_gaps <- cov |> group_by(year) |> summarise(tot = sum(n), .groups = "drop") |>
  filter(tot == 0)
WET <- c(1, 2, 3, 12)
wet_cols <- tibble(x0 = c(0.5, 11.5), x1 = c(3.5, 12.5))
BLUE_WET <- "#7BA8C4"

heat <- ggplot(cov, aes(mes, factor(year, levels = years))) +
  geom_tile(aes(fill = ifelse(n > 0, n, NA)), colour = "white", linewidth = 0.7) +
  geom_rect(data = wet_cols, aes(xmin = x0, xmax = x1, ymin = -Inf, ymax = Inf),
            inherit.aes = FALSE, fill = BLUE_WET, alpha = 0.16) +
  annotate("rect", xmin = 0.5, xmax = 12.5,
           ymin = match(year_gaps$year, years) - 0.5,
           ymax = match(year_gaps$year, years) + 0.5,
           fill = ACCENT, alpha = 0.10) +
  geom_text(data = filter(cov, n > 0), aes(label = n), size = 2.2,
            colour = "white", fontface = "bold") +
  geom_text(data = year_gaps, aes(x = 6.5, y = factor(year, levels = years),
                                  label = "no campaign"),
            inherit.aes = FALSE, size = 2.2, colour = ACCENT, fontface = "italic") +
  scale_fill_gradient(low = "#8FB4C7", high = "#1F4D66", na.value = "grey96",
                      guide = "none") +
  scale_x_discrete(drop = FALSE, labels = function(x) substr(x, 1, 1)) +
  labs(x = NULL, y = NULL) +
  theme(panel.grid = element_blank(), axis.line = element_blank(),
        axis.ticks = element_blank(), plot.margin = margin(0, 0, 2, 2))

# marginal superior: match-ups por mes
seas2 <- seas |> mutate(mes = factor(map_strict(month_label, MES_ES, "mes"),
                                     levels = MES))
top <- ggplot(seas2, aes(mes, n)) +
  geom_rect(data = wet_cols, aes(xmin = x0, xmax = x1, ymin = -Inf, ymax = Inf),
            inherit.aes = FALSE, fill = BLUE_WET, alpha = 0.16) +
  geom_col(fill = "#2E6E8E", width = 0.72) +
  geom_text(data = filter(seas2, n > 0), aes(label = n), vjust = -0.35,
            size = 2.1, colour = INK) +
  annotate("text", x = 2, y = 300, label = "wet season:\nno match-ups",
           size = 2.1, colour = "#3F6F8C", lineheight = 0.95, fontface = "italic") +
  scale_x_discrete(drop = FALSE) +
  scale_y_continuous(limits = c(0, 470), breaks = c(0, 200, 400),
                     expand = expansion(mult = c(0, 0))) +
  labs(x = NULL, y = "Match-ups", tag = "a") +
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(),
        axis.line.x = element_blank(), panel.grid.major.y = element_blank(),
        axis.title.y = element_text(size = 7), plot.margin = margin(2, 0, 0, 2))

# marginal derecho: match-ups por anio y sensor
SENS <- c(LS = "Landsat 8/9", S2 = "Sentinel-2")
ys <- dist |> count(year, sensor) |>
  mutate(sensor = factor(map_strict(sensor, SENS, "sensor"), levels = rev(SENS))) |>
  right_join(tibble(year = years), by = "year") |>
  mutate(n = replace_na(n, 0))
right <- ggplot(filter(ys, n > 0),
                aes(n, factor(year, levels = years), fill = sensor)) +
  geom_col(width = 0.72) +
  scale_fill_manual(values = PAL_SENSOR, name = NULL,
                    breaks = unname(SENS)) +
  scale_y_discrete(drop = FALSE, limits = as.character(years)) +
  scale_x_continuous(breaks = c(0, 50, 100), expand = expansion(mult = c(0, 0.05))) +
  labs(x = "Match-ups", y = NULL) +
  theme(axis.text.y = element_blank(), axis.ticks.y = element_blank(),
        axis.line.y = element_blank(), panel.grid.major.y = element_blank(),
        axis.title.x = element_text(size = 7),
        legend.position = "inside", legend.position.inside = c(1, 1),
        legend.justification = c(1, 0), legend.key.size = unit(6, "pt"),
        legend.text = element_text(size = 6), legend.background = element_blank(),
        plot.margin = margin(0, 2, 2, 0))

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

pc <- pc + labs(tag = "b")
design <- "
AA#D
BCCD
"
fig <- wrap_plots(A = top, B = heat, C = right, D = pc, design = design) +
  plot_layout(design = c(area(1, 1, 1, 2), area(2, 1, 2, 2), area(2, 3, 2, 3),
                         area(1, 4, 2, 4)),
              widths = c(1, 1.9, 0.62, 1.55), heights = c(0.32, 1)) &
  theme(plot.tag = element_text(size = 10, face = "bold"))
save_fig(fig, "fig05_data_inventory", W2, 105)
cat("fig05 lista\n")
