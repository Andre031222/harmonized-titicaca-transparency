# ============================================================================
# figS2_trends.R -- Tendencias del Secchi in situ y su fragilidad (p10)
#
# (a) Mediana anual por zona (epoca seca) en bruto y ajustada por localizacion
#     (anomalia respecto a la climatologia de cada localizacion), con la recta
#     de Sen de la serie seca completa; la campana de lluvias de dic 2012 se
#     muestra aparte (hueco)
# (b) Pendiente de Sen con IC 95 % de Gilbert y P exacta de Mann-Kendall para
#     cada variante: seca / seca + lluvias, bruto / ajustado, anio inicial
# (c) Lecturas por zona y anio detras de cada mediana
# ============================================================================

source(file.path(local({a <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(a)) dirname(normalizePath(sub("^--file=", "", a[1]))) else getwd()}),
  "theme_titicaca.R"))
suppressPackageStartupMessages(library(jsonlite))

tr   <- read_tidy("annual_trends.csv")
tt   <- read_tidy("trend_tests.csv")
sens <- read_tidy("trend_start_year_sensitivity.csv")
vchg <- read_tidy("trend_verdict_changes.csv")
nzy  <- read_tidy("trend_n_zone_year.csv")
tj   <- fromJSON(file.path(ROOT, "results/metrics/temporal.json"))
med_raw <- read_csv(file.path(ROOT, "data/processed/insitu_annual_medians.csv"),
                    show_col_types = FALSE)
med_loc <- read_csv(file.path(ROOT,
                              "data/processed/insitu_location_anomaly_medians.csv"),
                    show_col_types = FALSE)

# las medianas de annual_trends deben ser las de data/processed
chk_raw <- tr |> filter(adjust == "raw") |>
  inner_join(med_raw, by = c("zona", "year", "season"))
chk_loc <- tr |> filter(adjust == "location") |>
  inner_join(med_loc, by = c("series", "zona", "year", "season"))
stopifnot(nrow(chk_raw) == sum(tr$adjust == "raw"),
          nrow(chk_loc) == sum(tr$adjust == "location"),
          all(abs(chk_raw$value - chk_raw$secchi_median) < 1e-6),
          all(abs(chk_loc$value - chk_loc$anomaly_median) < 1e-6))

ADJ <- c(raw = "Annual median (m)", location = "Location-adjusted anomaly (m)")
SER <- c(dry = "Dry season", dry_wet = "Dry + Dec 2012 wet")
WET_YEAR <- tj$wet_season_campaign$years
YEARS <- range(tj$years_with_data)

# ------------------------------------------------------------------ (a) --
pa_df <- tr |>
  filter(series == "dry" | (series == "dry_wet" & year %in% WET_YEAR)) |>
  mutate(zl = zone_factor(zona),
         adj = factor(map_strict(adjust, ADJ, "ajuste"), levels = ADJ),
         wet = year %in% WET_YEAR)
stopifnot(all(pa_df$season[pa_df$wet] == "wet"),
          all(pa_df$season[!pa_df$wet] == "dry"))
dry_pts <- filter(pa_df, !wet)

# recta de Sen de la serie seca completa: pendiente del pipeline, ordenada en
# el origen = mediana de (y - b t) (convencion de Sen)
sen_main <- tt |> filter(series == "dry") |>
  mutate(zl = zone_factor(zona),
         adj = factor(map_strict(adjust, ADJ, "ajuste"), levels = ADJ))
sen_lines <- dry_pts |> select(zl, adj, year, value) |>
  inner_join(select(sen_main, zl, adj, sen_slope, start_year,
                                          end_year), by = c("zl", "adj")) |>
  group_by(zl, adj) |>
  summarise(b = median(value - sen_slope[1] * year), slope = sen_slope[1],
            x0 = start_year[1], x1 = end_year[1], .groups = "drop") |>
  mutate(y0 = b + slope * x0, y1 = b + slope * x1)
lab_a <- sen_main |>
  mutate(ptxt = ifelse(p_exact < 0.001, "P < 0.001",
                       sprintf("P = %.3f", p_exact)),
         txt = sprintf("Sen %+.2f m/yr [%.2f, %.2f]\n%s, %d years",
                       sen_slope, sen_lo95, sen_hi95, ptxt, n_years))
stopifnot(nrow(lab_a) == 6)

# margen superior en cada fila para que la etiqueta no pise los datos
rng <- pa_df |> group_by(adj) |>
  summarise(lo = min(value), hi = max(value), .groups = "drop") |>
  mutate(top = hi + 0.55 * (hi - lo))
lab_a <- lab_a |> left_join(rng, by = "adj")
blank <- rng |> tidyr::crossing(zl = factor(ZONE_ORDER, levels = ZONE_ORDER)) |>
  tidyr::pivot_longer(c(lo, top), values_to = "value") |> mutate(year = YEARS[1])

pa <- ggplot(pa_df, aes(year, value)) +
  geom_blank(data = blank) +
  geom_hline(data = tibble(adj = factor(ADJ[["location"]], levels = ADJ)),
             aes(yintercept = 0), colour = INK_2, linewidth = 0.3, linetype = "22") +
  geom_segment(data = sen_lines, aes(x = x0, xend = x1, y = y0, yend = y1,
                                     colour = zl),
               linewidth = 0.55, linetype = "42") +
  geom_line(data = dry_pts, aes(colour = zl), linewidth = 0.35, alpha = 0.6) +
  geom_point(data = dry_pts, aes(fill = zl), shape = 21, colour = "white",
             stroke = 0.25, size = 1.7) +
  geom_point(data = filter(pa_df, wet), aes(colour = zl), shape = 21,
             fill = "white", stroke = 0.6, size = 1.7) +
  geom_text(data = lab_a, aes(x = YEARS[1], y = top, label = txt), hjust = 0,
            vjust = 1, size = 2.3, lineheight = 0.95, colour = INK) +
  facet_grid(adj ~ zl, scales = "free_y", switch = "y") +
  scale_colour_manual(values = PAL_ZONE, guide = "none") +
  scale_fill_manual(values = PAL_ZONE, guide = "none") +
  scale_x_continuous(breaks = seq(2011, 2024, 3), limits = YEARS + c(-0.4, 0.4)) +
  labs(x = NULL, y = NULL) +
  theme(strip.placement = "outside",
        strip.text.y.left = element_text(angle = 90, face = "plain", hjust = 0.5,
                                         size = 7.4),
        panel.spacing.y = unit(10, "pt"))

# ------------------------------------------------------------------ (b) --
# solo series distintas: pedir 2012 en la serie seca da la misma que 2013
pb_df <- sens |> filter(distinct_series) |>
  mutate(zl = zone_factor(zona),
         variant = factor(sprintf("%s\n%s", SER[series],
                                  c(raw = "raw", location = "location-adjusted")[adjust]),
                          levels = c(sprintf("%s\n%s", rep(SER, each = 2),
                                             c("raw", "location-adjusted")))),
         start = factor(start_year, levels = rev(sort(unique(start_year)))),
         sig = p_exact < 0.05,
         ptxt = ifelse(p_exact < 0.001, "P < 0.001", sprintf("P = %.3f", p_exact)))
stopifnot(!anyNA(pb_df$variant), !anyNA(pb_df$p_exact))
# cada combinacion serie x ajuste x zona tiene tantas filas como anios
# iniciales distintos declara trend_verdict_changes
nchk <- pb_df |> count(series, adjust, zona) |>
  inner_join(vchg, by = c("series", "adjust", "zona"))
stopifnot(all(nchk$n == nchk$n_distinct_series))

X_LO <- min(pb_df$sen_lo95); X_HI <- max(pb_df$sen_hi95)
X_P  <- X_HI + 0.08 * (X_HI - X_LO)
pb <- ggplot(pb_df, aes(sen_slope, start, colour = zl)) +
  geom_vline(xintercept = 0, colour = INK_2, linewidth = 0.3) +
  geom_errorbar(aes(xmin = sen_lo95, xmax = sen_hi95), width = 0,
                linewidth = 0.5, orientation = "y") +
  geom_point(aes(shape = sig), fill = "white", size = 1.6, stroke = 0.6) +
  geom_text(aes(x = X_P, label = ptxt, fontface = ifelse(sig, "bold", "plain")),
            hjust = 0, size = 2.25, colour = INK) +
  facet_grid(variant ~ zl, scales = "free_y", space = "free_y", switch = "y") +
  scale_colour_manual(values = PAL_ZONE, guide = "none") +
  scale_shape_manual(values = c(`TRUE` = 16, `FALSE` = 21),
                     labels = c(`TRUE` = "Mann–Kendall P < 0.05",
                                `FALSE` = "P ≥ 0.05"), name = NULL,
                     breaks = c("TRUE", "FALSE")) +
  scale_x_continuous(breaks = seq(-0.5, 0.5, 0.5),
                     expand = expansion(mult = c(0.03, 0))) +
  coord_cartesian(xlim = c(X_LO, X_P + 0.62 * (X_HI - X_LO))) +
  labs(x = "Sen slope (m per year) with Gilbert 95% CI", y = "Series start year") +
  guides(shape = guide_legend(override.aes = list(colour = INK))) +
  theme(strip.placement = "outside",
        strip.text.y.left = element_text(angle = 0, face = "plain", hjust = 1,
                                         size = 6.6, lineheight = 0.95),
        axis.text.y = element_text(size = 6.6),
        panel.grid.major.y = element_blank(),
        panel.border = element_rect(fill = NA, colour = "grey80", linewidth = 0.3),
        panel.spacing.y = unit(3, "pt"),
        strip.switch.pad.grid = unit(5, "pt"),
        legend.position = "bottom", legend.key.size = unit(7, "pt"),
        legend.text = element_text(size = 6.6),
        legend.margin = margin(t = -4))

# ------------------------------------------------------------------ (c) --
reads <- nzy |> mutate(zl = zone_factor(zona)) |>
  complete(year = seq(YEARS[1], YEARS[2]), zl, fill = list(n = 0))
no_camp <- setdiff(seq(YEARS[1], YEARS[2]), tj$years_with_data)
stopifnot(sum(nzy$n) == tj$n_readings_with_secchi)
# una etiqueta por bloque de anios consecutivos sin campana
camp_gap <- vapply(split(no_camp, cumsum(c(1, diff(no_camp) != 1))), mean, 0)
y_c <- max(count(nzy, year, wt = n)$n)
pc <- ggplot(reads, aes(year, n, fill = zl)) +
  annotate("rect", xmin = no_camp - 0.5, xmax = no_camp + 0.5, ymin = -Inf,
           ymax = Inf, fill = "grey94") +
  geom_col(width = 0.75) +
  annotate("text", x = WET_YEAR, y = y_c * 1.04, label = "wet", size = 2.2,
           colour = INK_2, fontface = "italic", vjust = 0) +
  annotate("text", x = camp_gap, y = y_c * 0.8, label = "no campaign",
           angle = 90, size = 2.2, colour = INK_2, fontface = "italic") +
  scale_fill_manual(values = PAL_ZONE, name = NULL) +
  scale_x_continuous(breaks = seq(2011, 2024, 3), expand = c(0.02, 0)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.1))) +
  labs(x = NULL, y = "Secchi readings") +
  theme(legend.position = "bottom", legend.direction = "vertical",
        legend.key.size = unit(7, "pt"), legend.text = element_text(size = 6.6),
        legend.margin = margin(t = -4))

# ---------------------------------------------------------------- montaje --
# sin envolver, patchwork alinea (c) con las facetas de (b) y despega su eje x
fig <- pa / ((pb | wrap_elements(full = pc)) + plot_layout(widths = c(2.6, 1))) +
  plot_layout(heights = c(1, 1.25)) + tags_abc()
save_fig(fig, "figS2_trends", W2, 215)
cat("figS2 lista\n")
