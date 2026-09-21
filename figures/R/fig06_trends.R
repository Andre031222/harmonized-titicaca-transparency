# ============================================================================
# fig06_trends.R -- Tendencias, y por que hay que reportarlas con cuidado
#
# (a) Franjas: mediana anual de Secchi in-situ por zona y anio, 2011-2024,
#     con la pendiente de Sen y la P de Mann-Kendall del registro completo
# (b) Matriz zona x anio inicial: pendiente de Sen, IC 95% y P. Empezar en
#     2013 vuelve "significativas" dos zonas: el veredicto lo fija el anio
#     inicial, no el lago.
# (c) Lecturas por anio y zona: cuanto dato hay detras de cada franja
# ============================================================================

source(file.path(local({a <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(a)) dirname(normalizePath(sub("^--file=", "", a[1]))) else getwd()}),
  "theme_titicaca.R"))

tr   <- read_tidy("annual_trends.csv")
sens <- read_tidy("trend_start_year_sensitivity.csv")
nmed <- read_csv(file.path(ROOT, "data/processed/insitu_annual_medians.csv"),
                 show_col_types = FALSE)

ZONES  <- c("Bahia de Puno", "Lago Menor", "Lago Mayor")
YEARS  <- 2011:2024
NO_CAMP <- c(2020, 2021, 2023)
SECCHI_PAL <- c("#7A3E1D", "#C2582C", "#E8B04A", "#8FC1B5", "#2E6E8E", "#16324F")

# ------------------------------------------------------------------ (a) --
stripes <- expand.grid(zone_label = ZONES, year = YEARS, stringsAsFactors = FALSE) |>
  left_join(tr |> select(zone_label, year, secchi_median), by = c("zone_label", "year")) |>
  mutate(zl = zone_factor(zone_label),
         state = case_when(!is.na(secchi_median) ~ "data",
                           year %in% NO_CAMP ~ "no campaign",
                           TRUE ~ "no reading"))
stopifnot(sum(stripes$state == "data") == nrow(tr))

lab_a <- tr |> distinct(zone_label, sen_slope_m_per_yr, p_value, n_years) |>
  mutate(zl = zone_factor(zone_label),
         txt = sprintf("Sen %+.2f m/yr\nP = %.2f, %d years",
                       sen_slope_m_per_yr, p_value, n_years))

pa <- ggplot(stripes, aes(year, zl)) +
  geom_tile(data = filter(stripes, state == "data"), aes(fill = secchi_median),
            colour = "white", linewidth = 0.6, height = 0.86) +
  geom_tile(data = filter(stripes, state != "data"), fill = "grey93",
            colour = "white", linewidth = 0.6, height = 0.86) +
  geom_text(data = filter(stripes, state == "data"),
            aes(label = sprintf("%.1f", secchi_median),
                colour = secchi_median > 7.5 & secchi_median < 11),
            size = 2.1, fontface = "bold") +
  geom_text(data = filter(stripes, state == "no reading"), label = "–",
            size = 2.4, colour = INK_2) +
  annotate("text", x = NO_CAMP, y = 2, label = "no campaign", angle = 90,
           size = 2.1, colour = INK_2, fontface = "italic") +
  geom_text(data = lab_a, aes(x = 2024.75, y = zl, label = txt), hjust = 0,
            size = 2.2, lineheight = 1, colour = INK) +
  scale_fill_gradientn(colours = SECCHI_PAL, limits = c(4, 14), oob = squish,
                       name = "Median\nSecchi (m)", breaks = c(4, 8, 12),
                       guide = guide_colourbar(barwidth = unit(4, "pt"),
                                               barheight = unit(34, "pt"))) +
  scale_colour_manual(values = c(`TRUE` = INK, `FALSE` = "white"), guide = "none") +
  scale_x_continuous(breaks = seq(2011, 2024, 1), labels = function(x)
    ifelse(x %% 2 == 1, x, ""), expand = expansion(add = c(0.1, 0.1))) +
  scale_y_discrete(limits = rev(levels(stripes$zl))) +
  coord_cartesian(xlim = c(2010.5, 2024.5), clip = "off") +
  labs(x = NULL, y = NULL) +
  theme(panel.grid = element_blank(), axis.line = element_blank(),
        axis.ticks.y = element_blank(), legend.position = "left",
        legend.title = element_text(size = 6.5),
        plot.margin = margin(4, 72, 2, 2))

# ------------------------------------------------------------------ (b) --
# IC del 95% de la pendiente de Sen (Gilbert 1987) sobre anios reales, con la
# varianza de Mann-Kendall corregida por empates, como en p05
sen_ci <- function(x, t, z = 1.96) {
  n <- length(x)
  pr <- combn(n, 2)
  sl <- sort((x[pr[2, ]] - x[pr[1, ]]) / (t[pr[2, ]] - t[pr[1, ]]))
  ties <- table(x); ties <- ties[ties > 1]
  v <- (n * (n - 1) * (2 * n + 5) - sum(ties * (ties - 1) * (2 * ties + 5))) / 18
  C <- z * sqrt(v); N <- length(sl)
  lo <- max(1, floor((N - C) / 2)); hi <- min(N, ceiling((N + C) / 2) + 1)
  c(slope = median(sl), lo = sl[lo], hi = sl[hi])
}
ci <- sens |> rowwise() |>
  mutate(ci = list(with(filter(nmed, zona == zone, year >= start_year) |>
                          arrange(year), sen_ci(secchi_median, year)))) |>
  ungroup() |>
  mutate(chk = vapply(ci, `[[`, 0, "slope"), lo = vapply(ci, `[[`, 0, "lo"),
         hi = vapply(ci, `[[`, 0, "hi"), zl = zone_factor(zone_label),
         txt = sprintf("%+.2f\n[%.2f, %.2f]\nP = %.3f",
                       sen_slope_m_per_yr, lo, hi, p_value))
stopifnot(all(abs(ci$chk - ci$sen_slope_m_per_yr) < 1e-3))

pb <- ggplot(ci, aes(factor(start_year), zl, fill = sen_slope_m_per_yr)) +
  geom_tile(colour = "white", linewidth = 1.1) +
  geom_tile(data = filter(ci, significant_at_005), fill = NA, colour = INK,
            linewidth = 0.8, width = 0.93, height = 0.9) +
  geom_text(aes(label = txt, fontface = ifelse(significant_at_005, "bold", "plain"),
                colour = abs(sen_slope_m_per_yr) > 0.25),
            size = 2.05, lineheight = 0.95) +
  scale_colour_manual(values = c(`TRUE` = "white", `FALSE` = INK), guide = "none") +
  scale_fill_gradient2(low = "#B35806", mid = "#F7F7F7", high = "#2E6E8E",
                       midpoint = 0, limits = c(-0.35, 0.35), oob = squish,
                       name = "Sen slope\n(m/yr)", breaks = c(-0.3, 0, 0.3),
                       guide = guide_colourbar(barwidth = unit(4, "pt"),
                                               barheight = unit(34, "pt"))) +
  scale_x_discrete(position = "top", expand = c(0, 0)) +
  scale_y_discrete(limits = rev(levels(ci$zl)), expand = c(0, 0)) +
  labs(x = "Series starts in", y = NULL) +
  theme(axis.line = element_blank(), axis.ticks = element_blank(),
        panel.grid = element_blank(), legend.title = element_text(size = 6.5))

# ------------------------------------------------------------------ (c) --
reads <- nmed |> mutate(zl = zone_factor(zona)) |>
  complete(year = YEARS, zl, fill = list(n = 0))
pc <- ggplot(reads, aes(factor(year), n, fill = zl)) +
  annotate("rect", xmin = c(9.5, 12.5), xmax = c(11.5, 13.5), ymin = -Inf,
           ymax = Inf, fill = "grey93") +
  geom_col(width = 0.75) +
  scale_fill_manual(values = PAL_ZONE, name = NULL) +
  scale_x_discrete(labels = function(x) ifelse(as.integer(x) %% 3 == 2011 %% 3,
                                               x, "")) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.05))) +
  labs(x = NULL, y = "Secchi readings") +
  theme(panel.grid.major.x = element_blank(), legend.position = "top",
        legend.key.size = unit(6, "pt"), legend.text = element_text(size = 6.5),
        legend.margin = margin(b = -4))

fig <- pa / ((pb | pc) + plot_layout(widths = c(1.15, 1))) +
  plot_layout(heights = c(0.62, 1)) + tags_abc()
save_fig(fig, "fig06_trends_and_start_year", W2, 125)
cat("fig06 lista\n")
