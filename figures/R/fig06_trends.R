# ============================================================================
# fig06_trends.R -- Trends, y por que hay que reportarlas con cuidado
#
# (a) Medianas anuales de transparencia in-situ por zona, 2011-2024, con la
#     pendiente de Sen sobre el registro COMPLETO
# (b) El quinto error, que no llegamos a cometer: si la serie se empieza en
#     2013 en vez de 2011, dos zonas pasan a "trend significant". El
#     veredicto lo fija el anio inicial, not the lake.
# ============================================================================

source(file.path(local({a <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(a)) dirname(normalizePath(sub("^--file=", "", a[1]))) else getwd()}),
  "theme_titicaca.R"))

tr   <- read_tidy("annual_trends.csv")
sens <- read_tidy("trend_start_year_sensitivity.csv")

tr   <- tr   |> mutate(zl = zone_factor(zone_label))
sens <- sens |> mutate(zl = zone_factor(zone_label))

# ---------------------------------------------------------------- panel (a) --
lab <- tr |> distinct(zl, sen_slope_m_per_yr, p_value, n_years) |>
  mutate(txt = sprintf("Sen %+.2f m/yr,  P = %.2f,  n = %d years",
                       sen_slope_m_per_yr, p_value, n_years))

gaps <- data.frame(year = c(2020, 2021, 2023))

pa <- ggplot(tr, aes(year, secchi_median, colour = zl)) +
  geom_vline(data = gaps, aes(xintercept = year), inherit.aes = FALSE,
             colour = ACCENT, linetype = "12", linewidth = 0.35) +
  geom_smooth(method = "lm", formula = y ~ x, se = FALSE, linewidth = 0.5,
              linetype = "22", alpha = 0.6) +
  geom_line(linewidth = 0.55) +
  geom_point(size = 1.7) +
  geom_label(data = lab, aes(x = 2011, y = Inf, label = txt), hjust = 0,
             vjust = 1.35, size = 2.35, inherit.aes = FALSE, colour = INK_2,
             fill = alpha("white", 0.85), label.size = 0,
             label.padding = unit(1.4, "pt")) +
  facet_wrap(~zl, nrow = 1) +
  scale_colour_manual(values = PAL_ZONE, guide = "none") +
  scale_x_continuous(breaks = seq(2011, 2024, 3)) +
  scale_y_continuous(expand = expansion(mult = c(0.10, 0.28))) +
  labs(x = NULL, y = "Median Secchi (m)")

# ---------------------------------------------------------------- panel (b) --
pb <- ggplot(sens, aes(factor(start_year), sen_slope_m_per_yr)) +
  geom_hline(yintercept = 0, colour = INK, linewidth = 0.4) +
  geom_segment(aes(xend = factor(start_year), yend = 0,
                   colour = significant_at_005), linewidth = 0.85) +
  geom_point(aes(colour = significant_at_005, shape = significant_at_005),
             size = 2.5) +
  geom_label(aes(label = sprintf("P = %.3f", p_value)), vjust = -0.85,
             size = 2.15, colour = INK_2, fill = alpha("white", 0.85),
             label.size = 0, label.padding = unit(1.2, "pt"),
             show.legend = FALSE) +
  facet_wrap(~zl, nrow = 1) +
  scale_colour_manual(values = c(`TRUE` = ACCENT, `FALSE` = NEUTRAL),
                      labels = c(`TRUE` = "P < 0.05", `FALSE` = "Not significant"),
                      name = NULL) +
  scale_shape_manual(values = c(`TRUE` = 17, `FALSE` = 16),
                     labels = c(`TRUE` = "P < 0.05", `FALSE` = "Not significant"),
                     name = NULL) +
  scale_y_continuous(limits = c(-0.18, 0.62),
                     expand = expansion(mult = c(0.04, 0.14))) +
  labs(x = "Start year of the series", y = "Sen's slope (m/year)") +
  theme(legend.position = "bottom", legend.margin = margin(t = -4))

fig <- pa / pb + plot_layout(heights = c(1, 1.05)) + tags_abc()
save_fig(fig, "fig06_trends_and_start_year", W2, 150)
cat("fig06 lista\n")
