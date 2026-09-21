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

# anos sin campana, como franjas
gaps <- data.frame(x0 = c(2019.5, 2022.5), x1 = c(2021.5, 2023.5))

# la recta es la de Sen (pendiente del CSV, intercepto = mediana de y - b x),
# no un ajuste por minimos cuadrados que no es lo que se reporta
sen_line <- tr |> group_by(zl) |>
  summarise(b = first(sen_slope_m_per_yr),
            a = median(secchi_median - first(sen_slope_m_per_yr) * year),
            x0 = min(year), x1 = max(year), .groups = "drop") |>
  mutate(y0 = a + b * x0, y1 = a + b * x1)

pa <- ggplot(tr, aes(year, secchi_median, colour = zl)) +
  geom_rect(data = gaps, aes(xmin = x0, xmax = x1, ymin = -Inf, ymax = Inf),
            inherit.aes = FALSE, fill = "grey90", alpha = 0.8) +
  geom_segment(data = sen_line, aes(x = x0, xend = x1, y = y0, yend = y1),
               linewidth = 0.55, linetype = "22") +
  geom_line(linewidth = 0.55) +
  geom_point(size = 1.7) +
  geom_label(data = lab, aes(x = 2011, y = Inf, label = txt), hjust = 0,
             vjust = 1.35, size = 2.35, inherit.aes = FALSE, colour = INK_2,
             fill = alpha("white", 0.85), label.size = 0,
             label.padding = unit(1.4, "pt")) +
  annotate("text", x = 2021.5, y = -Inf, label = "no campaign", vjust = -0.6,
           size = 1.9, colour = INK_2, fontface = "italic") +
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
  geom_label(aes(label = sprintf("P = %.3f", p_value),
                 vjust = ifelse(sen_slope_m_per_yr < 0, 1.6, -0.85)),
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
  scale_y_continuous(limits = c(-0.3, 0.62),
                     expand = expansion(mult = c(0.04, 0.14))) +
  labs(x = "Start year of the series", y = "Sen's slope (m/year)") +
  theme(legend.position = "inside", legend.position.inside = c(0.995, 0.98),
        legend.justification = c(1, 1), legend.key.size = unit(7, "pt"),
        legend.background = element_blank())

fig <- pa / pb + plot_layout(heights = c(1, 1.05)) + tags_abc()
save_fig(fig, "fig06_trends_and_start_year", W2, 150)
cat("fig06 lista\n")
