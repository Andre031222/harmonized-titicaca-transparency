# ============================================================================
# fig2_harmonization.R -- Los coeficientes de banda de HLS no vuelven
# intercambiables a Landsat 8/9 y Sentinel-2 sobre el agua del lago.
#
# (a) Cuanto mueven los coeficientes HLS (pendiente + intercepto) la mediana
#     de la reflectancia Sen2Cor de Sentinel-2 sobre el agua, por banda, y el
#     tamano del intercepto frente a esa mediana.
# (b) Exactitud balanceada del clasificador de sensor (validacion dejando una
#     campana fuera) para las tres rutas (producto HLS v2.0, SR de agencia +
#     coeficientes HLS, SR de agencia sin ajuste) y cuatro muestras (todas,
#     eventos pareados, pares a +-3 dias, pares a +-1 dia), con IC 95 % por
#     bootstrap de campanas, frente a la distribucion de 100 controles
#     negativos (violin gris).
# (c) Recalibraciones locales (OLS, RMA, mapeo de cuantiles) en pares y en
#     pares casi simultaneos, frente a la banda de controles.
# La dispersion pareada Landsat vs Sentinel-2 esta en figS4_paired_scatter.R.
#
# Entradas: results/tidy/harmonization_separability.csv,
# harmonization_controls.csv, harmonization_control_draws.csv,
# harmonization_bandpass_shift.csv. Ningun numero escrito a mano.
# ============================================================================

source(file.path(local({a <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(a)) dirname(normalizePath(sub("^--file=", "", a[1]))) else getwd()}),
  "theme_titicaca.R"))

sep   <- read_tidy("harmonization_separability.csv")
ctrl  <- read_tidy("harmonization_controls.csv")
draws <- read_tidy("harmonization_control_draws.csv")
shift <- read_tidy("harmonization_bandpass_shift.csv")

BAND_ORDER <- c("Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2")
stopifnot(setequal(shift$band_label, BAND_ORDER))

PRIMARY_GROUPING <- "campaign"
ROUTE_LAB <- c(hls    = "NASA HLS v2.0\nproduct",
               sr     = "Agency SR +\nHLS coefficients",
               sr_raw = "Agency SR,\nunadjusted")
TEST_LAB  <- c(all = "All", paired = "Paired\nevents",
               pm3d = "Pairs\n±3 d", pm1d_anyimg = "Pairs\n±1 d")
RECAL_LAB <- c(none = "None", ols = "OLS", rma = "RMA",
               qm_bands = "Quantile map.,\nbands", qm_features = "Quantile map.,\nall features")
SAMPLE_LAB <- c(paired = "Paired events", pm3d = "Pairs ±3 d")

# Que distribucion de controles corresponde a cada prueba: la misma regla que
# usa p03 para la columna control_ba_hi_dates (sr_raw comparte filas con sr;
# las pruebas de pares cercanos usan los controles de pares cercanos).
ctrl_key <- function(route, test) {
  tibble(c_route = ifelse(route == "sr_raw", "sr", route),
         c_sample = ifelse(startsWith(test, "pm"), "pm3d", "all"))
}
# control negativo por fechas, los dos sensores juntos (200 repeticiones)
draws_d <- draws |> filter(control == "dates")
ctrl_q <- draws_d |> group_by(route, sample) |>
  summarise(q_lo = quantile(balanced_accuracy, 0.025, na.rm = TRUE),
            .groups = "drop")
# comprobacion: el umbral del CSV es el maximo de los 97.5 % por sensor
thr_chk <- ctrl |> filter(control == "dates") |> group_by(route, sample) |>
  summarise(thr = max(ba_hi), .groups = "drop")

GREY_V <- "#CFCFCF"
PAL_EXC <- c(`TRUE` = INK, `FALSE` = "white")

# ---------------------------------------------------------------- panel (a) --
pa_df <- shift |>
  mutate(band_label = factor(band_label, levels = BAND_ORDER),
         icpt_pct = 100 * abs_intercept_over_s2_median,
         dir = ifelse(median_shift_pct < 0, "down", "up"))
ylim_a <- range(c(pa_df$median_shift_pct, pa_df$icpt_pct, 0))
ylim_a <- ylim_a + c(-0.16, 0.16) * diff(ylim_a)
pa <- ggplot(pa_df, aes(band_label)) +
  geom_hline(yintercept = 0, colour = INK, linewidth = 0.35) +
  geom_col(aes(y = median_shift_pct), width = 0.45, fill = PAL_SENSOR[["Sentinel-2"]],
           position = position_nudge(x = -0.18)) +
  geom_text(aes(x = as.numeric(band_label) - 0.18, y = median_shift_pct,
                label = sprintf("%+.0f%%", median_shift_pct),
                vjust = ifelse(median_shift_pct < 0, 1.5, -0.6)),
            size = 2.0, colour = INK, fontface = "bold") +
  geom_point(aes(x = as.numeric(band_label) + 0.36, y = icpt_pct,
                 shape = "|Intercept| / S2 median"),
             size = 1.9, colour = INK, fill = "white", stroke = 0.55) +
  scale_shape_manual(values = 23, name = NULL) +
  scale_y_continuous(labels = function(x) paste0(x, "%"),
                     breaks = breaks_pretty(n = 5)) +
  scale_x_discrete(labels = function(x) sub("SWIR", "SWIR\n", x)) +
  coord_cartesian(ylim = ylim_a) +
  labs(x = NULL, y = "Change in the Sentinel-2 median\nwater reflectance (HLS coefficients)") +
  theme(axis.ticks.x = element_blank(), panel.grid.major.y = element_blank(),
        axis.text.x = element_text(lineheight = 0.9),
        legend.position = "inside", legend.position.inside = c(0.02, 1.0),
        legend.justification = c(0, 1), legend.text = element_text(size = 6.6))

# ---------------------------------------------------------------- panel (b) --
pb_df <- sep |>
  filter(grouping == PRIMARY_GROUPING, featset == "all12", recal == "none") |>
  mutate(route_f = factor(map_strict(route, ROUTE_LAB, "ruta"), levels = ROUTE_LAB),
         test_f = factor(map_strict(test, TEST_LAB, "prueba"), levels = TEST_LAB),
         exc = as.character(exceeds_control))
stopifnot(nrow(pb_df) == nrow(distinct(pb_df, route, test)))
# un violin de controles detras de cada prueba
vio_b <- pb_df |> select(route, test, route_f, test_f) |>
  bind_cols(ctrl_key(pb_df$route, pb_df$test)) |>
  inner_join(draws_d, by = c(c_route = "route", c_sample = "sample"),
             relationship = "many-to-many")
stopifnot(nrow(vio_b) == nrow(pb_df) * 200)
thr_b <- pb_df |> distinct(route_f, test_f, control_ba_hi_dates)
YL <- c(min(draws_d$balanced_accuracy, na.rm = TRUE) - 0.02, 1.02)

pb <- ggplot(pb_df, aes(test_f)) +
  geom_hline(yintercept = 0.5, colour = INK_2, linetype = "22", linewidth = 0.3) +
  geom_violin(data = vio_b, aes(y = balanced_accuracy), fill = GREY_V,
              colour = NA, width = 0.75, scale = "width") +
  geom_segment(data = thr_b, aes(x = as.numeric(test_f) - 0.3,
                                 xend = as.numeric(test_f) + 0.3,
                                 y = control_ba_hi_dates, yend = control_ba_hi_dates),
               colour = NEUTRAL, linewidth = 0.45) +
  geom_errorbar(aes(ymin = ba_lo, ymax = ba_hi), width = 0, linewidth = 0.5,
                colour = INK) +
  geom_point(aes(y = balanced_accuracy, fill = exc), shape = 21, size = 1.9,
             stroke = 0.6, colour = INK) +
  geom_text(aes(x = as.numeric(test_f) + 0.17, y = balanced_accuracy,
                label = sprintf("%.2f", balanced_accuracy)),
            hjust = 0, size = 2.1, colour = INK_2) +
  facet_grid(~route_f, scales = "free_x", space = "free_x") +
  scale_fill_manual(values = PAL_EXC, guide = "none") +
  scale_x_discrete(expand = expansion(add = c(0.55, 0.75))) +
  scale_y_continuous(breaks = seq(0.3, 1, 0.1), limits = YL,
                     expand = expansion(0)) +
  labs(x = NULL, y = "Balanced accuracy,\nLandsat vs Sentinel-2") +
  theme(axis.ticks.x = element_blank(), axis.text.x = element_text(lineheight = 0.9),
        strip.text = element_text(size = 7, face = "plain"))

# ---------------------------------------------------------------- panel (c) --
pc_df <- sep |>
  filter(grouping == PRIMARY_GROUPING, featset == "all12",
         test %in% names(SAMPLE_LAB), route %in% unique(sep$route[sep$recal != "none"])) |>
  mutate(route_f = factor(map_strict(route, ROUTE_LAB, "ruta"), levels = ROUTE_LAB),
         sample_f = factor(map_strict(test, SAMPLE_LAB, "muestra"), levels = SAMPLE_LAB),
         recal_f = factor(map_strict(recal, RECAL_LAB, "recalibracion"),
                          levels = rev(RECAL_LAB)),
         exc = as.character(exceeds_control)) |>
  droplevels() |>
  mutate(facet_lab = factor(paste0(sub("\n", " ", route_f), "\n", sample_f)),
         facet_lab = factor(facet_lab, levels = unique(facet_lab[order(route_f, sample_f)])))
stopifnot(nrow(pc_df) == nrow(distinct(pc_df, route, test, recal)))
# banda de controles: del 2.5 % de las repeticiones juntas al umbral de p03
band_c <- pc_df |> distinct(route, test, route_f, sample_f, control_ba_hi_dates) |>
  bind_cols(ctrl_key(pc_df |> distinct(route, test, route_f, sample_f,
                                       control_ba_hi_dates) |> pull(route),
                     pc_df |> distinct(route, test, route_f, sample_f,
                                       control_ba_hi_dates) |> pull(test))) |>
  left_join(ctrl_q, by = c(c_route = "route", c_sample = "sample")) |>
  mutate(facet_lab = factor(paste0(sub("\n", " ", route_f), "\n", sample_f),
                            levels = levels(pc_df$facet_lab)))
stopifnot(!anyNA(band_c$q_lo))
XL_C <- c(min(c(band_c$q_lo, pc_df$ba_lo)) - 0.02, 1.03)

pc <- ggplot(pc_df, aes(y = recal_f)) +
  geom_rect(data = band_c, aes(xmin = q_lo, xmax = control_ba_hi_dates,
                               ymin = -Inf, ymax = Inf),
            inherit.aes = FALSE, fill = GREY_V, alpha = 0.8) +
  geom_vline(data = band_c, aes(xintercept = control_ba_hi_dates),
             colour = NEUTRAL, linewidth = 0.45) +
  geom_errorbar(aes(xmin = ba_lo, xmax = ba_hi), width = 0, linewidth = 0.5,
                colour = INK, orientation = "y") +
  geom_point(aes(x = balanced_accuracy, fill = exc), shape = 21, size = 1.9,
             stroke = 0.6, colour = INK) +
  facet_wrap(~facet_lab, nrow = 1) +
  scale_fill_manual(values = PAL_EXC, guide = "none") +
  scale_x_continuous(limits = XL_C, breaks = seq(0.4, 1, 0.2),
                     expand = expansion(0)) +
  labs(x = "Balanced accuracy, Landsat vs Sentinel-2", y = NULL) +
  theme(panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_line(colour = "#EDEDED", linewidth = 0.25),
        axis.text.y = element_text(lineheight = 0.85, size = 6.6),
        strip.text = element_text(size = 7, face = "plain", lineheight = 0.95),
        panel.spacing.x = unit(10, "pt"))

# ---------------------------------------------------------------- montaje ----
top <- (pa | pb) + plot_layout(widths = c(0.75, 1.5))
fig <- (top / pc) + plot_layout(heights = c(1.3, 0.9)) + tags_abc()
save_fig(fig, "fig2_harmonization", W2, 150)

# hechos para el pie de figura
cat("\n-- datos para el pie --\n")
print(as.data.frame(pb_df |> select(route, test, n, n_campaigns, balanced_accuracy,
                                    ba_lo, ba_hi, control_ba_hi_dates, exceeds_control)))
print(as.data.frame(pc_df |> select(route, test, recal, balanced_accuracy, ba_lo,
                                    ba_hi, control_ba_hi_dates, exceeds_control)))
print(as.data.frame(thr_chk))
cat("fig2 lista\n")
