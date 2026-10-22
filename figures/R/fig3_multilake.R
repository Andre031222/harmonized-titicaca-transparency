# ============================================================================
# fig3_multilake.R -- La diferencia residual entre HLSL30 (Landsat 8/9) y
# HLSS30 (Sentinel-2) no es propia del Titicaca: aparece en pares del mismo
# dia sobre agua abierta de 28 lagos grandes, y crece cuanto mas clara es el
# agua.
#
# (a) Exactitud balanceada del clasificador de sensor (validacion agrupada por
#     fecha) frente al indice de claridad (mediana de la reflectancia verde de
#     Sentinel-2, eje log), un punto por lago con tamano = n pares; controles
#     negativos (mitades de fechas dentro de un sensor) en gris; rho de
#     Spearman anotado. Titicaca resaltado.
# (b) Razon de medianas Landsat / Sentinel-2 en azul, verde y rojo frente al
#     mismo indice de claridad, con la referencia 1:1 (razon = 1).
# (c) Pruebas conjuntas: agrupada por lago-fecha, dejando lagos fuera y
#     control negativo (S2 contra S2).
#
# Entradas: results/tidy/multilake_hls_separability.csv,
# results/tidy/multilake_hls_bands.csv, results/metrics/multilake_hls.json
# (todas escritas por pipeline/p12_multilake_hls.py). Ningun numero a mano.
# ============================================================================

source(file.path(local({a <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(a)) dirname(normalizePath(sub("^--file=", "", a[1]))) else getwd()}),
  "theme_titicaca.R"))

sep   <- read_tidy("multilake_hls_separability.csv") |> filter(tested)
bands <- read_tidy("multilake_hls_bands.csv")
# p12 escribe NaN literal (JSON no valido) en algunos campos por lago; se
# lee como null. Los campos usados aqui no estan afectados.
met   <- jsonlite::fromJSON(gsub("\\bNaN\\b", "null", paste(readLines(
  file.path(ROOT, "results", "metrics", "multilake_hls.json"), warn = FALSE), collapse = "\n")))

stopifnot(nrow(sep) == met$n_lakes_tested)

# rho de Spearman: se recalcula para obtener un P no redondeado (el JSON
# guarda p = 0.0 por redondeo a 4 decimales) y se comprueba contra el JSON.
ct <- suppressWarnings(cor.test(sep$s2_green_median, sep$balanced_accuracy,
                                method = "spearman", exact = FALSE))
rho_json <- met$spearman_vs_s2_green$balanced_accuracy_vs_clarity$rho
stopifnot(abs(unname(ct$estimate) - rho_json) < 1e-3)

is_tc <- function(x) grepl("^Titicaca", x)
COL_TC   <- ACCENT
COL_LAKE <- "#2E6E8E"
COL_CTRL <- "#A8A8A8"

xlim_g <- range(sep$s2_green_median) * c(0.8, 1.25)
x_breaks <- c(0.002, 0.005, 0.01, 0.02, 0.05, 0.1)
x_lab <- "Median Sentinel-2 green reflectance (clarity index)"

# ---------------------------------------------------------------- panel (a) --
ctrl <- sep |>
  select(lake, s2_green_median, control_LS_bacc, control_S2_bacc) |>
  pivot_longer(starts_with("control"), names_to = "ctrl", values_to = "bacc")
ctrl_rng <- range(ctrl$bacc)

MAX_SIZE_A <- 4.2
# Posiciones de las etiquetas, puestas a mano y revisadas en el PNG: con 28
# lagos apinados en x = 0.005-0.017, ggrepel dejaba nombres pegados al punto
# vecino (etiqueta ambigua). Son coordenadas de DISENO (donde va el texto), no
# resultados: la linea guia sale siempre del punto real del CSV.
# lx, ly: posicion del texto; hj: justificacion horizontal; seg: linea guia.
LAB_POS <- tribble(
  ~lake,            ~lx,    ~ly,   ~hj,  ~seg,
  "Ladoga",         0.0022, 0.998, 0.3,  FALSE,
  "Tahoe",          0.0046, 1.012, 0.5,  FALSE,
  "Superior",       0.0056, 1.037, 1,    TRUE,
  "Titicaca Mayor", 0.0068, 1.037, 0,    TRUE,
  "Baikal",         0.0088, 0.985, 0.5,  FALSE,
  "Vanern",         0.0120, 1.002, 0,    TRUE,
  "Issyk-Kul",      0.0120, 0.978, 0,    TRUE,
  "Victoria",       0.0120, 0.954, 0,    TRUE,
  "Titicaca Menor", 0.0190, 0.930, 0,    FALSE,
  "Tanganyika",     0.0044, 0.958, 1,    TRUE,
  "Taupo",          0.0044, 0.936, 1,    TRUE,
  "Michigan",       0.0044, 0.914, 1,    TRUE,
  "Malawi",         0.0044, 0.892, 1,    TRUE,
  "Ontario",        0.0070, 0.855, 0.5,  TRUE,
  "Garda",          0.0094, 0.877, 0.5,  TRUE,
  "Okeechobee",     0.0130, 0.856, 0.5,  TRUE,
  "Geneva",         0.0172, 0.882, 0,    TRUE,
  "Van",            0.0182, 0.910, 0,    FALSE,
  "Taihu",          0.0859, 0.905, 0.5,  FALSE,
  "Chapala",        0.0695, 0.865, 0.5,  FALSE,
  "Balaton",        0.1129, 0.815, 0.75, FALSE,
  "Poyang",         0.0793, 0.742, 0.5,  FALSE,
  "Erie",           0.0210, 0.808, 1,    FALSE,
  "Nasser",         0.0314, 0.774, 0.5,  FALSE,
  "Nicaragua",      0.0258, 0.843, 1,    FALSE,
  "Qinghai",        0.0300, 0.821, 0.5,  FALSE,
  "Winnipeg",       0.0405, 0.809, 0,    FALSE,
  "Great Slave",    0.0370, 0.868, 0.5,  FALSE)
stopifnot(setequal(LAB_POS$lake, sep$lake))
pa_df <- sep |> mutate(tc = is_tc(lake)) |> left_join(LAB_POS, by = "lake")
# la guia termina un poco antes del texto, del lado en que esta justificado
seg_df <- pa_df |> filter(seg) |>
  mutate(xend = lx * case_when(hj == 1 ~ 1.03, hj == 0 ~ 1 / 1.03, TRUE ~ 1),
         yend = ly + ifelse(hj == 0.5, ifelse(ly > balanced_accuracy, -0.007, 0.007), 0))
rho_lab <- sprintf("Spearman~rho == %.2f*','~~%s*','~~italic(n) == %d",
                   rho_json, p_fmt(ct$p.value), nrow(sep))

pa <- ggplot(pa_df, aes(s2_green_median, balanced_accuracy)) +
  geom_hline(yintercept = 0.5, linetype = "22", colour = INK_2, linewidth = 0.3) +
  geom_point(data = ctrl, aes(y = bacc), inherit.aes = TRUE, shape = 1,
             size = 1.1, colour = COL_CTRL, stroke = 0.4) +
  geom_segment(data = seg_df, aes(xend = xend, yend = yend), colour = "#9A9A9A",
               linewidth = 0.2) +
  geom_point(aes(size = n_pairs, fill = tc), shape = 21, colour = "white",
             stroke = 0.3, alpha = 0.95) +
  geom_text(aes(lx, ly, label = lake, hjust = hj, colour = tc,
                fontface = ifelse(tc, "bold", "plain")), size = 2.4) +
  annotate("text", x = xlim_g[2], y = 0.5, label = "chance", hjust = 1.05,
           vjust = -0.5, size = 2.35, colour = INK_2) +
  annotate("text", x = xlim_g[1] * 1.04, y = ctrl_rng[1] - 0.02, hjust = 0, vjust = 1,
           size = 2.35, colour = INK_2,
           label = "Negative controls (date halves within one sensor)") +
  annotate("text", x = xlim_g[2], y = 1.06, label = rho_lab, parse = TRUE,
           hjust = 1, vjust = 1, size = 2.4, colour = INK) +
  scale_fill_manual(values = c(`TRUE` = COL_TC, `FALSE` = COL_LAKE), guide = "none") +
  scale_colour_manual(values = c(`TRUE` = COL_TC, `FALSE` = INK), guide = "none") +
  scale_size_area(max_size = MAX_SIZE_A, name = "Same-day pairs",
                  breaks = c(2000, 4000, 8000), labels = label_comma()) +
  guides(size = guide_legend(override.aes = list(fill = COL_LAKE, colour = "white"))) +
  scale_x_log10(limits = xlim_g, breaks = x_breaks, labels = label_number(drop0trailing = TRUE),
                expand = c(0, 0)) +
  scale_y_continuous(limits = c(0.3, 1.06), breaks = seq(0.3, 1, 0.1),
                     expand = c(0, 0)) +
  annotation_logticks(sides = "b", short = unit(1, "pt"), mid = unit(1.6, "pt"),
                      long = unit(2.2, "pt"), linewidth = 0.25, colour = INK_2) +
  labs(x = x_lab, y = "Balanced accuracy, Landsat vs Sentinel-2") +
  theme(legend.position = "inside", legend.position.inside = c(0.03, 0.47),
        legend.justification = c(0, 0), legend.direction = "horizontal",
        legend.title.position = "top")

# ---------------------------------------------------------------- panel (b) --
BAND_LAB <- c(B2 = "Blue", B3 = "Green", B4 = "Red")
PAL_BAND <- c(Blue = "#0072B2", Green = "#009E73", Red = "#D55E00")
LOW_SIG <- 1e-3   # mediana de S2 por debajo de 0.001: razon inestable
pb_df <- bands |>
  filter(band %in% names(BAND_LAB)) |>
  left_join(select(sep, lake, s2_green_median), by = "lake") |>
  mutate(band = factor(BAND_LAB[band], levels = BAND_LAB),
         low = median_S2 < LOW_SIG, tc = is_tc(lake))
n_na <- sum(is.na(pb_df$ratio_LS_S2))
pb_plot <- filter(pb_df, !is.na(ratio_LS_S2))

pb <- ggplot(pb_plot, aes(s2_green_median, ratio_LS_S2, colour = band)) +
  geom_hline(yintercept = 1, colour = INK, linewidth = 0.35) +
  geom_smooth(data = filter(pb_plot, !low), aes(group = band), method = "loess",
              formula = y ~ x, se = FALSE, span = 1, linewidth = 0.5) +
  geom_point(aes(shape = low), size = 1.2, stroke = 0.45) +
  geom_point(data = filter(pb_plot, tc), shape = 0, size = 2.6, stroke = 0.4,
             colour = COL_TC, show.legend = FALSE) +
  scale_colour_manual(values = PAL_BAND, name = NULL) +
  scale_shape_manual(values = c(`FALSE` = 16, `TRUE` = 1), guide = "none") +
  scale_x_log10(limits = xlim_g, breaks = x_breaks, labels = label_number(drop0trailing = TRUE),
                expand = c(0, 0)) +
  scale_y_log10(breaks = c(0.25, 0.5, 1, 2), labels = label_number(drop0trailing = TRUE)) +
  annotation_logticks(sides = "b", short = unit(1, "pt"), mid = unit(1.6, "pt"),
                      long = unit(2.2, "pt"), linewidth = 0.25, colour = INK_2) +
  labs(x = "Median S2 green reflectance", y = "Landsat / Sentinel-2 median ratio") +
  guides(colour = guide_legend(override.aes = list(linewidth = 0, size = 1.6))) +
  theme(legend.position = "inside", legend.position.inside = c(1, 1),
        legend.justification = c(1, 1), legend.key.size = unit(7, "pt"),
        legend.spacing.y = unit(0, "pt"))

# ---------------------------------------------------------------- panel (c) --
POOL_LAB <- c(pooled_lake_date = "Grouped by\nlake-date",
              pooled_leave_lake_out = "Leave-lake-\nout",
              pooled_control_S2 = "Control\n(S2 vs S2)")
pc_df <- tibble(test = names(POOL_LAB),
                bacc = sapply(names(POOL_LAB), \(k) met[[k]]$balanced_accuracy),
                n = sapply(names(POOL_LAB), \(k) met[[k]]$n)) |>
  mutate(test = factor(POOL_LAB[test], levels = POOL_LAB),
         ctrl = grepl("Control", test))
pc <- ggplot(pc_df, aes(test, bacc)) +
  geom_col(aes(fill = ctrl), width = 0.6) +
  geom_hline(yintercept = 0.5, linetype = "22", colour = INK_2, linewidth = 0.3) +
  geom_text(aes(label = sprintf("%.2f", bacc)), vjust = -0.5, size = 2.3, colour = INK) +
  scale_fill_manual(values = c(`FALSE` = COL_LAKE, `TRUE` = COL_CTRL), guide = "none") +
  scale_y_continuous(limits = c(0, 1.08), breaks = seq(0, 1, 0.25),
                     expand = c(0, 0)) +
  labs(x = NULL, y = "Pooled balanced accuracy") +
  theme(axis.text.x = element_text(size = 6.6, lineheight = 0.9))

# ------------------------------------------------------------------ montaje --
fig <- pa + (pb / pc + plot_layout(heights = c(1.25, 1))) +
  plot_layout(widths = c(1.65, 1)) + tags_abc()

save_fig(fig, "fig3_multilake", W2, 135)
cat(sprintf("  rho = %.3f (json %.3f), P = %.2g, n lagos = %d, razones NA = %d, debiles = %d\n",
            unname(ct$estimate), rho_json, ct$p.value, nrow(sep), n_na,
            sum(pb_plot$low)))
