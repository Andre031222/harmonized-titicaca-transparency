# ============================================================================
# fig10_data_flow.R -- Flujo de datos: del registro de IMARPE a cada analisis
#
# Todos los conteos salen de los archivos del repositorio salvo los dos del
# registro completo (881 mediciones, 734 con Secchi), que no se redistribuye
# y se documentan en la Seccion 2.2 del articulo.
# ============================================================================

source(file.path(local({a <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(a)) dirname(normalizePath(sub("^--file=", "", a[1]))) else getwd()}),
  "theme_titicaca.R"))

RECORD_N <- 881; RECORD_SECCHI <- 734          # registro IMARPE 2011-2024

s2 <- read_csv(file.path(ROOT, "data/processed/matchups_s2.csv"), show_col_types = FALSE)
ls <- read_csv(file.path(ROOT, "data/processed/matchups_ls.csv"), show_col_types = FALSE)
d  <- read_csv(file.path(ROOT, "data/processed/analysis_dataset.csv"), show_col_types = FALSE)
sep <- read_tidy("sensor_separability.csv")

# una medicion es un par (estacion, fecha); row_id es solo el indice de cada tabla
meas <- bind_rows(s2 |> select(station, date, secchi, matched),
                  ls |> select(station, date, secchi, matched)) |>
  mutate(matched = as.logical(matched))
cand    <- nrow(distinct(meas, station, date))
cand_sd <- nrow(distinct(filter(meas, !is.na(secchi)), station, date))
hit     <- nrow(distinct(filter(meas, matched), station, date))
n_camp  <- n_distinct(format(as.Date(meas$date), "%Y"))   # una campana por anio
m_s2 <- sum(as.logical(s2$matched)); m_ls <- sum(as.logical(ls$matched))
a_s2 <- sum(d$sensor == "S2");       a_ls <- sum(d$sensor == "LS")
ev   <- d |> distinct(station, campaign_date, sensor) |>
  count(station, campaign_date)
n_ev <- nrow(ev); n_both <- sum(ev$n == 2)
n_pm3 <- sep$n[sep$test == "pm3d_hls"]
fmtn <- function(x) format(x, big.mark = ",")

# --- cajas: x, y del centro, ancho, alto, texto, tipo -----------------------
B <- tribble(
  ~id, ~x, ~y, ~w, ~h, ~kind, ~head, ~body,
  "rec", 0.36, 0.93, 0.50, 0.10, "main", "IMARPE in-situ record, 2011–2024",
    sprintf("%s measurements · %s with Secchi · 13 field campaigns",
            fmtn(RECORD_N), fmtn(RECORD_SECCHI)),
  "era", 0.36, 0.76, 0.50, 0.10, "main", "Landsat 8/9 era, 2013–2024",
    sprintf("%s measurements in %d campaigns · %s with Secchi", fmtn(cand), n_camp, fmtn(cand_sd)),
  "mu",  0.36, 0.57, 0.50, 0.12, "main", "Satellite–field match-ups",
    sprintf("%s in total: %s Landsat 8/9 OLI · %s Sentinel-2 MSI\n±10 days, true station coordinate, cloud-masked",
            fmtn(m_ls + m_s2), fmtn(m_ls), fmtn(m_s2)),
  "sec", 0.36, 0.38, 0.50, 0.12, "core", "Secchi match-ups (analysis dataset)",
    sprintf("%s: %s Landsat 8/9 · %s Sentinel-2\n%s stations · %s sampling dates · 9 field campaigns",
            fmtn(a_ls + a_s2), fmtn(a_ls), fmtn(a_s2),
            n_distinct(d$station), n_distinct(d$campaign_date)),
  "ev",  0.36, 0.19, 0.50, 0.10, "main", "Unique field events",
    sprintf("%s events · %s observed by both sensors", fmtn(n_ev), fmtn(n_both)),
  # usos, a la derecha
  "u1", 0.82, 0.57, 0.30, 0.10, "use", "Retrievability (Section 3.9)",
    "Chl-a, suspended solids, temperature;\nthermal band on Landsat match-ups",
  "u2", 0.82, 0.41, 0.30, 0.10, "use", "Retrieval and validation",
    "Random Forest, baselines, null-model\ngrid, conformal intervals (3.2–3.8)",
  "u3", 0.82, 0.27, 0.30, 0.10, "use", "Sensor two-sample test (3.1)",
    sprintf("all match-ups; paired events;\n±3-day pairs (%d match-ups)", n_pm3),
  "u4", 0.82, 0.13, 0.30, 0.10, "use", "Pseudo-replication check (3.2)",
    "one record per field event"
)
# exclusiones, a la izquierda de cada flecha principal
X <- tribble(
  ~y, ~lab,
  0.845, sprintf("%d measurements from the %d campaigns\nof 2011–2012, before Landsat 8/9,\nincl. the only wet-season campaign",
                 RECORD_N - cand, 13 - n_camp),
  0.665, sprintf("%d measurements with no cloud-free\nimage of either sensor", cand - hit),
  0.475, sprintf("%d match-ups without a Secchi reading", (m_ls + m_s2) - (a_ls + a_s2)),
  0.285, sprintf("%d match-ups collapsed: second sensor\nor duplicate record of the same event",
                 (a_ls + a_s2) - n_ev)
)
FILL <- c(main = "#F2F2F2", core = "#DCE9F0", use = "white")
EDGE <- c(main = "#8C8C8C", core = "#2E6E8E", use = "#8C8C8C")

B <- B |> mutate(xmin = x - w / 2, xmax = x + w / 2, ymin = y - h / 2, ymax = y + h / 2)
chain <- c("rec", "era", "mu", "sec", "ev")
arr <- tibble(from = head(chain, -1), to = tail(chain, -1)) |>
  left_join(B |> select(from = id, y0 = ymin, x0 = x), by = "from") |>
  left_join(B |> select(to = id, y1 = ymax), by = "to")
uses <- tibble(from = c("mu", "sec", "sec", "ev"), to = c("u1", "u2", "u3", "u4")) |>
  left_join(B |> select(from = id, x0 = xmax, y0 = y), by = "from") |>
  left_join(B |> select(to = id, x1 = xmin, y1 = y), by = "to")

ARROW <- arrow(length = unit(4, "pt"), type = "closed")
p <- ggplot() +
  geom_segment(data = arr, aes(x = x0, xend = x0, y = y0, yend = y1 + 0.004),
               colour = INK_2, linewidth = 0.45, arrow = ARROW) +
  geom_segment(data = uses, aes(x = x0, xend = x1 - 0.004, y = y0, yend = y1),
               colour = "#9DBCCD", linewidth = 0.4, arrow = ARROW) +
  geom_text(data = X, aes(x = 0.115, y = y, label = lab), hjust = 1, size = 2.35,
            colour = ACCENT, lineheight = 0.95) +
  geom_segment(data = X, aes(x = 0.125, xend = 0.355, y = y, yend = y),
               colour = ACCENT, linewidth = 0.3, linetype = "22") +
  geom_rect(data = B, aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax,
                          fill = kind, colour = kind), linewidth = 0.45) +
  geom_text(data = B, aes(x = x, y = ymax - 0.022, label = head), fontface = "bold",
            size = 2.75, colour = INK, vjust = 1) +
  geom_text(data = B, aes(x = x, y = ymax - 0.052, label = body), size = 2.4,
            colour = INK_2, vjust = 1, lineheight = 0.95) +
  scale_fill_manual(values = FILL, guide = "none") +
  scale_colour_manual(values = EDGE, guide = "none") +
  coord_cartesian(xlim = c(-0.22, 0.98), ylim = c(0.07, 0.99), expand = FALSE) +
  theme_void()

save_fig(p, "fig10_data_flow", W2, 125)
cat("fig10 lista\n")
