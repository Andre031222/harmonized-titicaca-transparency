# ============================================================================
# theme_titicaca.R -- tema, paleta y utilidades compartidas por todas las
# figuras del articulo. Un solo sitio para que las 9 figuras se lean como un
# sistema y no como nueve estilos distintos.
#
# Criterios de revista (Elsevier / JGLR):
#   - texto minimo 7 pt al tamano final de columna. OJO: una figura de 190 mm
#     se reduce a ~140 mm en el PDF de revision (74 %), asi que el tamano base
#     es 11 pt para que al escalar siga por encima de 8 pt.
#   - figuras de 1 columna = 90 mm, 1.5 columnas = 140 mm, 2 columnas = 190 mm
#   - sin dependencia del color para leer el mensaje (paleta segura para
#     daltonismo y legible en escala de grises)
# ============================================================================

# Anade la libreria auxiliar donde se instalaron ragg/ggtext/ggdist.
.extra_lib <- path.expand("~/R/library")
if (dir.exists(.extra_lib)) .libPaths(c(.libPaths(), .extra_lib))

suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(tidyr); library(readr)
  library(scales);  library(patchwork); library(forcats); library(stringr)
})

# --- rutas ------------------------------------------------------------------
# Sube directorios hasta encontrar results/tidy, para que los scripts corran
# igual desde figures/R/ que desde la raiz del proyecto.
find_root <- function(start = getwd()) {
  p <- normalizePath(start, mustWork = FALSE)
  for (i in 1:6) {
    if (dir.exists(file.path(p, "results", "tidy"))) return(p)
    p <- normalizePath(file.path(p, ".."), mustWork = FALSE)
  }
  stop("No encuentro results/tidy/ subiendo desde ", start)
}
ROOT <- find_root()
TIDY <- file.path(ROOT, "results", "tidy")
FIGR <- file.path(ROOT, "results", "figures_r")
dir.create(FIGR, showWarnings = FALSE, recursive = TRUE)

# --- anchos de columna (mm) -------------------------------------------------
W1  <- 90;  W15 <- 140; W2 <- 190

# --- paleta -----------------------------------------------------------------
# Zonas troficas: de turbio (calido) a claro (frio), ordenadas por
# transparencia media, de modo que el color codifique la variable real.
PAL_ZONE <- c("Bahia de Puno" = "#C2582C",
              "Minor Lake"    = "#D9A441",
              "Major Lake"    = "#2E6E8E")

PAL_SENSOR <- c("Sentinel-2"  = "#2E6E8E",
                "Landsat 8/9" = "#C2582C")

# Semaforo de veredicto, usado en armonizacion y validacion.
PAL_VERDICT <- c("bueno" = "#3D7A57", "aceptable" = "#7FA663",
                 "debil" = "#D9A441", "roto" = "#C2582C")

INK    <- "#1A1A1A"
INK_2  <- "#4A4A4A"
GRID   <- "#DFDFDF"
ACCENT <- "#C2582C"
NEUTRAL<- "#8C8C8C"

# --- tema -------------------------------------------------------------------
theme_titicaca <- function(base_size = 11, base_family = "") {
  theme_minimal(base_size = base_size, base_family = base_family) +
    theme(
      text             = element_text(colour = INK),
      plot.title       = element_text(size = rel(1.05), face = "bold",
                                      colour = INK, hjust = 0,
                                      margin = margin(b = 2)),
      plot.subtitle    = element_text(size = rel(0.92), colour = INK_2,
                                      hjust = 0, margin = margin(b = 7),
                                      lineheight = 1.15),
      plot.caption     = element_text(size = rel(0.80), colour = INK_2,
                                      hjust = 0, margin = margin(t = 7),
                                      lineheight = 1.15),
      plot.title.position   = "plot",
      plot.caption.position = "plot",
      axis.title       = element_text(size = rel(0.95), colour = INK_2),
      axis.text        = element_text(size = rel(0.88), colour = INK_2),
      axis.ticks       = element_line(colour = GRID, linewidth = 0.3),
      axis.ticks.length = unit(2, "pt"),
      panel.grid.major = element_line(colour = GRID, linewidth = 0.3),
      panel.grid.minor = element_blank(),
      panel.spacing    = unit(9, "pt"),
      strip.text       = element_text(size = rel(0.92), face = "bold",
                                      colour = INK, hjust = 0,
                                      margin = margin(b = 3, t = 3)),
      strip.background = element_blank(),
      legend.title     = element_text(size = rel(0.90), colour = INK_2),
      legend.text      = element_text(size = rel(0.88), colour = INK_2),
      legend.key.size  = unit(9, "pt"),
      legend.margin    = margin(0, 0, 0, 0),
      legend.box.margin= margin(0, 0, 0, 0),
      plot.margin      = margin(6, 8, 5, 6)
    )
}

theme_set(theme_titicaca())

# --- utilidades -------------------------------------------------------------
read_tidy <- function(name) {
  readr::read_csv(file.path(TIDY, name), show_col_types = FALSE,
                  progress = FALSE)
}

#' Guarda en PNG (300 dpi, para revision) y PDF vectorial (para produccion).
save_fig <- function(plot, name, width_mm, height_mm, dpi = 300) {
  png_path <- file.path(FIGR, paste0(name, ".png"))
  pdf_path <- file.path(FIGR, paste0(name, ".pdf"))
  dev <- if (requireNamespace("ragg", quietly = TRUE)) ragg::agg_png else "png"
  ggsave(png_path, plot, width = width_mm, height = height_mm, units = "mm",
         dpi = dpi, device = dev, bg = "white")
  ggsave(pdf_path, plot, width = width_mm, height = height_mm, units = "mm",
         device = cairo_pdf, bg = "white")
  cat(sprintf("  guardado  %-42s %3.0f x %3.0f mm\n",
              basename(png_path), width_mm, height_mm))
  invisible(png_path)
}

#' Etiqueta de metricas lista para anotar dentro de un panel.
metric_label <- function(r2, rmse, n = NULL, mae = NULL) {
  s <- sprintf("R² = %.3f\nRMSE = %.2f m", r2, rmse)
  if (!is.null(mae)) s <- paste0(s, sprintf("\nMAE = %.2f m", mae))
  if (!is.null(n))   s <- paste0(s, sprintf("\nn = %d", n))
  s
}

#' Etiqueta de panel (a), (b), ... con el estilo de la revista.
tag_theme <- function() {
  theme(plot.tag = element_text(size = rel(1.05), face = "bold", colour = INK),
        plot.tag.position = c(0.005, 0.995))
}

cat("theme_titicaca.R cargado | ROOT =", ROOT, "\n")
