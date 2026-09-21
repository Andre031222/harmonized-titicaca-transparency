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
# Los tres sectores del lago son toponimos, no terminos traducibles: el
# manuscrito los nombra en espanol y las figuras deben coincidir. El pipeline
# los exporta en dos formas (zona en mayusculas, zone_label sin tildes); la
# tilde se pone aqui, en la capa de presentacion, sin tocar las claves de los
# datos.
#
# zone_factor() aborta ante una zona que no reconoce. Esto no es paranoia: una
# busqueda fallida devuelve NA en silencio, y asi fue como fig04, fig06 y
# fig07 llegaron a un PDF con "Lago Mayor" y "Lago Menor" fundidos en una sola
# categoria "NA" sin que nada avisara.
ZONE_ORDER <- c("Bahía de Puno", "Lago Menor", "Lago Mayor")
ZONE_NAME  <- c("BAHIA PUNO"    = "Bahía de Puno",
                "LAGO MENOR"    = "Lago Menor",
                "LAGO MAYOR"    = "Lago Mayor",
                "Bahia de Puno" = "Bahía de Puno",
                "Lago Menor"    = "Lago Menor",
                "Lago Mayor"    = "Lago Mayor")

# Traduce valores de datos a etiquetas de presentacion y aborta si aparece un
# valor que el mapa no cubre. Traducir los NIVELES de un factor en vez de los
# datos manda los valores no traducidos a NA sin avisar; asi es como agosto
# (359 match-ups) acabo dibujado como una barra llamada "NA".
map_strict <- function(x, map, what = "valor") {
  out <- unname(map[as.character(x)])
  bad <- unique(as.character(x)[is.na(out)])
  if (length(bad)) stop(what, " no reconocido: ", paste(bad, collapse = ", "))
  out
}

zone_factor <- function(x) {
  out <- unname(ZONE_NAME[as.character(x)])
  bad <- unique(as.character(x)[is.na(out)])
  if (length(bad)) stop("zona no reconocida: ", paste(bad, collapse = ", "),
                        "  (revisar ZONE_NAME en theme_titicaca.R)")
  factor(out, levels = ZONE_ORDER)
}

PAL_ZONE <- setNames(c("#C2582C", "#D9A441", "#2E6E8E"), ZONE_ORDER)

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
# Estilo de las revistas Nature: la figura no lleva titulos ni subtitulos en
# los paneles (la explicacion va en el pie), solo la letra del panel en negrita
# minuscula; ejes en L, tipografia de 7-8 pt al tamano de impresion y una guia
# horizontal tenue en lugar de cuadricula.
theme_titicaca <- function(base_size = 8, base_family = "") {
  theme_classic(base_size = base_size, base_family = base_family) +
    theme(
      text              = element_text(colour = INK),
      plot.title        = element_blank(),
      plot.subtitle     = element_blank(),
      plot.caption      = element_blank(),
      axis.line         = element_line(colour = INK, linewidth = 0.3),
      axis.ticks        = element_line(colour = INK, linewidth = 0.3),
      axis.ticks.length = unit(2.2, "pt"),
      axis.title        = element_text(size = rel(1.0), colour = INK),
      axis.text         = element_text(size = rel(0.9), colour = INK_2),
      panel.grid.major.y = element_line(colour = "#EDEDED", linewidth = 0.25),
      panel.spacing     = unit(8, "pt"),
      strip.text        = element_text(size = rel(0.95), face = "bold",
                                       colour = INK, hjust = 0,
                                       margin = margin(b = 2, t = 2)),
      strip.background  = element_blank(),
      legend.title      = element_text(size = rel(0.9), colour = INK),
      legend.text       = element_text(size = rel(0.85), colour = INK_2),
      legend.key.size   = unit(8, "pt"),
      legend.margin     = margin(0, 0, 0, 0),
      legend.box.margin = margin(0, 0, 0, 0),
      legend.background = element_blank(),
      plot.tag          = element_text(size = 10, face = "bold", colour = INK),
      plot.tag.position = c(0, 1),
      plot.margin       = margin(8, 6, 4, 4)
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

#' Letras de panel a, b, c ... en negrita minuscula, como en Nature.
tags_abc <- function() {
  plot_annotation(tag_levels = "a",
                  theme = theme(plot.margin = margin(2, 2, 2, 2)))
}

#' Nube de lluvia: media densidad + caja estrecha + puntos. Requiere ggdist.
raincloud <- function(fill_alpha = 0.55, point_size = 0.7, width = 0.55,
                      side = "right") {
  list(
    ggdist::stat_halfeye(aes(fill = after_scale(alpha(colour, fill_alpha))),
                         adjust = 0.8, width = width, justification = -0.18,
                         .width = 0, point_colour = NA, side = side),
    geom_boxplot(width = 0.12, outlier.shape = NA, linewidth = 0.3,
                 fill = "white", colour = INK),
    ggbeeswarm::geom_quasirandom(size = point_size, alpha = 0.55, width = 0.13,
                                 shape = 16))
}

#' "n = ..." bajo cada grupo de un eje discreto.
n_labels <- function(df, x, y_pos, size = 2.3) {
  counts <- dplyr::count(df, {{ x }})
  geom_text(data = counts, aes(x = {{ x }}, y = y_pos, label = paste0("n = ", n)),
            inherit.aes = FALSE, size = size, colour = INK_2, vjust = 1)
}

#' Valor P con el formato de las revistas Nature: P = 0.03, P = 2.1 x 10^-5.
p_fmt <- function(p) {
  if (is.na(p)) return("")
  if (p >= 0.001) return(sprintf("italic(P) == %.3f", p))
  e <- floor(log10(p)); m <- p / 10^e
  sprintf("italic(P) == %.1f %%*%% 10^%d", m, e)
}

cat("theme_titicaca.R cargado | ROOT =", ROOT, "\n")
