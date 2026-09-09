# Resuelve la carpeta del script tanto con Rscript como con source() interactivo,
# y carga el tema compartido. Todas las figuras empiezan con:
#   source(file.path(dirname(sys.frame(1)$ofile %||% "."), "_bootstrap.R"))
# ...pero eso falla  Rscript, asi que se usa este patron en su lugar.
.script_dir <- local({
  a <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(a)) dirname(normalizePath(sub("^--file=", "", a[1]))) else getwd()
})
source(file.path(.script_dir, "theme_titicaca.R"))
