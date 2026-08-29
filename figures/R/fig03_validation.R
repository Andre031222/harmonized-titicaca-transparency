# ============================================================================
# fig03_validation.R -- Por que la unidad de bloqueo es la CAMPANA
#
# (a) Modelos nulos: una unidad de bloqueo solo funciona si su propio nulo
#     colapsa bajo ella. El nulo de campana cae a ~0; los de zona y estacion no.
# (b) Jerarquia de validacion: aleatorio == estacion >> campana > anio, y el
#     colapso total bajo extrapolacion a una zona no vista.
# ============================================================================

source(file.path(local({a <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(a)) dirname(normalizePath(sub("^--file=", "", a[1]))) else getwd()}),
  "theme_titicaca.R"))

nulls <- read_tidy("null_models.csv")
hier  <- read_tidy("validation_hierarchy.csv")

# ---------------------------------------------------------------- panel (a) --
pa_df <- nulls |>
  mutate(label = recode(null_model,
                        zona           = "Media de la ZONA",
                        station        = "Media de la ESTACIÓN",
                        campaign_date  = "Media de la CAMPAÑA"),
         collapsed = R2 < 0.05,
         label = fct_reorder(label, R2))

pa <- ggplot(pa_df, aes(R2, label, fill = collapsed)) +
  geom_vline(xintercept = 0, colour = INK, linewidth = 0.4) +
  geom_col(width = 0.55) +
  geom_text(aes(label = sprintf("%+.3f", R2),
                hjust = ifelse(R2 > 0.05, -0.18, 1.18)),
            size = 2.6, colour = INK, fontface = "bold") +
  scale_fill_manual(values = c(`TRUE` = "#3D7A57", `FALSE` = ACCENT),
                    guide = "none") +
  scale_x_continuous(limits = c(-0.12, 0.40),
                     breaks = seq(-0.1, 0.4, 0.1),
                     expand = expansion(mult = c(0, 0.02))) +
  labs(title = "(a)  Sólo el bloqueo por campaña neutraliza su propio nulo",
       subtitle = paste("R² de modelos que sólo predicen la media del grupo, con",
                        "folds bloqueados por campaña.\nUna unidad de bloqueo",
                        "sirve si, y sólo si, su propio nulo colapsa a cero bajo",
                        "ella"),
       x = "R² del modelo nulo", y = NULL)

# ---------------------------------------------------------------- panel (b) --
LAB <- c(random_kfold          = "K-fold aleatorio",
         by_station            = "Bloqueo por ESTACIÓN",
         by_campaign_date      = "Bloqueo por CAMPAÑA",
         by_year               = "Bloqueo por AÑO",
         one_record_per_event  = "Un registro por evento",
         `LAGO MENOR`          = "Extrapolar a Lago Menor",
         `BAHIA PUNO`          = "Extrapolar a Bahía de Puno",
         `LAGO MAYOR`          = "Extrapolar a Lago Mayor")

GRP <- c(random_kfold = "Diseño de validación", by_station = "Diseño de validación",
         by_campaign_date = "Diseño de validación", by_year = "Diseño de validación",
         one_record_per_event = "Diseño de validación",
         `LAGO MENOR` = "Extrapolación a zona no vista",
         `BAHIA PUNO` = "Extrapolación a zona no vista",
         `LAGO MAYOR` = "Extrapolación a zona no vista")

pb_df <- hier |>
  filter(case %in% names(LAB)) |>
  mutate(name = LAB[case], grp = GRP[case],
         grp = factor(grp, levels = c("Diseño de validación",
                                      "Extrapolación a zona no vista")),
         primary = case == "by_campaign_date",
         name = fct_reorder(name, R2))

pb <- ggplot(pb_df, aes(R2, name)) +
  geom_vline(xintercept = 0, colour = INK, linewidth = 0.4) +
  geom_segment(aes(x = 0, xend = R2, yend = name, colour = grp),
               linewidth = 0.9) +
  geom_point(aes(colour = grp, size = primary)) +
  geom_text(aes(label = sprintf("%+.3f", R2),
                hjust = ifelse(R2 > 0.05, -0.28, 1.28)),
            size = 2.5, colour = INK, fontface = "bold") +
  geom_text(aes(label = sprintf("RMSE %.2f m", RMSE), x = 1.02),
            hjust = 1, size = 2.3, colour = INK_2) +
  scale_colour_manual(values = c("Diseño de validación" = "#2E6E8E",
                                 "Extrapolación a zona no vista" = ACCENT),
                      name = NULL) +
  scale_size_manual(values = c(`TRUE` = 3.1, `FALSE` = 2.0), guide = "none") +
  scale_x_continuous(limits = c(-0.30, 1.04), breaks = seq(-0.2, 0.8, 0.2),
                     expand = expansion(mult = c(0.01, 0))) +
  # Facetas arriba, no a la izquierda: la tira lateral chocaba con las
  # etiquetas del eje y.
  facet_wrap(~grp, ncol = 1, scales = "free_y") +
  labs(title = "(b)  El bloqueo por estación no cambia nada; la extrapolación lo rompe todo",
       subtitle = paste("R² fuera de fold del Random Forest; el punto grande es",
                        "el diseño primario.\nQue el bloqueo por estación coincida",
                        "con el aleatorio no prueba ausencia de fuga:\nprueba que",
                        "no retiene nada"),
       x = "R² fuera de fold", y = NULL) +
  theme(legend.position = "none",
        strip.text = element_text(face = "bold", colour = INK_2,
                                  size = rel(0.88), hjust = 0))

fig <- pa / pb + plot_layout(heights = c(1, 2.05))
save_fig(fig, "fig03_validation_design", W2, 168)
cat("fig03 lista\n")
