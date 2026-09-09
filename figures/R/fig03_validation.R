# ============================================================================
# fig03_validation.R -- Por que la unidad de bloqueo es la CAMPANA
#
# (a) Modelos nulos: una unidad de bloqueo solo funciona si su propio nulo
#     colapsa  ella. El nulo de campana cae a ~0; los de zona y estacion no.
# (b) Jerarquia de validacion: aleatorio == estacion >> campana > anio, y el
#     colapso total  extrapolacion a una zona no vista.
# ============================================================================

source(file.path(local({a <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(a)) dirname(normalizePath(sub("^--file=", "", a[1]))) else getwd()}),
  "theme_titicaca.R"))

nulls <- read_tidy("null_models.csv")
hier  <- read_tidy("validation_hierarchy.csv")

# ---------------------------------------------------------------- panel (a) --
pa_df <- nulls |>
  mutate(label = recode(null_model,
                        zona           = "ZONE mean",
                        station        = "STATION mean",
                        campaign_date  = "CAMPAIGN mean"),
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
  labs(title = "(a)  Only campaign blocking neutralizes its own null model",
       subtitle = paste("R² of models predicting only the group mean, with",
                        "folds blocked by campaign.\nA blocking unit",
                        "is effective if and only if its null model collapses to zero under it",
                        ""),
       x = "Null model R²", y = NULL)

# ---------------------------------------------------------------- panel (b) --
LAB <- c(random_kfold          = "Random K-fold",
         by_station            = "STATION blocking",
         by_campaign_date      = "CAMPAIGN blocking",
         by_year               = "YEAR blocking",
         one_record_per_event  = "One record per event",
         `LAGO MENOR`          = "Extrapolate to Minor Lake",
         `BAHIA PUNO`          = "Extrapolate to Puno Bay",
         `LAGO MAYOR`          = "Extrapolate to Major Lake")

GRP <- c(random_kfold = "Validation design", by_station = "Validation design",
         by_campaign_date = "Validation design", by_year = "Validation design",
         one_record_per_event = "Validation design",
         `LAGO MENOR` = "Extrapolation to unseen zone",
         `BAHIA PUNO` = "Extrapolation to unseen zone",
         `LAGO MAYOR` = "Extrapolation to unseen zone")

pb_df <- hier |>
  filter(case %in% names(LAB)) |>
  mutate(name = LAB[case], grp = GRP[case],
         grp = factor(grp, levels = c("Validation design",
                                      "Extrapolation to unseen zone")),
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
  scale_colour_manual(values = c("Validation design" = "#2E6E8E",
                                 "Extrapolation to unseen zone" = ACCENT),
                      name = NULL) +
  scale_size_manual(values = c(`TRUE` = 3.1, `FALSE` = 2.0), guide = "none") +
  scale_x_continuous(limits = c(-0.30, 1.04), breaks = seq(-0.2, 0.8, 0.2),
                     expand = expansion(mult = c(0.01, 0))) +
  # Facetas arriba, no a la izquierda: la tira lateral chocaba con las
  # etiquetas del eje y.
  facet_wrap(~grp, ncol = 1, scales = "free_y") +
  labs(title = "(b) Station blocking changes nothing; extrapolation breaks everything",
       subtitle = paste("Out-of-fold R² of the Random Forest; the large point is",
                        "the primary design.\nThe fact that station blocking matches",
                        "with random does not prove absence of leakage:\nit proves that it",
                        "retains nothing"),
       x = "Out-of-fold R²", y = NULL) +
  theme(legend.position = "none",
        strip.text = element_text(face = "bold", colour = INK_2,
                                  size = rel(0.88), hjust = 0))

fig <- pa / pb + plot_layout(heights = c(1, 2.05))
save_fig(fig, "fig03_validation_design", W2, 168)
cat("fig03 lista\n")
