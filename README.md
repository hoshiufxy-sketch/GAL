# Figure code

Plotting and campaign code for the manuscript.

## Layout

```
code/            all Python
data/            all inputs and recorded results
wet-lab data/    the experimental workbook
```

## code/

**Libraries** — imported by the scripts below, not run directly.

| file | provides |
|---|---|
| `ensemble_learner.py` | `make_learner(family, seed)` → `xgb_ens`, `gp`, `dnn_ens`; each returns `fit()` and `predict()` giving a mean and a spread |
| `run.py` | `features(frame)` → the 128-descriptor matrix (length, composition, 1–3-mer frequencies, positional one-hot, five physicochemical terms) |
| `gal/` | `extract_complete_features(spacer)` — the ViennaRNA descriptors |

`run.py` has a command-line entry point left over from the original working tree; run it as
a module, do not execute the file.

**Figure scripts** — each writes its figure beside itself in `code/`.

| script | figure |
|---|---|
| `fig1_fit_quality_bars.py` | R² and Spearman ρ, in-sample against grouped out-of-fold, three learners |
| `fig2_plot_landscape_neighbour_en.py` | expression landscape, Hamming-distance profile, distance-1 yield |
| `make_fig3_strategies.py` | top-20 found per round, six acquisition arms, one panel per feature set |
| `make_figE_gal_vs_al_3learner.py` | cumulative hit rate, GAL against AL, three learners per feature set |
| `make_fig_best_value_comparison.py` | highest value reached by each approach |
| `make_fig_ccac_motif.py` | CCAC distribution, the two downstream bases, the model's ranking |
| `make_fig_ccac_length_bars.py` | expression by spacer length at fixed 5′-CCAC |
| `make_fig_ccac_position_bars.py` | expression by CCAC start position, designed spacers |
| `make_fig_ccac_position_bars_unified.py` | the same, pooled across library and designed set |
| `make_fig_rnafold_split.py` | base-pair-height and positional-entropy profiles |
| `plot_violin.py` | expression distribution of initial library, Round 1, Round 2 |
| `annotate_wt.py` | the above with wild-type significance annotation |

**Campaign script**

| script | produces |
|---|---|
| `run_strategies_fig3.py` | `data/strategies_fig3.csv`, the recorded campaigns behind `make_fig3_strategies.py`. Full run ≈ 15 min. |

## data/

`sequence_means.csv`, `nodes.csv`, `hamming1_pairs.csv`, `distance_profiles.csv`,
`audit.json` and `fit_quality_3models.csv` are the measured library and its derived
structural tables.

`strategies_fig3.csv` and `curves_3learner.csv` are campaign results over the 421-member
library. `curves_3features_nt.csv` is a separate campaign over the 417 members for which
all three feature sets exist; `make_fig3_strategies.py` and
`make_figE_gal_vs_al_3learner.py` draw their third panel from it.

`ccac28_extracted.csv` holds the 28 designed CCAC spacers. `New Data.xlsx` is the
experimental workbook. The remaining `.json` files are extraction caches for the violin
figures.

## Running

```
pip install -r requirements.txt
cd code
python make_fig3_strategies.py
```

Each figure script runs in place; nothing needs to be configured.
