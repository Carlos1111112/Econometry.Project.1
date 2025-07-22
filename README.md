# Supply and Demand Equilibrium Script

This script fits configurable models (linear or non-linear) to supply and demand
observations, computes the market equilibrium and price elasticities, and plots
the resulting curves. Data can be entered manually or loaded from CSV files.

## Usage

```
python Econometry.8.py \
    --mode csv \
    --demand-file demand.csv \
    --supply-file supply.csv \
    --model demand:poly3,supply:exp \
    --no-plot
```

Models are selected with `--model` and may include `linear`, `exp`, `logistic`,
`power`, or polynomial forms like `poly3`. Use `--output-json` to save a JSON
report and `--log-scale` to plot using logarithmic axes.

If `--mode csv` is used, both `--demand-file` and `--supply-file` must be
provided. When not using `--no-plot` the supply and demand curves will be shown
interactively. Results are printed to the terminal and written to
`equilibrium_report.txt`.
