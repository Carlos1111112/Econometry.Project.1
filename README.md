# Supply and Demand Equilibrium Script

This script fits linear models to supply and demand observations, computes the
market equilibrium and price elasticities, and plots the resulting curves.
Data can be entered manually or loaded from CSV files using command line
options.

## Usage

```
python Econometry.8.py [--mode manual|csv] [--demand-file FILE] [--supply-file FILE] [--no-plot]
```

If `--mode csv` is used, both `--demand-file` and `--supply-file` must be
provided. When not using `--no-plot` the supply and demand curves will be shown
interactively. Results are printed to the terminal and written to
`equilibrium_report.txt`.
