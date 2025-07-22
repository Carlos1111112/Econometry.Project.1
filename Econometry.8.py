"""Supply and demand equilibrium analysis with flexible model fitting."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, fsolve


# ===========================
# Data handling
# ===========================

def load_manual_data(kind: str) -> Tuple[np.ndarray, np.ndarray]:
    """Prompt the user to manually enter ``(price, quantity)`` pairs."""
    prices: list[float] = []
    quantities: list[float] = []
    print(f"\nEntering data for {kind} equation. Use -1 to finish.")
    while True:
        try:
            p = float(input("Price: $"))
            if p == -1:
                break
            if p < 0:
                print("Price cannot be negative.")
                continue
            q = float(input("Quantity: "))
            if q < 0:
                print("Quantity cannot be negative.")
                continue
            prices.append(p)
            quantities.append(q)
        except ValueError:
            print("Invalid number. Try again.")
    return np.array(prices), np.array(quantities)


def load_csv_data(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load ``(price, quantity)`` data from a CSV file with two columns."""
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        rows = [row for row in reader if row]
    if not rows:
        raise ValueError(f"No data found in {path}")
    prices, quantities = zip(*[(float(r[0]), float(r[1])) for r in rows])
    return np.array(prices), np.array(quantities)


# ===========================
# Model definitions
# ===========================

def linear_func(P: np.ndarray, a: float, b: float) -> np.ndarray:
    """Simple linear function ``Q = a*P + b``."""
    return a * P + b


def exp_func(P: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Exponential function ``Q = a * exp(b*P) + c``."""
    return a * np.exp(b * P) + c


def logistic_func(P: np.ndarray, L: float, k: float, x0: float) -> np.ndarray:
    """Logistic curve with carrying capacity ``L``."""
    return L / (1 + np.exp(-k * (P - x0)))


def power_func(P: np.ndarray, a: float, b: float) -> np.ndarray:
    """Power law ``Q = a * P**b``."""
    return a * P ** b


def create_poly_func(degree: int) -> Callable[[np.ndarray, float], np.ndarray]:
    """Return a polynomial model of a given degree."""

    def func(P: np.ndarray, *coeffs: float) -> np.ndarray:
        return np.polyval(coeffs, P)

    func.__name__ = f"poly{degree}"
    return func


MODEL_REGISTRY: Dict[str, Callable] = {
    "linear": linear_func,
    "exp": exp_func,
    "logistic": logistic_func,
    "power": power_func,
}


# ===========================
# Fitting utilities
# ===========================

@dataclass
class FitResult:
    func: Callable[[np.ndarray], np.ndarray]
    params: Tuple[float, ...]
    r2: float


def get_model(name: str, degree: int | None = None) -> Callable:
    """Return a model function based on ``name``.

    Polynomial models can specify the degree either in the name (e.g. ``poly3``)
    or via the ``degree`` argument.
    """

    if name.startswith("poly"):
        deg = int(name[4:]) if len(name) > 4 else degree or 2
        return create_poly_func(deg)
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'")
    return MODEL_REGISTRY[name]


def fit_model(
    x: np.ndarray, y: np.ndarray, model_name: str, degree: int | None = None
) -> FitResult:
    """Fit a named model to data and return :class:`FitResult`."""

    model = get_model(model_name, degree)
    guess = np.ones(model.__code__.co_argcount - 1)
    popt, _ = curve_fit(model, x, y, p0=guess, maxfev=10000)

    def fitted_func(P: np.ndarray) -> np.ndarray:
        return model(P, *popt)

    residuals = y - fitted_func(x)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return FitResult(fitted_func, tuple(popt), float(r2))


def numerical_elasticity(func: Callable[[float], float], price: float) -> float:
    """Compute price elasticity using a central difference derivative."""

    eps = 1e-6
    q = func(price)
    dqdp = (func(price + eps) - func(price - eps)) / (2 * eps)
    if q == 0:
        raise ValueError("Quantity is zero; elasticity undefined")
    return dqdp * price / q


def find_equilibrium(
    demand: Callable[[float], float],
    supply: Callable[[float], float],
    guess: float,
) -> Tuple[float, float]:
    """Solve for the price where demand equals supply."""

    root = fsolve(lambda p: demand(p) - supply(p), x0=guess)
    price_eq = float(root[0])
    quantity_eq = float(demand(price_eq))
    return price_eq, quantity_eq


# ===========================
# Plotting and reporting
# ===========================

def plot_curves(
    demand: Callable[[np.ndarray], np.ndarray],
    supply: Callable[[np.ndarray], np.ndarray],
    price_eq: float,
    quantity_eq: float,
    x_label: str,
    y_label: str,
    log_scale: bool,
    filename: str | None = None,
) -> None:
    """Plot demand and supply curves and the equilibrium point."""

    prices = np.linspace(0, price_eq * 1.5 if price_eq > 0 else 10, 200)
    plt.figure()
    plt.plot(prices, demand(prices), label="Demand")
    plt.plot(prices, supply(prices), label="Supply")
    plt.scatter([price_eq], [quantity_eq], color="red", label="Equilibrium")
    if log_scale:
        plt.yscale("log")
        plt.xscale("log")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    else:
        plt.show()


def write_report(report: dict, to_json: bool) -> None:
    """Write report dictionary to ``txt`` or ``json`` file."""

    if to_json:
        with open("equilibrium_report.json", "w") as f:
            json.dump(report, f, indent=2)
    else:
        with open("equilibrium_report.txt", "w") as f:
            for k, v in report.items():
                f.write(f"{k}: {v}\n")


# ===========================
# CLI
# ===========================

def parse_model_option(option: str) -> Dict[str, str]:
    """Parse ``--model`` option into a mapping of curve names."""

    mapping: Dict[str, str] = {"demand": "linear", "supply": "linear"}
    pairs = [p for p in option.split(",") if p]
    for pair in pairs:
        if ":" not in pair:
            continue
        side, model = pair.split(":", 1)
        if side in ("demand", "supply"):
            mapping[side] = model
    return mapping


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Supply and demand analysis")
    parser.add_argument("--mode", choices=["manual", "csv"], default="manual")
    parser.add_argument("--demand-file")
    parser.add_argument("--supply-file")
    parser.add_argument("--model", default="demand:linear,supply:linear")
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--output-json", action="store_true")
    parser.add_argument("--log-scale", action="store_true")
    parser.add_argument("--xlabel", default="Price")
    parser.add_argument("--ylabel", default="Quantity")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.mode == "csv":
        if not args.demand_file or not args.supply_file:
            parser.error("CSV mode requires --demand-file and --supply-file")
        x_d, y_d = load_csv_data(args.demand_file)
        x_s, y_s = load_csv_data(args.supply_file)
    else:
        x_d, y_d = load_manual_data("demand")
        x_s, y_s = load_manual_data("supply")

    models = parse_model_option(args.model)
    demand_fit = fit_model(x_d, y_d, models["demand"], args.degree)
    supply_fit = fit_model(x_s, y_s, models["supply"], args.degree)

    guess = np.mean(np.concatenate([x_d, x_s]))
    price_eq, quantity_eq = find_equilibrium(demand_fit.func, supply_fit.func, guess)

    elasticity_d = numerical_elasticity(demand_fit.func, price_eq)
    elasticity_s = numerical_elasticity(supply_fit.func, price_eq)

    report = {
        "demand_model": models["demand"],
        "supply_model": models["supply"],
        "demand_params": demand_fit.params,
        "supply_params": supply_fit.params,
        "r2_demand": demand_fit.r2,
        "r2_supply": supply_fit.r2,
        "equilibrium_price": price_eq,
        "equilibrium_quantity": quantity_eq,
        "elasticity_demand": elasticity_d,
        "elasticity_supply": elasticity_s,
    }

    write_report(report, args.output_json)

    if not args.no_plot:
        plot_curves(
            demand_fit.func,
            supply_fit.func,
            price_eq,
            quantity_eq,
            args.xlabel,
            args.ylabel,
            args.log_scale,
            None,
        )


if __name__ == "__main__":
    main()
