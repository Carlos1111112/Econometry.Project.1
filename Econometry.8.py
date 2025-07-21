"""Tools for computing supply and demand equilibrium."""

import argparse
import numpy as np
import matplotlib.pyplot as plt

# === Helper functions ===
def get_manual_data(kind: str):
    """Prompt the user to manually enter price--quantity pairs."""

    prices, quantities = [], []
    print(f"\nEntering data for the {kind} equation:")

    while True:
        try:
            price = float(input("Enter a price (or -1 to finish): $"))
            if price == -1:
                break
            if price < 0:
                print("Price cannot be negative. Try again.")
                continue

            quantity = float(input("Enter the corresponding quantity: "))
            if quantity < 0:
                print("Quantity cannot be negative. Try again.")
                continue

            prices.append(price)
            quantities.append(quantity)

        except ValueError:
            print("Invalid input. Please enter a number.")

    return np.array(prices), np.array(quantities)


def get_csv_data(filepath: str):
    """Load price--quantity data from a CSV file."""

    data = np.loadtxt(filepath, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1]


def fit_line(x: np.ndarray, y: np.ndarray):
    """Return slope and intercept of the best fitting line."""

    return np.polyfit(x, y, 1)


def calculate_equilibrium(coef_d: np.ndarray, coef_s: np.ndarray):
    """Return price and quantity at which demand equals supply."""

    m_d, b_d = coef_d
    m_s, b_s = coef_s

    if np.isclose(m_d, m_s):
        raise ValueError("Supply and demand curves are parallel; no equilibrium.")

    price_eq = (b_s - b_d) / (m_d - m_s)
    quantity_eq = m_d * price_eq + b_d
    return price_eq, quantity_eq


def compute_elasticity(coef: np.ndarray, price: float, quantity: float):
    """Return price elasticity for a linear model Q = mP + b."""

    m = coef[0]
    if quantity == 0:
        raise ValueError("Quantity cannot be zero when computing elasticity.")
    return m * price / quantity


# === Main script ===
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute the market equilibrium from supply and demand data",
    )
    parser.add_argument(
        "--mode",
        choices=["manual", "csv"],
        default="manual",
        help="input mode",
    )
    parser.add_argument("--demand-file", help="CSV file with demand observations")
    parser.add_argument("--supply-file", help="CSV file with supply observations")
    parser.add_argument(
        "--no-plot", action="store_true", help="disable displaying the plot"
    )

    args = parser.parse_args()

    try:
        if args.mode == "manual":
            x_d, y_d = get_manual_data("demand")
            x_s, y_s = get_manual_data("supply")
        else:
            if not args.demand_file or not args.supply_file:
                parser.error(
                    "CSV mode requires both --demand-file and --supply-file."
                )
            x_d, y_d = get_csv_data(args.demand_file)
            x_s, y_s = get_csv_data(args.supply_file)

        # Fit linear models
        coef_d = fit_line(x_d, y_d)
        coef_s = fit_line(x_s, y_s)

        # Compute equilibrium and elasticities
        price_eq, quantity_eq = calculate_equilibrium(coef_d, coef_s)
        elasticity_d = compute_elasticity(coef_d, price_eq, quantity_eq)
        elasticity_s = compute_elasticity(coef_s, price_eq, quantity_eq)

        print(f"Demand equation: Qd = {coef_d[1]:.4f} + {coef_d[0]:.4f}P")
        print(f"Supply equation: Qs = {coef_s[1]:.4f} + {coef_s[0]:.4f}P")
        print(f"Equilibrium price: ${price_eq:.4f}")
        print(f"Equilibrium quantity: {quantity_eq:.4f}")
        print(
            f"Price elasticity of demand at equilibrium: {elasticity_d:.4f}"
        )
        print(
            f"Price elasticity of supply at equilibrium: {elasticity_s:.4f}"
        )

        with open("equilibrium_report.txt", "w") as f:
            f.write(f"Demand equation: Qd = {coef_d[1]:.4f} + {coef_d[0]:.4f}P\n")
            f.write(f"Supply equation: Qs = {coef_s[1]:.4f} + {coef_s[0]:.4f}P\n")
            f.write(f"Equilibrium price: ${price_eq:.4f}\n")
            f.write(f"Equilibrium quantity: {quantity_eq:.4f}\n")
            f.write(f"Price elasticity of demand: {elasticity_d:.4f}\n")
            f.write(f"Price elasticity of supply: {elasticity_s:.4f}\n")

        if not args.no_plot:
            prices = np.linspace(0, max(x_d.max(), x_s.max()) * 1.1, 100)
            demand_curve = coef_d[0] * prices + coef_d[1]
            supply_curve = coef_s[0] * prices + coef_s[1]

            plt.plot(prices, demand_curve, label="Demand")
            plt.plot(prices, supply_curve, label="Supply")
            plt.scatter([price_eq], [quantity_eq], color="red", label="Equilibrium")
            plt.title("Supply and Demand Curves")
            plt.xlabel("Price")
            plt.ylabel("Quantity")
            plt.legend()
            plt.xlim(left=0)
            plt.tight_layout()
            plt.savefig("supply_demand_curves.png")
            plt.show()

    except ValueError as exc:
        print(f"Error: {exc}")


if __name__ == "__main__":
    main()
