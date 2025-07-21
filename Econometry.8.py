import numpy as np
import matplotlib.pyplot as plt

# === Helper functions ===
def get_manual_data(kind: str):
    """Prompt the user to enter price–quantity pairs manually."""
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
    """Load price–quantity data from a CSV file (two columns, header row)."""
    data = np.loadtxt(filepath, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1]


def fit_line(x: np.ndarray, y: np.ndarray):
    """
    Fit a linear model y = m x + b to the data.
    Returns an array [m, b].
    """
    return np.polyfit(x, y, 1)


def calculate_equilibrium(coef_d: np.ndarray, coef_s: np.ndarray):
    """
    Given demand coef_d = [m_d, b_d] and supply coef_s = [m_s, b_s],
    solve for equilibrium price and quantity:
      m_d * P + b_d = m_s * P + b_s
    """
    m_d, b_d = coef_d
    m_s, b_s = coef_s

    price_eq = (b_s - b_d) / (m_d - m_s)
    quantity_eq = m_d * price_eq + b_d
    return price_eq, quantity_eq


def compute_elasticity(coef: np.ndarray, price: float, quantity: float):
    """
    Compute price elasticity of demand (or supply):
      elasticity = (dQ/dP) * (P / Q)
    For a linear model Q = m P + b, dQ/dP = m.
    """
    m = coef[0]
    return m * price / quantity


# === Main script ===
if __name__ == "__main__":
    try:
        mode = input("Select input mode (manual or csv): ").strip().lower()
        if mode not in ("manual", "csv"):
            raise ValueError("Mode must be 'manual' or 'csv'.")

        if mode == "manual":
            x_d, y_d = get_manual_data("demand")
            x_s, y_s = get_manual_data("supply")
        else:
            demand_file = input("Path to demand CSV file: ").strip()
            supply_file = input("Path to supply CSV file: ").strip()
            x_d, y_d = get_csv_data(demand_file)
            x_s, y_s = get_csv_data(supply_file)

        # Fit linear models
        coef_d = fit_line(x_d, y_d)  # [slope_d, intercept_d]
        coef_s = fit_line(x_s, y_s)  # [slope_s, intercept_s]

        # Compute equilibrium
        price_eq, quantity_eq = calculate_equilibrium(coef_d, coef_s)
        elasticity_d = compute_elasticity(coef_d, price_eq, quantity_eq)

        # Plot supply and demand curves
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

        # Save results to a text report
        with open("equilibrium_report.txt", "w") as f:
            f.write(f"Demand equation: Qd = {coef_d[1]:.4f} + {coef_d[0]:.4f}P\n")
            f.write(f"Supply equation: Qs = {coef_s[1]:.4f} + {coef_s[0]:.4f}P\n")
            f.write(f"Equilibrium price: ${price_eq:.4f}\n")
            f.write(f"Equilibrium quantity: {quantity_eq:.4f}\n")
            f.write(f"Price elasticity of demand: {elasticity_d:.4f}\n")

    except ValueError as e:
        print(f"Error: {e}")
