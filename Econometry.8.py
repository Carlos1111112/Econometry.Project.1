import numpy as np
import matplotlib.pyplot as plt


# === Auxiliary functions ===
def request_manual_data(data_type):
    prices, quantities = [], []
    print(f"\nFor the {data_type} equation:")
    while True:
        try:
            price = float(input("Enter a price (or type -1 to finish): $"))
            if price == -1:
                break
            if price < 0:
                print("Price cannot be negative. Please try again.")
                continue

            quantity = float(input("Enter the corresponding quantity: "))
            if quantity < 0:
                print("Quantity cannot be negative. Please try again.")
                continue

            prices.append(price)
            quantities.append(quantity)
        except ValueError:
            print("Invalid input. Please enter a numeric value.")
    return np.array(prices), np.array(quantities)


def load_data_from_csv(filename, data_type):
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    prices = data[:, 0]
    quantities = data[:, 1]
    print(f"\nLoaded {len(prices)} observations for {data_type} from '{filename}'.")
    return prices, quantities


def calculate_fit(prices, quantities, data_type):
    # Try linear fit
    coef_lin = np.polyfit(prices, quantities, 1)
    y_lin = np.polyval(coef_lin, prices)
    error_lin = np.mean((quantities - y_lin) ** 2)
    ss_tot = np.sum((quantities - np.mean(quantities)) ** 2)
    r2_lin = 1 - (np.sum((quantities - y_lin) ** 2) / ss_tot) if ss_tot > 0 else 0.0

    # If linear model is weak, try quadratic
    if error_lin > 1e-2 or r2_lin < 0.8:
        print(f"Warning: Linear fit for {data_type} has high error ({error_lin:.6f}) or low R² ({r2_lin:.6f}). Trying quadratic model.")
        coef_quad = np.polyfit(prices, quantities, 2)
        y_quad = np.polyval(coef_quad, prices)
        error_quad = np.mean((quantities - y_quad) ** 2)
        r2_quad = 1 - (np.sum((quantities - y_quad) ** 2) / ss_tot) if ss_tot > 0 else 0.0

        if error_quad < error_lin and r2_quad > r2_lin:
            print(f"Quadratic fit selected for {data_type} (error: {error_quad:.6f}, R²: {r2_quad:.6f}).")
            # Return [slope, intercept, quad_coeff] for consistency
            return np.array([coef_quad[1], coef_quad[0], coef_quad[2]])
        else:
            print(f"Keeping linear fit for {data_type} (error: {error_lin:.6f}, R²: {r2_lin:.6f}).")

    else:
        print(f"Linear fit sufficient for {data_type} (error: {error_lin:.6f}, R²: {r2_lin:.6f}).")

    # Return [slope, intercept]
    return np.array([coef_lin[0], coef_lin[1]])


def calculate_equilibrium(coef_d, coef_o):
    # coef = [slope, intercept] or [slope, intercept, quad_coeff]
    if len(coef_d) == 2 and len(coef_o) == 2:
        # Linear demand and supply
        m_d, b_d = coef_d
        m_o, b_o = coef_o
        price_eq = (b_o - b_d) / (m_d - m_o)
        quantity_eq = m_d * price_eq + b_d
    else:
        # For quadratic cases, solve m2*p^2 + m1*p + m0 = 0
        # Construct p-dependent polynomials for Qd(p) and Qo(p)
        # We'll solve Qd(p) = Qo(p)
        # Let coef_d = [m_d, b_d, a_d] representing a_d * p^2 + m_d * p + b_d
        a_d, m_d, b_d = coef_d[2], coef_d[0], coef_d[1]
        a_o, m_o, b_o = coef_o[2], coef_o[0], coef_o[1]
        # (a_d - a_o)p^2 + (m_d - m_o)p + (b_d - b_o) = 0
        A = a_d - a_o
        B = m_d - m_o
        C = b_d - b_o
        roots = np.roots([A, B, C])
        # Choose the real, positive root
        price_eq = next((r.real for r in roots if np.isreal(r) and r.real > 0), None)
        quantity_eq = np.polyval(coef_d[::-1], price_eq)
    return price_eq, quantity_eq


def price_elasticity(slope, price, quantity):
    return slope * (price / quantity)


# === Main flow ===
try:
    # Choose data entry mode: manual or CSV
    mode = input("Type 'manual' to enter data by hand or 'csv' to load from a file: ").strip().lower()
    if mode == 'csv':
        file_d = input("Enter CSV filename for demand (with .csv): ").strip()
        file_o = input("Enter CSV filename for supply (with .csv): ").strip()
        prices_d, quantities_d = load_data_from_csv(file_d, "demand")
        prices_o, quantities_o = load_data_from_csv(file_o, "supply")
    else:
        prices_d, quantities_d = request_manual_data("demand")
        prices_o, quantities_o = request_manual_data("supply")

    # Compute fits
    coef_d = calculate_fit(prices_d, quantities_d, "demand")
    coef_o = calculate_fit(prices_o, quantities_o, "supply")

    # Compute equilibrium
    price_eq, quantity_eq = calculate_equilibrium(coef_d, coef_o)

    # Compute elasticity at equilibrium
    elasticity_d = price_elasticity(coef_d[0], price_eq, quantity_eq)

    # Display results
    eq_d_str = (
        "Qd = "
        + " + ".join([f"{c:.8f}" for c in coef_d[::-1]]).replace("+ -", "- ")
        + "P"
        + (" + ..." if len(coef_d) > 2 else "")
    )
    eq_o_str = (
        "Qs = "
        + " + ".join([f"{c:.8f}" for c in coef_o[::-1]]).replace("+ -", "- ")
        + "P"
        + (" + ..." if len(coef_o) > 2 else "")
    )

    print("\n=== Results ===")
    print(f"Demand equation: {eq_d_str}")
    print(f"Supply equation: {eq_o_str}")
    print(f"Equilibrium price: ${price_eq:.8f}")
    print(f"Equilibrium quantity: {quantity_eq:.8f} units")
    print(f"Price elasticity of demand at equilibrium: {elasticity_d:.8f}")

    # Plot curves
    p_min = min(prices_d.min(), prices_o.min()) - 5
    p_max = max(prices_d.max(), prices_o.max()) + 5
    p = np.linspace(p_min, p_max, 400)

    def eval_curve(coefs, p_vals):
        if len(coefs) == 3:
            return np.polyval(coefs[::-1], p_vals)
        else:
            return coefs[0] * p_vals + coefs[1]

    q_d = eval_curve(coef_d, p)
    q_o = eval_curve(coef_o, p)

    plt.figure(figsize=(10, 7))
    plt.plot(p, q_d, label="Demand", linewidth=2)
    plt.plot(p, q_o, label="Supply", linewidth=2)
    plt.xlabel("Price")
    plt.ylabel("Quantity")
    plt.title("Supply and Demand Curves")
    plt.legend()
    plt.xlim(left=0)
    plt.tight_layout()
    plt.savefig("supply_demand_curves.png")
    plt.show()

    # Save report to text file
    with open("equilibrium_report.txt", "w") as f:
        f.write(f"Demand equation: {eq_d_str}\n")
        f.write(f"Supply equation: {eq_o_str}\n")
        f.write(f"Equilibrium price: ${price_eq:.8f}\n")
        f.write(f"Equilibrium quantity: {quantity_eq:.8f} units\n")
        f.write(f"Price elasticity of demand: {elasticity_d:.8f}\n")

except ValueError as e:
    print(f"Error: {e}")