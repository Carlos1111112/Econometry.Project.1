import numpy as np
import matplotlib.pyplot as plt


# === Funciones auxiliares ===
def pedir_datos_manual(tipo):
    precios, cantidades = [], []
    print(f"\nPara la ecuación de {tipo}:")
    while True:
        try:
            precio = float(input("Ingresa un precio (o escribe -1 para terminar): $"))
            if precio == -1:
                break
            if precio < 0:
                print("El precio no puede ser negativo. Intenta de nuevo.")
                continue
            cantidad = float(input(f"¿Cuánta es la {tipo} cuando el precio es ${precio:.2f}? "))
            if cantidad < 0:
                print("La cantidad no puede ser negativa. Intenta de nuevo.")
                continue
            precios.append(precio)
            cantidades.append(cantidad)
        except ValueError:
            print("Entrada no válida. Intenta de nuevo.")
    if len(precios) < 2:
        raise ValueError(f"Se necesitan al menos dos puntos para {tipo}.")
    return np.array(precios), np.array(cantidades)


def calcular_ajuste(precios, cantidades, tipo):
    coef = np.polyfit(precios, cantidades, 1)
    if tipo == "demanda" and coef[0] > 0:
        raise ValueError("La pendiente de la demanda debe ser negativa.")
    if tipo == "oferta" and coef[0] < 0:
        raise ValueError("La pendiente de la oferta debe ser positiva.")
    return coef


def calcular_equilibrio(coef_d, coef_o):
    a_d, b_d = coef_d
    a_o, b_o = coef_o
    if a_d == a_o:
        raise ValueError("Las pendientes son iguales, no hay equilibrio único.")
    precio_eq = (b_o - b_d) / (a_d - a_o)
    cantidad_eq = a_d * precio_eq + b_d
    return precio_eq, cantidad_eq


def elasticidad_precio(pendiente, precio, cantidad):
    return pendiente * (precio / cantidad)


# === Flujo principal ===
try:
    precios_d, cantidades_d = pedir_datos_manual("demanda")
    precios_o, cantidades_o = pedir_datos_manual("oferta")

    coef_d = calcular_ajuste(precios_d, cantidades_d, "demanda")
    coef_o = calcular_ajuste(precios_o, cantidades_o, "oferta")

    # Cálculo del equilibrio
    precio_eq, cantidad_eq = calcular_equilibrio(coef_d, coef_o)

    # Elasticidad en el punto de equilibrio
    elasticidad_d = elasticidad_precio(coef_d[0], precio_eq, cantidad_eq)

    # Mostrar resultados sin redondeo prematuro
    print("\n=== Resultados ===")
    print(f"Ecuación de demanda: Qd = {coef_d[1]:.4f} + {coef_d[0]:.4f}P")
    print(f"Ecuación de oferta: Qs = {coef_o[1]:.4f} + {coef_o[0]:.4f}P")
    print(f"Precio de equilibrio: ${precio_eq:.4f}")
    print(f"Cantidad de equilibrio: {cantidad_eq:.4f} unidades")
    print(f"Elasticidad precio de la demanda en equilibrio: {elasticidad_d:.4f}")

    # Graficar
    p_min = min(min(precios_d), min(precios_o)) - 2
    p_max = max(max(precios_d), max(precios_o)) + 2
    p = np.linspace(p_min, p_max, 200)
    q_d = coef_d[1] + coef_d[0] * p
    q_o = coef_o[1] + coef_o[0] * p

    plt.figure(figsize=(8, 6))
    plt.plot(p, q_d, 'b-', label="Demanda", linewidth=2)
    plt.plot(p, q_o, 'orange', label="Oferta", linewidth=2)
    plt.plot(precio_eq, cantidad_eq, 'ro', label="Equilibrio", markersize=8)

    # Añadir puntos de datos
    plt.plot(precios_d, cantidades_d, 'bo', alpha=0.5)
    plt.plot(precios_o, cantidades_o, 'o', color='orange', alpha=0.5)

    plt.xlabel("Precio ($)", fontsize=12)
    plt.ylabel("Cantidad", fontsize=12)
    plt.title("Curvas de Oferta y Demanda", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plt.tight_layout()
    plt.savefig("curvas_oferta_demanda.png")
    plt.show()

    # Guardar resultados en txt
    with open("reporte_equilibrio.txt", "w") as f:
        f.write(f"Ecuación de demanda: Qd = {coef_d[1]:.4f} + {coef_d[0]:.4f}P\n")
        f.write(f"Ecuación de oferta: Qs = {coef_o[1]:.4f} + {coef_o[0]:.4f}P\n")
        f.write(f"Precio de equilibrio: ${precio_eq:.4f}\n")
        f.write(f"Cantidad de equilibrio: {cantidad_eq:.4f} unidades\n")
        f.write(f"Elasticidad precio de la demanda: {elasticidad_d:.4f}\n")

except ValueError as e:
    print(f"Error: {e}")
