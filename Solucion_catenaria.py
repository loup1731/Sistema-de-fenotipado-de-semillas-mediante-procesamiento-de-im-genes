
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# =============================
# DEFINICIÓN IMPLICITA
# =============================

def ecuacion(a, L, H):
    return np.cosh(L/(2*a)) - (1 + H/a)

# =============================
# PARÁMETROS DE LA CATENARIA
# =============================

L = 0.204 # Diametro de la base en metros
H = 0.12 # altura máxima de la catenaria en x=0

a0 = 0.05 # Valor inicial aproximado

# =============================
# SOLUCION
# =============================

a_sol = fsolve(ecuacion, a0, args=(L, H))[0]

print("valor de a:", a_sol)

# =============================
# GRAFICA
# =============================

x = np.linspace(-L/2, L/2, 500)
y = -a_sol * np.cosh(x/a_sol) + (H + a_sol)

plt.plot(x, y, label = "Catenaria invertida")
plt.axhline(0,color = 'gray', linestyle='--')
plt.title("Perfil del domo: catenaria invertida")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.show()




