import numpy as np
import matplotlib.pyplot as plt

# =========================
# PARÁMETROS DEL LED
# =========================

# Número de rayos por LED
n_rays = 200

# Ángulo del haz del LED en grados
beam_angle_deg = 115
beam_angle_rad = np.radians(beam_angle_deg)
half_beam = beam_angle_rad / 2

# Longitud de la tira de LED
led_length = 0.0035  # en metros

# Distancia del centro al punto de la base del LED (horizontal)
led_distance = 0.102 - 0.0035#- 0.005#- 0.0035#- 0.0025  # distancia desde el centro

# Ángulo de inclinación de la tira LED respecto a la horizontal
led_incl_deg = 0  #45 #0
led_incl_rad = np.radians(led_incl_deg)

# =========================
# CÁLCULO DE LA GEOMETRÍA DEL LED
# =========================

# LED derecho
xB_r = led_distance
yB = 0
xA_r = xB_r + led_length * np.cos(led_incl_rad)
yA_r = yB + led_length * np.sin(led_incl_rad)

# LED izquierdo (simétrico)
xB_l = -led_distance
xA_l = xB_l - led_length * np.cos(led_incl_rad)
yA_l = yB + led_length * np.sin(led_incl_rad)

# Centro de cada tira (punto de emisión)
xc_r, yc_r = (xA_r + xB_r) / 2, (yA_r + yB) / 2
xc_l, yc_l = (xA_l + xB_l) / 2, (yA_l + yB) / 2

# =========================
# FUNCIÓN PARA OBTENER DIRECCIONES DE RAYOS
# =========================

def ray_directions(center_angle, n_rays, half_beam):
    if n_rays == 1:
        return np.array([center_angle])
    else:
        return np.linspace(-half_beam, half_beam, n_rays) + center_angle

# Dirección perpendicular a la tira (normal)
vec_r = np.array([xB_r - xA_r, yB - yA_r])
vec_l = np.array([xB_l - xA_l, yB - yA_l])
normal_r = np.array([vec_r[1], -vec_r[0]])
normal_l = np.array([-vec_l[1], vec_l[0]])  # hacia el interior

# Ángulo de la normal (dirección central de emisión)
theta_r = np.arctan2(normal_r[1], normal_r[0])
theta_l = np.arctan2(normal_l[1], normal_l[0])

# Ángulos de los rayos
angles_r = ray_directions(theta_r, n_rays, half_beam)
angles_l = ray_directions(theta_l, n_rays, half_beam)

# =========================
# DOMO
# =========================

# Seleccionar tipo de domo
tipo_domo = 3     # 1: semicircular, 2: cónico, 3: catenaria
r =  0.102
H = 0.12
a_catenaria =   0.056486757337818924 #0.0362 #0.0671  # Usar solo si tipo_domo == 3


def generar_domo(tipo_domo=1, r=0.15, H=0.25, a=None, N=500):
    """
    Genera los puntos (x, y) del perfil del domo según el tipo seleccionado.

    Parámetros:
    - tipo_domo: int
        1: Semicircular
        2: Cónico
        3: Catenaria invertida
    - r: float
        Radio de la base del domo
    - H: float
        Altura del domo
    - a: float
        Parámetro de la catenaria (solo requerido si tipo_domo = 3)
    - N: int
        Número de puntos para definir el perfil

    Retorna:
    - x_domo: array
    - y_domo: array
    """

    x = np.linspace(-r, r, N)

    if tipo_domo == 1:
        # Semicírculo: x² + y² = r² -> y = sqrt(r² - x²)
        y = np.sqrt(r**2 - x**2)

    elif tipo_domo == 2:
        # Cono: línea recta desde (-r, 0) a (0, H) a (r, 0)
        x1 = np.linspace(-r, 0, N // 2)
        y1 = H * (x1 + r) / r
        x2 = np.linspace(0, r, N // 2)
        y2 = H * (r - x2) / r
        x = np.concatenate([x1, x2])
        y = np.concatenate([y1, y2])

    elif tipo_domo == 3:
        if a is None:
            raise ValueError("Para tipo_domo = 3 (catenaria) se debe proporcionar el parámetro 'a'")
        y = -a * np.cosh(x / a) + (H + a)

    else:
        raise ValueError("tipo_domo debe ser 1 (semicírculo), 2 (cono) o 3 (catenaria invertida)")

    return x, y

x_domo, y_domo = generar_domo(tipo_domo, r, H, a_catenaria)


# =========================
# RAY TRACING: choques y reflejos hasta la base
# =========================

EPS = 1e-8               # desplazamiento post-choque para evitar autointersección
MAX_BOUNCES = 1000       # límite por rayo (solo seguridad)
TMAX_SCAN = 5.0          # tope para escaneo de catenaria (múltiplos del tamaño del domo)
SCAN_STEPS = 64          # pasos iniciales para encontrar cambio de signo en catenaria
BIS_TOL = 1e-9           # tolerancia para bisección en catenaria
BIS_MAX_IT = 100

# ----- utilidades geométricas -----
def normalize(v):
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n

def reflect(d, n):
    # d y n unitarios
    return d - 2.0 * np.dot(d, n) * n

# ----- base y=0 dentro de [-R, R] -----
def intersect_ray_with_base(p0, d, R):
    x0, y0 = p0
    dx, dy = d
    if dy >= 0:
        return None  # nunca bajará a la base
    t = (0.0 - y0) / dy
    if t <= 0:
        return None
    x_hit = x0 + t * dx
    if -R - 1e-12 <= x_hit <= R + 1e-12:
        return (t, np.array([x_hit, 0.0]))
    return None

# ----- DOMO: perfiles y derivadas -----
def dome_y(x, shape, R, H=None, a=None):
    x = np.asarray(x)
    if shape == 1:  # semicircular
        return np.sqrt(np.maximum(0.0, R**2 - x**2))
    elif shape == 2:  # cónico
        m = H / R
        y = np.where(x <= 0, m * (x + R), -m * (x - R))
        return np.maximum(0.0, y)
    elif shape == 3:  # catenaria invertida
        if H is None:
            # cerrar en base cuando no se especifica, pero aquí asumimos ya dado en tu script
            H = a * (np.cosh(R / a) - 1.0)
        return H - (a * np.cosh(x / a) - a)
    else:
        raise ValueError("shape debe ser 1, 2 o 3")

def dome_dy_dx(x, shape, R, H=None, a=None):
    # derivada f'(x) = dy/dx (útil para normales)
    if shape == 1:
        # y = sqrt(R^2 - x^2)  => y' = -x / sqrt(R^2 - x^2) (cuidar x=±R)
        denom = np.sqrt(np.maximum(1e-30, R**2 - x**2))
        return -x / denom
    elif shape == 2:
        # cónico: pendiente constante por tramo
        m = H / R
        return m if x <= 0 else -m
    elif shape == 3:
        # y = H - (a cosh(x/a) - a) => y' = -sinh(x/a)
        return -np.sinh(x / a)
    else:
        raise ValueError("shape debe ser 1, 2 o 3")

# ----- normales internas -----
def surface_normal_internal(p, shape, R, H=None, a=None):
    x, y = p
    if shape == 1:
        # círculo centro (0,0): normal externa = p/||p||; interna = -externa
        n_ext = normalize(p)
        n_int = -n_ext
        return n_int
    elif shape == 2:
        mprime = dome_dy_dx(x, shape, R, H, a)  # +m o -m
        n0 = np.array([-mprime, 1.0])
        n = normalize(n0)
        # orientar hacia el interior: para x<0 interior apunta +x; para x>0 interior -x; pero mejor
        # garantizamos que apunte "hacia el centro" (0,0):
        to_center = normalize(-p)
        if np.dot(n, to_center) < 0:
            n = -n
        return n
    elif shape == 3:
        mprime = dome_dy_dx(x, shape, R, H, a)  # -sinh(x/a)
        n0 = np.array([-mprime, 1.0])  # (-f'(x), 1)
        n = normalize(n0)
        # orientar hacia el centro:
        to_center = normalize(-p)
        if np.dot(n, to_center) < 0:
            n = -n
        return n
    else:
        raise ValueError("shape debe ser 1, 2 o 3")

# ----- intersecciones con el domo -----
def intersect_semicircle(p0, d, R):
    # círculo x^2 + y^2 = R^2, tomar solución con y>=0 (semicírculo superior)
    x0, y0 = p0
    dx, dy = d
    A = dx*dx + dy*dy
    B = 2*(x0*dx + y0*dy)
    C = x0*x0 + y0*y0 - R*R
    disc = B*B - 4*A*C
    if disc < 0:
        return None
    sqrt_disc = np.sqrt(disc)
    t1 = (-B - sqrt_disc) / (2*A)
    t2 = (-B + sqrt_disc) / (2*A)
    ts = [t for t in (t1, t2) if t > 1e-12]
    if not ts:
        return None
    t_hit = min(ts)
    p_hit = p0 + t_hit * d
    if p_hit[1] < -1e-12:
        return None
    return (t_hit, p_hit)

def intersect_cone(p0, d, R, H):
    # dos segmentos de recta: y = m(x+R) para x∈[-R,0], y = -m(x-R) para x∈[0,R]
    x0, y0 = p0
    dx, dy = d
    m = H / R
    hits = []

    # LADO IZQUIERDO: y = m(x+R), x in [-R, 0]
    # Param rayo: x = x0 + t dx, y = y0 + t dy
    # Igualamos: y0 + t dy = m (x0 + t dx + R)
    # => t (dy - m dx) = m (x0 + R) - y0
    denom = (dy - m*dx)
    if abs(denom) > 1e-15:
        t = (m*(x0 + R) - y0) / denom
        if t > 1e-12:
            xh = x0 + t*dx
            if -R - 1e-12 <= xh <= 0 + 1e-12:
                yh = y0 + t*dy
                if yh >= -1e-12:
                    hits.append((t, np.array([xh, yh])))

    # LADO DERECHO: y = -m(x - R), x in [0, R]
    denom = (dy + m*dx)
    if abs(denom) > 1e-15:
        t = (-m*(x0 - R) - y0) / denom
        if t > 1e-12:
            xh = x0 + t*dx
            if 0 - 1e-12 <= xh <= R + 1e-12:
                yh = y0 + t*dy
                if yh >= -1e-12:
                    hits.append((t, np.array([xh, yh])))

    if not hits:
        return None
    # escoger el primer choque
    t_hit, p_hit = min(hits, key=lambda x: x[0])
    return (t_hit, p_hit)

def intersect_catenary(p0, d, R, H, a):
    # Resolver g(t) = y0 + t dy - f(x0 + t dx) = 0 para t>0
    # f(x) = H - (a cosh(x/a) - a)
    x0, y0 = p0
    dx, dy = d

    def g(t):
        x = x0 + t*dx
        return y0 + t*dy - (H - (a*np.cosh(x/a) - a))

    # escaneo para encontrar cambio de signo
    t_low = 1e-12
    g_low = g(t_low)
    t_high = max(R, H) * 1.5
    t_high = max(t_high, 0.2)
    # ampliar si hace falta
    for _ in range(SCAN_STEPS):
        g_high = g(t_high)
        if np.sign(g_low) == 0:
            return (t_low, p0 + t_low*d)
        if np.sign(g_high) == 0:
            return (t_high, p0 + t_high*d)
        if np.sign(g_low) != np.sign(g_high):
            break
        t_high *= 1.5
        if t_high > TMAX_SCAN:
            # no hay raíz adelante (probablemente irá a base)
            return None
    else:
        # no encontró cambio de signo
        return None

    # bisección
    a_t, b_t = t_low, t_high
    fa, fb = g(a_t), g(b_t)
    if np.sign(fa) == 0:
        return (a_t, p0 + a_t*d)
    if np.sign(fb) == 0:
        return (b_t, p0 + b_t*d)

    for _ in range(BIS_MAX_IT):
        c_t = 0.5*(a_t + b_t)
        fc = g(c_t)
        if abs(fc) < BIS_TOL or (b_t - a_t) < BIS_TOL:
            return (c_t, p0 + c_t*d)
        if np.sign(fa) * np.sign(fc) <= 0:
            b_t, fb = c_t, fc
        else:
            a_t, fa = c_t, fc

    # si llega aquí, retorno mejor c_t
    return (c_t, p0 + c_t*d)

def intersect_ray_with_dome_exact(p0, d, shape, R, H=None, a=None):
    if shape == 1:
        return intersect_semicircle(p0, d, R)
    elif shape == 2:
        return intersect_cone(p0, d, R, H)
    elif shape == 3:
        return intersect_catenary(p0, d, R, H, a)
    else:
        raise ValueError("shape debe ser 1, 2 o 3")

# ----- trazado hasta la base -----
def trace_until_base(p0, d, shape, R, H=None, a=None):
    """
    Retorna lista de segmentos [(p_ini, p_fin), ...] desde el origen del rayo
    rebotando en el domo hasta tocar la base y=0 dentro de [-R, R].
    """
    segments = []
    p = np.array(p0, dtype=float)
    d = normalize(np.array(d, dtype=float))

    for _ in range(MAX_BOUNCES):
        hit_base = intersect_ray_with_base(p, d, R)
        hit_dome = intersect_ray_with_dome_exact(p, d, shape, R, H, a)

        t_base = hit_base[0] if hit_base is not None else np.inf
        t_dome = hit_dome[0] if hit_dome is not None else np.inf

        if np.isinf(t_base) and np.isinf(t_dome):
            # No hay más eventos hacia adelante: terminamos
            break

        if t_base < t_dome:
            # Termina en la base
            p_next = hit_base[1]
            segments.append((p.copy(), p_next.copy()))
            return segments

        # Rebote en domo
        p_hit = hit_dome[1]
        segments.append((p.copy(), p_hit.copy()))
        n = surface_normal_internal(p_hit, shape, R, H, a)
        # orientar normal hacia interior respecto a dirección
        if np.dot(d, n) > 0:
            n = -n
        d = reflect(d, n)
        p = p_hit + EPS * d

    # Si llegamos aquí, excedimos MAX_BOUNCES (situación anómala)
    # Devolvemos lo acumulado, pero conviene revisar parámetros si pasa.
    return segments

# =========================
# ILUMINACIÓN EN LA BASE (histograma de impactos)
# =========================

def ray_final_base_hit(p0, angle, shape, R, H, a):
    """
    Traza un rayo hasta la base y retorna x_hit (float) si impacta en [-R, R],
    o None si no llega a la base.
    """
    d = np.array([np.cos(angle), np.sin(angle)], dtype=float)
    segs = trace_until_base(p0, d, shape, R, H, a)
    if not segs:
        return None
    # El último segmento termina en la base si todo va bien
    _, p_fin = segs[-1]
    if abs(p_fin[1]) <= 1e-8 and -R - 1e-9 <= p_fin[0] <= R + 1e-9:
        return float(p_fin[0])
    return None

def recolectar_impactos_base(origenes_y_angulos, shape, R, H, a):
    """
    origenes_y_angulos: lista de tuplas [(p0, [ang1, ang2, ...]), ...]
    Retorna: lista de x_hits (floats) de todos los rayos que tocaron la base.
    """
    hits = []
    for p0, angs in origenes_y_angulos:
        for ang in angs:
            xh = ray_final_base_hit(np.array(p0, dtype=float), ang, shape, R, H, a)
            if xh is not None:
                hits.append(xh)
    return hits

def histograma_base(x_hits, R, nbins=100, normalizar=True):
    """
    Construye histograma en [-R, R].
    Retorna: (counts, bin_edges, bin_centers)
    - counts: si normalizar=True, sum(counts)=1 (densidad aproximada).
    """
    counts, edges = np.histogram(x_hits, bins=nbins, range=(-R, R))
    if normalizar and counts.sum() > 0:
        counts = counts.astype(float) / counts.sum()
    centers = 0.5 * (edges[:-1] + edges[1:])
    return counts, edges, centers

# =========================
# VISUALIZACIÓN
# =========================

def generar_titulo():
    """
    Genera un título dinámico para la gráfica usando las variables globales del script.
    Incluye tipo de domo, número de rayos, inclinación de LEDs, ángulo de haz,
    distancia al centro, R, H y a (si aplica).
    """
    # Usamos las variables globales directamente
    global tipo_domo, n_rays, led_incl_deg, beam_angle_deg, led_distance, r, H, a_catenaria

    nombres_domo = {1: "Semicircular", 2: "Cónico", 3: "Catenaria invertida"}
    plural = "rayo" if n_rays == 1 else "rayos"

    # Primera línea
    t1 = f"Domo {nombres_domo.get(tipo_domo, 'Desconocido')} | {n_rays} {plural}/LED | Inclinación: {led_incl_deg:.1f}°"

    # Segunda línea
    t2 = f"Haz: {beam_angle_deg:.1f}° | Distancia LED: {led_distance:.3f} m | R={r:.3f} m"

    # Altura o parámetros adicionales según tipo de domo
    if tipo_domo in (1, 2) and H is not None:
        t2 += f" | H={H:.3f} m"
    if tipo_domo == 3 and a_catenaria is not None:
        t2 += f" | H={H:.3f} m | a={a_catenaria:.4f} m"

    return t1 + "\n" + t2


fig, ax = plt.subplots(figsize=(8, 6))

# Domo
x_domo, y_domo = generar_domo(tipo_domo, r, H, a_catenaria)
ax.plot(x_domo, y_domo, 'r', lw=2, label='Domo')

# Tiras
ax.plot([xA_r, xB_r], [yA_r, yB], 'b', lw=2, label='Tira derecha')
ax.plot([xA_l, xB_l], [yA_l, yB], 'b', lw=2, label='Tira izquierda')

# Función auxiliar para trazar todos los segmentos de un rayo
def draw_ray_path(p0, angle, shape, R, H, a, ax, max_len_plot=10.0):
    d = np.array([np.cos(angle), np.sin(angle)])
    segs = trace_until_base(p0, d, shape, R, H, a)
    for s0, s1 in segs:
        ax.plot([s0[0], s1[0]], [s0[1], s1[1]], lw=0.9, alpha=0.9, color='g')

# Rayos desde cada tira (centro de tira)
p0_r = np.array([xc_r, yc_r])
p0_l = np.array([xc_l, yc_l])

for ang in angles_r:
    draw_ray_path(p0_r, ang, tipo_domo, r, H, a_catenaria, ax)

for ang in angles_l:
    draw_ray_path(p0_l, ang, tipo_domo, r, H, a_catenaria, ax)

ax.set_aspect('equal', 'box')
ax.set_xlim(-1.1*r, 1.1*r)
ax.set_ylim(-0.01, max(0.1, y_domo.max()*1.05))
ax.grid(True)
ax.set_title(generar_titulo())
#ax.set_title(titulo)


ax.legend()
plt.tight_layout()
plt.show()

#Visualización del histograma: 
    
# 1) Recolectar impactos en la base de todos los rayos (tira derecha e izquierda)
p0_r = (xc_r, yc_r)
p0_l = (xc_l, yc_l)
origenes_y_angulos = [
    (p0_r, angles_r),
    (p0_l, angles_l),
]

x_hits = recolectar_impactos_base(origenes_y_angulos, tipo_domo, r, H, a_catenaria)

# 2) Histograma
nbins = 80  # ajusta resolución a gusto
counts, edges, centers = histograma_base(x_hits, r, nbins=nbins, normalizar=True)

# 3) Gráfica del histograma (debajo de tu figura principal)
plt.figure(figsize=(9, 3.2))
plt.bar(centers, counts, width=(2*r)/nbins, align='center', alpha=0.8, edgecolor='k')

plt.axhline(1/nbins, linestyle='--', linewidth=1, color='red', alpha=0.6) # linea roja. 

plt.xlim(-r, r)
plt.xlabel('x en la base (m)')
plt.ylabel('Fracción de rayos')

# Usar el mismo título del domo + indicación de histograma
plt.title(generar_titulo() + "\nDistribución de impactos en la base (normalizada)")

plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# =========================
# VISUALIZACIÓN V1
# =========================

# fig, ax = plt.subplots(figsize=(8, 4))
# ax.plot([xA_r, xB_r], [yA_r, yB], 'b', lw=2)
# ax.plot([xA_l, xB_l], [yA_l, yB], 'b', lw=2)
# ax.plot(0, 0, 'ko', label='Centro')

# # Dibujar rayos
# ray_length = 0.1 #Modifica la longitud de los rayos.  

# for angle in angles_r:
#     dx, dy = np.cos(angle), np.sin(angle)
#     ax.arrow(xc_r, yc_r, dx * ray_length, dy * ray_length, head_width=0.002, color='green', alpha=0.6)

# for angle in angles_l:
#     dx, dy = np.cos(angle), np.sin(angle)
#     ax.arrow(xc_l, yc_l, dx * ray_length, dy * ray_length, head_width=0.002, color='green', alpha=0.6)

# ax.set_aspect('equal')
# ax.set_xlim(-0.2, 0.2)
# ax.set_ylim(-0.01, 0.1)
# ax.grid(True)
# ax.set_title("Emisión de rayos desde LEDs inclinados")
# plt.tight_layout()
# plt.show()

# # Visualizar Domo
# plt.plot(x_domo, y_domo, 'r', label='Perfil del domo')
# plt.axis('equal')
# plt.grid(True)
# plt.title("Perfil del domo seleccionado")
# plt.legend()
# plt.show()

# =========================
# VISUALIZACIÓN V2
# =========================

# fig, ax = plt.subplots(figsize=(8, 6))

# # Domo
# #x_domo, y_domo = generar_domo(tipo_domo, r, H if tipo_domo!=3 else None, a_catenaria)
# x_domo, y_domo = generar_domo(tipo_domo, r, H, a_catenaria)
# ax.plot(x_domo, y_domo, 'r', lw=2, label='Domo')

# # Tiras
# ax.plot([xA_r, xB_r], [yA_r, yB], 'b', lw=2, label='Tira derecha')
# ax.plot([xA_l, xB_l], [yA_l, yB], 'b', lw=2, label='Tira izquierda')

# # Rayos
# ray_length = 0.1
# for angle in angles_r:
#     dx, dy = np.cos(angle), np.sin(angle)
#     ax.arrow(xc_r, yc_r, dx*ray_length, dy*ray_length, head_width=0.002, color='g', alpha=0.7)
# for angle in angles_l:
#     dx, dy = np.cos(angle), np.sin(angle)
#     ax.arrow(xc_l, yc_l, dx*ray_length, dy*ray_length, head_width=0.002, color='g', alpha=0.7)

# ax.set_aspect('equal', 'box')
# ax.set_xlim(-1.1*r, 1.1*r)
# ax.set_ylim(-0.01, max(0.1, y_domo.max()*1.05))
# ax.grid(True)
# ax.set_title("Domo + Tiras + Rayos")
# ax.legend()
# plt.tight_layout()
# plt.show()