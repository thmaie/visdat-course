import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import find_peaks


# ======================================================
# EIGENFREQUENZEN FINDEN
# ======================================================
def find_eigenfrequencies(
    frequency,
    tf,
    height_ratio=0.2,
    min_distance=5,
):
    """
    Findet Eigenfrequenzen aus einer Übertragungsfunktion

    Returns
    -------
    eigenfreqs : ndarray
        Gefundene Eigenfrequenzen
    peak_indices : ndarray
        Indizes der Peaks
    """
    peaks, props = find_peaks(
        np.abs(tf),
        height=np.max(np.abs(tf)) * height_ratio,
        distance=min_distance,
    )

    return frequency[peaks], peaks


# ======================================================
# MODE AUFBAUEN
# ======================================================
def build_mode(z_nodes, z_meas, u_meas):
    """
    Interpoliert Messwerte auf alle Knoten (Modeform)
    """
    z_ext = np.r_[0.0, z_meas]
    u_ext = np.r_[0.0, u_meas]

    order = np.argsort(z_ext)
    z_ext = z_ext[order]
    u_ext = u_ext[order]

    z_ext, idx = np.unique(z_ext, return_index=True)
    u_ext = u_ext[idx]

    if len(z_ext) < 2:
        # zu wenige Punkte → einfach konstante Auslenkung
        u_all = np.zeros_like(z_nodes)
        u_all[z_nodes == 0] = 0.0
        if len(u_ext) > 0:
            u_all += u_ext[0]
        return u_all

    interp = PchipInterpolator(z_ext, u_ext, extrapolate=True)
    u_all = interp(z_nodes)

    # Einspannung
    u_all[z_nodes == 0] = 0.0
    return u_all


# ======================================================
# MODE PLOTTEN (UNIVERSELL)
# ======================================================
def plot_mode(
    ax,
    node_ids,
    nodes,          # dict: {"x":..., "y":..., "z":...}
    edges,
    plot_lines,     # [(color, displacement_array)]
    axis_a="x",
    axis_b="z",
    disp_axis="x",
    scale=0.4,
):
    """
    Universeller Mode-Plotter
    """

    ax.clear()

    A = nodes[axis_a]
    B = nodes[axis_b]
    D = nodes[disp_axis]

    for e in edges:
        i1 = np.where(node_ids == e["n1"])[0][0]
        i2 = np.where(node_ids == e["n2"])[0][0]

        # Geometrie
        ax.plot(
            [A[i1], A[i2]],
            [B[i1], B[i2]],
            color="0.8",
            lw=0.8,
        )

        # Modeform(en)
        for color, u in plot_lines:
            ax.plot(
                [A[i1] + scale * u[i1],
                 A[i2] + scale * u[i2]],
                [B[i1], B[i2]],
                color,
                lw=1.4,
            )

    ax.set_aspect("equal")
    ax.set_xlabel(axis_a)
    ax.set_ylabel(axis_b)
    ax.grid(True)
