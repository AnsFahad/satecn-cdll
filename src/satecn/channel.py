"""Uplink rate models: IoT-UAV air-to-ground (Eqs. 8-9) and IoT-LEO (Eq. 10)."""
from __future__ import annotations

import numpy as np

C_LIGHT = 3.0e8


def _db2lin(x):
    return 10.0 ** (np.asarray(x) / 10.0)


def fspl_db(d_m, f_hz):
    return 20.0 * np.log10(np.maximum(d_m, 1.0)) + 20.0 * np.log10(f_hz) + 20.0 * np.log10(4 * np.pi / C_LIGHT)


def uav_path_loss_lin(iot_xy: np.ndarray, uav_xy: np.ndarray, cfg: dict) -> np.ndarray:
    """Average LoS/NLoS path loss L_{i,u}(t) of Eq. (8), linear scale, shape (N, U)."""
    h = cfg["altitude_m"]
    r = np.linalg.norm(iot_xy[:, None, :] - uav_xy[None, :, :], axis=-1)
    d = np.sqrt(r ** 2 + h ** 2)
    theta = np.degrees(np.arctan2(h, r))
    a, b = cfg["env_a"], cfg["env_b"]
    p_los = 1.0 / (1.0 + a * np.exp(-b * (theta - a)))
    base = fspl_db(d, cfg["carrier_hz"])
    pl_los = _db2lin(base + cfg["eta_los_db"])
    pl_nlos = _db2lin(base + cfg["eta_nlos_db"])
    return p_los * pl_los + (1.0 - p_los) * pl_nlos


def uav_rate(path_loss_lin: np.ndarray, power_w: np.ndarray, cfg: dict) -> np.ndarray:
    """Eq. (9): R = B log2(1 + P / (L N0 B)). power_w broadcasts against path_loss_lin."""
    B = cfg["bandwidth_hz"]
    n0 = _db2lin(cfg["noise_dbm_hz"] + cfg["noise_figure_db"]) * 1e-3  # W/Hz
    return B * np.log2(1.0 + power_w / (path_loss_lin * n0 * B))


def leo_gain(range_km: np.ndarray, cfg: dict, rng: np.random.Generator, n_iot: int) -> np.ndarray:
    """g_i * H_{i,l}(t) / N0 per watt for each (IoT, satellite), shape (N, L).

    Free-space loss over the slant range, antenna gains, and Rician small-scale fading.
    """
    pl = fspl_db(range_km * 1e3, cfg["carrier_hz"])
    g_db = cfg["iot_gain_dbi"] + cfg["sat_gain_dbi"] - pl
    k = cfg["rician_k"]
    L = range_km.shape[0]
    los = np.sqrt(k / (k + 1))
    sc = np.sqrt(1 / (2 * (k + 1)))
    h = los + sc * (rng.standard_normal((n_iot, L)) + 1j * rng.standard_normal((n_iot, L)))
    fading = np.abs(h) ** 2
    n0 = _db2lin(cfg["noise_dbm_hz"] + cfg["noise_figure_db"]) * 1e-3 * cfg["bandwidth_hz"]
    return _db2lin(g_db)[None, :] * fading / n0


def leo_rate(gain_per_w: np.ndarray, power_w, cfg: dict) -> np.ndarray:
    """Eq. (10): R = B log2(1 + P g H / N0)."""
    return cfg["bandwidth_hz"] * np.log2(1.0 + np.asarray(power_w) * gain_per_w)
