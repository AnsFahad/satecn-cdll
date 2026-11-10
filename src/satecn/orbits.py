"""LEO geometry: Starlink TLE propagation (sgp4) with a synthetic-orbit fallback.

Provides per-second elevation and slant range of each screened satellite over the IoT area.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

R_EARTH_KM = 6378.137
MU_EARTH = 398600.4418  # km^3/s^2


def load_tle(path: str | Path) -> list[tuple[str, str, str]]:
    """Read a 3-line TLE file into (name, line1, line2) tuples."""
    lines = [l.rstrip() for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    out, i = [], 0
    while i + 2 < len(lines):
        if lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
            out.append((lines[i].strip(), lines[i + 1], lines[i + 2]))
            i += 3
        else:
            i += 1
    if not out:
        raise ValueError(f"no valid TLE triplets found in {path}")
    return out


def _jd(dt: datetime) -> tuple[float, float]:
    """Julian date split into (whole, fraction) as expected by sgp4."""
    from sgp4.api import jday
    return jday(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second + dt.microsecond * 1e-6)


def _gmst_rad(jd_full: np.ndarray) -> np.ndarray:
    d = jd_full - 2451545.0
    return np.deg2rad((280.46061837 + 360.98564736629 * d) % 360.0)


def _ground_ecef(lat_deg: float, lon_deg: float, alt_km: float = 0.0) -> np.ndarray:
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    r = R_EARTH_KM + alt_km
    return np.array([r * math.cos(lat) * math.cos(lon), r * math.cos(lat) * math.sin(lon), r * math.sin(lat)])


def _elev_range(sat_ecef: np.ndarray, lat_deg: float, lon_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Elevation (deg) and slant range (km) for positions shaped (..., 3)."""
    gs = _ground_ecef(lat_deg, lon_deg)
    up = gs / np.linalg.norm(gs)
    rel = sat_ecef - gs
    rng = np.linalg.norm(rel, axis=-1)
    sin_el = np.clip((rel @ up) / np.maximum(rng, 1e-9), -1.0, 1.0)
    return np.degrees(np.arcsin(sin_el)), rng


def propagate_tle(tles, start: datetime, n_seconds: int):
    """Propagate TLEs with sgp4; return ECEF positions (S, T, 3) km and altitudes (S,)."""
    from sgp4.api import Satrec, SatrecArray
    sats = SatrecArray([Satrec.twoline2rv(l1, l2) for _, l1, l2 in tles])
    jd0, fr0 = _jd(start)
    fr = fr0 + np.arange(n_seconds) / 86400.0
    jd = np.full(n_seconds, jd0)
    err, r_teme, _ = sats.sgp4(jd, fr)  # (S, T, 3)
    theta = _gmst_rad(jd + fr)  # rotate TEME -> ECEF (polar motion ignored)
    c, s = np.cos(theta), np.sin(theta)
    x = c * r_teme[..., 0] + s * r_teme[..., 1]
    y = -s * r_teme[..., 0] + c * r_teme[..., 1]
    pos = np.stack([x, y, r_teme[..., 2]], axis=-1)
    pos[err != 0] = np.nan
    alt = np.nanmean(np.linalg.norm(pos, axis=-1), axis=1) - R_EARTH_KM
    return pos, alt


def _longest_run(mask: np.ndarray) -> np.ndarray:
    """Longest run of True along the last axis."""
    best = np.zeros(mask.shape[:-1], dtype=int)
    cur = np.zeros_like(best)
    for t in range(mask.shape[-1]):
        cur = np.where(mask[..., t], cur + 1, 0)
        best = np.maximum(best, cur)
    return best


def screen_tles(tles, start: datetime, window_s: int, lat: float, lon: float,
                min_elev: float = 55.0, min_vis_s: int = 90, alt_range=(520.0, 580.0)):
    """Screening used in Sec. V-A: altitude band, min elevation, continuous visibility."""
    pos, alt = propagate_tle(tles, start, window_s)
    el, _ = _elev_range(pos, lat, lon)
    el = np.nan_to_num(el, nan=-90.0)
    ok_alt = (alt >= alt_range[0]) & (alt <= alt_range[1])
    run = _longest_run(el >= min_elev)
    keep = np.where(ok_alt & (run >= min_vis_s))[0]
    return keep, el[keep]


@dataclass
class LEOGeometry:
    """Per-second elevation/slant range of L satellites over a window."""
    elev_deg: np.ndarray   # (L, T)
    range_km: np.ndarray   # (L, T)
    min_elev: float
    source: str

    @property
    def n_sats(self) -> int:
        return self.elev_deg.shape[0]

    @property
    def window(self) -> int:
        return self.elev_deg.shape[1]

    def visible(self, t: int) -> np.ndarray:
        return self.elev_deg[:, t % self.window] >= self.min_elev

    def mean_visible(self) -> float:
        return float((self.elev_deg >= self.min_elev).sum(axis=0).mean())


def tle_geometry(tle_path: str | Path, start_iso: str, window_s: int, lat: float, lon: float,
                 min_elev: float) -> LEOGeometry:
    tles = load_tle(tle_path)
    start = datetime.fromisoformat(start_iso).replace(tzinfo=timezone.utc)
    pos, _ = propagate_tle(tles, start, window_s)
    el, rng = _elev_range(pos, lat, lon)
    el = np.nan_to_num(el, nan=-90.0)
    rng = np.nan_to_num(rng, nan=1e5)
    return LEOGeometry(el, rng, min_elev, f"TLE:{Path(tle_path).name}")


def synthetic_geometry(n_sats: int, window_s: int, min_elev: float, seed: int = 0,
                       alt_range=(520.0, 580.0)) -> LEOGeometry:
    """Fallback when sgp4/TLE is unavailable: straight-overhead-ish circular passes.

    Each satellite follows a great-circle pass whose closest approach and timing are random,
    reproducing short (~2-4 min) high-elevation passes similar to the TLE screening.
    """
    rng_ = np.random.default_rng(seed)
    t = np.arange(window_s)
    el = np.empty((n_sats, window_s))
    rg = np.empty((n_sats, window_s))
    for s in range(n_sats):
        h = rng_.uniform(*alt_range)
        r = R_EARTH_KM + h
        w = math.sqrt(MU_EARTH / r ** 3)  # rad/s
        t0 = rng_.uniform(0, window_s)
        cross = rng_.uniform(0.0, 0.035)  # cross-track offset angle (rad)
        along = w * (t - t0)
        # central angle between sub-satellite point and ground point
        cang = np.arccos(np.clip(np.cos(along) * np.cos(cross), -1, 1))
        d = np.sqrt(R_EARTH_KM ** 2 + r ** 2 - 2 * R_EARTH_KM * r * np.cos(cang))
        sin_el = (r * np.cos(cang) - R_EARTH_KM) / d
        el[s] = np.degrees(np.arcsin(np.clip(sin_el, -1, 1)))
        rg[s] = d
    return LEOGeometry(el, rg, min_elev, "synthetic")


def build_geometry(cfg: dict, root: Path) -> LEOGeometry:
    """Use TLE + sgp4 when available, otherwise the synthetic fallback."""
    leo = cfg["leo"]
    tle_path = root / leo["tle_file"]
    if leo.get("use_tle", True) and tle_path.exists():
        try:
            return tle_geometry(tle_path, leo["window_start_utc"], leo["window_s"],
                                cfg["area"]["lat"], cfg["area"]["lon"], leo["min_elev_deg"])
        except ImportError:
            print("[orbits] sgp4 not installed; using synthetic LEO geometry")
    return synthetic_geometry(leo["n_sats"], leo["window_s"], leo["min_elev_deg"], seed=leo.get("seed", 0))
