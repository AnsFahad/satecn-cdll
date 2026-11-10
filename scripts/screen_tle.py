"""Screen a full Starlink TLE file down to the satellites used in the simulation (Sec. V-A).

Usage:
  python scripts/screen_tle.py --tle starlink_full.txt --start 2026-11-08T00:00:00 --out data/starlink_screened.txt
Download the full file from https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle
"""
import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satecn.orbits import load_tle, screen_tles  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tle", required=True)
    ap.add_argument("--start", default="2026-11-08T00:00:00")
    ap.add_argument("--window", type=int, default=600)
    ap.add_argument("--lat", type=float, default=42.3732)
    ap.add_argument("--lon", type=float, default=-72.5199)
    ap.add_argument("--target", type=int, default=18, help="desired number of screened satellites")
    ap.add_argument("--scan-minutes", type=int, default=120, help="search window starts over this span")
    ap.add_argument("--alt-min", type=float, default=440.0)
    ap.add_argument("--alt-max", type=float, default=600.0)
    ap.add_argument("--out", default="data/starlink_screened.txt")
    a = ap.parse_args()

    tles = load_tle(a.tle)
    t0 = datetime.fromisoformat(a.start).replace(tzinfo=timezone.utc)
    best = None
    for m in range(0, a.scan_minutes, 10):
        start = t0 + timedelta(minutes=m)
        keep, el = screen_tles(tles, start, a.window, a.lat, a.lon, alt_range=(a.alt_min, a.alt_max))
        vis = (el >= 55).sum(axis=0).mean() if len(keep) else 0.0
        print(f"start={start.isoformat()} screened={len(keep)} mean_visible={vis:.2f}")
        score = abs(len(keep) - a.target)
        if best is None or score < best[0]:
            best = (score, start, keep, vis)
        if score == 0:
            break
    _, start, keep, vis = best
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for k in keep:
            f.write("\n".join(tles[k]) + "\n")
    print(f"wrote {len(keep)} satellites to {out} (window start {start.isoformat()}, mean visible {vis:.2f})")


if __name__ == "__main__":
    main()
