"""
WMM2025 magnetic declination and inclination from spherical harmonic model.

Public API:
    get_mag_field(lat_deg, lon_deg, alt_m=0.0, year=None)
        -> (declination_deg, inclination_deg)

Coefficients are lazy-loaded from WMM2025.COF on first call.

Algorithm follows NOAA WMM-2020 Technical Note (Chulliat et al., 2020):
  Schmidt quasi-normal associated Legendre recursion (3 cases: diagonal,
  sub-diagonal, general), then NED field summation.
"""

import math
from datetime import datetime
from pathlib import Path

_COEFFS = None  # lazy cache: list of (n, m, g, h, gdot, hdot)

_COF_PATH = (
    Path(__file__).resolve().parents[2]
    / "WMM2025COF" / "WMM2025COF" / "WMM2025.COF"
)

_WMM_EPOCH  = 2025.0
_REF_RADIUS = 6371200.0   # WMM reference sphere radius (m)


def _load_coeffs():
    global _COEFFS
    if _COEFFS is not None:
        return _COEFFS
    coeffs = []
    with open(_COF_PATH, "r") as fh:
        for line in fh:
            parts = line.split()
            if not parts or parts[0].startswith("9999"):
                break
            try:
                n, m = int(parts[0]), int(parts[1])
            except (ValueError, IndexError):
                continue
            if n < 1:
                continue
            g, h, gdot, hdot = (float(parts[2]), float(parts[3]),
                                 float(parts[4]), float(parts[5]))
            coeffs.append((n, m, g, h, gdot, hdot))
    _COEFFS = coeffs
    return _COEFFS


def _decimal_year(dt=None):
    if dt is None:
        dt = datetime.utcnow()
    y = dt.year
    start = datetime(y, 1, 1)
    end   = datetime(y + 1, 1, 1)
    return y + (dt - start).total_seconds() / (end - start).total_seconds()


def _schmidt_legendre(n_max, cos_theta, sin_theta):
    """
    Schmidt quasi-normal associated Legendre functions P[n][m] and dP[n][m]/dtheta.

    Recursion (3 cases from NOAA WMM Tech Note, Appendix C):
      diagonal:     P[n][n] = sqrt((2n-1)/(2n)) * sin_theta * P[n-1][n-1]
      sub-diagonal: P[n][n-1] = sqrt(2n-1) * cos_theta * P[n-1][n-1]  (K2=0 of general case)
      general m<=n-2: P[n][m] = K1*cos*P[n-1][m] - K2*P[n-2][m]
        K1 = (2n-1)/sqrt((n-m)*(n+m))
        K2 = sqrt((n-m-1)*(n+m-1)/((n-m)*(n+m)))

    The sub-diagonal formula is actually the general formula evaluated at m=n-1
    (K2 vanishes because n-m-1=0).  It is split out only for clarity.
    """
    P  = [[0.0] * (n_max + 1) for _ in range(n_max + 1)]
    dP = [[0.0] * (n_max + 1) for _ in range(n_max + 1)]

    P[0][0]  = 1.0
    dP[0][0] = 0.0

    if n_max < 1:
        return P, dP

    P[1][0]  =  cos_theta
    dP[1][0] = -sin_theta
    P[1][1]  =  sin_theta
    dP[1][1] =  cos_theta

    for n in range(2, n_max + 1):
        # ── diagonal m == n ──────────────────────────────────────────
        sd = math.sqrt((2.0 * n - 1.0) / (2.0 * n))
        P[n][n]  = sin_theta * sd * P[n - 1][n - 1]
        dP[n][n] = (cos_theta * sd * P[n - 1][n - 1]
                    + sin_theta * sd * dP[n - 1][n - 1])

        # ── m = 0 .. n-1 (sub-diagonal + general, unified formula) ──
        for m in range(n):
            nm  = (n - m) * (n + m)   # = n²-m²
            k1  = (2.0 * n - 1.0) / math.sqrt(nm)
            if n - 2 >= m:
                k2 = math.sqrt((n - m - 1.0) * (n + m - 1.0) / nm)
                P[n][m]  = (k1 * cos_theta * P[n - 1][m]
                            - k2 * P[n - 2][m])
                dP[n][m] = (k1 * (-sin_theta * P[n - 1][m]
                                  + cos_theta * dP[n - 1][m])
                            - k2 * dP[n - 2][m])
            else:
                # m == n-1: k2 = 0 (n-m-1 = 0)
                P[n][m]  = k1 * cos_theta * P[n - 1][m]
                dP[n][m] = k1 * (-sin_theta * P[n - 1][m]
                                  + cos_theta * dP[n - 1][m])

    return P, dP


def get_mag_field(lat_deg: float, lon_deg: float,
                  alt_m: float = 0.0,
                  year: float | None = None) -> tuple[float, float]:
    """
    Compute WMM2025 magnetic declination and inclination.

    Args:
        lat_deg: Geodetic latitude  (degrees, -90 to +90)
        lon_deg: Geodetic longitude (degrees, -180 to +360)
        alt_m:   Height above WGS-84 ellipsoid (metres)
        year:    Decimal year (e.g. 2026.3).  Defaults to current UTC date.

    Returns:
        (declination_deg, inclination_deg)
        Positive declination = magnetic north is east of true north.
        Positive inclination = field points into Earth (normal for N hemisphere).
    """
    if year is None:
        year = _decimal_year()

    coeffs = _load_coeffs()
    dt     = year - _WMM_EPOCH

    lat_rad = math.radians(lat_deg)
    lon_rad = math.radians(lon_deg)
    theta   = math.pi / 2.0 - lat_rad   # geocentric colatitude (approx geodetic)

    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    if abs(sin_theta) < 1e-10:   # guard against geographic poles
        sin_theta = 1e-10

    r    = _REF_RADIUS + alt_m
    a    = _REF_RADIUS
    n_max = 12

    P, dP = _schmidt_legendre(n_max, cos_theta, sin_theta)

    # Accumulate NED field components directly:
    #   X (North) = -B_theta = +Σ (a/r)^{n+2} * gh_cos * dP
    #   Y (East)  =  B_phi   = -(1/sinθ) * Σ m * (a/r)^{n+2} * gh_sin * P
    #   Z (Down)  = -B_r     = -Σ (n+1) * (a/r)^{n+2} * gh_cos * P
    X = 0.0
    Y = 0.0
    Z = 0.0

    for (n, m, g0, h0, gdot, hdot) in coeffs:
        g = g0 + gdot * dt
        h = h0 + hdot * dt

        ratio   = (a / r) ** (n + 2)
        cos_ml  = math.cos(m * lon_rad)
        sin_ml  = math.sin(m * lon_rad)
        gh_cos  = g * cos_ml + h * sin_ml          # g cos(mλ) + h sin(mλ)
        gh_sin  = -g * sin_ml + h * cos_ml         # -g sin(mλ) + h cos(mλ)  (for B_phi)

        X += ratio * gh_cos * dP[n][m]
        Y -= m * ratio * gh_sin * P[n][m] / sin_theta
        Z -= (n + 1) * ratio * gh_cos * P[n][m]

    declination_deg = math.degrees(math.atan2(Y, X))
    inclination_deg = math.degrees(math.atan2(Z, math.sqrt(X * X + Y * Y)))

    return declination_deg, inclination_deg
