"""
Toy detector model: photon transport to a forward EM calorimeter.

- Geometry: plane at z = z_cal, circular active area of radius R.
- Escape: exponential survival in target with lambda = (9/7) X0
  (pair production).
- Measurement: smear (x,y) on calorimeter and smear energy.
- Reconstruction: eta, phi from (x_meas, y_meas, z_cal - z_vtx).

This is intentionally simple but consistent and L-dependent via:
- path length in material before exiting target
- vertex depth z_vtx which affects geometry and separation at the
  calorimeter
"""

import numpy as np

# Radiation length of polyethylene
X0 = 4.7736 / 0.93  # cm


def intersect_calo_plane(photon_p4, z_vtx_cm, z_cal_cm):
    """
    Intersect photon trajectory with a plane z = z_cal.

    Parameters
    ----------
    photon_p4 : ROOT.TLorentzVector
        Photon 4-vector in LAB.
    z_vtx_cm : float
        Production vertex z (cm).
    z_cal_cm : float
        Calorimeter plane position z (cm).

    Returns
    -------
    ok : bool
        True if the photon goes forward and intersects the plane.
    x_cm, y_cm : float
        Intersection coordinates on the plane (cm).
    """
    px = photon_p4.Px()
    py = photon_p4.Py()
    pz = photon_p4.Pz()
    p = np.sqrt(px * px + py * py + pz * pz)

    if p <= 0.0:
        return False, 0.0, 0.0

    nx, ny, nz = px / p, py / p, pz / p

    if nz <= 0.0:
        return False, 0.0, 0.0

    t = (z_cal_cm - z_vtx_cm) / nz
    if t <= 0.0:
        return False, 0.0, 0.0

    x = 0.0 + t * nx
    y = 0.0 + t * ny
    return True, x, y


def passes_calo_aperture(x_cm, y_cm, R_cm):
    """
    Circular active area cut.

    Parameters
    ----------
    x_cm, y_cm : float
        Impact point on calo plane.
    R_cm : float
        Active radius.

    Returns
    -------
    ok : bool
        True if inside active area.
    """
    r2 = x_cm * x_cm + y_cm * y_cm
    return r2 <= (R_cm * R_cm)


def photon_escape_prob(photon_p4, z_vtx_cm, L_cm, R_tgt_cm):
    """
    Survival probability for a photon to exit the target.

    Use a simple high-energy approximation for pair production length:
        lambda_pair = (9/7) * X0

    Verify both forward and lateral exit.

    Parameters
    ----------
    photon_p4 : ROOT.TLorentzVector
        Photon 4-vector in LAB.
    z_vtx_cm : float
        Production depth inside target.
    L_cm : float
        Target thickness.
    R_tgt_cm : float
        Target radius (cm).

    Returns
    -------
    p_esc : float
        Escape probability in [0,1].
    """
    if z_vtx_cm >= L_cm:
        return 0.0

    px, py, pz = photon_p4.Px(), photon_p4.Py(), photon_p4.Pz()
    p = np.sqrt(px * px + py * py + pz * pz)
    nx, ny, nz = px / p, py / p, pz / p

    if not np.isfinite(nz) or nz <= 0.0:
        return 0.0

    # forward path length
    l_front = (L_cm - z_vtx_cm) / nz

    # lateral path length
    nT2 = np.sqrt(nx * nx + ny * ny)
    if nT2 > 0.0 and np.isfinite(nT2):
        nT = float(np.sqrt(nT2))
        l_side = R_tgt_cm / nT
    else:
        # exactly along z -> never exits laterally
        l_side = float("inf")

    lambda_pair = (9.0 / 7.0) * X0
    p_esc = np.exp(-min(l_front, l_side) / lambda_pair)

    # Clamp to [0, 1] against tiny numerical overshoots
    if p_esc < 0.0:
        return 0.0
    if p_esc > 1.0:
        return 1.0
    return p_esc


def smear_energy(rng, E_GeV, a=0.12, b=0.02, c=0.0):
    """
    Implement simple calorimeter energy resolution model.

    sigma_E/E = sqrt( (a/sqrt(E))^2 + b^2 + (c/E)^2 ), with E in GeV.

    Parameters
    ----------
    rng : np.random.Generator
    E_GeV : float
    a, b, c : float
        Resolution parameters.

    Returns
    -------
    E_meas : float
        Smeared energy (GeV), clipped to be >= 0.
    """
    E = max(E_GeV, 1e-9)
    rel = np.sqrt((a / np.sqrt(E)) ** 2 + b**2 + (c / E) ** 2)
    sigma = rel * E
    E_meas = rng.normal(E, sigma)
    return max(E_meas, 0.0)


def smear_position(rng, x_cm, y_cm, sigma_xy_cm=0.5):
    """
    Gaussian smearing of the shower centroid on calo plane.

    Parameters
    ----------
    sigma_xy_cm : float
        Position resolution (cm).

    Returns
    -------
    x_meas, y_meas : float
        Measured hit coordinates.
    """
    x_meas = rng.normal(x_cm, sigma_xy_cm)
    y_meas = rng.normal(y_cm, sigma_xy_cm)
    return x_meas, y_meas


def eta_phi_from_hit(x_cm, y_cm, z_vtx_cm, z_cal_cm):
    """
    Reconstruct direction (eta, phi) from measured hit position.

    Parameters
    ----------
    x_cm, y_cm : float
        Measured hit coordinates.
    z_vtx_cm : float
    z_cal_cm : float

    Returns
    -------
    eta, phi : float
        Reconstructed direction in (eta, phi) space.
    """
    dz = z_cal_cm - z_vtx_cm
    if dz <= 0.0:
        return float("nan"), float("nan")

    r = np.sqrt(x_cm * x_cm + y_cm * y_cm)
    theta = np.atan2(r, dz)
    eta = -np.log(np.tan(0.5 * theta))
    phi = np.atan2(y_cm, x_cm)
    return eta, phi


def deltaR(eta1, phi1, eta2, phi2):
    """
    Compute deltaR = sqrt(deta^2 + dphi^2).

    Returns
    -------
    dR : float
        Delta R separation between the two clusters.
    """
    dphi = np.atan2(np.sin(phi1 - phi2), np.cos(phi1 - phi2))
    deta = eta1 - eta2
    return np.sqrt(deta * deta + dphi * dphi)


def detect_two_photons(
    rng,
    g1_lab,
    g2_lab,
    z_vtx_cm,
    L_cm,
    z_cal_cm,
    R_calo_cm,
    R_tgt_cm,
    E_thr_GeV=0.1,
    deltaR_min=0.02,
    sigma_xy_cm=0.5,
    res_a=0.12,
    res_b=0.02,
):
    """
    Full toy detection chain for two photons.

    Steps (per photon):

    - escape in target (stochastic)
    - intersect plane and aperture cut
    - smear energy and position
    - apply energy threshold on MEASURED energy

    Finally:

    - reconstruct eta/phi from smeared (x,y)
    - apply cluster separation deltaR > deltaR_min

    Parameters
    ----------
    rng : np.random.Generator
    g1_lab, g2_lab : ROOT.TLorentzVector
        Photon 4-vectors in LAB.
    z_vtx_cm : float
        Production vertex z (cm).
    L_cm : float
        Target thickness (cm).
    z_cal_cm : float
        Calorimeter plane position z (cm).
    R_calo_cm : float
        Calorimeter active radius (cm).
    R_tgt_cm : float
        Target radius (cm).
    E_thr_GeV : float
        Energy threshold on MEASURED energy (GeV).
    deltaR_min : float
        Minimum separation between the two clusters.
    sigma_xy_cm : float
        Position resolution (cm).
    res_a, res_b : float
        Energy resolution parameters.

    Returns
    -------
    ok : bool
        True if the event is reconstructed as two resolved photons.
    out : dict
        If ok, contains measured kinematics for both photons and
        deltaR.
    """
    # ---------- escape ----------
    p1 = photon_escape_prob(g1_lab, z_vtx_cm, L_cm, R_tgt_cm)
    p2 = photon_escape_prob(g2_lab, z_vtx_cm, L_cm, R_tgt_cm)
    if rng.random() > p1:
        return False, {}
    if rng.random() > p2:
        return False, {}

    # ---------- geometry ----------
    ok1, x1, y1 = intersect_calo_plane(g1_lab, z_vtx_cm, z_cal_cm)
    ok2, x2, y2 = intersect_calo_plane(g2_lab, z_vtx_cm, z_cal_cm)
    if not ok1 or not ok2:
        return False, {}
    if not passes_calo_aperture(x1, y1, R_calo_cm):
        return False, {}
    if not passes_calo_aperture(x2, y2, R_calo_cm):
        return False, {}

    # ---------- measurement ----------
    E1_meas = smear_energy(rng, g1_lab.E(), a=res_a, b=res_b, c=0.0)
    E2_meas = smear_energy(rng, g2_lab.E(), a=res_a, b=res_b, c=0.0)

    # threshold on measured energy
    if E1_meas < E_thr_GeV or E2_meas < E_thr_GeV:
        return False, {}

    x1m, y1m = smear_position(rng, x1, y1, sigma_xy_cm=sigma_xy_cm)
    x2m, y2m = smear_position(rng, x2, y2, sigma_xy_cm=sigma_xy_cm)

    eta1, phi1 = eta_phi_from_hit(x1m, y1m, z_vtx_cm, z_cal_cm)
    eta2, phi2 = eta_phi_from_hit(x2m, y2m, z_vtx_cm, z_cal_cm)

    if not (np.isfinite(eta1) and np.isfinite(eta2)):
        return False, {}

    dR = deltaR(eta1, phi1, eta2, phi2)
    if dR < deltaR_min:
        return False, {}

    out = {
        "E1_meas": E1_meas,
        "E2_meas": E2_meas,
        "eta1_meas": eta1,
        "eta2_meas": eta2,
        "phi1_meas": phi1,
        "phi2_meas": phi2,
        "dR_meas": dR,
        "x1_meas_cm": x1m,
        "y1_meas_cm": y1m,
        "x2_meas_cm": x2m,
        "y2_meas_cm": y2m,
    }
    return True, out
