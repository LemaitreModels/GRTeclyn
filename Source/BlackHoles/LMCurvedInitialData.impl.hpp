/* GRTeclyn
 * Copyright 2022 The GRTL collaboration.
 * Please refer to LICENSE in GRTeclyn's root directory.
 */

#if !defined(LMCURVEDINITIALDATA_HPP_)
#error "This file should only be included through LMCurvedInitialData.hpp"
#endif

#ifndef LMCURVEDINITIALDATA_IMPL_HPP_
#define LMCURVEDINITIALDATA_IMPL_HPP_

#include "BinaryBHInitialData.hpp" // for the Lapse enum
#include "TensorAlgebra.hpp"

#include <AMReX.H>

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>

// --------------------------------------------------------------------------
// Barycentric interpolation (same rule as LMInitialData::bary)
// --------------------------------------------------------------------------
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE void LMCurvedInitialData::bary(amrex::Real xq,
                                                  const amrex::Real *nodes,
                                                  const amrex::Real *w, int n,
                                                  amrex::Real *out)
{
    for (int j = 0; j < n; ++j)
    {
        if (std::abs(xq - nodes[j]) < 1e-13)
        {
            for (int k = 0; k < n; ++k)
            {
                out[k] = 0.0;
            }
            out[j] = 1.0;
            return;
        }
    }
    amrex::Real sum = 0.0;
    for (int j = 0; j < n; ++j)
    {
        out[j] = w[j] / (xq - nodes[j]);
        sum += out[j];
    }
    for (int j = 0; j < n; ++j)
    {
        out[j] /= sum;
    }
}

// --------------------------------------------------------------------------
// One quasi-isotropic Kerr hole (verbatim from operators.py)
// --------------------------------------------------------------------------
// _kerr_psi: psi_QI = W/r with W = [(k + m r + r^2)^2 + a^2 r^2 mu^2]^(1/4),
// k = (m^2 - a^2)/4, mu = z/r.  W is smooth and nonzero at r = 0.
[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
LMCurvedInitialData::kerr_psi(const amrex::Real X[3], amrex::Real m,
                              amrex::Real a)
{
    const amrex::Real r =
        std::sqrt(X[0] * X[0] + X[1] * X[1] + X[2] * X[2]);
    const amrex::Real mu = X[2] / r;
    const amrex::Real k  = 0.25 * (m * m - a * a);
    const amrex::Real P  = k + m * r + r * r;
    const amrex::Real W =
        std::pow(P * P + a * a * r * r * mu * mu, 0.25);
    return W / r;
}

// _kerr_metric: gt_ij = delta_ij + a^2 h w_i w_j, w = (-y, x, 0),
// h = (1 + 2 m rbar / Sigma) / (Sigma r^2).  At a = 0 the update is exactly 0.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
LMCurvedInitialData::kerr_metric(const amrex::Real X[3], amrex::Real m,
                                 amrex::Real a, amrex::Real gt[3][3])
{
    const amrex::Real r2 = X[0] * X[0] + X[1] * X[1] + X[2] * X[2];
    const amrex::Real r  = std::sqrt(r2);
    const amrex::Real k  = 0.25 * (m * m - a * a);
    const amrex::Real rbar  = r + m + k / r;
    const amrex::Real Sigma = rbar * rbar + a * a * (X[2] * X[2] / r2);
    const amrex::Real h = (1.0 + 2.0 * m * rbar / Sigma) / (Sigma * r2);
    const amrex::Real om[3] = {-X[1], X[0], 0.0};
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            gt[i][j] = ((i == j) ? 1.0 : 0.0) +
                       (a * a * h) * om[i] * om[j];
        }
    }
}

// _kerr_At: the closed-form Cartesian At_ij of one QI Kerr hole, built from the
// two nonvanishing spherical components with the sin^k(theta) prefactors folded
// into the Cartesian vectors, so nothing divides by rho on the axis.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
LMCurvedInitialData::kerr_At(const amrex::Real X[3], amrex::Real m,
                             amrex::Real a, amrex::Real At[3][3])
{
    const amrex::Real r2 = X[0] * X[0] + X[1] * X[1] + X[2] * X[2];
    const amrex::Real r  = std::sqrt(r2);
    const amrex::Real k  = 0.25 * (m * m - a * a);
    const amrex::Real rbar   = r + m + k / r;
    const amrex::Real rbar_r = 1.0 - k / r2;
    const amrex::Real Sigma  = rbar * rbar + a * a * (X[2] * X[2] / r2);
    const amrex::Real Delta  = r2 * rbar_r * rbar_r;
    const amrex::Real sin2   = (X[0] * X[0] + X[1] * X[1]) / r2;
    const amrex::Real Acal =
        (rbar * rbar + a * a) * (rbar * rbar + a * a) - Delta * a * a * sin2;
    const amrex::Real dA_drbar =
        4.0 * rbar * (rbar * rbar + a * a) -
        a * a * sin2 * (2.0 * rbar - 2.0 * m);
    const amrex::Real sA = std::sqrt(Acal);
    const amrex::Real E =
        -m * a * (Acal - rbar * dA_drbar) / (r2 * r2 * r * Sigma * sA);
    const amrex::Real F = -2.0 * m * a * a * a * rbar * rbar_r * X[2] /
                          (r2 * r2 * Sigma * sA);
    const amrex::Real om[3] = {-X[1], X[0], 0.0};
    // u = z X / r^2 - e_z  (r * grad(theta) * sin(theta))
    const amrex::Real uv[3] = {X[2] * X[0] / r2, X[2] * X[1] / r2,
                               X[2] * X[2] / r2 - 1.0};
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            At[i][j] = E * (X[i] * om[j] + om[i] * X[j]) +
                       F * (uv[i] * om[j] + om[i] * uv[j]);
        }
    }
}

// --------------------------------------------------------------------------
// The shipped spectral fields — one (A,B,phi) pass shared by all seven
// --------------------------------------------------------------------------
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
LMCurvedInitialData::fields_at(amrex::Real x, amrex::Real y, amrex::Real z,
                               amrex::Real *out) const
{
    const amrex::Real b   = m_params.b;
    const amrex::Real rho = std::sqrt(x * x + y * y);
    const amrex::Real phi = std::atan2(y, x);

    const amrex::Real r1 = std::sqrt(rho * rho + (z - b) * (z - b));
    const amrex::Real r2 = std::sqrt(rho * rho + (z + b) * (z + b));
    const amrex::Real xi = (r1 + r2) / (2.0 * b);
    amrex::Real Bq       = (r2 - r1) / (2.0 * b);
    Bq                   = std::min(std::max(Bq, -1.0), 1.0);
    amrex::Real ratio    = (xi - 1.0) / (xi + 1.0);
    ratio                = std::min(std::max(ratio, 0.0), 1.0);
    const amrex::Real Aq = std::sqrt(ratio);

    amrex::Real tA[LM_CURVED_MAX_NODES];
    amrex::Real tB[LM_CURVED_MAX_NODES];
    bary(Aq, m_params.A, m_params.wA, m_params.nA, tA);
    bary(Bq, m_params.B, m_params.wB, m_params.nB, tB);

    const int nA                   = m_params.nA;
    const int nB                   = m_params.nB;
    const amrex::Real one_minus_B2 = 1.0 - Bq * Bq;

    for (int f = 0; f < LM_CURVED_NFIELDS; ++f)
    {
        amrex::Real acc = 0.0;
        for (int t = 0; t < m_params.ncos; ++t)
        {
            amrex::Real coef = 0.0;
            for (int i = 0; i < nA; ++i)
            {
                amrex::Real inner = 0.0;
                for (int j = 0; j < nB; ++j)
                {
                    inner +=
                        tB[j] *
                        m_params.C[f][(i * nB + j) * m_params.ncos + t];
                }
                coef += tA[i] * inner;
            }
            const int k = m_params.cos_m[t];
            const amrex::Real w =
                (k == 0) ? 1.0 : std::pow(one_minus_B2, 0.5 * k);
            acc += w * coef * std::cos(k * phi);
        }
        for (int t = 0; t < m_params.nsin; ++t)
        {
            amrex::Real coef = 0.0;
            for (int i = 0; i < nA; ++i)
            {
                amrex::Real inner = 0.0;
                for (int j = 0; j < nB; ++j)
                {
                    inner +=
                        tB[j] *
                        m_params.S[f][(i * nB + j) * m_params.nsin + t];
                }
                coef += tA[i] * inner;
            }
            const int k         = m_params.sin_m[t];
            const amrex::Real w = std::pow(one_minus_B2, 0.5 * k); // k >= 1
            acc += w * coef * std::sin(k * phi);
        }
        out[f] = acc;
    }
}

// --------------------------------------------------------------------------
// The consumer state
// --------------------------------------------------------------------------
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
LMCurvedInitialData::consumer_state(amrex::Real x, amrex::Real y,
                                    amrex::Real z, amrex::Real &psi_out,
                                    amrex::Real &chi_out,
                                    Tensor<2, amrex::Real> &h_out,
                                    Tensor<2, amrex::Real> &A_out) const
{
    const amrex::Real b = m_params.b;

    amrex::Real fld[LM_CURVED_NFIELDS];
    fields_at(x, y, z, fld);
    const amrex::Real u = fld[0];
    const amrex::Real Lb[3][3] = {{fld[1], fld[2], fld[3]},
                                  {fld[2], fld[4], fld[5]},
                                  {fld[3], fld[5], fld[6]}};

    // Hole-centred coordinates: A at (0,0,+b), B at (0,0,-b);
    // a = chi m, spin along z with the sign carried by chi.
    const amrex::Real YA[3] = {x, y, z - b};
    const amrex::Real YB[3] = {x, y, z + b};
    const amrex::Real mA = m_params.m_A, mB = m_params.m_B;
    const amrex::Real aA = m_params.chi_A * mA;
    const amrex::Real aB = m_params.chi_B * mB;

    // Attenuations: each hole is switched off by proximity to the OPPOSITE
    // puncture, f_(+-) = 1 - exp[-(r_(-+)/omega)^p], omega = omega_over_b * b.
    // The outer cutoff F == 1 (format 3 fixes F_kind 'none').
    const amrex::Real rA =
        std::sqrt(YA[0] * YA[0] + YA[1] * YA[1] + YA[2] * YA[2]);
    const amrex::Real rB =
        std::sqrt(YB[0] * YB[0] + YB[1] * YB[1] + YB[2] * YB[2]);
    const amrex::Real omega = m_params.omega_over_b * b;
    const amrex::Real pexp  = static_cast<amrex::Real>(m_params.atten_p);
    const amrex::Real fA = 1.0 - std::exp(-std::pow(rB / omega, pexp));
    const amrex::Real fB = 1.0 - std::exp(-std::pow(rA / omega, pexp));

    // The superposed conformal metric and Psi.
    amrex::Real gA[3][3], gB[3][3], gt[3][3];
    kerr_metric(YA, mA, aA, gA);
    kerr_metric(YB, mB, aB, gB);
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            const amrex::Real delta = (i == j) ? 1.0 : 0.0;
            gt[i][j] = delta + fA * (gA[i][j] - delta) + fB * (gB[i][j] - delta);
        }
    }
    const amrex::Real Psi = kerr_psi(YA, mA, aA) + kerr_psi(YB, mB, aB) - 1.0;

    // det and inverse of gt (symmetric 3x3, closed form).
    const amrex::Real det =
        gt[0][0] * (gt[1][1] * gt[2][2] - gt[1][2] * gt[2][1]) -
        gt[0][1] * (gt[1][0] * gt[2][2] - gt[1][2] * gt[2][0]) +
        gt[0][2] * (gt[1][0] * gt[2][1] - gt[1][1] * gt[2][0]);
    amrex::Real gti[3][3];
    gti[0][0] = (gt[1][1] * gt[2][2] - gt[1][2] * gt[2][1]) / det;
    gti[0][1] = (gt[0][2] * gt[2][1] - gt[0][1] * gt[2][2]) / det;
    gti[0][2] = (gt[0][1] * gt[1][2] - gt[0][2] * gt[1][1]) / det;
    gti[1][0] = gti[0][1];
    gti[1][1] = (gt[0][0] * gt[2][2] - gt[0][2] * gt[2][0]) / det;
    gti[1][2] = (gt[0][2] * gt[1][0] - gt[0][0] * gt[1][2]) / det;
    gti[2][0] = gti[0][2];
    gti[2][1] = gti[1][2];
    gti[2][2] = (gt[0][0] * gt[1][1] - gt[0][1] * gt[1][0]) / det;

    // Mt = (At(+) + At(-)), NOT attenuated, trace-removed against gt; then
    // At = Mt + Lb, trace-removed AGAIN against the local gt (the second
    // removal deletes the interpolated Lb's truncation-level trace) — the same
    // two-step order as the exporter's reference recipe.
    amrex::Real AtA[3][3], AtB[3][3], Mt[3][3];
    kerr_At(YA, mA, aA, AtA);
    kerr_At(YB, mB, aB, AtB);
    amrex::Real tr = 0.0;
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            Mt[i][j] = AtA[i][j] + AtB[i][j];
        }
    }
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            tr += gti[i][j] * Mt[i][j];
        }
    }
    amrex::Real At[3][3];
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            At[i][j] = Mt[i][j] - gt[i][j] * (tr / 3.0) + Lb[i][j];
        }
    }
    tr = 0.0;
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            tr += gti[i][j] * At[i][j];
        }
    }
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            At[i][j] -= gt[i][j] * (tr / 3.0);
        }
    }

    // The BSSN state.
    const amrex::Real psi   = Psi + u;
    const amrex::Real Dm13  = std::pow(det, -1.0 / 3.0);
    const amrex::Real chi   = std::pow(psi, -4.0) * Dm13;
    const amrex::Real scale = chi * std::pow(psi, -2.0);
    psi_out                 = psi;
    chi_out                 = chi;
    FOR (i, j)
    {
        h_out[i][j] = gt[i][j] * Dm13;
        A_out[i][j] = scale * At[i][j];
    }
}

// --------------------------------------------------------------------------
// Setting the state
// --------------------------------------------------------------------------
AMREX_FORCE_INLINE AMREX_GPU_DEVICE void
LMCurvedInitialData::operator()(int ix, int iy, int iz,
                                const amrex::Array4<amrex::Real> &state) const
{
    const amrex::CellData<amrex::Real> &cell = state.cellData(ix, iy, iz);
    Coordinates coords(amrex::IntVect(ix, iy, iz), m_dx, m_params.center);

    amrex::Real psi_here = 0.0, chi = 0.0;
    Tensor<2, amrex::Real> h_here, A_here;
    consumer_state(coords.x, coords.y, coords.z, psi_here, chi, h_here,
                   A_here);

    cell[c_chi] = chi;
    FOR2_SYM(i, j)
    {
        cell[VAR_IDX(c_h11, i, j)] = h_here[i][j];
        cell[VAR_IDX(c_A11, i, j)] = A_here[i][j];
    }
    // K == 0 exactly in this sector (maximal slicing per hole survives the
    // attenuated sum); the caller has zeroed every component already, so K,
    // Theta, the shift and Gamma^i stay zero here.  Gamma^i is then computed
    // from h_ij by GammaCalculator over the grid — see BinaryBHLevel.

    switch (m_params.initial_lapse)
    {
    case Lapse::ONE:
        cell[c_lapse] = 1.0;
        break;
    case Lapse::PRE_COLLAPSED:
        cell[c_lapse] = std::sqrt(chi);
        break;
    case Lapse::CHI:
        cell[c_lapse] = chi;
        break;
    default:
        amrex::Abort("LMCurvedInitialData: unsupported initial lapse");
    }
}

// --------------------------------------------------------------------------
// Validation
// --------------------------------------------------------------------------
inline void LMCurvedInitialData::validate_staggering(const params_t &params,
                                                     double dx, double tol)
{
    // psi_QI diverges at each puncture exactly as psi_BL does — W(0) =
    // sqrt(m^2 - a^2)/2 > 0 below extremality — so the staggering requirement
    // is the flat class's, verbatim.
    for (int sign = -1; sign <= 1; sign += 2)
    {
        double coord[3] = {params.center[0], params.center[1],
                           params.center[2] + sign * params.b};
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            const double t   = coord[d] / dx - 0.5;
            const double dev = std::abs(t - std::round(t));
            if (dev < tol)
            {
                amrex::Abort(
                    "LMCurvedInitialData: puncture lands within " +
                    std::to_string(tol) +
                    " cells of a cell CENTRE in direction " +
                    std::to_string(d) + " (dx = " + std::to_string(dx) +
                    "); psi_QI diverges there.  Change N_full or L_full so "
                    "the punctures fall on cell corners.");
            }
        }
    }
}

[[nodiscard]] inline std::array<double, 4>
LMCurvedInitialData::validate(const std::string &reference_file) const
{
    std::ifstream f(reference_file);
    if (!f)
    {
        amrex::Abort("LMCurvedInitialData::validate: cannot open " +
                     reference_file);
    }
    double dpsi = 0.0, dchi = 0.0, dh = 0.0, dA = 0.0;
    long n = 0;
    std::string line;
    while (std::getline(f, line))
    {
        if (line.empty() || line[0] == '#')
        {
            continue;
        }
        std::istringstream ss(line);
        double x = 0, y = 0, z = 0, psi_ref = 0, chi_ref = 0;
        double h_ref[6] = {}, A_ref[6] = {};
        ss >> x >> y >> z >> psi_ref >> chi_ref;
        for (double &v : h_ref)
        {
            ss >> v;
        }
        for (double &v : A_ref)
        {
            ss >> v;
        }
        amrex::Real psi_got = 0.0, chi_got = 0.0;
        Tensor<2, amrex::Real> h_got, A_got;
        consumer_state(x, y, z, psi_got, chi_got, h_got, A_got);
        dpsi = std::max(dpsi, std::abs(psi_got - psi_ref));
        dchi = std::max(dchi, std::abs(chi_got - chi_ref));
        const int iu[6] = {0, 0, 0, 1, 1, 2};
        const int ju[6] = {0, 1, 2, 1, 2, 2};
        for (int k = 0; k < 6; ++k)
        {
            dh = std::max(dh, std::abs(h_got[iu[k]][ju[k]] - h_ref[k]));
            dA = std::max(dA, std::abs(A_got[iu[k]][ju[k]] - A_ref[k]));
        }
        ++n;
    }
    amrex::Print() << "LMCurvedInitialData::validate: " << n
                   << " reference points, max|dpsi| = " << dpsi
                   << ", max|dchi| = " << dchi << ", max|dh| = " << dh
                   << ", max|dA| = " << dA << "\n";
    return {dpsi, dchi, dh, dA};
}

#endif /* LMCURVEDINITIALDATA_IMPL_HPP_ */
