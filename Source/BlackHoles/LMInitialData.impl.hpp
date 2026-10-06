/* GRTeclyn
 * Copyright 2022 The GRTL collaboration.
 * Please refer to LICENSE in GRTeclyn's root directory.
 */

#if !defined(LMINITIALDATA_HPP_)
#error "This file should only be included through LMInitialData.hpp"
#endif

#ifndef LMINITIALDATA_IMPL_HPP_
#define LMINITIALDATA_IMPL_HPP_

#include "BinaryBHInitialData.hpp" // for the Lapse enum
#include "TensorAlgebra.hpp"

#include <AMReX.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// --------------------------------------------------------------------------
// Barycentric interpolation
// --------------------------------------------------------------------------
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE void LMInitialData::bary(amrex::Real xq,
                                           const amrex::Real *nodes,
                                           const amrex::Real *w, int n,
                                           amrex::Real *out)
{
    // An exact node hit would divide by zero; the limit is the unit vector.
    // This is the case a naive implementation gets wrong, and query points do
    // land on nodes (the reference-table check exercises it).
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
// The spectral correction u
// --------------------------------------------------------------------------
[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
LMInitialData::u(amrex::Real x, amrex::Real y, amrex::Real z) const
{
    const amrex::Real b   = m_params.b;
    const amrex::Real rho = std::sqrt(x * x + y * y);
    const amrex::Real phi = std::atan2(y, x);

    // Inverse ABT map: r1, r2 to the punctures at (0,0,±b);
    // xi = (r1+r2)/2b, B = (r2-r1)/2b, A = sqrt((xi-1)/(xi+1)).
    const amrex::Real r1 = std::sqrt(rho * rho + (z - b) * (z - b));
    const amrex::Real r2 = std::sqrt(rho * rho + (z + b) * (z + b));
    const amrex::Real xi = (r1 + r2) / (2.0 * b);
    amrex::Real Bq       = (r2 - r1) / (2.0 * b);
    Bq                   = std::min(std::max(Bq, -1.0), 1.0);
    amrex::Real ratio    = (xi - 1.0) / (xi + 1.0);
    ratio                = std::min(std::max(ratio, 0.0), 1.0);
    const amrex::Real Aq = std::sqrt(ratio);

    amrex::Real tA[LM_MAX_NODES];
    amrex::Real tB[LM_MAX_NODES];
    bary(Aq, m_params.A, m_params.wA, m_params.nA, tA);
    bary(Bq, m_params.B, m_params.wB, m_params.nB, tB);

    // Interpolate the phi-modal coefficients in (A,B) — B inner, A outer, the
    // same order as the reference evaluator — restore the axis factor at the
    // query point, then sum the trig series.
    // The (A,B) rule is linear, so interpolating the coefficients and then
    // summing the series equals interpolating per phi-plane then in phi.
    //
    // The stored C,S are the SMOOTH factor (format 2): the wavenumber-k part of
    // u carries w(k,B) = (1-B^2)^(k/2) at the axis, which is not a polynomial in
    // B — for odd k not even smooth at B=+-1 — so it is divided out before the
    // solver's polynomial interpolation and multiplied back here.  Applying it
    // AFTER the (A,B) interpolation is what makes this evaluator the truncated
    // spectral expansion exactly rather than an approximation to it.
    // `one_minus_B2 >= 0` because Bq is clamped to [-1,1] above; it is 0 exactly
    // on the axis, where every k>0 mode correctly contributes nothing.
    const int nA                  = m_params.nA;
    const int nB                  = m_params.nB;
    const amrex::Real one_minus_B2 = 1.0 - Bq * Bq;
    amrex::Real out               = 0.0;

    for (int t = 0; t < m_params.ncos; ++t)
    {
        amrex::Real coef = 0.0;
        for (int i = 0; i < nA; ++i)
        {
            amrex::Real inner = 0.0;
            for (int j = 0; j < nB; ++j)
            {
                inner += tB[j] * m_params.C[(i * nB + j) * m_params.ncos + t];
            }
            coef += tA[i] * inner;
        }
        const int k = m_params.cos_m[t];
        const amrex::Real w =
            (k == 0) ? 1.0 : std::pow(one_minus_B2, 0.5 * k);
        out += w * coef * std::cos(k * phi);
    }
    for (int t = 0; t < m_params.nsin; ++t)
    {
        amrex::Real coef = 0.0;
        for (int i = 0; i < nA; ++i)
        {
            amrex::Real inner = 0.0;
            for (int j = 0; j < nB; ++j)
            {
                inner += tB[j] * m_params.S[(i * nB + j) * m_params.nsin + t];
            }
            coef += tA[i] * inner;
        }
        const int k = m_params.sin_m[t];
        const amrex::Real w = std::pow(one_minus_B2, 0.5 * k); // k >= 1 here
        out += w * coef * std::sin(k * phi);
    }
    return out;
}

[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
LMInitialData::psi_BL(amrex::Real x, amrex::Real y, amrex::Real z) const
{
    const amrex::Real b = m_params.b;
    const amrex::Real rA =
        std::sqrt(x * x + y * y + (z - b) * (z - b));
    const amrex::Real rB =
        std::sqrt(x * x + y * y + (z + b) * (z + b));
    return 1.0 + m_params.m_A / (2.0 * rA) + m_params.m_B / (2.0 * rB);
}

[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
LMInitialData::psi(amrex::Real x, amrex::Real y, amrex::Real z) const
{
    return psi_BL(x, y, z) + u(x, y, z);
}

// --------------------------------------------------------------------------
// Closed-form Bowen-York Ahat^{ij}
// --------------------------------------------------------------------------
[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Tensor::Rank2
LMInitialData::Ahat(amrex::Real x, amrex::Real y, amrex::Real z) const
{
    Tensor::Rank2 out;
    FOR (i, j) { out(i, j) = 0.0; }

    for (int puncture = 0; puncture < 2; ++puncture)
    {
        const amrex::Real zc =
            (puncture == 0) ? (z - m_params.b) : (z + m_params.b);
        const amrex::Real *P =
            (puncture == 0) ? m_params.P_A : m_params.P_B;
        const amrex::Real *Sv =
            (puncture == 0) ? m_params.S_A : m_params.S_B;

        const amrex::Real r =
            std::max(std::sqrt(x * x + y * y + zc * zc), 1e-12);
        const amrex::Real n[3] = {x / r, y / r, zc / r};

        // momentum: (3/2r^2)[P^i n^j + n^i P^j - (delta^ij - n^i n^j)(P.n)]
        // (identical expression to BoostedBHInitialData::Aij)
        if (P[0] != 0.0 || P[1] != 0.0 || P[2] != 0.0)
        {
            const amrex::Real Pn = n[0] * P[0] + n[1] * P[1] + n[2] * P[2];
            FOR (i, j)
            {
                const amrex::Real delta = (i == j) ? 1.0 : 0.0;
                out(i, j) += 1.5 *
                             (P[i] * n[j] + P[j] * n[i] -
                              (delta - n[i] * n[j]) * Pn) /
                             (r * r);
            }
        }
        // spin: (3/r^3)(v^i n^j + n^i v^j),  v = S x n
        if (Sv[0] != 0.0 || Sv[1] != 0.0 || Sv[2] != 0.0)
        {
            const amrex::Real v[3] = {Sv[1] * n[2] - Sv[2] * n[1],
                                      Sv[2] * n[0] - Sv[0] * n[2],
                                      Sv[0] * n[1] - Sv[1] * n[0]};
            FOR (i, j)
            {
                out(i, j) += 3.0 * (v[i] * n[j] + n[i] * v[j]) / (r * r * r);
            }
        }
    }
    return out;
}

// --------------------------------------------------------------------------
// Setting the state
// --------------------------------------------------------------------------
AMREX_FORCE_INLINE AMREX_GPU_DEVICE void
LMInitialData::operator()(int ix, int iy, int iz,
                          const amrex::Array4<amrex::Real> &state) const
{
    const amrex::CellData<amrex::Real> &cell = state.cellData(ix, iy, iz);
    Coordinates coords(amrex::IntVect(ix, iy, iz), m_dx, m_params.center);

    const amrex::Real psi_here = psi(coords.x, coords.y, coords.z);
    const amrex::Real chi      = std::pow(psi_here, -4.0);
    cell[c_chi]                = chi;

    FOR2_SYM(i, j)
    {
        cell[sym_var_idx(c_h11, i, j)] = TensorAlgebra::delta(i, j);
    }

    // A_ij(CCZ4) = chi * K_ij with K_ij = psi^{-2} Ahat_ij, i.e.
    // A_ij = psi^{-6} Ahat_ij = chi^{3/2} Ahat_ij — the same relation
    // BinaryBHInitialData uses.  Ahat is analytically trace-free with h_ij =
    // delta_ij, so no make_trace_free is applied here either: this keeps the
    // two initial-data paths differing ONLY in psi and Ahat.
    Tensor::Rank2 Ahat_here = Ahat(coords.x, coords.y, coords.z);
    FOR2_SYM(i, j)
    {
        cell[sym_var_idx(c_A11, i, j)] = std::pow(chi, 1.5) * Ahat_here(i, j);
    }

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
        amrex::Abort("LMInitialData: unsupported initial lapse");
    }
}

// --------------------------------------------------------------------------
// Validation
// --------------------------------------------------------------------------
inline void LMInitialData::validate_staggering(const params_t &params,
                                               double dx, double tol)
{
    // A puncture sits at grid coordinate center[d] + (+/-)b in z.  Cell centres
    // are at (k+0.5)*dx, so the puncture is ON a cell centre when
    // (coord/dx - 0.5) is an integer.  Require it to be at least `tol` cells
    // away, in every direction.
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
                    "LMInitialData: puncture lands within " +
                    std::to_string(tol) +
                    " cells of a cell CENTRE in direction " +
                    std::to_string(d) + " (dx = " + std::to_string(dx) +
                    "); psi diverges there.  Change N_full or L_full so the "
                    "punctures fall on cell corners.");
            }
        }
    }
}

[[nodiscard]] inline std::array<double, 2>
LMInitialData::validate(const std::string &reference_file) const
{
    std::ifstream f(reference_file);
    if (!f)
    {
        amrex::Abort("LMInitialData::validate: cannot open " + reference_file);
    }
    // Parse on the host, evaluate ON THE DEVICE: the coefficients live in
    // amrex::Gpu::DeviceVector (LMSpectralData), so a host-side psi()/Ahat()
    // dereferences device memory -- harmless on a CPU build, a segfault on a
    // CUDA one (Hortense job 14416404).  See LMCurvedInitialData::validate.
    std::vector<amrex::Real> pts;  // x y z per point
    std::vector<double> ref;       // psi Ahat[6] per point (7)
    std::string line;
    while (std::getline(f, line))
    {
        if (line.empty() || line[0] == '#')
        {
            continue;
        }
        std::istringstream ss(line);
        double v[10] = {};
        for (double &x : v)
        {
            ss >> x;
        }
        pts.insert(pts.end(), v, v + 3);
        ref.insert(ref.end(), v + 3, v + 10);
    }
    const int n = static_cast<int>(pts.size() / 3);

    amrex::Gpu::DeviceVector<amrex::Real> d_pts(pts.size());
    amrex::Gpu::DeviceVector<amrex::Real> d_out(static_cast<std::size_t>(7) * n);
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, pts.begin(), pts.end(),
                          d_pts.begin());
    const LMInitialData self = *this;
    const amrex::Real *P     = d_pts.data();
    amrex::Real *O           = d_out.data();
    // The table is in puncture-centred coordinates, so evaluate directly.
    amrex::ParallelFor(n,
                       [=] AMREX_GPU_DEVICE(int k) noexcept
                       {
                           const amrex::Real x = P[3 * k], y = P[3 * k + 1],
                                             z = P[3 * k + 2];
                           amrex::Real *o      = O + 7 * k;
                           o[0]                = self.psi(x, y, z);
                           Tensor::Rank2 A     = self.Ahat(x, y, z);
                           o[1]                = A(0, 0);
                           o[2]                = A(0, 1);
                           o[3]                = A(0, 2);
                           o[4]                = A(1, 1);
                           o[5]                = A(1, 2);
                           o[6]                = A(2, 2);
                       });
    std::vector<amrex::Real> got(d_out.size());
    amrex::Gpu::copyAsync(amrex::Gpu::deviceToHost, d_out.begin(), d_out.end(),
                          got.begin());
    amrex::Gpu::streamSynchronize();

    double dpsi_max = 0.0;
    double dA_max   = 0.0;
    for (int k = 0; k < n; ++k)
    {
        const amrex::Real *g = got.data() + 7 * k;
        const double *r      = ref.data() + 7 * k;
        dpsi_max             = std::max(dpsi_max, std::abs(g[0] - r[0]));
        for (int m = 1; m < 7; ++m)
        {
            dA_max = std::max(dA_max, std::abs(g[m] - r[m]));
        }
    }
    amrex::Print() << "LMInitialData::validate: " << n << " reference points, "
                   << "max|dpsi| = " << dpsi_max
                   << ", max|dAhat| = " << dA_max << "\n";
    return {dpsi_max, dA_max};
}

#endif /* LMINITIALDATA_IMPL_HPP_ */
