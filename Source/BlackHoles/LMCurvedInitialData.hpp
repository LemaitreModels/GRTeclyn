/* GRTeclyn
 * Copyright 2022 The GRTL collaboration.
 * Please refer to LICENSE in GRTeclyn's root directory.
 */

// Set LM-initial-data CONFORMALLY CURVED spectral puncture initial data.
//
// The curved successor of `LMInitialData` (format 3 against its format 2), for
// the spinning-at-rest sector: two quasi-isotropic conformally-Kerr punctures,
// K == 0 exactly, conformal metric != delta.  The stock `BinaryBH` compute
// hard-wires h_ij = delta_ij, so this class is required rather than convenient:
// handing it curved data would flatten them without error.
//
// The division of labour follows the exporter
// (`lm...curved_puncture.validation.export_grteclyn`, format 3):
//
//   * CLOSED FORMS, evaluated here from header parameters (they are steep or
//     singular at the punctures and must not be sampled): the per-hole
//     quasi-isotropic psi_QI = (Sigma/r^2)^(1/4), conformal metric
//     gt_ij = delta_ij + a^2 h w_i w_j, and extrinsic-curvature seed At_ij;
//     the attenuated superposition with f_(+-) = 1 - exp[-(r_(-+)/omega)^p]
//     (each hole switched off by proximity to the OPPOSITE puncture; the outer
//     cutoff F == 1 in this format); Mt_ij = At(+) + At(-) trace-removed
//     against the superposed gt (NOT attenuated).  All transcribed from the
//     solver's own `operators.py` / `pair.py` closed forms.
//
//   * SHIPPED SPECTRAL FIELDS, interpolated exactly like format 2's u (tensor
//     barycentric in (A,B) on the smooth factor, axis factor w(k,B) restored at
//     the query point, cos/sin series summed): the solved correction u and the
//     six components of (L\~b)_ij — shipping the assembled (L\~b) spares this
//     class both the spectral derivative of b^i and the Christoffels of gt.
//
// The BSSN state, with D = det(gt):
//
//     psi   = psi_QI(+) + psi_QI(-) - 1 + u
//     chi   = psi^-4 D^-(1/3)
//     h_ij  = gt_ij D^-(1/3)                    (unit determinant)
//     At    = Mt + Lb, trace-removed against the LOCAL gt (off the nodes the
//             interpolated Lb carries a truncation-level trace; the removed
//             part is interpolation error, not physics)
//     A_ij  = chi psi^-2 At_ij ;   K = 0 ;   Gamma^i from GammaCalculator
//             on the grid AFTER this functor has filled h_ij (ghosts included).
//
// GOTCHA: psi_QI diverges at the punctures (W(0) = sqrt(m^2-a^2)/2 > 0), same
// staggering requirement as the flat class; `validate_staggering` asserts it.
// Pass `lm_curved_id_reference_file` to check this whole reconstruction against
// the exporter's table (see LMCurvedInitialData::validate) before trusting a
// single evolved step.

#ifndef LMCURVEDINITIALDATA_HPP_
#define LMCURVEDINITIALDATA_HPP_

#include "Coordinates.hpp"
#include "DimensionDefinitions.hpp"
#include "StateVariables.hpp"
#include "Tensor.hpp"

#include <AMReX_Array4.H>
#include <AMReX_REAL.H>

#include <array>
#include <string>

//! Maximum nodes per spectral direction (stack temporaries in the kernel);
//! same bound as the flat reader's LM_MAX_NODES, under its own name so the two
//! headers stay independent.
constexpr int LM_CURVED_MAX_NODES = 128;

//! The exported field count and order: u, then the symmetric (L~b)_ij upper
//! triangle (xx, xy, xz, yy, yz, zz).  Fixed by format 3.
constexpr int LM_CURVED_NFIELDS = 7;

class LMCurvedInitialData
{
  public:
    //! Trivially-copyable view onto data owned by LMCurvedSpectralData.
    struct params_t
    {
        double b{};
        double m_A{};
        double m_B{};
        double chi_A{}; //!< a_A = chi_A m_A, spin along +z (sign included)
        double chi_B{};
        double omega_over_b{};
        int atten_p{};
        std::array<double, AMREX_SPACEDIM> center{};
        int nA{};
        int nB{};
        int nphi{};
        int ncos{};
        int nsin{};
        const amrex::Real *A{nullptr};
        const amrex::Real *B{nullptr};
        const amrex::Real *wA{nullptr};
        const amrex::Real *wB{nullptr};
        const int *cos_m{nullptr};
        const int *sin_m{nullptr};
        const amrex::Real *C[LM_CURVED_NFIELDS] = {};
        const amrex::Real *S[LM_CURVED_NFIELDS] = {};
        int initial_lapse{};
    };

    AMREX_FORCE_INLINE
    explicit LMCurvedInitialData(params_t a_params, double a_dx)
        : m_params(a_params), m_dx(a_dx)
    {
    }

    AMREX_GPU_DEVICE void
    operator()(int ix, int iy, int iz,
               const amrex::Array4<amrex::Real> &state) const;

    //! All shipped fields at a point relative to the grid centre — one inverse
    //! map and one barycentric pass, shared across the seven fields.
    AMREX_GPU_HOST_DEVICE void fields_at(amrex::Real x, amrex::Real y,
                                         amrex::Real z,
                                         amrex::Real *out) const;

    //! The full consumer state at a point: psi, chi, h_ij, A_ij (BSSN).
    AMREX_GPU_HOST_DEVICE void
    consumer_state(amrex::Real x, amrex::Real y, amrex::Real z,
                   amrex::Real &psi_out, amrex::Real &chi_out,
                   Tensor::Rank2 &h_out,
                   Tensor::Rank2 &A_out) const;

    //! Abort unless every puncture is clear of a cell centre by `tol` cells.
    static void validate_staggering(const params_t &params, double dx,
                                    double tol = 0.05);

    //! Compare (psi, chi, h, A) against the exporter's reference table;
    //! returns (max|dpsi|, max|dchi|, max|dh|, max|dA|).
    [[nodiscard]] std::array<double, 4>
    validate(const std::string &reference_file) const;

  protected:
    params_t m_params;
    double m_dx;

    //! Barycentric weights at `xq`, normalised; exact-node hits handled.
    //! Same rule as LMInitialData::bary (kept local so the two classes stay
    //! independently header-only; the reference-table gate covers drift).
    static AMREX_GPU_HOST_DEVICE void
    bary(amrex::Real xq, const amrex::Real *nodes, const amrex::Real *w, int n,
         amrex::Real *out);

    //! One quasi-isotropic Kerr hole, hole-centred X, spin a along z:
    //! psi_QI, conformal metric gt_ij, and the At_ij seed.  Verbatim
    //! transcriptions of operators.py's _kerr_psi/_kerr_metric/_kerr_At.
    static AMREX_GPU_HOST_DEVICE amrex::Real
    kerr_psi(const amrex::Real X[3], amrex::Real m, amrex::Real a);
    static AMREX_GPU_HOST_DEVICE void
    kerr_metric(const amrex::Real X[3], amrex::Real m, amrex::Real a,
                amrex::Real gt[3][3]);
    static AMREX_GPU_HOST_DEVICE void
    kerr_At(const amrex::Real X[3], amrex::Real m, amrex::Real a,
            amrex::Real At[3][3]);
};

#include "LMCurvedInitialData.impl.hpp"

#endif /* LMCURVEDINITIALDATA_HPP_ */
