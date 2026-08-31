/* GRTeclyn
 * Copyright 2022 The GRTL collaboration.
 * Please refer to LICENSE in GRTeclyn's root directory.
 */

// Set LM-initial-data spectral puncture initial data on the grid.
//
// The data is conformally flat and maximally sliced, exactly like
// `BinaryBHInitialData`, and differs from it only in the conformal factor:
// instead of the analytic O(P^2) Bowen-York approximation
//
//     psi ~ 1 + sum_A [ m_A/2r_A + P_A^2 psi_2(r_A,cos th_A)/m_A^2 ]
//
// it carries the *solved* conformal factor
//
//     psi = psi_BL + u,   psi_BL = 1 + m_A/(2 r_A) + m_B/(2 r_B),
//
// with the correction `u` supplied spectrally by `LMSpectralData`.  The
// extrinsic curvature is the same closed-form Bowen-York tensor in both cases
// (verified against `BoostedBHInitialData::Aij`, which is the identical
// expression), so a run of this class against a run of the stock class is a
// comparison of *initial data*, with the discretisation, the variable
// conversion and the constraint operator held fixed.
//
// The evaluation is a direct port of
// `lm.initial_data.validation.export_grteclyn.eval_u` / `.eval_Ahat`, which are
// themselves gated against the solver's own evaluator.  Pass
// `lm_id_reference_file` at runtime to check the port against a table of
// reference values (see LMInitialData::validate).
//
// GOTCHA: psi_BL diverges at the punctures.  Choose the domain and n_cell so
// that no cell *centre* coincides with a puncture — the standard puncture-code
// staggering.  `validate_staggering` asserts it rather than trusting it.

#ifndef LMINITIALDATA_HPP_
#define LMINITIALDATA_HPP_

#include "Coordinates.hpp"
#include "DimensionDefinitions.hpp"
#include "StateVariables.hpp"
#include "Tensor.hpp"

#include <AMReX_Array4.H>
#include <AMReX_REAL.H>

#include <array>

//! Maximum number of nodes per spectral direction, so the per-point
//! barycentric temporaries can live on the stack (no allocation in the kernel).
constexpr int LM_MAX_NODES = 128;

class LMInitialData
{
  public:
    //! Trivially-copyable view onto data owned by LMSpectralData.
    struct params_t
    {
        double b{};
        double m_A{};
        double m_B{};
        double P_A[3]{};
        double P_B[3]{};
        double S_A[3]{};
        double S_B[3]{};
        std::array<double, AMREX_SPACEDIM> center{};
        int nA{};   //!< number of A nodes (= Na+1)
        int nB{};   //!< number of B nodes
        int nphi{};
        int ncos{};
        int nsin{};
        const amrex::Real *A{nullptr};
        const amrex::Real *B{nullptr};
        const amrex::Real *wA{nullptr};
        const amrex::Real *wB{nullptr};
        const int *cos_m{nullptr};
        const int *sin_m{nullptr};
        const amrex::Real *C{nullptr};
        const amrex::Real *S{nullptr};
        int initial_lapse{};
    };

    AMREX_FORCE_INLINE
    explicit LMInitialData(params_t a_params, double a_dx)
        : m_params(a_params), m_dx(a_dx)
    {
    }

    AMREX_GPU_DEVICE void
    operator()(int ix, int iy, int iz,
               const amrex::Array4<amrex::Real> &state) const;

    //! The spectral correction u at a point relative to the grid centre.
    [[nodiscard]] AMREX_GPU_HOST_DEVICE amrex::Real u(amrex::Real x,
                                                      amrex::Real y,
                                                      amrex::Real z) const;

    //! psi_BL = 1 + m_A/(2 r_A) + m_B/(2 r_B).
    [[nodiscard]] AMREX_GPU_HOST_DEVICE amrex::Real psi_BL(amrex::Real x,
                                                           amrex::Real y,
                                                           amrex::Real z) const;

    //! The full conformal factor psi = psi_BL + u.
    [[nodiscard]] AMREX_GPU_HOST_DEVICE amrex::Real psi(amrex::Real x,
                                                        amrex::Real y,
                                                        amrex::Real z) const;

    //! Closed-form Bowen-York Ahat^{ij} (momentum + spin, both punctures).
    [[nodiscard]] AMREX_GPU_HOST_DEVICE Tensor::Rank2
    Ahat(amrex::Real x, amrex::Real y, amrex::Real z) const;

    //! Abort unless every puncture is clear of a cell centre by `tol` cells.
    static void validate_staggering(const params_t &params, double dx,
                                    double tol = 0.05);

    //! Compare psi/Ahat against a reference table produced by
    //! `export_grteclyn.dump_reference_table`; returns (max|dpsi|, max|dAhat|).
    [[nodiscard]] std::array<double, 2>
    validate(const std::string &reference_file) const;

  protected:
    params_t m_params;
    double m_dx;

    //! Barycentric interpolation weights of one node set at `xq`, normalised so
    //! they sum to one; exact-node hits are handled by returning a unit vector.
    static AMREX_GPU_HOST_DEVICE void
    bary(amrex::Real xq, const amrex::Real *nodes, const amrex::Real *w, int n,
         amrex::Real *out);
};

#include "LMInitialData.impl.hpp"

#endif /* LMINITIALDATA_HPP_ */
