/* GRTeclyn
 * Copyright 2022 The GRTL collaboration.
 * Please refer to LICENSE in GRTeclyn's root directory.
 */

// Initialise Gamma^i = h^{jk} Gamma^i_{jk} from the h_ij already on the grid.
//
// The AMReX port of GRChombo's GammaCalculator: the in-tree
// Source/CCZ4/GammaCalculator.hpp is unported heritage code (it includes
// VarsTools.hpp, which this port does not carry, and its own comment says so),
// so the curved initial-data path carries its own ported functor, built from
// the same pieces the ported Constraints class uses (CCZ4Vars, CCZ4D1Vars,
// CCZ4Geometry, FourthOrderDerivatives).
//
// Every other BinaryBH initial-data branch has h_ij = delta_ij, so Gamma^i = 0
// and the zero-fill is exact; only the conformally curved channel needs this.
// Run it over the VALID region after the state (ghosts included) is filled:
// the derivative stencil then reads directly-evaluated ghost values, so no
// FillPatch is required at t = 0.

#ifndef LMCURVEDGAMMAINIT_HPP_
#define LMCURVEDGAMMAINIT_HPP_

#include "CCZ4D1Vars.hpp"
#include "CCZ4Geometry.hpp"
#include "CCZ4Vars.hpp"
#include "FourthOrderDerivatives.hpp"
#include "StateVariables.hpp"
#include "Tensor.hpp"

#include <AMReX_Array4.H>
#include <AMReX_REAL.H>

class LMCurvedGammaInit
{
  public:
    AMREX_FORCE_INLINE
    explicit LMCurvedGammaInit(double a_dx) : m_deriv(a_dx) {}

    AMREX_GPU_DEVICE void
    operator()(int ix, int iy, int iz,
               const amrex::Array4<amrex::Real> &state,
               const amrex::Array4<const amrex::Real> &state_c) const
    {
        // Reads only h_ij (through the stencil); writes only Gamma^i at this
        // cell — different components, so the in-place update is race-free.
        const amrex::CellData<const amrex::Real> in = state_c.cellData(ix, iy, iz);
        CCZ4Vars vars(in);
        const CCZ4D1Vars d1(ix, iy, iz, state_c, m_deriv);
        const auto h_UU  = CCZ4Geometry::compute_inverse_metric(vars);
        const auto chris = CCZ4Geometry::compute_christoffel(d1, h_UU);
        const amrex::CellData<amrex::Real> out = state.cellData(ix, iy, iz);
        out[c_Gamma1] = chris.contracted[0];
        out[c_Gamma2] = chris.contracted[1];
        out[c_Gamma3] = chris.contracted[2];
    }

  protected:
    FourthOrderDerivatives m_deriv;
};

#endif /* LMCURVEDGAMMAINIT_HPP_ */
