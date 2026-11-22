/* GRTeclyn -- see LICENSE. */
#ifndef LMHISTORYINITIALDATA_HPP_
#define LMHISTORYINITIALDATA_HPP_

#include "Coordinates.hpp"
#include "StateVariables.hpp"
#include <AMReX.H>
#include <AMReX_Array4.H>
#include "LMHistoryInterpolation.hpp"
#include <array>

// Full ADM input (format 5), with no analytic seed or K=0 restriction.
// Ownership lives in LMHistoryGridData; capture this view by value on device.
class LMHistoryInitialData
{
    static_assert(AMREX_SPACEDIM == 3, "format 5 currently requires 3D Cartesian evolution");
  public:
    AMREX_FORCE_INLINE LMHistoryInitialData(LMHistory::View view,
        std::array<double, AMREX_SPACEDIM> center, double dx)
        : m_view(view), m_center(center), m_dx(dx) {}

    AMREX_GPU_DEVICE void operator()(int ix, int iy, int iz,
        const amrex::Array4<amrex::Real> &state) const
    {
        Coordinates coords(amrex::IntVect(ix, iy, iz), m_dx, m_center);
        const double point[3] = {coords.x, coords.y, coords.z};
        double fields[LMHistory::nfields], converted[LMHistory::nstate];
        if (!LMHistory::evaluate(m_view, point, fields) ||
            !LMHistory::decompose(fields, converted))
            amrex::Abort("LM history: query outside export grid or invalid interpolated slice");
        amrex::CellData<amrex::Real> cell = state.cellData(ix, iy, iz);
        // Zero auxiliary fields, then retain physical trace K and supplied gauge.
        for (int n = 0; n < cell.nComp(); ++n) cell[n] = 0.0;
        cell[c_chi] = converted[0];
        for (int f = 0; f < 6; ++f)
        {
            // CCZ4StateVariables stores xx,xy,xz,yy,yz,zz contiguously.
            cell[c_h11+f] = converted[1+f];
            cell[c_A11+f] = converted[7+f];
        }
        cell[c_K] = converted[13];
        cell[c_lapse] = converted[14];
        for (int d = 0; d < 3; ++d) cell[c_shift1+d] = converted[15+d];
        // Gamma^i is computed from h by GammaCalculator after all ghosts fill.
    }
  private:
    LMHistory::View m_view;
    std::array<double, AMREX_SPACEDIM> m_center;
    double m_dx;
};
#endif
