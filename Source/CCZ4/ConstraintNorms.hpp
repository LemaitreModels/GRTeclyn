/* GRTeclyn
 * Copyright 2022 The GRTL collaboration.
 * Please refer to LICENSE in GRTeclyn's root directory.
 */

// Volume-weighted norms of the Hamiltonian and momentum constraints over the
// AMR hierarchy, written to a small JSON file.
//
// WHY THIS FILE EXISTS.  `Constraints` (Source/CCZ4/Constraints.hpp) is fully
// ported to AMReX and registered in the derive list, so `Ham, Mom1-3` are
// available as derived quantities with the fourth-order stencils.  What was
// *not* ported from GRChombo is `AMRReductions`, which is why the norm block in
// `BinaryBHLevel::specificPostTimeStep` sits inside `#if 0`.  This provides the
// missing reduction using AMReX's own primitives.
//
// CONVENTIONS (chosen to be comparable with an external monitor):
//   * the reported "L2" is the VOLUME-NORMALISED root-mean-square,
//         L2(f) = sqrt( sum_cells |f|^2 dV / sum_cells dV ),
//     i.e. GRChombo's `AMRReductions::norm(..., normalize_by_volume = true)`.
//     It is an RMS, not an integral, so it does not grow with the box size.
//   * the momentum norm is applied to the flat magnitude
//         |M| = sqrt(M1^2 + M2^2 + M3^2).
//   * covered cells (those refined by a finer level) are excluded via
//     `makeFineMask`, so the total is a leaf-cell norm and each cell is counted
//     exactly once.  Per-level numbers are reported as well: for a uniform
//     `max_level = 0` run there is a single level and the two coincide.
//   * cells within `r_excl` of any puncture are excluded (psi_BL diverges
//     there, so |H| is a large cancellation), and `border_cells` cells are
//     dropped from the coarse-level domain edge (the fourth-order stencil
//     reaches into ghost cells filled from the outer boundary condition).
//     Both mirror the mask of the external monitor being compared against.

#ifndef CONSTRAINTNORMS_HPP_
#define CONSTRAINTNORMS_HPP_

// GRTeclyn includes
#include "Constraints.hpp"

// AMReX includes
#include <AMReX_AmrLevel.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParReduce.H>
#include <AMReX_REAL.H>

// System includes
#include <array>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace ConstraintNorms
{

//! Norms on one refinement level (leaf cells only).
struct LevelNorms
{
    int level{};
    double dx{};
    double volume{};   //!< total volume of the counted cells
    long n_cells{};    //!< number of counted cells
    double rms_Ham{};
    double max_Ham{};
    double rms_Mom{};
    double max_Mom{};
};

//! Norms over the whole hierarchy plus the per-level breakdown.
struct Result
{
    std::vector<LevelNorms> levels;
    double volume{};
    long n_cells{};
    double rms_Ham{};
    double max_Ham{};
    double rms_Mom{};
    double max_Mom{};
};

//! Options selecting which cells enter the norms.
struct Options
{
    double r_excl{0.0}; //!< exclude cells with r < r_excl of any puncture
    int border_cells{0};  //!< drop this many cells from the level-0 domain edge
    std::vector<std::array<double, AMREX_SPACEDIM>> punctures{};
};

/// Reduce one level's derived constraint MultiFab.
/// `mask` is 1.0 on cells to count and 0.0 on covered cells.
inline LevelNorms reduce_level(const amrex::MultiFab &constraints,
                               const amrex::MultiFab &mask,
                               const amrex::Geometry &geom,
                               const amrex::Box &interior, int level,
                               const Options &opts)
{
    const auto dx_arr = geom.CellSizeArray();
    const auto problo = geom.ProbLoArray();
    const amrex::Real dV =
        AMREX_D_TERM(dx_arr[0], *dx_arr[1], *dx_arr[2]); // cell volume

    // Puncture centres into a plain device-copyable buffer.
    const int n_punc = static_cast<int>(opts.punctures.size());
    amrex::Gpu::DeviceVector<amrex::Real> d_punc(
        static_cast<std::size_t>(AMREX_SPACEDIM * n_punc));
    {
        amrex::Vector<amrex::Real> h_punc(
            static_cast<std::size_t>(AMREX_SPACEDIM * n_punc));
        for (int p = 0; p < n_punc; ++p)
        {
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
            {
                h_punc[static_cast<std::size_t>(AMREX_SPACEDIM * p + d)] =
                    opts.punctures[static_cast<std::size_t>(p)][d];
            }
        }
        if (n_punc > 0)
        {
            amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_punc.begin(),
                                  h_punc.end(), d_punc.begin());
            amrex::Gpu::streamSynchronize();
        }
    }
    const amrex::Real *punc_ptr    = d_punc.dataPtr();
    const amrex::Real r_excl_sq    = opts.r_excl * opts.r_excl;

    amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum,
                     amrex::ReduceOpSum, amrex::ReduceOpMax, amrex::ReduceOpMax>
        reduce_op;
    amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real, amrex::Real,
                      amrex::Real, amrex::Real>
        reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;

    for (amrex::MFIter mfi(constraints); mfi.isValid(); ++mfi)
    {
        const amrex::Box bx = mfi.validbox() & interior;
        if (!bx.ok())
        {
            continue;
        }
        const auto &c = constraints.const_array(mfi);
        const auto &m = mask.const_array(mfi);
        reduce_op.eval(bx, reduce_data,
                       [=] AMREX_GPU_DEVICE(int i, int j, int k) -> ReduceTuple
                       {
                           amrex::Real keep = m(i, j, k);
                           if (r_excl_sq > 0.0)
                           {
                               const amrex::Real x =
                                   problo[0] + (i + 0.5) * dx_arr[0];
                               const amrex::Real y =
                                   problo[1] + (j + 0.5) * dx_arr[1];
                               const amrex::Real z =
                                   problo[2] + (k + 0.5) * dx_arr[2];
                               for (int p = 0; p < n_punc; ++p)
                               {
                                   const amrex::Real ddx =
                                       x - punc_ptr[AMREX_SPACEDIM * p + 0];
                                   const amrex::Real ddy =
                                       y - punc_ptr[AMREX_SPACEDIM * p + 1];
                                   const amrex::Real ddz =
                                       z - punc_ptr[AMREX_SPACEDIM * p + 2];
                                   if (ddx * ddx + ddy * ddy + ddz * ddz <
                                       r_excl_sq)
                                   {
                                       keep = 0.0;
                                   }
                               }
                           }
                           const amrex::Real ham = c(i, j, k, 0);
                           const amrex::Real mom_sq =
                               c(i, j, k, 1) * c(i, j, k, 1) +
                               c(i, j, k, 2) * c(i, j, k, 2) +
                               c(i, j, k, 3) * c(i, j, k, 3);
                           // (weight, count, sum H^2, sum |M|^2, max|H|,
                           //  max|M|); excluded cells contribute 0, which is
                           //  safe for the maxima since both are >= 0.
                           return {keep * dV,
                                   keep,
                                   keep * dV * ham * ham,
                                   keep * dV * mom_sq,
                                   keep * std::abs(ham),
                                   keep * std::sqrt(mom_sq)};
                       });
    }

    ReduceTuple hv     = reduce_data.value(reduce_op);
    double sum_w       = amrex::get<0>(hv);
    double sum_n       = amrex::get<1>(hv);
    double sum_ham_sq  = amrex::get<2>(hv);
    double sum_mom_sq  = amrex::get<3>(hv);
    double max_ham     = amrex::get<4>(hv);
    double max_mom     = amrex::get<5>(hv);

    amrex::ParallelDescriptor::ReduceRealSum(sum_w);
    amrex::ParallelDescriptor::ReduceRealSum(sum_n);
    amrex::ParallelDescriptor::ReduceRealSum(sum_ham_sq);
    amrex::ParallelDescriptor::ReduceRealSum(sum_mom_sq);
    amrex::ParallelDescriptor::ReduceRealMax(max_ham);
    amrex::ParallelDescriptor::ReduceRealMax(max_mom);

    LevelNorms out;
    out.level   = level;
    out.dx      = dx_arr[0];
    out.volume  = sum_w;
    out.n_cells = static_cast<long>(std::llround(sum_n));
    out.rms_Ham = (sum_w > 0.0) ? std::sqrt(sum_ham_sq / sum_w) : 0.0;
    out.rms_Mom = (sum_w > 0.0) ? std::sqrt(sum_mom_sq / sum_w) : 0.0;
    out.max_Ham = max_ham;
    out.max_Mom = max_mom;
    return out;
}

/// Compute the constraint norms over the whole hierarchy at time `time`.
/// `gramr` must be the (GR)AMR object; `AMR` is templated only to avoid a
/// header dependency on GRAMR from Source/CCZ4.
template <class AMR>
Result compute(AMR &gramr, amrex::Real time, const Options &opts)
{
    Result res;
    // `derive` calls FillPatch internally, so ghost cells are filled; we
    // reduce over valid cells only and therefore ask for none.
    auto mfs        = gramr.derive(Constraints::name, time, 0);
    const int finest = gramr.finestLevel();

    double sum_w = 0.0, sum_n = 0.0, sum_h2 = 0.0, sum_m2 = 0.0;
    double max_h = 0.0, max_m = 0.0;

    for (int lev = 0; lev <= finest; ++lev)
    {
        const amrex::MultiFab &cmf = *mfs[lev];
        const amrex::Geometry &geom =
            gramr.getLevel(lev).Geom();

        // Interior box: shrink the level-0 domain by border_cells, then
        // refine to this level, so the same physical shell is dropped on
        // every level.
        amrex::Box interior = gramr.getLevel(0).Domain();
        if (opts.border_cells > 0)
        {
            interior.grow(-opts.border_cells);
        }
        interior.refine(gramr.getLevel(lev).Domain().length(0) /
                        gramr.getLevel(0).Domain().length(0));

        // Leaf mask: 1 where this level is the finest covering the cell.
        amrex::MultiFab mask(cmf.boxArray(), cmf.DistributionMap(), 1, 0);
        if (lev < finest)
        {
            const amrex::BoxArray &fba = gramr.getLevel(lev + 1).boxArray();
            const amrex::IntVect ratio(
                gramr.getLevel(lev + 1).Domain().length(0) /
                gramr.getLevel(lev).Domain().length(0));
            mask = amrex::makeFineMask(cmf.boxArray(), cmf.DistributionMap(),
                                       fba, ratio, 1.0 /*crse*/, 0.0 /*fine*/);
        }
        else
        {
            mask.setVal(1.0);
        }

        LevelNorms ln = reduce_level(cmf, mask, geom, interior, lev, opts);
        res.levels.push_back(ln);

        sum_w += ln.volume;
        sum_n += static_cast<double>(ln.n_cells);
        sum_h2 += ln.rms_Ham * ln.rms_Ham * ln.volume;
        sum_m2 += ln.rms_Mom * ln.rms_Mom * ln.volume;
        max_h = std::max(max_h, ln.max_Ham);
        max_m = std::max(max_m, ln.max_Mom);
    }

    res.volume  = sum_w;
    res.n_cells = static_cast<long>(std::llround(sum_n));
    res.rms_Ham = (sum_w > 0.0) ? std::sqrt(sum_h2 / sum_w) : 0.0;
    res.rms_Mom = (sum_w > 0.0) ? std::sqrt(sum_m2 / sum_w) : 0.0;
    res.max_Ham = max_h;
    res.max_Mom = max_m;
    return res;
}

/// JSON has no NaN/Inf literal, and a bare `nan` in the output silently breaks
/// every downstream parser (Python's `json` accepts `NaN` but not `-nan`).  Emit
/// `null` instead, so a non-finite norm is unmistakable rather than fatal.
inline std::string jnum(double v)
{
    if (!std::isfinite(v))
    {
        return "null";
    }
    std::ostringstream s;
    s << std::setprecision(17) << std::scientific << v;
    return s.str();
}

/// Write the result as JSON (IO processor only).  `extra` is inserted verbatim
/// as additional top-level members (caller supplies `"key": value,` pairs).
inline void write_json(const Result &res, const std::string &filename,
                       amrex::Real time, const Options &opts,
                       const std::string &extra = "")
{
    if (!amrex::ParallelDescriptor::IOProcessor())
    {
        return;
    }
    std::ofstream f(filename);
    f << std::setprecision(17) << std::scientific;
    f << "{\n";
    if (!extra.empty())
    {
        f << "  " << extra << "\n";
    }
    f << "  \"time\": " << jnum(time) << ",\n";
    f << "  \"r_excl\": " << jnum(opts.r_excl) << ",\n";
    f << "  \"border_cells\": " << opts.border_cells << ",\n";
    f << "  \"norm\": \"volume-normalised rms over leaf cells\",\n";
    f << "  \"n_cells\": " << res.n_cells << ",\n";
    f << "  \"volume\": " << jnum(res.volume) << ",\n";
    f << "  \"L2_Ham\": " << jnum(res.rms_Ham) << ",\n";
    f << "  \"L2_Mom\": " << jnum(res.rms_Mom) << ",\n";
    f << "  \"Linf_Ham\": " << jnum(res.max_Ham) << ",\n";
    f << "  \"Linf_Mom\": " << jnum(res.max_Mom) << ",\n";
    f << "  \"levels\": [\n";
    for (std::size_t i = 0; i < res.levels.size(); ++i)
    {
        const LevelNorms &l = res.levels[i];
        f << "    {\"level\": " << l.level << ", \"dx\": " << jnum(l.dx)
          << ", \"n_cells\": " << l.n_cells << ", \"volume\": "
          << jnum(l.volume) << ", \"L2_Ham\": " << jnum(l.rms_Ham)
          << ", \"L2_Mom\": " << jnum(l.rms_Mom) << ", \"Linf_Ham\": "
          << jnum(l.max_Ham) << ", \"Linf_Mom\": " << jnum(l.max_Mom)
          << "}" << (i + 1 < res.levels.size() ? "," : "") << "\n";
    }
    f << "  ]\n}\n";
    f.close();
}

} // namespace ConstraintNorms

#endif /* CONSTRAINTNORMS_HPP_ */
