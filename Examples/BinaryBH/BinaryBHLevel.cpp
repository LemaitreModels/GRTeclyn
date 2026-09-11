/* GRTeclyn
 * Copyright 2022 The GRTL collaboration.
 * Please refer to LICENSE in GRTeclyn's root directory.
 */

#include "BinaryBHLevel.hpp"

#include "AlgebraicConstraintsEnforcer.hpp"
#include "BinaryBHInitialData.hpp"
#include "CCZ4RHS.hpp"
#include "ChiTagger.hpp"
#include "ConstraintNorms.hpp"
#include "Constraints.hpp"
#include "ExtractionTagger.hpp"
#include "GammaCalculator.hpp"
#include "FourthOrderDerivatives.hpp"
#include "LMCurvedInitialData.hpp"
#include "LMCurvedSpectralData.hpp"
#include "LMInitialData.hpp"
#include "LMSpectralData.hpp"
#include "PositiveChiAndLapse.hpp"
#include "PunctureTagger.hpp"
#include "PunctureTracker.hpp"
#include "SixthOrderDerivatives.hpp"
#include "TwoPuncturesInitialData.hpp"
#include "Weyl4.hpp"
#include "WeylExtraction.hpp"

BHAmr<BinaryBHLevel::num_punctures> *BinaryBHLevel::get_bh_amr_ptr()
{
    return dynamic_cast<BHAmr<num_punctures> *>(get_gr_amr_ptr());
}

PunctureTracker<BinaryBHLevel::num_punctures> &
BinaryBHLevel::get_puncture_tracker()
{
    return get_bh_amr_ptr()->get_puncture_tracker();
}

void BinaryBHLevel::variableSetUp()
{
    BL_PROFILE("BinaryBHLevel::variableSetUp()");

    // Set up the state variables
    state_variable_set_up();

    Constraints::set_up(state_index);

    Weyl4::set_up(state_index);
}

// Things to do during the advance step after RK4 steps
void BinaryBHLevel::specific_advance()
{
    amrex::MultiFab &state_new = get_new_data(state_index);
    const auto &state_arrays   = state_new.arrays();

    // The classes to be used
    AlgebraicConstraintsEnforcer algebraic_constraints_enforcer;
    PositiveChiAndLapse positive_chi_lapse;

    // Enforce det(h)=1, the trace free A_ij condition and positive chi and
    // lapse
    amrex::ParallelFor(state_new,
                       [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
                       {
                           algebraic_constraints_enforcer(ix, iy, iz,
                                                          state_arrays[box_no]);
                           positive_chi_lapse(ix, iy, iz, state_arrays[box_no]);
                       });
}

// This initial data uses an approximation for the metric which
// is valid for small boosts
void BinaryBHLevel::initData()
{
    BL_PROFILE("BinaryBHLevel::initData");
    if (get_gr_amr_ptr()->Verbose() > 0)
    {
        amrex::Print() << "BinaryBHLevel::initData " << Level() << "\n";
    }
#ifdef USE_TWOPUNCTURES
    TwoPuncturesInitialData two_punctures_initial_data(Geom().CellSize(0));

    two_punctures_initial_data.solve(); // only solves first time

    amrex::MultiFab &state_new = get_new_data(state_index);
#ifdef AMREX_USE_GPU
    amrex::MFInfo mf_info;
    mf_info.SetArena(amrex::The_Cpu_Arena());
    amrex::MultiFab host_state(state_new.boxArray(),
                               state_new.DistributionMap(), state_new.nComp(),
                               state_new.nGrowVect(), mf_info);
#else
    amrex::MultiFab &host_state = state_new;
#endif

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(state_new, amrex::TilingIfNotGPU()); mfi.isValid();
         ++mfi)
    {
        const amrex::Box &grown_tile_box = mfi.growntilebox();
        const auto &state_array          = host_state.array(mfi);

        amrex::LoopOnCpu(
            grown_tile_box, [=](int ix, int iy, int iz)
            { two_punctures_initial_data(ix, iy, iz, state_array); });
#ifdef AMREX_USE_GPU
        // Copy to device
        amrex::Gpu::htod_memcpy_async(
            state_new[mfi].dataPtr(), host_state[mfi].dataPtr(),
            host_state[mfi].size() * sizeof(amrex::Real));
#endif
    }

#else
    amrex::Real dx = Geom().CellSize(0);
    // First set everything to zero (to avoid undefinded values in constraints)
    // then calculate initial data
    amrex::MultiFab &state_new = get_new_data(state_index);
    const auto &state_arrays   = state_new.arrays();

    // LM-initial-data parameters.  Upstream PR #215 (refactor_params2) removed
    // SimulationParameters' data members, so these are read here, at the point
    // of use, from a prefixed table — the pattern upstream itself now uses (cf.
    // GRParmParse("puncture_tracking") below).  Keys are "lm.*" so they cannot
    // collide with an upstream key now or later.
    GRParmParse lm_pp("lm");
    std::string lm_id_file;
    std::string lm_curved_id_file;
    lm_pp.queryAdd("id_file", lm_id_file);
    lm_pp.queryAdd("curved_id_file", lm_curved_id_file);

    // Domain centre, hoisted: BOTH LM channels need it, and the flat branch
    // used to compute it inside its own scope.  BaseParameterChecker::
    // check_params() computes the default -- including the shift for
    // reflective boundaries -- and queryAdd()s it into the "geometry" table at
    // startup, so the key is present by the time initial data is built.  The
    // loop is only a fallback for a caller that bypassed check_params.
    std::array<amrex::Real, AMREX_SPACEDIM> lm_center{};
    for (int i = 0; i < AMREX_SPACEDIM; ++i)
    {
        lm_center[i] = 0.5 * (Geom().ProbLo(i) + Geom().ProbHi(i));
    }
    GRParmParse geom_pp("geometry");
    geom_pp.queryAdd("center", lm_center);

    // Curved channel FIRST, and mutually exclusive with the flat one.
    if (!lm_curved_id_file.empty())
    {
        // LM-initial-data CONFORMALLY CURVED spectral initial data (format 3,
        // spinning-at-rest sector).  Runtime-selected like the flat channel
        // below, and mutually exclusive with it.
        if (!lm_id_file.empty())
        {
            amrex::Abort("BinaryBHLevel: lm.id_file and lm.curved_id_file are "
                         "mutually exclusive — pick one initial-data channel");
        }

        // Companion keys, read at point of use like the flat channel above.
        std::string lm_curved_id_reference_file;
        amrex::Real lm_curved_id_reference_tol = 1.0e-10;
        lm_pp.queryAdd("curved_id_reference_file", lm_curved_id_reference_file);
        lm_pp.queryAdd("curved_id_reference_tol", lm_curved_id_reference_tol);

        static const LMCurvedSpectralData lm_curved_data(
            lm_curved_id_file);
        LMCurvedInitialData::params_t lm_params =
            lm_curved_data.params(lm_center, Lapse::PRE_COLLAPSED);
        LMCurvedInitialData lm_initial_data(lm_params, dx);

        static_assert(std::is_trivially_copyable_v<LMCurvedInitialData>,
                      "LMCurvedInitialData needs to be device copyable");

        // psi_QI diverges at the punctures: refuse a grid that samples one.
        LMCurvedInitialData::validate_staggering(lm_params, dx);

        if (Level() == 0)
        {
            for (const auto &line : lm_curved_data.provenance())
            {
                amrex::Print() << "LM-curved-ID:" << line << "\n";
            }
            if (!lm_curved_id_reference_file.empty())
            {
                auto err = lm_initial_data.validate(
                    lm_curved_id_reference_file);
                const double tol = lm_curved_id_reference_tol;
                if (err[0] > tol || err[1] > tol || err[2] > tol ||
                    err[3] > tol)
                {
                    amrex::Abort(
                        "LMCurvedInitialData: the C++ reconstruction disagrees "
                        "with the exported reference table beyond "
                        "lm_curved_id_reference_tol");
                }
            }
        }

        amrex::ParallelFor(
            state_new, state_new.nGrowVect(),
            [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
            {
                amrex::CellData<amrex::Real> cell =
                    state_arrays[box_no].cellData(ix, iy, iz);
                for (int n = 0; n < cell.nComp(); ++n)
                {
                    cell[n] = 0.;
                }
                lm_initial_data(ix, iy, iz, state_arrays[box_no]);
            });

        // Gamma^i of the curved conformal metric.  Every other branch has
        // h_ij = delta so Gamma^i = 0 and the zero-fill above is exact; here
        // it is not.  This used LMCurvedGammaInit, a local port written because
        // the in-tree GammaCalculator was unported heritage code at the old
        // pin.  Upstream has since ported it (Source/CCZ4/GammaCalculator.hpp),
        // computing the identical quantity -- h_UU from CCZ4Vars, d1_h via
        // d1_sym_tensor(..., c_h11), contracted Christoffel into c_Gamma1+i --
        // and templated on the derivative order, so unlike the local copy it
        // also works at 6th order.  The local port is therefore deleted rather
        // than re-ported.  The ParallelFor above filled the ghost cells by
        // direct evaluation (nGrowVect), so the derivative stencils are valid
        // without a FillPatch; iterate the VALID region only -- the write
        // touches only the Gamma components, so the in-place update is
        // race-free.
        amrex::Gpu::streamSynchronize();
        GammaCalculator<> gamma_init(dx);
        static_assert(std::is_trivially_copyable_v<GammaCalculator<>>,
                      "GammaCalculator needs to be device copyable");
        amrex::ParallelFor(
            state_new,
            [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
            { gamma_init(ix, iy, iz, state_arrays[box_no]); });
    }
    else if (!lm_id_file.empty())
    {
        // Deliberately `get`, not `queryAdd`: once lm.id_file is set the run is
        // an LM-ID run, and a mistyped companion key must abort rather than
        // silently take a default.  Silent defaulting is exactly how the
        // parameter rename in PR #215 bites (see GRTECLYN_UPDATE_BRIEF.md).
        std::string lm_id_reference_file;
        amrex::Real lm_id_reference_tol = 1.0e-10;
        lm_pp.queryAdd("id_reference_file", lm_id_reference_file);
        lm_pp.queryAdd("id_reference_tol", lm_id_reference_tol);

        // LM-initial-data spectral initial data.  Runtime-selected, so this is
        // the SAME executable, stencils and variable conversion as the analytic
        // branch below — the comparison is then of initial data alone.
        // Read once per process (the file is read-only static data used by
        // every level).
        static const LMSpectralData lm_data(lm_id_file);
        LMInitialData::params_t lm_params =
            lm_data.params(lm_center, Lapse::PRE_COLLAPSED);
        LMInitialData lm_initial_data(lm_params, dx);

        static_assert(std::is_trivially_copyable_v<LMInitialData>,
                      "LMInitialData needs to be device copyable");

        // psi_BL diverges at the punctures: refuse a grid that samples one.
        LMInitialData::validate_staggering(lm_params, dx);

        if (Level() == 0)
        {
            for (const auto &line : lm_data.provenance())
            {
                amrex::Print() << "LM-ID:" << line << "\n";
            }
            if (!lm_id_reference_file.empty())
            {
                auto err = lm_initial_data.validate(lm_id_reference_file);
                if (err[0] > lm_id_reference_tol ||
                    err[1] > lm_id_reference_tol)
                {
                    amrex::Abort(
                        "LMInitialData: the C++ evaluation disagrees with the "
                        "exported reference table beyond lm_id_reference_tol");
                }
            }
        }

        amrex::ParallelFor(
            state_new, state_new.nGrowVect(),
            [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
            {
                amrex::CellData<amrex::Real> cell =
                    state_arrays[box_no].cellData(ix, iy, iz);
                for (int n = 0; n < cell.nComp(); ++n)
                {
                    cell[n] = 0.;
                }
                lm_initial_data(ix, iy, iz, state_arrays[box_no]);
            });
    }
    else
    {
        // Set up the compute class for the BinaryBH initial data.  Upstream
        // PR #215: it reads bh1.*/bh2.* itself, so no params are threaded in.
        BinaryBHInitialData binary_initial_data(dx);

        static_assert(std::is_trivially_copyable_v<BinaryBHInitialData>,
                      "BinaryBHInitialData needs to be device copyable");

        amrex::ParallelFor(
            state_new, state_new.nGrowVect(),
            [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
            {
                amrex::CellData<amrex::Real> cell =
                    state_arrays[box_no].cellData(ix, iy, iz);
                for (int n = 0; n < cell.nComp(); ++n)
                {
                    cell[n] = 0.;
                }
                binary_initial_data(ix, iy, iz, state_arrays[box_no]);
            });
    }
#endif
    amrex::Gpu::streamSynchronize();

    if (get_bh_amr_ptr()->puncture_tracking_enabled() && Level() == 0)
    {
        // need to set the puncture coordinates as we use it for the puncture
        // tagging
        BoostedBHInitialData::params_t bh1_params(1);
        BoostedBHInitialData::params_t bh2_params(2);
#ifdef USE_TWOPUNCTURES
        two_punctures_initial_data.set_bh_params(bh1_params, bh2_params);
#else
        bh1_params.fill_params();
        bh2_params.fill_params();
#endif

        get_puncture_tracker().set_puncture_coords(
            {bh1_params.center[0], bh1_params.center[1], bh1_params.center[2],
             bh2_params.center[0], bh2_params.center[1], bh2_params.center[2]});
        // can't call start_from_initial_punctures() because we need the full
        // AMR grid first
    }
}

// Calculate RHS during RK4 substeps
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void BinaryBHLevel::specific_eval_rhs(amrex::MultiFab &a_soln,
                                      amrex::MultiFab &a_rhs,
                                      const amrex::Real /*a_time*/)
{
    BL_PROFILE("BinaryBHLevel::specific_eval_rhs()");
    const auto &soln_arrays       = a_soln.arrays();
    const auto &const_soln_arrays = a_soln.const_arrays();
    const auto &rhs_arrays        = a_rhs.arrays();
    const auto soln_ghosts        = a_soln.nGrowVect();

    // The classes to be used
    AlgebraicConstraintsEnforcer algebraic_constraints_enforcer;
    PositiveChiAndLapse positive_chi_lapse;

    // Enforce positive chi and lapse, det(h)=1 and trace free A
    amrex::ParallelFor(a_soln, soln_ghosts,
                       [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
                       {
                           algebraic_constraints_enforcer(ix, iy, iz,
                                                          soln_arrays[box_no]);
                           positive_chi_lapse(ix, iy, iz, soln_arrays[box_no]);
                       });

    // Calculate CCZ4 right hand side using dynamic derivative order
    if (m_evolution_spatial_derivative_order == 4)
    {
        CCZ4RHS<FourthOrderDerivatives> ccz4rhs(Geom().CellSize(0));
        MovingPunctureGauge<FourthOrderDerivatives> moving_puncture_gauge(
            Geom().CellSize(0));

        // NB: These are split up to avoid having to pre-compute all the
        //  first and second derivatives in memory on the GPU at once.

        amrex::ParallelFor(
            a_rhs,
            [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
            {
                ccz4rhs.compute_chi_and_h_ij(ix, iy, iz, rhs_arrays[box_no],
                                             const_soln_arrays[box_no]);
            });

        amrex::ParallelFor(
            a_rhs,
            [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
            {
                ccz4rhs.compute_A_ij_and_Theta_and_Gamma(
                    ix, iy, iz, rhs_arrays[box_no], const_soln_arrays[box_no]);
            });

        amrex::ParallelFor(
            a_rhs,
            [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
            {
                moving_puncture_gauge.calculate_rhs(
                    ix, iy, iz, rhs_arrays[box_no], const_soln_arrays[box_no]);

                ccz4rhs.apply_dissipation(ix, iy, iz, rhs_arrays[box_no],
                                          const_soln_arrays[box_no]);
            });
    }
    else if (m_evolution_spatial_derivative_order == 6)
    {
        CCZ4RHS<SixthOrderDerivatives> ccz4rhs(Geom().CellSize(0));
        MovingPunctureGauge<SixthOrderDerivatives> moving_puncture_gauge(
            Geom().CellSize(0));

        // NB: These are split up to avoid having to pre-compute all the
        //  first and second derivatives in memory on the GPU at once.

        amrex::ParallelFor(
            a_rhs,
            [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
            {
                ccz4rhs.compute_chi_and_h_ij(ix, iy, iz, rhs_arrays[box_no],
                                             const_soln_arrays[box_no]);
            });

        amrex::ParallelFor(
            a_rhs,
            [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
            {
                ccz4rhs.compute_A_ij_and_Theta_and_Gamma(
                    ix, iy, iz, rhs_arrays[box_no], const_soln_arrays[box_no]);
            });

        amrex::ParallelFor(
            a_rhs,
            [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
            {
                moving_puncture_gauge.calculate_rhs(
                    ix, iy, iz, rhs_arrays[box_no], const_soln_arrays[box_no]);

                ccz4rhs.apply_dissipation(ix, iy, iz, rhs_arrays[box_no],
                                          const_soln_arrays[box_no]);
            });
    }

    amrex::Gpu::streamSynchronize();
}

// enforce algebraic constraints during RK4 substeps
void BinaryBHLevel::specific_update_ode(amrex::MultiFab &a_soln)
{

    AlgebraicConstraintsEnforcer algebraic_constraints_enforcer;
    const auto soln_ghosts = amrex::IntVect(0); // zero ghost cells

    // Enforce the det(h)=1 and trace free A_ij conditions
    const auto &soln_arrays = a_soln.arrays();
    amrex::ParallelFor(
        a_soln, soln_ghosts,
        [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
        { algebraic_constraints_enforcer(ix, iy, iz, soln_arrays[box_no]); });

    amrex::Gpu::streamSynchronize();
}

void BinaryBHLevel::pre_tag_cells()
{
    amrex::MultiFab &state_new = get_new_data(state_index);
    const auto current_time    = get_state_data(state_index).curTime();

    // Fill ghosts for chi to calculate second derivatives
    // 4th-order d2 requires 2 ghost cells
    const int num_ghosts = 2;
    const int num_comps  = 1;

    FillPatch(*this, state_new, num_ghosts, current_time, state_index, c_chi,
              num_comps);
}

void BinaryBHLevel::tag_cells(amrex::TagBoxArray &a_tag_box_array,
                              amrex::Real a_regrid_threshold)
{
    BL_PROFILE("BinaryBHLevel::tag_cells()");
    amrex::MultiFab &state_new = get_new_data(state_index);

    const auto &tag_arrays         = a_tag_box_array.arrays();
    const auto &state_const_arrays = state_new.const_arrays();

    ChiTagger chi_tagger(Geom().CellSize(0), a_regrid_threshold);

    GRParmParse pp;
    spherical_extraction_params_t extraction_params("weyl_extraction");
    extraction_params.fill_params();
    ExtractionTagger extraction_tagger(Geom().CellSize(0), Level(),
                                       extraction_params);

    constexpr auto num_puncture_coords =
        static_cast<std::size_t>(AMREX_SPACEDIM * num_punctures);
    std::array<amrex::Real, num_puncture_coords> puncture_coords{};
    const bool puncture_tracking_enabled =
        get_bh_amr_ptr()->puncture_tracking_enabled();

    if (puncture_tracking_enabled)
    {
        puncture_coords = get_puncture_tracker().get_puncture_coords();
    }

    amrex::Real bh1_mass{};
    amrex::Real bh2_mass{};
    pp.get("bh1.mass", bh1_mass);
    pp.get("bh2.mass", bh2_mass);

    PunctureTagger<num_punctures> puncture_tagger(
        Geom().CellSize(0), Level(), get_gr_amr_ptr()->maxLevel(),
        puncture_coords, {bh1_mass, bh2_mass});

    amrex::ParallelFor(state_new, amrex::IntVect(0),
                       [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
                       {
                           chi_tagger(ix, iy, iz, tag_arrays[box_no],
                                      state_const_arrays[box_no]);

                           extraction_tagger(ix, iy, iz, tag_arrays[box_no]);

                           if (puncture_tracking_enabled)
                           {
                               puncture_tagger(ix, iy, iz, tag_arrays[box_no]);
                           }
                       });

    amrex::Gpu::streamSynchronize();
}

void BinaryBHLevel::specific_post_init()
{
    BL_PROFILE("BinaryBHLevel::specific_post_init()");

    if (get_bh_amr_ptr()->puncture_tracking_enabled() && Level() == 0)
    {
        get_puncture_tracker().start_from_initial_punctures();
    }

    // Constraint norms of the INITIAL DATA (t = 0).  post_init cascades from
    // the finest level down, so at Level() == 0 the whole hierarchy exists and
    // the hierarchy-wide `derive` below is safe.  Measuring here rather than in
    // specificPostTimeStep means the run needs no evolution step at all.
    // (The norm block in specificPostTimeStep is still #if 0'd: it uses
    // GRChombo's AMRReductions, which was never ported to AMReX.)
    GRParmParse lm_norm_pp("lm");
    bool calculate_constraint_norms = false;
    lm_norm_pp.queryAdd("calculate_constraint_norms",
                        calculate_constraint_norms);

    if (calculate_constraint_norms && Level() == 0)
    {
        ConstraintNorms::Options opts;
        lm_norm_pp.queryAdd("constraint_norm_exclusion_radius", opts.r_excl);
        lm_norm_pp.queryAdd("constraint_norm_border_cells", opts.border_cells);

        // Puncture centres.  Same construction upstream uses for the puncture
        // tracker: params_t reads bh<N>.* itself via fill_params().
        BoostedBHInitialData::params_t bh1_params(1);
        BoostedBHInitialData::params_t bh2_params(2);
#ifdef USE_TWOPUNCTURES
        two_punctures_initial_data.set_bh_params(bh1_params, bh2_params);
#else
        bh1_params.fill_params();
        bh2_params.fill_params();
#endif
        opts.punctures.push_back(bh1_params.center);
        opts.punctures.push_back(bh2_params.center);

        const amrex::Real time = get_state_data(state_index).curTime();
        auto res = ConstraintNorms::compute(*get_gr_amr_ptr(), time, opts);

        amrex::Print() << "Constraint norms at t = " << time
                       << " (volume-normalised rms over leaf cells, "
                       << res.n_cells << " cells)\n"
                       << "  L2_Ham   = " << res.rms_Ham << "\n"
                       << "  L2_Mom   = " << res.rms_Mom << "\n"
                       << "  Linf_Ham = " << res.max_Ham << "\n"
                       << "  Linf_Mom = " << res.max_Mom << "\n";

        std::ostringstream extra;
        extra << std::setprecision(17);
        extra << "\"n_cell_level0\": "
              << get_gr_amr_ptr()->getLevel(0).Domain().length(0) << ",";
        // NOTE: output location changed with the port.  The old
        // simParams().data_path is gone; this reads "lm.data_path", default
        // empty (i.e. the run directory), which is where every campaign job
        // already looks.
        std::string lm_data_path;
        lm_norm_pp.queryAdd("data_path", lm_data_path);
        ConstraintNorms::write_json(res, lm_data_path + "constraint_norms.json",
                                    time, opts, extra.str());
    }
}

void BinaryBHLevel::specific_post_restart()
{
    BL_PROFILE("BinaryBHLevel::specific_post_restart()");

    if (get_bh_amr_ptr()->puncture_tracking_enabled() && Level() == 0)
    {
        std::string restart_checkpoint{};
        GRParmParse pp("amr");
        pp.get("restart", restart_checkpoint);
        get_puncture_tracker().restart(restart_checkpoint);
    }
}

void BinaryBHLevel::specific_post_plotfile(const std::string &a_dir,
                                           std::ostream &a_os)
{
    if (get_bh_amr_ptr()->puncture_tracking_enabled() && Level() == 0)
    {
        get_puncture_tracker().write_plotfile(a_dir);
    }
}

void BinaryBHLevel::specific_post_checkpoint(const std::string &a_chk_dir,
                                             std::ostream & /*a_os*/)
{
    if (get_bh_amr_ptr()->puncture_tracking_enabled() && Level() == 0)
    {
        get_puncture_tracker().checkpoint(a_chk_dir);
    }
}

void BinaryBHLevel::specific_post_timestep()
{
    BL_PROFILE("BinaryBHLevel::specific_post_timestep");

    if (get_bh_amr_ptr()->puncture_tracking_enabled())
    {
        GRParmParse puncture_tracking_pp("puncture_tracking");
        int puncture_tracking_level{};
        puncture_tracking_pp.get("level", puncture_tracking_level);
        int puncture_tracking_writeout_level{};
        puncture_tracking_pp.get("writeout_level",
                                 puncture_tracking_writeout_level);

        // do puncture tracking on requested level
        if (Level() == puncture_tracking_level)
        {
            BL_PROFILE("PunctureTracking");

            // only do the write out when we're at at a multiple of the
            // writeout_level
            bool write_punctures =
                at_level_timestep_multiple(puncture_tracking_writeout_level);
            amrex::Real current_time = get_state_data(state_index).curTime();
            amrex::Real dt           = get_gr_amr_ptr()->dtLevel(Level());
            get_puncture_tracker().track(current_time, dt, write_punctures);
        }
    }

    spherical_extraction_params_t extraction_params("weyl_extraction");
    extraction_params.fill_params();

    if (extraction_params.enabled)
    {
        const int min_level = extraction_params.min_extraction_level();
        bool calculate_weyl = at_level_timestep_multiple(min_level);

        if (calculate_weyl && Level() == min_level)
        {
            amrex::Real m_time       = get_state_data(state_index).curTime();
            amrex::Real m_dt         = get_gr_amr_ptr()->dtLevel(Level());
            amrex::Real restart_time = get_gr_amr_ptr()->get_restart_time();
            bool first_step          = (m_time <= m_dt);

            WeylExtraction my_extraction(extraction_params, m_dt, m_time,
                                         first_step, restart_time);
            my_extraction.execute_query(&get_bh_amr_ptr()->m_weyl_interpolator);
        }
    }
}
