/* GRTeclyn
 * Copyright 2022 The GRTL collaboration.
 * Please refer to LICENSE in GRTeclyn's root directory.
 */

#include "BinaryBHLevel.hpp"
#include "BinaryBHInitialData.hpp"
#include "CCZ4RHS.hpp"
#include "ChiTagger.hpp"
#include "ConstraintNorms.hpp"
#include "Constraints.hpp"
#include "ExtractionTagger.hpp"
#include "GammaCalculator.hpp"
#include "LMCurvedInitialData.hpp"
#include "LMCurvedSpectralData.hpp"
#include "LMInitialData.hpp"
#include "LMSpectralData.hpp"
#include "PositiveChiAndLapse.hpp"
#include "PunctureTagger.hpp"
#include "PunctureTracker.hpp"
// xxxxx #include "SixthOrderDerivatives.hpp"
#include "TraceARemoval.hpp"
#include "TwoPuncturesInitialData.hpp"
#include "Weyl4.hpp"
#include "WeylExtraction.hpp"

BHAMR<BinaryBHLevel::num_punctures> *BinaryBHLevel::get_bhamr_ptr()
{
    return dynamic_cast<BHAMR<num_punctures> *>(get_gramr_ptr());
}

PunctureTracker<BinaryBHLevel::num_punctures> &
BinaryBHLevel::get_puncture_tracker()
{
    return get_bhamr_ptr()->get_puncture_tracker();
}

void BinaryBHLevel::variableSetUp()
{
    BL_PROFILE("BinaryBHLevel::variableSetUp()");

    // Set up the state variables
    stateVariableSetUp();

    Constraints::set_up(state_index);

    Weyl4::set_up(state_index);
}

// Things to do during the advance step after RK4 steps
void BinaryBHLevel::specificAdvance()
{
    amrex::MultiFab &state_new = get_new_data(state_index);
    const auto &state_arrays   = state_new.arrays();

    // The classes to be used
    TraceARemoval trace_A_removal;
    PositiveChiAndLapse positive_chi_lapse;

    // Enforce the trace free A_ij condition and positive chi and lapse
    amrex::ParallelFor(state_new,
                       [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
                       {
                           trace_A_removal(ix, iy, iz, state_arrays[box_no]);
                           positive_chi_lapse(ix, iy, iz, state_arrays[box_no]);
                       });
}

// This initial data uses an approximation for the metric which
// is valid for small boosts
void BinaryBHLevel::initData()
{
    BL_PROFILE("BinaryBHLevel::initialData");
    if (m_verbosity > 0)
    {
        amrex::Print() << "BinaryBHLevel::initialData " << Level() << "\n";
    }
#ifdef USE_TWOPUNCTURES
    // xxxxx USE_TWOPUNCTURES todo
    TwoPuncturesInitialData two_punctures_initial_data(
        m_dx, m_p.center, m_tp_amr.m_two_punctures);
    // Can't use simd with this initial data
    BoxLoops::loop(two_punctures_initial_data, m_state_new, m_state_new,
                   INCLUDE_GHOST_CELLS, disable_simd());
#else
    double dx = Geom().CellSize(0);
    // First set everything to zero (to avoid undefinded values in constraints)
    // then calculate initial data
    amrex::MultiFab &state_new = get_new_data(state_index);
    const auto &state_arrays   = state_new.arrays();

    if (!simParams().lm_curved_id_file.empty())
    {
        // LM-initial-data CONFORMALLY CURVED spectral initial data (format 3,
        // spinning-at-rest sector).  Runtime-selected like the flat channel
        // below, and mutually exclusive with it.
        if (!simParams().lm_id_file.empty())
        {
            amrex::Abort("BinaryBHLevel: lm_id_file and lm_curved_id_file are "
                         "mutually exclusive — pick one initial-data channel");
        }
        static const LMCurvedSpectralData lm_curved_data(
            simParams().lm_curved_id_file);
        LMCurvedInitialData::params_t lm_params =
            lm_curved_data.params(simParams().center, Lapse::PRE_COLLAPSED);
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
            if (!simParams().lm_curved_id_reference_file.empty())
            {
                auto err = lm_initial_data.validate(
                    simParams().lm_curved_id_reference_file);
                const double tol = simParams().lm_curved_id_reference_tol;
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
        // it is not, and the stock GammaCalculator computes it from the h_ij
        // just written.  The ParallelFor above filled the ghost cells by
        // direct evaluation (nGrowVect), so the derivative stencils are valid
        // without a FillPatch.
        amrex::Gpu::streamSynchronize();
        BoxLoops::loop(GammaCalculator(m_dx), state_new, state_new,
                       EXCLUDE_GHOST_CELLS);
    }
    else if (!simParams().lm_id_file.empty())
    {
        // LM-initial-data spectral initial data.  Runtime-selected, so this is
        // the SAME executable, stencils and variable conversion as the analytic
        // branch below — the comparison is then of initial data alone.
        // Read once per process (the file is read-only static data used by
        // every level).
        static const LMSpectralData lm_data(simParams().lm_id_file);
        LMInitialData::params_t lm_params =
            lm_data.params(simParams().center, Lapse::PRE_COLLAPSED);
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
            if (!simParams().lm_id_reference_file.empty())
            {
                auto err =
                    lm_initial_data.validate(simParams().lm_id_reference_file);
                if (err[0] > simParams().lm_id_reference_tol ||
                    err[1] > simParams().lm_id_reference_tol)
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
        // Set up the compute class for the BinaryBH initial data
        BinaryBHInitialData binary_initial_data(simParams().bh1_params,
                                                simParams().bh2_params, dx);

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

    if (simParams().puncture_tracking_enabled && Level() == 0)
    {
        // need to set the puncture coordinates as we use it for the puncture
        // tagging
        get_puncture_tracker().set_puncture_coords(
            {simParams().bh1_params.center[0], simParams().bh1_params.center[1],
             simParams().bh1_params.center[2], simParams().bh2_params.center[0],
             simParams().bh2_params.center[1],
             simParams().bh2_params.center[2]});
        // can't call start_from_initial_punctures() because we need the full
        // AMR grid first
    }
}

// Calculate RHS during RK4 substeps
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void BinaryBHLevel::specificEvalRHS(amrex::MultiFab &a_soln,
                                    amrex::MultiFab &a_rhs,
                                    const double /*a_time*/)
{
    BL_PROFILE("BinaryBHLevel::specificEvalRHS()");
    const auto &soln_arrays       = a_soln.arrays();
    const auto &const_soln_arrays = a_soln.const_arrays();
    const auto &rhs_arrays        = a_rhs.arrays();
    const auto soln_ghosts        = a_soln.nGrowVect();

    // The classes to be used
    TraceARemoval trace_A_removal;
    PositiveChiAndLapse positive_chi_lapse;

    // Enforce positive chi and lapse and trace free A
    amrex::ParallelFor(a_soln, soln_ghosts,
                       [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
                       {
                           trace_A_removal(ix, iy, iz, soln_arrays[box_no]);
                           positive_chi_lapse(ix, iy, iz, soln_arrays[box_no]);
                       });

    // Calculate CCZ4 right hand side
    if (simParams().max_spatial_derivative_order == 4)
    {
        CCZ4RHS<MovingPunctureGauge, FourthOrderDerivatives> ccz4rhs(
            simParams().ccz4_params, Geom().CellSize(0), simParams().sigma,
            simParams().formulation);

        amrex::ParallelFor(
            a_rhs,
            [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
            {
                ccz4rhs(ix, iy, iz, rhs_arrays[box_no],
                        const_soln_arrays[box_no]);
            });
    }
    else if (simParams().max_spatial_derivative_order == 6)
    {
        amrex::Abort("xxxxx max_spatial_derivative_order == 6 todo");
#if 0
        CCZ4RHS<MovingPunctureGauge, SixthOrderDerivatives>
            ccz4rhs(simParams().ccz4_params, Geom().CellSize(0), simParams().sigma,
                    simParams().formulation);
        amrex::ParallelFor(a_rhs,
        [=] AMREX_GPU_DEVICE (int box_no, int ix, int iy, int iz)
        {
            amrex::CellData<amrex::Real const> state = const_soln_arrays[box_no].cellData(i,j,k);
            amrex::CellData<amrex::Real> rhs = rhs_arrays[box_no].cellData(ix,iy,iz);
            ccz4rhs.compute(rhs, state);
        });
#endif
    }

    amrex::Gpu::streamSynchronize();
}

// enforce trace removal during RK4 substeps
void BinaryBHLevel::specificUpdateODE(amrex::MultiFab &a_soln)
{

    TraceARemoval trace_A_removal;
    const auto soln_ghosts = amrex::IntVect(0); // zero ghost cells

    // Enforce the trace free A_ij condition
    const auto &soln_arrays = a_soln.arrays();
    amrex::ParallelFor(a_soln, soln_ghosts,
                       [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
                       { trace_A_removal(ix, iy, iz, soln_arrays[box_no]); });

    amrex::Gpu::streamSynchronize();
}

void BinaryBHLevel::pre_tag_cells()
{
    amrex::MultiFab &state_new = get_new_data(state_index);
    const auto current_time    = get_state_data(state_index).curTime();

    // Just fill 2 ghosts for chi to calculate second derivatives
    const int nghost = 2;
    const int ncomp  = 1;
    FillPatch(*this, state_new, nghost, current_time, state_index, c_chi,
              ncomp);
}

void BinaryBHLevel::tag_cells(amrex::TagBoxArray &a_tag_box_array,
                              amrex::Real a_regrid_threshold)
{
    BL_PROFILE("BinaryBHLevel::tag_cells()");
    amrex::MultiFab &state_new = get_new_data(state_index);

    const auto &tag_arrays         = a_tag_box_array.arrays();
    const auto &state_const_arrays = state_new.const_arrays();

    ChiTagger chi_tagger(Geom().CellSize(0), a_regrid_threshold);

    ExtractionTagger extraction_tagger(Geom().CellSize(0), Level(),
                                       simParams().extraction_params,
                                       simParams().activate_extraction);

    const bool puncture_tracking_enabled =
        simParams().puncture_tracking_enabled;
    constexpr auto num_puncture_coords =
        static_cast<std::size_t>(AMREX_SPACEDIM * num_punctures);
    std::array<amrex::Real, num_puncture_coords> puncture_coords{};

    if (puncture_tracking_enabled)
    {
        puncture_coords = get_puncture_tracker().get_puncture_coords();
    }

    PunctureTagger<num_punctures> puncture_tagger(
        Geom().CellSize(0), Level(), get_gramr_ptr()->maxLevel(),
        puncture_coords,
        {simParams().bh1_params.mass, simParams().bh2_params.mass});

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

    if (simParams().puncture_tracking_enabled)
    {
        get_puncture_tracker().start_from_initial_punctures();
    }

    // Constraint norms of the INITIAL DATA (t = 0).  post_init cascades from
    // the finest level down, so at Level() == 0 the whole hierarchy exists and
    // the hierarchy-wide `derive` below is safe.  Measuring here rather than in
    // specificPostTimeStep means the run needs no evolution step at all.
    // (The norm block in specificPostTimeStep is still #if 0'd: it uses
    // GRChombo's AMRReductions, which was never ported to AMReX.)
    if (simParams().calculate_constraint_norms && Level() == 0)
    {
        ConstraintNorms::Options opts;
        opts.r_excl       = simParams().constraint_norm_exclusion_radius;
        opts.border_cells = simParams().constraint_norm_border_cells;
        opts.punctures.push_back(simParams().bh1_params.center);
        opts.punctures.push_back(simParams().bh2_params.center);

        const amrex::Real time = get_state_data(state_index).curTime();
        auto res = ConstraintNorms::compute(*get_gramr_ptr(), time, opts);

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
              << get_gramr_ptr()->getLevel(0).Domain().length(0) << ",";
        ConstraintNorms::write_json(res,
                                    simParams().data_path +
                                        "constraint_norms.json",
                                    time, opts, extra.str());
    }
}

void BinaryBHLevel::specific_post_restart()
{
    BL_PROFILE("BinaryBHLevel::specific_post_restart()");

    if (simParams().puncture_tracking_enabled)
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
    if (simParams().puncture_tracking_enabled)
    {
        get_puncture_tracker().write_plotfile(a_dir);
    }
}

void BinaryBHLevel::specific_post_checkpoint(const std::string &a_chk_dir,
                                             std::ostream & /*a_os*/)
{
    if (simParams().puncture_tracking_enabled)
    {
        get_puncture_tracker().checkpoint(a_chk_dir);
    }
}

void BinaryBHLevel::specificPostTimeStep()
{
    BL_PROFILE("BinaryBHLevel::specificPostTimeStep");

    // do puncture tracking on requested level
    if (simParams().puncture_tracking_enabled &&
        Level() == simParams().puncture_tracking_level)
    {
        BL_PROFILE("PunctureTracking");

        // only do the write out when we're at at a multiple of the
        // writeout_level
        bool write_punctures = at_level_timestep_multiple(
            simParams().puncture_tracking_writeout_level);
        amrex::Real current_time = get_state_data(state_index).curTime();
        amrex::Real dt           = get_gramr_ptr()->dtLevel(Level());
        get_puncture_tracker().track(current_time, dt, write_punctures);
    }

    // Weyl extraction
    if (simParams().activate_extraction)
    {
        int min_level = simParams().extraction_params.min_extraction_level();
        bool calculate_weyl = at_level_timestep_multiple(min_level);

        if (calculate_weyl && Level() == min_level)
        {
            amrex::Real m_time       = get_state_data(state_index).curTime();
            amrex::Real m_dt         = get_gramr_ptr()->dtLevel(Level());
            amrex::Real restart_time = get_gramr_ptr()->get_restart_time();
            bool first_step          = (m_time <= m_dt);

            WeylExtraction my_extraction(simParams().extraction_params, m_dt,
                                         m_time, first_step, restart_time);
            my_extraction.execute_query(&get_bhamr_ptr()->m_weyl_interpolator);
        }
    }

#if 0
//xxxxx specificPostTimeStep

    if (m_p.calculate_constraint_norms)
    {
        fillAllGhosts();
        BoxLoops::loop(Constraints(m_dx, c_Ham, Interval(c_Mom1, c_Mom3)),
                       m_state_new, m_state_diagnostics, EXCLUDE_GHOST_CELLS);
        if (m_level == 0)
        {
            AMRReductions<VariableType::derived> amr_reductions(m_gr_amr);
            double L2_Ham = amr_reductions.norm(c_Ham);
            double L2_Mom = amr_reductions.norm(Interval(c_Mom1, c_Mom3));
            SmallDataIO constraints_file(m_p.data_path + "constraint_norms",
                                         m_dt, m_time, m_restart_time,
                                         SmallDataIO::APPEND, first_step);
            constraints_file.remove_duplicate_time_data();
            if (first_step)
            {
                constraints_file.write_header_line({"L^2_Ham", "L^2_Mom"});
            }
            constraints_file.write_time_data_line({L2_Ham, L2_Mom});
        }
    }

#endif
}
