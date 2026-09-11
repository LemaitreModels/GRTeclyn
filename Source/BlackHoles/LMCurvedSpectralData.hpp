/* GRTeclyn
 * Copyright 2022 The GRTL collaboration.
 * Please refer to LICENSE in GRTeclyn's root directory.
 */

// Reader/owner for an LMID-curved-puncture spectral export (format 3).
//
// Same division of labour as LMSpectralData / LMInitialData: the compute class
// `LMCurvedInitialData` must be trivially copyable (captured by value into an
// `amrex::ParallelFor` lambda), so it cannot own the spectral arrays; this
// class owns them and hands out raw pointers.
//
// FILE FORMAT: the plain ASCII token stream written by
// `lemaitre.initial_data.curved_puncture.validation.export_grteclyn`
// (`format 3`, sector `spinning`) — `#` comments then `<key> <values...>`
// records.  Against format 2 it carries SEVEN fields instead of one (`u` plus
// the six components of (L~b)_ij), a conformally CURVED closed-form background
// (quasi-isotropic Kerr, spins chi_A/chi_B along z, attenuation omega/p), and
// no Bowen-York momenta or spins.  The two formats are NOT interchangeable in
// either direction — both readers refuse the mismatch by equality.

#ifndef LMCURVEDSPECTRALDATA_HPP_
#define LMCURVEDSPECTRALDATA_HPP_

#include "LMCurvedInitialData.hpp"

#include <AMReX.H>
#include <AMReX_Gpu.H>
#include <AMReX_REAL.H>

#include <array>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

class LMCurvedSpectralData
{
  public:
    explicit LMCurvedSpectralData(const std::string &a_filename)
    {
        read(a_filename);
    }

    [[nodiscard]] LMCurvedInitialData::params_t
    params(const std::array<double, AMREX_SPACEDIM> &a_center,
           int a_initial_lapse) const
    {
        LMCurvedInitialData::params_t p{};
        p.b            = m_b;
        p.m_A          = m_m_A;
        p.m_B          = m_m_B;
        p.chi_A        = m_chi_A;
        p.chi_B        = m_chi_B;
        p.omega_over_b = m_omega_over_b;
        p.atten_p      = m_atten_p;
        p.center       = a_center;
        p.nA           = m_nA;
        p.nB           = m_nB;
        p.nphi         = m_nphi;
        p.ncos         = m_ncos;
        p.nsin         = m_nsin;
        p.A            = m_A.dataPtr();
        p.B            = m_B.dataPtr();
        p.wA           = m_wA.dataPtr();
        p.wB           = m_wB.dataPtr();
        p.cos_m        = m_cos_m.dataPtr();
        p.sin_m        = m_sin_m.dataPtr();
        for (int f = 0; f < LM_CURVED_NFIELDS; ++f)
        {
            p.C[f] = m_C[f].dataPtr();
            p.S[f] = m_S[f].dataPtr();
        }
        p.initial_lapse = a_initial_lapse;
        return p;
    }

    [[nodiscard]] double b() const { return m_b; }
    [[nodiscard]] const std::vector<std::string> &provenance() const
    {
        return m_provenance;
    }

  private:
    //! The exported field names, in file order (fixed by format 3).
    static const char *field_name(int f)
    {
        static const char *names[LM_CURVED_NFIELDS] = {
            "u", "Lb_xx", "Lb_xy", "Lb_xz", "Lb_yy", "Lb_yz", "Lb_zz"};
        return names[f];
    }

    void read(const std::string &filename)
    {
        std::ifstream f(filename);
        if (!f)
        {
            amrex::Abort("LMCurvedSpectralData: cannot open " + filename);
        }

        std::vector<double> A, B;
        std::vector<double> C[LM_CURVED_NFIELDS], S[LM_CURVED_NFIELDS];
        std::vector<int> cos_m, sin_m;
        std::vector<std::string> fields;
        std::string sector, F_kind;
        int format = -1, Na = -1, Nb = -1, ncos = -1, nsin = -1;
        std::string line;
        while (std::getline(f, line))
        {
            if (line.empty())
            {
                continue;
            }
            if (line[0] == '#')
            {
                m_provenance.push_back(line.substr(1));
                continue;
            }
            std::istringstream ss(line);
            std::string key;
            ss >> key;
            if (key == "format")
            {
                ss >> format;
            }
            else if (key == "sector")
            {
                ss >> sector;
            }
            else if (key == "b")
            {
                ss >> m_b;
            }
            else if (key == "m_A")
            {
                ss >> m_m_A;
            }
            else if (key == "m_B")
            {
                ss >> m_m_B;
            }
            else if (key == "chi_A")
            {
                ss >> m_chi_A;
            }
            else if (key == "chi_B")
            {
                ss >> m_chi_B;
            }
            else if (key == "omega_over_b")
            {
                ss >> m_omega_over_b;
            }
            else if (key == "atten_p")
            {
                ss >> m_atten_p;
            }
            else if (key == "F_kind")
            {
                ss >> F_kind;
            }
            else if (key == "F_s" || key == "F_q" || key == "F_k")
            {
                // parsed for completeness; unused when F_kind == none
            }
            else if (key == "Na")
            {
                ss >> Na;
            }
            else if (key == "Nb")
            {
                ss >> Nb;
            }
            else if (key == "Nphi")
            {
                ss >> m_nphi;
            }
            else if (key == "ncos")
            {
                ss >> ncos;
            }
            else if (key == "nsin")
            {
                ss >> nsin;
            }
            else if (key == "cos_m")
            {
                int v = 0;
                while (ss >> v)
                {
                    cos_m.push_back(v);
                }
            }
            else if (key == "sin_m")
            {
                int v = 0;
                while (ss >> v)
                {
                    sin_m.push_back(v);
                }
            }
            else if (key == "fields")
            {
                std::string v;
                while (ss >> v)
                {
                    fields.push_back(v);
                }
            }
            else if (key == "A" || key == "B")
            {
                std::vector<double> &dst = (key == "A") ? A : B;
                double v                 = 0.0;
                while (ss >> v)
                {
                    dst.push_back(v);
                }
            }
            else if (key.rfind("C_", 0) == 0 || key.rfind("S_", 0) == 0)
            {
                const bool is_c        = key[0] == 'C';
                const std::string name = key.substr(2);
                int idx                = -1;
                for (int k = 0; k < LM_CURVED_NFIELDS; ++k)
                {
                    if (name == field_name(k))
                    {
                        idx = k;
                    }
                }
                if (idx < 0)
                {
                    amrex::Abort("LMCurvedSpectralData: unknown field record " +
                                 key + " in " + filename);
                }
                std::vector<double> &dst = is_c ? C[idx] : S[idx];
                double v                 = 0.0;
                while (ss >> v)
                {
                    dst.push_back(v);
                }
            }
        }

        if (format != 3)
        {
            amrex::Abort(
                "LMCurvedSpectralData: " + filename + " has format " +
                std::to_string(format) +
                ", expected 3 (format 2 is the conformally-flat channel — its "
                "reader is LMSpectralData and the two data sets are not "
                "interchangeable)");
        }
        if (sector != "spinning")
        {
            amrex::Abort("LMCurvedSpectralData: " + filename + " has sector '" +
                         sector +
                         "'; this consumer implements the spinning-at-rest "
                         "sector (K == 0) only");
        }
        if (F_kind != "none")
        {
            amrex::Abort("LMCurvedSpectralData: " + filename +
                         " has F_kind '" + F_kind +
                         "'; this consumer implements F == 1 only, which is "
                         "how the sector ships");
        }
        if (static_cast<int>(fields.size()) != LM_CURVED_NFIELDS)
        {
            amrex::Abort("LMCurvedSpectralData: field-set mismatch in " +
                         filename);
        }
        for (int k = 0; k < LM_CURVED_NFIELDS; ++k)
        {
            if (fields[static_cast<size_t>(k)] != field_name(k))
            {
                amrex::Abort("LMCurvedSpectralData: field order mismatch in " +
                             filename);
            }
        }
        m_nA = Na + 1; // Na is the Chebyshev order: Na+1 nodes
        m_nB = Nb;
        if (m_nA <= 0 || m_nB <= 0 || m_nphi <= 0)
        {
            amrex::Abort("LMCurvedSpectralData: " + filename +
                         " missing Na/Nb/Nphi");
        }
        if (m_nA > LM_CURVED_MAX_NODES || m_nB > LM_CURVED_MAX_NODES)
        {
            amrex::Abort("LMCurvedSpectralData: node count exceeds "
                         "LM_CURVED_MAX_NODES; raise it in "
                         "LMCurvedInitialData.hpp");
        }
        m_ncos = ncos;
        m_nsin = nsin;
        if (static_cast<int>(cos_m.size()) != m_ncos ||
            static_cast<int>(sin_m.size()) != m_nsin)
        {
            amrex::Abort("LMCurvedSpectralData: mode-count mismatch in " +
                         filename);
        }
        if (static_cast<int>(A.size()) != m_nA ||
            static_cast<int>(B.size()) != m_nB)
        {
            amrex::Abort("LMCurvedSpectralData: node-count mismatch in " +
                         filename);
        }
        for (int k = 0; k < LM_CURVED_NFIELDS; ++k)
        {
            if (static_cast<int>(C[k].size()) != m_nA * m_nB * m_ncos ||
                static_cast<int>(S[k].size()) != m_nA * m_nB * m_nsin)
            {
                amrex::Abort(
                    "LMCurvedSpectralData: coefficient-count mismatch for "
                    "field " +
                    std::string(field_name(k)) + " in " + filename);
            }
        }
        if (std::abs(m_chi_A) >= 1.0 || std::abs(m_chi_B) >= 1.0)
        {
            amrex::Abort("LMCurvedSpectralData: |chi| >= 1 in " + filename);
        }

        // Barycentric weights — computed here rather than shipped, so the file
        // stays a description of the field and the consumer owns its own
        // interpolation (same rule as LMSpectralData).
        std::vector<double> wA(m_nA, 1.0), wB(m_nB, 1.0);
        for (int j = 0; j < m_nA; ++j)
        {
            double prod = 1.0;
            for (int k = 0; k < m_nA; ++k)
            {
                if (k != j)
                {
                    prod *= (A[j] - A[k]);
                }
            }
            wA[j] = 1.0 / prod;
        }
        for (int j = 0; j < m_nB; ++j)
        {
            double prod = 1.0;
            for (int k = 0; k < m_nB; ++k)
            {
                if (k != j)
                {
                    prod *= (B[j] - B[k]);
                }
            }
            wB[j] = 1.0 / prod;
        }

        auto to_dev_real = [](const std::vector<double> &v,
                              amrex::Gpu::DeviceVector<amrex::Real> &out)
        {
            out.resize(v.size());
            if (!v.empty())
            {
                amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, v.begin(),
                                      v.end(), out.begin());
            }
        };
        auto to_dev_int = [](const std::vector<int> &v,
                             amrex::Gpu::DeviceVector<int> &out)
        {
            out.resize(v.size());
            if (!v.empty())
            {
                amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, v.begin(),
                                      v.end(), out.begin());
            }
        };
        to_dev_real(A, m_A);
        to_dev_real(B, m_B);
        to_dev_real(wA, m_wA);
        to_dev_real(wB, m_wB);
        to_dev_int(cos_m, m_cos_m);
        to_dev_int(sin_m, m_sin_m);
        for (int k = 0; k < LM_CURVED_NFIELDS; ++k)
        {
            to_dev_real(C[k], m_C[k]);
            to_dev_real(S[k], m_S[k]);
        }
        amrex::Gpu::streamSynchronize();
    }

    double m_b{}, m_m_A{}, m_m_B{}, m_chi_A{}, m_chi_B{};
    double m_omega_over_b{};
    int m_atten_p{-1};
    int m_nA{-1}, m_nB{-1}, m_nphi{-1}, m_ncos{-1}, m_nsin{-1};
    amrex::Gpu::DeviceVector<amrex::Real> m_A, m_B, m_wA, m_wB;
    amrex::Gpu::DeviceVector<amrex::Real> m_C[LM_CURVED_NFIELDS];
    amrex::Gpu::DeviceVector<amrex::Real> m_S[LM_CURVED_NFIELDS];
    amrex::Gpu::DeviceVector<int> m_cos_m, m_sin_m;
    std::vector<std::string> m_provenance;
};

#endif /* LMCURVEDSPECTRALDATA_HPP_ */
