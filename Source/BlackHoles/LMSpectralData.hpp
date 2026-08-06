/* GRTeclyn
 * Copyright 2022 The GRTL collaboration.
 * Please refer to LICENSE in GRTeclyn's root directory.
 */

// Reader/owner for an LM-initial-data spectral export.
//
// The compute class `LMInitialData` has to satisfy
// `std::is_trivially_copyable_v` (it is captured by value into an
// `amrex::ParallelFor` lambda), so it cannot own the spectral arrays.  This
// class owns them and hands out raw pointers, in the same division of labour as
// `TP::TwoPunctures` / `TwoPuncturesInitialData`.
//
// FILE FORMAT: the plain ASCII token stream written by
// `lm.initial_data.validation.export_grteclyn` (`format 1`) — `#` comments then
// `<key> <values...>` records.  ASCII rather than JSON/HDF5 precisely so that
// this reader needs no library.  The physics content is
//
//     psi(x) = psi_BL(x) + u(x),  psi_BL = 1 + m_A/(2 r_A) + m_B/(2 r_B)
//
// with punctures A at (0,0,+b) and B at (0,0,-b) relative to the grid centre,
// and `u` given on the ABT prolate-spheroidal grid as
//
//     u(A,B,phi) = sum_t C[i][j][t] cos(cos_m[t] phi)
//                + sum_t S[i][j][t] sin(sin_m[t] phi)
//
// interpolated in (A,B) by a tensor-product barycentric rule.  See
// LMInitialData.hpp for the evaluation.

#ifndef LMSPECTRALDATA_HPP_
#define LMSPECTRALDATA_HPP_

#include "LMInitialData.hpp"

#include <AMReX.H>
#include <AMReX_Gpu.H>
#include <AMReX_REAL.H>

#include <array>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

class LMSpectralData
{
  public:
    explicit LMSpectralData(const std::string &a_filename)
    {
        read(a_filename);
    }

    /// Trivially-copyable view for the compute class.  `center` is the grid
    /// centre the punctures are placed relative to.
    [[nodiscard]] LMInitialData::params_t
    params(const std::array<double, AMREX_SPACEDIM> &a_center,
           int a_initial_lapse) const
    {
        LMInitialData::params_t p{};
        p.b   = m_b;
        p.m_A = m_m_A;
        p.m_B = m_m_B;
        for (int d = 0; d < 3; ++d)
        {
            p.P_A[d] = m_P_A[d];
            p.P_B[d] = m_P_B[d];
            p.S_A[d] = m_S_A[d];
            p.S_B[d] = m_S_B[d];
        }
        p.center        = a_center;
        p.nA            = m_nA;
        p.nB            = m_nB;
        p.nphi          = m_nphi;
        p.ncos          = m_ncos;
        p.nsin          = m_nsin;
        p.A             = m_A.dataPtr();
        p.B             = m_B.dataPtr();
        p.wA            = m_wA.dataPtr();
        p.wB            = m_wB.dataPtr();
        p.cos_m         = m_cos_m.dataPtr();
        p.sin_m         = m_sin_m.dataPtr();
        p.C             = m_C.dataPtr();
        p.S             = m_S.dataPtr();
        p.initial_lapse = a_initial_lapse;
        return p;
    }

    [[nodiscard]] double b() const { return m_b; }
    [[nodiscard]] const std::vector<std::string> &provenance() const
    {
        return m_provenance;
    }

  private:
    void read(const std::string &filename)
    {
        std::ifstream f(filename);
        if (!f)
        {
            amrex::Abort("LMSpectralData: cannot open " + filename);
        }

        std::vector<double> A, B, C, S;
        std::vector<int> cos_m, sin_m;
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
            else if (key == "P_A")
            {
                ss >> m_P_A[0] >> m_P_A[1] >> m_P_A[2];
            }
            else if (key == "P_B")
            {
                ss >> m_P_B[0] >> m_P_B[1] >> m_P_B[2];
            }
            else if (key == "S_A")
            {
                ss >> m_S_A[0] >> m_S_A[1] >> m_S_A[2];
            }
            else if (key == "S_B")
            {
                ss >> m_S_B[0] >> m_S_B[1] >> m_S_B[2];
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
            else if (key == "A" || key == "B" || key == "C" || key == "S")
            {
                std::vector<double> &dst =
                    (key == "A") ? A : (key == "B") ? B : (key == "C") ? C : S;
                double v = 0.0;
                while (ss >> v)
                {
                    dst.push_back(v);
                }
            }
        }

        if (format != 1)
        {
            amrex::Abort("LMSpectralData: " + filename + " has format " +
                         std::to_string(format) + ", expected 1");
        }
        m_nA = Na + 1; // Na is the Chebyshev order: Na+1 nodes
        m_nB = Nb;
        if (m_nA <= 0 || m_nB <= 0 || m_nphi <= 0)
        {
            amrex::Abort("LMSpectralData: " + filename + " missing Na/Nb/Nphi");
        }
        if (m_nA > LM_MAX_NODES || m_nB > LM_MAX_NODES)
        {
            amrex::Abort("LMSpectralData: node count exceeds LM_MAX_NODES; "
                         "raise it in LMInitialData.hpp");
        }
        m_ncos = ncos;
        m_nsin = nsin;
        if (static_cast<int>(cos_m.size()) != m_ncos ||
            static_cast<int>(sin_m.size()) != m_nsin)
        {
            amrex::Abort("LMSpectralData: mode-count mismatch in " + filename);
        }
        if (static_cast<int>(A.size()) != m_nA ||
            static_cast<int>(B.size()) != m_nB)
        {
            amrex::Abort("LMSpectralData: node-count mismatch in " + filename);
        }
        if (static_cast<int>(C.size()) != m_nA * m_nB * m_ncos ||
            static_cast<int>(S.size()) != m_nA * m_nB * m_nsin)
        {
            amrex::Abort("LMSpectralData: coefficient-count mismatch in " +
                         filename);
        }

        // Barycentric weights of each node set — computed here rather than
        // shipped, so the file stays a description of the field and the
        // consumer owns its own interpolation.
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
        to_dev_real(C, m_C);
        to_dev_real(S, m_S);
        to_dev_int(cos_m, m_cos_m);
        to_dev_int(sin_m, m_sin_m);
        amrex::Gpu::streamSynchronize();
    }

    double m_b{}, m_m_A{}, m_m_B{};
    double m_P_A[3]{}, m_P_B[3]{}, m_S_A[3]{}, m_S_B[3]{};
    int m_nA{-1}, m_nB{-1}, m_nphi{-1}, m_ncos{-1}, m_nsin{-1};
    amrex::Gpu::DeviceVector<amrex::Real> m_A, m_B, m_wA, m_wB, m_C, m_S;
    amrex::Gpu::DeviceVector<int> m_cos_m, m_sin_m;
    std::vector<std::string> m_provenance;
};

#endif /* LMSPECTRALDATA_HPP_ */
