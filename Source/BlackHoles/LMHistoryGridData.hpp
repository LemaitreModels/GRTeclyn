/* GRTeclyn -- see LICENSE. */
#ifndef LMHISTORYGRIDDATA_HPP_
#define LMHISTORYGRIDDATA_HPP_

// Include the device functor first so its shared numerical core is compiled
// with AMReX's host/device qualifiers, including when the host parser uses it.
#include "LMHistoryInitialData.hpp"
#include "LMHistoryFile.hpp"
#include <AMReX_Gpu.H>
#include <array>
#include <type_traits>

class LMHistoryGridData
{
  public:
    explicit LMHistoryGridData(const std::string &filename)
    {
        static_assert(std::is_same_v<amrex::Real, double>,
                      "format 5 requires a double-precision GRTeclyn build");
        try
        {
            const LMHistoryFile file(filename);
            m_view.stencil = file.stencil;
            for (int d = 0; d < 3; ++d)
            {
                m_view.n[d] = static_cast<int>(file.axes[d].size());
                upload(file.axes[d], m_axes[d]);
                m_view.axis[d] = m_axes[d].dataPtr();
            }
            upload(file.values, m_values);
            m_view.values = m_values.dataPtr();
            // Complete copies before host vectors die (and before first use).
            amrex::Gpu::streamSynchronize();
            amrex::Print() << "LM history: format 5, " << m_view.n[0] << " x "
                << m_view.n[1] << " x " << m_view.n[2] << ", stencil "
                << m_view.stencil << ", lapse " << file.lapse_choice
                << ", shift " << file.shift_choice << "\n";
        }
        catch (const std::exception &error) { amrex::Abort(error.what()); }
    }
    LMHistory::View view() const { return m_view; }
  private:
    static void upload(const std::vector<double> &source,
                       amrex::Gpu::DeviceVector<double> &dest)
    {
        dest.resize(source.size());
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
                             source.begin(), source.end(), dest.begin());
    }
    LMHistory::View m_view{};
    std::array<amrex::Gpu::DeviceVector<double>, 3> m_axes;
    amrex::Gpu::DeviceVector<double> m_values;
};
#endif
