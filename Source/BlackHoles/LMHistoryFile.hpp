/* GRTeclyn -- see LICENSE. */
#ifndef LMHISTORYFILE_HPP_
#define LMHISTORYFILE_HPP_

#include "LMHistoryInterpolation.hpp"
#include <array>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Host parser. Both the AMReX owner and standalone test use this exact reader.
// Throws on malformed/version-incompatible data; the AMReX owner aborts with
// that diagnostic. No dependency on Python or a particular PM evaluator.
class LMHistoryFile
{
  public:
    std::array<std::vector<double>, 3> axes;
    std::vector<double> values;
    int stencil{};
    std::string lapse_choice, shift_choice;

    static const std::array<std::string, LMHistory::nfields> &field_names()
    {
        static const std::array<std::string, LMHistory::nfields> names = {
            "gamma_xx", "gamma_xy", "gamma_xz", "gamma_yy", "gamma_yz", "gamma_zz",
            "K_xx", "K_xy", "K_xz", "K_yy", "K_yz", "K_zz", "lapse",
            "shift_x", "shift_y", "shift_z"};
        return names;
    }

    explicit LMHistoryFile(const std::string &filename)
    {
        std::ifstream input(filename);
        if (!input) fail("cannot open " + filename);
        std::map<std::string, std::vector<std::string>> rows;
        std::string line;
        while (std::getline(input, line))
        {
            line = line.substr(0, line.find('#'));
            std::istringstream stream(line);
            std::string key, token;
            if (!(stream >> key)) continue;
            if (rows.count(key)) fail("duplicate record " + key);
            std::vector<std::string> tokens;
            while (stream >> token) tokens.push_back(token);
            rows.emplace(key, std::move(tokens));
        }
        auto scalar = [&](const std::string &key) -> std::string
        {
            const auto it = rows.find(key);
            if (it == rows.end() || it->second.size() != 1)
                fail("expected one token for " + key);
            return it->second[0];
        };
        // Equality, never a minimum-version check: formats 2/3/4 have different
        // fields, interpolation and background reconstruction.
        if (scalar("format") != "5") fail("expected format 5");
        if (scalar("grid") != "cartesian") fail("expected Cartesian grid");
        stencil = integer(scalar("stencil"));
        if (stencil != 2 && stencil != 4 && stencil != 6 && stencil != 8)
            fail("stencil must be 2, 4, 6, or 8");
        lapse_choice = scalar("lapse_choice");
        shift_choice = scalar("shift_choice");
        if (lapse_choice != "provided" && lapse_choice != "one" &&
            lapse_choice != "precollapsed") fail("unsupported lapse_choice");
        if (shift_choice != "provided" && shift_choice != "zero")
            fail("unsupported shift_choice");
        if (rows["fields"] != std::vector<std::string>(field_names().begin(),
                                                       field_names().end()))
            fail("field order mismatch");
        std::vector<std::string> known = {"format", "grid", "stencil", "Nx", "Ny",
            "Nz", "x", "y", "z", "lapse_choice", "shift_choice", "fields"};
        std::size_t count = 1;
        for (int d = 0; d < 3; ++d)
        {
            const std::string axis(1, "xyz"[d]);
            const int n = integer(scalar("N" + axis));
            if (n < stencil) fail("node count below stencil");
            const auto &r = rows[axis];
            if (r.size() != static_cast<std::size_t>(n)) fail("node count mismatch");
            for (const auto &v : r) axes[d].push_back(number(v));
            for (int i = 1; i < n; ++i)
                if (!(axes[d][i] > axes[d][i-1])) fail("nodes not increasing");
            if (count > std::numeric_limits<std::size_t>::max()/n/LMHistory::nfields)
                fail("grid size overflow");
            count *= n;
        }
        // Validate counts BEFORE allocation; unknown/missing keys are errors.
        for (const auto &name : field_names())
        {
            known.push_back("data_" + name);
            if (rows["data_" + name].size() != count) fail("field count mismatch");
        }
        if (rows.size() != known.size()) fail("unknown or missing records");
        for (const auto &key : known)
            if (!rows.count(key)) fail("missing record " + key);
        values.resize(count*LMHistory::nfields);
        for (int f = 0; f < LMHistory::nfields; ++f)
            for (std::size_t i = 0; i < count; ++i)
                values[i*LMHistory::nfields+f] = number(rows["data_" + field_names()[f]][i]);
        for (std::size_t i = 0; i < count; ++i)
        {
            const double *v = values.data()+i*LMHistory::nfields;
            double state[LMHistory::nstate];
            if (!LMHistory::decompose(v, state)) fail("invalid metric/lapse at node");
            const double expected = lapse_choice == "one" ? 1.0 : std::sqrt(state[0]);
            if (lapse_choice != "provided" &&
                std::abs(v[12]-expected) > 2e-13*expected)
                fail("lapse disagrees with policy");
            if (shift_choice == "zero" && (v[13] != 0 || v[14] != 0 || v[15] != 0))
                fail("shift disagrees with policy");
        }
    }

    LMHistory::View view() const
    {
        LMHistory::View v{};
        v.stencil = stencil;
        for (int d = 0; d < 3; ++d)
        {
            v.n[d] = static_cast<int>(axes[d].size());
            v.axis[d] = axes[d].data();
        }
        v.values = values.data();
        return v;
    }

  private:
    [[noreturn]] static void fail(const std::string &why)
    { throw std::runtime_error("LMHistoryFile: " + why); }
    static int integer(const std::string &s)
    {
        std::size_t used = 0;
        const long n = std::stol(s, &used);
        if (used != s.size() || n < 0 || n > std::numeric_limits<int>::max())
            fail("invalid integer");
        return static_cast<int>(n);
    }
    static double number(const std::string &s)
    {
        std::size_t used = 0;
        const double n = std::stod(s, &used);
        if (used != s.size() || !std::isfinite(n)) fail("invalid finite number");
        return n;
    }
};
#endif
