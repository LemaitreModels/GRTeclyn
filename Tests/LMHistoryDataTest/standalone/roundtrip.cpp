/* GRTeclyn -- see LICENSE. Standalone check of the production format-5 core.
 * From parent: c++ -std=c++17 -I../../Source/BlackHoles standalone/roundtrip.cpp -o history-reader
 * history-reader slice.lmid points.txt -> sixteen ADM + eighteen CCZ4 fields.
 * Without points.txt, validates only. Failures return 2 with a diagnostic.
 */
#include "LMHistoryFile.hpp"
#include <iomanip>
#include <iostream>

int main(int argc, char **argv)
{
    try
    {
        if (argc < 2 || argc > 3) throw std::runtime_error("usage: reader slice [points]");
        const LMHistoryFile file(argv[1]);
        if (argc == 2) { std::cout << "format 5 accepted\n"; return 0; }
        std::ifstream points(argv[2]);
        if (!points) throw std::runtime_error("cannot open points");
        double p[3];
        std::cout << std::setprecision(17);
        while (points >> p[0])
        {
            if (!(points >> p[1] >> p[2])) throw std::runtime_error("incomplete point");
            double fields[LMHistory::nfields], state[LMHistory::nstate];
            if (!LMHistory::evaluate(file.view(), p, fields) ||
                !LMHistory::decompose(fields, state))
                throw std::runtime_error("outside export grid or invalid interpolated slice");
            for (double v : fields) std::cout << v << ' ';
            for (double v : state) std::cout << v << ' ';
            std::cout << '\n';
        }
        if (!points.eof()) throw std::runtime_error("malformed point");
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
