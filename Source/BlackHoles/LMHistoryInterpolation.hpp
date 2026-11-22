/* GRTeclyn -- see LICENSE. */
#ifndef LMHISTORYINTERPOLATION_HPP_
#define LMHISTORYINTERPOLATION_HPP_

#include <cmath>
#ifdef AMREX_GPU_HOST_DEVICE
#define LM_HISTORY_HD AMREX_GPU_HOST_DEVICE
#else
#define LM_HISTORY_HD
#endif

// Portable numerical core, shared by the actual device functor and the
// standalone reader check. No AMReX storage or host I/O in these functions.
namespace LMHistory
{
constexpr int nfields = 16;
constexpr int nstate = 18;
constexpr int max_stencil = 8;

struct View
{
    int n[3]{};
    int stencil{};
    const double *axis[3]{};
    const double *values{}; // point-major [x][y][z][field], field fastest
};

LM_HISTORY_HD inline bool weights(double q, const double *axis, int n,
                                   int count, int &start, double *w)
{
    if (!std::isfinite(q) || q < axis[0] || q > axis[n - 1]) return false;
    int lo = 0, hi = n;
    while (lo < hi)
    {
        const int mid = lo + (hi - lo) / 2;
        if (axis[mid] <= q) lo = mid + 1;
        else hi = mid;
    }
    start = lo - count / 2;
    if (start < 0) start = 0;
    if (start > n - count) start = n - count;
    for (int i = 0; i < count; ++i)
    {
        w[i] = 1.0;
        for (int j = 0; j < count; ++j)
            if (i != j) w[i] *= (q - axis[start + j]) /
                                     (axis[start + i] - axis[start + j]);
    }
    return true;
}

// gamma is packed xx,xy,xz,yy,yz,zz. Positive principal minors test SPD.
LM_HISTORY_HD inline bool inverse(const double *g, double *inv, double &det)
{
    inv[0] = g[3] * g[5] - g[4] * g[4];
    inv[1] = g[2] * g[4] - g[1] * g[5];
    inv[2] = g[1] * g[4] - g[2] * g[3];
    inv[3] = g[0] * g[5] - g[2] * g[2];
    inv[4] = g[1] * g[2] - g[0] * g[4];
    inv[5] = g[0] * g[3] - g[1] * g[1];
    det = g[0] * inv[0] + g[1] * inv[1] + g[2] * inv[2];
    if (!(g[0] > 0 && inv[5] > 0 && det > 0 && std::isfinite(det)))
        return false;
    for (int i = 0; i < 6; ++i) inv[i] /= det;
    return true;
}

// CCZ4 definitions: chi=det(gamma)^(-1/3), h=chi gamma,
// trK=gamma^ij K_ij, A=chi(K_ij-gamma_ij trK/3). Keep the FULL trace.
// Output: chi, h[6], A[6], trK, lapse, shift^i[3].
LM_HISTORY_HD inline bool decompose(const double *fields, double *state)
{
    for (int f = 0; f < nfields; ++f)
        if (!std::isfinite(fields[f])) return false;
    double inv[6], det;
    if (!inverse(fields, inv, det) || !(fields[12] > 0)) return false;
    const double chi = 1.0 / std::cbrt(det);
    const double trace = inv[0]*fields[6] + inv[3]*fields[9] + inv[5]*fields[11] +
        2.0*(inv[1]*fields[7] + inv[2]*fields[8] + inv[4]*fields[10]);
    state[0] = chi;
    for (int i = 0; i < 6; ++i)
    {
        state[1+i] = chi * fields[i];
        state[7+i] = chi * (fields[6+i] - fields[i]*trace/3.0);
    }
    state[13] = trace;
    for (int i = 0; i < 4; ++i) state[14+i] = fields[12+i];
    for (int i = 0; i < nstate; ++i)
        if (!std::isfinite(state[i])) return false;
    return true;
}

LM_HISTORY_HD inline bool evaluate(const View &v, const double *point,
                                    double *fields)
{
    double w[3][max_stencil];
    int start[3];
    if (v.stencil < 2 || v.stencil > max_stencil) return false;
    for (int d = 0; d < 3; ++d)
        if (!weights(point[d], v.axis[d], v.n[d], v.stencil, start[d], w[d]))
            return false;
    for (int f = 0; f < nfields; ++f) fields[f] = 0.0;
    for (int i = 0; i < v.stencil; ++i)
        for (int j = 0; j < v.stencil; ++j)
            for (int k = 0; k < v.stencil; ++k)
            {
                const auto index = ((static_cast<unsigned long long>(start[0]+i)*
                    v.n[1] + start[1]+j)*v.n[2] + start[2]+k)*nfields;
                const double weight = w[0][i]*w[1][j]*w[2][k];
                for (int f = 0; f < nfields; ++f)
                    fields[f] += weight * v.values[index+f];
            }
    double state[nstate];
    return decompose(fields, state);
}
} // namespace LMHistory
#undef LM_HISTORY_HD
#endif
