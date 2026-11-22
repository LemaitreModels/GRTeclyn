# Format-5 standalone reader check

This executable uses the production host parser, interpolation and CCZ4
conversion. It needs a C++17 compiler, without an AMReX build. From this directory:

```sh
c++ -std=c++17 -O2 -I../../Source/BlackHoles standalone/roundtrip.cpp -o /tmp/lm-history-reader
```

Run the radiative-puncture leaf's `tests/test_export_grteclyn.py` with
`LM_HISTORY_READER=/tmp/lm-history-reader` for Python-to-C++ round trips,
malformed/version rejection and a boosted-Schwarzschild vacuum constraint smoke.
The leaf supplies analytic fixtures; no evolution code is imported by Python.
Without the environment variable the leaf skips C++ checks explicitly.

The source is in `standalone/` so the Tests makefile does not collect its `main`
into the normal doctest executable.

The standalone check does not compile or execute the AMReX owner, device functor,
BinaryBH initialisation or GammaCalculator. A native CPU build and two-step GPU
smoke are required before production. The runtime key is `lm.history_id_file`;
the binary must be built without `USE_TWOPUNCTURES`. The finite export grid
must cover every requested cell centre, including ghosts. There is no
extrapolation, puncture regularisation, or automatic resolution certification.
