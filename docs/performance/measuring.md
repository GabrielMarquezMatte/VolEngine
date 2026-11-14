# Measuring

How to answer, in seconds, where the time goes and whether a change helped. The official numbers in
[Benchmarks](benchmarks.md) come from the generic `gcc-bench` build (no `-march=native`), because the shipped
binaries are generic.

Run every executable with `/c/msys64/ucrt64/bin` first on `PATH` (Git's mingw64 `libstdc++` lacks symbols):

```
export PATH=/c/msys64/ucrt64/bin:$PATH
cmake --preset gcc-bench
cmake --build --preset gcc-bench
```

## Quick suite (about 35 s)

```
python tests/benchmarks/quick.py run --out build/gcc-bench/quick.json
```

Three repetitions of a fixed filter (`QUICK_FILTER` in `tests/benchmarks/quick.py`):
- `BM_Loglik` (`double` and `Dual`) of four plan 2a cases;
- `BM_FitOne` of five cases, each with `iterations` and `evaluations` counters;
- `BM_TccGridSample/threads:1`: 18 fits (one per order and ARMA of the TCC grid, the distributions in rotation)
  on one series, about 1 s.

## A/B

Measure before and after in the same session, one right after the other. On this machine, runs on different days
move some medians by more than their spread, so never compare against an old table.

```
python tests/benchmarks/quick.py run --out build/gcc-bench/quick_a.json
# change, rebuild
python tests/benchmarks/quick.py run --out build/gcc-bench/quick_b.json
python tests/benchmarks/quick.py compare build/gcc-bench/quick_a.json build/gcc-bench/quick_b.json
```

`compare` prints the medians, the ratio B/A and `noise` when |B − A| is within the larger standard deviation.

## Grid profile (about 11 s for the full grid)

```
build/gcc-bench/volengine_grid_profile [--series N] [--threads N] [--filter TEXT] [--standard-errors on|off] [--json PATH]
```

Runs the 1080-fit grid of `BM_TccGrid` (or a slice: `--series 1..20`, `--filter` on labels such as
`sgarch_p3q3_arma11_std`) once through the library's batch path. It prints:
- `wall_s`, the sum of the fits' own times, and `pool_use` = sum / (threads × wall);
- tables by spec, distribution, order and ARMA, with iterations and evaluations per fit, evaluations per
  iteration, the optimizer's share of the fit time, and the status counts.

With 16 threads the per-fit times include the slowdown of two threads per core, so `fit_s_sum` exceeds the
one-thread total.

Compare two profiles, for example before and after an optimizer change:

```
python tests/benchmarks/quick.py compare-profile build/gcc-bench/profile_a.json build/gcc-bench/profile_b.json
```

It prints the time and evaluation ratios, the worst relative log-likelihood change per fit and which fit it is,
and the status counts of each side. A change to the optimizer must not lower any log-likelihood beyond 1e-9
relative.

## Sampling profile (AMD uProf)

AMD uProf on Windows does not symbolize the MinGW-gcc build: `AMDuProfCLI report` crashes on a session of a DWARF
binary, and GCC's `-gcodeview` fails in the assembler. The `msvc-profile` preset builds the grid profile with PDB
symbols, and serves only to locate hot spots. MSVC code is slower than GCC's (20 s against 7.7 s on the same
two-series slice), so read the shares as indicative; how much a change gains always comes from `gcc-bench`.

```
cmake --preset msvc-profile
cmake --build --preset msvc-profile --target volengine_grid_profile
"C:/Program Files/AMD/AMDuProf/bin/AMDuProfCLI.exe" collect --config tbp -o build/uprof build/msvc-profile/Release/volengine_grid_profile.exe --series 2 --threads 1
"C:/Program Files/AMD/AMDuProf/bin/AMDuProfCLI.exe" report -i build/uprof/<session> --bin-path <abs path to build/msvc-profile/Release> --symbol-path <same>
```

The report is `report.csv` in the session directory; without `--bin-path` and `--symbol-path` the functions show
as bare addresses.
