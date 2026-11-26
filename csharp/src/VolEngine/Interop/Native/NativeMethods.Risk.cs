using System.Runtime.InteropServices;

namespace VolEngine.Interop;

internal static unsafe partial class NativeMethods
{
    [LibraryImport(LibraryName)]
    internal static partial int ve_risk_empirical(PanelHandle returns, double* weights, int kind, double confidence,
                                                  double* var, double* es);

    [LibraryImport(LibraryName)]
    internal static partial int ve_risk_parametric(int dist, double skew, double shape, double ghlambda, double mean,
                                                   double sigma, double confidence, double* var, double* es);
}
