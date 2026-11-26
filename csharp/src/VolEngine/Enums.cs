namespace VolEngine;

/// <summary>The status a native call returns; every value other than <see cref="Ok"/> is thrown as an exception that carries it.</summary>
public enum Status
{
    /// <summary>The call succeeded.</summary>
    Ok,
    /// <summary>An argument was rejected; the message says which and why.</summary>
    InvalidArgument,
    /// <summary>A handle was missing or of the wrong kind.</summary>
    InvalidHandle,
    /// <summary>The operation does not apply to this model or option.</summary>
    Unsupported,
    /// <summary>The native library could not allocate memory.</summary>
    OutOfMemory,
    /// <summary>An unexpected failure inside the native library.</summary>
    InternalError,
    /// <summary>The call stopped on a cancellation request; the binding throws it as an
    /// <see cref="OperationCanceledException"/>, not a <see cref="VolEngineException"/>.</summary>
    Cancelled,
}

/// <summary>
/// The conditional variance model. Each fGARCH submodel is its own value, as in the C interface: the omitted
/// eta blocks and the power lambda follow the submodel (FGarchGarch has lambda = 2 fixed, FGarchTGarch lambda = 1,
/// FGarchNGarch and FGarchAparch lambda free, ...).
/// </summary>
public enum Model
{
    /// <summary>Standard GARCH.</summary>
    SGarch,
    /// <summary>Exponential GARCH (gamma1..gammap).</summary>
    EGarch,
    /// <summary>GJR-GARCH (gamma1..gammap).</summary>
    GjrGarch,
    /// <summary>Asymmetric power ARCH (gamma1..gammap, delta).</summary>
    Aparch,
    /// <summary>Integrated GARCH: beta_q is derived as 1 - sum alpha - the other betas.</summary>
    IGarch,
    /// <summary>Component GARCH (eta11 = rho, eta21 = phi).</summary>
    CsGarch,
    /// <summary>fGARCH GARCH submodel: no extra block, lambda = 2 fixed.</summary>
    FGarchGarch,
    /// <summary>fGARCH TGARCH submodel: eta11..eta1p, lambda = 1 fixed.</summary>
    FGarchTGarch,
    /// <summary>fGARCH AVGARCH submodel: eta11..eta1p, eta21..eta2p, lambda = 1 fixed.</summary>
    FGarchAvGarch,
    /// <summary>fGARCH NGARCH submodel: lambda free.</summary>
    FGarchNGarch,
    /// <summary>fGARCH NAGARCH submodel: eta21..eta2p, lambda = 2 fixed.</summary>
    FGarchNaGarch,
    /// <summary>fGARCH APARCH submodel: eta11..eta1p, lambda free.</summary>
    FGarchAparch,
    /// <summary>fGARCH GJRGARCH submodel: eta11..eta1p, lambda = 2 fixed.</summary>
    FGarchGjrGarch,
    /// <summary>fGARCH ALLGARCH submodel: eta11..eta1p, eta21..eta2p, lambda free.</summary>
    FGarchAllGarch,
    /// <summary>
    /// RiskMetrics' EWMA, sigma2_t = lambda sigma2_(t-1) + (1 - lambda) e_(t-1)^2: the iGARCH(1, 1) with omega = 0.
    /// omega and alpha1 are derived (0 and 1 - lambda); lambda in [0, 1] is estimated unless fixed (RiskMetrics fixes
    /// it at 0.94). Order (1, 1) only; variance targeting and simulating from the spec are rejected.
    /// </summary>
    Ewma,
}

/// <summary>
/// The innovation distribution, standardized to zero mean and unit variance. Distribution parameters come last, in the
/// order skew, shape, ghlambda.
/// </summary>
public enum Distribution
{
    /// <summary>Normal.</summary>
    Norm,
    /// <summary>Student t; shape nu &gt; 2.</summary>
    Std,
    /// <summary>Generalized error; shape nu &gt; 0.</summary>
    Ged,
    /// <summary>Fernandez-Steel skewed Student t; skew xi &gt; 0, shape nu &gt; 2.</summary>
    Sstd,
    /// <summary>Fernandez-Steel skewed GED; skew xi &gt; 0, shape nu &gt; 0.</summary>
    Sged,
    /// <summary>Normal inverse Gaussian (ghyp with lambda = -1/2); skew rho in (-1, 1), shape zeta in (0, 1e6].</summary>
    Nig,
    /// <summary>Generalized hyperbolic; skew rho in (-1, 1), shape zeta in (0, 1e6], ghlambda real.</summary>
    Ghyp,
    /// <summary>Johnson SU; skew nu real, shape tau &gt; 0.</summary>
    Jsu,
}

/// <summary>How the pre-sample variance is initialized.</summary>
public enum VarInit
{
    /// <summary>The sample mean of the squared residuals (rugarch).</summary>
    Sample,
    /// <summary>The exponential backcast of arch.</summary>
    Backcast,
}

/// <summary>The outcome of a fit. Check it before using the values.</summary>
public enum FitStatus
{
    /// <summary>The optimizer converged.</summary>
    Converged,
    /// <summary>The optimizer stopped at the iteration limit.</summary>
    MaxIter,
    /// <summary>
    /// No usable optimum: the values are those of the best finite point reached, or NaN with empty series when there
    /// was none.
    /// </summary>
    Failed,
    /// <summary>Too few observations: NaN values and empty series.</summary>
    InsufficientData,
}

/// <summary>The covariance behind standard errors, t values and p values.</summary>
public enum StandardErrorKind
{
    /// <summary>The inverse Hessian.</summary>
    Hessian,
    /// <summary>The QML sandwich.</summary>
    Robust,
}

/// <summary>The second-stage correlation model.</summary>
public enum DccModel
{
    /// <summary>DCC (Engle).</summary>
    Dcc,
    /// <summary>Asymmetric DCC.</summary>
    Adcc,
    /// <summary>Flexible (grouped) DCC.</summary>
    Fdcc,
    /// <summary>Constant correlation; needs a copula.</summary>
    Static,
}

/// <summary>The multivariate distribution of a DCC.</summary>
public enum DccDistribution
{
    /// <summary>Multivariate normal.</summary>
    MvNorm,
    /// <summary>Multivariate Student t (mshape).</summary>
    MvT,
    /// <summary>Multivariate Laplace.</summary>
    MvLaplace,
}

/// <summary>How a Copula-DCC turns the margins into uniforms.</summary>
public enum Copula
{
    /// <summary>No copula: the plain DCC.</summary>
    None,
    /// <summary>Each fit's own cdf.</summary>
    Parametric,
    /// <summary>Ranks.</summary>
    Empirical,
}

/// <summary>The estimator of a static copula correlation.</summary>
public enum StaticMethod
{
    /// <summary>The library default (Kendall).</summary>
    Default,
    /// <summary>Kendall's tau.</summary>
    Kendall,
    /// <summary>Maximum likelihood (C1.. are the free entries of A, R = cov2cor(A A')).</summary>
    Ml,
}

internal enum Criterion
{
    Aic,
    Bic,
    Hqic,
    Sic,
}

internal enum CriterionScale
{
    PerObs,
    Total,
}
