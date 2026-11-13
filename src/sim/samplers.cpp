#include "sim/samplers.hpp"

#include <cmath>
#include <numbers>
#include <variant>

namespace ve::sim
{

namespace
{

// log of x^(lambda - 1) exp(-omega (x + 1 / x) / 2), the GIG of Hormann and Leydold with chi = psi = omega.
double log_kernel(double x, double lambda, double omega)
{
    return (lambda - 1.0) * std::log(x) - 0.5 * omega * (x + 1.0 / x);
}

// The mode of the kernel, written for lambda < 1 without the cancellation of (lambda - 1) + sqrt(...).
double kernel_mode(double lambda, double omega)
{
    if (lambda >= 1.0)
    {
        return (lambda - 1.0 + std::sqrt((lambda - 1.0) * (lambda - 1.0) + omega * omega)) / omega;
    }
    return omega / (1.0 - lambda + std::sqrt((1.0 - lambda) * (1.0 - lambda) + omega * omega));
}

// Ratio of uniforms about the origin: (u, v) uniform in [0, u+] x [0, 1], x = u / v, accepted when
// v^2 <= g(x) / g(mode). u+ = max x sqrt(g(x) / g(mode)), at x = (1 + lambda + sqrt((1 + lambda)^2 + omega^2)) / omega.
double rou_without_shift(Stream& stream, double lambda, double omega)
{
    double mode = kernel_mode(lambda, omega);
    double top = log_kernel(mode, lambda, omega);
    double xu = (1.0 + lambda + std::sqrt((1.0 + lambda) * (1.0 + lambda) + omega * omega)) / omega;
    double u_plus = xu * std::exp(0.5 * (log_kernel(xu, lambda, omega) - top));
    for (;;)
    {
        double u = u_plus * stream.uniform();
        double v = stream.uniform();
        double x = u / v;
        if (2.0 * std::log(v) <= log_kernel(x, lambda, omega) - top)
        {
            return x;
        }
    }
}

struct ShiftBounds
{
    double lower;
    double upper;
};

// The extremes of (x - mode) sqrt(g(x) / g(mode)): the roots of x^3 + a x^2 + b x + c with a = -(mode + 2 (lambda +
// 1) / omega), b = 2 (lambda - 1) mode / omega - 1, c = mode (Dagpunar 1989), the largest above the mode and the
// middle one below it, by the trigonometric form of Cardano.
ShiftBounds shift_bounds(double lambda, double omega, double mode, double top)
{
    double a = -(mode + 2.0 * (lambda + 1.0) / omega);
    double b = 2.0 * (lambda - 1.0) * mode / omega - 1.0;
    double c = mode;
    double p = b - a * a / 3.0;
    double q = 2.0 * a * a * a / 27.0 - a * b / 3.0 + c;
    double angle = std::acos(-q / (2.0 * std::sqrt(-(p * p * p) / 27.0)));
    double radius = 2.0 * std::sqrt(-p / 3.0);
    double above = radius * std::cos(angle / 3.0) - a / 3.0;
    double below = radius * std::cos(angle / 3.0 + 4.0 * std::numbers::pi / 3.0) - a / 3.0;
    auto bound = [&](double x) { return (x - mode) * std::exp(0.5 * (log_kernel(x, lambda, omega) - top)); };
    return {bound(below), bound(above)};
}

// Ratio of uniforms about the mode, for lambda > 2 or omega > 3: x = u / v + mode.
double rou_with_shift(Stream& stream, double lambda, double omega)
{
    double mode = kernel_mode(lambda, omega);
    double top = log_kernel(mode, lambda, omega);
    ShiftBounds bounds = shift_bounds(lambda, omega, mode, top);
    for (;;)
    {
        double u = bounds.lower + (bounds.upper - bounds.lower) * stream.uniform();
        double v = stream.uniform();
        double x = u / v + mode;
        if (x > 0.0 && 2.0 * std::log(v) <= log_kernel(x, lambda, omega) - top)
        {
            return x;
        }
    }
}

// The dominating function of the non-T-concave case (0 <= lambda < 1, small omega): g(mode) on (0, x0),
// exp(-omega) x^(lambda - 1) on [x0, 2 / omega) and x*^(lambda - 1) exp(-omega x / 2) on [x*, inf), with
// x0 = omega / (1 - lambda) and x* = max(x0, 2 / omega).
struct Hat
{
    double x0;
    double xstar;
    double k1;
    double k2;
    double k3;
    double a1;
    double a2;
    double a3;
};

Hat hat(double lambda, double omega)
{
    Hat h{};
    h.x0 = omega / (1.0 - lambda);
    h.xstar = std::max(h.x0, 2.0 / omega);
    h.k1 = std::exp(log_kernel(kernel_mode(lambda, omega), lambda, omega));
    h.a1 = h.k1 * h.x0;
    if (h.x0 < 2.0 / omega)
    {
        h.k2 = std::exp(-omega);
        h.a2 = lambda == 0.0 ? h.k2 * std::log(2.0 / (omega * omega))
                             : h.k2 * (std::pow(2.0 / omega, lambda) - std::pow(h.x0, lambda)) / lambda;
    }
    h.k3 = std::pow(h.xstar, lambda - 1.0);
    h.a3 = 2.0 * h.k3 * std::exp(-0.5 * h.xstar * omega) / omega;
    return h;
}

struct Candidate
{
    double x;
    double height;
};

Candidate hat_draw(const Hat& h, double lambda, double omega, double area)
{
    if (area <= h.a1)
    {
        return {h.x0 * area / h.a1, h.k1};
    }
    if (area <= h.a1 + h.a2)
    {
        double rest = area - h.a1;
        double x = lambda == 0.0 ? omega * std::exp(rest * std::exp(omega))
                                 : std::pow(std::pow(h.x0, lambda) + lambda * rest / h.k2, 1.0 / lambda);
        return {x, h.k2 * std::pow(x, lambda - 1.0)};
    }
    double rest = area - h.a1 - h.a2;
    double x = -2.0 / omega * std::log(std::exp(-0.5 * h.xstar * omega) - rest * omega / (2.0 * h.k3));
    return {x, h.k3 * std::exp(-0.5 * x * omega)};
}

double non_t_concave(Stream& stream, double lambda, double omega)
{
    Hat h = hat(lambda, omega);
    double total = h.a1 + h.a2 + h.a3;
    for (;;)
    {
        Candidate c = hat_draw(h, lambda, omega, total * stream.uniform());
        if (stream.uniform() * c.height <= std::exp(log_kernel(c.x, lambda, omega)))
        {
            return c.x;
        }
    }
}

// The GIG with chi = psi = omega and lambda >= 0.
double standard_gig(Stream& stream, double lambda, double omega)
{
    if (lambda > 2.0 || omega > 3.0)
    {
        return rou_with_shift(stream, lambda, omega);
    }
    if (lambda >= 1.0 - 2.25 * omega * omega || omega > 0.2)
    {
        return rou_without_shift(stream, lambda, omega);
    }
    return non_t_concave(stream, lambda, omega);
}

// norm, std (N sqrt((nu - 2) / (2 G)), G ~ Gamma(nu / 2)) and ged (|z| = lambda (2 G)^(1 / nu), G ~ Gamma(1 / nu)).
double draw_symmetric(const dist::LogDensity<double>& density, Stream& stream)
{
    if (density.dist == Dist::Std)
    {
        double nu = density.scale + 2.0;
        return normal(stream) * std::sqrt(density.scale / (2.0 * gamma(stream, 0.5 * nu)));
    }
    if (density.dist == Dist::Ged)
    {
        double magnitude = density.scale * std::pow(2.0 * gamma(stream, 1.0 / density.power), 1.0 / density.power);
        return stream.uniform() < 0.5 ? -magnitude : magnitude;
    }
    return normal(stream);
}

double draw_from(const dist::LogDensity<double>& density, Stream& stream)
{
    return draw_symmetric(density, stream);
}

// Fernandez-Steel: y = xi |x| with probability xi^2 / (1 + xi^2), else -|x| / xi, and z = (y - shift) / stretch.
double draw_from(const dist::SkewedLogDensity<double>& density, Stream& stream)
{
    double x = std::abs(draw_symmetric(density.kernel, stream));
    double xi = density.skew;
    double y = stream.uniform() < xi * xi / (1.0 + xi * xi) ? xi * x : -x / xi;
    return (y - density.shift) / density.stretch;
}

// z = location + beta W + sqrt(W) N, beta = rho alpha, W ~ GIG(order, delta^2, alpha^2 - beta^2).
double draw_from(const dist::HyperbolicLogDensity<double>& density, Stream& stream)
{
    double beta = density.rho * density.alpha;
    double psi = density.alpha * density.cosine * (density.alpha * density.cosine);
    double w = gig(stream, density.order, density.delta * density.delta, psi);
    return density.location + beta * w + std::sqrt(w) * normal(stream);
}

// z = center + scale sinh((N + skew) / shape).
double draw_from(const dist::JohnsonLogDensity<double>& density, Stream& stream)
{
    return density.center + density.scale * std::sinh((normal(stream) + density.skew) / density.shape);
}

}

double normal(Stream& stream)
{
    double radius = std::sqrt(-2.0 * std::log(stream.uniform()));
    return radius * std::cos(2.0 * std::numbers::pi * stream.uniform());
}

double gamma(Stream& stream, double shape)
{
    if (shape < 1.0)
    {
        double boost = std::pow(stream.uniform(), 1.0 / shape);
        return gamma(stream, shape + 1.0) * boost;
    }
    double d = shape - 1.0 / 3.0;
    double c = 1.0 / std::sqrt(9.0 * d);
    for (;;)
    {
        double x = normal(stream);
        double v = 1.0 + c * x;
        if (v <= 0.0)
        {
            continue;
        }
        v = v * v * v;
        if (std::log(stream.uniform()) < 0.5 * x * x + d - d * v + d * std::log(v))
        {
            return d * v;
        }
    }
}

double gig(Stream& stream, double lambda, double chi, double psi)
{
    double omega = std::sqrt(chi * psi);
    double scale = std::sqrt(chi / psi);
    if (lambda < 0.0)
    {
        return scale / standard_gig(stream, -lambda, omega);
    }
    return scale * standard_gig(stream, lambda, omega);
}

Sampler::Sampler(Dist dist, const dist::ShapeParameters<double>& parameters)
    : density_(dist::make_density(dist, parameters))
{
}

double Sampler::draw(Stream& stream) const
{
    return std::visit([&stream](const auto& density) { return draw_from(density, stream); }, density_);
}

}
