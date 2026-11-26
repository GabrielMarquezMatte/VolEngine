#pragma once

#include "ad/dual.hpp"
#include "core/spec.hpp"
#include "dist/densities.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <span>
#include <variant>

namespace ve::dist
{

namespace detail
{

constexpr std::size_t kGaussLegendreNodes = 64;

struct QuadratureNode
{
    double t;
    double weight;
};

// The Gauss-Legendre rule on [0, 1].
const std::array<QuadratureNode, kGaussLegendreNodes>& gauss_legendre();

// norm, std and ged keep their closed forms.
inline bool is_symmetric(Dist dist)
{
    return dist == Dist::Norm || dist == Dist::Std || dist == Dist::Ged;
}

// The sstd has the tail of the t.
inline bool has_student_tail(Dist dist)
{
    return dist == Dist::Std || dist == Dist::Sstd;
}

template <class T>
bool in_domain_of(Dist dist, const ShapeParameters<T>& parameters)
{
    ShapeParameters<double> values{ad::value_of(parameters.skew), ad::value_of(parameters.shape),
                                   ad::value_of(parameters.ghlambda)};
    return in_domain(dist, values);
}

template <class T>
bool infinite_moment(Dist dist, const ShapeParameters<T>& parameters, const T& power)
{
    return has_student_tail(dist) && ad::value_of(parameters.shape) <= ad::value_of(power);
}

// exp(log_value), exactly 0 with no derivative where log_value is -inf: a ged with a large shape has 0.5 (|z| /
// scale)^shape overflow in the far tails the quadrature visits, and exp(-inf) times the infinite slope of that power in
// the shape is NaN, not the 0 that is the derivative of a density that is 0 there.
template <class T>
T exp_or_zero(const T& log_value)
{
    using std::exp;
    if (ad::value_of(log_value) == -std::numeric_limits<double>::infinity())
    {
        return T{0.0};
    }
    return exp(log_value);
}

template <class T>
T density_at(const Density<T>& density, const T& z)
{
    return exp_or_zero(log_density(density, z));
}

// (|z - eta2| - eta1 (z - eta2))^power f(z).
template <class T>
struct ShockPowerIntegrand
{
    Density<T> density;
    T eta1;
    T eta2;
    T power;

    T operator()(const T& z) const
    {
        using std::abs;
        using std::pow;
        T shift = z - eta2;
        return pow(abs(shift) - eta1 * shift, power) * density_at(density, z);
    }

    // The log of the integrand, -inf where it is 0 (on the side that |eta1| = 1 cancels).
    T log_value(const T& z) const
    {
        using std::abs;
        using std::log;
        T shift = z - eta2;
        T base = abs(shift) - eta1 * shift;
        if (!(ad::value_of(base) > 0.0))
        {
            return T{-std::numeric_limits<double>::infinity()};
        }
        return power * log(base) + log_density(density, z);
    }
};

// f(z).
template <class T>
struct DensityIntegrand
{
    Density<T> density;

    T operator()(const T& z) const
    {
        return density_at(density, z);
    }

    T log_value(const T& z) const
    {
        return log_density(density, z);
    }
};

// weight times the integrand at z, taken from logs where the integrand itself is lost but the product is not: where it
// is not finite (|z - eta2|^power overflowed to inf against an f(z) that underflowed to 0), and where it is 0 on a map
// that reaches far (exponent above 1), whose far nodes of a std or sstd tail have an f(z) that underflows while
// |z - eta2|^power and the weight are as large as its inverse. Under the other maps the density falls exponentially,
// and a node where it underflows holds nothing.
template <class T, class Integrand>
T weighted_value(const Integrand& integrand, const T& z, double weight, bool reaches_far)
{
    T value = integrand(z);
    double plain = ad::value_of(value);
    bool lost = !std::isfinite(plain) || (plain == 0.0 && reaches_far);
    if (!lost)
    {
        return weight * value;
    }
    return exp_or_zero(integrand.log_value(z) + std::log(weight));
}

// int over [edge + 1, inf) (direction 1) or (-inf, edge - 1] (direction -1), through
// z = edge + direction (1 - t)^-exponent.
template <class T, class Integrand>
T power_tail_integral(const Integrand& integrand, const T& edge, double direction, double exponent)
{
    T sum{0.0};
    for (const QuadratureNode& node : gauss_legendre())
    {
        double complement = 1.0 - node.t;
        double distance = std::pow(complement, -exponent);
        double jacobian = exponent * distance / complement;
        sum = sum + weighted_value(integrand, edge + direction * distance, node.weight * jacobian, exponent > 1.0);
    }
    return sum;
}

// int over the distances [e^lower, e^upper] from edge (direction 1, or their mirror), through
// z = edge + direction e^w, w in [lower, upper]. The terms are summed from logs, so that a power that overflows, or a
// density that underflows, far out keeps its node.
template <class T, class Integrand>
T logarithmic_segment(const Integrand& integrand, const T& edge, double direction, double lower, double upper)
{
    double width = upper - lower;
    T sum{0.0};
    for (const QuadratureNode& node : gauss_legendre())
    {
        double log_distance = lower + width * node.t;
        double log_weight = std::log(node.weight * width) + log_distance;
        sum = sum + exp_or_zero(integrand.log_value(edge + direction * std::exp(log_distance)) + log_weight);
    }
    return sum;
}

// int beyond z of an integrand g that is |z|^-(1 + decay) times a constant, to a relative O(1 / |z|):
// |z| g(z) / decay.
template <class T, class Integrand>
T power_law_beyond(const Integrand& integrand, const T& z, const T& decay)
{
    using std::abs;
    using std::log;
    return exp_or_zero(log(abs(z)) + integrand.log_value(z)) / decay;
}

// How tail_integral integrates one tail: through z = edge + direction (1 - t)^-exponent while reach is 0, else
// through z = edge + direction e^w out to reach; with a decay above 0 (a std or sstd tail), in pieces of w and with
// the power law |z|^-(1 + decay) beyond reach.
template <class T>
struct TailMap
{
    double exponent = 1.0;
    double reach = 0.0;
    T decay{0.0};
};

template <class T>
struct TailMaps
{
    TailMap<T> below;
    TailMap<T> above;
};

// The pieces of w are [0, 1], [1, 2], [2, 4], ...: |z|^power f(z) turns from the core to the power law at
// |z| ~ sqrt((nu - 2) power), within ~1 / sqrt(power) in w, which one piece out to w ~ 46 would leave to a few nodes.
template <class T, class Integrand>
T student_reach_integral(const Integrand& integrand, const T& edge, double direction, const TailMap<T>& map)
{
    double end = std::log(map.reach);
    T sum = logarithmic_segment(integrand, edge, direction, 0.0, 1.0);
    for (double lower = 1.0; lower < end; lower *= 2.0)
    {
        sum = sum + logarithmic_segment(integrand, edge, direction, lower, std::min(2.0 * lower, end));
    }
    return sum + power_law_beyond(integrand, edge + direction * map.reach, map.decay);
}

template <class T, class Integrand>
T tail_integral(const Integrand& integrand, const T& edge, double direction, const TailMap<T>& map)
{
    if (!(map.reach > 0.0))
    {
        return power_tail_integral(integrand, edge, direction, map.exponent);
    }
    if (!(ad::value_of(map.decay) > 0.0))
    {
        return logarithmic_segment(integrand, edge, direction, 0.0, std::log(map.reach));
    }
    return student_reach_integral(integrand, edge, direction, map);
}

// A zero-width segment is skipped: its nodes would all sit on the cusp, where the slope of a power below 1 is
// infinite.
template <class T, class Integrand>
T segment_integral(const Integrand& integrand, const T& lower, const T& upper)
{
    T width = upper - lower;
    if (ad::value_of(width) == 0.0)
    {
        return T{0.0};
    }
    T sum{0.0};
    for (const QuadratureNode& node : gauss_legendre())
    {
        sum = sum + node.weight * integrand(lower + width * node.t);
    }
    return width * sum;
}

// A std or sstd integrand decays as |z|^(power - nu - 1), which z = edge + (1 - t)^-k turns into
// (1 - t)^(k (nu - power) - 1): smooth enough at t = 1 once k (nu - power) >= 2. The last node (1 - t ~ 3.5e-4) sits
// at z ~ e^(8 k), and past k (power + 1) = kMaxStudentTailSpan its |z|^(power + 1) nears the top of the double range
// (e^709), while as nu nears power a growing share of the moment lies beyond any double. Such a tail goes on the
// logarithmic map out to kStudentReach, and beyond it is the integral of the power law, which holds there to a
// relative 1 / kStudentReach.
constexpr double kMaxStudentTailSpan = 80.0;
constexpr double kStudentReach = 1e20;

// An sged or ged integrand decays as exp(-a |z|^nu), which k = 1 / nu turns into exp(-a (1 - t)^-1), flat at
// t = 1, where a nu below 1 would leave mass beyond the last node. k stops at kMaxSgedTailExponent, where the
// last node (1 - t ~ 3.5e-4) sits at z ~ 1e34 and its powers stay finite.
constexpr double kMaxSgedTailExponent = 10.0;

// A nig or ghyp tail is ~|x|^(lambda - 1) exp(-|x| / L), with the decay length L = 1 / (alpha (1 - rho)) above and
// 1 / (alpha (1 + rho)) below, and z = edge + (1 - t)^-1 resolves an L up to 1. A longer one (a small zeta) goes on
// the logarithmic map, out to kHyperbolicDecays + 2 max(0, lambda - 1 + power) decay lengths, beyond which
// |x|^(lambda - 1 + power) exp(-|x| / L) has a share of its integral below 1e-17.
constexpr double kHyperbolicDecays = 50.0;

template <class T>
TailMap<T> student_tail(const T& shape, const T& power)
{
    T decay = shape - power;
    double exponent = std::max(1.0, 2.0 / ad::value_of(decay));
    if (exponent * (ad::value_of(power) + 1.0) > kMaxStudentTailSpan)
    {
        return {1.0, kStudentReach, decay};
    }
    return {exponent};
}

template <class T>
TailMap<T> hyperbolic_tail(const HyperbolicLogDensity<T>& density, double direction, const T& power)
{
    double decay_length = 1.0 / (ad::value_of(density.alpha) * (1.0 - direction * ad::value_of(density.rho)));
    if (!(decay_length > 1.0))
    {
        return {};
    }
    double polynomial = std::max(0.0, ad::value_of(density.order) - 1.0 + ad::value_of(power));
    return {1.0, decay_length * (kHyperbolicDecays + 2.0 * polynomial)};
}

template <class T>
TailMaps<T> tail_maps(Dist dist, const ShapeParameters<T>& parameters, const Density<T>& density, const T& power)
{
    if (has_student_tail(dist))
    {
        TailMap<T> tail = student_tail(parameters.shape, power);
        return {tail, tail};
    }
    if (dist == Dist::Sged || dist == Dist::Ged)
    {
        TailMap<T> tail{std::clamp(1.0 / ad::value_of(parameters.shape), 1.0, kMaxSgedTailExponent)};
        return {tail, tail};
    }
    const HyperbolicLogDensity<T>* hyperbolic = std::get_if<HyperbolicLogDensity<T>>(&density);
    if (hyperbolic == nullptr)
    {
        return {};
    }
    return {hyperbolic_tail(*hyperbolic, -1.0, power), hyperbolic_tail(*hyperbolic, 1.0, power)};
}

// The narrow side of a Fernandez-Steel density holds its mass within a few z-scales of the kink, and that
// scale falls as 1 / xi^2 with the skew. A side whose scale is below kRefinedBelow gets split points at
// scale 4^k from the kink, out to kRefinedBelow; at most kMaxRefinements of them, so a scale below
// 0.5 4^-12 (xi ~ 6000) starts with a first piece wider than the scale.
constexpr std::size_t kMaxRefinements = 12;
constexpr double kRefinedBelow = 0.5;
constexpr double kRefinementRatio = 4.0;

// A ged or sged kernel exp(-|y / scale|^nu / 2) with a large nu is flat to |y| ~ scale and falls within ~scale / nu of
// it, an edge no piece of unit scale resolves: the quadrature missed 3e-6 of a moment at nu = 20 and 1.5% at nu = 1000.
// From kEdgeShapeFrom on, each edge gets split points at kEdgeOffsets edge widths (scale / nu) from it, which span the
// zone where |y / scale|^nu goes from e^-8 to e^8 (the density from flat to e^-1490). The points are values only, as
// every split point that is not a kink or the cusp: the density is smooth there.
constexpr double kEdgeShapeFrom = 6.0;
constexpr std::array<double, 9> kEdgeOffsets{-8.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0};

// The z-scale of the density on each side of its kink: 1 / (xi sigma) below and xi / sigma above for sstd and
// sged, 1 for the others. They are scaled by the width of the kernel's core when below 1: lambda for ged and sged,
// where a small nu puts the core within lambda << 1 of the kink, and sqrt(nu - 2) for std and sstd.
struct SideScales
{
    double below;
    double above;
};

template <class T>
double kernel_core(const LogDensity<T>& kernel)
{
    if (kernel.dist == Dist::Ged)
    {
        return std::min(1.0, ad::value_of(kernel.scale));
    }
    if (kernel.dist == Dist::Std)
    {
        return std::min(1.0, std::sqrt(ad::value_of(kernel.scale)));
    }
    return 1.0;
}

template <class T>
SideScales side_scales(const Density<T>& density)
{
    const LogDensity<T>* symmetric = std::get_if<LogDensity<T>>(&density);
    if (symmetric != nullptr)
    {
        double core = kernel_core(*symmetric);
        return {core, core};
    }
    const SkewedLogDensity<T>* skewed = std::get_if<SkewedLogDensity<T>>(&density);
    if (skewed == nullptr)
    {
        return {1.0, 1.0};
    }
    double skew = ad::value_of(skewed->skew);
    double stretch = ad::value_of(skewed->stretch);
    double core = kernel_core(skewed->kernel);
    return {core / (skew * stretch), core * skew / stretch};
}

// The split points of one quadrature.
template <class T>
struct Breakpoints
{
    std::array<T, 2 * kMaxRefinements + 3 + 2 * kEdgeOffsets.size()> points{};
    std::size_t count = 0;

    void add(const T& point)
    {
        points[count] = point;
        ++count;
    }

    void sort()
    {
        std::sort(points.begin(), points.begin() + static_cast<std::ptrdiff_t>(count),
                  [](const T& a, const T& b) { return ad::value_of(a) < ad::value_of(b); });
    }

    std::span<const T> view() const
    {
        return {points.data(), count};
    }
};

template <class T>
void add_refinements(Breakpoints<T>& breakpoints, const T& kink, double direction, double scale)
{
    double smallest = kRefinedBelow * std::pow(kRefinementRatio, -static_cast<double>(kMaxRefinements));
    for (double distance = std::max(scale, smallest); distance < kRefinedBelow; distance *= kRefinementRatio)
    {
        breakpoints.add(kink + direction * distance);
    }
}

// A nig or ghyp with delta below kRefinedBelow peaks within a few delta of its location, which gets a split
// point and the refinements of a kink at the scale delta. As |rho| nears 1 the minimum of g, at the offset
// rho delta / cosine from the location (see hyperbolic.hpp), moves away from it; once that offset reaches
// kRefinedBelow the peak at the location has either widened or been suppressed by exp(-alpha g), and the mass
// lies about the minimum, which the refinements follow instead. The points are values only: the density is
// smooth there.
template <class T>
void add_location_refinements(Breakpoints<T>& breakpoints, const Density<T>& density)
{
    const HyperbolicLogDensity<T>* hyperbolic = std::get_if<HyperbolicLogDensity<T>>(&density);
    if (hyperbolic == nullptr)
    {
        return;
    }
    double delta = ad::value_of(hyperbolic->delta);
    if (!(delta < kRefinedBelow))
    {
        return;
    }
    double center = ad::value_of(hyperbolic->location);
    double minimum_offset = ad::value_of(hyperbolic->rho) * delta / ad::value_of(hyperbolic->cosine);
    if (std::fabs(minimum_offset) >= kRefinedBelow)
    {
        center += minimum_offset;
    }
    T point{center};
    breakpoints.add(point);
    add_refinements(breakpoints, point, -1.0, delta);
    add_refinements(breakpoints, point, 1.0, delta);
}

template <class T>
void add_edge_points(Breakpoints<T>& breakpoints, double position, double width)
{
    for (double offset : kEdgeOffsets)
    {
        breakpoints.add(T{position + offset * width});
    }
}

// The edges of a ged (at +-scale) or of the kernel of an sged (at -scale / (skew stretch) and scale skew / stretch from
// the kink), when its shape is above kEdgeShapeFrom.
template <class T>
void add_edge_refinements(Breakpoints<T>& breakpoints, const Density<T>& density, const T& kink)
{
    const LogDensity<T>* kernel = std::get_if<LogDensity<T>>(&density);
    double below = 1.0;
    double above = 1.0;
    if (kernel == nullptr)
    {
        const SkewedLogDensity<T>* skewed = std::get_if<SkewedLogDensity<T>>(&density);
        if (skewed == nullptr)
        {
            return;
        }
        kernel = &skewed->kernel;
        double skew = ad::value_of(skewed->skew);
        double stretch = ad::value_of(skewed->stretch);
        below = 1.0 / (skew * stretch);
        above = skew / stretch;
    }
    double shape = ad::value_of(kernel->power);
    if (kernel->dist != Dist::Ged || !(shape > kEdgeShapeFrom))
    {
        return;
    }
    double scale = ad::value_of(kernel->scale);
    double center = ad::value_of(kink);
    add_edge_points(breakpoints, center - scale * below, scale * below / shape);
    add_edge_points(breakpoints, center + scale * above, scale * above / shape);
}

template <class T>
void add_side_refinements(Breakpoints<T>& breakpoints, const Density<T>& density, const T& kink)
{
    SideScales scales = side_scales(density);
    add_refinements(breakpoints, kink, -1.0, scales.below);
    add_refinements(breakpoints, kink, 1.0, scales.above);
    add_location_refinements(breakpoints, density);
    add_edge_refinements(breakpoints, density, kink);
}

// int over [points[0] - 1, points.back()], one piece between each pair of neighbouring points.
template <class T, class Integrand>
T breakpoint_sum(const Integrand& integrand, std::span<const T> points)
{
    T sum = segment_integral(integrand, points[0] - 1.0, points[0]);
    for (std::size_t i = 1; i < points.size(); ++i)
    {
        sum = sum + segment_integral(integrand, points[i - 1], points[i]);
    }
    return sum;
}

// The integrand has a cusp at z = eta2, and the density is not smooth at its kink (0 for ged, z* for sstd
// and sged): the quadrature is split at both, so that no piece holds a kink. With eta2 on the kink both split
// points follow eta2, so that no piece differentiates the cusp of a power below 1.
template <class T>
Breakpoints<T> shock_breakpoints(const ShockPowerIntegrand<T>& integrand)
{
    T kink = density_kink(integrand.density);
    if (ad::value_of(integrand.eta2) == ad::value_of(kink))
    {
        kink = integrand.eta2;
    }
    Breakpoints<T> breakpoints;
    breakpoints.add(kink);
    breakpoints.add(integrand.eta2);
    add_side_refinements(breakpoints, integrand.density, kink);
    breakpoints.sort();
    return breakpoints;
}

template <class T>
T shock_power_quadrature(const ShockPowerIntegrand<T>& integrand, const TailMaps<T>& tails)
{
    Breakpoints<T> breakpoints = shock_breakpoints(integrand);
    std::span<const T> points = breakpoints.view();
    T last = points.back();
    T near = breakpoint_sum(integrand, points) + segment_integral(integrand, last, last + 1.0);
    return tail_integral(integrand, points.front(), -1.0, tails.below) + near +
           tail_integral(integrand, last, 1.0, tails.above);
}

// The kink when it lies below 0, 0, and the refinements below 0.
template <class T>
Breakpoints<T> below_zero_breakpoints(const Density<T>& density)
{
    T kink = density_kink(density);
    T lower{0.0};
    if (ad::value_of(kink) < 0.0)
    {
        lower = kink;
    }
    Breakpoints<T> breakpoints;
    breakpoints.add(lower);
    breakpoints.add(T{0.0});
    add_side_refinements(breakpoints, density, kink);
    breakpoints.sort();
    while (ad::value_of(breakpoints.points[breakpoints.count - 1]) > 0.0)
    {
        --breakpoints.count;
    }
    return breakpoints;
}

// int over (-inf, 0).
template <class T>
T below_zero_quadrature(const DensityIntegrand<T>& integrand, const TailMaps<T>& tails)
{
    Breakpoints<T> breakpoints = below_zero_breakpoints(integrand.density);
    std::span<const T> points = breakpoints.view();
    return tail_integral(integrand, points.front(), -1.0, tails.below) + breakpoint_sum(integrand, points);
}

// A jsu is z = center + scale sinh((r + skew) / shape) of a standard normal r, and a small shape puts its
// mass in a peak of width ~scale, which no fixed grid in z resolves. Its moments are taken in r instead,
// where (|z - eta2| - eta1 (z - eta2))^power phi(r) is smooth but at the cusp r(eta2). For large |r| it is
// ~exp(power |r| / shape - r^2 / 2), which peaks at |r| = power / shape, so the mass lies within
// kJohnsonReach of [-power / shape, power / shape] and of the cusp (exp(-50) of the peak beyond). Pieces are
// at most kJohnsonPiece wide, where 64 nodes integrate a unit Gaussian to rounding; past kJohnsonMaxPieces
// per side (power / shape > ~118, where the moment overflows anyway) they widen.
constexpr double kJohnsonReach = 10.0;
constexpr double kJohnsonPiece = 8.0;
constexpr double kJohnsonMaxPieces = 16.0;
// sinh(u) overflows past |u| ~ 710; beyond kSinhLimit, log |center - eta2 + scale sinh(u)| is
// log(scale) + |u| - log 2 to rounding.
constexpr double kSinhLimit = 700.0;

// r(z) = shape asinh((z - center) / scale) - skew, increasing in z.
template <class T>
T johnson_normal_of(const JohnsonLogDensity<T>& density, const T& z)
{
    using std::asinh;
    return density.shape * asinh((z - density.center) / density.scale) - density.skew;
}

template <class T>
struct SignedLog
{
    T log;
    double sign;
};

// log |z(r) - eta2| and its sign, for u = (r + skew) / shape.
template <class T>
SignedLog<T> johnson_log_distance(const JohnsonLogDensity<T>& density, const T& eta2, const T& u)
{
    using std::abs;
    using std::log;
    using std::sinh;
    if (std::fabs(ad::value_of(u)) > kSinhLimit)
    {
        double sign = ad::value_of(u) < 0.0 ? -1.0 : 1.0;
        return {log(density.scale) + abs(u) - std::numbers::ln2, sign};
    }
    T distance = density.center - eta2 + density.scale * sinh(u);
    double sign = ad::value_of(distance) < 0.0 ? -1.0 : 1.0;
    return {log(abs(distance)), sign};
}

// (|z - eta2| - eta1 (z - eta2))^power phi(r), at z = z(r).
template <class T>
struct JohnsonShockIntegrand
{
    JohnsonLogDensity<T> density;
    T eta1;
    T eta2;
    T power;

    T operator()(const T& r) const
    {
        using std::exp;
        using std::log;
        T u = (r + density.skew) / density.shape;
        SignedLog<T> distance = johnson_log_distance(density, eta2, u);
        T log_shock = power * (distance.log + log(1.0 - eta1 * distance.sign));
        return exp(log_shock - 0.5 * (r * r) - 0.5 * std::log(2.0 * std::numbers::pi));
    }
};

// int over [lower, upper] in equal pieces.
template <class T>
T johnson_pieces(const JohnsonShockIntegrand<T>& integrand, const T& lower, const T& upper)
{
    double width = ad::value_of(upper) - ad::value_of(lower);
    double pieces = std::clamp(std::ceil(width / kJohnsonPiece), 1.0, kJohnsonMaxPieces);
    T step = (upper - lower) / pieces;
    T sum{0.0};
    for (double k = 0.0; k < pieces; k += 1.0)
    {
        sum = sum + segment_integral(integrand, lower + step * k, lower + step * (k + 1.0));
    }
    return sum;
}

// NaN where the density overflows (shape below ~0.053, where scale is 0 and center 0 or NaN) or the range
// is not finite, which the pieces would otherwise turn into an empty sum, 0.
template <class T>
T johnson_shock_moment(const JohnsonShockIntegrand<T>& integrand)
{
    T cusp = johnson_normal_of(integrand.density, integrand.eta2);
    double peak = ad::value_of(integrand.power) / ad::value_of(integrand.density.shape);
    double lower = std::min(-peak, ad::value_of(cusp)) - kJohnsonReach;
    double upper = std::max(peak, ad::value_of(cusp)) + kJohnsonReach;
    double scale = ad::value_of(integrand.density.scale);
    bool computable = scale > 0.0 && std::isfinite(scale) && std::isfinite(ad::value_of(cusp)) &&
                      std::isfinite(lower) && std::isfinite(upper);
    if (!computable)
    {
        return T{std::numeric_limits<double>::quiet_NaN()};
    }
    return johnson_pieces(integrand, T{lower}, cusp) + johnson_pieces(integrand, cusp, T{upper});
}

// P(z < 0) = Phi(r(0)).
template <class T>
T johnson_probability_below_zero(const JohnsonLogDensity<T>& density)
{
    using std::erfc;
    return 0.5 * erfc(-(johnson_normal_of(density, T{0.0}) / std::numbers::sqrt2));
}

}

// E(|z - eta2| - eta1 (z - eta2))^power, by quadrature (in the normal variable for jsu); +inf for std and
// sstd with shape <= power, NaN otherwise outside the domain (see in_domain).
template <class T>
T shifted_moment(Dist dist, const ShapeParameters<T>& parameters, const T& eta1, const T& eta2,
                 const T& power)
{
    if (detail::infinite_moment(dist, parameters, power))
    {
        return T{std::numeric_limits<double>::infinity()};
    }
    if (!detail::in_domain_of(dist, parameters))
    {
        return T{std::numeric_limits<double>::quiet_NaN()};
    }
    if (dist == Dist::Jsu)
    {
        JohnsonLogDensity<T> density = johnson_log_density(parameters.skew, parameters.shape);
        return detail::johnson_shock_moment(detail::JohnsonShockIntegrand<T>{density, eta1, eta2, power});
    }
    detail::ShockPowerIntegrand<T> integrand{make_density(dist, parameters), eta1, eta2, power};
    return detail::shock_power_quadrature(integrand, detail::tail_maps(dist, parameters, integrand.density, power));
}

// E|z|^power: the closed form for norm, std and ged, the quadrature for the others; +inf for std and sstd
// with shape <= power, NaN for the others outside the domain.
template <class T>
T abs_moment(Dist dist, const ShapeParameters<T>& parameters, const T& power)
{
    if (detail::infinite_moment(dist, parameters, power))
    {
        return T{std::numeric_limits<double>::infinity()};
    }
    if (!detail::is_symmetric(dist))
    {
        return shifted_moment(dist, parameters, T{0.0}, T{0.0}, power);
    }
    if (dist == Dist::Std)
    {
        return detail::student_abs_moment(parameters.shape, power);
    }
    if (dist == Dist::Ged)
    {
        return detail::ged_abs_moment(parameters.shape, power);
    }
    return detail::normal_abs_moment(power);
}

// E(|z| - eta1 z)^power, which is ((1 - eta1)^power + (1 + eta1)^power) / 2 E|z|^power for a symmetric
// density.
template <class T>
T asymmetric_moment(Dist dist, const ShapeParameters<T>& parameters, const T& eta1, const T& power)
{
    using std::pow;
    if (!detail::is_symmetric(dist))
    {
        return shifted_moment(dist, parameters, eta1, T{0.0}, power);
    }
    T asymmetry = 0.5 * (pow(1.0 - eta1, power) + pow(1.0 + eta1, power));
    return asymmetry * abs_moment(dist, parameters, power);
}

// P(z < 0): exactly 0.5 for norm, std and ged, Phi(r(0)) for jsu, the quadrature of f over (-inf, 0) for the
// others; NaN outside the domain but for norm, std and ged.
template <class T>
T probability_below_zero(Dist dist, const ShapeParameters<T>& parameters)
{
    if (detail::is_symmetric(dist))
    {
        return T{0.5};
    }
    if (!detail::in_domain_of(dist, parameters))
    {
        return T{std::numeric_limits<double>::quiet_NaN()};
    }
    if (dist == Dist::Jsu)
    {
        return detail::johnson_probability_below_zero(johnson_log_density(parameters.skew, parameters.shape));
    }
    detail::DensityIntegrand<T> integrand{make_density(dist, parameters)};
    return detail::below_zero_quadrature(integrand, detail::tail_maps(dist, parameters, integrand.density, T{0.0}));
}

}
