#pragma once

namespace ve::diagnostics
{

// Upper tails taken directly (Boost.Math gamma_q and ibeta, in tails.cpp), never as 1 - cdf; NaN stays NaN.
double gamma_upper(double x, double shape, double scale);
double chi_squared_upper(double x, double degrees);
double student_two_sided(double t, double degrees);

}
