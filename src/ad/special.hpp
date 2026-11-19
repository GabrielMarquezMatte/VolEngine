#pragma once

namespace ve::ad
{

// Digamma for x > 0; NaN below -1e6, where the recurrence over the shift to the asymptotic range would not end.
double digamma(double x);

// log Gamma(x) for x > 0, without writing the global signgam that std::lgamma writes (a data race between threads),
// and in double on every thread. Like std::lgamma, a pole or an argument outside the domain gives inf or NaN.
double log_gamma(double x);

}
