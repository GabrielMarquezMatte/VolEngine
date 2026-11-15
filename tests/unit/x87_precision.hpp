#pragma once

#include <utility>

namespace ve::testing
{

// f() with the x87 precision-control field at 64-bit, then at 53-bit: a pool thread and the Python main thread
// of MinGW run the unit at those two settings, and a long double computation differs between them. Elsewhere (no
// x87 unit) both calls run with the ordinary settings.
template <class Function>
auto at_both_x87_precisions(const Function& f)
{
    using Result = decltype(f());
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
    unsigned short saved = 0;
    __asm__ volatile("fnstcw %0" : "=m"(saved));
    const auto set = [](unsigned short word) { __asm__ volatile("fldcw %0" : : "m"(word)); };
    set(static_cast<unsigned short>((saved & ~0x0300U) | 0x0300U));
    Result extended = f();
    set(static_cast<unsigned short>((saved & ~0x0300U) | 0x0200U));
    Result rounded = f();
    set(saved);
    return std::pair<Result, Result>{std::move(extended), std::move(rounded)};
#else
    return std::pair<Result, Result>{f(), f()};
#endif
}

}
