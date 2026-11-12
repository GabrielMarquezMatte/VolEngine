#pragma once

#include "dist/symmetric.hpp"

namespace ve::dist
{

// Distribution parameters of one fit; the ones its distribution lacks are 0.
template <class T>
struct ShapeParameters
{
    T skew{0.0};
    T shape{0.0};
    T ghlambda{0.0};
};

}
