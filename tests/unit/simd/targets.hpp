#pragma once

#include <cstdint>
#include <hwy/targets.h>
#include <iostream>
#include <vector>

namespace ve::test
{

// Every SIMD target this machine can run and the build compiled, for kernels that must agree on all of them.
inline std::vector<int64_t> runnable_targets()
{
    return hwy::SupportedAndGeneratedTargets();
}

// Restricts the runtime dispatch to one target for the lifetime of the object; zero restores every supported target.
class ForcedTarget
{
public:
    explicit ForcedTarget(int64_t target)
    {
        std::cout << "[ target ] " << hwy::TargetName(target) << '\n';
        hwy::SetSupportedTargetsForTest(target);
    }
    ForcedTarget(const ForcedTarget&) = delete;
    ForcedTarget& operator=(const ForcedTarget&) = delete;
    ~ForcedTarget()
    {
        hwy::SetSupportedTargetsForTest(0);
    }
};

}
