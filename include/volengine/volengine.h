#ifndef VOLENGINE_VOLENGINE_H
#define VOLENGINE_VOLENGINE_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#if defined(VE_BUILDING)
#define VE_API __declspec(dllexport)
#else
#define VE_API __declspec(dllimport)
#endif
#else
#define VE_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C"
{
#endif

#define VE_ABI_VERSION 0

VE_API int32_t ve_abi_version(void);

#ifdef __cplusplus
}
#endif

#endif
