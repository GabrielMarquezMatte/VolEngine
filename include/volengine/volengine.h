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

typedef int32_t ve_status;
#define VE_OK 0
#define VE_INVALID_ARGUMENT 1
#define VE_INVALID_HANDLE 2
#define VE_UNSUPPORTED 3
#define VE_OUT_OF_MEMORY 4
#define VE_INTERNAL_ERROR 5

typedef struct ve_context ve_context;

VE_API const char* ve_last_error(void);

VE_API ve_status ve_context_create(ve_context** out);
VE_API ve_status ve_context_set_threads(ve_context* ctx, int32_t threads);
VE_API ve_status ve_context_threads(const ve_context* ctx, int32_t* out);
VE_API void ve_context_free(ve_context* ctx);

#ifdef __cplusplus
}
#endif

#endif
