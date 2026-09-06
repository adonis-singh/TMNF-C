/* Host/device portability for the physics sources.
 *
 * The same .c files are compiled twice: by the C compiler for the CPU
 * library and by nvcc as CUDA C++ for the GPU library. TMNF_HD marks a
 * function as callable from both; it expands to nothing outside nvcc, so the
 * C build is unchanged. tmnf_abort/tmnf_fail are the fail-fast primitives:
 * abort() on the host, a device trap (host-visible launch error) on the GPU.
 */
#ifndef TMNF_HD_H
#define TMNF_HD_H

#if defined(__CUDACC__)
#define TMNF_HD __host__ __device__
#else
#define TMNF_HD
#endif

/* The SSE paths in gm.h and collision.c are host-only: nvcc's device pass
 * inherits the host compiler's __SSE2__ but has no intrinsics, so it takes
 * the scalar branches, which perform the same binary32 operations in the
 * same order (the lanes are exact copies of the scalar transcription). */
#if defined(__SSE2__) && !defined(__CUDA_ARCH__)
#define TMNF_SSE 1
#else
#define TMNF_SSE 0
#endif

#if defined(__cplusplus) && !defined(_Static_assert)
#define _Static_assert static_assert
#endif
/* vec_env.h's operation guard is a C11 atomic; CUDA C++ only needs the
 * struct to parse (it never touches a TmnfVecEnv). */
#if defined(__cplusplus) && !defined(_Atomic)
#define _Atomic
#endif

#include <stdio.h>
#include <stdlib.h>

#if defined(__GNUC__) || defined(__clang__)
#define TMNF_NORETURN __attribute__((noreturn))
#else
#define TMNF_NORETURN
#endif

TMNF_HD TMNF_NORETURN static inline void tmnf_abort(void)
{
#if defined(__CUDA_ARCH__)
	__trap();
#else
	abort();
#endif
}

TMNF_HD TMNF_NORETURN static inline void tmnf_fail(const char *message)
{
#if defined(__CUDA_ARCH__)
	printf("tmnf device: %s\n", message);
	__trap();
#else
	fprintf(stderr, "tmnf: %s\n", message);
	abort();
#endif
}

#endif /* TMNF_HD_H */
