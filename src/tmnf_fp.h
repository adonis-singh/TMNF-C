/* Floating-point policy for the bit-exact TMNF physics port.
 *
 * CORRECTED MODEL (from golden traces): the live game runs x87 with
 * precision-control = 24 bits, NOT 53. The CRT initially sets PC=53
 * (__setdefaultprecision, 0x00404078), but Direct3D device creation reprograms
 * the x87 control word to single precision (PC=24) before physics runs. Under
 * PC=24 every x87 arithmetic result is rounded to a 24-bit significand, i.e.
 * IEEE single precision, differing from strict float32 only in the wider 15-bit
 * exponent range, which car-physics magnitudes never reach. Therefore:
 *
 *   x87(PC=24) op result  ==  correctly-rounded float32 op    (bit-identical)
 *
 * So each arithmetic operation is done in float precision (round to 24-bit
 * significand after every op). The helpers below make this explicit: compute
 * the operation in double (exact for a single float*float or float+float) and
 * round once to float. This equals native float arithmetic on x86-64 SSE
 * (FLT_EVAL_METHOD=0) and is fast and vectorizable for the RL environment.
 *
 * Rules for porters:
 *   - Do every arithmetic op with x87_mul/x87_add/x87_sub (or plain float ops),
 *     so each intermediate rounds to 24-bit. Do NOT accumulate in double.
 *   - Preserve the exact operation tree and associativity from the .asm
 *     (use tools/x87trace.py to derive it; the decompiler's order is unreliable).
 *   - Float->int is truncation toward zero: use ftol().
 *   - Square root: x87 fsqrt under PC=24 rounds to 24-bit; use x87_sqrt().
 */
#ifndef TMNF_FP_H
#define TMNF_FP_H

#include "tmnf_hd.h"
#include <math.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Read a float lvalue as double for a single exact op before re-rounding. */
#define F(x) ((double)(x))

/* Round a double-valued single-op result to the x87 PC=24 (float) significand. */
TMNF_HD static inline float x87_r24(double value) { return (float)value; }

/* For one operation on two binary32 operands, the binary64 result rounded to
 * binary32 equals the directly rounded binary32 result: double rounding is
 * innocuous for + - * / and sqrt whenever the wide format carries at least
 * 2 * 24 + 2 significand bits, and binary64 carries 53. GCC narrows the
 * x87_mul/x87_add/x87_sub forms to float instructions on its own (verified
 * with objdump: mulss/addss/subss, no *sd variants); sqrt is written as sqrtf
 * because GCC keeps the wide form for it. tests/x87_fp_identity.c checks
 * sqrt for every binary32 and samples the other operations. */
#if defined(__CUDA_ARCH__)
/* The device spells the same operations as round-to-nearest binary32
 * intrinsics: identical bits by the argument above, and FP64 issues at 1/64
 * of the FP32 rate on the RTX 5090. -prec-div=true and -prec-sqrt=true make
 * __fdiv_rn/__fsqrt_rn the plain / and sqrtf too; the intrinsics are
 * explicit so no flag can change them. */
TMNF_HD static inline float x87_mul(float a, float b) { return __fmul_rn(a, b); }
TMNF_HD static inline float x87_add(float a, float b) { return __fadd_rn(a, b); }
TMNF_HD static inline float x87_sub(float a, float b) { return __fsub_rn(a, b); }
TMNF_HD static inline float x87_div(float a, float b) { return __fdiv_rn(a, b); }
TMNF_HD static inline float x87_rcp(float x) { return __fdiv_rn(1.0f, x); }
TMNF_HD static inline float x87_sqrt(float x) { return __fsqrt_rn(x); }
#else
TMNF_HD static inline float x87_mul(float a, float b) { return x87_r24(F(a) * F(b)); }
TMNF_HD static inline float x87_add(float a, float b) { return x87_r24(F(a) + F(b)); }
TMNF_HD static inline float x87_sub(float a, float b) { return x87_r24(F(a) - F(b)); }
/* fdiv under PC=24 on two binary32 operands (1.0 is exact): same argument. */
TMNF_HD static inline float x87_div(float a, float b) { return a / b; }
TMNF_HD static inline float x87_rcp(float x) { return 1.0f / x; }

/* x87 fsqrt under PC=24 rounds to 24-bit. */
TMNF_HD static inline float x87_sqrt(float x) { return sqrtf(x); }
#endif

/* Transcendentals on float arguments, rounded to float. The host evaluates
 * glibc's double functions; the device evaluates CUDA's and corrects the
 * exhaustively enumerated inputs where the two differ after rounding
 * (src/cuda/tmnf_dev_libm.h). Every call site in the port passes a float
 * widened with F(), so a float argument loses nothing. */
#if defined(__CUDACC__)
#include "cuda/tmnf_dev_libm.h"
#endif
#if defined(__CUDA_ARCH__)
TMNF_HD static inline float x87_sin(float x) { return tmnf_dev_sin_r24(x); }
TMNF_HD static inline float x87_cos(float x) { return tmnf_dev_cos_r24(x); }
TMNF_HD static inline float x87_exp(float x) { return tmnf_dev_exp_r24(x); }
TMNF_HD static inline float x87_atan2(float y, float x)
{
	return tmnf_dev_atan2_r24(y, x);
}
#else
TMNF_HD static inline float x87_sin(float x) { return x87_r24(sin(F(x))); }
TMNF_HD static inline float x87_cos(float x) { return x87_r24(cos(F(x))); }
TMNF_HD static inline float x87_exp(float x) { return x87_r24(exp(F(x))); }
TMNF_HD static inline float x87_atan2(float y, float x)
{
	return x87_r24(atan2(F(y), F(x)));
}
#endif

/* _ftol / FISTP with round-toward-zero == C truncation toward zero. x86
 * cvttsd2si yields the integer indefinite 0x80000000 for NaN and out-of-range
 * values; the device conversion would saturate, so it is spelled out. */
TMNF_HD static inline int32_t ftol(double x)
{
#if defined(__CUDA_ARCH__)
	if (!(x > -2147483649.0 && x < 2147483648.0))
		return INT32_MIN;
#endif
	return (int32_t)x;
}

#ifdef __cplusplus
}
#endif

#endif /* TMNF_FP_H */
