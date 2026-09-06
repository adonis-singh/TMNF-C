/* Constants and one-instruction helpers shared by vehicle_model3/4/5.c. */
#ifndef TMNF_VEHICLE_OLDMODELS_COMMON_H
#define TMNF_VEHICLE_OLDMODELS_COMMON_H

#include "tmnf_hd.h"
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "tmnf_fp.h"
#include "vehicle_oldmodels.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 0x00D0AC60: float 1e-10, the wheel-axis length gate. */
#define OLDMODELS_LENGTH_EPSILON 0x1.b7cdfcp-34f
/* 0x00BA38FC: float 1e-5. */
#define OLDMODELS_CONTROL_EPSILON 0x1.4f8b58p-17f
/* 0x00B362C0: float 0.1 promoted to double. */
#define OLDMODELS_POINT_ONE 0x1.99999ap-4
/* 0x00B36110: float pi promoted to double. */
#define OLDMODELS_PI 0x1.921fb6p+1
#define OLDMODELS_UINT32_RANGE 0x1p32f

TMNF_HD static inline float x87_mul_double(float value, double multiplier)
{
	return x87_r24(F(value) * multiplier);
}

TMNF_HD static inline float oldmodels_div(float numerator, float denominator)
{
	return x87_r24(F(numerator) / F(denominator));
}

TMNF_HD static inline float oldmodels_abs(float value)
{
	return (float)fabs(F(value));
}

TMNF_HD static inline float oldmodels_sin(float value)
{
	return x87_sin(value);
}

TMNF_HD static inline float oldmodels_cos(float value)
{
	return x87_cos(value);
}

TMNF_HD static inline float oldmodels_atan2(float y, float x)
{
	return x87_atan2(y, x);
}

/* `(int)value < 0` in the decompilation: the sign bit, not a compare. */
TMNF_HD static inline float oldmodels_sign_bits(float value)
{
	uint32_t bits;

	memcpy(&bits, &value, sizeof(bits));
	return (bits & UINT32_C(0x80000000)) != 0 ? -1.0f : 1.0f;
}

/* FCOM against zero: negative or zero picks -1. */
TMNF_HD static inline float oldmodels_sign_compare(float value)
{
	return F(value) <= 0.0 ? -1.0f : 1.0f;
}

TMNF_HD const TMNFVehicleGroundMaterial *VehicleOldModels_WheelMaterial(
	const CSceneVehicleCarOldModelsContext *context,
	const CSceneVehicleCarWheel *wheel);

#ifdef __cplusplus
}
#endif

#endif
