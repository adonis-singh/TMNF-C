#ifndef TMNF_VEHICLE_MODEL6_LONGITUDINAL_H
#define TMNF_VEHICLE_MODEL6_LONGITUDINAL_H

#include "tmnf_hd.h"
#include <stddef.h>

#include "vehicle_model6_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Region 5 also reads tuning +0x160, which is absent from the shared scalar
 * view. Callers of this region provide this complete object and point
 * VehicleModel6Context.scalars at its common prefix.
 */
typedef struct {
	VehicleModel6TuningScalars common;
	float s160;
} VehicleModel6LongitudinalScalars;

_Static_assert(offsetof(VehicleModel6LongitudinalScalars, common) == 0,
	"longitudinal scalar prefix");

/* 0x007C5FB8..0x007C67D1, Model 6 region 5. */
TMNF_HD void VehicleModel6_ApplyLongitudinalForces(
	VehicleModel6Context *context, const VehicleModel6Inputs *inputs,
	VehicleModel6Scratch *scratch);

#ifdef __cplusplus
}
#endif

#endif /* TMNF_VEHICLE_MODEL6_LONGITUDINAL_H */
