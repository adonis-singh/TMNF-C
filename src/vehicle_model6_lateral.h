#ifndef TMNF_VEHICLE_MODEL6_LATERAL_H
#define TMNF_VEHICLE_MODEL6_LATERAL_H

#include "tmnf_hd.h"
#include "vehicle_model6_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Model 6 region 4: [0x007C5BB3, 0x007C5FB8). */
TMNF_HD void VehicleModel6_ApplyLateralFriction(
	VehicleModel6Context *context, const VehicleModel6Inputs *inputs,
	VehicleModel6Scratch *scratch);

#ifdef __cplusplus
}
#endif

#endif /* TMNF_VEHICLE_MODEL6_LATERAL_H */
