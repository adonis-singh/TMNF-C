#ifndef TMNF_VEHICLE_MODEL6_WHEELS_H
#define TMNF_VEHICLE_MODEL6_WHEELS_H

#include "tmnf_hd.h"
#include "vehicle_model6_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 0x007C4822..0x007C53B9 */
TMNF_HD void VehicleModel6_ProcessWheelContacts(
	VehicleModel6Context *context, const VehicleModel6Inputs *inputs,
	VehicleModel6Scratch *scratch);

#ifdef __cplusplus
}
#endif

#endif /* TMNF_VEHICLE_MODEL6_WHEELS_H */
