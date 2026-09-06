#ifndef TMNF_VEHICLE_MODEL6_DRIVE_H
#define TMNF_VEHICLE_MODEL6_DRIVE_H

#include "tmnf_hd.h"
#include "vehicle_model6_common.h"

#ifdef __cplusplus
extern "C" {
#endif

TMNF_HD VehicleModel6Flow VehicleModel6_UpdateDriveState(
	VehicleModel6Context *context, const VehicleModel6Inputs *inputs,
	VehicleModel6Scratch *scratch);

#ifdef __cplusplus
}
#endif

#endif /* TMNF_VEHICLE_MODEL6_DRIVE_H */
