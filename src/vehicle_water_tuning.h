#ifndef TMNF_VEHICLE_WATER_TUNING_H
#define TMNF_VEHICLE_WATER_TUNING_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * CSceneVehicleCarTuning +0x210 and +0x214: the two CFuncKeysReal curves that
 * 0x007C2910 CSceneVehicleCar::ApplyWaterForces evaluates at the entry
 * ratio to scale the water-entry impulse (vertical: +0x210, horizontal:
 * +0x214). The vehicle snapshot does not carry these two curves (its dumper
 * captures the other 21 tuning curves), so they are embedded from a live dump
 * of tuning key 29 in the A04-Acrobatic process. Every tuning key holds the
 * same StadiumCar values. Float bits, interpolation 0 (linear).
 */
static const float TMNF_WATER_IMPULSE_POSITIONS[4] = {
	0x0p+0f, 0x1.99999ap-2f, 0x1p-1f, 0x1p+0f,
};
static const float TMNF_WATER_IMPULSE_VERTICAL_VALUES[4] = {
	0x1.ccccccp-1f, 0x1.ccccccp-1f, 0x1.8p+0f, 0x1.8p+0f,
};
static const float TMNF_WATER_IMPULSE_HORIZONTAL_VALUES[4] = {
	0x1.666666p-1f, 0x1.666666p-1f, 0x1.333334p-1f, 0x1.333334p-1f,
};

#ifdef __cplusplus
}
#endif

#endif
