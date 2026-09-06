#ifndef TMNF_MODEL6_TRACE_H
#define TMNF_MODEL6_TRACE_H

#include <stdint.h>

#define TMNF_MODEL6_TRACE_MAGIC "TMNFM6G1"
#define TMNF_MODEL6_TRACE_VERSION 1u
#define TMNF_MODEL6_CURVE_COUNT 12u

enum TmnfModel6CurveSlot {
	TMNF_M6_CURVE_ACCEL,
	TMNF_M6_CURVE_STEER_SLOWDOWN,
	TMNF_M6_CURVE_STEER_DRIVE_TORQUE,
	TMNF_M6_CURVE_MAX_SIDE_FRICTION,
	TMNF_M6_CURVE_SLIPPING_ACCEL,
	TMNF_M6_CURVE_DAMPER_MODULATION,
	TMNF_M6_CURVE_REAR_GEAR_ACCEL,
	TMNF_M6_CURVE_ROLLOVER_RATIO,
	TMNF_M6_CURVE_BURNOUT_RADIUS,
	TMNF_M6_CURVE_BURNOUT_LATERAL_SPEED,
	TMNF_M6_CURVE_DONUT_ROLLOVER,
	TMNF_M6_CURVE_BURNOUT_ROLLOVER,
};

#pragma pack(push, 1)

struct TmnfModel6Curve {
	uint32_t object_id;
	uint32_t positions_id;
	uint32_t values_id;
	uint32_t count;
	int32_t interpolation;
	uint32_t positions_offset;
	uint32_t values_offset;
};

struct TmnfModel6Material {
	uint32_t object_id;
	float values[4];
};

struct TmnfModel6TraceHeader {
	char magic[8];
	uint32_t version;
	uint32_t total_size;
	uint32_t phase;
	uint32_t car_id;
	uint32_t tuning_id;
	uint32_t wheels_id;
	uint32_t item_id;
	uint32_t corpus_id;
	uint32_t dyna_id;
	uint32_t params_id;
	uint32_t state_id;
	uint32_t model_iso_id;
	uint32_t body_reference_id;
	uint32_t wheel_count;
	uint32_t ground_id_count;
	uint32_t ground_material_count;
	uint32_t car_offset;
	uint32_t tuning_offset;
	uint32_t wheels_offset;
	uint32_t dyna_offset;
	uint32_t params_offset;
	uint32_t state_offset;
	uint32_t curves_offset;
	uint32_t ground_ids_offset;
	uint32_t ground_materials_offset;
	uint32_t model_iso_offset;
	uint32_t body_reference_offset;
	uint32_t curve_data_offset;
	uint32_t tick;
	float model_value;
	float lateral_force_factor;
	float longitudinal_force_factor;
	float steering_angle;
	int32_t grounded;
	uint8_t existing_force[12];
	uint8_t local_speed[12];
	uint8_t local_angular_speed[12];
	uint8_t material[16];
	int32_t sliding;
	float brake_force;
	uint8_t source_exe_sha256[32];
	uint8_t source_track_sha256[32];
};

#pragma pack(pop)

_Static_assert(sizeof(struct TmnfModel6Curve) == 0x1c,
	"Model6 curve descriptor size");

#endif
