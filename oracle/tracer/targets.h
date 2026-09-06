#ifndef TMNF_TRACER_TARGETS_H
#define TMNF_TRACER_TARGETS_H

#include <stdint.h>

enum CaptureKind {
	CAP_THIS,
	CAP_THIS_OFFSET,
	CAP_STACK_POINTER,
	CAP_STACK_SCALAR,
	CAP_CURRENT_TICK,
	CAP_DEREF_THIS,
	CAP_DYNA_REPLACEMENTS,
	CAP_VEHICLE_WHEEL_SPEED,
	CAP_VEHICLE_DYNA_STATE,
	CAP_VEHICLE_DYNA_TEMP_STATE,
	CAP_VEHICLE_CORPUS_ISO,
	CAP_VEHICLE_DYNA_PARAMS,
	CAP_VEHICLE_TUNING,
	CAP_VEHICLE_WHEELS,
	CAP_VEHICLE_TUNING_BUFFER,
	CAP_CONTACT_OTHER_ISO
};

struct CaptureDirective {
	uint32_t tag;
	enum CaptureKind kind;
	uint32_t offset;
	uint32_t add;
	uint32_t len;
};

#define MAX_DIRECTIVES 12
#define MAX_PATCH_BYTES 12

struct TargetSpec {
	uint32_t va;
	const char *name;
	uint8_t patch_len;
	uint8_t expected[MAX_PATCH_BYTES];
	uint8_t n_in;
	uint8_t n_out;
	struct CaptureDirective in[MAX_DIRECTIVES];
	struct CaptureDirective out[MAX_DIRECTIVES];
};

extern const struct TargetSpec g_targets[];
extern const uint32_t g_target_count;

#endif
