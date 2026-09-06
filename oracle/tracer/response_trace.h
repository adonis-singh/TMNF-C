#ifndef TMNF_RESPONSE_TRACE_H
#define TMNF_RESPONSE_TRACE_H

#include <stdint.h>

#define TMNF_RESPONSE_TRACE_MAGIC "TMNFRSP1"
#define TMNF_RESPONSE_TRACE_VERSION 1u
#define TMNF_RESPONSE_SURFACE_MATERIAL_COUNT 31u

enum TmnfResponseTraceKind {
	TMNF_RESPONSE_SOLVE_IMPULSE = 1,
	TMNF_RESPONSE_COMPUTE_COLLISION_RESPONSE = 2
};

#pragma pack(push, 1)

struct TmnfResponseTraceHeader {
	char magic[8];
	uint32_t version;
	uint32_t kind;
	uint32_t total_size;
	uint32_t collision_count;
	uint32_t body_count;
	uint32_t material_count;
	uint32_t tree_count;
	uint32_t contact_count;
	uint32_t event_count;
	uint32_t replacement_count;
	uint32_t collisions_offset;
	uint32_t bodies_offset;
	uint32_t materials_offset;
	uint32_t contacts_offset;
	uint32_t events_offset;
	uint32_t replacements_offset;
	uint32_t surface_materials_offset;
};

struct TmnfResponseCollision {
	uint32_t body1_id;
	uint32_t tree1_id;
	uint32_t body2_id;
	uint32_t tree2_id;
	uint8_t collision[0x38];
	uint32_t material_id;
};

struct TmnfResponseBody {
	uint32_t id;
	uint32_t classification_flags;
	uint32_t response_flags;
	float response_weight;
	uint8_t iso[0x30];
	uint32_t dyna_present;
	uint32_t has_contact_sink;
	uint32_t contact_target_va;
	uint32_t dirty_flag;
	int32_t mode;
	uint32_t replacement_index;
	uint32_t replacement_count;
	uint32_t replacement_capacity;
	uint8_t params[0x44];
	uint8_t state[0xb4];
};

struct TmnfResponseMaterial {
	uint32_t id;
	uint8_t bytes[0x14];
};

struct TmnfResponseContact {
	uint32_t present;
	uint32_t body_id;
	uint32_t tree_id;
	uint16_t surface_material;
	uint16_t reserved0a;
	uint8_t normal[0x0c];
	uint8_t position[0x0c];
	uint8_t relative_speed[0x0c];
	uint8_t replacement[0x0c];
	uint32_t accepted;
	uint32_t other_body_id;
	uint32_t other_tree_id;
	uint16_t other_surface_material;
	uint16_t reserved4a;
};

struct TmnfResponseContactEvent {
	uint32_t target_va;
	uint32_t item_body_id;
	struct TmnfResponseContact input;
	struct TmnfResponseContact output;
};

#pragma pack(pop)

_Static_assert(sizeof(struct TmnfResponseTraceHeader) == 0x4c,
	"response trace header size");
_Static_assert(sizeof(struct TmnfResponseCollision) == 0x4c,
	"response collision size");
_Static_assert(sizeof(struct TmnfResponseBody) == 0x158,
	"response body size");
_Static_assert(sizeof(struct TmnfResponseMaterial) == 0x18,
	"response material size");
_Static_assert(sizeof(struct TmnfResponseContact) == 0x50,
	"response contact size");
_Static_assert(sizeof(struct TmnfResponseContactEvent) == 0xa8,
	"response contact event size");

#endif
