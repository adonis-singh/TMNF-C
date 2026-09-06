#ifndef TMNF_DETECT_TRACE_H
#define TMNF_DETECT_TRACE_H

#include <stdint.h>

#define TMNF_DETECT_TRACE_MAGIC "TMNFDET1"
#define TMNF_DETECT_TRACE_VERSION 1u

enum TmnfDetectTraceKind {
	TMNF_DETECT_SPHERE_MESH = 1,
	TMNF_DETECT_BOX_MESH = 2,
	TMNF_DETECT_SPHERE_SPHERE = 3,
	TMNF_DETECT_SURF_DISPATCH = 4,
	TMNF_DETECT_PLUG_SURFACE = 5,
	TMNF_DETECT_BOX_TEST_INTER = 6,
	TMNF_DETECT_BOX_SET_MULT = 7,
	TMNF_DETECT_MERGE = 8,
	TMNF_DETECT_QSORT = 9,
	TMNF_DETECT_ELLIPSOID_MESH = 10,
};

#pragma pack(push, 1)

struct TmnfDetectTraceHeader {
	char magic[8];
	uint32_t version;
	uint32_t kind;
	uint32_t total_size;
	uint32_t return_value;
	uint32_t located_count;
	uint32_t surface_count;
	uint32_t iso_count;
	uint32_t plug_count;
	uint32_t buffer_count;
	uint32_t vertex_count;
	uint32_t face_count;
	uint32_t node_count;
	uint32_t material_id_count;
	uint32_t located_offset;
	uint32_t surfaces_offset;
	uint32_t isos_offset;
	uint32_t plugs_offset;
	uint32_t buffers_offset;
	uint32_t vertices_offset;
	uint32_t faces_offset;
	uint32_t nodes_offset;
	uint32_t material_ids_offset;
	uint32_t collision_records_offset;
	uint32_t collision_record_count;
	uint32_t boxes_offset;
	uint32_t box_count;
	uint32_t comparator_va;
};

struct TmnfDetectLocated {
	uint32_t surface_id;
	uint32_t iso_id;
	uint32_t is_located;
};

struct TmnfDetectSurface {
	uint32_t id;
	uint16_t material_index;
	uint8_t type;
	uint8_t reserved;
	uint8_t shape[0x18];
	uint32_t vertex_index;
	uint32_t vertex_count;
	uint32_t face_index;
	uint32_t face_count;
	uint32_t node_index;
	uint32_t node_count;
};

struct TmnfDetectPlugSurface {
	uint32_t surface_id;
	uint32_t material_index;
	uint32_t material_count;
};

struct TmnfDetectBuffer {
	uint32_t id;
	uint32_t count;
	uint32_t capacity;
	uint32_t record_index;
	uint32_t active;
	uint32_t has_active;
};

#pragma pack(pop)

_Static_assert(sizeof(struct TmnfDetectTraceHeader) == 0x74,
	"detect trace header size");
_Static_assert(sizeof(struct TmnfDetectLocated) == 0x0c,
	"detect located size");
_Static_assert(sizeof(struct TmnfDetectSurface) == 0x38,
	"detect surface size");
_Static_assert(sizeof(struct TmnfDetectPlugSurface) == 0x0c,
	"detect plug surface size");
_Static_assert(sizeof(struct TmnfDetectBuffer) == 0x18,
	"detect buffer size");

#endif
