#ifndef TMNF_BROADPHASE_TRACE_H
#define TMNF_BROADPHASE_TRACE_H

#include <stdint.h>

#include "detect_trace.h"

#define TMNF_BROADPHASE_TRACE_MAGIC "TMNFBP01"
#define TMNF_BROADPHASE_TRACE_VERSION 2u

enum TmnfBroadphaseTraceKind {
	TMNF_BROADPHASE_GROUP_SPEEDS = 1,
	TMNF_BROADPHASE_GROUP_PERFORM = 2,
	TMNF_BROADPHASE_TREE_STATIC = 3,
	TMNF_BROADPHASE_PREPARE = 4,
	TMNF_BROADPHASE_COMPUTE_COLLISION = 5,
	TMNF_BROADPHASE_DETECT_BETWEEN = 6,
	TMNF_BROADPHASE_DETECT_CORPUS = 7,
};

#pragma pack(push, 1)

struct TmnfBroadphaseTraceHeader {
	char magic[8];
	uint32_t version;
	uint32_t kind;
	uint32_t total_size;
	uint32_t return_value;
	uint32_t root_id;
	uint32_t argument_ids[4];
	uint32_t group_count;
	uint32_t corpus_id_count;
	uint32_t corpus_count;
	uint32_t dyna_count;
	uint32_t state_count;
	uint32_t speed_count;
	uint32_t speed_value_count;
	uint32_t device_count;
	uint32_t table_count;
	uint32_t table_value_count;
	uint32_t zone_count;
	uint32_t static_entry_count;
	uint32_t tree_count;
	uint32_t tree_child_id_count;
	uint32_t iso_count;
	uint32_t plug_count;
	uint32_t surface_count;
	uint32_t vertex_count;
	uint32_t face_count;
	uint32_t node_count;
	uint32_t material_id_count;
	uint32_t buffer_count;
	uint32_t collision_record_count;
	uint32_t merge_buffer_id_count;
	uint32_t groups_offset;
	uint32_t corpus_ids_offset;
	uint32_t corpora_offset;
	uint32_t dynas_offset;
	uint32_t states_offset;
	uint32_t speeds_offset;
	uint32_t speed_values_offset;
	uint32_t devices_offset;
	uint32_t tables_offset;
	uint32_t table_values_offset;
	uint32_t zones_offset;
	uint32_t static_entries_offset;
	uint32_t trees_offset;
	uint32_t tree_child_ids_offset;
	uint32_t isos_offset;
	uint32_t plugs_offset;
	uint32_t surfaces_offset;
	uint32_t vertices_offset;
	uint32_t faces_offset;
	uint32_t nodes_offset;
	uint32_t material_ids_offset;
	uint32_t buffers_offset;
	uint32_t collision_records_offset;
	uint32_t merge_buffer_ids_offset;
};

struct TmnfBroadphaseGroup {
	uint32_t id;
	uint32_t zone_index;
	uint32_t corpus_array_id;
	uint32_t corpus_index;
	uint32_t corpus_count;
	uint32_t corpus_capacity;
	uint32_t speed_id;
	uint32_t device_array_id;
	uint32_t device_index;
	uint32_t device_count;
	uint32_t static_array_id;
	uint32_t static_entry_index;
	uint32_t static_entry_count;
	uint32_t static_entry_capacity;
	uint8_t bytes[0x44];
};

struct TmnfBroadphaseCorpus {
	uint32_t id;
	uint32_t dyna_id;
	uint32_t flags;
	uint32_t tree_id;
	uint8_t bytes[0x5c];
};

struct TmnfBroadphaseDyna {
	uint32_t id;
	uint32_t state_id;
	uint8_t bytes[0x344];
};

struct TmnfBroadphaseState {
	uint32_t id;
	uint8_t bytes[0xb4];
};

struct TmnfBroadphaseSpeed {
	uint32_t id;
	uint32_t value_index;
	uint32_t count;
	uint32_t capacity;
};

struct TmnfBroadphaseDevice {
	uint32_t id;
	uint32_t group_id;
	uint32_t table_id;
	uint8_t bytes[0x1c];
};

struct TmnfBroadphaseTable {
	uint32_t id;
	uint32_t value_index;
	uint32_t value_count;
	uint32_t rows;
	uint32_t columns;
	uint32_t reserved;
	uint32_t stride;
};

struct TmnfBroadphaseZone {
	uint32_t id;
	uint32_t current_material_id;
	uint32_t current_corpus1_id;
	uint32_t current_corpus2_id;
	uint32_t general_buffer_id;
	uint32_t static_group_id;
	uint32_t merge_array_id;
	uint32_t merge_buffer_index;
	uint32_t merge_buffer_count;
	uint32_t merge_buffer_capacity;
	uint8_t bytes[0x1ac];
};

struct TmnfBroadphaseStaticEntry {
	uint32_t id;
	uint32_t skip_count;
	uint8_t box[0x18];
	uint8_t iso[0x30];
	uint32_t tree_flags;
	uint32_t surface_id;
	uint32_t tree_id;
	uint32_t corpus_id;
};

struct TmnfBroadphaseTree {
	uint32_t id;
	uint32_t flags;
	uint32_t surface_id;
	uint32_t buffer_id;
	uint32_t child_index;
	uint32_t child_count;
	uint8_t box[0x18];
	uint8_t local_iso[0x30];
};

struct TmnfBroadphaseIso {
	uint32_t id;
	uint8_t bytes[0x30];
};

struct TmnfBroadphasePlug {
	uint32_t id;
	uint32_t surface_id;
	uint32_t material_index;
	uint32_t material_count;
};

#pragma pack(pop)

_Static_assert(sizeof(struct TmnfBroadphaseTraceHeader) == 0xec,
	"broadphase trace header size");
_Static_assert(sizeof(struct TmnfBroadphaseGroup) == 0x7c,
	"broadphase group size");
_Static_assert(sizeof(struct TmnfBroadphaseCorpus) == 0x6c,
	"broadphase corpus size");
_Static_assert(sizeof(struct TmnfBroadphaseDyna) == 0x34c,
	"broadphase dyna size");
_Static_assert(sizeof(struct TmnfBroadphaseState) == 0xb8,
	"broadphase state size");
_Static_assert(sizeof(struct TmnfBroadphaseSpeed) == 0x10,
	"broadphase speed size");
_Static_assert(sizeof(struct TmnfBroadphaseDevice) == 0x28,
	"broadphase device size");
_Static_assert(sizeof(struct TmnfBroadphaseTable) == 0x1c,
	"broadphase table size");
_Static_assert(sizeof(struct TmnfBroadphaseZone) == 0x1d4,
	"broadphase zone size");
_Static_assert(sizeof(struct TmnfBroadphaseStaticEntry) == 0x60,
	"broadphase static entry size");
_Static_assert(sizeof(struct TmnfBroadphaseTree) == 0x60,
	"broadphase tree size");
_Static_assert(sizeof(struct TmnfBroadphaseIso) == 0x34,
	"broadphase iso size");
_Static_assert(sizeof(struct TmnfBroadphasePlug) == 0x10,
	"broadphase plug size");

#endif
