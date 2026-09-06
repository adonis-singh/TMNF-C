/* Immutable TMNF race-route snapshot.
 *
 * The oracle dumps the live race's spawn state, ordered trigger volumes, lap
 * metadata, and a dense road-center reference polyline. Relative file offsets
 * are fixed to direct pointers in a private mmap and then made read-only.
 */
#ifndef TMNF_ROUTE_H
#define TMNF_ROUTE_H

#include "tmnf_hd.h"
#include <stddef.h>
#include <stdint.h>

#include "collision.h"
#include "hms_state.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
	TMNF_ROUTE_VERSION = 3,
	TMNF_ROUTE_SECTION_COUNT = 5,
};

typedef enum {
	TMNF_ROUTE_METADATA = 0,
	TMNF_ROUTE_START = 1,
	TMNF_ROUTE_CHECKPOINTS = 2,
	TMNF_ROUTE_FINISH = 3,
	TMNF_ROUTE_REFERENCE = 4,
} TmnfRouteSectionId;

typedef enum {
	TMNF_ROUTE_WAYPOINT_START = 0,
	TMNF_ROUTE_WAYPOINT_FINISH = 1,
	TMNF_ROUTE_WAYPOINT_CHECKPOINT = 2,
	TMNF_ROUTE_WAYPOINT_NONE = 3,
	TMNF_ROUTE_WAYPOINT_START_FINISH = 4,
} TmnfRouteWaypointType;

enum {
	TMNF_ROUTE_MULTILAP = 1u << 0,
};

typedef struct {
	uint64_t offset;
	uint32_t count;
	uint32_t stride;
} TmnfRouteSection;

typedef struct {
	char magic[8];                  /* "TMNFROU1" */
	uint32_t version;
	uint32_t endian;                /* 0x12345678 */
	uint32_t header_size;           /* 0xE0 */
	uint32_t section_count;         /* 5 */
	uint64_t file_size;
	uint8_t exe_sha256[32];
	uint8_t track_sha256[32];
	TmnfRouteSection sections[TMNF_ROUTE_SECTION_COUNT];
	uint8_t payload_sha256[32];
	uint8_t reserved[16];
} TmnfRouteHeader;

/* On disk, the four *_rel fields are offsets from the file base. */
typedef struct {
	uint32_t lap_count;
	uint32_t checkpoint_count;
	uint32_t finish_count;          /* alternatives; checkpoint_count + finish_count <= 64 */
	uint32_t reference_count;       /* serialized checkpoint_count + 2 */
	uint32_t total_race_checkpoints;
	uint32_t race_checkpoint_limit;
	uint32_t flags;
	uint32_t centerline_count;      /* dense reference-section records */
	uint64_t start_rel;
	uint64_t checkpoints_rel;
	uint64_t finish_rel;
	uint64_t reference_rel;
} TmnfRouteMetadata;

/* Deterministic, pointer-free prefix of CHmsStateDyna. The two trailing game
 * dwords are excluded because the final one is a process-local pointer. */
typedef struct {
	GmQuat quat;
	GmMat3 rot;
	GmVec3 pos;
	GmVec3 lin_vel;
	GmVec3 lin_vel_added;
	GmVec3 ang_vel;
	GmVec3 force;
	GmVec3 torque;
	GmMat3 inv_inertia_world;
	GmVec3 not_tweaked_lin_vel;
} TmnfRouteInitialState;

/* spawn is CGameCtnBlock::GetSpawnLoc (0x0060B410) of the start block: the
 * isometry a respawn places the car at when the current lap has no
 * checkpoint yet (analysis/respawn.md). */
typedef struct {
	GmIso4 transform;
	TmnfRouteInitialState initial_state;
	uint32_t block_index;
	uint32_t waypoint_type;
	GmIso4 spawn;
} TmnfRouteStart;

/* box is the exact local root CPlugTree AABB used by TMNF collision. The box
 * lives in the tree's parent frame, so transform is the corpus isometry,
 * excluding the root CPlugTree::Location.
 * spawn is CGameCtnBlock::GetSpawnLoc of the checkpoint block; no_respawn is
 * CGameCtnBlockInfo+0x120, which makes OnCheckpoint (0x0047C330) keep the
 * previous spawn location instead of this one. */
typedef struct {
	uint32_t race_index;
	uint32_t block_index;
	uint32_t waypoint_type;
	uint32_t tree_flags;
	GmBoxAligned box;
	GmIso4 transform;
	GmIso4 spawn;
	uint32_t no_respawn;
	uint32_t reserved;
} TmnfRouteTrigger;

typedef struct {
	GmVec3 position;
	float arc_length;
	float half_width;
	uint32_t leg_index;
} TmnfRouteReferencePoint;

_Static_assert(sizeof(TmnfRouteSection) == 0x10, "route section size");
_Static_assert(sizeof(TmnfRouteHeader) == 0xE0, "route header size");
_Static_assert(sizeof(TmnfRouteMetadata) == 0x40, "route metadata size");
_Static_assert(sizeof(TmnfRouteInitialState) == 0xAC, "route state size");
_Static_assert(sizeof(TmnfRouteStart) == 0x114, "route start size");
_Static_assert(sizeof(TmnfRouteTrigger) == 0x90, "route trigger size");
_Static_assert(
	sizeof(TmnfRouteReferencePoint) == 0x18, "route reference size");

typedef struct {
	float arc_length;
	float lateral_offset;
	float half_width;
	uint32_t segment_index;
	uint32_t centerline_segment_index;
	uint32_t segments_tested;
} TmnfRouteProjection;

/* Projection BVH built at load; leaves carry a centerline segment. Exposed so
 * a device copy of the route can be sized and uploaded. */
typedef struct TmnfRouteBvhNode {
	float minimum[3];
	float maximum[3];
	uint32_t left;
	uint32_t right;
	uint32_t segment;
} TmnfRouteBvhNode;

typedef struct {
	void *mapping;
	size_t mapping_size;
	const TmnfRouteHeader *header;
	const TmnfRouteMetadata *metadata;
	const TmnfRouteStart *start;
	const TmnfRouteTrigger *checkpoints;
	const TmnfRouteTrigger *finish; /* finish_count entries; index zero owns the reference line */
	const TmnfRouteReferencePoint *centerline;
	TmnfRouteBvhNode *projection_bvh;
	uint32_t projection_bvh_count;
	uint32_t projection_bvh_height;
} TmnfRoute;

TmnfRoute *TmnfRoute_Load(
	const char *path, const uint8_t expected_track_sha256[32]);
void TmnfRoute_Unload(TmnfRoute *route);

TMNF_HD const TmnfRouteStart *TmnfRoute_GetStart(const TmnfRoute *route);
TMNF_HD uint32_t TmnfRoute_GetCheckpointCount(const TmnfRoute *route);
TMNF_HD const TmnfRouteTrigger *TmnfRoute_GetCheckpoint(
	const TmnfRoute *route, uint32_t index);
TMNF_HD const TmnfRouteTrigger *TmnfRoute_GetFinish(const TmnfRoute *route);
TMNF_HD uint32_t TmnfRoute_GetReferencePointCount(const TmnfRoute *route);
TMNF_HD const TmnfRouteReferencePoint *TmnfRoute_GetReferencePoints(
	const TmnfRoute *route);
TMNF_HD float TmnfRoute_GetReferenceLength(const TmnfRoute *route);

/* Projects exactly onto the nearest finite 3D polyline segment through an
 * immutable BVH. lateral_offset is the non-negative Euclidean distance,
 * half_width is linearly interpolated, segment_index is the checkpoint leg,
 * centerline_segment_index is the physical segment, and ties choose the first
 * physical segment. */
TMNF_HD TmnfRouteProjection TmnfRoute_Project(
	const TmnfRoute *route, const GmVec3 *world_position);

#ifdef __cplusplus
}
#endif

#endif /* TMNF_ROUTE_H */
