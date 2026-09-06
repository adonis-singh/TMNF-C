/* Immutable static-track collision snapshot.
 *
 * The oracle dumps TMNF's already-flattened static collision octree. Loading
 * this file avoids reimplementing the GBX asset pipeline and leaves the native
 * hot path with direct, read-only pointers into an mmap.
 */
#ifndef TMNF_TRACK_H
#define TMNF_TRACK_H

#include <stddef.h>
#include <stdint.h>

#include "collision.h"
#include "collision_response.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
	TMNF_TRACK_VERSION = 3,
	TMNF_TRACK_SECTION_COUNT = 12,
	TMNF_TRACK_MATERIAL_COUNT = 31,
};

typedef enum {
	TMNF_TRACK_ENTRIES = 0,
	TMNF_TRACK_SURFACES = 1,
	TMNF_TRACK_MESHES = 2,
	TMNF_TRACK_VERTICES = 3,
	TMNF_TRACK_FACES = 4,
	TMNF_TRACK_NODES = 5,
	TMNF_TRACK_MATERIAL_IDS = 6,
	TMNF_TRACK_MATERIAL_DATA = 7,
	TMNF_TRACK_COLLISION_PAIRS = 8,
	/* CHmsCorpus+0x18 location of static corpus id k at index k-1; this is
	 * what CHmsCorpus::GetLocation (0x005474A0) returns for a static body
	 * and differs from the flattened tree isos when a block's collision
	 * tree carries a local transform. */
	TMNF_TRACK_CORPUS_ISOS = 9,
	/* CHmsZone+0x154 water map header and its width*height cells. */
	TMNF_TRACK_WATER = 10,
	TMNF_TRACK_WATER_CELLS = 11,
} TmnfTrackSectionId;

typedef struct {
	uint64_t offset;
	uint32_t count;
	uint32_t stride;
} TmnfTrackSection;

typedef struct {
	char magic[8];                  /* "TMNFTRK1" */
	uint32_t version;
	uint32_t endian;                /* 0x12345678 */
	uint32_t header_size;           /* 0x150 */
	uint32_t section_count;         /* 12 */
	uint64_t file_size;
	uint8_t exe_sha256[32];
	uint8_t track_sha256[32];
	TmnfTrackSection sections[TMNF_TRACK_SECTION_COUNT];
	uint8_t payload_sha256[32];
	uint8_t reserved[16];
} TmnfTrackHeader;

/* On disk, *_rel fields are offsets from the file base. The loader replaces
 * them with native pointers in its private writable mapping, then mprotects
 * the complete mapping read-only. These layouts deliberately match the native
 * collision structs after fixup. */
typedef struct {
	uint32_t skip_count;
	GmBoxAligned box;
	GmIso4 iso;
	uint32_t tree_flags;
	uint64_t surface_rel;
	uint32_t tree_id;
	uint32_t corpus_id;
} TmnfTrackStaticEntry;

typedef struct {
	uint64_t mesh_rel;
	uint64_t material_ids_rel;
	uint32_t material_count;
	uint32_t reserved;
} TmnfTrackSurface;

typedef struct {
	GmSurf base;
	uint32_t vertex_count;
	uint32_t pad0c;
	uint64_t vertices_rel;
	uint32_t face_count;
	uint32_t pad1c;
	uint64_t faces_rel;
	uint32_t node_count;
	uint32_t pad2c;
	uint64_t nodes_rel;
} TmnfTrackMesh;

typedef struct {
	float friction;
	float restitution;
} TmnfTrackMaterialData;

typedef struct {
	uint32_t raw[5];
} TmnfTrackCollisionPair;

typedef struct {
	float cell_x;                   /* map +0x00 */
	float cell_z;                   /* map +0x04 */
	float origin_x;                 /* map +0x08 */
	float origin_z;                 /* map +0x0c */
	uint32_t width;                 /* map +0x10 */
	uint32_t height;                /* map +0x14 */
	uint32_t default_cell;          /* map +0x18, low byte */
	float level;                    /* owner +0x178 */
	float floor;                    /* owner +0x17c */
	uint32_t reserved;
} TmnfTrackWaterHeader;

_Static_assert(sizeof(TmnfTrackSection) == 0x10, "track section size");
_Static_assert(sizeof(TmnfTrackHeader) == 0x150, "track header size");
_Static_assert(sizeof(TmnfTrackWaterHeader) == 0x28, "track water size");
_Static_assert(sizeof(TmnfTrackStaticEntry) == 0x60, "track entry size");
_Static_assert(sizeof(TmnfTrackSurface) == 0x18, "track surface size");
_Static_assert(sizeof(TmnfTrackMesh) == 0x38, "track mesh size");
_Static_assert(sizeof(TmnfTrackMaterialData) == 0x08, "track material size");
_Static_assert(sizeof(TmnfTrackCollisionPair) == 0x14, "track pair size");

_Static_assert(
	sizeof(TmnfTrackStaticEntry) == sizeof(HmsStaticCollisionEntry),
	"snapshot/native static-entry layouts differ");
_Static_assert(
	offsetof(TmnfTrackStaticEntry, surface_rel) ==
		offsetof(HmsStaticCollisionEntry, surface),
	"snapshot/native surface pointer offsets differ");
_Static_assert(
	sizeof(TmnfTrackSurface) == sizeof(CPlugSurface),
	"snapshot/native surface layouts differ");
_Static_assert(
	sizeof(TmnfTrackMesh) == sizeof(GmSurfMesh),
	"snapshot/native mesh layouts differ");

/*
 * GmMap2<unsigned char> water map reached by 0x007C2910 through
 * CHmsZone+0x168 (+0x154 map, +0x178 surface level, +0x17c floor), recorded
 * live by the tracer (TMNF_TRACK_WATER, TMNF_TRACK_WATER_CELLS). Stadium is a
 * 32x32 grid of 32 m cells at level 8.0 whose set cells are exactly the blocks
 * with physical-material-13 faces; Desert is 45x45 at 62.0 and has set cells
 * without any such face, so the cells are taken from the game, not derived.
 */
enum {
	TMNF_WATER_MATERIAL = 13,
};

typedef struct TmnfTrackWater {
	float cell_x;
	float cell_z;
	float origin_x;
	float origin_z;
	uint32_t width;
	uint32_t height;
	uint8_t default_cell;
	const uint8_t *cells;           /* width * height, from the snapshot */
	uint32_t cell_count;            /* number of set cells */
	float level;
	float floor;
} TmnfTrackWater;

typedef struct {
	void *mapping;
	size_t mapping_size;
	const TmnfTrackHeader *header;
	const HmsStaticCollisionEntry *entries;
	uint32_t entry_count;
	const CPlugSurface *surfaces;
	uint32_t surface_count;
	const GmSurfMesh *meshes;
	uint32_t mesh_count;
	const TmnfTrackMaterialData *materials;
	const TmnfTrackCollisionPair *collision_pairs;
	uint32_t collision_pair_count;
	const GmIso4 *corpus_isos;      /* indexed by corpus_ref - 1 */
	uint32_t corpus_count;
	TmnfTrackWater water;
	/* Region-restricted static tree copies per grid cell (see track.c).
	 * cell_count == 0 when the track has no bounded leaves. */
	TmnfStaticGrid grid;
	size_t grid_node_count;
	size_t grid_mesh_slot_count;
	/* Response bodies of the static corpora, indexed by corpus_ref; present
	 * marks the referenced ones. Immutable, shared by every world. */
	CHmsResponseBody *static_response_bodies;
	uint8_t *static_response_present;
	uint32_t static_response_count;
} TmnfTrack;

/*
 * expected_track_sha256 is the SHA-256 of the source Challenge.Gbx. The
 * loader also requires the snapshot to target the canonical TMNF 2.11.26
 * executable.
 *
 * Loading performs all pointer fixups, then makes the complete mapping
 * read-only. One loaded handle may be bound into any number of worlds and
 * read concurrently. The caller must keep it alive until those worlds are
 * destroyed.
 */
TmnfTrack *TmnfTrack_Load(
	const char *path, const uint8_t expected_track_sha256[32]);
void TmnfTrack_BindStaticGroup(
	const TmnfTrack *track, CHmsCollisionManager_SGroup *group);
void TmnfTrack_Unload(TmnfTrack *track);

#ifdef __cplusplus
}
#endif

#endif /* TMNF_TRACK_H */
