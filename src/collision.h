#ifndef TMNF_COLLISION_H
#define TMNF_COLLISION_H

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "tmnf_hd.h"

#include "fastbuffer.h"
#include "gm.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
	GmVec3 center;
	GmVec3 half_extent;
} GmBoxAligned;

typedef enum {
	GM_SURF_SPHERE = 0,
	GM_SURF_ELLIPSOID = 1,
	GM_SURF_PLANE = 2,
	GM_SURF_QUAD_HEIGHT = 3,
	GM_SURF_TRIANGLE_HEIGHT = 4,
	GM_SURF_POLYGON = 5,
	GM_SURF_BOX = 6,
	GM_SURF_MESH = 7,
	GM_SURF_CYLINDER = 8,
	GM_SURF_TYPE_COUNT = 9
} GmSurfType;

typedef struct {
	uint32_t vtable;
	uint16_t material_index;
	uint8_t type;
	uint8_t reserved;
} GmSurf;

typedef struct {
	GmSurf base;
	float radius;
} GmSurfSphere;

typedef struct {
	GmSurf base;
	GmVec3 radii;
} GmSurfEllipsoid;

typedef struct {
	GmSurf base;
	GmVec3 center;
	GmVec3 half_extent;
} GmSurfBox;

typedef struct {
	GmSurf base;
	GmVec3 vertices[4];
	uint8_t vertex_count;
	uint8_t reserved0[3];
	GmVec3 normal;
	uint32_t one_sided;
} GmSurfPolygon;

typedef struct {
	GmVec3 normal;
	uint32_t reserved0c;
	uint32_t vertex[3];
	uint16_t material_index;
	uint16_t reserved1e;
} GmSurfMeshFace;

typedef struct {
	uint32_t skip_count;
	GmBoxAligned box;
	uint32_t face_index;
} GmSurfMeshNode;

typedef struct {
	GmSurf base;
	uint32_t vertex_count;
	GmVec3 *vertices;
	uint32_t face_count;
	GmSurfMeshFace *faces;
	uint32_t node_count;
	GmSurfMeshNode *nodes;
} GmSurfMesh;

/* Sphere/mesh edge directions and side planes depend only on the immutable
 * triangle. Keep the exact scalar operations shared by loading and fallback
 * queries, so caching does not change their rounding. */
typedef struct {
	GmVec3 direction;
	GmVec3 side;
} TmnfSphereMeshEdge;

typedef struct {
	float xyz[3][4];
	TmnfSphereMeshEdge edges[3];
	float padding[2];
} TmnfSphereFaceEdges;

TMNF_HD static inline TmnfSphereMeshEdge TmnfSphereMeshEdge_Prepare(
	const GmVec3 *vertex, const GmVec3 *next, const GmVec3 *normal) {
	GmVec3 edge = {
		x87_sub(next->x, vertex->x),
		x87_sub(next->y, vertex->y),
		x87_sub(next->z, vertex->z),
	};
	float squared = x87_add(x87_add(x87_mul(edge.x, edge.x),
		x87_mul(edge.y, edge.y)), x87_mul(edge.z, edge.z));
	if (0x1.b7cdfcp-34f < squared) {
		float length = x87_sqrt(squared);
		float inverse = x87_rcp(length);
		edge.x = F(inverse) * F(edge.x);
		edge.y = F(edge.y) * F(inverse);
		edge.z = F(inverse) * F(edge.z);
	}
	GmVec3 side = {
		x87_sub(x87_mul(edge.y, normal->z), x87_mul(edge.z, normal->y)),
		x87_sub(x87_mul(normal->x, edge.z), x87_mul(normal->z, edge.x)),
		x87_sub(x87_mul(normal->y, edge.x), x87_mul(edge.y, normal->x)),
	};
	TmnfSphereMeshEdge result = { edge, side };
	return result;
}

typedef struct {
	GmVec3 separation;
	GmVec3 normal;
	GmVec3 position;
	uint16_t material1;
	uint16_t material2;
	uint32_t flags;
	GmVec3 face_normal;
} GmCollision;

typedef struct SHmsPhysicalCollision {
	uint32_t corpus1;
	uint32_t tree1;
	uint32_t corpus2;
	uint32_t tree2;
	GmCollision collision;
	uint32_t material;
} SHmsPhysicalCollision;

typedef struct {
	uint32_t vtable;
	CFastBufferLayout32 collisions;
} CHmsCollisionBufferLayout32;

typedef struct {
	CHmsCollisionBufferLayout32 base;
	uint32_t active;
} SHmsSphereBufferContactLayout32;

typedef struct {
	CFastBuffer_SHmsPhysicalCollision collisions;
} CHmsCollisionBuffer;

typedef struct {
	CHmsCollisionBuffer base;
	uint32_t active;
} SHmsSphereBufferContact;

typedef struct {
	uint32_t surface1;
	uint32_t iso1;
	uint32_t surface2;
	uint32_t iso2;
} SPlugSurfaceLocatedPairLayout32;

/* A mesh list is a header slot (node count) followed by node_count entries:
 * indices into the mesh's flattened node array, restricted to a mesh-space
 * region, with list-local skip counts. It yields the same visit sequence as
 * the full array for any query box lying inside that region (see track.c). */
typedef struct {
	uint32_t node;          /* index into GmSurfMesh.nodes */
	uint32_t skip;
} TmnfMeshListEntry;

typedef union {
	uint32_t node_count;
	TmnfMeshListEntry entry;
} TmnfMeshPoolSlot;

/* Uniform grid over one mesh in mesh space; each cell holds the slot of the
 * mesh list built for the cell expanded by that level's query budget E.
 * Levels are ordered by growing budget; a query whose box center lies in a
 * cell and whose half extents are all <= max_half (E less the rounding
 * margin) lies inside that cell's list region and uses it. */
enum { TMNF_MESH_GRID_LEVELS = 2 };

typedef struct {
	double origin[3];
	double cell_size;
	double inv_cell_size;
	double max_half;
	uint32_t dims[3];
	uint32_t cell_count;
	const uint32_t *cells;               /* header slot per cell */
} TmnfMeshGridLevel;

typedef struct {
	TmnfMeshGridLevel levels[TMNF_MESH_GRID_LEVELS];
	const TmnfMeshPoolSlot *pool;
	const TmnfSphereFaceEdges *sphere_edges; /* per face, immutable */
} TmnfMeshGrid;

/* Optional per-query acceleration for a static mesh (built by track.c from the
 * immutable track, see TmnfStaticGrid). inverse_iso is GmIso4_SetInverse(iso)
 * precomputed; mesh_grid, when non-NULL, restricts the node scan. */
typedef struct {
	const GmIso4 *inverse_iso;
	const TmnfMeshGrid *mesh_grid;
} TmnfMeshQueryAccel;

typedef struct {
	GmSurf *surf;
	const GmIso4 *iso;
	uint32_t is_located;
	const TmnfMeshQueryAccel *accel;
} LocatedGmSurf;

/* One node of a region-restricted copy of the static collision tree. */
enum {
	TMNF_CELL_NODE_INTERNAL = UINT32_MAX,   /* not a leaf */
	TMNF_CELL_NODE_EMPTY = UINT32_MAX - 1,  /* leaf no query here can hit */
};

typedef struct {
	GmBoxAligned box;
	uint32_t skip;
	uint32_t entry_index;
} TmnfStaticCellNode;

typedef struct {
	double origin[3];
	double cell_size;
	double inv_cell_size;
	double expand;          /* query box half-extent budget, E */
	double margin;          /* float rounding guard, delta */
	uint32_t dims[3];
	uint32_t cell_count;
	const uint32_t *cell_offsets;        /* per cell, into nodes */
	const uint32_t *cell_counts;         /* per cell; lists are shared */
	const TmnfStaticCellNode *nodes;
	const TmnfMeshPoolSlot *mesh_pool;
	const TmnfMeshGrid *mesh_grids;      /* one per track mesh */
	const TmnfMeshGrid *const *entry_mesh_grids; /* per entry, may be NULL */
	const GmIso4 *entry_inverse_isos;    /* one per static entry */
} TmnfStaticGrid;

_Static_assert(sizeof(TmnfStaticCellNode) == 32, "static cell node size");
_Static_assert(sizeof(TmnfMeshPoolSlot) == 8, "mesh pool slot size");

typedef struct CPlugSurface {
	GmSurf *geom;
	const uint8_t *material_ids;
	uint32_t material_count;
} CPlugSurface;

typedef struct {
	uint32_t vtable;
	uint8_t pad04[0x10];
	uint32_t geom;
	CFastBufferLayout32 materials;
} CPlugSurfaceLayout32;

typedef struct {
	uint32_t vtable;
	uint8_t pad04[0x30];
	uint32_t surf;
	uint32_t reserved38;
} CPlugSurfaceGeomLayout32;

typedef struct CPlugTree CPlugTree;

struct CPlugTree {
	uint32_t object_ref;
	uint32_t flags;
	GmBoxAligned box;
	GmIso4 local_iso;
	CPlugSurface *surface;
	SHmsSphereBufferContact *contact_buffer;
	uint32_t child_count;
	CPlugTree **children;
};

typedef struct {
	uint32_t vtable;
	uint8_t pad04[0x30];
	GmBoxAligned box;
	uint32_t reserved4c;
	uint32_t collision_buffer;
	uint8_t pad54[0x08];
	GmIso4 local_iso;
	uint32_t surface;
	uint8_t pad90[0x0c];
	uint32_t flags;
	uint8_t pad_a0[0x0c];
} CPlugTreeLayout32;

typedef struct CHmsCorpus {
	uint32_t object_ref;
	uint32_t flags;
	uint32_t group_index;
	void *dyna;
	GmIso4 local_iso;
	const GmIso4 *live_iso;
	CPlugTree *tree;
} CHmsCorpus;

typedef struct {
	uint8_t pad00[0x18];
	GmIso4 local_iso;
	uint32_t scene_ref;
	uint8_t pad4c[0x08];
	uint32_t group_index;
	uint32_t dyna;
} CHmsCorpusLayout32;

typedef struct {
	int32_t *data;
	uint32_t rows;
	uint32_t columns;
	uint32_t reserved;
	uint32_t stride;
} CFastRectTableInt;

typedef struct {
	uint32_t data;
	uint32_t rows;
	uint32_t columns;
	uint32_t reserved;
	uint32_t stride;
} CFastRectTableIntLayout32;

typedef struct CHmsCollisionManager_SGroup CHmsCollisionManager_SGroup;
typedef struct CollisionRuntime CollisionRuntime;

typedef struct {
	CHmsCollisionManager_SGroup *group;
	uint32_t material_ref;
	CFastRectTableInt perform;
} CPlugMaterial_SDeviceMat;

typedef struct {
	uint32_t group;
	uint32_t material_ref;
	CFastRectTableIntLayout32 perform;
} CPlugMaterial_SDeviceMatLayout32;

typedef struct {
	uint32_t skip_count;
	GmBoxAligned box;
	GmIso4 iso;
	uint32_t tree_flags;
	CPlugSurface *surface;
	uint32_t tree_ref;
	uint32_t corpus_ref;
} HmsStaticCollisionEntry;

typedef struct {
	uint32_t skip_count;
	GmBoxAligned box;
	GmIso4 iso;
	uint32_t surface;
	uint32_t tree;
	uint32_t corpus;
} HmsStaticCollisionEntryLayout32;

struct CHmsCollisionManager_SGroup {
	const CollisionRuntime *runtime;
	uint32_t corpus_count;
	CHmsCorpus **corpora;
	float *speed_sq;
	uint32_t device_mat_count;
	CPlugMaterial_SDeviceMat *device_mats;
	uint32_t priority;
	uint32_t is_static;
	uint32_t static_entry_count;
	HmsStaticCollisionEntry *static_entries;
	const TmnfStaticGrid *static_grid;   /* NULL: scan static_entries */
};

typedef struct {
	uint8_t pad00[0x0c];
	CFastBufferLayout32 corpora;
	CFastBufferLayout32 speed_sq;
	uint32_t device_mat_count;
	uint32_t device_mats;
	uint32_t reserved2c;
	CFastBufferLayout32 static_entries;
	uint32_t priority;
	uint32_t is_static;
} CHmsCollisionManager_SGroupLayout32;

typedef struct {
	CPlugTree *tree1;
	const GmIso4 *iso1;
	CPlugTree *tree2;
	const GmIso4 *iso2;
} SPlugTreeLocatedPair;

typedef int (*GmCollisionHandler)(
	const LocatedGmSurf *, const LocatedGmSurf *, CHmsCollisionBuffer *);

typedef struct {
	GmCollisionHandler sphere_box;
	GmCollisionHandler sphere_ellipsoid;
	GmCollisionHandler sphere_polygon;
	GmCollisionHandler ellipsoid_polygon;
	GmCollisionHandler sphere_mesh;
	GmCollisionHandler ellipsoid_mesh;
	GmCollisionHandler box_box;
	GmCollisionHandler box_mesh;
	GmCollisionHandler mesh_mesh;
} CollisionShapeDispatch;

struct CollisionRuntime {
	void (*get_linear_speed)(const CHmsCorpus *, GmVec3 *);
	CollisionShapeDispatch shapes;
};

typedef struct {
	CHmsCollisionManager_SGroup groups[5];
	const CollisionRuntime *runtime;
	uint32_t current_material;
	uint32_t current_corpus1;
	uint32_t current_corpus2;
	CHmsCollisionBuffer *general_buffer;
	CHmsCollisionManager_SGroup *static_group;
	SHmsSphereBufferContact **merge_buffers;
	uint32_t merge_buffer_count;
	uint32_t merge_buffer_capacity;
} CHmsCollisionManager_SZone;

typedef struct {
	CHmsCollisionManager_SGroupLayout32 groups[5];
	uint8_t pad154[0x30];
	uint32_t current_material;
	uint32_t current_corpus1;
	uint32_t current_corpus2;
	uint32_t general_buffer;
	uint32_t static_group;
	uint32_t reserved198;
	uint32_t dependency;
	CFastBufferLayout32 merge_buffers;
} CHmsCollisionManager_SZoneLayout32;

_Static_assert(sizeof(GmBoxAligned) == 0x18, "GmBoxAligned size");
_Static_assert(offsetof(GmBoxAligned, center) == 0x00, "GmBoxAligned center");
_Static_assert(offsetof(GmBoxAligned, half_extent) == 0x0c, "GmBoxAligned half extent");
_Static_assert(sizeof(GmSurf) == 0x08, "GmSurf header size");
_Static_assert(offsetof(GmSurf, material_index) == 0x04, "GmSurf material");
_Static_assert(offsetof(GmSurf, type) == 0x06, "GmSurf type");
_Static_assert(sizeof(GmSurfSphere) == 0x0c, "GmSurfSphere size");
_Static_assert(offsetof(GmSurfSphere, radius) == 0x08, "GmSurfSphere radius");
_Static_assert(sizeof(GmSurfEllipsoid) == 0x14, "GmSurfEllipsoid size");
_Static_assert(offsetof(GmSurfEllipsoid, radii) == 0x08, "GmSurfEllipsoid radii");
_Static_assert(sizeof(GmSurfBox) == 0x20, "GmSurfBox size");
_Static_assert(offsetof(GmSurfBox, center) == 0x08, "GmSurfBox center");
_Static_assert(offsetof(GmSurfBox, half_extent) == 0x14, "GmSurfBox extent");
_Static_assert(sizeof(GmSurfPolygon) == 0x4c, "GmSurfPolygon size");
_Static_assert(offsetof(GmSurfPolygon, vertex_count) == 0x38, "polygon vertex count");
_Static_assert(offsetof(GmSurfPolygon, normal) == 0x3c, "polygon normal");
_Static_assert(offsetof(GmSurfPolygon, one_sided) == 0x48, "polygon one sided");
_Static_assert(sizeof(GmSurfMeshFace) == 0x20, "mesh face size");
_Static_assert(offsetof(GmSurfMeshFace, vertex) == 0x10, "mesh face vertices");
_Static_assert(offsetof(GmSurfMeshFace, material_index) == 0x1c,
	"mesh face material");
_Static_assert(sizeof(GmSurfMeshNode) == 0x20, "mesh node size");
_Static_assert(offsetof(GmSurfMeshNode, box) == 0x04, "mesh node box");
_Static_assert(offsetof(GmSurfMeshNode, face_index) == 0x1c, "mesh node face");
_Static_assert(sizeof(GmCollision) == 0x38, "GmCollision size");
_Static_assert(offsetof(GmCollision, normal) == 0x0c, "collision normal");
_Static_assert(offsetof(GmCollision, position) == 0x18, "collision position");
_Static_assert(offsetof(GmCollision, material1) == 0x24, "collision material1");
_Static_assert(offsetof(GmCollision, material2) == 0x26, "collision material2");
_Static_assert(offsetof(GmCollision, flags) == 0x28, "collision flags");
_Static_assert(offsetof(GmCollision, face_normal) == 0x2c, "collision face normal");
_Static_assert(sizeof(SHmsPhysicalCollision) == 0x4c, "physical collision size");
_Static_assert(offsetof(SHmsPhysicalCollision, collision) == 0x10, "physical collision data");
_Static_assert(offsetof(SHmsPhysicalCollision, material) == 0x48, "physical material");
_Static_assert(sizeof(CHmsCollisionBufferLayout32) == 0x10, "collision buffer layout");
_Static_assert(sizeof(SHmsSphereBufferContactLayout32) == 0x14, "sphere buffer layout");
_Static_assert(offsetof(SHmsSphereBufferContactLayout32, active) == 0x10, "sphere active");
_Static_assert(sizeof(SPlugSurfaceLocatedPairLayout32) == 0x10, "surface pair layout");
_Static_assert(sizeof(CPlugSurfaceLayout32) == 0x24, "CPlugSurface size");
_Static_assert(offsetof(CPlugSurfaceLayout32, geom) == 0x14, "CPlugSurface geom");
_Static_assert(offsetof(CPlugSurfaceLayout32, materials) == 0x18, "surface materials");
_Static_assert(sizeof(CPlugSurfaceGeomLayout32) == 0x3c, "CPlugSurfaceGeom size");
_Static_assert(offsetof(CPlugSurfaceGeomLayout32, surf) == 0x34, "surface geom surf");
_Static_assert(sizeof(CPlugTreeLayout32) == 0xac, "CPlugTree size");
_Static_assert(offsetof(CPlugTreeLayout32, box) == 0x34, "tree box");
_Static_assert(offsetof(CPlugTreeLayout32, collision_buffer) == 0x50,
	"tree collision buffer");
_Static_assert(offsetof(CPlugTreeLayout32, local_iso) == 0x5c, "tree local iso");
_Static_assert(offsetof(CPlugTreeLayout32, surface) == 0x8c, "tree surface");
_Static_assert(offsetof(CPlugTreeLayout32, flags) == 0x9c, "tree flags");
_Static_assert(sizeof(CHmsCorpusLayout32) == 0x5c, "CHmsCorpus size");
_Static_assert(offsetof(CHmsCorpusLayout32, local_iso) == 0x18, "corpus iso");
_Static_assert(offsetof(CHmsCorpusLayout32, scene_ref) == 0x48, "corpus scene");
_Static_assert(offsetof(CHmsCorpusLayout32, group_index) == 0x54, "corpus group index");
_Static_assert(offsetof(CHmsCorpusLayout32, dyna) == 0x58, "corpus dyna");
_Static_assert(sizeof(CFastRectTableIntLayout32) == 0x14, "rect table size");
_Static_assert(offsetof(CFastRectTableIntLayout32, stride) == 0x10, "rect stride");
_Static_assert(sizeof(CPlugMaterial_SDeviceMatLayout32) == 0x1c, "device mat size");
_Static_assert(offsetof(CPlugMaterial_SDeviceMatLayout32, perform) == 0x08,
	"device mat table");
_Static_assert(sizeof(HmsStaticCollisionEntryLayout32) == 0x58, "static entry size");
_Static_assert(offsetof(HmsStaticCollisionEntryLayout32, box) == 0x04, "static box");
_Static_assert(offsetof(HmsStaticCollisionEntryLayout32, iso) == 0x1c, "static iso");
_Static_assert(offsetof(HmsStaticCollisionEntryLayout32, surface) == 0x4c,
	"static surface");
_Static_assert(sizeof(CHmsCollisionManager_SGroupLayout32) == 0x44, "collision group size");
_Static_assert(offsetof(CHmsCollisionManager_SGroupLayout32, corpora) == 0x0c,
	"group corpora");
_Static_assert(offsetof(CHmsCollisionManager_SGroupLayout32, speed_sq) == 0x18,
	"group speed");
_Static_assert(offsetof(CHmsCollisionManager_SGroupLayout32, device_mats) == 0x28,
	"group device mats");
_Static_assert(offsetof(CHmsCollisionManager_SGroupLayout32, static_entries) == 0x30,
	"group static entries");
_Static_assert(offsetof(CHmsCollisionManager_SGroupLayout32, priority) == 0x3c,
	"group priority");
_Static_assert(offsetof(CHmsCollisionManager_SGroupLayout32, is_static) == 0x40,
	"group static");
_Static_assert(sizeof(CHmsCollisionManager_SZoneLayout32) == 0x1ac, "collision zone size");
_Static_assert(offsetof(CHmsCollisionManager_SZoneLayout32, current_material) == 0x184,
	"zone material");
_Static_assert(offsetof(CHmsCollisionManager_SZoneLayout32, current_corpus1) == 0x188,
	"zone corpus1");
_Static_assert(offsetof(CHmsCollisionManager_SZoneLayout32, current_corpus2) == 0x18c,
	"zone corpus2");
_Static_assert(offsetof(CHmsCollisionManager_SZoneLayout32, general_buffer) == 0x190,
	"zone buffer");
_Static_assert(offsetof(CHmsCollisionManager_SZoneLayout32, static_group) == 0x194,
	"zone static group");
_Static_assert(offsetof(CHmsCollisionManager_SZoneLayout32, merge_buffers) == 0x1a0,
	"zone merge buffers");

TMNF_HD void CollisionRuntime_Init(CollisionRuntime *runtime);
TMNF_HD void CHmsCollisionManager_SZone_Init(
	CHmsCollisionManager_SZone *zone, const CollisionRuntime *runtime);
TMNF_HD void CHmsCollisionManager_SZone_Destroy(CHmsCollisionManager_SZone *zone);

/* A query box held in the form the overlap test consumes, so a box that is
 * tested against many others is loaded (or, for a box just produced by
 * GmBoxAligned_SetMultQuery, never stored and reloaded) once. */
#if TMNF_SSE
typedef struct {
	__m128 center;          /* [cx cy cz -] */
	__m128 half;            /* [hx hy hz -], ready for every overlap test */
} GmBoxQuery;

static inline GmBoxQuery GmBoxAligned_Query(const GmBoxAligned *box) {
	GmBoxQuery q;
	q.center = _mm_loadu_ps(&box->center.x);
	__m128 tail = _mm_loadu_ps(&box->center.z);
	q.half = _mm_shuffle_ps(tail, tail, _MM_SHUFFLE(3, 3, 2, 1));
	return q;
}

/* 0x00537530  Tests overlap of two center/half-extent AABBs. */
/* UNVALIDATED */
/* Three axes at once. The per-lane subtract, sign clear, add and compare
 * are the same IEEE binary32 operations as the scalar form below, and the
 * result does not depend on the order the axes are examined in (NaN fails
 * every axis). Loads: [cx cy cz hx] from the center and [cz hx hy hz] from
 * center.z, both inside the 24-byte box. */
static inline int GmBoxQuery_TestInter(
	const GmBoxQuery *self, const GmBoxAligned *other) {
	__m128 other_center = _mm_loadu_ps(&other->center.x);
	__m128 other_tail = _mm_loadu_ps(&other->center.z);
	__m128 other_half = _mm_shuffle_ps(
		other_tail, other_tail, _MM_SHUFFLE(3, 3, 2, 1));
	__m128 delta = _mm_sub_ps(other_center, self->center);
	__m128 absolute_delta = _mm_andnot_ps(_mm_set1_ps(-0.0f), delta);
	__m128 extent = _mm_add_ps(other_half, self->half);
	int inside = _mm_movemask_ps(_mm_cmple_ps(absolute_delta, extent));
	return (inside & 7) == 7;
}

/* 0x008E5230  Transforms an AABB by an affine isometry. */
/* VALIDATED: 714/714 golden records. */
/* center = ((col0*cx + col1*cy) + col2*cz) + t and half = (|col0|*hx +
 * |col1|*hy) + |col2|*hz, the same association per component as the scalar
 * form below. Columns are extracted once for both. */
static inline GmBoxQuery GmBoxAligned_SetMultQuery(
	GmBoxAligned *self, const GmBoxAligned *source, const GmIso4 *iso) {
	GmIsoChunks k = gm_iso_load(iso);
	__m128 c0, c1, c2;
	gm_iso_cols(k, &c0, &c1, &c2);
	__m128 center = _mm_add_ps(gm_combine3(c0, source->center.x,
		c1, source->center.y, c2, source->center.z), gm_iso_translation(k));
	__m128 sign = _mm_set1_ps(-0.0f);
	__m128 half = gm_combine3(_mm_andnot_ps(sign, c0), source->half_extent.x,
		_mm_andnot_ps(sign, c1), source->half_extent.y,
		_mm_andnot_ps(sign, c2), source->half_extent.z);
	gm_store3(&self->center.x, center);
	gm_store3(&self->half_extent.x, half);
	GmBoxQuery q;
	q.center = center;
	q.half = half;
	return q;
}
#else
typedef struct {
	GmBoxAligned box;
} GmBoxQuery;

TMNF_HD static inline GmBoxQuery GmBoxAligned_Query(const GmBoxAligned *box) {
	GmBoxQuery q = { *box };
	return q;
}

TMNF_HD static inline int GmBoxQuery_TestInter(
	const GmBoxQuery *self_query, const GmBoxAligned *other) {
	const GmBoxAligned *self = &self_query->box;
#if defined(__CUDA_ARCH__)
	/* The SSE form's binary32 operations (a float sum or difference of two
	 * floats rounded from double is the binary32 result, 53 >= 2 * 24 + 2);
	 * the device has no fast double path. */
	float dz = fabsf(x87_sub(other->center.z, self->center.z));
	float ez = x87_add(other->half_extent.z, self->half_extent.z);
	if (!(dz <= ez)) {
		return 0;
	}
	float dy = fabsf(x87_sub(other->center.y, self->center.y));
	float ey = x87_add(other->half_extent.y, self->half_extent.y);
	if (!(dy <= ey)) {
		return 0;
	}
	float dx = fabsf(x87_sub(other->center.x, self->center.x));
	float ex = x87_add(other->half_extent.x, self->half_extent.x);
	return dx <= ex;
#else
	float delta = (float)(F(other->center.z) - F(self->center.z));
	float absolute_delta = (float)fabs(F(delta));
	float extent =
		(float)(F(other->half_extent.z) + F(self->half_extent.z));
	if (!(absolute_delta <= extent)) {
		return 0;
	}

	delta = (float)(F(other->center.y) - F(self->center.y));
	absolute_delta = (float)fabs(F(delta));
	extent = (float)(F(other->half_extent.y) + F(self->half_extent.y));
	if (!(absolute_delta <= extent)) {
		return 0;
	}

	delta = (float)(F(other->center.x) - F(self->center.x));
	absolute_delta = (float)fabs(F(delta));
	extent = (float)(F(other->half_extent.x) + F(self->half_extent.x));
	return absolute_delta <= extent;
#endif
}

TMNF_HD static inline GmBoxQuery GmBoxAligned_SetMultQuery(
	GmBoxAligned *self, const GmBoxAligned *source, const GmIso4 *iso) {
	self->center.x = x87_add(
		x87_add(
			x87_add(
				x87_mul(source->center.x, iso->m[0]),
				x87_mul(iso->m[1], source->center.y)),
			x87_mul(iso->m[2], source->center.z)),
		iso->t[0]);
	self->center.y = x87_add(
		x87_add(
			x87_add(
				x87_mul(iso->m[4], source->center.y),
				x87_mul(source->center.x, iso->m[3])),
			x87_mul(iso->m[5], source->center.z)),
		iso->t[1]);
	self->center.z = x87_add(
		x87_add(
			x87_add(
				x87_mul(iso->m[7], source->center.y),
				x87_mul(iso->m[6], source->center.x)),
			x87_mul(iso->m[8], source->center.z)),
		iso->t[2]);

	self->half_extent.x = x87_add(
		x87_add(
			x87_mul(fabsf(iso->m[1]), source->half_extent.y),
			x87_mul(fabsf(iso->m[0]), source->half_extent.x)),
		x87_mul(fabsf(iso->m[2]), source->half_extent.z));
	self->half_extent.y = x87_add(
		x87_add(
			x87_mul(fabsf(iso->m[4]), source->half_extent.y),
			x87_mul(fabsf(iso->m[3]), source->half_extent.x)),
		x87_mul(fabsf(iso->m[5]), source->half_extent.z));
	self->half_extent.z = x87_add(
		x87_add(
			x87_mul(fabsf(iso->m[7]), source->half_extent.y),
			x87_mul(fabsf(iso->m[6]), source->half_extent.x)),
		x87_mul(fabsf(iso->m[8]), source->half_extent.z));
	return GmBoxAligned_Query(self);
}
#endif

TMNF_HD static inline int GmBoxAligned_TestInter(
	const GmBoxAligned *self, const GmBoxAligned *other) {
	GmBoxQuery q = GmBoxAligned_Query(self);
	return GmBoxQuery_TestInter(&q, other);
}

TMNF_HD static inline void GmBoxAligned_SetMult(
	GmBoxAligned *self, const GmBoxAligned *source, const GmIso4 *iso) {
	(void)GmBoxAligned_SetMultQuery(self, source, iso);
}

/* 0x00538090  Initializes a collision buffer with capacity 50. */
/* UNVALIDATED */
TMNF_HD CHmsCollisionBuffer *CHmsCollisionBuffer_Init(CHmsCollisionBuffer *self);
TMNF_HD void CHmsCollisionBuffer_Destroy(CHmsCollisionBuffer *self);
TMNF_HD GmCollision *CHmsCollisionBuffer_AddCollision(CHmsCollisionBuffer *self);
TMNF_HD GmCollision *CHmsCollisionBuffer_GetCollision(
	CHmsCollisionBuffer *self, uint32_t index);
TMNF_HD uint32_t CHmsCollisionBuffer_GetCount(const CHmsCollisionBuffer *self);

/* 0x00539880  Initializes a mergeable per-sphere collision buffer. */
/* UNVALIDATED */
TMNF_HD SHmsSphereBufferContact *SHmsSphereBufferContact_Init(
	SHmsSphereBufferContact *self);

/* 0x00538100  Merges a per-sphere buffer into the destination. */
/* UNVALIDATED */
TMNF_HD void SHmsSphereBufferContact_MergeAndAddToCollisions(
	SHmsSphereBufferContact *self, CHmsCollisionBuffer *destination);

/* 0x00547C80  Orders physical collisions for response processing. */
/* UNVALIDATED */
TMNF_HD int SHmsPhysicalCollision_Compare(
	const SHmsPhysicalCollision *a, const SHmsPhysicalCollision *b);

/* 0x008F49D0  Computes sphere/sphere overlap and contact data. */
/* UNVALIDATED */
TMNF_HD int GmCollision_Sphere_Sphere(
	const LocatedGmSurf *sphere1, const LocatedGmSurf *sphere2,
	CHmsCollisionBuffer *buffer);

/* 0x008EA2D0  Computes sphere contacts against a triangle mesh. */
/* UNVALIDATED */
TMNF_HD int GmCollision_Sphere_Mesh(
	const LocatedGmSurf *sphere, const LocatedGmSurf *mesh,
	CHmsCollisionBuffer *buffer);

/* 0x008EADC0  Computes ellipsoid contacts against a triangle mesh. */
/* VALIDATED: 128/128 graph-complete golden records. */
TMNF_HD int GmCollision_Ellipsoid_Mesh(
	const LocatedGmSurf *ellipsoid, const LocatedGmSurf *mesh,
	CHmsCollisionBuffer *buffer);

/* 0x008F5200  Tests an oriented box against a triangle mesh. */
/* UNVALIDATED */
TMNF_HD int GmCollision_Box_Mesh(
	const LocatedGmSurf *box, const LocatedGmSurf *mesh,
	CHmsCollisionBuffer *buffer);

/* 0x008E8890  Dispatches a geometry pair and fixes reversed output. */
/* UNVALIDATED */
TMNF_HD int GmSurf_ComputeCollision(
	const LocatedGmSurf *surf1, const LocatedGmSurf *surf2,
	CHmsCollisionBuffer *buffer, const CollisionRuntime *runtime);

/* 0x00537150  Computes surface collision and remaps material indices. */
/* UNVALIDATED */
TMNF_HD int CPlugSurface_ComputeCollision(
	const CPlugSurface *surface1, const GmIso4 *iso1,
	const CPlugSurface *surface2, const GmIso4 *iso2,
	CHmsCollisionBuffer *buffer, const CollisionRuntime *runtime);

/* 0x00537E80  Computes squared speeds for one non-static group. */
/* UNVALIDATED */
TMNF_HD void CHmsCollisionManager_SGroup_ComputeNonStaticCorpusInfos(
	CHmsCollisionManager_SGroup *self);

/* 0x00537F30  Fills pairwise collision-enable tables for one group. */
/* UNVALIDATED */
TMNF_HD void CHmsCollisionManager_SGroup_ComputeIsToPerformCollisions(
	CHmsCollisionManager_SGroup *self);

/* 0x0053A0E0  Prepares all five collision groups. */
/* UNVALIDATED */
TMNF_HD void CHmsCollisionManager_SZone_PrepareCollisions(
	CHmsCollisionManager_SZone *self);

/* 0x0053A120  Collides a tree against the current static collision tree. */
/* UNVALIDATED */
/* active = 0 runs no detection: on the device the call joins the warp's
 * cooperative traversal for the lanes that do (tmnf_warp.h). */
TMNF_HD void CHmsCollisionManager_SZone_DetectCollisionBetweenTreeAndStaticCollisionTree(
	CHmsCollisionManager_SZone *self, const GmIso4 *iso, CPlugTree *tree,
	int active);

/* 0x0053A3D0  Collides tree two's root/subtree against tree one's root. */
/* UNVALIDATED */
TMNF_HD int CHmsCollisionManager_SZone_ComputeCollisionTree1RootOnly(
	CHmsCollisionManager_SZone *self, const SPlugTreeLocatedPair *pair,
	const GmBoxAligned *tree1_box);

/* 0x0053A660  Collides tree one's root/subtree against tree two's root. */
/* UNVALIDATED */
TMNF_HD int CHmsCollisionManager_SZone_ComputeCollisionTree2RootOnly(
	CHmsCollisionManager_SZone *self, const SPlugTreeLocatedPair *pair,
	const GmBoxAligned *tree2_box);

/* 0x0053A8F0  Recursively collides two located plug trees. */
/* UNVALIDATED */
TMNF_HD int CHmsCollisionManager_SZone_ComputeCollision(
	CHmsCollisionManager_SZone *self, const SPlugTreeLocatedPair *pair);

/* 0x0053AFB0  Builds root pairs and collides two corpora. */
/* UNVALIDATED */
TMNF_HD void CHmsCollisionManager_SZone_DetectCollisionBetween(
	CHmsCollisionManager_SZone *self, CHmsCorpus *corpus1,
	CHmsCorpus *corpus2);

/* 0x0053B1C0  Detects all enabled collisions for one corpus. */
/* UNVALIDATED */
TMNF_HD void CHmsCollisionManager_SZone_DetectCollisionsCorpus(
	CHmsCollisionManager_SZone *self, CHmsCollisionBuffer *buffer,
	CHmsCorpus *corpus, int active);

#ifdef __cplusplus
}
#endif

#endif
