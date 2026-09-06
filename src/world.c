#include "world.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "track.h"
#include "vehicle_model6.h"
#include "vehicle_respawn.h"
#include "vehicle_oldmodels.h"
#include "vehicle_fake_contact_mask.h"
#include "surface_material.h"
#include "vehicle_water_tuning.h"

enum {
	CAR_SIZE = 0x878,
	VEHICLE_STRUCT_SIZE = 0x50,
	TUNING_SIZE = 0x3ac,
	WHEEL_SIZE = 0x2fc,
	DYNA_SIZE = 0x344,
	PARAMS_SIZE = 0x5c,
	STATE_SIZE = 0xb4,
	ACTIVE_CURVE_COUNT = 21,
	PRIMARY_CURVE_COUNT = 2,
	GEARBOX_COUNT = 4,
	GEARBOX_VALUE_COUNT = 6,
};

enum {
	CURVE_ACCEL,
	CURVE_LATERAL_CONTACT_SLOWDOWN,
	CURVE_STEER_SLOWDOWN,
	CURVE_STEER_DRIVE_TORQUE,
	CURVE_MAX_SIDE_FRICTION,
	CURVE_ROLLOVER_LATERAL,
	CURVE_ROLLOVER_ANGLE,
	CURVE_M4_STEER_RADIUS,
	CURVE_M4_MAX_FRICTION,
	CURVE_M5_SLIPPING_ACCEL,
	CURVE_WATER_FRICTION,
	CURVE_M6_DAMPER_MODULATION,
	CURVE_M6_REAR_GEAR_ACCEL,
	CURVE_M6_ROLLOVER_RATIO,
	CURVE_M6_BURNOUT_RADIUS,
	CURVE_M6_BURNOUT_LATERAL_SPEED,
	CURVE_M6_DONUT_ROLLOVER,
	CURVE_M6_BURNOUT_ROLLOVER,
	CURVE_AIR_VERTICAL,
	CURVE_STEERING_ANGLE,
	CURVE_EFFECT,
};

/* Tuning field offsets of the 21 active curves and the two primary curves.
 * The link stage reads them on both the host and the device, and CUDA
 * cannot read a host constant array from device code, so the initializer
 * is shared between a host and a device copy. */
#define ACTIVE_CURVE_OFFSETS_INIT { \
	0x034, 0x068, 0x078, 0x0a0, 0x0ac, 0x0b8, 0x0bc, \
	0x1b4, 0x1bc, 0x1e0, 0x218, 0x224, 0x230, 0x250, \
	0x25c, 0x260, 0x288, 0x2a4, 0x36c, 0x378, 0x380, \
}
#define PRIMARY_CURVE_OFFSETS_INIT { 0x044, 0x048 }

static const uint32_t ACTIVE_CURVE_OFFSETS[ACTIVE_CURVE_COUNT] =
	ACTIVE_CURVE_OFFSETS_INIT;
static const uint32_t PRIMARY_CURVE_OFFSETS[PRIMARY_CURVE_COUNT] =
	PRIMARY_CURVE_OFFSETS_INIT;
static const uint32_t GEARBOX_OFFSETS[GEARBOX_COUNT] = {
	0x2c4, 0x2d4, 0x2e0, 0x304,
};

enum {
	VEHICLE_CURVE_OWNER_STRUCT = 0,
	VEHICLE_CURVE_OWNER_TUNING = 1,
};

#define TMNF_VEHICLE_SNAPSHOT_MAGIC "TMNFM6G1"
#define TMNF_VEHICLE_SNAPSHOT_VERSION 4u

/* Vehicle collision nodes. The snapshot lists the nodes below the recorded
 * root in pre-order; a record's kind field holds the kind in its low byte and
 * the node's direct child count in bits 8..15. Stadium cars are eight
 * ellipsoid leaves; United cars nest body shapes (a body node may carry a
 * surface and children) and use sphere wheels. */
enum {
	VEHICLE_COLLISION_TREE_BODY = 0,
	VEHICLE_COLLISION_TREE_WHEEL = 1,
	VEHICLE_COLLISION_TREE_ROOT = 2,
	VEHICLE_COLLISION_KIND_MASK = 0xffu,
	VEHICLE_COLLISION_CHILD_SHIFT = 8,
	VEHICLE_COLLISION_MAX_NODES = 8,
};

_Static_assert(VEHICLE_COLLISION_MAX_NODES <= TMNF_WORLD_CONTACT_BUFFER_COUNT,
	"one contact buffer per vehicle collision node");

#pragma pack(push, 1)

struct TmnfVehicleSnapshotHeader {
	char magic[8];
	uint32_t version;
	uint32_t total_size;
	uint32_t header_size;
	uint32_t phase;
	uint32_t tick;
	uint32_t car_id;
	uint32_t vehicle_struct_id;
	uint32_t tuning_container_id;
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
	uint32_t tuning_count;
	uint32_t active_tuning_key;
	uint32_t curve_count;
	uint32_t gearbox_count;
	uint32_t car_offset;
	uint32_t vehicle_struct_offset;
	uint32_t tuning_descriptors_offset;
	uint32_t wheels_offset;
	uint32_t dyna_offset;
	uint32_t params_offset;
	uint32_t state_offset;
	uint32_t curve_descriptors_offset;
	uint32_t gearbox_descriptors_offset;
	uint32_t ground_ids_offset;
	uint32_t ground_materials_offset;
	uint32_t model_iso_offset;
	uint32_t body_reference_offset;
	uint32_t curve_data_offset;
	uint32_t gearbox_data_offset;
	uint32_t collision_child_count;
	uint32_t collision_root_offset;
	uint32_t collision_children_offset;
	uint32_t collision_material_ids_offset;
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

struct TmnfVehicleTuningDescriptor {
	uint32_t key;
	uint32_t object_id;
	uint32_t raw_offset;
	uint32_t raw_size;
};

struct TmnfVehicleCurveDescriptor {
	uint32_t owner_kind;
	uint32_t owner_key;
	uint32_t field_offset;
	uint32_t object_id;
	uint32_t positions_id;
	uint32_t values_id;
	uint32_t count;
	int32_t interpolation;
	uint32_t positions_offset;
	uint32_t values_offset;
};

struct TmnfVehicleGearboxDescriptor {
	uint32_t owner_key;
	uint32_t field_offset;
	uint32_t buffer_id;
	uint32_t data_id;
	uint32_t count;
	uint32_t data_offset;
};

struct TmnfVehicleMaterial {
	uint32_t object_id;
	float values[4];
	/* Fake-contact descriptor read by 0x007C3C00: material +0x24 mask
	 * present, +0x28/+0x2c periods, +0x30 impulse scale, +0x34 limit. The
	 * mask is always the committed 128x128 image; the dumper checks it. */
	uint32_t fake_contact_mask;
	float fake_contact_period_x;
	float fake_contact_period_z;
	float fake_contact_impulse_scale;
	float fake_contact_impulse_limit;
};

struct TmnfVehicleCollisionTree {
	uint32_t object_id;
	uint32_t flags;
	uint32_t surface_id;
	uint32_t geometry_id;
	uint32_t kind;
	uint32_t wheel_index;
	uint32_t material_index;
	uint32_t material_count;
	uint16_t geometry_material_index;
	uint8_t geometry_type;
	uint8_t geometry_reserved;
	GmBoxAligned box;
	GmIso4 local_iso;
	uint8_t shape[24];
};

#pragma pack(pop)

_Static_assert(sizeof(struct TmnfVehicleSnapshotHeader) == 320,
	"vehicle snapshot header size");
_Static_assert(sizeof(struct TmnfVehicleTuningDescriptor) == 16,
	"vehicle tuning descriptor size");
_Static_assert(sizeof(struct TmnfVehicleCurveDescriptor) == 40,
	"vehicle curve descriptor size");
_Static_assert(sizeof(struct TmnfVehicleGearboxDescriptor) == 24,
	"vehicle gearbox descriptor size");
_Static_assert(sizeof(struct TmnfVehicleMaterial) == 40,
	"vehicle material size");
_Static_assert(sizeof(struct TmnfVehicleCollisionTree) == 132,
	"vehicle collision tree size");

static const uint8_t TMNF_21126_EXE_SHA256[32] = {
	0x38, 0x47, 0xcf, 0x9f, 0x20, 0xbf, 0xc6, 0x39,
	0x14, 0x45, 0x00, 0x60, 0xed, 0x52, 0x8c, 0x12,
	0x10, 0x4f, 0x74, 0x3d, 0x96, 0xad, 0x23, 0xd6,
	0xe7, 0x6a, 0xbd, 0x17, 0x8d, 0xe8, 0xc8, 0x4f,
};

/* The part of a world the physics never writes after World_Create: the
 * decoded vehicle (tuning, curves, materials, gear tables), the collision
 * shapes and static structures and the contexts that only hold pointers.
 * One copy is shared by every environment on the device (one address for
 * all lanes, served by L1); tests/world_cold_readonly.c proves the
 * invariant on the CPU with a read-only mapping. */
struct TmnfWorldCold {
	uint8_t *vehicle_blob;
	uint32_t vehicle_blob_size;
	const struct TmnfVehicleSnapshotHeader *vehicle_header;
	const struct TmnfVehicleTuningDescriptor *tuning_descriptors;
	const struct TmnfVehicleCurveDescriptor *curve_descriptors;
	const struct TmnfVehicleGearboxDescriptor *gearbox_descriptors;
	const struct TmnfVehicleCollisionTree *collision_root;
	const struct TmnfVehicleCollisionTree *collision_children;
	const uint8_t *collision_material_ids;
	const uint8_t *raw_car;
	const uint8_t *raw_vehicle_struct;
	const uint8_t *raw_tuning;
	const uint8_t *raw_wheels;
	const uint8_t *raw_dyna;
	const TmnfTrack *track;
	CSceneVehicleCarTuning vehicle_tuning;
	CSceneVehicleCarTuningAux aux_tuning;
	TMNFVehicleContactTuning contact_tuning;
	CSceneVehicleCarTuningCurveSet curve_set;
	CFuncKeysReal active_curves[ACTIVE_CURVE_COUNT];
	CFuncKeysReal primary_curves[PRIMARY_CURVE_COUNT];
	TMNFVehicleAuxCurve aux_curves[2];
	int steering_angle_curve_present;
	CFuncKeysReal water_impulse_curves[2];
	/* Descriptor index of each active and primary curve (find_curve at
	 * decode time), so that relinking is a lookup and not a scan of the
	 * descriptor table. */
	uint32_t active_curve_index[ACTIVE_CURVE_COUNT];
	uint32_t primary_curve_index[PRIMARY_CURVE_COUNT];
	/* CFuncKeys_Compile of every curve above, in link_curves order; the
	 * host owns it, a device world links a device copy. */
	float *curve_bounds;
	uint32_t curve_bound_count;
	TMNFVehicleGroundMaterial ground_materials[TMNF_MAX_GROUND_MATERIALS];
	const TMNFVehicleGroundMaterial *ground_material_ptrs[TMNF_MAX_GROUND_MATERIALS];
	uint32_t ground_material_ids[TMNF_SURFACE_MATERIAL_COUNT];
	CSceneVehicleCarModel6Tuning model6_tuning;
	CSceneVehicleCarModel6Context model6;
	CSceneVehicleCarOldModelsTuning oldmodels_tuning;
	CSceneVehicleCarOldModelsContext oldmodels;
	TMNFVehicleComputeTuning compute_tuning;
	TMNFVehicleComputeContext compute;
	float gear_ratios[6];
	float gear_upshift[6];
	float gear_downshift[6];
	float gear_aux[6];
	CHmsItem item;
	CHmsCorpus *item_corpora[1];
	void *item_zones[1];

	/* Node i of the snapshot's pre-order list lives in slot i; wheel
	 * leaves use the hot wheel_trees[wheel_index] (their iso moves with
	 * the wheel) so the vehicle code can find them by pointer. Sphere
	 * shapes use the ellipsoid storage (same base, radius in radii.x). */
	GmSurfEllipsoid wheel_ellipsoids[4];
	CPlugSurface wheel_surfaces[4];
	GmSurfEllipsoid body_ellipsoids[VEHICLE_COLLISION_MAX_NODES];
	CPlugSurface body_surfaces[VEHICLE_COLLISION_MAX_NODES];
	CPlugTree body_trees[VEHICLE_COLLISION_MAX_NODES];
	uint32_t body_tree_count;
	uint32_t wheel_tree_refs[4];
	uint32_t body_tree_refs[VEHICLE_COLLISION_MAX_NODES];
	CPlugTree *collision_child_slots[VEHICLE_COLLISION_MAX_NODES];
	CPlugTree *collision_child_pointers[VEHICLE_COLLISION_MAX_NODES];
	CPlugTree root_tree;
	CHmsCorpus collision_corpus;
	CHmsCorpus *dynamic_corpora[1];
	CPlugMaterial_SDeviceMat static_device;
	CollisionRuntime collision_runtime;
	CHmsResponseBody player_response_body;
	CHmsResponseMaterial response_material;
	CHmsForceFieldUniform gravity;
};

/* The mutable part: dynamics, vehicle and contact state, the per-tick
 * collision structures. This is what a device environment copies, diffs and
 * keeps in its lane-interleaved local memory. */
struct TmnfWorld {
	struct TmnfWorldCold *cold;

	CHmsStateDyna live_state;
	CHmsStateDyna committed_state;
	CHmsDynaParams dyna_params;
	CHmsDyna dyna;

	CSceneVehicleCar vehicle;
	CSceneVehicleCarWheel wheels[4];

	CSceneVehicleCarWheelAux aux_wheels[4];
	CSceneVehicleCarAuxContext aux;

	TMNFVehicleContactWheelState contact_wheels[4];
	TMNFVehicleContactTimer timer;
	TMNFVehicleContactContext contact;

	CSceneVehicleCarModel6State model6_state;
	CSceneVehicleCarOldModelsState oldmodels_state;
	TMNFVehicleComputeState compute_state;

	CPlugTree wheel_trees[4];
	float dynamic_speed_sq[1];

	/* Per-sphere contact buffers, one per collision child in root order,
	 * and the zone's merge list. Both hold fixed-capacity storage so that a
	 * device world never allocates. */
	SHmsSphereBufferContact contact_buffers[VEHICLE_COLLISION_MAX_NODES];
	SHmsSphereBufferContact *merge_storage[TMNF_WORLD_MERGE_CAPACITY];

	CHmsCollisionManager_SZone collision_zone;

	TmnfPhysicsCorpus physics_corpus;
	TmnfPhysicsWorld physics;
};

#if defined(__cplusplus)
static_assert(sizeof(struct TmnfWorld) <= TMNF_WORLD_MAX_BYTES,
	"raise TMNF_WORLD_MAX_BYTES");
#else
_Static_assert(sizeof(struct TmnfWorld) <= TMNF_WORLD_MAX_BYTES,
	"raise TMNF_WORLD_MAX_BYTES");
#endif

TMNF_HD static void world_fail(const char *message)
{
	tmnf_fail(message);
}

static void *allocate(size_t size)
{
	void *memory = calloc(1, size);
	if (memory == NULL)
		world_fail("out of memory");
	return memory;
}

TMNF_HD static uint32_t raw_u32(const uint8_t *raw, uint32_t offset)
{
	uint32_t value;
	memcpy(&value, raw + offset, sizeof(value));
	return value;
}

static int32_t raw_i32(const uint8_t *raw, uint32_t offset)
{
	int32_t value;
	memcpy(&value, raw + offset, sizeof(value));
	return value;
}

static float raw_f32(const uint8_t *raw, uint32_t offset)
{
	float value;
	memcpy(&value, raw + offset, sizeof(value));
	return value;
}

static const uint8_t *graph_section(
	const uint8_t *blob, uint32_t size, uint32_t offset, uint32_t length)
{
	if (offset > size || length > size - offset)
		world_fail("vehicle graph section is out of bounds");
	return blob + offset;
}

static uint8_t *read_vehicle_file(const char *path, uint32_t *size)
{
	FILE *file = fopen(path, "rb");
	if (file == NULL)
		world_fail("cannot open vehicle snapshot");
	if (fseek(file, 0, SEEK_END) != 0)
		world_fail("cannot size vehicle snapshot");
	long length = ftell(file);
	if (length < 0 || (unsigned long)length > UINT32_MAX)
		world_fail("vehicle snapshot size is invalid");
	if (fseek(file, 0, SEEK_SET) != 0)
		world_fail("cannot rewind vehicle snapshot");
	uint8_t *bytes = (uint8_t *)allocate((size_t)length);
	if (fread(bytes, 1, (size_t)length, file) != (size_t)length)
		world_fail("cannot read vehicle snapshot");
	if (fgetc(file) != EOF)
		world_fail("vehicle snapshot has trailing data");
	if (fclose(file) != 0)
		world_fail("cannot close vehicle snapshot");
	*size = (uint32_t)length;
	return bytes;
}

static int contains_offset(
	const uint32_t *offsets, uint32_t count, uint32_t offset)
{
	for (uint32_t i = 0; i < count; ++i) {
		if (offsets[i] == offset)
			return 1;
	}
	return 0;
}

static const struct TmnfVehicleSnapshotHeader *validate_vehicle_graph(
	const uint8_t *blob, uint32_t size,
	const uint8_t expected_track_sha256[32])
{
	if (size < sizeof(struct TmnfVehicleSnapshotHeader))
		world_fail("truncated vehicle graph");
	const struct TmnfVehicleSnapshotHeader *header =
		(const struct TmnfVehicleSnapshotHeader *)blob;
	if (memcmp(header->magic, TMNF_VEHICLE_SNAPSHOT_MAGIC, 8) != 0 ||
		header->version != TMNF_VEHICLE_SNAPSHOT_VERSION ||
		header->total_size != size ||
		header->header_size != sizeof(*header) ||
		header->phase != 0 ||
		header->car_id == 0 ||
		header->vehicle_struct_id == 0 ||
		header->tuning_container_id == 0 ||
		header->wheels_id == 0 ||
		header->item_id == 0 ||
		header->corpus_id == 0 ||
		header->dyna_id == 0 ||
		header->params_id == 0 ||
		header->state_id == 0 ||
		header->model_iso_id == 0 ||
		header->body_reference_id == 0 ||
		header->wheel_count != 4 ||
		header->ground_id_count != TMNF_SURFACE_MATERIAL_COUNT ||
		header->ground_material_count == 0 ||
		header->ground_material_count > TMNF_MAX_GROUND_MATERIALS ||
		header->tuning_count == 0 ||
		header->tuning_count > 64 ||
		header->active_tuning_key >= header->tuning_count ||
		header->curve_count == 0 ||
		header->curve_count > 4096 ||
		header->collision_child_count == 0 ||
		header->collision_child_count >
			VEHICLE_COLLISION_MAX_NODES ||
		header->tuning_count > UINT32_MAX / GEARBOX_COUNT ||
		header->gearbox_count !=
			header->tuning_count * GEARBOX_COUNT ||
		memcmp(header->source_exe_sha256,
			TMNF_21126_EXE_SHA256, 32) != 0 ||
		memcmp(header->source_track_sha256,
			expected_track_sha256, 32) != 0) {
		world_fail("invalid vehicle graph header");
	}
	(void)graph_section(blob, size, header->car_offset, CAR_SIZE);
	const uint8_t *vehicle_struct = graph_section(
		blob, size, header->vehicle_struct_offset, VEHICLE_STRUCT_SIZE);
	(void)graph_section(blob, size, header->wheels_offset, 4 * WHEEL_SIZE);
	(void)graph_section(blob, size, header->dyna_offset, DYNA_SIZE);
	(void)graph_section(blob, size, header->params_offset, PARAMS_SIZE);
	(void)graph_section(blob, size, header->state_offset, STATE_SIZE);
	(void)graph_section(blob, size, header->ground_ids_offset, 31 * 4);
	(void)graph_section(blob, size, header->ground_materials_offset,
		header->ground_material_count *
			sizeof(struct TmnfVehicleMaterial));
	(void)graph_section(
		blob, size, header->model_iso_offset, sizeof(GmIso4));
	(void)graph_section(
		blob, size, header->body_reference_offset, sizeof(GmVec3));
	(void)graph_section(blob, size, header->curve_data_offset, 0);
	(void)graph_section(blob, size, header->gearbox_data_offset, 0);
	const struct TmnfVehicleCollisionTree *collision_root =
		(const struct TmnfVehicleCollisionTree *)graph_section(
			blob, size, header->collision_root_offset,
			sizeof(*collision_root));
	const struct TmnfVehicleCollisionTree *collision_children =
		(const struct TmnfVehicleCollisionTree *)graph_section(
			blob, size, header->collision_children_offset,
			header->collision_child_count *
				sizeof(*collision_children));
	if (collision_root->object_id == 0 ||
		(collision_root->flags & 0x80u) == 0 ||
		collision_root->surface_id != 0 ||
		collision_root->geometry_id != 0 ||
		collision_root->kind != VEHICLE_COLLISION_TREE_ROOT ||
		collision_root->wheel_index != UINT32_MAX ||
		collision_root->material_count != 0) {
		world_fail("invalid vehicle collision root");
	}
	uint32_t wheel_mask = 0;
	uint32_t collision_material_count = 0;
	uint32_t open_children = 0;
	for (uint32_t i = 0; i < header->collision_child_count; ++i) {
		const struct TmnfVehicleCollisionTree *tree =
			&collision_children[i];
		uint32_t kind = tree->kind & VEHICLE_COLLISION_KIND_MASK;
		uint32_t child_count =
			tree->kind >> VEHICLE_COLLISION_CHILD_SHIFT;
		if (tree->object_id == 0 ||
			(tree->flags & 0x80u) == 0 ||
			child_count > 0xffu ||
			tree->material_index != collision_material_count) {
			world_fail("invalid vehicle collision node");
		}
		/* Pre-order: a record either fills a slot opened by an earlier
		 * parent or is a direct child of the root. */
		if (open_children != 0)
			open_children--;
		open_children += child_count;
		if (tree->surface_id == 0) {
			if (kind != VEHICLE_COLLISION_TREE_BODY ||
				tree->geometry_id != 0 ||
				tree->wheel_index != UINT32_MAX ||
				tree->material_count != 0 ||
				child_count == 0) {
				world_fail("invalid surface-less vehicle collision node");
			}
			continue;
		}
		if (tree->geometry_id == 0 ||
			(tree->flags & 0x04u) == 0 ||
			(tree->geometry_type != GM_SURF_ELLIPSOID &&
				tree->geometry_type != GM_SURF_SPHERE) ||
			tree->geometry_material_index != 0 ||
			tree->material_count != 1) {
			world_fail("invalid vehicle collision shape");
		}
		const uint8_t *material = graph_section(
			blob, size,
			header->collision_material_ids_offset +
				tree->material_index,
			tree->material_count);
		if (*material >= TMNF_SURFACE_MATERIAL_COUNT)
			world_fail("invalid vehicle collision material");
		if (kind == VEHICLE_COLLISION_TREE_BODY) {
			if (tree->wheel_index != UINT32_MAX)
				world_fail("invalid vehicle body collision node");
		} else if (kind == VEHICLE_COLLISION_TREE_WHEEL) {
			if (tree->wheel_index >= 4 ||
				child_count != 0 ||
				(wheel_mask & (1u << tree->wheel_index)) != 0) {
				world_fail("invalid vehicle wheel collision node");
			}
			wheel_mask |= 1u << tree->wheel_index;
		} else {
			world_fail("invalid vehicle collision node kind");
		}
		collision_material_count += tree->material_count;
	}
	if (wheel_mask != 0x0fu || open_children != 0)
		world_fail("incomplete vehicle collision tree");
	(void)graph_section(
		blob, size, header->collision_material_ids_offset,
		collision_material_count);

	const struct TmnfVehicleTuningDescriptor *tunings =
		(const struct TmnfVehicleTuningDescriptor *)graph_section(
			blob, size, header->tuning_descriptors_offset,
			header->tuning_count * sizeof(*tunings));
	for (uint32_t i = 0; i < header->tuning_count; ++i) {
		if (tunings[i].key != i || tunings[i].object_id == 0 ||
			tunings[i].raw_size != TUNING_SIZE) {
			world_fail("invalid vehicle tuning descriptor");
		}
		(void)graph_section(
			blob, size, tunings[i].raw_offset, TUNING_SIZE);
	}

	const struct TmnfVehicleCurveDescriptor *curves =
		(const struct TmnfVehicleCurveDescriptor *)graph_section(
			blob, size, header->curve_descriptors_offset,
			header->curve_count * sizeof(*curves));
	for (uint32_t i = 0; i < header->curve_count; ++i) {
		const uint8_t *owner;
		if (curves[i].owner_kind == VEHICLE_CURVE_OWNER_STRUCT) {
			if (curves[i].owner_key != 0 ||
				!contains_offset(
					PRIMARY_CURVE_OFFSETS,
					PRIMARY_CURVE_COUNT,
					curves[i].field_offset)) {
				world_fail("invalid vehicle-struct curve owner");
			}
			owner = vehicle_struct;
		} else if (
			curves[i].owner_kind == VEHICLE_CURVE_OWNER_TUNING) {
			if (curves[i].owner_key >= header->tuning_count ||
				!contains_offset(
					ACTIVE_CURVE_OFFSETS,
					ACTIVE_CURVE_COUNT,
					curves[i].field_offset)) {
				world_fail("invalid tuning curve owner");
			}
			owner = blob + tunings[curves[i].owner_key].raw_offset;
		} else {
			world_fail("invalid vehicle curve owner kind");
		}
		if (curves[i].object_id == 0 || curves[i].positions_id == 0 ||
			curves[i].values_id == 0 || curves[i].count == 0 ||
			curves[i].count > UINT32_MAX / 4 ||
			raw_u32(owner, curves[i].field_offset) == 0) {
			world_fail("invalid vehicle curve descriptor");
		}
		(void)graph_section(blob, size, curves[i].positions_offset,
			curves[i].count * 4);
		(void)graph_section(blob, size, curves[i].values_offset,
			curves[i].count * 4);
		for (uint32_t j = 0; j < i; ++j) {
			if (curves[j].owner_kind == curves[i].owner_kind &&
				curves[j].owner_key == curves[i].owner_key &&
				curves[j].field_offset == curves[i].field_offset) {
				world_fail("duplicate vehicle curve descriptor");
			}
		}
	}

	const struct TmnfVehicleGearboxDescriptor *gearboxes =
		(const struct TmnfVehicleGearboxDescriptor *)graph_section(
			blob, size, header->gearbox_descriptors_offset,
			header->gearbox_count * sizeof(*gearboxes));
	for (uint32_t i = 0; i < header->gearbox_count; ++i) {
		if (gearboxes[i].owner_key >= header->tuning_count ||
			!contains_offset(
				GEARBOX_OFFSETS, GEARBOX_COUNT,
				gearboxes[i].field_offset) ||
			gearboxes[i].buffer_id == 0 ||
			gearboxes[i].data_id == 0 ||
			gearboxes[i].count != GEARBOX_VALUE_COUNT) {
			world_fail("invalid vehicle gearbox descriptor");
		}
		const uint8_t *owner =
			blob + tunings[gearboxes[i].owner_key].raw_offset;
		if (raw_u32(owner, gearboxes[i].field_offset) !=
				GEARBOX_VALUE_COUNT ||
			raw_u32(owner, gearboxes[i].field_offset + 4) == 0) {
			world_fail("vehicle gearbox does not match raw tuning");
		}
		(void)graph_section(
			blob, size, gearboxes[i].data_offset,
			GEARBOX_VALUE_COUNT * sizeof(float));
		for (uint32_t j = 0; j < i; ++j) {
			if (gearboxes[j].owner_key ==
					gearboxes[i].owner_key &&
				gearboxes[j].field_offset ==
					gearboxes[i].field_offset) {
				world_fail("duplicate vehicle gearbox descriptor");
			}
		}
	}

	const struct TmnfVehicleMaterial *materials =
		(const struct TmnfVehicleMaterial *)(blob +
			header->ground_materials_offset);
	for (uint32_t i = 0; i < header->ground_material_count; ++i) {
		if (materials[i].object_id == 0)
			world_fail("vehicle ground material has null identity");
		if (materials[i].fake_contact_mask > 1)
			world_fail("vehicle ground material mask is not a flag");
	}
	return header;
}

static void decode_wheel(CSceneVehicleCarWheel *wheel, const uint8_t *raw)
{
	memset(wheel, 0, sizeof(*wheel));
	wheel->active = raw_i32(raw, 0x000);
	wheel->steerable = raw_i32(raw, 0x004);
	wheel->radius = raw_f32(raw, 0x008);
	memcpy(wheel->field70, raw + 0x070, sizeof(wheel->field70));
	wheel->fielda0 = raw_i32(raw, 0x0a0);
	wheel->fielda4 = raw_i32(raw, 0x0a4);
	memcpy(&wheel->offset_from_vehicle, raw + 0x0a8, sizeof(GmVec3));
	memcpy(&wheel->real_time, raw + 0x0b4, sizeof(wheel->real_time));
	wheel->field15c = raw_i32(raw, 0x15c);
	memcpy(&wheel->contact_relative_local_distance,
		raw + 0x160, sizeof(GmVec3));
}

static const struct TmnfVehicleCurveDescriptor *lookup_curve(
	const TmnfWorld *world, uint32_t owner_kind, uint32_t owner_key,
	uint32_t field_offset)
{
	const struct TmnfVehicleCurveDescriptor *result = NULL;
	for (uint32_t i = 0; i < world->cold->vehicle_header->curve_count; ++i) {
		const struct TmnfVehicleCurveDescriptor *curve =
			&world->cold->curve_descriptors[i];
		if (curve->owner_kind == owner_kind &&
			curve->owner_key == owner_key &&
			curve->field_offset == field_offset) {
			if (result != NULL)
				world_fail("ambiguous vehicle curve lookup");
			result = curve;
		}
	}
	return result;
}

static const struct TmnfVehicleCurveDescriptor *find_curve(
	const TmnfWorld *world, uint32_t owner_kind, uint32_t owner_key,
	uint32_t field_offset)
{
	const struct TmnfVehicleCurveDescriptor *result =
		lookup_curve(world, owner_kind, owner_key, field_offset);
	if (result == NULL)
		world_fail("vehicle curve lookup failed");
	return result;
}

/* Values only; the position/value arrays are bound by link_curves. */
static void decode_curve(
	TmnfWorld *world, CFuncKeysReal *curve, uint32_t owner_kind,
	uint32_t owner_key, uint32_t field_offset, uint32_t *index)
{
	const struct TmnfVehicleCurveDescriptor *raw =
		find_curve(world, owner_kind, owner_key, field_offset);
	curve->keys.count = raw->count;
	curve->interpolation = raw->interpolation;
	*index = (uint32_t)(raw - world->cold->curve_descriptors);
}

static void decode_curves(TmnfWorld *world)
{
	uint32_t key = world->cold->vehicle_header->active_tuning_key;
	for (uint32_t i = 0; i < ACTIVE_CURVE_COUNT; ++i) {
		/* Tuning +0x378 (steering angle from speed) is a nullable
		 * pointer in the game; the Desert tuning leaves it null and
		 * CSceneVehicleCar then uses a fixed 30 degrees. */
		if (i == CURVE_STEERING_ANGLE &&
			lookup_curve(world, VEHICLE_CURVE_OWNER_TUNING, key,
				ACTIVE_CURVE_OFFSETS[i]) == NULL) {
			world->cold->steering_angle_curve_present = 0;
			continue;
		}
		if (i == CURVE_STEERING_ANGLE)
			world->cold->steering_angle_curve_present = 1;
		decode_curve(
			world, &world->cold->active_curves[i],
			VEHICLE_CURVE_OWNER_TUNING, key,
			ACTIVE_CURVE_OFFSETS[i], &world->cold->active_curve_index[i]);
	}
	for (uint32_t i = 0; i < PRIMARY_CURVE_COUNT; ++i) {
		decode_curve(
			world, &world->cold->primary_curves[i],
			VEHICLE_CURVE_OWNER_STRUCT, 0,
			PRIMARY_CURVE_OFFSETS[i], &world->cold->primary_curve_index[i]);
	}
	for (uint32_t i = 0; i < 2; ++i) {
		const CFuncKeysReal *source =
			&world->cold->active_curves[CURVE_AIR_VERTICAL + i];
		world->cold->aux_curves[i].count = source->keys.count;
		world->cold->aux_curves[i].interpolation = source->interpolation;
	}
	world->cold->curve_set.m5_slipping_accel_scale =
		raw_f32(world->cold->raw_tuning, 0x1e4);
	world->cold->curve_set.damper_max = raw_f32(world->cold->raw_tuning, 0x11c);
	world->cold->curve_set.damper_min = raw_f32(world->cold->raw_tuning, 0x120);
	for (uint32_t i = 0; i < 2; ++i) {
		world->cold->water_impulse_curves[i].keys.count = 4;
		world->cold->water_impulse_curves[i].interpolation = 0;
	}
}

static const struct TmnfVehicleGearboxDescriptor *find_gearbox(
	const TmnfWorld *world, uint32_t owner_key, uint32_t field_offset)
{
	const struct TmnfVehicleGearboxDescriptor *result = NULL;
	for (uint32_t i = 0; i < world->cold->vehicle_header->gearbox_count; ++i) {
		const struct TmnfVehicleGearboxDescriptor *gearbox =
			&world->cold->gearbox_descriptors[i];
		if (gearbox->owner_key == owner_key &&
			gearbox->field_offset == field_offset) {
			if (result != NULL)
				world_fail("ambiguous vehicle gearbox lookup");
			result = gearbox;
		}
	}
	if (result == NULL)
		world_fail("vehicle gearbox lookup failed");
	return result;
}

static void decode_gearboxes(TmnfWorld *world)
{
	float *destinations[GEARBOX_COUNT] = {
		world->cold->gear_ratios,
		world->cold->gear_upshift,
		world->cold->gear_downshift,
		world->cold->gear_aux,
	};
	uint32_t key = world->cold->vehicle_header->active_tuning_key;
	for (uint32_t i = 0; i < GEARBOX_COUNT; ++i) {
		const struct TmnfVehicleGearboxDescriptor *raw =
			find_gearbox(world, key, GEARBOX_OFFSETS[i]);
		memcpy(
			destinations[i],
			world->cold->vehicle_blob + raw->data_offset,
			GEARBOX_VALUE_COUNT * sizeof(float));
	}
}

static void decode_vehicle_tuning(TmnfWorld *world)
{
	const uint8_t *raw = world->cold->raw_tuning;
	CSceneVehicleCarTuning *tuning = &world->cold->vehicle_tuning;
	memset(tuning, 0, sizeof(*tuning));
	tuning->damper_max = raw_f32(raw, 0x11c);
	tuning->damper_min = raw_f32(raw, 0x120);
	tuning->gear_ratio_count = 6;
	tuning->engine_model = raw_i32(raw, 0x354);
	tuning->forward_speed_limit_scale = raw_f32(raw, 0x02c);
	tuning->engine_rpm_reverse_accel = raw_f32(raw, 0x2ec);
	tuning->engine_rpm_accel = raw_f32(raw, 0x2f0);
	tuning->engine_rpm_decel = raw_f32(raw, 0x2f4);
	tuning->engine_rpm_turbo_decel = raw_f32(raw, 0x31c);
	tuning->engine_rpm_follow_accel = raw_f32(raw, 0x320);
	tuning->engine_rpm_low_accel = raw_f32(raw, 0x324);
	tuning->engine_rpm_high_decel = raw_f32(raw, 0x328);
	tuning->speed_32c = raw_f32(raw, 0x32c);
	tuning->speed_330 = raw_f32(raw, 0x330);
	tuning->speed_334 = raw_f32(raw, 0x334);
	tuning->speed_338 = raw_f32(raw, 0x338);
	tuning->suspension_model = raw_i32(raw, 0x350);
	tuning->suspension_stiffness = raw_f32(raw, 0x114);
	tuning->suspension_damping = raw_f32(raw, 0x118);
	tuning->suspension_rest_length = raw_f32(raw, 0x124);
	tuning->suspension_scale = raw_f32(raw, 0x128);
}

static void decode_aux(TmnfWorld *world)
{
	const uint8_t *raw = world->cold->raw_tuning;
	const uint8_t *car = world->cold->raw_car;
	CSceneVehicleCarTuningAux *tuning = &world->cold->aux_tuning;
	tuning->steering_speed_base = raw_f32(raw, 0x06c);
	tuning->steering_speed_scale = raw_f32(raw, 0x070);
	tuning->steering_slew_rate = raw_f32(raw, 0x094);
	tuning->air_torque_linear = raw_f32(raw, 0x158);
	tuning->air_torque_quadratic = raw_f32(raw, 0x15c);
	tuning->suspension_follow_rate = raw_f32(raw, 0x194);
	tuning->air_control_window_ticks = raw_u32(raw, 0x364);
	tuning->air_reversal_threshold = raw_f32(raw, 0x368);
	tuning->water_buoyancy = raw_f32(raw, 0x204);
	tuning->water_entry_speed_threshold = raw_f32(raw, 0x208);
	tuning->water_entry_speed_minimum = raw_f32(raw, 0x20c);
	tuning->water_angular_drag_linear = raw_f32(raw, 0x21c);
	tuning->water_angular_drag_quadratic = raw_f32(raw, 0x220);
	/* Tuning +0x210/+0x214 are absent from the snapshot; see
	 * vehicle_water_tuning.h. */

	for (uint32_t i = 0; i < 4; ++i) {
		const uint8_t *wheel = world->cold->raw_wheels + i * WHEEL_SIZE;
		memcpy(&world->aux_wheels[i].surface_source,
			wheel + 0x010, sizeof(GmIso4));
		memcpy(&world->aux_wheels[i].surface_location,
			wheel + 0x040, sizeof(GmIso4));
	}

	world->aux.wheel_count = 4;
	world->aux.integration_flags = raw_u32(car, 0x2f4);
	world->aux.turbo_epoch_tick = raw_u32(car, 0x5d0);
	world->aux.air_control_immediate = raw_i32(car, 0x5d4);
	world->aux.air_control_locked = raw_i32(car, 0x5e4);
	world->aux.steering_value = raw_f32(car, 0x5e8);
	world->aux.turbo_progress = raw_f32(car, 0x5f0);
	world->aux.turbo_factor = raw_f32(car, 0x5f4);
	world->aux.turbo_start_tick = raw_u32(car, 0x5f8);
	world->aux.turbo_end_tick = raw_u32(car, 0x5fc);
	world->aux.turbo_type =
		(TMNFVehicleTurboType)raw_i32(car, 0x600);
	world->aux.roulette_token = raw_u32(car, 0x604);
	world->aux.roulette_value = raw_f32(car, 0x608);
	world->aux.air_control_tick = raw_u32(car, 0x614);
	memcpy(&world->aux.air_control_speed, car + 0x618, sizeof(GmVec3));
	world->aux.roulette_modulus = UINT32_MAX;
	memcpy(&world->aux.body_box, car + 0x1dc, sizeof(GmBoxAligned));
	/* car+0x26c is a process-local audio object, not vehicle state. The
	 * headless native world has no audio source attached. */
	world->aux.turbo_sound_attached = 0;
}

static void decode_contact_tuning(TmnfWorld *world)
{
	const uint8_t *raw = world->cold->raw_tuning;
	TMNFVehicleContactTuning *tuning = &world->cold->contact_tuning;
	tuning->friction_force = raw_f32(raw, 0x058);
	tuning->extra_friction_force = raw_f32(raw, 0x05c);
	tuning->slope_adherence_min = raw_f32(raw, 0x0d4);
	tuning->slope_adherence_max = raw_f32(raw, 0x0d8);
	tuning->slope_secondary_min = raw_f32(raw, 0x0dc);
	tuning->slope_secondary_max = raw_f32(raw, 0x0e0);
	tuning->angular_y_scale = raw_f32(raw, 0x0e8);
	tuning->angular_xz_scale = raw_f32(raw, 0x0ec);
	tuning->damper_max = raw_f32(raw, 0x11c);
	tuning->max_angular_speed = raw_f32(raw, 0x14c);
	tuning->max_linear_speed_delta = raw_f32(raw, 0x150);
	tuning->body_tangent_ratio = raw_f32(raw, 0x170);
	tuning->body_tangent_ratio_material4 = raw_f32(raw, 0x174);
	tuning->restitution_air_material4 = raw_f32(raw, 0x178);
	tuning->restitution_air = raw_f32(raw, 0x17c);
	tuning->restitution_ground = raw_f32(raw, 0x184);
	tuning->restitution_ground_material4 = raw_f32(raw, 0x18c);
	tuning->lateral_linear = raw_f32(raw, 0x1a8);
	tuning->lateral_quadratic = raw_f32(raw, 0x1ac);
	tuning->lateral_ground_scale = raw_f32(raw, 0x1c0);
	tuning->lateral_contact_duration_ticks = raw_u32(raw, 0x1e8);
	tuning->wheel_contact_model = raw_i32(raw, 0x350);
	tuning->friction_model = raw_i32(raw, 0x354);
}

#define MODEL6_TUNE(field, offset) \
	world->cold->model6_tuning.field = raw_f32(world->cold->raw_tuning, offset)

static void decode_model6(TmnfWorld *world)
{
	const uint8_t *car = world->cold->raw_car;
	MODEL6_TUNE(forward_speed_limit_scale, 0x02c);
	MODEL6_TUNE(reverse_speed_limit_scale, 0x030);
	MODEL6_TUNE(brake_base, 0x040);
	MODEL6_TUNE(brake_speed_scale, 0x044);
	MODEL6_TUNE(forward_brake_limit_sliding, 0x048);
	MODEL6_TUNE(forward_brake_limit, 0x04c);
	MODEL6_TUNE(speed_limit_force, 0x060);
	MODEL6_TUNE(vertical_force_scale, 0x064);
	MODEL6_TUNE(wheel_steer_sine_limit, 0x074);
	MODEL6_TUNE(steer_slowdown_scale, 0x07c);
	MODEL6_TUNE(wheel_torque_scale, 0x098);
	MODEL6_TUNE(sliding_steer_torque_scale, 0x09c);
	MODEL6_TUNE(lateral_force_scale, 0x0a4);
	MODEL6_TUNE(sliding_lateral_limit_scale, 0x0b0);
	MODEL6_TUNE(lateral_overflow_blend, 0x0b4);
	MODEL6_TUNE(wheel_overflow_blend, 0x0e4);
	MODEL6_TUNE(vertical_force_divisor, 0x160);
	MODEL6_TUNE(traction_loss_scale, 0x200);
	MODEL6_TUNE(burnout_trigger_scale, 0x228);
	MODEL6_TUNE(burnout_trigger_limit, 0x22c);
	MODEL6_TUNE(rollover_axis_min_length, 0x234);
	MODEL6_TUNE(rollover_torque_x_scale, 0x238);
	MODEL6_TUNE(rollover_torque_z_scale, 0x23c);
	MODEL6_TUNE(sliding_brake_scale, 0x240);
	MODEL6_TUNE(braking_lateral_limit_scale, 0x244);
	MODEL6_TUNE(reverse_brake_limit_sliding, 0x248);
	MODEL6_TUNE(reverse_brake_limit, 0x24c);
	MODEL6_TUNE(burnout_speed_max, 0x254);
	MODEL6_TUNE(burnout_speed_min, 0x258);
	MODEL6_TUNE(donut_lateral_force_scale, 0x264);
	MODEL6_TUNE(donut_yaw_angle_scale, 0x268);
	MODEL6_TUNE(donut_steer_linear, 0x26c);
	MODEL6_TUNE(donut_steer_quadratic, 0x270);
	MODEL6_TUNE(donut_countersteer_scale, 0x274);
	MODEL6_TUNE(donut_radius_exponent, 0x278);
	MODEL6_TUNE(donut_radial_speed_exponent, 0x27c);
	MODEL6_TUNE(donut_radius_min, 0x280);
	MODEL6_TUNE(donut_lateral_speed_limit, 0x284);
	MODEL6_TUNE(donut_normal_angle_limit, 0x28c);
	MODEL6_TUNE(donut_angle_positive_limit, 0x290);
	MODEL6_TUNE(donut_angle_negative_limit, 0x294);
	world->cold->model6_tuning.burnout_enter_ticks =
		raw_u32(world->cold->raw_tuning, 0x298);
	MODEL6_TUNE(burnout_enter_accel_scale, 0x29c);
	MODEL6_TUNE(burnout_enter_lateral_scale, 0x2a0);
	world->cold->model6_tuning.burnout_exit_ticks =
		raw_u32(world->cold->raw_tuning, 0x2a8);
	MODEL6_TUNE(burnout_exit_accel_scale, 0x2ac);
	MODEL6_TUNE(burnout_exit_extra_accel, 0x2b8);
	MODEL6_TUNE(material6_longitudinal_scale, 0x33c);
	MODEL6_TUNE(material6_gas_denominator, 0x340);
	MODEL6_TUNE(material6_vertical_shape, 0x344);
	MODEL6_TUNE(material6_vertical_scale, 0x348);

	memcpy(&world->model6_state.pivot_position, car + 0x1dc, 12);
	memcpy(&world->model6_state.pivot_axis, car + 0x1e8, 12);
	world->model6_state.reverse_mode = raw_i32(car, 0x5c4);
	world->model6_state.reverse_speed_threshold = raw_f32(car, 0x5cc);
	world->model6_state.contact_block_count = raw_i32(car, 0x5d8);
	world->model6_state.side_contact = raw_i32(car, 0x5dc);
	world->model6_state.last_sliding_tick = raw_u32(car, 0x62c);
	world->model6_state.sliding_start_tick = raw_u32(car, 0x630);
	world->model6_state.sliding_elapsed_ticks = raw_u32(car, 0x634);
	memcpy(&world->model6_state.model_iso, car + 0x6a4, 0x30);
	memcpy(&world->model6_state.rollover_axis, car + 0x6d4, 12);
	memcpy(&world->model6_state.orbit_center, car + 0x6e0, 12);
	world->model6_state.orbit_initial_radius = raw_f32(car, 0x6ec);
	world->model6_state.orbit_radius = raw_f32(car, 0x6f0);
	world->model6_state.burnout_start_tick = raw_u32(car, 0x6f4);
	world->model6_state.burnout_transition_tick = raw_u32(car, 0x6f8);
	memcpy(&world->model6_state.orbit_axis, car + 0x6fc, 12);
	world->model6_state.orbit_sign = raw_f32(car, 0x708);
	world->model6_state.axle_width = raw_f32(car, 0x840);
}

#undef MODEL6_TUNE

/* Models 3, 4 and 5 share the tuning block and the car fields Model 6 uses
 * (world->model6_state); only their private fields are decoded here. Offsets
 * are the ones listed in vehicle_oldmodels.h. */
#define OLDMODELS_TUNE(field, offset) \
	world->cold->oldmodels_tuning.field = raw_f32(world->cold->raw_tuning, offset)

static void decode_oldmodels(TmnfWorld *world)
{
	const uint8_t *car = world->cold->raw_car;

	OLDMODELS_TUNE(forward_speed_limit_scale, 0x02c);
	OLDMODELS_TUNE(reverse_speed_limit_scale, 0x030);
	OLDMODELS_TUNE(brake_base, 0x040);
	OLDMODELS_TUNE(brake_speed_scale, 0x044);
	OLDMODELS_TUNE(brake_limit_sliding, 0x048);
	OLDMODELS_TUNE(brake_limit, 0x04c);
	OLDMODELS_TUNE(speed_limit_force, 0x060);
	OLDMODELS_TUNE(vertical_force_scale, 0x064);
	OLDMODELS_TUNE(wheel_steer_sine_limit, 0x074);
	OLDMODELS_TUNE(steer_slowdown_scale, 0x07c);
	world->cold->oldmodels_tuning.m5_steer_gate_needs_sliding =
		raw_i32(world->cold->raw_tuning, 0x080);
	world->cold->oldmodels_tuning.m5_steer_gate_sliding_ticks =
		raw_u32(world->cold->raw_tuning, 0x084);
	OLDMODELS_TUNE(wheel_torque_scale, 0x098);
	OLDMODELS_TUNE(sliding_steer_torque_scale, 0x09c);
	OLDMODELS_TUNE(lateral_force_scale, 0x0a4);
	OLDMODELS_TUNE(sliding_lateral_limit_scale, 0x0b0);
	OLDMODELS_TUNE(lateral_overflow_blend, 0x0b4);
	OLDMODELS_TUNE(longitudinal_torque_scale, 0x0c0);
	OLDMODELS_TUNE(wheel_overflow_blend, 0x0e4);
	OLDMODELS_TUNE(vertical_force_divisor, 0x160);
	OLDMODELS_TUNE(m4_steer_torque_speed_scale, 0x19c);
	OLDMODELS_TUNE(m4_yaw_damping_linear, 0x1a0);
	OLDMODELS_TUNE(m4_yaw_damping_quadratic, 0x1a4);
	OLDMODELS_TUNE(m4_drift_exit_lateral_speed, 0x1b0);
	OLDMODELS_TUNE(m4_drift_steer_radius_scale, 0x1b8);
	OLDMODELS_TUNE(m4_drift_angle_rate, 0x1cc);
	OLDMODELS_TUNE(m4_drift_angle_radius_scale, 0x1d0);
	OLDMODELS_TUNE(m4_drift_angle_limit, 0x1d8);
	OLDMODELS_TUNE(m4_drift_entry_angle_scale, 0x1dc);
	OLDMODELS_TUNE(m5_longitudinal_torque_limit, 0x1f4);
	world->cold->oldmodels_tuning.m5_steer_gate_ticks =
		raw_u32(world->cold->raw_tuning, 0x1fc);
	OLDMODELS_TUNE(traction_loss_scale, 0x200);

	world->oldmodels_state.m4_drift_angle = raw_f32(car, 0x638);
	world->oldmodels_state.m4_drift_steer_sign = raw_f32(car, 0x63c);
	world->oldmodels_state.m4_drift_state = raw_i32(car, 0x640);
	world->oldmodels_state.m5_last_steer_tick = raw_u32(car, 0x648);
	world->oldmodels_state.m5_last_steer_sliding = raw_i32(car, 0x64c);
}

#undef OLDMODELS_TUNE

static void decode_compute(TmnfWorld *world)
{
	const uint8_t *car = world->cold->raw_car;
	const uint8_t *raw = world->cold->raw_tuning;
	TMNFVehicleComputeTuning *tuning = &world->cold->compute_tuning;
	tuning->event_c_level1_max = raw_f32(raw, 0x028);
	tuning->grounded_drag_term = raw_f32(raw, 0x058);
	tuning->active_contact_stop_threshold = raw_f32(raw, 0x0a4);
	tuning->normal_turbo_factor = raw_f32(raw, 0x0f0);
	tuning->roulette_turbo_factor = raw_f32(raw, 0x0f4);
	tuning->normal_turbo_duration = raw_u32(raw, 0x0f8);
	tuning->roulette_turbo_duration = raw_u32(raw, 0x0fc);
	tuning->air_impulse_scale = raw_f32(raw, 0x104);
	tuning->special_force_field_scale = raw_f32(raw, 0x108);
	tuning->airborne_linear_drag = raw_f32(raw, 0x154);
	tuning->grounded_force_field_scale = raw_f32(raw, 0x160);
	tuning->airborne_force_field_scale = raw_f32(raw, 0x164);
	tuning->normalized_force_divisor = raw_f32(raw, 0x228);
	tuning->effect_curve_bias = raw_f32(raw, 0x384);
	tuning->event_ab_level2_min = raw_f32(raw, 0x398);
	tuning->event_ab_trigger = raw_f32(raw, 0x39c);
	tuning->event_c_trigger = raw_f32(raw, 0x3a0);

	TMNFVehicleComputeState *state = &world->compute_state;
	state->simulation_gate = raw_f32(car, 0x1e8);
	state->event_level_c = raw_u32(car, 0x1fc);
	state->event_source_c = car[0x200];
	state->event_source_ab = car[0x201];
	memcpy(&state->spring_c, car + 0x214, sizeof(GmSpringFloat));
	memcpy(&state->spring_a, car + 0x228, sizeof(GmSpringFloat));
	state->contact_rise = raw_f32(car, 0x23c);
	state->contact_decay = raw_f32(car, 0x240);
	state->history_force_limit = raw_f32(car, 0x244);
	state->history_force_scale = raw_f32(car, 0x24c);
	state->spring_value_limit = raw_f32(car, 0x250);
	state->local_speed_limit = raw_f32(car, 0x2e0);
	state->air_effect_threshold = raw_f32(car, 0x05c);
	state->brake_input_scale = raw_f32(car, 0x5a8);
	state->grounded_drag_scale = raw_f32(car, 0x5ac);
	state->computed_brake_force = raw_f32(car, 0x5b0);
	state->state_5d8 = raw_i32(car, 0x5d8);
	state->air_impulse_cooldown_tick = raw_u32(car, 0x610);
	state->effect_accumulator = raw_f32(car, 0x624);
	state->last_force_tick = raw_u32(car, 0x650);
	state->event_level_a = raw_u32(car, 0x654);
	state->event_level_b = raw_u32(car, 0x658);
	state->peak_event_level_b = raw_u32(car, 0x660);
	state->peak_event_level_a = raw_u32(car, 0x664);
	state->peak_event_level_c = raw_u32(car, 0x668);
	state->peak_event_source_ab = car[0x66c];
	state->peak_event_source_c = car[0x66d];
	state->event_metric_a = raw_f32(car, 0x670);
	state->event_metric_b = raw_f32(car, 0x674);
	state->event_metric_c = raw_f32(car, 0x678);
	memcpy(&state->normalized_force, car + 0x6d4, sizeof(GmVec3));
	state->air_effect_mode = raw_i32(car, 0x74c);
}

TMNF_HD static void collision_get_linear_speed(
	const CHmsCorpus *corpus, GmVec3 *speed)
{
	CHmsDyna_GetLinearSpeed((CHmsDyna *)corpus->dyna, speed);
}

TMNF_HD static CHmsResponseBody *resolve_response_body(
	void *user, uint32_t corpus_ref)
{
	TmnfWorld *world = (TmnfWorld *)user;
	if (corpus_ref == world->cold->collision_corpus.object_ref)
		return &world->cold->player_response_body;
	const TmnfTrack *track = world->cold->track;
	if (corpus_ref < track->static_response_count &&
		track->static_response_present[corpus_ref] != 0) {
		return &track->static_response_bodies[corpus_ref];
	}
	return NULL;
}

TMNF_HD static const CHmsResponseMaterial *resolve_response_material(
	void *user, uint32_t material_ref)
{
	TmnfWorld *world = (TmnfWorld *)user;
	return material_ref == 1 ? &world->cold->response_material : NULL;
}

TMNF_HD static const GmIso4 *resolve_body_iso(
	void *user, const CHmsResponseBody *body)
{
	(void)user;
	if (body->dyna != NULL)
		return (const GmIso4 *)&body->dyna->liveState->rot;
	return &body->iso;
}

TMNF_HD static uint32_t response_body_token(
	void *user, const CHmsResponseBody *body)
{
	(void)user;
	return body->corpus_ref;
}

TMNF_HD static void absorb_player_contact(
	void *user, CHmsResponseBody *body, CHmsPhysicalContact *contact)
{
	TmnfWorld *world = (TmnfWorld *)user;
	if (body != &world->cold->player_response_body)
		world_fail("contact delivered to the wrong response body");
	CSceneVehicleCar_AbsorbContact(&world->contact, contact);
	world->model6_state.side_contact = world->contact.side_contact;
}

TMNF_HD static void set_surface_location(
	void *user, void *surface_tree, const GmIso4 *location)
{
	TmnfWorld *world = (TmnfWorld *)user;
	for (uint32_t i = 0; i < 4; ++i) {
		if (surface_tree == &world->wheel_trees[i]) {
			/* location->t was just written by scalar stores. */
			GmIso4 *local = &world->wheel_trees[i].local_iso;
#if TMNF_SSE
			_mm_storeu_ps(local->m, _mm_loadu_ps(location->m));
			_mm_storeu_ps(local->m + 4, _mm_loadu_ps(location->m + 4));
			_mm_storeu_ps(local->m + 8, gm_gather4(location->m + 8));
#else
			*local = *location;
#endif
			world->wheel_trees[i].box.center.x = location->t[0];
			world->wheel_trees[i].box.center.y = location->t[1];
			world->wheel_trees[i].box.center.z = location->t[2];
			world->contact_wheels[i].impulse_point.x = location->t[0];
			world->contact_wheels[i].impulse_point.y = location->t[1];
			world->contact_wheels[i].impulse_point.z = location->t[2];
			return;
		}
	}
	world_fail("surface update references an unknown wheel tree");
}

TMNF_HD static void finish_vehicle_integration(
	void *user, CSceneVehicleCar *vehicle)
{
	TmnfWorld *world = (TmnfWorld *)user;
	if (vehicle != &world->vehicle)
		world_fail("integration callback references the wrong vehicle");
	world->model6_state.contact_block_count =
		world->compute_state.state_5d8;
	world->model6_state.side_contact = world->contact.side_contact;
	world->model6_state.rollover_axis =
		world->compute_state.normalized_force;
}

TMNF_HD static void post_vehicle_force(void *user, CSceneVehicleCar *vehicle)
{
	TmnfWorld *world = (TmnfWorld *)user;
	if (vehicle != &world->vehicle)
		world_fail("force callback references the wrong vehicle");
	world->vehicle.engine.braking_factor =
		world->compute_state.computed_brake_force;
}

/* Values of the collision graph; the pointers are bound by link_collision. */
static void decode_collision_world(TmnfWorld *world)
{
	uint32_t node_count = world->cold->vehicle_header->collision_child_count;
	for (uint32_t i = 0; i < node_count; ++i) {
		const struct TmnfVehicleCollisionTree *raw =
			&world->cold->collision_children[i];
		uint32_t kind = raw->kind & VEHICLE_COLLISION_KIND_MASK;
		GmSurfEllipsoid *shape;
		CPlugSurface *surface;
		CPlugTree *tree;
		if (kind == VEHICLE_COLLISION_TREE_WHEEL) {
			uint32_t wheel_index = raw->wheel_index;
			shape = &world->cold->wheel_ellipsoids[wheel_index];
			surface = &world->cold->wheel_surfaces[wheel_index];
			tree = &world->wheel_trees[wheel_index];
			world->cold->wheel_tree_refs[wheel_index] = raw->object_id;
			world->contact_wheels[wheel_index].impulse_point =
				raw->box.center;
		} else {
			shape = &world->cold->body_ellipsoids[i];
			surface = &world->cold->body_surfaces[i];
			tree = &world->cold->body_trees[i];
			world->cold->body_tree_refs[world->cold->body_tree_count++] =
				raw->object_id;
		}
		if (raw->surface_id != 0) {
			shape->base.material_index =
				raw->geometry_material_index;
			shape->base.type = raw->geometry_type;
			shape->base.reserved = raw->geometry_reserved;
			memcpy(&shape->radii, raw->shape, sizeof(shape->radii));
			surface->material_count = raw->material_count;
		}
		tree->object_ref = raw->object_id;
		tree->flags = raw->flags;
		tree->box = raw->box;
		tree->local_iso = raw->local_iso;
	}
	world->cold->root_tree.object_ref = world->cold->collision_root->object_id;
	world->cold->root_tree.flags = world->cold->collision_root->flags;
	world->cold->root_tree.box = world->cold->collision_root->box;
	world->cold->root_tree.local_iso = world->cold->collision_root->local_iso;

	world->cold->collision_corpus.flags = 1u << 13;
	world->cold->collision_corpus.group_index = 0;

	CHmsCollisionManager_SGroup *dynamic =
		&world->collision_zone.groups[0];
	dynamic->corpus_count = 1;
	dynamic->device_mat_count = 1;
	dynamic->priority = 0;
	world->cold->static_device.material_ref = 1;
	world->cold->static_device.perform.rows = 1;
	world->cold->static_device.perform.columns = 0;

	world->cold->response_material.category = 3;
	world->cold->response_material.response_mode = 1;
	world->cold->response_material.side_enabled[0] = 1;
	world->cold->response_material.side_enabled[1] = 1;

	world->cold->player_response_body.corpus_ref =
		world->cold->collision_corpus.object_ref;
	world->cold->player_response_body.classification_flags = 0x19847004u;
	world->cold->player_response_body.response_flags = 0xff618001u;
	world->cold->player_response_body.response_weight = 1.0f;
	world->cold->player_response_body.has_contact_sink = 1;
	for (uint32_t i = 0; i < VEHICLE_COLLISION_MAX_NODES; ++i) {
		world->contact_buffers[i].active = 0;
		world->contact_buffers[i].base.collisions.count = 0;
	}
}

static void decode_vehicle(TmnfWorld *world)
{
	memcpy(&world->live_state,
		world->cold->vehicle_blob + world->cold->vehicle_header->state_offset,
		sizeof(world->live_state));
	memcpy(&world->dyna_params,
		world->cold->vehicle_blob + world->cold->vehicle_header->params_offset,
		sizeof(world->dyna_params));
	memcpy(&world->dyna.tempState,
		world->cold->raw_dyna + 0x274, sizeof(world->dyna.tempState));
	world->committed_state = world->dyna.tempState;
	world->dyna.clampAngular = raw_i32(world->cold->raw_dyna, 0x0c0);
	world->dyna.maxAngularSpeed = raw_f32(world->cold->raw_dyna, 0x0c4);
	world->dyna.dirtyFlag = raw_i32(world->cold->raw_dyna, 0x33c);
	world->dyna.mode = raw_i32(world->cold->raw_dyna, 0x340);

	decode_curves(world);
	decode_gearboxes(world);
	decode_vehicle_tuning(world);
	for (uint32_t i = 0; i < 4; ++i)
		decode_wheel(
			&world->wheels[i], world->cold->raw_wheels + i * WHEEL_SIZE);

	const uint8_t *car = world->cold->raw_car;
	world->vehicle.input_gas = raw_f32(car, 0x050);
	world->vehicle.input_brake = raw_f32(car, 0x054);
	world->vehicle.input_steer = raw_f32(car, 0x058);
	world->vehicle.wheel_count = 4;
	memcpy(&world->vehicle.engine, car + 0x59c, sizeof(world->vehicle.engine));
	memcpy(&world->vehicle.current_local_speed, car + 0x70c, 12);
	memcpy(&world->vehicle.total_force_added, car + 0x818, 12);
	memcpy(&world->vehicle.total_impulse_added, car + 0x824, 12);
	world->vehicle.engine_mode = raw_i32(car, 0x2e4);
	world->vehicle.turbo_active = raw_i32(car, 0x628);
	world->vehicle.drive_mode = raw_i32(car, 0x69c);
	world->vehicle.engine_limit_flag = raw_i32(car, 0x744);
	world->vehicle.gear_downshift_flag = raw_i32(car, 0x748);
	world->vehicle.force_wheel_speed = raw_i32(car, 0x6a0);
	world->vehicle.flag_60c = raw_i32(car, 0x60c);
	world->vehicle.block_wheel_speed = raw_i32(car, 0x73c);
	world->vehicle.forced_wheel_speed = raw_f32(world->cold->raw_tuning, 0x2bc);

	decode_aux(world);
	decode_contact_tuning(world);

	const struct TmnfVehicleSnapshotHeader *header =
		world->cold->vehicle_header;
	memcpy(world->cold->ground_material_ids,
		world->cold->vehicle_blob + header->ground_ids_offset,
		sizeof(world->cold->ground_material_ids));
	for (uint32_t i = 0; i < TMNF_SURFACE_MATERIAL_COUNT; ++i) {
		if (world->cold->ground_material_ids[i] >= header->ground_material_count)
			world_fail("ground-id table maps a physical material outside the material manager");
	}
	const struct TmnfVehicleMaterial *raw_materials =
		(const struct TmnfVehicleMaterial *)(world->cold->vehicle_blob +
			header->ground_materials_offset);
	for (uint32_t i = 0; i < header->ground_material_count; ++i) {
		TMNFVehicleGroundMaterial *material = &world->cold->ground_materials[i];
		const struct TmnfVehicleMaterial *raw = &raw_materials[i];
		memset(material, 0, sizeof(*material));
		memcpy(material->values, raw->values, sizeof(raw->values));
		/* The bump mask is one shared 128x128 image in every collection's
		 * material manager (src/vehicle_fake_contact_mask.h); the record
		 * says whether this material references it. The pointer itself is
		 * bound by link_vehicle; here it only records that there is one. */
		material->fake_contact_mask =
			raw->fake_contact_mask != 0 ? TMNF_FAKE_CONTACT_MASK : NULL;
		material->fake_contact_period_x = raw->fake_contact_period_x;
		material->fake_contact_period_z = raw->fake_contact_period_z;
		material->fake_contact_impulse_scale =
			raw->fake_contact_impulse_scale;
		material->fake_contact_impulse_limit =
			raw->fake_contact_impulse_limit;
	}
	world->timer.tick_time = header->tick;
	world->contact.wheel_count = 4;
	world->contact.ground_material_index_count = TMNF_SURFACE_MATERIAL_COUNT;
	world->contact.ground_material_count = header->ground_material_count;
	world->contact.wheel_contact_absorb_count =
		raw_u32(car, 0x67c);
	world->contact.body_contact_count = raw_u32(car, 0x680);
	memcpy(&world->contact.body_contact_position_sum,
		car + 0x684, sizeof(GmVec3));
	memcpy(&world->contact.body_contact_normal_sum,
		car + 0x690, sizeof(GmVec3));
	world->contact.friction_input_selector = raw_i32(car, 0x5c4);
	world->contact.side_contact = raw_i32(car, 0x5dc);
	world->contact.last_side_contact_tick = raw_u32(car, 0x5e0);
	world->contact.airborne_friction_gate = raw_i32(car, 0x5e4);

	decode_model6(world);
	decode_oldmodels(world);
	decode_compute(world);
	world->cold->item.corpus_count = 1;
}

/* ---- pointer topology ---------------------------------------------------
 *
 * Every pointer in the world is written here and nowhere else, as a function
 * of the world's own address and the link sources. The host runs it once at
 * World_Create; a device world is a byte copy of a host world relinked in
 * place with device sources. Data written by the decode functions is never
 * touched, so the same function serves both address spaces.
 */

TMNF_HD static void link_blob(TmnfWorld *world, const uint8_t *blob)
{
	const struct TmnfVehicleSnapshotHeader *header =
		(const struct TmnfVehicleSnapshotHeader *)blob;
	world->cold->vehicle_blob = (uint8_t *)blob;
	world->cold->vehicle_header = header;
	world->cold->tuning_descriptors =
		(const struct TmnfVehicleTuningDescriptor *)(
			blob + header->tuning_descriptors_offset);
	world->cold->curve_descriptors =
		(const struct TmnfVehicleCurveDescriptor *)(
			blob + header->curve_descriptors_offset);
	world->cold->gearbox_descriptors =
		(const struct TmnfVehicleGearboxDescriptor *)(
			blob + header->gearbox_descriptors_offset);
	world->cold->collision_root =
		(const struct TmnfVehicleCollisionTree *)(
			blob + header->collision_root_offset);
	world->cold->collision_children =
		(const struct TmnfVehicleCollisionTree *)(
			blob + header->collision_children_offset);
	world->cold->collision_material_ids =
		blob + header->collision_material_ids_offset;
	world->cold->raw_car = blob + header->car_offset;
	world->cold->raw_vehicle_struct = blob + header->vehicle_struct_offset;
	world->cold->raw_tuning = blob +
		world->cold->tuning_descriptors[header->active_tuning_key].raw_offset;
	world->cold->raw_wheels = blob + header->wheels_offset;
	world->cold->raw_dyna = blob + header->dyna_offset;
}

/* The key bounds of every curve live in one table (sources->curve_bounds),
 * lower then upper per curve, in the order link_curves visits them. */
TMNF_HD static void link_key_bounds(
	CFuncKeys *keys, const float *bounds, uint32_t *cursor)
{
	keys->lower_bounds = bounds + *cursor;
	keys->upper_bounds = bounds + *cursor + keys->count;
	*cursor += 2 * keys->count;
}

TMNF_HD static void link_curve(
	TmnfWorld *world, CFuncKeysReal *curve, uint32_t descriptor_index,
	const float *bounds, uint32_t *cursor)
{
	const struct TmnfVehicleCurveDescriptor *raw =
		&world->cold->curve_descriptors[descriptor_index];
	curve->keys.positions =
		(const float *)(world->cold->vehicle_blob + raw->positions_offset);
	curve->values =
		(const float *)(world->cold->vehicle_blob + raw->values_offset);
	link_key_bounds(&curve->keys, bounds, cursor);
}

TMNF_HD static void link_curves(
	TmnfWorld *world, const TmnfWorldLinkSources *sources)
{
	uint32_t cursor = 0;
	for (uint32_t i = 0; i < ACTIVE_CURVE_COUNT; ++i) {
		link_curve(
			world, &world->cold->active_curves[i],
			world->cold->active_curve_index[i], sources->curve_bounds, &cursor);
	}
	for (uint32_t i = 0; i < PRIMARY_CURVE_COUNT; ++i) {
		link_curve(
			world, &world->cold->primary_curves[i],
			world->cold->primary_curve_index[i], sources->curve_bounds, &cursor);
	}
	for (uint32_t i = 0; i < 2; ++i) {
		const CFuncKeysReal *source =
			&world->cold->active_curves[CURVE_AIR_VERTICAL + i];
		world->cold->aux_curves[i].positions = source->keys.positions;
		world->cold->aux_curves[i].lower_bounds = source->keys.lower_bounds;
		world->cold->aux_curves[i].upper_bounds = source->keys.upper_bounds;
		world->cold->aux_curves[i].values = source->values;
		world->cold->water_impulse_curves[i].keys.positions =
			sources->water_impulse_positions;
		link_key_bounds(&world->cold->water_impulse_curves[i].keys,
			sources->curve_bounds, &cursor);
	}
	if (cursor != world->cold->curve_bound_count)
		world_fail("curve bound table size mismatch");
	world->cold->water_impulse_curves[0].values =
		sources->water_impulse_vertical_values;
	world->cold->water_impulse_curves[1].values =
		sources->water_impulse_horizontal_values;

	CSceneVehicleCarTuningCurveSet *set = &world->cold->curve_set;
	set->accel_from_speed = &world->cold->active_curves[CURVE_ACCEL];
	set->rollover_lateral_from_speed =
		&world->cold->active_curves[CURVE_ROLLOVER_LATERAL];
	set->max_side_friction_from_speed =
		&world->cold->active_curves[CURVE_MAX_SIDE_FRICTION];
	set->lateral_contact_slowdown_from_speed =
		&world->cold->active_curves[CURVE_LATERAL_CONTACT_SLOWDOWN];
	set->steer_slowdown_from_speed =
		&world->cold->active_curves[CURVE_STEER_SLOWDOWN];
	set->rollover_lateral_coef_from_angle =
		&world->cold->active_curves[CURVE_ROLLOVER_ANGLE];
	set->steer_drive_torque_from_speed =
		&world->cold->active_curves[CURVE_STEER_DRIVE_TORQUE];
	set->m4_steer_radius_from_speed =
		&world->cold->active_curves[CURVE_M4_STEER_RADIUS];
	set->m4_max_friction_force_from_speed =
		&world->cold->active_curves[CURVE_M4_MAX_FRICTION];
	set->m5_slipping_accel_from_speed =
		&world->cold->active_curves[CURVE_M5_SLIPPING_ACCEL];
	set->water_friction_from_speed =
		&world->cold->active_curves[CURVE_WATER_FRICTION];
	set->m6_damper_modulation =
		&world->cold->active_curves[CURVE_M6_DAMPER_MODULATION];
	set->m6_rear_gear_accel_from_speed =
		&world->cold->active_curves[CURVE_M6_REAR_GEAR_ACCEL];
	set->m6_rollover_lateral_from_speed_ratio =
		&world->cold->active_curves[CURVE_M6_ROLLOVER_RATIO];
	set->m6_burnout_radius_from_speed =
		&world->cold->active_curves[CURVE_M6_BURNOUT_RADIUS];
	set->m6_lateral_speed_from_burnout_radius =
		&world->cold->active_curves[CURVE_M6_BURNOUT_LATERAL_SPEED];
	set->m6_donut_rollover_from_speed =
		&world->cold->active_curves[CURVE_M6_DONUT_ROLLOVER];
	set->m6_burnout_rollover_from_speed =
		&world->cold->active_curves[CURVE_M6_BURNOUT_ROLLOVER];
}

TMNF_HD static void link_vehicle(
	TmnfWorld *world, const TmnfWorldLinkSources *sources)
{
	world->dyna.params = &world->dyna_params;
	world->dyna.stateB = &world->committed_state;
	world->dyna.liveState = &world->live_state;
	world->dyna.replacementBuf.data = sources->replacements;
	world->dyna.replacementBuf.capacity = sources->replacement_capacity;
	world->dyna.replacementBuf.count = 0;

	CSceneVehicleCarTuning *tuning = &world->cold->vehicle_tuning;
	tuning->accel_curve = &world->cold->active_curves[CURVE_ACCEL];
	tuning->damper_modulation_curve =
		&world->cold->active_curves[CURVE_M6_DAMPER_MODULATION];
	tuning->gear_ratios = world->cold->gear_ratios;
	tuning->gear_upshift = world->cold->gear_upshift;
	tuning->gear_downshift = world->cold->gear_downshift;
	tuning->gear_aux = world->cold->gear_aux;

	world->vehicle.hms_item = &world->cold->item;
	world->vehicle.dyna_state = &world->live_state;
	world->vehicle.dyna_params = &world->dyna_params;
	world->vehicle.tuning = &world->cold->vehicle_tuning;
	world->vehicle.wheels = world->wheels;
	for (uint32_t i = 0; i < 4; ++i) {
		world->wheels[i].surface_handler = &world->wheel_trees[i];
		world->wheels[i].history = &sources->wheel_history[i];
		world->contact_wheels[i].wheel = &world->wheels[i];
	}

	CSceneVehicleCarTuningAux *aux_tuning = &world->cold->aux_tuning;
	aux_tuning->air_vertical_curve = &world->cold->aux_curves[0];
	aux_tuning->steering_angle_curve = world->cold->steering_angle_curve_present
		? &world->cold->aux_curves[1] : NULL;
	aux_tuning->water_impulse_vertical_curve =
		&world->cold->water_impulse_curves[0];
	aux_tuning->water_impulse_horizontal_curve =
		&world->cold->water_impulse_curves[1];
	aux_tuning->water_friction_curve =
		&world->cold->active_curves[CURVE_WATER_FRICTION];
	aux_tuning->water_map = &sources->track->water;

	world->aux.vehicle = &world->vehicle;
	world->aux.tuning = aux_tuning;
	world->aux.wheels = world->aux_wheels;
	world->aux.runtime = world;
	world->aux.play_turbo_sound = NULL;
	world->aux.set_surface_location = set_surface_location;
	world->aux.finish_integration = finish_vehicle_integration;

	world->cold->contact_tuning.curves = &world->cold->curve_set;
	for (uint32_t i = 0; i < world->cold->vehicle_header->ground_material_count; ++i) {
		world->cold->ground_material_ptrs[i] = &world->cold->ground_materials[i];
		if (world->cold->ground_materials[i].fake_contact_mask != NULL) {
			world->cold->ground_materials[i].fake_contact_mask =
				sources->fake_contact_mask;
		}
	}

	TMNFVehicleContactContext *contact = &world->contact;
	contact->vehicle = &world->vehicle;
	contact->tuning = &world->cold->contact_tuning;
	contact->wheels = world->contact_wheels;
	contact->ground_material_indices = world->cold->ground_material_ids;
	contact->ground_materials = world->cold->ground_material_ptrs;
	contact->timer = &world->timer;
	contact->wheel_tree_refs = world->cold->wheel_tree_refs;
	contact->body_tree_refs = world->cold->body_tree_refs;
	contact->air_control_immediate = &world->aux.air_control_immediate;
	contact->contact_block_count = &world->compute_state.state_5d8;
	contact->event_source_c = &world->compute_state.event_source_c;
	contact->event_source_ab = &world->compute_state.event_source_ab;
	contact->event_metric_a = &world->compute_state.event_metric_a;
	contact->event_metric_b = &world->compute_state.event_metric_b;
	contact->event_metric_c = &world->compute_state.event_metric_c;
	contact->vehicle_contact_rotation = &world->committed_state.rot;
	contact->resolve_body_iso = resolve_body_iso;
	contact->resolve_body_iso_user = world;

	world->cold->model6.vehicle = &world->vehicle;
	world->cold->model6.aux = &world->aux;
	world->cold->model6.contact = &world->contact;
	world->cold->model6.curves = &world->cold->curve_set;
	world->cold->model6.tuning = &world->cold->model6_tuning;
	world->cold->model6.state = &world->model6_state;
	world->cold->model6.model_iso_source =
		(const GmIso4 *)&world->live_state.rot;
	world->cold->model6.body_reference_position =
		(const GmVec3 *)(world->cold->vehicle_blob +
			world->cold->vehicle_header->body_reference_offset);

	world->cold->oldmodels.vehicle = &world->vehicle;
	world->cold->oldmodels.aux = &world->aux;
	world->cold->oldmodels.contact = &world->contact;
	world->cold->oldmodels.curves = &world->cold->curve_set;
	world->cold->oldmodels.tuning = &world->cold->oldmodels_tuning;
	world->cold->oldmodels.shared = &world->model6_state;
	world->cold->oldmodels.state = &world->oldmodels_state;

	world->cold->compute_tuning.effect_curve =
		&world->cold->active_curves[CURVE_EFFECT];
	world->cold->compute_tuning.contact_decay_curve = &world->cold->primary_curves[0];
	world->cold->compute_tuning.contact_rise_curve = &world->cold->primary_curves[1];

	world->cold->item_corpora[0] = &world->cold->collision_corpus;
	world->cold->item_zones[0] = &world->physics;
	world->cold->item.corpora = world->cold->item_corpora;
	world->cold->item.zones = world->cold->item_zones;

	world->cold->compute.vehicle = &world->vehicle;
	world->cold->compute.item = &world->cold->item;
	world->cold->compute.aux = &world->aux;
	world->cold->compute.contact = &world->contact;
	world->cold->compute.tuning = &world->cold->compute_tuning;
	world->cold->compute.state = &world->compute_state;
	world->cold->compute.model6 = &world->cold->model6;
	world->cold->compute.oldmodels = &world->cold->oldmodels;
	world->cold->compute.runtime = world;
	world->cold->compute.post_force = post_vehicle_force;
	world->cold->compute.contact_token = response_body_token;
}

TMNF_HD static uint32_t collision_node_child_count(
	const TmnfWorld *world, uint32_t index)
{
	return world->cold->collision_children[index].kind >>
		VEHICLE_COLLISION_CHILD_SHIFT;
}

/* Links `count` pre-order records starting at *next as the direct children
 * of `parent`, recursing into each child's subtree. Child pointer arrays are
 * carved from collision_child_pointers, which holds exactly one entry per
 * node. */
TMNF_HD static void link_collision_children(
	TmnfWorld *world, CPlugTree *parent, uint32_t count,
	uint32_t *next, uint32_t *pool_next)
{
	uint32_t node_count = world->cold->vehicle_header->collision_child_count;
	if (*pool_next + count > node_count)
		world_fail("vehicle collision tree construction failed");
	parent->children = world->cold->collision_child_pointers + *pool_next;
	parent->child_count = count;
	*pool_next += count;
	for (uint32_t k = 0; k < count; ++k) {
		if (*next >= node_count)
			world_fail("vehicle collision node is missing children");
		uint32_t index = (*next)++;
		CPlugTree *child = world->cold->collision_child_slots[index];
		parent->children[k] = child;
		link_collision_children(
			world, child, collision_node_child_count(world, index),
			next, pool_next);
	}
}

TMNF_HD static void link_collision(
	TmnfWorld *world, const TmnfWorldLinkSources *sources)
{
	CollisionRuntime_Init(&world->cold->collision_runtime);
	world->cold->collision_runtime.get_linear_speed =
		collision_get_linear_speed;
	CHmsCollisionManager_SZone *zone = &world->collision_zone;
	zone->runtime = &world->cold->collision_runtime;
	for (uint32_t i = 0; i < 5; i++)
		zone->groups[i].runtime = &world->cold->collision_runtime;
	zone->groups[4].is_static = 1;
	zone->groups[4].static_entry_count = sources->track->entry_count;
	zone->groups[4].static_entries =
		(HmsStaticCollisionEntry *)(uintptr_t)sources->track->entries;
	/* TmnfTrack_BindStaticGroup: the per-cell tree copies, when built. */
	zone->groups[4].static_grid = sources->track->grid.cell_count != 0 ?
		&sources->track->grid : NULL;
	zone->general_buffer = NULL;
	zone->static_group = NULL;
	zone->merge_buffers = world->merge_storage;
	zone->merge_buffer_count = 0;
	zone->merge_buffer_capacity = TMNF_WORLD_MERGE_CAPACITY;

	uint32_t node_count = world->cold->vehicle_header->collision_child_count;
	for (uint32_t i = 0; i < node_count; ++i) {
		const struct TmnfVehicleCollisionTree *raw =
			&world->cold->collision_children[i];
		uint32_t kind = raw->kind & VEHICLE_COLLISION_KIND_MASK;
		GmSurfEllipsoid *shape;
		CPlugSurface *surface;
		CPlugTree *tree;
		if (kind == VEHICLE_COLLISION_TREE_WHEEL) {
			uint32_t wheel_index = raw->wheel_index;
			shape = &world->cold->wheel_ellipsoids[wheel_index];
			surface = &world->cold->wheel_surfaces[wheel_index];
			tree = &world->wheel_trees[wheel_index];
		} else {
			shape = &world->cold->body_ellipsoids[i];
			surface = &world->cold->body_surfaces[i];
			tree = &world->cold->body_trees[i];
		}
		if (raw->surface_id != 0) {
			surface->geom = &shape->base;
			surface->material_ids =
				world->cold->collision_material_ids + raw->material_index;
			tree->surface = surface;
		} else {
			tree->surface = NULL;
		}
		tree->contact_buffer = &world->contact_buffers[i];
		world->cold->collision_child_slots[i] = tree;
	}
	/* Every buffer slot is bound, not only the node_count in use: a world
	 * copied back from the device carries device record pointers in the
	 * spare slots until this relinks them. */
	for (uint32_t i = 0; i < VEHICLE_COLLISION_MAX_NODES; ++i) {
		world->contact_buffers[i].base.collisions.data =
			sources->contact_records[i];
		world->contact_buffers[i].base.collisions.capacity =
			sources->contact_capacities[i];
		world->contact_buffers[i].base.collisions.count = 0;
		world->contact_buffers[i].active = 0;
	}
	/* Link the pre-order records. The root owns every record not claimed
	 * as a child by an earlier node. */
	uint32_t pending = 0;
	uint32_t root_child_count = 0;
	for (uint32_t i = 0; i < node_count; ++i) {
		if (pending != 0)
			pending--;
		else
			root_child_count++;
		pending += collision_node_child_count(world, i);
	}
	uint32_t next = 0;
	uint32_t pool_next = 0;
	link_collision_children(
		world, &world->cold->root_tree, root_child_count, &next, &pool_next);
	if (next != node_count || pool_next != node_count)
		world_fail("vehicle collision tree construction failed");
	world->cold->root_tree.surface = NULL;
	world->cold->root_tree.contact_buffer = NULL;

	world->cold->collision_corpus.dyna = &world->dyna;
	world->cold->collision_corpus.live_iso =
		(const GmIso4 *)&world->live_state.rot;
	world->cold->collision_corpus.tree = &world->cold->root_tree;

	CHmsCollisionManager_SGroup *dynamic = &zone->groups[0];
	world->cold->dynamic_corpora[0] = &world->cold->collision_corpus;
	dynamic->corpora = world->cold->dynamic_corpora;
	dynamic->speed_sq = world->dynamic_speed_sq;
	dynamic->device_mats = &world->cold->static_device;
	world->cold->static_device.group = &zone->groups[4];
	world->cold->static_device.perform.data = NULL;

	world->cold->player_response_body.dyna = &world->dyna;
	world->cold->player_response_body.absorb_contact = absorb_player_contact;
	world->cold->player_response_body.contact_user = world;
}

TMNF_HD static void link_physics(
	TmnfWorld *world, const TmnfWorldLinkSources *sources)
{
	world->cold->track = sources->track;
	world->physics_corpus.collision_corpus = &world->cold->collision_corpus;
	world->physics_corpus.dyna = &world->dyna;
	world->physics_corpus.vehicle = &world->vehicle;
	world->physics_corpus.vehicle_compute = &world->cold->compute;
	world->physics.force_fields = &world->cold->gravity;
	world->physics.corpora = &world->physics_corpus;
	world->physics.collision_zone = &world->collision_zone;
	world->physics.collision_buffer.collisions.data =
		sources->collision_records;
	world->physics.collision_buffer.collisions.capacity =
		sources->collision_capacity;
	world->physics.collision_buffer.collisions.count = 0;
	world->physics.response_zone.collisions = NULL;
	world->physics.response_zone.surface_materials =
		(const CPlugSurfaceMaterialData *)sources->track->materials;
	world->physics.response_zone.resolve_body = resolve_response_body;
	world->physics.response_zone.resolve_material =
		resolve_response_material;
	world->physics.response_zone.resolver_user = world;
	world->physics.route = sources->route;
}

TMNF_HD void World_LinkPointers(
	TmnfWorld *world, const TmnfWorldLinkSources *sources)
{
	world->cold = (struct TmnfWorldCold *)sources->cold;
	link_blob(world, sources->vehicle_blob);
	link_curves(world, sources);
	link_vehicle(world, sources);
	link_collision(world, sources);
	link_physics(world, sources);
}

TMNF_HD void World_LinkScratch(
	TmnfWorld *world, const TmnfWorldLinkSources *sources)
{
	for (uint32_t i = 0; i < 4; ++i)
		world->wheels[i].history = &sources->wheel_history[i];
	world->dyna.replacementBuf.data = sources->replacements;
	world->physics.collision_buffer.collisions.data =
		sources->collision_records;
	for (uint32_t i = 0; i < TMNF_WORLD_CONTACT_BUFFER_COUNT; ++i) {
		world->contact_buffers[i].base.collisions.data =
			sources->contact_records[i];
	}
}

void World_GetLinkSources(
	const TmnfWorld *world, TmnfWorldLinkSources *sources)
{
	memset(sources, 0, sizeof(*sources));
	sources->cold = world->cold;
	sources->wheel_history = world->wheels[0].history;
	sources->vehicle_blob = world->cold->vehicle_blob;
	sources->track = world->cold->track;
	sources->curve_bounds = world->cold->curve_bounds;
	sources->fake_contact_mask = TMNF_FAKE_CONTACT_MASK;
	sources->water_impulse_positions = TMNF_WATER_IMPULSE_POSITIONS;
	sources->water_impulse_vertical_values =
		TMNF_WATER_IMPULSE_VERTICAL_VALUES;
	sources->water_impulse_horizontal_values =
		TMNF_WATER_IMPULSE_HORIZONTAL_VALUES;
	sources->route = world->physics.route;
	sources->collision_records = world->physics.collision_buffer.collisions.data;
	sources->collision_capacity =
		world->physics.collision_buffer.collisions.capacity;
	for (uint32_t i = 0; i < TMNF_WORLD_CONTACT_BUFFER_COUNT; ++i) {
		sources->contact_records[i] =
			world->contact_buffers[i].base.collisions.data;
		sources->contact_capacities[i] =
			world->contact_buffers[i].base.collisions.capacity;
	}
	sources->replacements = world->dyna.replacementBuf.data;
	sources->replacement_capacity = world->dyna.replacementBuf.capacity;
}

/* Rounded up to the 64-byte allocation boundary worlds are placed on. */
TMNF_HD size_t World_Size(void)
{
	return (sizeof(TmnfWorld) + TMNF_WORLD_ALIGNMENT - 1) &
		~(size_t)(TMNF_WORLD_ALIGNMENT - 1);
}

TMNF_HD size_t World_ColdSize(void)
{
	return (sizeof(struct TmnfWorldCold) + TMNF_WORLD_ALIGNMENT - 1) &
		~(size_t)(TMNF_WORLD_ALIGNMENT - 1);
}

const void *World_GetCold(const TmnfWorld *world)
{
	return world->cold;
}

const uint8_t *World_GetVehicleBlob(const TmnfWorld *world, uint32_t *size)
{
	*size = world->cold->vehicle_blob_size;
	return world->cold->vehicle_blob;
}

const TmnfTrack *World_GetTrack(const TmnfWorld *world)
{
	return world->cold->track;
}

const TmnfPhysicsWorld *World_GetPhysicsWorldConst(const TmnfWorld *world)
{
	return &world->physics;
}

const float *World_GetCurveBounds(const TmnfWorld *world, uint32_t *count)
{
	*count = world->cold->curve_bound_count;
	return world->cold->curve_bounds;
}

static void finish_snapshot_tick(TmnfWorld *world)
{
	CHmsZoneDynamic_PhysicsStep2(&world->physics, 10);
}

static void select_tuning(const TmnfWorld *world, uint32_t key)
{
	if (key >= world->cold->vehicle_header->tuning_count)
		world_fail("vehicle tuning key is out of range");
	const struct TmnfVehicleTuningDescriptor *descriptor =
		&world->cold->tuning_descriptors[key];
	if (descriptor->key != key)
		world_fail("vehicle tuning key does not match descriptor");
}

static SHmsPhysicalCollision *allocate_records(uint32_t capacity)
{
	return (SHmsPhysicalCollision *)allocate(
		(size_t)capacity * sizeof(SHmsPhysicalCollision));
}

int World_HasLocalAssets(void) { return TMNF_HAS_GAME_MASK; }

TmnfWorld *World_Create(
	const TmnfTrack *track,
	const char *vehicle_snapshot_path)
{
	if (!TMNF_HAS_GAME_MASK)
		world_fail("local game mask missing; see docs/LOCAL_ASSETS.md and rebuild with TMNF_GAME_MASK");
	if (track == NULL || vehicle_snapshot_path == NULL)
		world_fail("track handle or vehicle snapshot path is null");
	TmnfWorld *world = (TmnfWorld *)aligned_alloc(
		TMNF_WORLD_ALIGNMENT, World_Size());
	if (world == NULL)
		world_fail("out of memory");
	memset(world, 0, World_Size());
	/* Page-aligned so that a test can map it read-only. */
	world->cold = (struct TmnfWorldCold *)aligned_alloc(
		4096, (World_ColdSize() + 4095) & ~(size_t)4095);
	if (world->cold == NULL)
		world_fail("out of memory");
	memset(world->cold, 0, (World_ColdSize() + 4095) & ~(size_t)4095);
	uint32_t blob_size;
	uint8_t *blob = read_vehicle_file(vehicle_snapshot_path, &blob_size);
	world->cold->vehicle_blob_size = blob_size;
	(void)validate_vehicle_graph(blob, blob_size, track->header->track_sha256);
	link_blob(world, blob);
	select_tuning(world, world->cold->vehicle_header->active_tuning_key);
	world->cold->track = track;

	decode_vehicle(world);
	CSceneVehicleCarWheelHistory *history = (CSceneVehicleCarWheelHistory *)allocate(
		4 * sizeof(*history));
	for (uint32_t i = 0; i < 4; ++i) {
		world->wheels[i].history = &history[i];
		memcpy(&history[i], world->cold->raw_wheels +
			i * TMNF_CSCENE_VEHICLE_CAR_WHEEL_GAME_SIZE + 0x16c,
			4 * sizeof(CSceneVehicleCarWheelState));
	}
	world->cold->collision_corpus.object_ref = track->static_response_count;
	decode_collision_world(world);
	world->contact.body_tree_count = world->cold->body_tree_count;

	/* Host scratch: the game's initial CFastBuffer capacities (0x32 for
	 * collision buffers, the captured replacement capacity), grown on
	 * demand by the buffers' 1.5x policy. */
	TmnfWorldLinkSources sources;
	memset(&sources, 0, sizeof(sources));
	sources.cold = world->cold;
	sources.wheel_history = world->wheels[0].history;
	sources.vehicle_blob = blob;
	sources.track = track;
	world->cold->curve_bound_count = 0;
	for (uint32_t i = 0; i < ACTIVE_CURVE_COUNT; ++i)
		world->cold->curve_bound_count += 2 * world->cold->active_curves[i].keys.count;
	for (uint32_t i = 0; i < PRIMARY_CURVE_COUNT; ++i)
		world->cold->curve_bound_count += 2 * world->cold->primary_curves[i].keys.count;
	for (uint32_t i = 0; i < 2; ++i)
		world->cold->curve_bound_count += 2 * world->cold->water_impulse_curves[i].keys.count;
	world->cold->curve_bounds = (float *)allocate(
		(size_t)world->cold->curve_bound_count * sizeof(float));
	sources.curve_bounds = world->cold->curve_bounds;
	sources.fake_contact_mask = TMNF_FAKE_CONTACT_MASK;
	sources.water_impulse_positions = TMNF_WATER_IMPULSE_POSITIONS;
	sources.water_impulse_vertical_values =
		TMNF_WATER_IMPULSE_VERTICAL_VALUES;
	sources.water_impulse_horizontal_values =
		TMNF_WATER_IMPULSE_HORIZONTAL_VALUES;
	sources.collision_capacity = 0x32;
	sources.collision_records = allocate_records(0x32);
	for (uint32_t i = 0; i < VEHICLE_COLLISION_MAX_NODES; ++i) {
		sources.contact_records[i] = allocate_records(0x32);
		sources.contact_capacities[i] = 0x32;
	}
	sources.replacement_capacity = raw_u32(world->cold->raw_dyna, 0x338);
	if (sources.replacement_capacity != 0) {
		sources.replacements = (GmVec3 *)allocate(
			(size_t)sources.replacement_capacity * sizeof(GmVec3));
	}
	World_LinkPointers(world, &sources);
	/* The bounds are a function of the linked positions: fill the table the
	 * pointers now address. */
	for (uint32_t i = 0; i < ACTIVE_CURVE_COUNT; ++i)
		CFuncKeys_CompileInto(&world->cold->active_curves[i].keys);
	for (uint32_t i = 0; i < PRIMARY_CURVE_COUNT; ++i)
		CFuncKeys_CompileInto(&world->cold->primary_curves[i].keys);
	for (uint32_t i = 0; i < 2; ++i)
		CFuncKeys_CompileInto(&world->cold->water_impulse_curves[i].keys);

	world->cold->gravity.active = 1;
	world->cold->gravity.value = (GmVec3){ 0.0f, -10.0f, 0.0f };
	world->physics_corpus.scene_flags = 0x2000u;
	world->physics_corpus.tick_time = world->timer.tick_time;
	world->physics.linear_drag_scale = 1.0f;
	world->physics.angular_drag_scale = 1.0f;
	world->physics.force_field_count = 1;
	world->physics.corpus_count = 1;
	world->physics.response_zone.surface_material_count =
		TMNF_TRACK_MATERIAL_COUNT;
	/*
	 * The snapshot begins at Model6 entry, after ComputeForces changed this
	 * field from the preceding grounded value to the airborne value. Restore
	 * the value consumed by the in-progress ComputeCorpusForces call.
	 */
	world->dyna_params.forceFieldScale =
		world->cold->compute_tuning.grounded_force_field_scale;
	finish_snapshot_tick(world);
	return world;
}

void World_Destroy(TmnfWorld *world)
{
	if (world == NULL)
		return;
	for (uint32_t i = 0; i < VEHICLE_COLLISION_MAX_NODES; ++i)
		CHmsCollisionBuffer_Destroy(&world->contact_buffers[i].base);
	CHmsCollisionBuffer_Destroy(&world->physics.collision_buffer);
	free(world->dyna.replacementBuf.data);
	free(world->wheels[0].history);
	free(world->cold->curve_bounds);
	free(world->cold->vehicle_blob);
	free(world->cold);
	free(world);
}

TMNF_HD TmnfPhysicsWorld *World_GetPhysicsWorld(TmnfWorld *world)
{
	return &world->physics;
}

TMNF_HD CSceneVehicleCar *World_GetPlayerVehicle(TmnfWorld *world)
{
	return &world->vehicle;
}

TMNF_HD const CHmsStateDyna *World_GetPlayerState(const TmnfWorld *world)
{
	return &world->live_state;
}

TMNF_HD const CFastBuffer_SHmsPhysicalCollision *World_GetLastCollisions(
	const TmnfWorld *world)
{
	return &world->physics.collision_buffer.collisions;
}

TMNF_HD const CHmsReplacementBuf *World_GetPlayerReplacements(
	const TmnfWorld *world)
{
	return &world->dyna.replacementBuf;
}

TMNF_HD uint32_t World_GetTimerTick(const TmnfWorld *world)
{
	return world->timer.tick_time;
}

void World_AdvanceTimer(TmnfWorld *world, uint32_t tick_ms)
{
	TmnfPhysicsCorpus_AdvanceTimer(&world->physics_corpus, tick_ms);
}

TMNF_HD void World_Respawn(TmnfWorld *world, const GmIso4 *spawn)
{
	if (world == NULL)
		world_fail("respawn argument is null");
	TmnfVehicle_Respawn(&world->physics_corpus, spawn);
}

TMNF_HD void World_GetPlayerObservation(
	const TmnfWorld *world, TmnfObservation *observation)
{
	const CHmsStateDyna *state = &world->live_state;
	observation->position = state->pos;
	observation->rotation = state->quat;
	observation->linear_speed = state->linVel;
	observation->angular_speed = state->angVel;
	for (uint32_t i = 0; i < 4; ++i)
		observation->wheel_speed[i] =
			world->wheels[i].real_time.field6c;
	observation->engine_rpm = world->vehicle.engine.rpm;
	observation->gear = world->vehicle.engine.gear;
}

TMNF_HD static void patch_model6_state(uint8_t *car, const TmnfWorld *world)
{
	const CSceneVehicleCarModel6State *state = &world->model6_state;
	memcpy(car + 0x1dc, &state->pivot_position, 12);
	memcpy(car + 0x1e8, &state->pivot_axis, 12);
	memcpy(car + 0x5c4, &state->reverse_mode, 4);
	memcpy(car + 0x5cc, &state->reverse_speed_threshold, 4);
	memcpy(car + 0x5d8, &state->contact_block_count, 4);
	memcpy(car + 0x5dc, &state->side_contact, 4);
	memcpy(car + 0x62c, &state->last_sliding_tick, 4);
	memcpy(car + 0x630, &state->sliding_start_tick, 4);
	memcpy(car + 0x634, &state->sliding_elapsed_ticks, 4);
	memcpy(car + 0x6a4, &state->model_iso, 0x30);
	memcpy(car + 0x6d4, &state->rollover_axis, 12);
	memcpy(car + 0x6e0, &state->orbit_center, 12);
	memcpy(car + 0x6ec, &state->orbit_initial_radius, 4);
	memcpy(car + 0x6f0, &state->orbit_radius, 4);
	memcpy(car + 0x6f4, &state->burnout_start_tick, 4);
	memcpy(car + 0x6f8, &state->burnout_transition_tick, 4);
	memcpy(car + 0x6fc, &state->orbit_axis, 12);
	memcpy(car + 0x708, &state->orbit_sign, 4);
	memcpy(car + 0x840, &state->axle_width, 4);
}

TMNF_HD static void patch_oldmodels_state(uint8_t *car, const TmnfWorld *world)
{
	const CSceneVehicleCarOldModelsState *state = &world->oldmodels_state;
	memcpy(car + 0x638, &state->m4_drift_angle, 4);
	memcpy(car + 0x63c, &state->m4_drift_steer_sign, 4);
	memcpy(car + 0x640, &state->m4_drift_state, 4);
	memcpy(car + 0x648, &state->m5_last_steer_tick, 4);
	memcpy(car + 0x64c, &state->m5_last_steer_sliding, 4);
}

TMNF_HD static void patch_compute_state(uint8_t *car, const TmnfWorld *world)
{
	const TMNFVehicleComputeState *state = &world->compute_state;
	memcpy(car + 0x1e8, &state->simulation_gate, 4);
	memcpy(car + 0x1fc, &state->event_level_c, 4);
	car[0x200] = state->event_source_c;
	car[0x201] = state->event_source_ab;
	memcpy(car + 0x214, &state->spring_c, sizeof(state->spring_c));
	memcpy(car + 0x228, &state->spring_a, sizeof(state->spring_a));
	memcpy(car + 0x23c, &state->contact_rise, 4);
	memcpy(car + 0x240, &state->contact_decay, 4);
	memcpy(car + 0x5b0, &state->computed_brake_force, 4);
	memcpy(car + 0x5d8, &state->state_5d8, 4);
	memcpy(car + 0x610, &state->air_impulse_cooldown_tick, 4);
	memcpy(car + 0x624, &state->effect_accumulator, 4);
	memcpy(car + 0x650, &state->last_force_tick, 4);
	memcpy(car + 0x654, &state->event_level_a, 4);
	memcpy(car + 0x658, &state->event_level_b, 4);
	memcpy(car + 0x660, &state->peak_event_level_b, 4);
	memcpy(car + 0x664, &state->peak_event_level_a, 4);
	memcpy(car + 0x668, &state->peak_event_level_c, 4);
	car[0x66c] = state->peak_event_source_ab;
	car[0x66d] = state->peak_event_source_c;
	memcpy(car + 0x670, &state->event_metric_a, 4);
	memcpy(car + 0x674, &state->event_metric_b, 4);
	memcpy(car + 0x678, &state->event_metric_c, 4);
	memcpy(car + 0x6d4, &state->normalized_force, 12);
	memcpy(car + 0x74c, &state->air_effect_mode, 4);
}

TMNF_HD void World_WritePlayerGameState(
	const TmnfWorld *world, uint8_t car[CAR_SIZE],
	uint8_t wheels[4 * WHEEL_SIZE])
{
	memcpy(car, world->cold->raw_car, CAR_SIZE);
	memcpy(wheels, world->cold->raw_wheels, 4 * WHEEL_SIZE);
	patch_model6_state(car, world);
	patch_oldmodels_state(car, world);
	patch_compute_state(car, world);

	const CSceneVehicleCar *vehicle = &world->vehicle;
	memcpy(car + 0x050, &vehicle->input_gas, 4);
	memcpy(car + 0x054, &vehicle->input_brake, 4);
	memcpy(car + 0x058, &vehicle->input_steer, 4);
	memcpy(car + 0x2e0, &world->compute_state.local_speed_limit, 4);
	memcpy(car + 0x2e4, &vehicle->engine_mode, 4);
	memcpy(car + 0x2e8, &vehicle->wheel_count, 4);
	memcpy(car + 0x59c, &vehicle->engine, sizeof(vehicle->engine));
	memcpy(car + 0x5d0, &world->aux.turbo_epoch_tick, 4);
	memcpy(car + 0x5d4, &world->aux.air_control_immediate, 4);
	memcpy(car + 0x5d8, &world->compute_state.state_5d8, 4);
	memcpy(car + 0x5dc, &world->contact.side_contact, 4);
	memcpy(car + 0x5e0, &world->contact.last_side_contact_tick, 4);
	memcpy(car + 0x5e4, &world->aux.air_control_locked, 4);
	memcpy(car + 0x5e8, &world->aux.steering_value, 4);
	memcpy(car + 0x5f0, &world->aux.turbo_progress, 4);
	memcpy(car + 0x5f4, &world->aux.turbo_factor, 4);
	memcpy(car + 0x5f8, &world->aux.turbo_start_tick, 4);
	memcpy(car + 0x5fc, &world->aux.turbo_end_tick, 4);
	memcpy(car + 0x600, &world->aux.turbo_type, 4);
	memcpy(car + 0x604, &world->aux.roulette_token, 4);
	memcpy(car + 0x608, &world->aux.roulette_value, 4);
	memcpy(car + 0x60c, &vehicle->flag_60c, 4);
	memcpy(car + 0x614, &world->aux.air_control_tick, 4);
	memcpy(car + 0x618, &world->aux.air_control_speed, 12);
	memcpy(car + 0x628, &vehicle->turbo_active, 4);
	memcpy(car + 0x69c, &vehicle->drive_mode, 4);
	memcpy(car + 0x6a0, &vehicle->force_wheel_speed, 4);
	memcpy(car + 0x70c, &vehicle->current_local_speed, 12);
	memcpy(car + 0x73c, &vehicle->block_wheel_speed, 4);
	memcpy(car + 0x744, &vehicle->engine_limit_flag, 4);
	memcpy(car + 0x748, &vehicle->gear_downshift_flag, 4);
	memcpy(car + 0x818, &vehicle->total_force_added, 12);
	memcpy(car + 0x824, &vehicle->total_impulse_added, 12);

	for (uint32_t i = 0; i < 4; ++i) {
		uint8_t *raw = wheels + i * WHEEL_SIZE;
		const CSceneVehicleCarWheel *wheel = &world->wheels[i];
		memcpy(raw + 0x000, &wheel->active, 4);
		memcpy(raw + 0x004, &wheel->steerable, 4);
		memcpy(raw + 0x008, &wheel->radius, 4);
		memcpy(raw + 0x010,
			&world->aux_wheels[i].surface_source, sizeof(GmIso4));
		memcpy(raw + 0x040,
			&world->aux_wheels[i].surface_location, sizeof(GmIso4));
		memcpy(raw + 0x070, wheel->field70, sizeof(wheel->field70));
		memcpy(raw + 0x0a0, &wheel->fielda0, 4);
		memcpy(raw + 0x0a4, &wheel->fielda4, 4);
		memcpy(raw + 0x0a8, &wheel->offset_from_vehicle, 12);
		memcpy(raw + 0x0b4, &wheel->real_time, sizeof(wheel->real_time));
		memcpy(raw + 0x15c, &wheel->field15c, 4);
		memcpy(raw + 0x160,
			&wheel->contact_relative_local_distance, 12);
		memcpy(raw + 0x16c, &wheel->history->previous_sync,
			sizeof(wheel->history->previous_sync));
		memcpy(raw + 0x1d0, &wheel->history->sync, sizeof(wheel->history->sync));
		memcpy(raw + 0x234, &wheel->history->field234, sizeof(wheel->history->field234));
		memcpy(raw + 0x298, &wheel->history->async_state,
			sizeof(wheel->history->async_state));
	}
}
