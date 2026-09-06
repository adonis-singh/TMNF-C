#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "collision.h"
#include "hms_dyna.h"
#include "trace_format.h"
#include "../oracle/tracer/broadphase_trace.h"

enum {
	PHYSICS_THIS_SIZE = 0x180,
	PHYSICS_COLLISION_COUNT_OFFSET = 0x15c,
};

typedef struct {
	const struct TmnfBroadphaseTraceHeader *header;
	const struct TmnfBroadphaseGroup *groups;
	const uint32_t *corpus_ids;
	const struct TmnfBroadphaseCorpus *corpora;
	const struct TmnfBroadphaseDyna *dynas;
	const struct TmnfBroadphaseState *states;
	const struct TmnfBroadphaseSpeed *speeds;
	const float *speed_values;
	const struct TmnfBroadphaseDevice *devices;
	const struct TmnfBroadphaseTable *tables;
	const int32_t *table_values;
	const struct TmnfBroadphaseZone *zones;
	const struct TmnfBroadphaseStaticEntry *static_entries;
	const struct TmnfBroadphaseTree *trees;
	const uint32_t *tree_child_ids;
	const struct TmnfBroadphaseIso *isos;
	const struct TmnfBroadphasePlug *plugs;
	const struct TmnfDetectSurface *surfaces;
	const GmVec3 *vertices;
	const GmSurfMeshFace *faces;
	const GmSurfMeshNode *nodes;
	const uint8_t *material_ids;
	const struct TmnfDetectBuffer *buffers;
	const SHmsPhysicalCollision *collision_records;
	const uint32_t *merge_buffer_ids;
} BroadphaseView;

typedef struct {
	uint32_t id;
	float *values;
	uint32_t count;
	uint32_t capacity;
} ReplaySpeed;

typedef struct {
	uint32_t id;
	int32_t *values;
	uint32_t rows;
	uint32_t columns;
	uint32_t reserved;
	uint32_t stride;
} ReplayTable;

typedef struct {
	CHmsCollisionManager_SGroup value;
	CHmsCorpus **corpora;
	CPlugMaterial_SDeviceMat *devices;
	HmsStaticCollisionEntry *static_entries;
} ReplayGroup;

typedef struct {
	GmSurf base;
	GmSurfSphere sphere;
	GmSurfEllipsoid ellipsoid;
	GmSurfBox box;
	GmSurfMesh mesh;
	GmVec3 *vertices;
	GmSurfMeshFace *faces;
	GmSurfMeshNode *nodes;
} ReplaySurface;

typedef struct {
	uint32_t id;
	SHmsSphereBufferContact sphere;
} ReplayBuffer;

typedef struct {
	CPlugTree value;
	CPlugTree **children;
} ReplayTree;

typedef struct {
	BroadphaseView input;
	CollisionRuntime runtime;
	ReplayGroup *groups;
	CHmsCorpus *corpora;
	CHmsDyna *dynas;
	CHmsStateDyna *states;
	ReplaySpeed *speeds;
	ReplayTable *tables;
	ReplaySurface *surfaces;
	CPlugSurface *plugs;
	uint8_t **plug_material_ids;
	ReplayBuffer *buffers;
	ReplayTree *trees;
	GmIso4 *isos;
	CHmsCollisionManager_SZone zone;
} ReplayGraph;

typedef struct {
	uint32_t va;
	const char *name;
	const char *file_name;
	int physics_window;
	int broadphase;
} Target;

static const Target targets[] = {
	{
		0x0053B1C0u,
		"CHmsCollisionManager::SZone::DetectCollisionsCorpus",
		"0053B1C0_SZone_DetectCollisionsCorpus.bin",
		0, 1,
	},
	{
		0x0053AFB0u,
		"CHmsCollisionManager::SZone::DetectCollisionBetween",
		"0053AFB0_SZone_DetectCollisionBetween.bin",
		0, 1,
	},
	{
		0x0053A8F0u,
		"CHmsCollisionManager::SZone::ComputeCollision",
		"0053A8F0_SZone_ComputeCollision.bin",
		0, 1,
	},
	{
		0x0053A120u,
		"CHmsCollisionManager::SZone::"
		"DetectCollisionBetweenTreeAndStaticCollisionTree",
		"0053A120_SZone_DetectCollisionBetweenTreeAndStaticCollisionTree.bin",
		0, 1,
	},
	{
		0x0053A0E0u,
		"CHmsCollisionManager::SZone::PrepareCollisions",
		"0053A0E0_SZone_PrepareCollisions.bin",
		0, 1,
	},
	{
		0x00537E80u,
		"CHmsCollisionManager::SGroup::ComputeNonStaticCorpusInfos",
		"00537E80_SGroup_ComputeNonStaticCorpusInfos.bin",
		0, 1,
	},
	{
		0x00537F30u,
		"CHmsCollisionManager::SGroup::ComputeIsToPerformCollisions",
		"00537F30_SGroup_ComputeIsToPerformCollisions.bin",
		0, 1,
	},
	{
		0x00549C90u,
		"CHmsZoneDynamic::PhysicsStep2",
		"00549C90_CHmsZoneDynamic_PhysicsStep2.bin",
		1, 0,
	},
};

static int diagnose;

static void fail(const char *message, const char *path)
{
	fprintf(stderr, "%s: %s\n", message, path);
	exit(2);
}

static void *allocate(size_t size)
{
	void *memory = calloc(1, size == 0 ? 1 : size);
	if (memory == NULL) {
		fprintf(stderr, "out of memory\n");
		exit(2);
	}
	return memory;
}

static void read_exact(
	FILE *file, void *destination, size_t size, const char *path)
{
	if (fread(destination, 1, size, file) != size)
		fail("unexpected EOF", path);
}

static uint32_t read_u32(FILE *file, const char *path)
{
	uint8_t bytes[4];
	read_exact(file, bytes, sizeof(bytes), path);
	return (uint32_t)bytes[0]
		| (uint32_t)bytes[1] << 8
		| (uint32_t)bytes[2] << 16
		| (uint32_t)bytes[3] << 24;
}

static uint16_t read_u16(FILE *file, const char *path)
{
	uint8_t bytes[2];
	read_exact(file, bytes, sizeof(bytes), path);
	return (uint16_t)(bytes[0] | (uint16_t)bytes[1] << 8);
}

static uint8_t *read_blob(FILE *file, uint32_t size, const char *path)
{
	uint8_t *blob = allocate(size);
	read_exact(file, blob, size, path);
	return blob;
}

static uint32_t raw_u32(const uint8_t *bytes, uint32_t offset)
{
	uint32_t value;
	memcpy(&value, bytes + offset, sizeof(value));
	return value;
}

static int range_valid(
	uint32_t offset, uint32_t count, uint32_t stride, uint32_t size)
{
	uint64_t end = (uint64_t)offset + (uint64_t)count * stride;
	return offset >= sizeof(struct TmnfBroadphaseTraceHeader)
		&& offset <= size && end <= size;
}

static BroadphaseView broadphase_view(const uint8_t *blob, uint32_t size)
{
	BroadphaseView view;
	const struct TmnfBroadphaseTraceHeader *h;
	memset(&view, 0, sizeof(view));
	if (size < sizeof(struct TmnfBroadphaseTraceHeader)) {
		fprintf(stderr, "truncated broadphase graph\n");
		exit(2);
	}
	h = (const struct TmnfBroadphaseTraceHeader *)blob;
	if (memcmp(h->magic, TMNF_BROADPHASE_TRACE_MAGIC, 8) != 0
	    || h->version != TMNF_BROADPHASE_TRACE_VERSION
	    || h->total_size != size
	    || !range_valid(h->groups_offset, h->group_count,
		    sizeof(struct TmnfBroadphaseGroup), size)
	    || !range_valid(h->corpus_ids_offset, h->corpus_id_count, 4, size)
	    || !range_valid(h->corpora_offset, h->corpus_count,
		    sizeof(struct TmnfBroadphaseCorpus), size)
	    || !range_valid(h->dynas_offset, h->dyna_count,
		    sizeof(struct TmnfBroadphaseDyna), size)
	    || !range_valid(h->states_offset, h->state_count,
		    sizeof(struct TmnfBroadphaseState), size)
	    || !range_valid(h->speeds_offset, h->speed_count,
		    sizeof(struct TmnfBroadphaseSpeed), size)
	    || !range_valid(h->speed_values_offset, h->speed_value_count, 4, size)
	    || !range_valid(h->devices_offset, h->device_count,
		    sizeof(struct TmnfBroadphaseDevice), size)
	    || !range_valid(h->tables_offset, h->table_count,
		    sizeof(struct TmnfBroadphaseTable), size)
	    || !range_valid(h->table_values_offset, h->table_value_count, 4,
		    size)
	    || !range_valid(h->zones_offset, h->zone_count,
		    sizeof(struct TmnfBroadphaseZone), size)
	    || !range_valid(h->static_entries_offset, h->static_entry_count,
		    sizeof(struct TmnfBroadphaseStaticEntry), size)
	    || !range_valid(h->trees_offset, h->tree_count,
		    sizeof(struct TmnfBroadphaseTree), size)
	    || !range_valid(h->tree_child_ids_offset, h->tree_child_id_count,
		    4, size)
	    || !range_valid(h->isos_offset, h->iso_count,
		    sizeof(struct TmnfBroadphaseIso), size)
	    || !range_valid(h->plugs_offset, h->plug_count,
		    sizeof(struct TmnfBroadphasePlug), size)
	    || !range_valid(h->surfaces_offset, h->surface_count,
		    sizeof(struct TmnfDetectSurface), size)
	    || !range_valid(h->vertices_offset, h->vertex_count,
		    sizeof(GmVec3), size)
	    || !range_valid(h->faces_offset, h->face_count,
		    sizeof(GmSurfMeshFace), size)
	    || !range_valid(h->nodes_offset, h->node_count,
		    sizeof(GmSurfMeshNode), size)
	    || !range_valid(h->material_ids_offset, h->material_id_count,
		    1, size)
	    || !range_valid(h->buffers_offset, h->buffer_count,
		    sizeof(struct TmnfDetectBuffer), size)
	    || !range_valid(h->collision_records_offset,
		    h->collision_record_count, sizeof(SHmsPhysicalCollision), size)
	    || !range_valid(h->merge_buffer_ids_offset,
		    h->merge_buffer_id_count, 4, size)) {
		fprintf(stderr, "invalid broadphase graph framing\n");
		exit(2);
	}
	view.header = h;
	view.groups = (const struct TmnfBroadphaseGroup *)(blob + h->groups_offset);
	view.corpus_ids = (const uint32_t *)(blob + h->corpus_ids_offset);
	view.corpora =
		(const struct TmnfBroadphaseCorpus *)(blob + h->corpora_offset);
	view.dynas = (const struct TmnfBroadphaseDyna *)(blob + h->dynas_offset);
	view.states =
		(const struct TmnfBroadphaseState *)(blob + h->states_offset);
	view.speeds =
		(const struct TmnfBroadphaseSpeed *)(blob + h->speeds_offset);
	view.speed_values = (const float *)(blob + h->speed_values_offset);
	view.devices =
		(const struct TmnfBroadphaseDevice *)(blob + h->devices_offset);
	view.tables =
		(const struct TmnfBroadphaseTable *)(blob + h->tables_offset);
	view.table_values = (const int32_t *)(blob + h->table_values_offset);
	view.zones = (const struct TmnfBroadphaseZone *)(blob + h->zones_offset);
	view.static_entries = (const struct TmnfBroadphaseStaticEntry *)(
		blob + h->static_entries_offset);
	view.trees = (const struct TmnfBroadphaseTree *)(blob + h->trees_offset);
	view.tree_child_ids =
		(const uint32_t *)(blob + h->tree_child_ids_offset);
	view.isos = (const struct TmnfBroadphaseIso *)(blob + h->isos_offset);
	view.plugs = (const struct TmnfBroadphasePlug *)(blob + h->plugs_offset);
	view.surfaces =
		(const struct TmnfDetectSurface *)(blob + h->surfaces_offset);
	view.vertices = (const GmVec3 *)(blob + h->vertices_offset);
	view.faces = (const GmSurfMeshFace *)(blob + h->faces_offset);
	view.nodes = (const GmSurfMeshNode *)(blob + h->nodes_offset);
	view.material_ids = blob + h->material_ids_offset;
	view.buffers =
		(const struct TmnfDetectBuffer *)(blob + h->buffers_offset);
	view.collision_records = (const SHmsPhysicalCollision *)(
		blob + h->collision_records_offset);
	view.merge_buffer_ids =
		(const uint32_t *)(blob + h->merge_buffer_ids_offset);
	return view;
}

static const struct TmnfBroadphaseCorpus *encoded_corpus(
	BroadphaseView view, uint32_t id)
{
	for (uint32_t i = 0; i < view.header->corpus_count; ++i) {
		if (view.corpora[i].id == id)
			return &view.corpora[i];
	}
	return NULL;
}

static const struct TmnfBroadphaseDyna *encoded_dyna(
	BroadphaseView view, uint32_t id)
{
	for (uint32_t i = 0; i < view.header->dyna_count; ++i) {
		if (view.dynas[i].id == id)
			return &view.dynas[i];
	}
	return NULL;
}

static const struct TmnfBroadphaseState *encoded_state(
	BroadphaseView view, uint32_t id)
{
	for (uint32_t i = 0; i < view.header->state_count; ++i) {
		if (view.states[i].id == id)
			return &view.states[i];
	}
	return NULL;
}

static uint32_t group_index_by_id(BroadphaseView view, uint32_t id)
{
	for (uint32_t i = 0; i < view.header->group_count; ++i) {
		if (view.groups[i].id == id)
			return i;
	}
	fprintf(stderr, "unknown broadphase group id %u\n", id);
	exit(2);
}

static uint32_t corpus_index_by_id(BroadphaseView view, uint32_t id)
{
	for (uint32_t i = 0; i < view.header->corpus_count; ++i) {
		if (view.corpora[i].id == id)
			return i;
	}
	fprintf(stderr, "unknown broadphase corpus id %u\n", id);
	exit(2);
}

static uint32_t dyna_index_by_id(BroadphaseView view, uint32_t id)
{
	for (uint32_t i = 0; i < view.header->dyna_count; ++i) {
		if (view.dynas[i].id == id)
			return i;
	}
	fprintf(stderr, "unknown broadphase dyna id %u\n", id);
	exit(2);
}

static uint32_t state_index_by_id(BroadphaseView view, uint32_t id)
{
	for (uint32_t i = 0; i < view.header->state_count; ++i) {
		if (view.states[i].id == id)
			return i;
	}
	fprintf(stderr, "unknown broadphase state id %u\n", id);
	exit(2);
}

static uint32_t speed_index_by_id(BroadphaseView view, uint32_t id)
{
	for (uint32_t i = 0; i < view.header->speed_count; ++i) {
		if (view.speeds[i].id == id)
			return i;
	}
	fprintf(stderr, "unknown broadphase speed id %u\n", id);
	exit(2);
}

static uint32_t table_index_by_id(BroadphaseView view, uint32_t id)
{
	for (uint32_t i = 0; i < view.header->table_count; ++i) {
		if (view.tables[i].id == id)
			return i;
	}
	fprintf(stderr, "unknown broadphase table id %u\n", id);
	exit(2);
}

static uint32_t surface_index_by_id(BroadphaseView view, uint32_t id)
{
	for (uint32_t i = 0; i < view.header->surface_count; ++i) {
		if (view.surfaces[i].id == id)
			return i;
	}
	fprintf(stderr, "unknown broadphase surface id %u\n", id);
	exit(2);
}

static uint32_t plug_index_by_id(BroadphaseView view, uint32_t id)
{
	for (uint32_t i = 0; i < view.header->plug_count; ++i) {
		if (view.plugs[i].id == id)
			return i;
	}
	fprintf(stderr, "unknown broadphase plug id %u\n", id);
	exit(2);
}

static uint32_t buffer_index_by_id(BroadphaseView view, uint32_t id)
{
	for (uint32_t i = 0; i < view.header->buffer_count; ++i) {
		if (view.buffers[i].id == id)
			return i;
	}
	fprintf(stderr, "unknown broadphase buffer id %u\n", id);
	exit(2);
}

static uint32_t tree_index_by_id(BroadphaseView view, uint32_t id)
{
	for (uint32_t i = 0; i < view.header->tree_count; ++i) {
		if (view.trees[i].id == id)
			return i;
	}
	fprintf(stderr, "unknown broadphase tree id %u\n", id);
	exit(2);
}

static uint32_t iso_index_by_id(BroadphaseView view, uint32_t id)
{
	for (uint32_t i = 0; i < view.header->iso_count; ++i) {
		if (view.isos[i].id == id)
			return i;
	}
	fprintf(stderr, "unknown broadphase iso id %u\n", id);
	exit(2);
}

static GmSurf *surface_geom(ReplaySurface *surface)
{
	switch (surface->base.type) {
	case GM_SURF_SPHERE:
		return &surface->sphere.base;
	case GM_SURF_ELLIPSOID:
		return &surface->ellipsoid.base;
	case GM_SURF_BOX:
		return &surface->box.base;
	case GM_SURF_MESH:
		return &surface->mesh.base;
	default:
		return &surface->base;
	}
}

static void replay_get_linear_speed(
	const CHmsCorpus *corpus, GmVec3 *speed)
{
	const CHmsDyna *dyna = (const CHmsDyna *)corpus->dyna;
	*speed = dyna->liveState->linVel;
}

static void initialize_graph(ReplayGraph *graph, BroadphaseView input)
{
	const struct TmnfBroadphaseTraceHeader *h = input.header;
	memset(graph, 0, sizeof(*graph));
	graph->input = input;
	CollisionRuntime_Init(&graph->runtime);
	graph->runtime.get_linear_speed = replay_get_linear_speed;
	graph->groups = allocate((size_t)h->group_count * sizeof(*graph->groups));
	graph->corpora = allocate((size_t)h->corpus_count * sizeof(*graph->corpora));
	graph->dynas = allocate((size_t)h->dyna_count * sizeof(*graph->dynas));
	graph->states = allocate((size_t)h->state_count * sizeof(*graph->states));
	graph->speeds = allocate((size_t)h->speed_count * sizeof(*graph->speeds));
	graph->tables = allocate((size_t)h->table_count * sizeof(*graph->tables));
	graph->surfaces =
		allocate((size_t)h->surface_count * sizeof(*graph->surfaces));
	graph->plugs = allocate((size_t)h->plug_count * sizeof(*graph->plugs));
	graph->plug_material_ids = allocate(
		(size_t)h->plug_count * sizeof(*graph->plug_material_ids));
	graph->buffers =
		allocate((size_t)h->buffer_count * sizeof(*graph->buffers));
	graph->trees = allocate((size_t)h->tree_count * sizeof(*graph->trees));
	graph->isos = allocate((size_t)h->iso_count * sizeof(*graph->isos));

	for (uint32_t i = 0; i < h->state_count; ++i)
		memcpy(&graph->states[i], input.states[i].bytes,
			sizeof(graph->states[i]));

	for (uint32_t i = 0; i < h->dyna_count; ++i) {
		const struct TmnfBroadphaseDyna *dyna = &input.dynas[i];
		if (dyna->state_id == 0 || encoded_state(input, dyna->state_id) == NULL)
			fail("dyna has invalid state id", "broadphase graph");
		graph->dynas[i].liveState =
			&graph->states[state_index_by_id(input, dyna->state_id)];
	}

	for (uint32_t i = 0; i < h->corpus_count; ++i) {
		const struct TmnfBroadphaseCorpus *corpus = &input.corpora[i];
		graph->corpora[i].object_ref = corpus->id;
		graph->corpora[i].flags = corpus->flags;
		graph->corpora[i].group_index = raw_u32(corpus->bytes, 0x54);
		memcpy(&graph->corpora[i].local_iso, corpus->bytes + 0x18,
			sizeof(graph->corpora[i].local_iso));
		if (corpus->dyna_id != 0) {
			if (encoded_dyna(input, corpus->dyna_id) == NULL)
				fail("corpus has invalid dyna id", "broadphase graph");
			uint32_t dyna_index =
				dyna_index_by_id(input, corpus->dyna_id);
			graph->corpora[i].dyna = &graph->dynas[dyna_index];
			graph->corpora[i].live_iso =
				(const GmIso4 *)&graph->dynas[dyna_index]
					.liveState->rot;
		}
	}

	for (uint32_t i = 0; i < h->speed_count; ++i) {
		const struct TmnfBroadphaseSpeed *speed = &input.speeds[i];
		if (speed->count > speed->capacity
		    || (uint64_t)speed->value_index + speed->count
			    > h->speed_value_count)
			fail("invalid speed slice", "broadphase graph");
		graph->speeds[i].id = speed->id;
		graph->speeds[i].count = speed->count;
		graph->speeds[i].capacity = speed->capacity;
		graph->speeds[i].values =
			allocate((size_t)speed->capacity * sizeof(float));
		memcpy(graph->speeds[i].values,
			&input.speed_values[speed->value_index],
			(size_t)speed->count * sizeof(float));
	}

	for (uint32_t i = 0; i < h->table_count; ++i) {
		const struct TmnfBroadphaseTable *table = &input.tables[i];
		uint64_t storage = (uint64_t)table->rows * table->stride;
		if (table->columns > table->stride
		    || storage != table->value_count
		    || (uint64_t)table->value_index + table->value_count
			    > h->table_value_count)
			fail("invalid table slice", "broadphase graph");
		graph->tables[i].id = table->id;
		graph->tables[i].rows = table->rows;
		graph->tables[i].columns = table->columns;
		graph->tables[i].reserved = table->reserved;
		graph->tables[i].stride = table->stride;
		graph->tables[i].values =
			allocate((size_t)table->value_count * sizeof(int32_t));
		memcpy(graph->tables[i].values,
			&input.table_values[table->value_index],
			(size_t)table->value_count * sizeof(int32_t));
	}

	for (uint32_t i = 0; i < h->surface_count; ++i) {
		const struct TmnfDetectSurface *encoded = &input.surfaces[i];
		ReplaySurface *surface = &graph->surfaces[i];
		surface->base.material_index = encoded->material_index;
		surface->base.type = encoded->type;
		surface->base.reserved = encoded->reserved;
		if (encoded->type == GM_SURF_SPHERE) {
			surface->sphere.base = surface->base;
			memcpy(&surface->sphere.radius, encoded->shape, 4);
		} else if (encoded->type == GM_SURF_ELLIPSOID) {
			surface->ellipsoid.base = surface->base;
			memcpy(&surface->ellipsoid.radii, encoded->shape, 0x0c);
		} else if (encoded->type == GM_SURF_BOX) {
			surface->box.base = surface->base;
			memcpy(&surface->box.center, encoded->shape, 0x18);
		} else if (encoded->type == GM_SURF_MESH) {
			if ((uint64_t)encoded->vertex_index + encoded->vertex_count
				    > h->vertex_count
			    || (uint64_t)encoded->face_index + encoded->face_count
				    > h->face_count
			    || (uint64_t)encoded->node_index + encoded->node_count
				    > h->node_count)
				fail("invalid mesh slice", "broadphase graph");
			surface->mesh.base = surface->base;
			surface->mesh.vertex_count = encoded->vertex_count;
			surface->vertices = allocate(
				(size_t)encoded->vertex_count * sizeof(GmVec3));
			memcpy(surface->vertices,
				&input.vertices[encoded->vertex_index],
				(size_t)encoded->vertex_count * sizeof(GmVec3));
			surface->mesh.vertices = surface->vertices;
			surface->mesh.face_count = encoded->face_count;
			surface->faces = allocate(
				(size_t)encoded->face_count * sizeof(GmSurfMeshFace));
			memcpy(surface->faces, &input.faces[encoded->face_index],
				(size_t)encoded->face_count * sizeof(GmSurfMeshFace));
			surface->mesh.faces = surface->faces;
			surface->mesh.node_count = encoded->node_count;
			surface->nodes = allocate(
				(size_t)encoded->node_count * sizeof(GmSurfMeshNode));
			memcpy(surface->nodes, &input.nodes[encoded->node_index],
				(size_t)encoded->node_count * sizeof(GmSurfMeshNode));
			surface->mesh.nodes = surface->nodes;
		}
	}
	for (uint32_t i = 0; i < h->plug_count; ++i) {
		const struct TmnfBroadphasePlug *encoded = &input.plugs[i];
		if ((uint64_t)encoded->material_index + encoded->material_count
		    > h->material_id_count)
			fail("invalid plug material slice", "broadphase graph");
		graph->plugs[i].geom = surface_geom(
			&graph->surfaces[
				surface_index_by_id(input, encoded->surface_id)]);
		graph->plug_material_ids[i] = allocate(encoded->material_count);
		memcpy(graph->plug_material_ids[i],
			&input.material_ids[encoded->material_index],
			encoded->material_count);
		graph->plugs[i].material_ids = graph->plug_material_ids[i];
		graph->plugs[i].material_count = encoded->material_count;
	}
	for (uint32_t i = 0; i < h->buffer_count; ++i) {
		const struct TmnfDetectBuffer *encoded = &input.buffers[i];
		ReplayBuffer *buffer = &graph->buffers[i];
		if (encoded->count > encoded->capacity
		    || (uint64_t)encoded->record_index + encoded->count
			    > h->collision_record_count)
			fail("invalid collision buffer slice", "broadphase graph");
		buffer->id = encoded->id;
		buffer->sphere.base.collisions.count = encoded->count;
		buffer->sphere.base.collisions.capacity = encoded->capacity;
		buffer->sphere.base.collisions.data = allocate(
			(size_t)encoded->capacity * sizeof(SHmsPhysicalCollision));
		memcpy(buffer->sphere.base.collisions.data,
			&input.collision_records[encoded->record_index],
			(size_t)encoded->count * sizeof(SHmsPhysicalCollision));
		buffer->sphere.active = encoded->active;
	}
	for (uint32_t i = 0; i < h->iso_count; ++i)
		memcpy(&graph->isos[i], input.isos[i].bytes, sizeof(GmIso4));
	for (uint32_t i = 0; i < h->tree_count; ++i) {
		const struct TmnfBroadphaseTree *encoded = &input.trees[i];
		ReplayTree *tree = &graph->trees[i];
		tree->value.object_ref = encoded->id;
		tree->value.flags = encoded->flags;
		memcpy(&tree->value.box, encoded->box, sizeof(tree->value.box));
		memcpy(&tree->value.local_iso, encoded->local_iso,
			sizeof(tree->value.local_iso));
		if (encoded->surface_id != 0)
			tree->value.surface = &graph->plugs[
				plug_index_by_id(input, encoded->surface_id)];
		if (encoded->buffer_id != 0)
			tree->value.contact_buffer = &graph->buffers[
				buffer_index_by_id(input, encoded->buffer_id)].sphere;
		if ((uint64_t)encoded->child_index + encoded->child_count
		    > h->tree_child_id_count)
			fail("invalid tree child slice", "broadphase graph");
		tree->value.child_count = encoded->child_count;
		tree->children = allocate(
			(size_t)encoded->child_count * sizeof(*tree->children));
		tree->value.children = tree->children;
	}
	for (uint32_t i = 0; i < h->corpus_count; ++i) {
		if (input.corpora[i].tree_id != 0)
			graph->corpora[i].tree = &graph->trees[
				tree_index_by_id(input, input.corpora[i].tree_id)]
					.value;
	}
	for (uint32_t i = 0; i < h->tree_count; ++i) {
		const struct TmnfBroadphaseTree *encoded = &input.trees[i];
		for (uint32_t j = 0; j < encoded->child_count; ++j) {
			uint32_t id =
				input.tree_child_ids[encoded->child_index + j];
			graph->trees[i].children[j] = &graph->trees[
				tree_index_by_id(input, id)].value;
		}
	}

	for (uint32_t i = 0; i < h->group_count; ++i) {
		const struct TmnfBroadphaseGroup *encoded = &input.groups[i];
		ReplayGroup *group = &graph->groups[i];
		group->value.runtime = &graph->runtime;
		group->value.priority = raw_u32(encoded->bytes, 0x3c);
		group->value.is_static = raw_u32(encoded->bytes, 0x40);
		if (encoded->speed_id != 0) {
			uint32_t speed_index =
				speed_index_by_id(input, encoded->speed_id);
			group->value.speed_sq = graph->speeds[speed_index].values;
		}
		if ((uint64_t)encoded->corpus_index + encoded->corpus_count
		    > h->corpus_id_count)
			fail("invalid group corpus slice", "broadphase graph");
		group->value.corpus_count = encoded->corpus_count;
		group->corpora = allocate(
			(size_t)encoded->corpus_count * sizeof(*group->corpora));
		for (uint32_t j = 0; j < encoded->corpus_count; ++j) {
			uint32_t id =
				input.corpus_ids[encoded->corpus_index + j];
			if (encoded_corpus(input, id) == NULL)
				fail("invalid group corpus id", "broadphase graph");
			group->corpora[j] =
				&graph->corpora[corpus_index_by_id(input, id)];
		}
		group->value.corpora = group->corpora;
		if ((uint64_t)encoded->device_index + encoded->device_count
		    > h->device_count)
			fail("invalid group device slice", "broadphase graph");
		group->value.device_mat_count = encoded->device_count;
		group->devices = allocate(
			(size_t)encoded->device_count * sizeof(*group->devices));
		group->value.device_mats = group->devices;
		if ((uint64_t)encoded->static_entry_index
			    + encoded->static_entry_count
		    > h->static_entry_count)
			fail("invalid static entry slice", "broadphase graph");
		group->value.static_entry_count = encoded->static_entry_count;
		group->static_entries = allocate(
			(size_t)encoded->static_entry_count
				* sizeof(*group->static_entries));
		group->value.static_entries = group->static_entries;
		for (uint32_t j = 0; j < encoded->static_entry_count; ++j) {
			const struct TmnfBroadphaseStaticEntry *entry =
				&input.static_entries[
					encoded->static_entry_index + j];
			HmsStaticCollisionEntry *actual =
				&group->static_entries[j];
			actual->skip_count = entry->skip_count;
			memcpy(&actual->box, entry->box, sizeof(actual->box));
			memcpy(&actual->iso, entry->iso, sizeof(actual->iso));
			actual->tree_flags = entry->tree_flags;
			if (entry->surface_id != 0)
				actual->surface = &graph->plugs[
					plug_index_by_id(
						input, entry->surface_id)];
			if (entry->tree_id != 0) {
				actual->tree_ref = entry->tree_id;
			}
			actual->corpus_ref = entry->corpus_id;
		}
	}

	for (uint32_t i = 0; i < h->group_count; ++i) {
		const struct TmnfBroadphaseGroup *encoded = &input.groups[i];
		ReplayGroup *group = &graph->groups[i];
		for (uint32_t j = 0; j < encoded->device_count; ++j) {
			const struct TmnfBroadphaseDevice *device =
				&input.devices[encoded->device_index + j];
			uint32_t table_index =
				table_index_by_id(input, device->table_id);
			ReplayTable *table = &graph->tables[table_index];
			group->devices[j].group =
				&graph->groups[
					group_index_by_id(input, device->group_id)].value;
			group->devices[j].material_ref =
				raw_u32(device->bytes, 4);
			group->devices[j].perform.data = table->values;
			group->devices[j].perform.rows = table->rows;
			group->devices[j].perform.columns = table->columns;
			group->devices[j].perform.reserved = table->reserved;
			group->devices[j].perform.stride = table->stride;
		}
	}
	for (uint32_t i = 0; i < h->group_count; ++i) {
		if (input.groups[i].zone_index < 5)
			graph->zone.groups[input.groups[i].zone_index] =
				graph->groups[i].value;
	}
	if (h->zone_count != 0) {
		const struct TmnfBroadphaseZone *zone = &input.zones[0];
		graph->zone.runtime = &graph->runtime;
		graph->zone.current_material = zone->current_material_id;
		graph->zone.current_corpus1 = zone->current_corpus1_id;
		graph->zone.current_corpus2 = zone->current_corpus2_id;
		if (zone->general_buffer_id != 0)
			graph->zone.general_buffer = &graph->buffers[
				buffer_index_by_id(input, zone->general_buffer_id)]
					.sphere.base;
		if (zone->static_group_id != 0)
			graph->zone.static_group = &graph->groups[
				group_index_by_id(input, zone->static_group_id)]
					.value;
		if ((uint64_t)zone->merge_buffer_index
			    + zone->merge_buffer_count
		    > h->merge_buffer_id_count)
			fail("invalid merge buffer slice", "broadphase graph");
		graph->zone.merge_buffer_count = zone->merge_buffer_count;
		graph->zone.merge_buffer_capacity = zone->merge_buffer_capacity;
		graph->zone.merge_buffers = allocate(
			(size_t)zone->merge_buffer_capacity
				* sizeof(*graph->zone.merge_buffers));
		for (uint32_t i = 0; i < zone->merge_buffer_count; ++i) {
			uint32_t id;
			memcpy(&id, (const uint8_t *)input.merge_buffer_ids +
				4 * (zone->merge_buffer_index + i), sizeof(id));
			graph->zone.merge_buffers[i] = &graph->buffers[
				buffer_index_by_id(input, id)].sphere;
		}
	}
}

static void destroy_graph(ReplayGraph *graph)
{
	for (uint32_t i = 0; i < graph->input.header->tree_count; ++i) {
		SHmsSphereBufferContact *contact =
			graph->trees[i].value.contact_buffer;
		int captured = 0;
		for (uint32_t j = 0;
		     j < graph->input.header->buffer_count; ++j) {
			if (contact == &graph->buffers[j].sphere) {
				captured = 1;
				break;
			}
		}
		if (contact != NULL && !captured) {
			CHmsCollisionBuffer_Destroy(&contact->base);
			free(contact);
		}
		free(graph->trees[i].children);
	}
	for (uint32_t i = 0; i < graph->input.header->group_count; ++i) {
		free(graph->groups[i].corpora);
		free(graph->groups[i].devices);
		free(graph->groups[i].static_entries);
	}
	for (uint32_t i = 0; i < graph->input.header->speed_count; ++i)
		free(graph->speeds[i].values);
	for (uint32_t i = 0; i < graph->input.header->table_count; ++i)
		free(graph->tables[i].values);
	for (uint32_t i = 0; i < graph->input.header->surface_count; ++i) {
		free(graph->surfaces[i].vertices);
		free(graph->surfaces[i].faces);
		free(graph->surfaces[i].nodes);
	}
	for (uint32_t i = 0; i < graph->input.header->plug_count; ++i)
		free(graph->plug_material_ids[i]);
	for (uint32_t i = 0; i < graph->input.header->buffer_count; ++i)
		free(graph->buffers[i].sphere.base.collisions.data);
	free(graph->zone.merge_buffers);
	free(graph->groups);
	free(graph->corpora);
	free(graph->dynas);
	free(graph->states);
	free(graph->speeds);
	free(graph->tables);
	free(graph->surfaces);
	free(graph->plugs);
	free(graph->plug_material_ids);
	free(graph->buffers);
	free(graph->trees);
	free(graph->isos);
}

static SHmsSphereBufferContact *actual_buffer_for_id(
	ReplayGraph *graph, BroadphaseView expected, uint32_t id)
{
	for (uint32_t i = 0; i < graph->input.header->buffer_count; ++i) {
		if (graph->input.buffers[i].id == id)
			return &graph->buffers[i].sphere;
	}
	for (uint32_t i = 0; i < expected.header->tree_count; ++i) {
		if (expected.trees[i].buffer_id == id) {
			uint32_t tree_index = tree_index_by_id(
				graph->input, expected.trees[i].id);
			return graph->trees[tree_index].value.contact_buffer;
		}
	}
	return NULL;
}

static uint32_t id_for_actual_buffer(
	ReplayGraph *graph, BroadphaseView expected,
	SHmsSphereBufferContact *buffer)
{
	for (uint32_t i = 0; i < graph->input.header->buffer_count; ++i) {
		if (buffer == &graph->buffers[i].sphere)
			return graph->input.buffers[i].id;
	}
	for (uint32_t i = 0; i < expected.header->tree_count; ++i) {
		uint32_t tree_index = tree_index_by_id(
			graph->input, expected.trees[i].id);
		if (buffer == graph->trees[tree_index].value.contact_buffer)
			return expected.trees[i].buffer_id;
	}
	return 0;
}

static int compare_collision_buffer(
	const SHmsSphereBufferContact *actual,
	const struct TmnfDetectBuffer *expected,
	const SHmsPhysicalCollision *records)
{
	if (actual == NULL
	    || actual->base.collisions.count != expected->count
	    || actual->base.collisions.capacity != expected->capacity
	    || (expected->has_active
		&& actual->active != expected->active))
		return 0;
	if (memcmp(actual->base.collisions.data,
		    &records[expected->record_index],
		    (size_t)expected->count * sizeof(SHmsPhysicalCollision)) != 0) {
		if (diagnose) {
			const uint8_t *actual_bytes =
				(const uint8_t *)actual->base.collisions.data;
			const uint8_t *expected_bytes =
				(const uint8_t *)&records[expected->record_index];
			size_t size =
				(size_t)expected->count
				* sizeof(SHmsPhysicalCollision);
			size_t offset = 0;
			while (offset < size
			       && actual_bytes[offset] == expected_bytes[offset])
				++offset;
			if (offset < size)
				fprintf(stderr,
					"collision byte=%zu actual=%02x expected=%02x\n",
					offset, actual_bytes[offset],
					expected_bytes[offset]);
		}
		return 0;
	}
	return 1;
}

static int compare_graph(
	ReplayGraph *graph, BroadphaseView expected, uint32_t return_value)
{
	if (graph->input.header->kind == TMNF_BROADPHASE_COMPUTE_COLLISION
	    && return_value != expected.header->return_value) {
		if (diagnose)
			fprintf(stderr, "return actual=%u expected=%u\n",
				return_value, expected.header->return_value);
		return 0;
	}
	if (graph->input.header->kind != expected.header->kind
	    || graph->input.header->root_id != expected.header->root_id
	    || graph->input.header->speed_count != expected.header->speed_count
	    || graph->input.header->table_count != expected.header->table_count)
		fail("broadphase output graph shape changed", "broadphase graph");
	for (uint32_t i = 0; i < expected.header->speed_count; ++i) {
		const struct TmnfBroadphaseSpeed *speed = &expected.speeds[i];
		uint32_t actual_index =
			speed_index_by_id(graph->input, speed->id);
		ReplaySpeed *actual = &graph->speeds[actual_index];
		if (actual->count != speed->count
		    || actual->capacity != speed->capacity
		    || (uint64_t)speed->value_index + speed->count
			    > expected.header->speed_value_count)
			return 0;
		if (memcmp(actual->values,
			    &expected.speed_values[speed->value_index],
			    (size_t)speed->count * sizeof(float)) != 0) {
			if (diagnose) {
				for (uint32_t j = 0; j < speed->count; ++j) {
					uint32_t actual_bits;
					uint32_t expected_bits;
					memcpy(&actual_bits, &actual->values[j], 4);
					memcpy(&expected_bits,
						&expected.speed_values[
							speed->value_index + j], 4);
					if (actual_bits != expected_bits) {
						fprintf(stderr,
							"speed[%u] actual=%08x expected=%08x\n",
							j, actual_bits, expected_bits);
						break;
					}
				}
			}
			return 0;
		}
	}
	for (uint32_t i = 0; i < expected.header->table_count; ++i) {
		const struct TmnfBroadphaseTable *table = &expected.tables[i];
		uint32_t actual_index =
			table_index_by_id(graph->input, table->id);
		ReplayTable *actual = &graph->tables[actual_index];
		if (actual->rows != table->rows
		    || actual->columns != table->columns
		    || actual->reserved != table->reserved
		    || actual->stride != table->stride
		    || (uint64_t)table->value_index + table->value_count
			    > expected.header->table_value_count)
			return 0;
		if (memcmp(actual->values,
			    &expected.table_values[table->value_index],
			    (size_t)table->value_count * sizeof(int32_t)) != 0) {
			if (diagnose) {
				for (uint32_t j = 0; j < table->value_count; ++j) {
					int32_t expected_value =
						expected.table_values[
							table->value_index + j];
					if (actual->values[j] != expected_value) {
						fprintf(stderr,
							"table[%u] actual=%d expected=%d\n",
							j, actual->values[j], expected_value);
						break;
					}
				}
			}
			return 0;
		}
	}
	if (expected.header->tree_count != graph->input.header->tree_count)
		return 0;
	for (uint32_t i = 0; i < expected.header->tree_count; ++i) {
		const struct TmnfBroadphaseTree *tree = &expected.trees[i];
		uint32_t tree_index =
			tree_index_by_id(graph->input, tree->id);
		SHmsSphereBufferContact *actual =
			graph->trees[tree_index].value.contact_buffer;
		if (tree->buffer_id == 0) {
			if (actual != NULL)
				return 0;
		} else if (actual_buffer_for_id(
			    graph, expected, tree->buffer_id) != actual) {
			return 0;
		}
	}
	if (expected.header->buffer_count != 0) {
		for (uint32_t i = 0; i < expected.header->buffer_count; ++i) {
			const struct TmnfDetectBuffer *buffer = &expected.buffers[i];
			if ((uint64_t)buffer->record_index + buffer->count
			    > expected.header->collision_record_count)
				fail("invalid expected collision slice",
					"broadphase graph");
			if (!compare_collision_buffer(
				    actual_buffer_for_id(
					    graph, expected, buffer->id),
				    buffer, expected.collision_records))
				return 0;
		}
	}
	if (expected.header->zone_count != 0) {
		const struct TmnfBroadphaseZone *zone = &expected.zones[0];
		uint32_t general_id = id_for_actual_buffer(
			graph, expected,
			(SHmsSphereBufferContact *)graph->zone.general_buffer);
		uint32_t static_group_id = 0;
		for (uint32_t i = 0; i < graph->input.header->group_count; ++i) {
			if (graph->zone.static_group == &graph->groups[i].value) {
				static_group_id = graph->input.groups[i].id;
				break;
			}
		}
		if (graph->zone.current_material != zone->current_material_id
		    || graph->zone.current_corpus1 != zone->current_corpus1_id
		    || graph->zone.current_corpus2 != zone->current_corpus2_id
		    || general_id != zone->general_buffer_id
		    || static_group_id != zone->static_group_id
		    || graph->zone.merge_buffer_count != zone->merge_buffer_count
		    || graph->zone.merge_buffer_capacity
			    != zone->merge_buffer_capacity
		    || (uint64_t)zone->merge_buffer_index
				    + zone->merge_buffer_count
			    > expected.header->merge_buffer_id_count)
			return 0;
		for (uint32_t i = 0; i < zone->merge_buffer_count; ++i) {
			uint32_t actual_id = id_for_actual_buffer(
				graph, expected, graph->zone.merge_buffers[i]);
			uint32_t expected_id;
			memcpy(&expected_id, (const uint8_t *)expected.merge_buffer_ids +
				4 * (zone->merge_buffer_index + i), sizeof(expected_id));
			if (actual_id != expected_id)
				return 0;
		}
	}
	return 1;
}

static int replay_graph(BroadphaseView input, BroadphaseView expected)
{
	ReplayGraph graph;
	int result;
	uint32_t return_value = 0;
	initialize_graph(&graph, input);
	switch (input.header->kind) {
	case TMNF_BROADPHASE_GROUP_SPEEDS: {
		uint32_t root_index =
			group_index_by_id(input, input.header->root_id);
		CHmsCollisionManager_SGroup_ComputeNonStaticCorpusInfos(
			&graph.groups[root_index].value);
		break;
	}
	case TMNF_BROADPHASE_GROUP_PERFORM: {
		uint32_t root_index =
			group_index_by_id(input, input.header->root_id);
		CHmsCollisionManager_SGroup_ComputeIsToPerformCollisions(
			&graph.groups[root_index].value);
		break;
	}
	case TMNF_BROADPHASE_TREE_STATIC: {
		uint32_t iso_index =
			iso_index_by_id(input, input.header->argument_ids[0]);
		uint32_t tree_index =
			tree_index_by_id(input, input.header->argument_ids[1]);
		CHmsCollisionManager_SZone_DetectCollisionBetweenTreeAndStaticCollisionTree(
			&graph.zone, &graph.isos[iso_index],
			&graph.trees[tree_index].value, 1);
		break;
	}
	case TMNF_BROADPHASE_PREPARE:
		CHmsCollisionManager_SZone_PrepareCollisions(&graph.zone);
		break;
	case TMNF_BROADPHASE_COMPUTE_COLLISION: {
		SPlugTreeLocatedPair pair = {
			&graph.trees[
				tree_index_by_id(input, input.header->argument_ids[0])]
				.value,
			&graph.isos[
				iso_index_by_id(input, input.header->argument_ids[1])],
			&graph.trees[
				tree_index_by_id(input, input.header->argument_ids[2])]
				.value,
			&graph.isos[
				iso_index_by_id(input, input.header->argument_ids[3])],
		};
		return_value = (uint32_t)
			CHmsCollisionManager_SZone_ComputeCollision(
				&graph.zone, &pair);
		break;
	}
	case TMNF_BROADPHASE_DETECT_BETWEEN:
		CHmsCollisionManager_SZone_DetectCollisionBetween(
			&graph.zone,
			&graph.corpora[corpus_index_by_id(
				input, input.header->argument_ids[0])],
			&graph.corpora[corpus_index_by_id(
				input, input.header->argument_ids[1])]);
		break;
	case TMNF_BROADPHASE_DETECT_CORPUS: {
		uint32_t buffer_index =
			buffer_index_by_id(input, input.header->argument_ids[0]);
		CHmsCollisionManager_SZone_DetectCollisionsCorpus(
			&graph.zone, &graph.buffers[buffer_index].sphere.base,
			&graph.corpora[corpus_index_by_id(
				input, input.header->argument_ids[1])], 1);
		break;
	}
	default:
		fprintf(stderr, "unsupported broadphase graph kind %u\n",
			input.header->kind);
		exit(2);
	}
	result = compare_graph(&graph, expected, return_value);
	destroy_graph(&graph);
	return result;
}

static uint32_t read_header(
	FILE *file, const Target *target, const char *path)
{
	char magic[8];
	uint32_t record_count;
	read_exact(file, magic, sizeof(magic), path);
	if (memcmp(magic, TMNF_TRACE_MAGIC, sizeof(magic)) != 0)
		fail("bad trace magic", path);
	if (read_u32(file, path) != target->va)
		fail("trace VA mismatch", path);
	record_count = read_u32(file, path);
	if (record_count == 0)
		fail("trace has no records", path);
	return record_count;
}

static void read_physics_buffer(
	FILE *file, const char *path, uint32_t expected_tag,
	uint32_t *address, uint8_t bytes[PHYSICS_THIS_SIZE])
{
	uint32_t tag = read_u32(file, path);
	*address = read_u32(file, path);
	uint32_t size = read_u32(file, path);
	if (tag != expected_tag || size != PHYSICS_THIS_SIZE)
		fail("PhysicsStep2 trace shape changed", path);
	read_exact(file, bytes, PHYSICS_THIS_SIZE, path);
}

static uint32_t inspect_physics(
	FILE *file, const char *path, uint32_t record_count)
{
	uint32_t changed_records = 0;
	uint8_t changed_offsets[PHYSICS_THIS_SIZE] = { 0 };
	for (uint32_t record = 0; record < record_count; ++record) {
		uint8_t input[PHYSICS_THIS_SIZE];
		uint8_t output[PHYSICS_THIS_SIZE];
		uint32_t input_address;
		uint32_t output_address;
		(void)read_u32(file, path);
		if (read_u16(file, path) != 1 || read_u16(file, path) != 1)
			fail("PhysicsStep2 record buffer count changed", path);
		read_physics_buffer(
			file, path, TAG_THIS, &input_address, input);
		read_physics_buffer(
			file, path, TAG_OUT_THIS, &output_address, output);
		if (input_address != output_address)
			fail("PhysicsStep2 THIS identity changed", path);
		int record_changed = 0;
		for (uint32_t offset = 0; offset < PHYSICS_THIS_SIZE; ++offset) {
			if (input[offset] != output[offset]) {
				changed_offsets[offset] = 1;
				record_changed = 1;
			}
		}
		changed_records += (uint32_t)record_changed;
	}
	for (uint32_t offset = 0; offset < PHYSICS_THIS_SIZE; ++offset) {
		if (changed_offsets[offset] != 0
		    && offset != PHYSICS_COLLISION_COUNT_OFFSET)
			fail("unexpected PhysicsStep2 root mutation", path);
	}
	return changed_records;
}

static uint32_t replay_broadphase(
	FILE *file, const char *path, uint32_t record_count)
{
	uint32_t passed = 0;
	diagnose = 1;
	for (uint32_t record = 0; record < record_count; ++record) {
		uint32_t input_tag;
		uint32_t input_address;
		uint32_t input_size;
		uint32_t output_tag;
		uint32_t output_address;
		uint32_t output_size;
		uint8_t *input_blob;
		uint8_t *output_blob;
		(void)read_u32(file, path);
		if (read_u16(file, path) != 1 || read_u16(file, path) != 1)
			fail("invalid broadphase record shape", path);
		input_tag = read_u32(file, path);
		input_address = read_u32(file, path);
		input_size = read_u32(file, path);
		input_blob = read_blob(file, input_size, path);
		output_tag = read_u32(file, path);
		output_address = read_u32(file, path);
		output_size = read_u32(file, path);
		output_blob = read_blob(file, output_size, path);
		if (input_tag != TAG_THIS || output_tag != TAG_OUT_THIS
		    || input_address != 0 || output_address != 0)
			fail("invalid broadphase graph buffer", path);
		BroadphaseView input = broadphase_view(input_blob, input_size);
		BroadphaseView output = broadphase_view(output_blob, output_size);
		int ok = replay_graph(input, output);
		free(input_blob);
		free(output_blob);
		if (ok) {
			++passed;
		} else if (diagnose) {
			fprintf(stderr, "%s record %u first failure\n", path, record);
			diagnose = 0;
		}
	}
	return passed;
}

static void inspect_empty(
	FILE *file, const char *path, uint32_t record_count)
{
	for (uint32_t record = 0; record < record_count; ++record) {
		(void)read_u32(file, path);
		if (read_u16(file, path) != 0 || read_u16(file, path) != 0)
			fail("broad-phase trace shape changed", path);
	}
}

int main(int argc, char **argv)
{
	uint32_t total_records = 0;
	uint32_t replayed_records = 0;
	uint32_t passed_records = 0;
	uint32_t unverifiable = 0;
	int success = 1;
	if (argc != 2) {
		fprintf(stderr, "usage: %s TRACE_DIRECTORY\n", argv[0]);
		return 2;
	}

	for (size_t i = 0; i < sizeof(targets) / sizeof(targets[0]); ++i) {
		const Target *target = &targets[i];
		char path[1024];
		int length = snprintf(
			path, sizeof(path), "%s/%s", argv[1], target->file_name);
		if (length < 0 || (size_t)length >= sizeof(path))
			fail("trace path too long", argv[1]);
		FILE *file = fopen(path, "rb");
		if (file == NULL)
			fail("cannot open trace", path);
		uint32_t record_count = read_header(file, target, path);
		total_records += record_count;
		if (target->broadphase) {
			uint32_t passed =
				replay_broadphase(file, path, record_count);
			replayed_records += record_count;
			passed_records += passed;
			printf("%08X %s: pass=%u/%u replayable=%u/%u\n",
				target->va, target->name, passed, record_count,
				record_count, record_count);
			if (passed != record_count)
				success = 0;
		} else if (target->physics_window) {
			uint32_t changed =
				inspect_physics(file, path, record_count);
			printf(
				"%08X %s: pass=N/A replayable=0/%u; "
				"only THIS[384] -> OUT_THIS[384], "
				"this+0x15C changed in %u records\n",
				target->va, target->name, record_count, changed);
			++unverifiable;
		} else {
			inspect_empty(file, path, record_count);
			printf(
				"%08X %s: pass=N/A replayable=0/%u; "
				"every record has 0 inputs and 0 outputs\n",
				target->va, target->name, record_count);
			++unverifiable;
		}
		if (fgetc(file) != EOF)
			fail("trailing trace bytes", path);
		fclose(file);
	}
	printf(
		"summary: pass=%u/%u replayable=%u/%u; "
		"%u/%zu functions unverifiable\n",
		passed_records, replayed_records,
		replayed_records, total_records, unverifiable,
		sizeof(targets) / sizeof(targets[0]));
	return success ? 0 : 1;
}
