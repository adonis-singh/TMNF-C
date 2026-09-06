#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "collision.h"
#include "fastbuffer.h"
#include "../oracle/tracer/detect_trace.h"

typedef struct {
	const struct TmnfDetectTraceHeader *header;
	const struct TmnfDetectLocated *located;
	const struct TmnfDetectSurface *surfaces;
	const GmIso4 *isos;
	const struct TmnfDetectPlugSurface *plugs;
	const struct TmnfDetectBuffer *buffers;
	const GmVec3 *vertices;
	const GmSurfMeshFace *faces;
	const GmSurfMeshNode *nodes;
	const uint8_t *material_ids;
	const SHmsPhysicalCollision *records;
	const GmBoxAligned *boxes;
} DetectView;

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
	SHmsSphereBufferContact sphere;
} ReplayBuffer;

typedef struct {
	DetectView input;
	ReplaySurface *surfaces;
	GmIso4 *isos;
	LocatedGmSurf *located;
	CPlugSurface *plugs;
	uint8_t **material_ids;
	ReplayBuffer *buffers;
} ReplayGraph;

static int diagnose;

static void die(const char *message, const char *path)
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

static uint32_t read_u32(FILE *file, const char *path)
{
	uint8_t bytes[4];
	if (fread(bytes, 1, sizeof(bytes), file) != sizeof(bytes))
		die("unexpected EOF", path);
	return (uint32_t)bytes[0]
		| (uint32_t)bytes[1] << 8
		| (uint32_t)bytes[2] << 16
		| (uint32_t)bytes[3] << 24;
}

static uint16_t read_u16(FILE *file, const char *path)
{
	uint8_t bytes[2];
	if (fread(bytes, 1, sizeof(bytes), file) != sizeof(bytes))
		die("unexpected EOF", path);
	return (uint16_t)(bytes[0] | (uint16_t)bytes[1] << 8);
}

static uint8_t *read_blob(FILE *file, uint32_t size, const char *path)
{
	uint8_t *blob = allocate(size);
	if (fread(blob, 1, size, file) != size)
		die("unexpected EOF", path);
	return blob;
}

static int range_valid(
	uint32_t offset, uint32_t count, uint32_t stride, uint32_t size)
{
	uint64_t end = (uint64_t)offset + (uint64_t)count * stride;
	return offset >= sizeof(struct TmnfDetectTraceHeader)
		&& offset <= size && end <= size;
}

static DetectView detect_view(const uint8_t *blob, uint32_t size)
{
	DetectView view;
	memset(&view, 0, sizeof(view));
	if (size < sizeof(struct TmnfDetectTraceHeader)) {
		fprintf(stderr, "truncated detection graph\n");
		exit(2);
	}
	view.header = (const struct TmnfDetectTraceHeader *)blob;
	const struct TmnfDetectTraceHeader *h = view.header;
	if (memcmp(h->magic, TMNF_DETECT_TRACE_MAGIC, 8) != 0
	    || h->version != TMNF_DETECT_TRACE_VERSION
	    || h->total_size != size
	    || !range_valid(h->located_offset, h->located_count,
		    sizeof(struct TmnfDetectLocated), size)
	    || !range_valid(h->surfaces_offset, h->surface_count,
		    sizeof(struct TmnfDetectSurface), size)
	    || !range_valid(h->isos_offset, h->iso_count, sizeof(GmIso4), size)
	    || !range_valid(h->plugs_offset, h->plug_count,
		    sizeof(struct TmnfDetectPlugSurface), size)
	    || !range_valid(h->buffers_offset, h->buffer_count,
		    sizeof(struct TmnfDetectBuffer), size)
	    || !range_valid(h->vertices_offset, h->vertex_count,
		    sizeof(GmVec3), size)
	    || !range_valid(h->faces_offset, h->face_count,
		    sizeof(GmSurfMeshFace), size)
	    || !range_valid(h->nodes_offset, h->node_count,
		    sizeof(GmSurfMeshNode), size)
	    || !range_valid(h->material_ids_offset, h->material_id_count, 1, size)
	    || !range_valid(h->collision_records_offset,
		    h->collision_record_count, sizeof(SHmsPhysicalCollision), size)
	    || !range_valid(h->boxes_offset, h->box_count,
		    sizeof(GmBoxAligned), size)) {
		fprintf(stderr, "invalid detection graph framing\n");
		exit(2);
	}
	view.located = (const struct TmnfDetectLocated *)(blob + h->located_offset);
	view.surfaces =
		(const struct TmnfDetectSurface *)(blob + h->surfaces_offset);
	view.isos = (const GmIso4 *)(blob + h->isos_offset);
	view.plugs =
		(const struct TmnfDetectPlugSurface *)(blob + h->plugs_offset);
	view.buffers =
		(const struct TmnfDetectBuffer *)(blob + h->buffers_offset);
	view.vertices = (const GmVec3 *)(blob + h->vertices_offset);
	view.faces = (const GmSurfMeshFace *)(blob + h->faces_offset);
	view.nodes = (const GmSurfMeshNode *)(blob + h->nodes_offset);
	view.material_ids = blob + h->material_ids_offset;
	view.records =
		(const SHmsPhysicalCollision *)(blob + h->collision_records_offset);
	view.boxes = (const GmBoxAligned *)(blob + h->boxes_offset);
	return view;
}

static ReplaySurface *surface_by_id(ReplayGraph *graph, uint32_t id)
{
	if (id == 0 || id > graph->input.header->surface_count) {
		fprintf(stderr, "invalid detection surface id %u\n", id);
		exit(2);
	}
	return &graph->surfaces[id - 1];
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

static void initialize_graph(ReplayGraph *graph, DetectView input)
{
	memset(graph, 0, sizeof(*graph));
	graph->input = input;
	const struct TmnfDetectTraceHeader *h = input.header;
	graph->surfaces =
		allocate((size_t)h->surface_count * sizeof(*graph->surfaces));
	graph->isos = allocate((size_t)h->iso_count * sizeof(*graph->isos));
	graph->located =
		allocate((size_t)h->located_count * sizeof(*graph->located));
	graph->plugs = allocate((size_t)h->plug_count * sizeof(*graph->plugs));
	graph->material_ids =
		allocate((size_t)h->plug_count * sizeof(*graph->material_ids));
	graph->buffers =
		allocate((size_t)h->buffer_count * sizeof(*graph->buffers));
	memcpy(graph->isos, input.isos, (size_t)h->iso_count * sizeof(GmIso4));

	for (uint32_t i = 0; i < h->surface_count; i++) {
		const struct TmnfDetectSurface *encoded = &input.surfaces[i];
		ReplaySurface *surface = &graph->surfaces[i];
		if (encoded->id != i + 1) {
			fprintf(stderr, "non-canonical detection surface ids\n");
			exit(2);
		}
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
				    > h->node_count) {
				fprintf(stderr, "invalid detection mesh slice\n");
				exit(2);
			}
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

	for (uint32_t i = 0; i < h->located_count; i++) {
		const struct TmnfDetectLocated *encoded = &input.located[i];
		graph->located[i].surf =
			surface_geom(surface_by_id(graph, encoded->surface_id));
		if (encoded->iso_id > h->iso_count) {
			fprintf(stderr, "invalid detection iso id\n");
			exit(2);
		}
		graph->located[i].iso =
			encoded->iso_id == 0 ? NULL : &graph->isos[encoded->iso_id - 1];
		graph->located[i].is_located = encoded->is_located;
	}

	for (uint32_t i = 0; i < h->plug_count; i++) {
		const struct TmnfDetectPlugSurface *encoded = &input.plugs[i];
		if ((uint64_t)encoded->material_index + encoded->material_count
		    > h->material_id_count) {
			fprintf(stderr, "invalid detection material slice\n");
			exit(2);
		}
		graph->plugs[i].geom =
			surface_geom(surface_by_id(graph, encoded->surface_id));
		graph->material_ids[i] = allocate(encoded->material_count);
		memcpy(graph->material_ids[i],
			&input.material_ids[encoded->material_index],
			encoded->material_count);
		graph->plugs[i].material_ids = graph->material_ids[i];
		graph->plugs[i].material_count = encoded->material_count;
	}

	for (uint32_t i = 0; i < h->buffer_count; i++) {
		const struct TmnfDetectBuffer *encoded = &input.buffers[i];
		ReplayBuffer *buffer = &graph->buffers[i];
		if (encoded->id != i + 1 || encoded->count > encoded->capacity
		    || (uint64_t)encoded->record_index + encoded->count
			    > h->collision_record_count) {
			fprintf(stderr, "invalid detection buffer slice\n");
			exit(2);
		}
		buffer->sphere.base.collisions.count = encoded->count;
		buffer->sphere.base.collisions.capacity = encoded->capacity;
		buffer->sphere.base.collisions.data = allocate(
			(size_t)encoded->capacity * sizeof(SHmsPhysicalCollision));
		memcpy(buffer->sphere.base.collisions.data,
			&input.records[encoded->record_index],
			(size_t)encoded->count * sizeof(SHmsPhysicalCollision));
		buffer->sphere.active = encoded->active;
	}
}

static void destroy_graph(ReplayGraph *graph)
{
	for (uint32_t i = 0; i < graph->input.header->surface_count; i++) {
		free(graph->surfaces[i].vertices);
		free(graph->surfaces[i].faces);
		free(graph->surfaces[i].nodes);
	}
	for (uint32_t i = 0; i < graph->input.header->plug_count; i++)
		free(graph->material_ids[i]);
	for (uint32_t i = 0; i < graph->input.header->buffer_count; i++)
		free(graph->buffers[i].sphere.base.collisions.data);
	free(graph->surfaces);
	free(graph->isos);
	free(graph->located);
	free(graph->plugs);
	free(graph->material_ids);
	free(graph->buffers);
}

static const char *collision_field(size_t offset)
{
	static const char *const fields[] = {
		"separation.x", "separation.y", "separation.z",
		"normal.x", "normal.y", "normal.z",
		"position.x", "position.y", "position.z",
		"materials", "flags",
		"face_normal.x", "face_normal.y", "face_normal.z",
	};
	if (offset < 0x24)
		return fields[offset / 4];
	if (offset < 0x28)
		return fields[9];
	if (offset < 0x2c)
		return fields[10];
	return fields[11 + (offset - 0x2c) / 4];
}

static int compare_bytes(
	const char *role, const void *actual_data,
	const void *expected_data, size_t size)
{
	const uint8_t *actual = actual_data;
	const uint8_t *expected = expected_data;
	if (memcmp(actual, expected, size) == 0)
		return 1;
	size_t offset = 0;
	while (offset < size && actual[offset] == expected[offset])
		offset++;
	uint32_t actual_word = 0;
	uint32_t expected_word = 0;
	size_t word = offset & ~(size_t)3;
	size_t available = size - word < 4 ? size - word : 4;
	memcpy(&actual_word, actual + word, available);
	memcpy(&expected_word, expected + word, available);
	if (diagnose)
		fprintf(stderr,
			"%s byte=%zu actual=%08x expected=%08x\n",
			role, offset, actual_word, expected_word);
	return 0;
}

static int compare_buffer(
	const ReplayBuffer *actual, const struct TmnfDetectBuffer *expected_desc,
	const SHmsPhysicalCollision *expected_records)
{
	const CFastBuffer_SHmsPhysicalCollision *buffer =
		&actual->sphere.base.collisions;
	if (buffer->count != expected_desc->count) {
		if (diagnose)
			fprintf(stderr, "collision count actual=%u expected=%u\n",
				buffer->count, expected_desc->count);
		return 0;
	}
	if (buffer->capacity != expected_desc->capacity) {
		if (diagnose)
			fprintf(stderr, "collision capacity actual=%u expected=%u\n",
				buffer->capacity, expected_desc->capacity);
		return 0;
	}
	for (uint32_t i = 0; i < buffer->count; i++) {
		const uint8_t *actual_bytes =
			(const uint8_t *)&buffer->data[i];
		const uint8_t *expected_bytes =
			(const uint8_t *)&expected_records[i];
		if (memcmp(actual_bytes, expected_bytes,
			    sizeof(SHmsPhysicalCollision)) != 0) {
			size_t offset = 0;
			while (actual_bytes[offset] == expected_bytes[offset])
				offset++;
			if (offset >= offsetof(SHmsPhysicalCollision, collision)
			    && offset < offsetof(SHmsPhysicalCollision, material)) {
				size_t collision_offset =
					offset - offsetof(SHmsPhysicalCollision, collision);
				uint32_t actual_word;
				uint32_t expected_word;
				size_t word = collision_offset & ~(size_t)3;
				memcpy(&actual_word,
					actual_bytes
						+ offsetof(SHmsPhysicalCollision, collision)
						+ word, 4);
				memcpy(&expected_word,
					expected_bytes
						+ offsetof(SHmsPhysicalCollision, collision)
						+ word, 4);
				if (diagnose)
					fprintf(stderr,
						"collision[%u].%s actual=%08x expected=%08x\n",
						i, collision_field(collision_offset),
						actual_word, expected_word);
			} else {
				compare_bytes("physical collision",
					actual_bytes, expected_bytes,
					sizeof(SHmsPhysicalCollision));
			}
			return 0;
		}
	}
	if (expected_desc->has_active
	    && actual->sphere.active != expected_desc->active) {
		if (diagnose)
			fprintf(stderr, "buffer active actual=%u expected=%u\n",
				actual->sphere.active, expected_desc->active);
		return 0;
	}
	return 1;
}

static int replay_graph(DetectView input, DetectView expected)
{
	if (input.header->kind != expected.header->kind) {
		fprintf(stderr, "detection graph kind changed\n");
		exit(2);
	}
	ReplayGraph graph;
	initialize_graph(&graph, input);
	int result = 1;
	uint32_t return_value = 0;
	CollisionRuntime runtime;
	CollisionRuntime_Init(&runtime);

	switch (input.header->kind) {
	case TMNF_DETECT_SPHERE_MESH:
		return_value = GmCollision_Sphere_Mesh(
			&graph.located[0], &graph.located[1],
			&graph.buffers[0].sphere.base);
		break;
	case TMNF_DETECT_ELLIPSOID_MESH:
		return_value = GmCollision_Ellipsoid_Mesh(
			&graph.located[0], &graph.located[1],
			&graph.buffers[0].sphere.base);
		break;
	case TMNF_DETECT_BOX_MESH:
		return_value = GmCollision_Box_Mesh(
			&graph.located[0], &graph.located[1],
			&graph.buffers[0].sphere.base);
		break;
	case TMNF_DETECT_SPHERE_SPHERE:
		return_value = GmCollision_Sphere_Sphere(
			&graph.located[0], &graph.located[1],
			&graph.buffers[0].sphere.base);
		break;
	case TMNF_DETECT_SURF_DISPATCH:
		if (input.header->comparator_va != 0x008EA2D0u
		    && input.header->comparator_va != 0x008F5200u
		    && input.header->comparator_va != 0x008F49D0u
		    && input.header->comparator_va != 0x008EADC0u) {
			if (diagnose)
				fprintf(stderr,
				"unimplemented dispatch target=%08x types=%u,%u\n",
				input.header->comparator_va,
				graph.located[0].surf->type,
				graph.located[1].surf->type);
			result = 0;
			break;
		}
		return_value = GmSurf_ComputeCollision(
			&graph.located[0], &graph.located[1],
			&graph.buffers[0].sphere.base, &runtime);
		break;
	case TMNF_DETECT_PLUG_SURFACE:
		if (input.header->surface_count != 2
		    || (surface_geom(&graph.surfaces[0])->type != GM_SURF_SPHERE
			&& surface_geom(&graph.surfaces[0])->type != GM_SURF_ELLIPSOID
			&& surface_geom(&graph.surfaces[0])->type
				!= GM_SURF_MESH)
		    || (surface_geom(&graph.surfaces[1])->type != GM_SURF_SPHERE
			&& surface_geom(&graph.surfaces[1])->type != GM_SURF_ELLIPSOID
			&& surface_geom(&graph.surfaces[1])->type
				!= GM_SURF_MESH)) {
			if (diagnose)
				fprintf(stderr, "unimplemented plug pair types=%u,%u\n",
					surface_geom(&graph.surfaces[0])->type,
					surface_geom(&graph.surfaces[1])->type);
			result = 0;
			break;
		}
		return_value = CPlugSurface_ComputeCollision(
			&graph.plugs[0], &graph.isos[0],
			&graph.plugs[1], &graph.isos[1],
			&graph.buffers[0].sphere.base, &runtime);
		break;
	case TMNF_DETECT_BOX_TEST_INTER:
		return_value = (uint32_t)GmBoxAligned_TestInter(
			&input.boxes[0], &input.boxes[1]);
		break;
	case TMNF_DETECT_BOX_SET_MULT: {
		GmBoxAligned box = input.boxes[0];
		GmBoxAligned_SetMult(&box, &input.boxes[1], &input.isos[0]);
		result = compare_bytes(
			"SetMult box", &box, &expected.boxes[0], sizeof(box));
		break;
	}
	case TMNF_DETECT_MERGE:
		SHmsSphereBufferContact_MergeAndAddToCollisions(
			&graph.buffers[0].sphere,
			&graph.buffers[1].sphere.base);
		break;
	case TMNF_DETECT_QSORT:
		if (input.header->comparator_va != 0x00547C80u) {
			fprintf(stderr, "unknown qsort comparator %08x\n",
				input.header->comparator_va);
			exit(2);
		}
		CFastBuffer_SHmsPhysicalCollision_QSort(
			&graph.buffers[0].sphere.base.collisions,
			SHmsPhysicalCollision_Compare);
		break;
	default:
		fprintf(stderr, "unsupported detection graph kind %u\n",
			input.header->kind);
		exit(2);
	}

	if ((input.header->kind == TMNF_DETECT_SPHERE_MESH
		|| input.header->kind == TMNF_DETECT_BOX_MESH
		|| input.header->kind == TMNF_DETECT_SPHERE_SPHERE
		|| input.header->kind == TMNF_DETECT_SURF_DISPATCH
		|| input.header->kind == TMNF_DETECT_PLUG_SURFACE
		|| input.header->kind == TMNF_DETECT_BOX_TEST_INTER)
	    && return_value != expected.header->return_value) {
		if (diagnose)
			fprintf(stderr, "return value actual=%u expected=%u\n",
				return_value, expected.header->return_value);
		result = 0;
	}
	if (result && expected.header->buffer_count != input.header->buffer_count) {
		fprintf(stderr, "output buffer identity changed\n");
		result = 0;
	}
	for (uint32_t i = 0; result && i < expected.header->buffer_count; i++) {
		const struct TmnfDetectBuffer *desc = &expected.buffers[i];
		result = compare_buffer(
			&graph.buffers[i], desc,
			&expected.records[desc->record_index]);
	}
	destroy_graph(&graph);
	return result;
}

static int replay_trace(const char *path, uint32_t expected_va)
{
	FILE *file = fopen(path, "rb");
	if (file == NULL)
		die("cannot open trace", path);
	char magic[8];
	if (fread(magic, 1, sizeof(magic), file) != sizeof(magic)
	    || memcmp(magic, "TMNFTRC1", 8) != 0)
		die("bad trace magic", path);
	uint32_t va = read_u32(file, path);
	uint32_t count = read_u32(file, path);
	if (va != expected_va)
		die("trace VA mismatch", path);
	uint32_t passed = 0;
	diagnose = 1;
	for (uint32_t i = 0; i < count; i++) {
		(void)read_u32(file, path);
		uint16_t input_count = read_u16(file, path);
		uint16_t output_count = read_u16(file, path);
		if (input_count != 1 || output_count != 1)
			die("invalid detection record shape", path);
		(void)read_u32(file, path);
		(void)read_u32(file, path);
		uint32_t input_size = read_u32(file, path);
		uint8_t *input_blob = read_blob(file, input_size, path);
		(void)read_u32(file, path);
		(void)read_u32(file, path);
		uint32_t output_size = read_u32(file, path);
		uint8_t *output_blob = read_blob(file, output_size, path);
		DetectView input = detect_view(input_blob, input_size);
		DetectView output = detect_view(output_blob, output_size);
		int ok = replay_graph(input, output);
		free(input_blob);
		free(output_blob);
		if (ok) {
			passed++;
		} else if (diagnose) {
			fprintf(stderr, "%s record %u first failure\n", path, i);
			diagnose = 0;
		}
	}
	if (fgetc(file) != EOF)
		die("trailing trace bytes", path);
	fclose(file);
	printf("%08X: %u/%u\n", va, passed, count);
	return passed == count;
}

int main(int argc, char **argv)
{
	if (argc != 2) {
		fprintf(stderr, "usage: %s TRACE_DIRECTORY\n", argv[0]);
		return 2;
	}
	static const struct {
		uint32_t va;
		const char *name;
	} targets[] = {
		{ 0x008EA2D0u, "GmCollision_Sphere_Mesh" },
		{ 0x008EADC0u, "GmCollision_Ellipsoid_Mesh" },
		{ 0x008F5200u, "GmCollision_Box_Mesh" },
		{ 0x008F49D0u, "GmCollision_Sphere_Sphere" },
		{ 0x008E8890u, "GmSurf_ComputeCollision" },
		{ 0x00537150u, "CPlugSurface_ComputeCollision" },
		{ 0x00537530u, "GmBoxAligned_TestInter" },
		{ 0x008E5230u, "GmBoxAligned_SetMult" },
		{ 0x00538100u,
			"SHmsSphereBufferContact_MergeAndAddToCollisions" },
		{ 0x00547DE0u, "CFastBuffer_SHmsPhysicalCollision_QSort" },
	};
	int success = 1;
	for (size_t i = 0; i < sizeof(targets) / sizeof(targets[0]); i++) {
		char path[1024];
		int length = snprintf(path, sizeof(path), "%s/%08X_%s.bin",
			argv[1], targets[i].va, targets[i].name);
		if (length < 0 || (size_t)length >= sizeof(path))
			die("trace path too long", argv[1]);
		if (!replay_trace(path, targets[i].va))
			success = 0;
	}
	return success ? 0 : 1;
}
