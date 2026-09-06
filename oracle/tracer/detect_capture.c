#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "detect_capture.h"
#include "detect_trace.h"

#define IMAGE_BASE 0x00400000u
#define PHYSICAL_COLLISION_SIZE 0x4cu

struct ByteVec {
	uint8_t *data;
	uint32_t size;
	uint32_t capacity;
};

struct IdMap {
	uint32_t *addresses;
	uint32_t count;
	uint32_t capacity;
};

struct DetectCapture {
	uint32_t va;
	uint32_t kind;
	uint32_t this_ptr;
	uint32_t entry_esp;
	uint32_t located[2];
	uint32_t located_count;
	uint32_t isos[2];
	uint32_t iso_count;
	uint32_t plugs[2];
	uint32_t plug_count;
	uint32_t buffers[2];
	uint32_t buffer_count;
	uint8_t buffer_has_active[2];
	uint32_t boxes[3];
	uint32_t box_count;
	uint32_t comparator;
	struct IdMap corpora;
	struct IdMap trees;
	struct IdMap materials;
	uint8_t *input;
	uint32_t input_size;
};

static HANDLE g_heap;
static uint32_t g_module_base;
static DetectCaptureFatalFn g_fatal;

static void fail(const char *message)
{
	g_fatal(message);
}

static void read_runtime(uint32_t address, void *output, uint32_t size)
{
	SIZE_T read = 0;
	char message[128];
	if (!ReadProcessMemory(
		    GetCurrentProcess(), (const void *)(uintptr_t)address,
		    output, size, &read)
	    || read != size) {
		_snprintf(
			message, sizeof(message),
			"cannot read detect address %08X size=%u", address, size);
		fail(message);
	}
}

static uint32_t read_u32(uint32_t address)
{
	uint32_t value;
	read_runtime(address, &value, sizeof(value));
	return value;
}

static uint16_t read_u16(uint32_t address)
{
	uint16_t value;
	read_runtime(address, &value, sizeof(value));
	return value;
}

static uint8_t read_u8(uint32_t address)
{
	uint8_t value;
	read_runtime(address, &value, sizeof(value));
	return value;
}

static void *heap_allocate(uint32_t size)
{
	void *memory = HeapAlloc(g_heap, HEAP_ZERO_MEMORY, size == 0 ? 1 : size);
	if (memory == NULL)
		fail("detect capture allocation failed");
	return memory;
}

static void *heap_resize(void *memory, uint32_t size)
{
	void *resized = memory == NULL
		? HeapAlloc(g_heap, HEAP_ZERO_MEMORY, size)
		: HeapReAlloc(g_heap, HEAP_ZERO_MEMORY, memory, size);
	if (resized == NULL)
		fail("detect capture resize failed");
	return resized;
}

static void vec_reserve(struct ByteVec *vec, uint32_t extra)
{
	uint32_t required;
	uint32_t capacity;
	if (extra > UINT32_MAX - vec->size)
		fail("detect trace size overflow");
	required = vec->size + extra;
	if (required <= vec->capacity)
		return;
	capacity = vec->capacity == 0 ? 512 : vec->capacity;
	while (capacity < required) {
		if (capacity > UINT32_MAX / 2)
			fail("detect trace capacity overflow");
		capacity *= 2;
	}
	vec->data = (uint8_t *)heap_resize(vec->data, capacity);
	vec->capacity = capacity;
}

static uint32_t vec_append(
	struct ByteVec *vec, const void *data, uint32_t size)
{
	uint32_t offset = vec->size;
	vec_reserve(vec, size);
	memcpy(vec->data + offset, data, size);
	vec->size += size;
	return offset;
}

static uint32_t static_va(uint32_t address)
{
	if (address < g_module_base)
		fail("detect target precedes module");
	return address - g_module_base + IMAGE_BASE;
}

static uint32_t map_address(
	uint32_t *addresses, uint32_t *count, uint32_t capacity,
	uint32_t address, const char *role)
{
	uint32_t i;
	char message[128];
	if (address == 0)
		return 0;
	for (i = 0; i < *count; ++i) {
		if (addresses[i] == address)
			return i + 1;
	}
	if (*count == capacity) {
		_snprintf(message, sizeof(message), "too many detect %s objects", role);
		fail(message);
	}
	addresses[*count] = address;
	++*count;
	return *count;
}

static uint32_t map_id(
	struct IdMap *map, uint32_t address, int allow_new, const char *role)
{
	uint32_t i;
	char message[128];
	if (address == 0)
		return 0;
	for (i = 0; i < map->count; ++i) {
		if (map->addresses[i] == address)
			return i + 1;
	}
	if (!allow_new) {
		_snprintf(message, sizeof(message),
			"unknown detect %s reference %08X", role, address);
		fail(message);
	}
	if (map->count == map->capacity) {
		uint32_t capacity = map->capacity == 0 ? 8 : map->capacity * 2;
		if (capacity < map->capacity)
			fail("detect identity map overflow");
		map->addresses = (uint32_t *)heap_resize(
			map->addresses, capacity * sizeof(*map->addresses));
		map->capacity = capacity;
	}
	map->addresses[map->count++] = address;
	return map->count;
}

static void append_surface(
	uint32_t address, uint32_t id,
	struct ByteVec *surfaces, struct ByteVec *vertices,
	struct ByteVec *faces, struct ByteVec *nodes)
{
	struct TmnfDetectSurface surface;
	uint32_t count;
	uint32_t data;
	memset(&surface, 0, sizeof(surface));
	surface.id = id;
	surface.material_index = read_u16(address + 4);
	surface.type = read_u8(address + 6);
	surface.reserved = read_u8(address + 7);
	if (surface.type == 0) {
		read_runtime(address + 8, surface.shape, 4);
	} else if (surface.type == 1) {
		read_runtime(address + 8, surface.shape, 0x0c);
	} else if (surface.type == 6) {
		read_runtime(address + 8, surface.shape, 0x18);
	} else if (surface.type == 7) {
		count = read_u32(address + 0x08);
		data = read_u32(address + 0x0c);
		if (count > 1000000u || (count != 0 && data == 0))
			fail("invalid detect mesh vertex buffer");
		surface.vertex_index = vertices->size / 0x0c;
		surface.vertex_count = count;
		if (count != 0)
			vec_reserve(vertices, count * 0x0c);
		read_runtime(data, vertices->data + vertices->size, count * 0x0c);
		vertices->size += count * 0x0c;

		count = read_u32(address + 0x10);
		data = read_u32(address + 0x14);
		if (count > 1000000u || (count != 0 && data == 0))
			fail("invalid detect mesh face buffer");
		surface.face_index = faces->size / 0x20;
		surface.face_count = count;
		if (count != 0)
			vec_reserve(faces, count * 0x20);
		read_runtime(data, faces->data + faces->size, count * 0x20);
		faces->size += count * 0x20;

		count = read_u32(address + 0x20);
		data = read_u32(address + 0x24);
		if (count > read_u32(address + 0x28) || count > 1000000u
		    || (count != 0 && data == 0))
			fail("invalid detect mesh node buffer");
		surface.node_index = nodes->size / 0x20;
		surface.node_count = count;
		if (count != 0)
			vec_reserve(nodes, count * 0x20);
		read_runtime(data, nodes->data + nodes->size, count * 0x20);
		nodes->size += count * 0x20;
	} else if (surface.type >= 9) {
		char message[96];
		_snprintf(
			message, sizeof(message),
			"unsupported detect surface type=%u address=%08X",
			surface.type, address);
		fail(message);
	}
	vec_append(surfaces, &surface, sizeof(surface));
}

static void append_buffer(
	struct DetectCapture *capture, uint32_t address, uint32_t id,
	int has_active, int output,
	struct ByteVec *buffers, struct ByteVec *records)
{
	struct TmnfDetectBuffer buffer;
	uint32_t data;
	memset(&buffer, 0, sizeof(buffer));
	buffer.id = id;
	buffer.count = read_u32(address + 4);
	data = read_u32(address + 8);
	buffer.capacity = read_u32(address + 12);
	if (buffer.count > buffer.capacity || buffer.capacity > 100000u
	    || (buffer.capacity != 0 && data == 0))
		fail("invalid detect collision buffer");
	buffer.record_index = records->size / PHYSICAL_COLLISION_SIZE;
	buffer.has_active = has_active != 0;
	if (has_active)
		buffer.active = read_u32(address + 0x10);
	if (buffer.count != 0) {
		uint32_t i;
		for (i = 0; i < buffer.count; ++i) {
			uint8_t encoded[PHYSICAL_COLLISION_SIZE];
			read_runtime(
				data + i * PHYSICAL_COLLISION_SIZE,
				encoded, sizeof(encoded));
			if (capture->kind == TMNF_DETECT_MERGE
			    || capture->kind == TMNF_DETECT_QSORT) {
				uint32_t value;
#define NORMALIZE(offset_, map_, role_) \
	do { \
		memcpy(&value, encoded + (offset_), 4); \
		value = map_id(&(map_), value, !output, (role_)); \
		memcpy(encoded + (offset_), &value, 4); \
	} while (0)
				NORMALIZE(0x00, capture->corpora, "corpus");
				NORMALIZE(0x04, capture->trees, "tree");
				NORMALIZE(0x08, capture->corpora, "corpus");
				NORMALIZE(0x0c, capture->trees, "tree");
				NORMALIZE(0x48, capture->materials, "material");
#undef NORMALIZE
			} else {
				memset(encoded, 0, 0x10);
				memset(encoded + 0x48, 0, 4);
			}
			vec_append(records, encoded, sizeof(encoded));
		}
	}
	vec_append(buffers, &buffer, sizeof(buffer));
}

static uint8_t *serialize(
	struct DetectCapture *capture, uint32_t return_value,
	int output, uint32_t *size)
{
	struct TmnfDetectTraceHeader header;
	struct ByteVec blob = { 0 };
	struct ByteVec located = { 0 };
	struct ByteVec surfaces = { 0 };
	struct ByteVec isos = { 0 };
	struct ByteVec plugs = { 0 };
	struct ByteVec buffers = { 0 };
	struct ByteVec vertices = { 0 };
	struct ByteVec faces = { 0 };
	struct ByteVec nodes = { 0 };
	struct ByteVec material_ids = { 0 };
	struct ByteVec records = { 0 };
	struct ByteVec boxes = { 0 };
	uint32_t surface_addresses[2] = { 0 };
	uint32_t surface_count = 0;
	uint32_t iso_addresses[2] = { 0 };
	uint32_t iso_count = 0;
	uint32_t i;

	memset(&header, 0, sizeof(header));
	memcpy(header.magic, TMNF_DETECT_TRACE_MAGIC, 8);
	header.version = TMNF_DETECT_TRACE_VERSION;
	header.kind = capture->kind;
	header.return_value = return_value;
	header.comparator_va = capture->comparator == 0
		? 0 : static_va(capture->comparator);

	if (!output) {
		for (i = 0; i < capture->iso_count; ++i)
			map_address(
				iso_addresses, &iso_count, 2,
				capture->isos[i], "iso");
		for (i = 0; i < capture->located_count; ++i) {
			struct TmnfDetectLocated encoded;
			uint32_t source = capture->located[i];
			uint32_t surface = read_u32(source);
			uint32_t iso = read_u32(source + 4);
			encoded.surface_id = map_address(
				surface_addresses, &surface_count, 2, surface, "surface");
			encoded.iso_id = map_address(
				iso_addresses, &iso_count, 2, iso, "iso");
			encoded.is_located = read_u32(source + 8);
			vec_append(&located, &encoded, sizeof(encoded));
		}
		for (i = 0; i < capture->plug_count; ++i) {
			struct TmnfDetectPlugSurface plug;
			uint32_t source = capture->plugs[i];
			uint32_t wrapper = read_u32(source + 0x14);
			uint32_t material_data = read_u32(source + 0x1c);
			if (wrapper == 0)
				fail("detect plug surface has no geometry wrapper");
			plug.surface_id = map_address(
				surface_addresses, &surface_count, 2,
				read_u32(wrapper + 0x34), "surface");
			plug.material_index = material_ids.size;
			plug.material_count = read_u32(source + 0x18);
			if (plug.material_count > read_u32(source + 0x20)
			    || plug.material_count > 65536u
			    || (plug.material_count != 0 && material_data == 0))
				fail("invalid detect plug material buffer");
			if (plug.material_count != 0) {
				vec_reserve(&material_ids, plug.material_count);
				for (uint32_t material = 0;
				     material < plug.material_count; ++material) {
					uint32_t pointer = read_u32(
						material_data + material * 4);
					if (pointer == 0)
						fail("detect plug surface has null material");
					material_ids.data[material_ids.size + material] =
						read_u8(pointer + 0x18);
				}
				material_ids.size += plug.material_count;
			}
			vec_append(&plugs, &plug, sizeof(plug));
		}
		for (i = 0; i < surface_count; ++i)
			append_surface(
				surface_addresses[i], i + 1, &surfaces,
				&vertices, &faces, &nodes);
		for (i = 0; i < iso_count; ++i) {
			uint8_t iso[0x30];
			read_runtime(iso_addresses[i], iso, sizeof(iso));
			vec_append(&isos, iso, sizeof(iso));
		}
		for (i = 0; i < capture->box_count; ++i) {
			uint8_t box[0x18];
			read_runtime(capture->boxes[i], box, sizeof(box));
			vec_append(&boxes, box, sizeof(box));
		}
	}

	for (i = 0; i < capture->buffer_count; ++i)
		append_buffer(
			capture, capture->buffers[i], i + 1,
			capture->buffer_has_active[i], output, &buffers, &records);
	if (output && capture->kind == TMNF_DETECT_BOX_SET_MULT) {
		uint8_t box[0x18];
		read_runtime(capture->boxes[0], box, sizeof(box));
		vec_append(&boxes, box, sizeof(box));
	}

	vec_append(&blob, &header, sizeof(header));
#define APPEND_SECTION(field_, count_field_, vector_, stride_) \
	do { \
		header.field_ = blob.size; \
		header.count_field_ = (vector_).size / (stride_); \
		if ((vector_).size != 0) \
			vec_append(&blob, (vector_).data, (vector_).size); \
	} while (0)
	APPEND_SECTION(located_offset, located_count, located,
		sizeof(struct TmnfDetectLocated));
	APPEND_SECTION(surfaces_offset, surface_count, surfaces,
		sizeof(struct TmnfDetectSurface));
	APPEND_SECTION(isos_offset, iso_count, isos, 0x30);
	APPEND_SECTION(plugs_offset, plug_count, plugs,
		sizeof(struct TmnfDetectPlugSurface));
	APPEND_SECTION(buffers_offset, buffer_count, buffers,
		sizeof(struct TmnfDetectBuffer));
	APPEND_SECTION(vertices_offset, vertex_count, vertices, 0x0c);
	APPEND_SECTION(faces_offset, face_count, faces, 0x20);
	APPEND_SECTION(nodes_offset, node_count, nodes, 0x20);
	APPEND_SECTION(material_ids_offset, material_id_count, material_ids, 1);
	APPEND_SECTION(collision_records_offset, collision_record_count,
		records, PHYSICAL_COLLISION_SIZE);
	APPEND_SECTION(boxes_offset, box_count, boxes, 0x18);
#undef APPEND_SECTION
	header.total_size = blob.size;
	memcpy(blob.data, &header, sizeof(header));

#define FREE_VEC(vector_) \
	do { \
		if ((vector_).data != NULL) \
			HeapFree(g_heap, 0, (vector_).data); \
	} while (0)
	FREE_VEC(located);
	FREE_VEC(surfaces);
	FREE_VEC(isos);
	FREE_VEC(plugs);
	FREE_VEC(buffers);
	FREE_VEC(vertices);
	FREE_VEC(faces);
	FREE_VEC(nodes);
	FREE_VEC(material_ids);
	FREE_VEC(records);
	FREE_VEC(boxes);
#undef FREE_VEC
	*size = blob.size;
	return blob.data;
}

void detect_capture_initialize(
	HANDLE heap, uint32_t module_base, DetectCaptureFatalFn fatal_fn)
{
	g_heap = heap;
	g_module_base = module_base;
	g_fatal = fatal_fn;
}

int detect_capture_is_target(uint32_t va)
{
	switch (va) {
	case 0x008EA2D0u:
	case 0x008EADC0u:
	case 0x008F5200u:
	case 0x008F49D0u:
	case 0x008E8890u:
	case 0x00537150u:
	case 0x00537530u:
	case 0x008E5230u:
	case 0x00538100u:
	case 0x00547DE0u:
		return 1;
	default:
		return 0;
	}
}

int detect_capture_is_race_target(uint32_t va)
{
	if (detect_capture_is_target(va))
		return 1;
	switch (va) {
	case 0x0053B1C0u:
	case 0x0053AFB0u:
	case 0x0053A8F0u:
	case 0x0053A120u:
	case 0x0053A3D0u:
	case 0x0053A660u:
	case 0x0053A0E0u:
	case 0x00537E80u:
	case 0x00537F30u:
		return 1;
	default:
		return 0;
	}
}

int detect_capture_race_ready(void)
{
	typedef int32_t (__attribute__((thiscall)) *RaceTimeFn)(void *race);
	uint32_t root = read_u32(
		g_module_base + 0x00D68C44u - IMAGE_BASE);
	uint32_t race;
	RaceTimeFn race_time;
	if (root == 0)
		return 0;
	race = read_u32(root + 0x454);
	if (race == 0 || read_u32(race + 0x18) != root
	    || read_u32(race + 0xc4) == 0)
		return 0;
	race_time = (RaceTimeFn)(uintptr_t)(
		g_module_base + 0x0047E680u - IMAGE_BASE);
	return race_time((void *)(uintptr_t)race) >= 0;
}

struct DetectCapture *detect_capture_create(
	uint32_t va, uint32_t this_ptr, uint32_t entry_esp)
{
	struct DetectCapture *capture =
		(struct DetectCapture *)heap_allocate(sizeof(*capture));
	capture->va = va;
	capture->this_ptr = this_ptr;
	capture->entry_esp = entry_esp;
	switch (va) {
	case 0x008EA2D0u:
		capture->kind = TMNF_DETECT_SPHERE_MESH;
		goto geometry;
	case 0x008EADC0u:
		capture->kind = TMNF_DETECT_ELLIPSOID_MESH;
		goto geometry;
	case 0x008F5200u:
		capture->kind = TMNF_DETECT_BOX_MESH;
		goto geometry;
	case 0x008F49D0u:
		capture->kind = TMNF_DETECT_SPHERE_SPHERE;
		goto geometry;
	case 0x008E8890u:
		capture->kind = TMNF_DETECT_SURF_DISPATCH;
geometry:
		capture->located[0] = read_u32(entry_esp + 4);
		capture->located[1] = read_u32(entry_esp + 8);
		capture->located_count = 2;
		capture->buffers[0] = read_u32(entry_esp + 12);
		capture->buffer_count = 1;
		if (va == 0x008E8890u) {
			uint32_t first = read_u32(capture->located[0]);
			uint32_t second = read_u32(capture->located[1]);
			uint32_t type1 = read_u8(first + 6);
			uint32_t type2 = read_u8(second + 6);
			uint32_t index;
			if (type1 > type2) {
				uint32_t swap = type1;
				type1 = type2;
				type2 = swap;
			}
			index = type1 * 9 + type2;
			capture->comparator = read_u32(
				g_module_base + 0x00D706E0u - IMAGE_BASE
				+ index * 4);
		}
		break;
	case 0x00537150u:
		this_ptr = read_u32(entry_esp + 4);
		capture->kind = TMNF_DETECT_PLUG_SURFACE;
		capture->plugs[0] = read_u32(this_ptr);
		capture->plugs[1] = read_u32(this_ptr + 8);
		capture->plug_count = 2;
		capture->isos[0] = read_u32(this_ptr + 4);
		capture->isos[1] = read_u32(this_ptr + 12);
		capture->iso_count = 2;
		capture->buffers[0] = read_u32(entry_esp + 8);
		capture->buffer_count = 1;
		break;
	case 0x00537530u:
		capture->kind = TMNF_DETECT_BOX_TEST_INTER;
		capture->boxes[0] = this_ptr;
		capture->boxes[1] = read_u32(entry_esp + 4);
		capture->box_count = 2;
		break;
	case 0x008E5230u:
		capture->kind = TMNF_DETECT_BOX_SET_MULT;
		capture->boxes[0] = this_ptr;
		capture->boxes[1] = read_u32(entry_esp + 4);
		capture->box_count = 2;
		capture->isos[0] = read_u32(entry_esp + 8);
		capture->iso_count = 1;
		break;
	case 0x00538100u:
		capture->kind = TMNF_DETECT_MERGE;
		capture->buffers[0] = this_ptr;
		capture->buffer_has_active[0] = 1;
		capture->buffers[1] = read_u32(entry_esp + 4);
		capture->buffer_count = 2;
		break;
	case 0x00547DE0u:
		capture->kind = TMNF_DETECT_QSORT;
		capture->buffers[0] = this_ptr - 4;
		capture->buffer_count = 1;
		capture->comparator = read_u32(entry_esp + 4);
		break;
	default:
		fail("unsupported detect capture target");
	}
	capture->input = serialize(capture, 0, 0, &capture->input_size);
	return capture;
}

const uint8_t *detect_capture_input(
	const struct DetectCapture *capture, uint32_t *size)
{
	*size = capture->input_size;
	return capture->input;
}

uint8_t *detect_capture_output(
	struct DetectCapture *capture, uint32_t return_value, uint32_t *size)
{
	return serialize(capture, return_value, 1, size);
}

void detect_capture_free_blob(uint8_t *blob)
{
	if (blob != NULL)
		HeapFree(g_heap, 0, blob);
}

void detect_capture_destroy(struct DetectCapture *capture)
{
	if (capture == NULL)
		return;
	if (capture->input != NULL)
		HeapFree(g_heap, 0, capture->input);
	if (capture->corpora.addresses != NULL)
		HeapFree(g_heap, 0, capture->corpora.addresses);
	if (capture->trees.addresses != NULL)
		HeapFree(g_heap, 0, capture->trees.addresses);
	if (capture->materials.addresses != NULL)
		HeapFree(g_heap, 0, capture->materials.addresses);
	HeapFree(g_heap, 0, capture);
}
