#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "model6_capture.h"
#include "model6_trace.h"

#define IMAGE_BASE 0x00400000u
#define TIMER_ROOT_VA 0x00D731E0u

struct ByteVec {
	uint8_t *data;
	uint32_t size;
	uint32_t capacity;
};

struct IdEntry {
	uint32_t address;
	uint32_t id;
};

struct IdMap {
	struct IdEntry *entries;
	uint32_t count;
	uint32_t capacity;
};

struct Model6Capture {
	uint32_t car;
	uint32_t tuning;
	uint32_t wheels;
	uint32_t wheel_count;
	uint32_t item;
	uint32_t corpus;
	uint32_t dyna;
	uint32_t params;
	uint32_t state;
	uint32_t model_iso;
	uint32_t body_reference;
	uint32_t ground_ids;
	uint32_t ground_id_count;
	uint32_t ground_material_ptrs;
	uint32_t ground_material_count;
	uint32_t curves[TMNF_MODEL6_CURVE_COUNT];
	uint32_t entry_esp;
	uint32_t tick;
	uint32_t sliding_ptr;
	uint32_t brake_ptr;
	struct TmnfModel6TraceHeader arguments;
	int debug;
	struct IdMap ids;
	uint8_t *input;
	uint32_t input_size;
};

static const uint32_t CURVE_OFFSETS[TMNF_MODEL6_CURVE_COUNT] = {
	0x034, 0x078, 0x0A0, 0x0AC, 0x1E0, 0x224,
	0x230, 0x250, 0x25C, 0x260, 0x288, 0x2A4
};

static const uint8_t TMNF_21126_EXE_SHA256[32] = {
	0x38, 0x47, 0xcf, 0x9f, 0x20, 0xbf, 0xc6, 0x39,
	0x14, 0x45, 0x00, 0x60, 0xed, 0x52, 0x8c, 0x12,
	0x10, 0x4f, 0x74, 0x3d, 0x96, 0xad, 0x23, 0xd6,
	0xe7, 0x6a, 0xbd, 0x17, 0x8d, 0xe8, 0xc8, 0x4f,
};

static HANDLE g_heap;
static uint32_t g_module_base;
static Model6CaptureFatalFn g_fatal;
static Model6CaptureFatalFn g_log;
static uint8_t g_track_sha256[32];
static volatile LONG g_capture_count;

static void fail(const char *message)
{
	g_fatal(message);
}

static void read_runtime(uint32_t address, void *output, uint32_t size)
{
	SIZE_T copied = 0;
	char message[128];
	if (address == 0
	    || !ReadProcessMemory(
		    GetCurrentProcess(), (const void *)(uintptr_t)address,
		    output, size, &copied)
	    || copied != size) {
		_snprintf(message, sizeof(message),
			"Model6 read failed address=%08X size=%u",
			address, size);
		fail(message);
	}
}

static uint32_t read_u32(uint32_t address)
{
	uint32_t value;
	read_runtime(address, &value, sizeof(value));
	return value;
}

static void *heap_resize(void *memory, uint32_t size)
{
	void *result = memory == NULL
		? HeapAlloc(g_heap, HEAP_ZERO_MEMORY, size)
		: HeapReAlloc(g_heap, HEAP_ZERO_MEMORY, memory, size);
	if (result == NULL)
		fail("Model6 capture allocation failed");
	return result;
}

static void vec_reserve(struct ByteVec *vec, uint32_t extra)
{
	uint32_t required;
	uint32_t capacity;
	if (extra > UINT32_MAX - vec->size)
		fail("Model6 blob overflow");
	required = vec->size + extra;
	if (required <= vec->capacity)
		return;
	capacity = vec->capacity == 0 ? 4096 : vec->capacity;
	while (capacity < required) {
		if (capacity > UINT32_MAX / 2)
			fail("Model6 blob capacity overflow");
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
	memcpy(vec->data + vec->size, data, size);
	vec->size += size;
	return offset;
}

static uint32_t vec_append_runtime(
	struct ByteVec *vec, uint32_t address, uint32_t size)
{
	uint32_t offset = vec->size;
	vec_reserve(vec, size);
	read_runtime(address, vec->data + vec->size, size);
	vec->size += size;
	return offset;
}

static uint32_t map_id(struct IdMap *map, uint32_t address)
{
	uint32_t i;
	if (address == 0)
		return 0;
	for (i = 0; i < map->count; ++i) {
		if (map->entries[i].address == address)
			return map->entries[i].id;
	}
	if (map->count == map->capacity) {
		uint32_t capacity = map->capacity == 0 ? 32 : map->capacity * 2;
		if (capacity < map->capacity)
			fail("Model6 identity map overflow");
		map->entries = (struct IdEntry *)heap_resize(
			map->entries, capacity * sizeof(*map->entries));
		map->capacity = capacity;
	}
	map->entries[map->count].address = address;
	map->entries[map->count].id = map->count + 1;
	++map->count;
	return map->count;
}

static uint32_t selected_tuning(uint32_t car)
{
	uint32_t container = read_u32(car + 0x64);
	uint32_t data;
	uint32_t index;
	if (container == 0)
		fail("Model6 car has no tuning container");
	data = read_u32(container + 0x18);
	index = read_u32(container + 0x24);
	if (data == 0)
		fail("Model6 tuning array is null");
	return read_u32(data + index * 4);
}

static uint32_t resolve_model_iso(uint32_t corpus)
{
	typedef uint32_t (__attribute__((thiscall)) *GetIsoFn)(uint32_t);
	uint32_t vtable = read_u32(corpus);
	GetIsoFn get_iso;
	if (vtable == 0)
		fail("Model6 corpus has no vtable");
	get_iso = (GetIsoFn)(uintptr_t)read_u32(vtable + 0x78);
	if (get_iso == NULL)
		fail("Model6 corpus has no location getter");
	return get_iso(corpus);
}

static uint32_t current_tick(void)
{
	uint32_t root = read_u32(
		g_module_base + TIMER_ROOT_VA - IMAGE_BASE);
	uint32_t timer = read_u32(root + 0x14);
	if (timer == 0)
		timer = root + 0xA0;
	return read_u32(timer + 0x1C);
}

static void discover_graph(struct Model6Capture *capture)
{
	uint32_t corpus_data;
	uint32_t manager;
	uint32_t i;
	capture->tuning = selected_tuning(capture->car);
	if (capture->debug) g_log("Model6 graph: tuning");
	capture->wheel_count = read_u32(capture->car + 0x2E8);
	capture->wheels = read_u32(capture->car + 0x2EC);
	if (capture->wheel_count != 4 || capture->wheels == 0)
		fail("Model6 capture requires exactly four wheels");
	capture->item = read_u32(capture->car + 0x28);
	if (capture->debug) g_log("Model6 graph: wheels and item");
	if (capture->item == 0 || read_u32(capture->item + 0x34) == 0)
		fail("Model6 car has no corpus");
	corpus_data = read_u32(capture->item + 0x38);
	if (corpus_data == 0)
		fail("Model6 corpus array is null");
	capture->corpus = read_u32(corpus_data);
	capture->dyna = read_u32(capture->corpus + 0x58);
	if (capture->dyna == 0)
		fail("Model6 corpus has no dynamics object");
	capture->params = read_u32(capture->dyna + 0x108);
	capture->state = read_u32(capture->dyna + 0x32C);
	if (capture->params == 0 || capture->state == 0)
		fail("Model6 dynamics graph is incomplete");
	if (capture->debug) g_log("Model6 graph: dynamics");
	capture->model_iso = resolve_model_iso(capture->corpus);
	capture->body_reference =
		read_u32(capture->item + 0x14) + 0x50;
	if (capture->model_iso == 0 || capture->body_reference == 0x50)
		fail("Model6 transform graph is incomplete");
	if (capture->debug) g_log("Model6 graph: transforms");
	capture->ground_id_count = read_u32(capture->car + 0x6C);
	capture->ground_ids = read_u32(capture->car + 0x70);
	manager = read_u32(capture->car + 0x68);
	if (manager == 0)
		fail("Model6 material manager is null");
	capture->ground_material_count = read_u32(manager + 0x14);
	capture->ground_material_ptrs = read_u32(manager + 0x18);
	if (capture->ground_id_count == 0 || capture->ground_ids == 0
	    || capture->ground_material_count == 0
	    || capture->ground_material_ptrs == 0)
		fail("Model6 material graph is incomplete");
	if (capture->ground_id_count > 1024
	    || capture->ground_material_count > 1024)
		fail("Model6 material graph is too large");
	if (capture->debug) g_log("Model6 graph: materials");
	for (i = 0; i < TMNF_MODEL6_CURVE_COUNT; ++i) {
		capture->curves[i] = read_u32(
			capture->tuning + CURVE_OFFSETS[i]);
		if (capture->curves[i] == 0)
			fail("Model6 tuning curve is null");
	}
	if (capture->debug) g_log("Model6 graph: curves");
	capture->tick = current_tick();
	if (capture->debug) g_log("Model6 graph: timer");
}

static void encode_arguments(
	struct Model6Capture *capture)
{
	uint32_t esp = capture->entry_esp;
	read_runtime(esp + 4, &capture->arguments.model_value, 4);
	read_runtime(read_u32(esp + 8), capture->arguments.existing_force, 12);
	read_runtime(esp + 12, &capture->arguments.lateral_force_factor, 4);
	read_runtime(esp + 16, &capture->arguments.longitudinal_force_factor, 4);
	read_runtime(read_u32(esp + 20), capture->arguments.local_speed, 12);
	read_runtime(
		read_u32(esp + 24), capture->arguments.local_angular_speed, 12);
	read_runtime(esp + 28, &capture->arguments.steering_angle, 4);
	capture->arguments.grounded = (int32_t)read_u32(esp + 32);
	read_runtime(read_u32(esp + 36), capture->arguments.material, 16);
	capture->sliding_ptr = read_u32(esp + 40);
	capture->brake_ptr = read_u32(esp + 44);
	read_runtime(capture->sliding_ptr, &capture->arguments.sliding, 4);
	read_runtime(capture->brake_ptr, &capture->arguments.brake_force, 4);
}

static uint8_t *serialize_graph(
	struct Model6Capture *capture, uint32_t phase, uint32_t *size)
{
	struct TmnfModel6TraceHeader header;
	struct ByteVec blob;
	struct TmnfModel6Curve descriptors[TMNF_MODEL6_CURVE_COUNT];
	uint32_t i;
	memset(&header, 0, sizeof(header));
	memset(&blob, 0, sizeof(blob));
	memset(descriptors, 0, sizeof(descriptors));
	memcpy(header.magic, TMNF_MODEL6_TRACE_MAGIC, 8);
	header.version = TMNF_MODEL6_TRACE_VERSION;
	header.phase = phase;
	header.car_id = map_id(&capture->ids, capture->car);
	header.tuning_id = map_id(&capture->ids, capture->tuning);
	header.wheels_id = map_id(&capture->ids, capture->wheels);
	header.item_id = map_id(&capture->ids, capture->item);
	header.corpus_id = map_id(&capture->ids, capture->corpus);
	header.dyna_id = map_id(&capture->ids, capture->dyna);
	header.params_id = map_id(&capture->ids, capture->params);
	header.state_id = map_id(&capture->ids, capture->state);
	header.model_iso_id = map_id(&capture->ids, capture->model_iso);
	header.body_reference_id =
		map_id(&capture->ids, capture->body_reference);
	header.wheel_count = capture->wheel_count;
	header.ground_id_count = capture->ground_id_count;
	header.ground_material_count = capture->ground_material_count;
	header.tick = capture->tick;
	memcpy(
		&header.model_value, &capture->arguments.model_value,
		offsetof(struct TmnfModel6TraceHeader, source_exe_sha256)
			- offsetof(struct TmnfModel6TraceHeader, model_value));
	if (phase != 0) {
		read_runtime(capture->sliding_ptr, &header.sliding, 4);
		read_runtime(capture->brake_ptr, &header.brake_force, 4);
	}
	memcpy(header.source_exe_sha256, TMNF_21126_EXE_SHA256, 32);
	memcpy(header.source_track_sha256, g_track_sha256, 32);
	if (capture->debug) g_log("Model6 serialize: arguments");
	vec_append(&blob, &header, sizeof(header));
	header.car_offset = vec_append_runtime(&blob, capture->car, 0x878);
	header.tuning_offset = vec_append_runtime(&blob, capture->tuning, 0x3AC);
	header.wheels_offset = vec_append_runtime(
		&blob, capture->wheels,
		capture->wheel_count * 0x2FC);
	header.dyna_offset = vec_append_runtime(&blob, capture->dyna, 0x344);
	header.params_offset = vec_append_runtime(&blob, capture->params, 0x5C);
	header.state_offset = vec_append_runtime(&blob, capture->state, 0xB4);
	if (capture->debug) g_log("Model6 serialize: raw graph");
	header.curves_offset = vec_append(
		&blob, descriptors, sizeof(descriptors));
	header.ground_ids_offset = vec_append_runtime(
		&blob, capture->ground_ids,
		capture->ground_id_count * 4);
	header.ground_materials_offset = blob.size;
	for (i = 0; i < capture->ground_material_count; ++i) {
		struct TmnfModel6Material material;
		uint32_t address = read_u32(
			capture->ground_material_ptrs + i * 4);
		if (address == 0)
			fail("Model6 ground material is null");
		material.object_id = map_id(&capture->ids, address);
		read_runtime(address + 0x14, material.values, 16);
		vec_append(&blob, &material, sizeof(material));
	}
	if (capture->debug) g_log("Model6 serialize: materials");
	header.model_iso_offset =
		vec_append_runtime(&blob, capture->model_iso, 0x30);
	header.body_reference_offset =
		vec_append_runtime(&blob, capture->body_reference, 0x0C);
	header.curve_data_offset = blob.size;
	for (i = 0; i < TMNF_MODEL6_CURVE_COUNT; ++i) {
		uint32_t curve = capture->curves[i];
		uint32_t position_count = read_u32(curve + 0x14);
		uint32_t positions = read_u32(curve + 0x18);
		uint32_t value_count = read_u32(curve + 0x20);
		uint32_t values = read_u32(curve + 0x24);
		if (position_count == 0 || position_count != value_count
		    || positions == 0 || values == 0 || position_count > 4096)
			fail("invalid Model6 curve buffers");
		descriptors[i].object_id = map_id(&capture->ids, curve);
		descriptors[i].positions_id = map_id(&capture->ids, positions);
		descriptors[i].values_id = map_id(&capture->ids, values);
		descriptors[i].count = position_count;
		descriptors[i].interpolation =
			(int32_t)read_u32(curve + 0x28);
		descriptors[i].positions_offset = vec_append_runtime(
			&blob, positions,
			position_count * 4);
		descriptors[i].values_offset = vec_append_runtime(
			&blob, values,
			position_count * 4);
	}
	if (capture->debug) g_log("Model6 serialize: curves");
	memcpy(blob.data + header.curves_offset,
	       descriptors, sizeof(descriptors));
	header.total_size = blob.size;
	memcpy(blob.data, &header, sizeof(header));
	*size = blob.size;
	return blob.data;
}

void model6_capture_initialize(
	HANDLE heap, uint32_t module_base, Model6CaptureFatalFn fatal_fn,
	Model6CaptureFatalFn log_fn)
{
	char track_sha256[65];
	uint32_t i;
	g_heap = heap;
	g_module_base = module_base;
	g_fatal = fatal_fn;
	g_log = log_fn;
	if (GetEnvironmentVariableA(
		    "TMNF_TRACK_SHA256", track_sha256,
		    sizeof(track_sha256)) != 64)
		fail("TMNF_TRACK_SHA256 must contain 64 hexadecimal characters");
	for (i = 0; i < 32; ++i) {
		unsigned int byte;
		if (sscanf(track_sha256 + i * 2, "%2x", &byte) != 1)
			fail("TMNF_TRACK_SHA256 is not hexadecimal");
		g_track_sha256[i] = (uint8_t)byte;
	}
}

struct Model6Capture *model6_capture_create(
	uint32_t this_ptr, uint32_t entry_esp)
{
	struct Model6Capture *capture = (struct Model6Capture *)HeapAlloc(
		g_heap, HEAP_ZERO_MEMORY, sizeof(*capture));
	if (capture == NULL)
		fail("Model6 capture allocation failed");
	capture->car = this_ptr;
	capture->entry_esp = entry_esp;
	capture->debug = InterlockedIncrement(&g_capture_count) == 1;
	discover_graph(capture);
	encode_arguments(capture);
	capture->input = serialize_graph(capture, 0, &capture->input_size);
	return capture;
}

const uint8_t *model6_capture_input(
	const struct Model6Capture *capture, uint32_t *size)
{
	*size = capture->input_size;
	return capture->input;
}

uint8_t *model6_capture_output(
	struct Model6Capture *capture, uint32_t *size)
{
	return serialize_graph(capture, 1, size);
}

void model6_capture_free_blob(uint8_t *blob)
{
	if (blob != NULL)
		HeapFree(g_heap, 0, blob);
}

void model6_capture_destroy(struct Model6Capture *capture)
{
	if (capture == NULL)
		return;
	if (capture->ids.entries != NULL)
		HeapFree(g_heap, 0, capture->ids.entries);
	if (capture->input != NULL)
		HeapFree(g_heap, 0, capture->input);
	HeapFree(g_heap, 0, capture);
}
