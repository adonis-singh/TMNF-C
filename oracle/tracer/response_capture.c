#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "response_capture.h"
#include "response_trace.h"

#define IMAGE_BASE 0x00400000u
#define SURFACE_MATERIAL_TABLE_VA 0x00D6EEC0u

struct ByteVec {
	uint8_t *data;
	uint32_t size;
	uint32_t capacity;
};

struct AddressEntry {
	uint32_t address;
	uint32_t id;
};

struct AddressMap {
	struct AddressEntry *entries;
	uint32_t count;
	uint32_t capacity;
};

struct ResponseCapture {
	uint32_t kind;
	uint32_t this_ptr;
	uint32_t physical;
	uint32_t contacts[2];
	struct AddressMap bodies;
	struct AddressMap trees;
	struct AddressMap materials;
	struct TmnfResponseContactEvent *events;
	uint32_t event_count;
	uint32_t event_capacity;
	uint8_t *input;
	uint32_t input_size;
};

static HANDLE g_heap;
static uint32_t g_module_base;
static ResponseCaptureFatalFn g_fatal;
static volatile LONG g_neg_probe_count;

static void fail(const char *message)
{
	g_fatal(message);
}

static uint32_t read_u32(uint32_t address)
{
	return *(const uint32_t *)(uintptr_t)address;
}

static void *heap_allocate(uint32_t size)
{
	void *memory = HeapAlloc(g_heap, HEAP_ZERO_MEMORY, size);
	if (memory == NULL)
		fail("response capture allocation failed");
	return memory;
}

static void *heap_resize(void *memory, uint32_t size)
{
	void *resized;
	if (memory == NULL)
		resized = HeapAlloc(g_heap, HEAP_ZERO_MEMORY, size);
	else
		resized = HeapReAlloc(g_heap, HEAP_ZERO_MEMORY, memory, size);
	if (resized == NULL)
		fail("response capture resize failed");
	return resized;
}

static void vec_reserve(struct ByteVec *vec, uint32_t extra)
{
	uint32_t required;
	uint32_t capacity;
	if (extra > UINT32_MAX - vec->size)
		fail("response trace blob overflow");
	required = vec->size + extra;
	if (required <= vec->capacity)
		return;
	capacity = vec->capacity == 0 ? 512 : vec->capacity;
	while (capacity < required) {
		if (capacity > UINT32_MAX / 2)
			fail("response trace capacity overflow");
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

static uint32_t map_find(const struct AddressMap *map, uint32_t address)
{
	uint32_t i;
	if (address == 0)
		return 0;
	for (i = 0; i < map->count; ++i) {
		if (map->entries[i].address == address)
			return map->entries[i].id;
	}
	return 0;
}

static uint32_t map_add(struct AddressMap *map, uint32_t address)
{
	uint32_t id;
	if (address == 0)
		return 0;
	id = map_find(map, address);
	if (id != 0)
		return id;
	if (map->count == map->capacity) {
		uint32_t capacity = map->capacity == 0 ? 8 : map->capacity * 2;
		if (capacity < map->capacity)
			fail("response address map overflow");
		map->entries = (struct AddressEntry *)heap_resize(
			map->entries, capacity * sizeof(*map->entries));
		map->capacity = capacity;
	}
	id = map->count + 1;
	map->entries[map->count].address = address;
	map->entries[map->count].id = id;
	++map->count;
	return id;
}

static uint32_t map_require(
	const struct AddressMap *map, uint32_t address, const char *role)
{
	uint32_t id;
	char message[128];
	if (address == 0)
		return 0;
	id = map_find(map, address);
	if (id != 0)
		return id;
	_snprintf(
		message, sizeof(message), "unknown response %s reference %08X",
		role, address);
	fail(message);
	return 0;
}

static uint32_t static_va(uint32_t address)
{
	if (address < g_module_base)
		fail("response callback target precedes module");
	return address - g_module_base + IMAGE_BASE;
}

static uint32_t body_item(uint32_t corpus)
{
	uint32_t item = read_u32(corpus + 0x48);
	if (item == 0)
		fail("response corpus has no item");
	return item;
}

static void discover_body(struct ResponseCapture *capture, uint32_t corpus)
{
	uint32_t item;
	uint32_t model;
	uint32_t sink_ref;
	uint32_t sink;
	uint32_t target;
	uint32_t dyna;
	if (corpus == 0)
		fail("null response corpus");
	if (map_find(&capture->bodies, corpus) != 0)
		return;
	map_add(&capture->bodies, corpus);
	item = body_item(corpus);
	model = read_u32(item + 0x14);
	if (model == 0)
		fail("response item has no response model");
	(void)read_u32(model + 0x18);
	sink_ref = read_u32(item + 0x24);
	if (sink_ref != 0) {
		sink = read_u32(sink_ref + 0x08);
		if (sink != 0) {
			uint32_t vtable = read_u32(sink);
			if (vtable == 0)
				fail("response sink has no vtable");
			target = static_va(read_u32(vtable + 0x0c));
			if (target != 0x007B39F0u && target != 0x0047CBA0u)
				fail("unknown response AbsorbContact target");
		}
	}
	dyna = read_u32(corpus + 0x58);
	if (dyna != 0) {
		uint32_t count = read_u32(dyna + 0x330);
		uint32_t data = read_u32(dyna + 0x334);
		uint32_t capacity = read_u32(dyna + 0x338);
		if (read_u32(dyna + 0x108) == 0
		    || read_u32(dyna + 0x32c) == 0)
			fail("response dyna graph is incomplete");
		if (count > capacity || (count != 0 && data == 0))
			fail("invalid response replacement buffer");
	}
}

static void discover_collision(
	struct ResponseCapture *capture, uint32_t collision)
{
	uint32_t material;
	discover_body(capture, read_u32(collision + 0x00));
	map_add(&capture->trees, read_u32(collision + 0x04));
	discover_body(capture, read_u32(collision + 0x08));
	map_add(&capture->trees, read_u32(collision + 0x0c));
	material = read_u32(collision + 0x48);
	if (material == 0)
		fail("physical collision has no response material");
	map_add(&capture->materials, material);
}

static void discover_contact(
	struct ResponseCapture *capture, uint32_t contact)
{
	if (contact == 0)
		return;
	discover_body(capture, read_u32(contact + 0x00));
	map_add(&capture->trees, read_u32(contact + 0x04));
	discover_body(capture, read_u32(contact + 0x40));
	map_add(&capture->trees, read_u32(contact + 0x44));
}

static uint32_t collision_count(const struct ResponseCapture *capture)
{
	if (capture->kind == TMNF_RESPONSE_SOLVE_IMPULSE)
		return 1;
	return read_u32(capture->this_ptr + 0x15c);
}

static uint32_t collision_data(const struct ResponseCapture *capture)
{
	if (capture->kind == TMNF_RESPONSE_SOLVE_IMPULSE)
		return capture->physical;
	return read_u32(capture->this_ptr + 0x160);
}

static void encode_collision(
	const struct ResponseCapture *capture, uint32_t source,
	struct TmnfResponseCollision *output)
{
	memset(output, 0, sizeof(*output));
	output->body1_id = map_require(
		&capture->bodies, read_u32(source + 0x00), "body");
	output->tree1_id = map_require(
		&capture->trees, read_u32(source + 0x04), "tree");
	output->body2_id = map_require(
		&capture->bodies, read_u32(source + 0x08), "body");
	output->tree2_id = map_require(
		&capture->trees, read_u32(source + 0x0c), "tree");
	memcpy(output->collision, (const void *)(uintptr_t)(source + 0x10), 0x38);
	output->material_id = map_require(
		&capture->materials, read_u32(source + 0x48), "material");
}

static uint32_t contact_target(uint32_t item)
{
	uint32_t sink_ref = read_u32(item + 0x24);
	uint32_t sink;
	uint32_t vtable;
	if (sink_ref == 0)
		return 0;
	sink = read_u32(sink_ref + 0x08);
	if (sink == 0)
		return 0;
	vtable = read_u32(sink);
	if (vtable == 0)
		fail("response sink has no vtable");
	return static_va(read_u32(vtable + 0x0c));
}

static void encode_body(
	const struct ResponseCapture *capture, uint32_t index,
	struct TmnfResponseBody *output, struct ByteVec *replacements)
{
	uint32_t corpus = capture->bodies.entries[index].address;
	uint32_t item = body_item(corpus);
	uint32_t model = read_u32(item + 0x14);
	uint32_t dyna = read_u32(corpus + 0x58);
	uint32_t target;
	memset(output, 0, sizeof(*output));
	output->id = capture->bodies.entries[index].id;
	output->classification_flags = read_u32(item + 0x18);
	output->response_flags = read_u32(item + 0x1c);
	memcpy(&output->response_weight, (const void *)(uintptr_t)(model + 0x18), 4);
	memcpy(output->iso, (const void *)(uintptr_t)(corpus + 0x18), 0x30);
	target = contact_target(item);
	output->has_contact_sink = target != 0;
	output->contact_target_va = target;
	if (dyna != 0) {
		uint32_t count = read_u32(dyna + 0x330);
		uint32_t data = read_u32(dyna + 0x334);
		uint32_t capacity = read_u32(dyna + 0x338);
		uint32_t params = read_u32(dyna + 0x108);
		uint32_t state = read_u32(dyna + 0x32c);
		if (count > capacity || (count != 0 && data == 0)
		    || params == 0 || state == 0)
			fail("invalid response dyna output graph");
		output->dyna_present = 1;
		output->dirty_flag = read_u32(dyna + 0x33c);
		output->mode = (int32_t)read_u32(dyna + 0x340);
		output->replacement_index = replacements->size / 0x0c;
		output->replacement_count = count;
		output->replacement_capacity = capacity;
		memcpy(output->params, (const void *)(uintptr_t)params, 0x44);
		memcpy(output->state, (const void *)(uintptr_t)state, 0xb4);
		if (count != 0)
			vec_append(
				replacements, (const void *)(uintptr_t)data,
				count * 0x0c);
	}
}

static void encode_contact(
	const struct ResponseCapture *capture, uint32_t source,
	struct TmnfResponseContact *output)
{
	memset(output, 0, sizeof(*output));
	if (source == 0)
		return;
	output->present = 1;
	output->body_id = map_require(
		&capture->bodies, read_u32(source + 0x00), "contact body");
	output->tree_id = map_require(
		&capture->trees, read_u32(source + 0x04), "contact tree");
	memcpy(&output->surface_material, (const void *)(uintptr_t)(source + 0x08), 2);
	memcpy(output->normal, (const void *)(uintptr_t)(source + 0x0c), 0x0c);
	memcpy(output->position, (const void *)(uintptr_t)(source + 0x18), 0x0c);
	memcpy(output->relative_speed, (const void *)(uintptr_t)(source + 0x24), 0x0c);
	memcpy(output->replacement, (const void *)(uintptr_t)(source + 0x30), 0x0c);
	output->accepted = read_u32(source + 0x3c);
	output->other_body_id = map_require(
		&capture->bodies, read_u32(source + 0x40), "other contact body");
	output->other_tree_id = map_require(
		&capture->trees, read_u32(source + 0x44), "other contact tree");
	memcpy(
		&output->other_surface_material,
		(const void *)(uintptr_t)(source + 0x48), 2);
}

static void probe_collision_neg(uint32_t physical)
{
	typedef void (__attribute__((thiscall)) *NegFn)(void *collision);
	uint8_t collision[0x38];
	NegFn function = (NegFn)(uintptr_t)(
		g_module_base + 0x00547E00u - IMAGE_BASE);
	memcpy(collision, (const void *)(uintptr_t)(physical + 0x10), 0x38);
	function(collision);
}

static uint32_t body_id_from_item(
	const struct ResponseCapture *capture, uint32_t item)
{
	uint32_t i;
	for (i = 0; i < capture->bodies.count; ++i) {
		if (body_item(capture->bodies.entries[i].address) == item)
			return capture->bodies.entries[i].id;
	}
	fail("unknown response callback item");
	return 0;
}

static uint8_t *serialize_graph(
	struct ResponseCapture *capture, int include_events, uint32_t *size)
{
	struct TmnfResponseTraceHeader header;
	struct ByteVec blob;
	struct ByteVec replacements;
	uint32_t count;
	uint32_t data;
	uint32_t i;
	memset(&header, 0, sizeof(header));
	memset(&blob, 0, sizeof(blob));
	memset(&replacements, 0, sizeof(replacements));
	memcpy(header.magic, TMNF_RESPONSE_TRACE_MAGIC, 8);
	header.version = TMNF_RESPONSE_TRACE_VERSION;
	header.kind = capture->kind;
	header.body_count = capture->bodies.count;
	header.material_count = capture->materials.count;
	header.tree_count = capture->trees.count;
	header.contact_count =
		capture->kind == TMNF_RESPONSE_SOLVE_IMPULSE ? 2 : 0;
	header.event_count = include_events ? capture->event_count : 0;
	vec_append(&blob, &header, sizeof(header));

	count = collision_count(capture);
	data = collision_data(capture);
	if (count != 0 && data == 0)
		fail("response collision buffer has no data");
	header.collision_count = count;
	header.collisions_offset = blob.size;
	for (i = 0; i < count; ++i) {
		struct TmnfResponseCollision collision;
		encode_collision(capture, data + i * 0x4c, &collision);
		vec_append(&blob, &collision, sizeof(collision));
	}

	header.bodies_offset = blob.size;
	for (i = 0; i < capture->bodies.count; ++i) {
		struct TmnfResponseBody body;
		encode_body(capture, i, &body, &replacements);
		vec_append(&blob, &body, sizeof(body));
	}

	header.materials_offset = blob.size;
	for (i = 0; i < capture->materials.count; ++i) {
		struct TmnfResponseMaterial material;
		material.id = capture->materials.entries[i].id;
		memcpy(
			material.bytes,
			(const void *)(uintptr_t)capture->materials.entries[i].address,
			sizeof(material.bytes));
		vec_append(&blob, &material, sizeof(material));
	}

	header.contacts_offset = blob.size;
	if (header.contact_count != 0) {
		for (i = 0; i < 2; ++i) {
			struct TmnfResponseContact contact;
			encode_contact(capture, capture->contacts[i], &contact);
			vec_append(&blob, &contact, sizeof(contact));
		}
	}

	header.events_offset = blob.size;
	if (header.event_count != 0)
		vec_append(
			&blob, capture->events,
			header.event_count * sizeof(*capture->events));

	header.replacements_offset = blob.size;
	header.replacement_count = replacements.size / 0x0c;
	if (replacements.size != 0)
		vec_append(&blob, replacements.data, replacements.size);

	header.surface_materials_offset = blob.size;
	vec_append(
		&blob,
		(const void *)(uintptr_t)(
			g_module_base + SURFACE_MATERIAL_TABLE_VA - IMAGE_BASE),
		TMNF_RESPONSE_SURFACE_MATERIAL_COUNT * 8);
	header.total_size = blob.size;
	memcpy(blob.data, &header, sizeof(header));
	if (replacements.data != NULL)
		HeapFree(g_heap, 0, replacements.data);
	*size = blob.size;
	return blob.data;
}

void response_capture_initialize(
	HANDLE heap, uint32_t module_base, ResponseCaptureFatalFn fatal_fn)
{
	g_heap = heap;
	g_module_base = module_base;
	g_fatal = fatal_fn;
}

struct ResponseCapture *response_capture_create(
	uint32_t kind, uint32_t this_ptr, uint32_t entry_esp)
{
	struct ResponseCapture *capture =
		(struct ResponseCapture *)heap_allocate(sizeof(*capture));
	uint32_t count;
	uint32_t data;
	uint32_t i;
	capture->kind = kind;
	capture->this_ptr = this_ptr;
	if (kind == TMNF_RESPONSE_SOLVE_IMPULSE) {
		capture->physical = read_u32(entry_esp + 4);
		capture->contacts[0] = read_u32(entry_esp + 8);
		capture->contacts[1] = read_u32(entry_esp + 12);
		if (capture->physical == 0)
			fail("SolveImpulse received null collision");
	} else if (kind != TMNF_RESPONSE_COMPUTE_COLLISION_RESPONSE) {
		fail("unknown response capture kind");
	}
	count = collision_count(capture);
	data = collision_data(capture);
	if (count != 0 && data == 0)
		fail("response collision buffer has no data");
	for (i = 0; i < count; ++i)
		discover_collision(capture, data + i * 0x4c);
	if (kind == TMNF_RESPONSE_COMPUTE_COLLISION_RESPONSE && count != 0
	    && InterlockedIncrement(&g_neg_probe_count) <= 2000)
		probe_collision_neg(data);
	if (kind == TMNF_RESPONSE_SOLVE_IMPULSE) {
		discover_contact(capture, capture->contacts[0]);
		discover_contact(capture, capture->contacts[1]);
	}
	capture->input = serialize_graph(capture, 0, &capture->input_size);
	return capture;
}

const uint8_t *response_capture_input(
	const struct ResponseCapture *capture, uint32_t *size)
{
	*size = capture->input_size;
	return capture->input;
}

uint8_t *response_capture_output(
	struct ResponseCapture *capture, uint32_t *size)
{
	return serialize_graph(capture, 1, size);
}

uint32_t response_capture_event_enter(
	struct ResponseCapture *capture, uint32_t target_va,
	uint32_t item, uint32_t contact)
{
	struct TmnfResponseContactEvent *event;
	uint32_t index;
	if (contact == 0)
		fail("AbsorbContact received null contact");
	if (capture->event_count == capture->event_capacity) {
		uint32_t capacity =
			capture->event_capacity == 0 ? 8 : capture->event_capacity * 2;
		if (capacity < capture->event_capacity)
			fail("response event buffer overflow");
		capture->events = (struct TmnfResponseContactEvent *)heap_resize(
			capture->events, capacity * sizeof(*capture->events));
		capture->event_capacity = capacity;
	}
	index = capture->event_count++;
	event = &capture->events[index];
	memset(event, 0, sizeof(*event));
	event->target_va = target_va;
	event->item_body_id = body_id_from_item(capture, item);
	encode_contact(capture, contact, &event->input);
	return index;
}

void response_capture_event_exit(
	struct ResponseCapture *capture, uint32_t event_index,
	uint32_t contact)
{
	if (event_index >= capture->event_count)
		fail("response callback event index is invalid");
	encode_contact(capture, contact, &capture->events[event_index].output);
}

void response_capture_free_blob(uint8_t *blob)
{
	if (blob != NULL)
		HeapFree(g_heap, 0, blob);
}

void response_capture_destroy(struct ResponseCapture *capture)
{
	if (capture == NULL)
		return;
	if (capture->bodies.entries != NULL)
		HeapFree(g_heap, 0, capture->bodies.entries);
	if (capture->trees.entries != NULL)
		HeapFree(g_heap, 0, capture->trees.entries);
	if (capture->materials.entries != NULL)
		HeapFree(g_heap, 0, capture->materials.entries);
	if (capture->events != NULL)
		HeapFree(g_heap, 0, capture->events);
	if (capture->input != NULL)
		HeapFree(g_heap, 0, capture->input);
	HeapFree(g_heap, 0, capture);
}
