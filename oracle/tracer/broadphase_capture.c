#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "broadphase_capture.h"
#include "broadphase_trace.h"

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

struct AddressList {
	uint32_t *addresses;
	uint32_t count;
	uint32_t capacity;
};

struct BroadphaseCapture {
	uint32_t kind;
	uint32_t root;
	uint32_t arguments[4];
	struct AddressMap ids;
	struct AddressList groups;
	struct AddressList corpus_groups;
	struct AddressList device_groups;
	struct AddressList static_groups;
	struct AddressList corpora;
	struct AddressList dynas;
	struct AddressList states;
	struct AddressList speeds;
	struct AddressList devices;
	struct AddressList tables;
	struct AddressList zones;
	struct AddressList trees;
	struct AddressList isos;
	struct AddressList plugs;
	struct AddressList surfaces;
	struct AddressList buffers;
	struct AddressList contact_buffers;
	uint8_t *input;
	uint32_t input_size;
};

static HANDLE g_heap;
static BroadphaseCaptureFatalFn g_fatal;
static const char *g_stage = "initialization";

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
			"broadphase read failed stage=%s address=%08X size=%u",
			g_stage, address, size);
		fail(message);
	}
}

static uint32_t read_u32(uint32_t address)
{
	uint32_t value;
	read_runtime(address, &value, sizeof(value));
	return value;
}

static uint32_t read_u32_named(uint32_t address, const char *role)
{
	uint32_t value;
	SIZE_T copied = 0;
	char message[160];
	if (address == 0
	    || !ReadProcessMemory(
		    GetCurrentProcess(), (const void *)(uintptr_t)address,
		    &value, sizeof(value), &copied)
	    || copied != sizeof(value)) {
		_snprintf(message, sizeof(message),
			"broadphase %s read failed address=%08X",
			role, address);
		fail(message);
	}
	return value;
}

static uint8_t read_u8_named(uint32_t address, const char *role)
{
	uint8_t value;
	SIZE_T copied = 0;
	char message[160];
	if (address == 0
	    || !ReadProcessMemory(
		    GetCurrentProcess(), (const void *)(uintptr_t)address,
		    &value, sizeof(value), &copied)
	    || copied != sizeof(value)) {
		_snprintf(message, sizeof(message),
			"broadphase %s read failed address=%08X",
			role, address);
		fail(message);
	}
	return value;
}

static void *heap_resize(void *memory, uint32_t size)
{
	void *result = memory == NULL
		? HeapAlloc(g_heap, HEAP_ZERO_MEMORY, size)
		: HeapReAlloc(g_heap, HEAP_ZERO_MEMORY, memory, size);
	if (result == NULL)
		fail("broadphase capture allocation failed");
	return result;
}

static void vec_reserve(struct ByteVec *vec, uint32_t extra)
{
	uint32_t required;
	uint32_t capacity;
	if (extra > UINT32_MAX - vec->size)
		fail("broadphase trace size overflow");
	required = vec->size + extra;
	if (required <= vec->capacity)
		return;
	capacity = vec->capacity == 0 ? 1024 : vec->capacity;
	while (capacity < required) {
		if (capacity > UINT32_MAX / 2)
			fail("broadphase trace capacity overflow");
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

static uint32_t map_find(
	const struct AddressMap *map, uint32_t address)
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
		uint32_t capacity = map->capacity == 0 ? 32 : map->capacity * 2;
		if (capacity < map->capacity)
			fail("broadphase identity map overflow");
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
	uint32_t id = map_find(map, address);
	char message[128];
	if (address == 0 || id != 0)
		return id;
	_snprintf(message, sizeof(message),
		"unknown broadphase %s address %08X", role, address);
	fail(message);
	return 0;
}

static int list_contains(
	const struct AddressList *list, uint32_t address)
{
	uint32_t i;
	for (i = 0; i < list->count; ++i) {
		if (list->addresses[i] == address)
			return 1;
	}
	return 0;
}

static void list_add(
	struct AddressList *list, struct AddressMap *ids, uint32_t address)
{
	if (address == 0)
		fail("null broadphase graph object");
	map_add(ids, address);
	if (list_contains(list, address))
		return;
	if (list->count == list->capacity) {
		uint32_t capacity = list->capacity == 0 ? 8 : list->capacity * 2;
		if (capacity < list->capacity)
			fail("broadphase address list overflow");
		list->addresses = (uint32_t *)heap_resize(
			list->addresses, capacity * sizeof(*list->addresses));
		list->capacity = capacity;
	}
	list->addresses[list->count++] = address;
}

static void validate_buffer(
	uint32_t count, uint32_t data, uint32_t capacity,
	uint32_t limit, const char *role)
{
	char message[128];
	if (count > capacity || capacity > limit
	    || (capacity != 0 && data == 0)) {
		_snprintf(message, sizeof(message),
			"invalid broadphase %s count=%u capacity=%u",
			role, count, capacity);
		fail(message);
	}
}

static void discover_speed(
	struct BroadphaseCapture *capture, uint32_t group)
{
	uint32_t count = read_u32(group + 0x18);
	uint32_t data = read_u32(group + 0x1c);
	uint32_t capacity = read_u32(group + 0x20);
	validate_buffer(count, data, capacity, 100000u, "speed buffer");
	if (data != 0)
		list_add(&capture->speeds, &capture->ids, data);
}

static void discover_tree(
	struct BroadphaseCapture *capture, uint32_t tree);

static void discover_corpus(
	struct BroadphaseCapture *capture, uint32_t corpus, int full)
{
	uint32_t dyna;
	uint32_t state;
	uint32_t scene;
	uint32_t model;
	uint32_t tree;
	list_add(&capture->corpora, &capture->ids, corpus);
	dyna = read_u32(corpus + 0x58);
	if (dyna != 0) {
		list_add(&capture->dynas, &capture->ids, dyna);
		state = read_u32(dyna + 0x32c);
		if (state == 0)
			fail("broadphase dyna has no live state");
		list_add(&capture->states, &capture->ids, state);
	}
	scene = read_u32(corpus + 0x48);
	if (scene == 0) {
		if (!full)
			return;
		fail("broadphase corpus has no scene object");
	}
	map_add(&capture->ids, scene);
	model = read_u32(scene + 0x14);
	if (model == 0) {
		if (!full)
			return;
		fail("broadphase corpus has no model");
	}
	map_add(&capture->ids, model);
	if (!full)
		return;
	tree = read_u32(model + 0x64);
	discover_tree(capture, tree);
}

static void discover_corpora(
	struct BroadphaseCapture *capture, uint32_t group)
{
	uint32_t count = read_u32(group + 0x0c);
	uint32_t data = read_u32(group + 0x10);
	uint32_t capacity = read_u32(group + 0x14);
	uint32_t i;
	validate_buffer(count, data, capacity, 100000u, "corpus buffer");
	list_add(&capture->corpus_groups, &capture->ids, group);
	if (data != 0)
		map_add(&capture->ids, data);
	for (i = 0; i < count; ++i) {
		uint32_t corpus = read_u32(data + i * 4);
		discover_corpus(capture, corpus, 0);
	}
}

static void discover_corpora_full(
	struct BroadphaseCapture *capture, uint32_t group)
{
	uint32_t count = read_u32(group + 0x0c);
	uint32_t data = read_u32(group + 0x10);
	uint32_t capacity = read_u32(group + 0x14);
	uint32_t i;
	validate_buffer(count, data, capacity, 100000u, "corpus buffer");
	list_add(&capture->corpus_groups, &capture->ids, group);
	if (data != 0)
		map_add(&capture->ids, data);
	for (i = 0; i < count; ++i)
		discover_corpus(capture, read_u32(data + i * 4), 1);
}

static void discover_group(
	struct BroadphaseCapture *capture, uint32_t group)
{
	uint32_t pointer;
	list_add(&capture->groups, &capture->ids, group);
	pointer = read_u32(group + 0x10);
	if (pointer != 0)
		map_add(&capture->ids, pointer);
	pointer = read_u32(group + 0x1c);
	if (pointer != 0)
		map_add(&capture->ids, pointer);
	pointer = read_u32(group + 0x28);
	if (pointer != 0)
		map_add(&capture->ids, pointer);
	pointer = read_u32(group + 0x34);
	if (pointer != 0)
		map_add(&capture->ids, pointer);
	discover_speed(capture, group);
}

static void discover_devices(
	struct BroadphaseCapture *capture, uint32_t group)
{
	uint32_t count = read_u32(group + 0x24);
	uint32_t data = read_u32(group + 0x28);
	uint32_t i;
	if (count > 100000u || (count != 0 && data == 0))
		fail("invalid broadphase device array");
	list_add(&capture->device_groups, &capture->ids, group);
	if (data != 0)
		map_add(&capture->ids, data);
	for (i = 0; i < count; ++i) {
		uint32_t device = data + i * 0x1c;
		uint32_t other = read_u32(device);
		uint32_t material = read_u32(device + 4);
		uint32_t table = device + 8;
		uint32_t rows = read_u32(table + 4);
		uint32_t columns = read_u32(table + 8);
		uint32_t stride = read_u32(table + 0x10);
		uint32_t values = read_u32(table);
		uint64_t value_count = (uint64_t)rows * stride;
		if (other == 0)
			fail("broadphase device has no group");
		if (columns > stride || value_count > 1000000u
		    || (value_count != 0 && values == 0))
			fail("invalid broadphase perform table");
		list_add(&capture->devices, &capture->ids, device);
		if (material != 0)
			map_add(&capture->ids, material);
		list_add(&capture->tables, &capture->ids, table);
		if (values != 0)
			map_add(&capture->ids, values);
		discover_group(capture, other);
	}
}

static void discover_record_refs(
	struct BroadphaseCapture *capture, uint32_t data, uint32_t count)
{
	g_stage = "discover collision references";
	uint32_t i;
	for (i = 0; i < count; ++i) {
		uint32_t record = data + i * 0x4c;
		uint32_t offsets[] = { 0x00, 0x04, 0x08, 0x0c, 0x48 };
		uint32_t j;
		for (j = 0; j < sizeof(offsets) / sizeof(offsets[0]); ++j) {
			uint32_t pointer = read_u32(record + offsets[j]);
			if (pointer != 0)
				map_add(&capture->ids, pointer);
		}
	}
}

static void discover_buffer(
	struct BroadphaseCapture *capture, uint32_t buffer, int has_active)
{
	uint32_t count;
	uint32_t data;
	uint32_t capacity;
	if (buffer == 0)
		return;
	g_stage = "discover collision buffer";
	list_add(&capture->buffers, &capture->ids, buffer);
	if (has_active)
		list_add(&capture->contact_buffers, &capture->ids, buffer);
	count = read_u32(buffer + 4);
	data = read_u32(buffer + 8);
	capacity = read_u32(buffer + 0x0c);
	validate_buffer(
		count, data, capacity, 100000u, "collision buffer");
	if (data != 0)
		map_add(&capture->ids, data);
	discover_record_refs(capture, data, count);
}

static void discover_surface(
	struct BroadphaseCapture *capture, uint32_t surface)
{
	g_stage = "discover surface";
	if (surface == 0)
		return;
	list_add(&capture->surfaces, &capture->ids, surface);
}

static void discover_plug(
	struct BroadphaseCapture *capture, uint32_t plug)
{
	uint32_t wrapper;
	uint32_t surface;
	uint32_t count;
	uint32_t data;
	uint32_t capacity;
	uint32_t i;
	if (plug == 0)
		return;
	g_stage = "discover plug surface";
	if (list_contains(&capture->plugs, plug))
		return;
	list_add(&capture->plugs, &capture->ids, plug);
	wrapper = read_u32_named(plug + 0x14, "plug wrapper");
	if (wrapper == 0)
		fail("broadphase plug has no geometry wrapper");
	map_add(&capture->ids, wrapper);
	surface = read_u32_named(wrapper + 0x34, "plug geometry");
	if (surface == 0)
		fail("broadphase plug has no geometry");
	discover_surface(capture, surface);
	count = read_u32(plug + 0x18);
	data = read_u32(plug + 0x1c);
	capacity = read_u32(plug + 0x20);
	validate_buffer(count, data, capacity, 65536u, "plug material buffer");
	if (data != 0)
		map_add(&capture->ids, data);
	for (i = 0; i < count; ++i) {
		uint32_t material = read_u32_named(
			data + i * 4, "plug material pointer");
		if (material == 0)
			fail("broadphase plug has null material");
		map_add(&capture->ids, material);
	}
}

static uint32_t tree_child_count(uint32_t tree)
{
	typedef uint32_t (__attribute__((thiscall)) *ChildCountFn)(uint32_t);
	uint32_t vtable;
	ChildCountFn function;
	g_stage = "read tree child count";
	vtable = read_u32_named(tree, "tree vtable");
	if (vtable == 0)
		fail("broadphase tree has no vtable");
	function = (ChildCountFn)(uintptr_t)read_u32_named(
		vtable + 0x7c, "tree child-count method");
	if (function == NULL)
		fail("broadphase tree has no child-count method");
	return function(tree);
}

static uint32_t tree_child_at(uint32_t tree, uint32_t index)
{
	typedef uint32_t (__attribute__((thiscall)) *ChildAtFn)(
		uint32_t, uint32_t);
	uint32_t vtable;
	ChildAtFn function;
	g_stage = "read tree child";
	vtable = read_u32_named(tree, "tree vtable");
	if (vtable == 0)
		fail("broadphase tree has no vtable");
	function = (ChildAtFn)(uintptr_t)read_u32_named(
		vtable + 0x80, "tree child method");
	if (function == NULL)
		fail("broadphase tree has no child method");
	return function(tree, index);
}

static void discover_tree(
	struct BroadphaseCapture *capture, uint32_t tree)
{
	uint32_t count;
	uint32_t i;
	uint32_t plug;
	uint32_t buffer;
	if (tree == 0)
		fail("null broadphase input tree");
	g_stage = "discover tree";
	if (list_contains(&capture->trees, tree))
		return;
	list_add(&capture->trees, &capture->ids, tree);
	plug = read_u32(tree + 0x8c);
	discover_plug(capture, plug);
	buffer = read_u32(tree + 0x50);
	discover_buffer(capture, buffer, 1);
	count = tree_child_count(tree);
	if (count > 100000u)
		fail("invalid broadphase tree child count");
	for (i = 0; i < count; ++i)
		discover_tree(capture, tree_child_at(tree, i));
}

static void discover_static_group(
	struct BroadphaseCapture *capture, uint32_t group)
{
	uint32_t count;
	uint32_t data;
	uint32_t capacity;
	uint32_t i;
	if (group == 0)
		fail("broadphase zone has no static group");
	g_stage = "discover static group";
	discover_group(capture, group);
	list_add(&capture->static_groups, &capture->ids, group);
	count = read_u32(group + 0x30);
	data = read_u32(group + 0x34);
	capacity = read_u32(group + 0x38);
	validate_buffer(count, data, capacity, 1000000u, "static entry buffer");
	if (data != 0)
		map_add(&capture->ids, data);
	for (i = 0; i < count; ++i) {
		g_stage = "discover static entry";
		uint32_t entry = data + i * 0x58;
		uint32_t plug = read_u32(entry + 0x4c);
		uint32_t tree = read_u32(entry + 0x50);
		uint32_t corpus = read_u32(entry + 0x54);
		map_add(&capture->ids, entry);
		discover_plug(capture, plug);
		if (tree != 0)
			map_add(&capture->ids, tree);
		if (corpus != 0)
			map_add(&capture->ids, corpus);
	}
}

static void discover_zone(
	struct BroadphaseCapture *capture, uint32_t zone, int deep_static)
{
	uint32_t pointer;
	uint32_t count;
	uint32_t data;
	uint32_t capacity;
	uint32_t i;
	g_stage = "discover zone";
	list_add(&capture->zones, &capture->ids, zone);
	pointer = read_u32(zone + 0x184);
	if (pointer != 0)
		map_add(&capture->ids, pointer);
	pointer = read_u32(zone + 0x188);
	if (pointer != 0)
		map_add(&capture->ids, pointer);
	pointer = read_u32(zone + 0x18c);
	if (pointer != 0)
		map_add(&capture->ids, pointer);
	pointer = read_u32(zone + 0x190);
	discover_buffer(capture, pointer, 0);
	pointer = read_u32(zone + 0x194);
	if (pointer != 0) {
		if (deep_static)
			discover_static_group(capture, pointer);
		else
			discover_group(capture, pointer);
	}
	pointer = read_u32(zone + 0x19c);
	if (pointer != 0)
		map_add(&capture->ids, pointer);
	count = read_u32(zone + 0x1a0);
	data = read_u32(zone + 0x1a4);
	capacity = read_u32(zone + 0x1a8);
	validate_buffer(count, data, capacity, 100000u, "merge buffer list");
	if (data != 0)
		map_add(&capture->ids, data);
	for (i = 0; i < count; ++i)
		discover_buffer(capture, read_u32(data + i * 4), 1);
}

static void refresh_output_graph(struct BroadphaseCapture *capture)
{
	uint32_t i;
	g_stage = "refresh output graph";
	for (i = 0; i < capture->trees.count; ++i) {
		uint32_t buffer = read_u32(capture->trees.addresses[i] + 0x50);
		discover_buffer(capture, buffer, 1);
	}
	for (i = 0; i < capture->zones.count; ++i) {
		uint32_t zone = capture->zones.addresses[i];
		uint32_t count = read_u32(zone + 0x1a0);
		uint32_t data = read_u32(zone + 0x1a4);
		uint32_t capacity = read_u32(zone + 0x1a8);
		uint32_t j;
		validate_buffer(
			count, data, capacity, 100000u, "merge buffer list");
		if (data != 0)
			map_add(&capture->ids, data);
		for (j = 0; j < count; ++j)
			discover_buffer(capture, read_u32(data + j * 4), 1);
		discover_buffer(capture, read_u32(zone + 0x190), 0);
	}
}

static int group_has_deep_section(
	const struct AddressList *groups, uint32_t group)
{
	return list_contains(groups, group);
}

static void normalize_pointer(
	const struct BroadphaseCapture *capture,
	uint8_t *bytes, uint32_t offset, const char *role)
{
	uint32_t address;
	uint32_t id;
	memcpy(&address, bytes + offset, 4);
	id = map_require(&capture->ids, address, role);
	memcpy(bytes + offset, &id, 4);
}

static void append_surface(
	const struct BroadphaseCapture *capture,
	uint32_t address, struct ByteVec *surfaces,
	struct ByteVec *vertices, struct ByteVec *faces,
	struct ByteVec *nodes)
{
	struct TmnfDetectSurface surface;
	uint32_t count;
	uint32_t data;
	memset(&surface, 0, sizeof(surface));
	g_stage = "serialize surface";
	surface.id = map_require(&capture->ids, address, "surface");
	surface.material_index = (uint16_t)read_u32(address + 4);
	read_runtime(address + 6, &surface.type, 1);
	read_runtime(address + 7, &surface.reserved, 1);
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
			fail("invalid broadphase mesh vertex buffer");
		surface.vertex_index = vertices->size / 0x0c;
		surface.vertex_count = count;
		if (count != 0) {
			vec_reserve(vertices, count * 0x0c);
			read_runtime(
				data, vertices->data + vertices->size,
				count * 0x0c);
			vertices->size += count * 0x0c;
		}
		count = read_u32(address + 0x10);
		data = read_u32(address + 0x14);
		if (count > 1000000u || (count != 0 && data == 0))
			fail("invalid broadphase mesh face buffer");
		surface.face_index = faces->size / 0x20;
		surface.face_count = count;
		if (count != 0) {
			vec_reserve(faces, count * 0x20);
			read_runtime(
				data, faces->data + faces->size,
				count * 0x20);
			faces->size += count * 0x20;
		}
		count = read_u32(address + 0x20);
		data = read_u32(address + 0x24);
		if (count > read_u32(address + 0x28) || count > 1000000u
		    || (count != 0 && data == 0))
			fail("invalid broadphase mesh node buffer");
		surface.node_index = nodes->size / 0x20;
		surface.node_count = count;
		if (count != 0) {
			vec_reserve(nodes, count * 0x20);
			read_runtime(
				data, nodes->data + nodes->size,
				count * 0x20);
			nodes->size += count * 0x20;
		}
	} else if (surface.type >= 9) {
		fail("unsupported broadphase surface type");
	}
	vec_append(surfaces, &surface, sizeof(surface));
}

static void append_buffer(
	const struct BroadphaseCapture *capture,
	uint32_t address, struct ByteVec *buffers,
	struct ByteVec *records)
{
	struct TmnfDetectBuffer buffer;
	uint32_t data;
	uint32_t i;
	memset(&buffer, 0, sizeof(buffer));
	g_stage = "serialize collision buffer";
	buffer.id = map_require(&capture->ids, address, "collision buffer");
	buffer.count = read_u32(address + 4);
	data = read_u32(address + 8);
	buffer.capacity = read_u32(address + 0x0c);
	validate_buffer(
		buffer.count, data, buffer.capacity,
		100000u, "collision buffer");
	buffer.record_index = records->size / 0x4c;
	buffer.has_active =
		list_contains(&capture->contact_buffers, address);
	if (buffer.has_active)
		buffer.active = read_u32(address + 0x10);
	for (i = 0; i < buffer.count; ++i) {
		uint8_t encoded[0x4c];
		uint32_t offsets[] = { 0x00, 0x04, 0x08, 0x0c, 0x48 };
		uint32_t j;
		read_runtime(data + i * 0x4c, encoded, sizeof(encoded));
		for (j = 0; j < sizeof(offsets) / sizeof(offsets[0]); ++j) {
			uint32_t pointer;
			uint32_t id;
			memcpy(&pointer, encoded + offsets[j], 4);
			id = map_require(
				&capture->ids, pointer, "collision reference");
			memcpy(encoded + offsets[j], &id, 4);
		}
		vec_append(records, encoded, sizeof(encoded));
	}
	vec_append(buffers, &buffer, sizeof(buffer));
}

static uint8_t *serialize(
	struct BroadphaseCapture *capture, uint32_t return_value,
	int output, uint32_t *size)
{
	struct TmnfBroadphaseTraceHeader header;
	struct ByteVec blob = { 0 };
	struct ByteVec groups = { 0 };
	struct ByteVec corpus_ids = { 0 };
	struct ByteVec corpora = { 0 };
	struct ByteVec dynas = { 0 };
	struct ByteVec states = { 0 };
	struct ByteVec speeds = { 0 };
	struct ByteVec speed_values = { 0 };
	struct ByteVec devices = { 0 };
	struct ByteVec tables = { 0 };
	struct ByteVec table_values = { 0 };
	struct ByteVec zones = { 0 };
	struct ByteVec static_entries = { 0 };
	struct ByteVec trees = { 0 };
	struct ByteVec tree_child_ids = { 0 };
	struct ByteVec isos = { 0 };
	struct ByteVec plugs = { 0 };
	struct ByteVec surfaces = { 0 };
	struct ByteVec vertices = { 0 };
	struct ByteVec faces = { 0 };
	struct ByteVec nodes = { 0 };
	struct ByteVec material_ids = { 0 };
	struct ByteVec buffers = { 0 };
	struct ByteVec collision_records = { 0 };
	struct ByteVec merge_buffer_ids = { 0 };
	uint32_t i;

	if (output)
		refresh_output_graph(capture);
	g_stage = "serialize header";
	memset(&header, 0, sizeof(header));
	memcpy(header.magic, TMNF_BROADPHASE_TRACE_MAGIC, 8);
	header.version = TMNF_BROADPHASE_TRACE_VERSION;
	header.kind = capture->kind;
	header.return_value = return_value;
	header.root_id = map_require(&capture->ids, capture->root, "root");
	for (i = 0; i < 4; ++i)
		header.argument_ids[i] = map_require(
			&capture->ids, capture->arguments[i], "argument");

	for (i = 0; i < capture->groups.count; ++i) {
		g_stage = "serialize group";
		uint32_t group_address = capture->groups.addresses[i];
		struct TmnfBroadphaseGroup group;
		uint32_t count;
		uint32_t data;
		uint32_t capacity;
		uint32_t j;
		memset(&group, 0, sizeof(group));
		group.zone_index = UINT32_MAX;
		group.id = map_require(&capture->ids, group_address, "group");
		if ((capture->kind == TMNF_BROADPHASE_PREPARE
		     || capture->kind == TMNF_BROADPHASE_DETECT_CORPUS)
		    && group_address >= capture->root
		    && group_address < capture->root + 5 * 0x44
		    && (group_address - capture->root) % 0x44 == 0)
			group.zone_index =
				(group_address - capture->root) / 0x44;
		read_runtime(group_address, group.bytes, sizeof(group.bytes));
		normalize_pointer(capture, group.bytes, 0x10, "corpus array");
		normalize_pointer(capture, group.bytes, 0x1c, "speed array");
		normalize_pointer(capture, group.bytes, 0x28, "device array");
		normalize_pointer(capture, group.bytes, 0x34, "static array");

		if (group_has_deep_section(
			    &capture->corpus_groups, group_address)) {
			count = read_u32(group_address + 0x0c);
			data = read_u32(group_address + 0x10);
			capacity = read_u32(group_address + 0x14);
			validate_buffer(
				count, data, capacity, 100000u, "corpus buffer");
			group.corpus_array_id =
				map_require(&capture->ids, data, "corpus array");
			group.corpus_index = corpus_ids.size / 4;
			group.corpus_count = count;
			group.corpus_capacity = capacity;
			for (j = 0; j < count; ++j) {
				uint32_t id = map_require(
					&capture->ids, read_u32(data + j * 4),
					"corpus");
				vec_append(&corpus_ids, &id, 4);
			}
		}

		data = read_u32(group_address + 0x1c);
		group.speed_id =
			map_require(&capture->ids, data, "speed array");
		if (group_has_deep_section(
			    &capture->device_groups, group_address)) {
			count = read_u32(group_address + 0x24);
			data = read_u32(group_address + 0x28);
			if (count > 100000u || (count != 0 && data == 0))
				fail("invalid broadphase output device array");
			group.device_array_id =
				map_require(&capture->ids, data, "device array");
			group.device_index =
				devices.size / sizeof(struct TmnfBroadphaseDevice);
			group.device_count = count;
			for (j = 0; j < count; ++j) {
				uint32_t address = data + j * 0x1c;
				struct TmnfBroadphaseDevice device;
				memset(&device, 0, sizeof(device));
				device.id = map_require(
					&capture->ids, address, "device");
				device.group_id = map_require(
					&capture->ids, read_u32(address),
					"device group");
				device.table_id = map_require(
					&capture->ids, address + 8, "table");
				read_runtime(address, device.bytes,
					sizeof(device.bytes));
				normalize_pointer(
					capture, device.bytes, 0,
					"device group");
				normalize_pointer(
					capture, device.bytes, 4,
					"device material");
				normalize_pointer(
					capture, device.bytes, 8,
					"table values");
				vec_append(&devices, &device, sizeof(device));
			}
		}
		if (group_has_deep_section(
			    &capture->static_groups, group_address)) {
			count = read_u32(group_address + 0x30);
			data = read_u32(group_address + 0x34);
			capacity = read_u32(group_address + 0x38);
			validate_buffer(
				count, data, capacity, 1000000u,
				"static entry buffer");
			group.static_array_id =
				map_require(&capture->ids, data, "static array");
			group.static_entry_index =
				static_entries.size
				/ sizeof(struct TmnfBroadphaseStaticEntry);
			group.static_entry_count = count;
			group.static_entry_capacity = capacity;
			for (j = 0; j < count; ++j) {
				uint32_t address = data + j * 0x58;
				struct TmnfBroadphaseStaticEntry entry;
				memset(&entry, 0, sizeof(entry));
				entry.id = map_require(
					&capture->ids, address, "static entry");
				entry.skip_count = read_u32(address);
				read_runtime(address + 4, entry.box,
					sizeof(entry.box));
				read_runtime(address + 0x1c, entry.iso,
					sizeof(entry.iso));
				if (read_u32(address + 0x4c) != 0
				    && read_u32(address + 0x50) != 0)
					entry.tree_flags = read_u32(
						read_u32(address + 0x50) + 0x9c);
				entry.surface_id = map_require(
					&capture->ids, read_u32(address + 0x4c),
					"static surface");
				entry.tree_id = map_require(
					&capture->ids, read_u32(address + 0x50),
					"static tree");
				entry.corpus_id = map_require(
					&capture->ids, read_u32(address + 0x54),
					"static corpus");
				vec_append(
					&static_entries, &entry, sizeof(entry));
			}
		}
		vec_append(&groups, &group, sizeof(group));
	}

	for (i = 0; i < capture->corpora.count; ++i) {
		g_stage = "serialize corpus";
		uint32_t address = capture->corpora.addresses[i];
		struct TmnfBroadphaseCorpus corpus;
		uint32_t scene;
		uint32_t model;
		uint32_t tree;
		memset(&corpus, 0, sizeof(corpus));
		corpus.id = map_require(&capture->ids, address, "corpus");
		read_runtime(address, corpus.bytes, sizeof(corpus.bytes));
		corpus.dyna_id = map_require(
			&capture->ids, read_u32(address + 0x58), "dyna");
		scene = read_u32(address + 0x48);
		if (scene != 0) {
			uint32_t scene_id = map_require(
				&capture->ids, scene, "corpus scene");
			corpus.flags = read_u32(scene + 0x18);
			model = read_u32(scene + 0x14);
			tree = model == 0 ? 0 : read_u32(model + 0x64);
			corpus.tree_id = map_find(&capture->ids, tree);
			memcpy(corpus.bytes + 0x48, &scene_id, 4);
		}
		memcpy(corpus.bytes + 0x58, &corpus.dyna_id, 4);
		vec_append(&corpora, &corpus, sizeof(corpus));
	}
	for (i = 0; i < capture->dynas.count; ++i) {
		g_stage = "serialize dyna";
		uint32_t address = capture->dynas.addresses[i];
		struct TmnfBroadphaseDyna dyna;
		memset(&dyna, 0, sizeof(dyna));
		dyna.id = map_require(&capture->ids, address, "dyna");
		read_runtime(address, dyna.bytes, sizeof(dyna.bytes));
		dyna.state_id = map_require(
			&capture->ids, read_u32(address + 0x32c), "state");
		memcpy(dyna.bytes + 0x32c, &dyna.state_id, 4);
		vec_append(&dynas, &dyna, sizeof(dyna));
	}
	for (i = 0; i < capture->states.count; ++i) {
		g_stage = "serialize dyna state";
		uint32_t address = capture->states.addresses[i];
		struct TmnfBroadphaseState state;
		state.id = map_require(&capture->ids, address, "state");
		read_runtime(address, state.bytes, sizeof(state.bytes));
		vec_append(&states, &state, sizeof(state));
	}
	for (i = 0; i < capture->speeds.count; ++i) {
		g_stage = "serialize speed array";
		uint32_t address = capture->speeds.addresses[i];
		struct TmnfBroadphaseSpeed speed;
		uint32_t group_address = 0;
		uint32_t j;
		for (j = 0; j < capture->groups.count; ++j) {
			uint32_t candidate = capture->groups.addresses[j];
			if (read_u32(candidate + 0x1c) == address) {
				group_address = candidate;
				break;
			}
		}
		if (group_address == 0)
			fail("orphan broadphase speed array");
		speed.id = map_require(&capture->ids, address, "speed array");
		speed.count = read_u32(group_address + 0x18);
		speed.capacity = read_u32(group_address + 0x20);
		validate_buffer(
			speed.count, address, speed.capacity,
			100000u, "speed buffer");
		speed.value_index = speed_values.size / 4;
		if (speed.count != 0) {
			vec_reserve(&speed_values, speed.count * 4);
			read_runtime(address,
				speed_values.data + speed_values.size,
				speed.count * 4);
			speed_values.size += speed.count * 4;
		}
		vec_append(&speeds, &speed, sizeof(speed));
	}
	for (i = 0; i < capture->tables.count; ++i) {
		g_stage = "serialize perform table";
		uint32_t address = capture->tables.addresses[i];
		struct TmnfBroadphaseTable table;
		uint32_t data = read_u32(address);
		uint64_t value_count;
		memset(&table, 0, sizeof(table));
		table.id = map_require(&capture->ids, address, "table");
		table.rows = read_u32(address + 4);
		table.columns = read_u32(address + 8);
		table.reserved = read_u32(address + 0x0c);
		table.stride = read_u32(address + 0x10);
		value_count = (uint64_t)table.rows * table.stride;
		if (table.columns > table.stride || value_count > 1000000u
		    || (value_count != 0 && data == 0))
			fail("invalid broadphase output table");
		map_require(&capture->ids, data, "table values");
		table.value_index = table_values.size / 4;
		table.value_count = (uint32_t)value_count;
		if (table.value_count != 0) {
			vec_reserve(&table_values, table.value_count * 4);
			read_runtime(data,
				table_values.data + table_values.size,
				table.value_count * 4);
			table_values.size += table.value_count * 4;
		}
		vec_append(&tables, &table, sizeof(table));
	}
	for (i = 0; i < capture->zones.count; ++i) {
		g_stage = "serialize zone";
		uint32_t address = capture->zones.addresses[i];
		struct TmnfBroadphaseZone zone;
		uint32_t count;
		uint32_t data;
		uint32_t capacity;
		uint32_t j;
		memset(&zone, 0, sizeof(zone));
		zone.id = map_require(&capture->ids, address, "zone");
		zone.current_material_id = map_require(
			&capture->ids, read_u32(address + 0x184),
			"current material");
		zone.current_corpus1_id = map_require(
			&capture->ids, read_u32(address + 0x188),
			"current corpus one");
		zone.current_corpus2_id = map_require(
			&capture->ids, read_u32(address + 0x18c),
			"current corpus two");
		zone.general_buffer_id = map_require(
			&capture->ids, read_u32(address + 0x190),
			"general buffer");
		zone.static_group_id = map_require(
			&capture->ids, read_u32(address + 0x194),
			"static group");
		count = read_u32(address + 0x1a0);
		data = read_u32(address + 0x1a4);
		capacity = read_u32(address + 0x1a8);
		validate_buffer(
			count, data, capacity, 100000u, "merge buffer list");
		zone.merge_array_id = map_require(
			&capture->ids, data, "merge buffer array");
		zone.merge_buffer_index = merge_buffer_ids.size / 4;
		zone.merge_buffer_count = count;
		zone.merge_buffer_capacity = capacity;
		for (j = 0; j < count; ++j) {
			uint32_t id = map_require(
				&capture->ids, read_u32(data + j * 4),
				"merge buffer");
			vec_append(&merge_buffer_ids, &id, 4);
		}
		read_runtime(address, zone.bytes, sizeof(zone.bytes));
		normalize_pointer(capture, zone.bytes, 0x184,
			"current material");
		normalize_pointer(capture, zone.bytes, 0x188,
			"current corpus one");
		normalize_pointer(capture, zone.bytes, 0x18c,
			"current corpus two");
		normalize_pointer(capture, zone.bytes, 0x190,
			"general buffer");
		normalize_pointer(capture, zone.bytes, 0x194,
			"static group");
		normalize_pointer(capture, zone.bytes, 0x19c,
			"zone dependency");
		normalize_pointer(capture, zone.bytes, 0x1a4,
			"merge buffer array");
		vec_append(&zones, &zone, sizeof(zone));
	}
	for (i = 0; i < capture->trees.count; ++i) {
		g_stage = "serialize tree";
		uint32_t address = capture->trees.addresses[i];
		struct TmnfBroadphaseTree tree;
		uint32_t count = tree_child_count(address);
		uint32_t j;
		memset(&tree, 0, sizeof(tree));
		if (count > 100000u)
			fail("invalid broadphase output tree child count");
		tree.id = map_require(&capture->ids, address, "tree");
		tree.flags = read_u32(address + 0x9c);
		tree.surface_id = map_require(
			&capture->ids, read_u32(address + 0x8c),
			"tree surface");
		tree.buffer_id = map_require(
			&capture->ids, read_u32(address + 0x50),
			"tree contact buffer");
		tree.child_index = tree_child_ids.size / 4;
		tree.child_count = count;
		for (j = 0; j < count; ++j) {
			uint32_t id = map_require(
				&capture->ids, tree_child_at(address, j),
				"tree child");
			vec_append(&tree_child_ids, &id, 4);
		}
		read_runtime(address + 0x34, tree.box, sizeof(tree.box));
		read_runtime(
			address + 0x5c, tree.local_iso, sizeof(tree.local_iso));
		vec_append(&trees, &tree, sizeof(tree));
	}
	for (i = 0; i < capture->isos.count; ++i) {
		g_stage = "serialize iso";
		uint32_t address = capture->isos.addresses[i];
		struct TmnfBroadphaseIso iso;
		iso.id = map_require(&capture->ids, address, "iso");
		read_runtime(address, iso.bytes, sizeof(iso.bytes));
		vec_append(&isos, &iso, sizeof(iso));
	}
	for (i = 0; i < capture->plugs.count; ++i) {
		g_stage = "serialize plug";
		uint32_t address = capture->plugs.addresses[i];
		uint32_t wrapper = read_u32(address + 0x14);
		uint32_t count = read_u32(address + 0x18);
		uint32_t data = read_u32(address + 0x1c);
		uint32_t capacity = read_u32(address + 0x20);
		struct TmnfBroadphasePlug plug;
		uint32_t j;
		validate_buffer(
			count, data, capacity, 65536u, "plug material buffer");
		plug.id = map_require(&capture->ids, address, "plug");
		plug.surface_id = map_require(
			&capture->ids, read_u32(wrapper + 0x34),
			"plug geometry");
		plug.material_index = material_ids.size;
		plug.material_count = count;
		for (j = 0; j < count; ++j) {
			uint32_t material = read_u32(data + j * 4);
			uint8_t material_id = read_u8_named(
				material + 0x18, "plug material id");
			vec_append(&material_ids, &material_id, 1);
		}
		vec_append(&plugs, &plug, sizeof(plug));
	}
	for (i = 0; i < capture->surfaces.count; ++i)
		append_surface(
			capture, capture->surfaces.addresses[i],
			&surfaces, &vertices, &faces, &nodes);
	for (i = 0; i < capture->buffers.count; ++i)
		append_buffer(
			capture, capture->buffers.addresses[i],
			&buffers, &collision_records);

	vec_append(&blob, &header, sizeof(header));
#define APPEND_SECTION(field_, count_field_, vector_, stride_) \
	do { \
		header.field_ = blob.size; \
		header.count_field_ = (vector_).size / (stride_); \
		if ((vector_).size != 0) \
			vec_append(&blob, (vector_).data, (vector_).size); \
	} while (0)
	APPEND_SECTION(groups_offset, group_count, groups,
		sizeof(struct TmnfBroadphaseGroup));
	APPEND_SECTION(corpus_ids_offset, corpus_id_count, corpus_ids, 4);
	APPEND_SECTION(corpora_offset, corpus_count, corpora,
		sizeof(struct TmnfBroadphaseCorpus));
	APPEND_SECTION(dynas_offset, dyna_count, dynas,
		sizeof(struct TmnfBroadphaseDyna));
	APPEND_SECTION(states_offset, state_count, states,
		sizeof(struct TmnfBroadphaseState));
	APPEND_SECTION(speeds_offset, speed_count, speeds,
		sizeof(struct TmnfBroadphaseSpeed));
	APPEND_SECTION(speed_values_offset, speed_value_count, speed_values, 4);
	APPEND_SECTION(devices_offset, device_count, devices,
		sizeof(struct TmnfBroadphaseDevice));
	APPEND_SECTION(tables_offset, table_count, tables,
		sizeof(struct TmnfBroadphaseTable));
	APPEND_SECTION(table_values_offset, table_value_count, table_values, 4);
	APPEND_SECTION(zones_offset, zone_count, zones,
		sizeof(struct TmnfBroadphaseZone));
	APPEND_SECTION(static_entries_offset, static_entry_count, static_entries,
		sizeof(struct TmnfBroadphaseStaticEntry));
	APPEND_SECTION(trees_offset, tree_count, trees,
		sizeof(struct TmnfBroadphaseTree));
	APPEND_SECTION(tree_child_ids_offset, tree_child_id_count,
		tree_child_ids, 4);
	APPEND_SECTION(isos_offset, iso_count, isos,
		sizeof(struct TmnfBroadphaseIso));
	APPEND_SECTION(plugs_offset, plug_count, plugs,
		sizeof(struct TmnfBroadphasePlug));
	APPEND_SECTION(surfaces_offset, surface_count, surfaces,
		sizeof(struct TmnfDetectSurface));
	APPEND_SECTION(vertices_offset, vertex_count, vertices, 0x0c);
	APPEND_SECTION(faces_offset, face_count, faces, 0x20);
	APPEND_SECTION(nodes_offset, node_count, nodes, 0x20);
	APPEND_SECTION(material_ids_offset, material_id_count, material_ids, 1);
	APPEND_SECTION(buffers_offset, buffer_count, buffers,
		sizeof(struct TmnfDetectBuffer));
	APPEND_SECTION(collision_records_offset, collision_record_count,
		collision_records, 0x4c);
	APPEND_SECTION(merge_buffer_ids_offset, merge_buffer_id_count,
		merge_buffer_ids, 4);
#undef APPEND_SECTION
	header.total_size = blob.size;
	memcpy(blob.data, &header, sizeof(header));

#define FREE_VEC(vector_) \
	do { \
		if ((vector_).data != NULL) \
			HeapFree(g_heap, 0, (vector_).data); \
	} while (0)
	FREE_VEC(groups);
	FREE_VEC(corpus_ids);
	FREE_VEC(corpora);
	FREE_VEC(dynas);
	FREE_VEC(states);
	FREE_VEC(speeds);
	FREE_VEC(speed_values);
	FREE_VEC(devices);
	FREE_VEC(tables);
	FREE_VEC(table_values);
	FREE_VEC(zones);
	FREE_VEC(static_entries);
	FREE_VEC(trees);
	FREE_VEC(tree_child_ids);
	FREE_VEC(isos);
	FREE_VEC(plugs);
	FREE_VEC(surfaces);
	FREE_VEC(vertices);
	FREE_VEC(faces);
	FREE_VEC(nodes);
	FREE_VEC(material_ids);
	FREE_VEC(buffers);
	FREE_VEC(collision_records);
	FREE_VEC(merge_buffer_ids);
#undef FREE_VEC
	*size = blob.size;
	return blob.data;
}

void broadphase_capture_initialize(
	HANDLE heap, BroadphaseCaptureFatalFn fatal_fn)
{
	g_heap = heap;
	g_fatal = fatal_fn;
}

int broadphase_capture_is_target(uint32_t va)
{
	return va == 0x00537E80u || va == 0x00537F30u
		|| va == 0x0053A0E0u || va == 0x0053A120u
		|| va == 0x0053A8F0u || va == 0x0053AFB0u
		|| va == 0x0053B1C0u;
}

struct BroadphaseCapture *broadphase_capture_create(
	uint32_t va, uint32_t this_ptr, uint32_t entry_esp)
{
	struct BroadphaseCapture *capture;
	capture = (struct BroadphaseCapture *)HeapAlloc(
		g_heap, HEAP_ZERO_MEMORY, sizeof(*capture));
	if (capture == NULL)
		fail("broadphase capture allocation failed");
	capture->root = this_ptr;
	switch (va) {
	case 0x00537E80u:
		capture->kind = TMNF_BROADPHASE_GROUP_SPEEDS;
		discover_group(capture, this_ptr);
		discover_corpora(capture, this_ptr);
		break;
	case 0x00537F30u:
		capture->kind = TMNF_BROADPHASE_GROUP_PERFORM;
		discover_group(capture, this_ptr);
		discover_devices(capture, this_ptr);
		break;
	case 0x0053A0E0u:
		capture->kind = TMNF_BROADPHASE_PREPARE;
		for (uint32_t i = 0; i < 5; ++i) {
			uint32_t group = this_ptr + i * 0x44;
			discover_group(capture, group);
			discover_corpora(capture, group);
			discover_devices(capture, group);
		}
		break;
	case 0x0053A120u:
		capture->kind = TMNF_BROADPHASE_TREE_STATIC;
		capture->arguments[0] = read_u32(entry_esp + 4);
		capture->arguments[1] = read_u32(entry_esp + 8);
		if (capture->arguments[0] == 0 || capture->arguments[1] == 0)
			fail("null broadphase static argument");
		list_add(
			&capture->isos, &capture->ids, capture->arguments[0]);
		discover_tree(capture, capture->arguments[1]);
		discover_zone(capture, this_ptr, 1);
		break;
	case 0x0053A8F0u: {
		uint32_t pair = read_u32(entry_esp + 4);
		capture->kind = TMNF_BROADPHASE_COMPUTE_COLLISION;
		if (pair == 0)
			fail("null broadphase tree pair");
		capture->arguments[0] = read_u32(pair);
		capture->arguments[1] = read_u32(pair + 4);
		capture->arguments[2] = read_u32(pair + 8);
		capture->arguments[3] = read_u32(pair + 0x0c);
		discover_tree(capture, capture->arguments[0]);
		list_add(
			&capture->isos, &capture->ids, capture->arguments[1]);
		discover_tree(capture, capture->arguments[2]);
		list_add(
			&capture->isos, &capture->ids, capture->arguments[3]);
		discover_zone(capture, this_ptr, 0);
		break;
	}
	case 0x0053AFB0u:
		capture->kind = TMNF_BROADPHASE_DETECT_BETWEEN;
		capture->arguments[0] = read_u32(entry_esp + 4);
		capture->arguments[1] = read_u32(entry_esp + 8);
		discover_corpus(capture, capture->arguments[0], 1);
		discover_corpus(capture, capture->arguments[1], 1);
		discover_zone(capture, this_ptr, 0);
		break;
	case 0x0053B1C0u:
		capture->kind = TMNF_BROADPHASE_DETECT_CORPUS;
		capture->arguments[0] = read_u32(entry_esp + 4);
		capture->arguments[1] = read_u32(entry_esp + 8);
		discover_buffer(capture, capture->arguments[0], 0);
		for (uint32_t i = 0; i < 5; ++i) {
			uint32_t group = this_ptr + i * 0x44;
			discover_group(capture, group);
			discover_corpora_full(capture, group);
			discover_devices(capture, group);
			discover_static_group(capture, group);
		}
		discover_corpus(capture, capture->arguments[1], 1);
		discover_zone(capture, this_ptr, 0);
		break;
	default:
		fail("unsupported broadphase capture target");
	}
	capture->input = serialize(capture, 0, 0, &capture->input_size);
	return capture;
}

const uint8_t *broadphase_capture_input(
	const struct BroadphaseCapture *capture, uint32_t *size)
{
	*size = capture->input_size;
	return capture->input;
}

uint8_t *broadphase_capture_output(
	struct BroadphaseCapture *capture, uint32_t return_value, uint32_t *size)
{
	return serialize(capture, return_value, 1, size);
}

void broadphase_capture_free_blob(uint8_t *blob)
{
	if (blob != NULL)
		HeapFree(g_heap, 0, blob);
}

void broadphase_capture_destroy(struct BroadphaseCapture *capture)
{
	if (capture == NULL)
		return;
#define FREE_LIST(list_) \
	do { \
		if ((list_).addresses != NULL) \
			HeapFree(g_heap, 0, (list_).addresses); \
	} while (0)
	FREE_LIST(capture->groups);
	FREE_LIST(capture->corpus_groups);
	FREE_LIST(capture->device_groups);
	FREE_LIST(capture->static_groups);
	FREE_LIST(capture->corpora);
	FREE_LIST(capture->dynas);
	FREE_LIST(capture->states);
	FREE_LIST(capture->speeds);
	FREE_LIST(capture->devices);
	FREE_LIST(capture->tables);
	FREE_LIST(capture->zones);
	FREE_LIST(capture->trees);
	FREE_LIST(capture->isos);
	FREE_LIST(capture->plugs);
	FREE_LIST(capture->surfaces);
	FREE_LIST(capture->buffers);
	FREE_LIST(capture->contact_buffers);
#undef FREE_LIST
	if (capture->ids.entries != NULL)
		HeapFree(g_heap, 0, capture->ids.entries);
	if (capture->input != NULL)
		HeapFree(g_heap, 0, capture->input);
	HeapFree(g_heap, 0, capture);
}
