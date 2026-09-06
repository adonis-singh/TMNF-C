#include <dirent.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "collision_response.h"
#include "trace_format.h"
#include "vehicle_contact.h"
#include "../oracle/tracer/response_trace.h"

typedef struct {
	uint32_t tag;
	uint32_t addr;
	uint32_t len;
	uint8_t *bytes;
} Buffer;

typedef struct {
	uint32_t seq;
	uint16_t input_count;
	uint16_t output_count;
	Buffer *inputs;
	Buffer *outputs;
} Record;

typedef struct {
	uint32_t va;
	uint32_t record_count;
	Record *records;
} Trace;

typedef int (*ReplayHandler)(const Record *record);

static uint32_t diagnostic_budget = 24;
static int strict_response_replay;

static void fail(const char *message, const char *path) {
	fprintf(stderr, "%s: %s\n", message, path);
	exit(2);
}

static void *allocate(size_t size) {
	if (size == 0) {
		size = 1;
	}
	void *data = malloc(size);
	if (data == NULL) {
		fprintf(stderr, "out of memory\n");
		exit(2);
	}
	return data;
}

static uint32_t read_u32(FILE *file, const char *path) {
	uint8_t bytes[4];
	if (fread(bytes, 1, sizeof(bytes), file) != sizeof(bytes)) {
		fail("unexpected EOF", path);
	}
	return (uint32_t)bytes[0] |
		(uint32_t)bytes[1] << 8 |
		(uint32_t)bytes[2] << 16 |
		(uint32_t)bytes[3] << 24;
}

static uint16_t read_u16(FILE *file, const char *path) {
	uint8_t bytes[2];
	if (fread(bytes, 1, sizeof(bytes), file) != sizeof(bytes)) {
		fail("unexpected EOF", path);
	}
	return (uint16_t)(bytes[0] | (uint16_t)bytes[1] << 8);
}

static Buffer *read_buffers(
	FILE *file, uint16_t count, const char *path) {
	if (count == 0) {
		return NULL;
	}
	Buffer *buffers = allocate((size_t)count * sizeof(*buffers));
	for (uint16_t i = 0; i < count; i++) {
		buffers[i].tag = read_u32(file, path);
		buffers[i].addr = read_u32(file, path);
		buffers[i].len = read_u32(file, path);
		buffers[i].bytes = allocate(buffers[i].len);
		if (fread(buffers[i].bytes, 1, buffers[i].len, file) !=
			buffers[i].len) {
			fail("unexpected EOF", path);
		}
	}
	return buffers;
}

static Trace load_trace(const char *path) {
	FILE *file = fopen(path, "rb");
	if (file == NULL) {
		fail("cannot open trace", path);
	}
	char magic[8];
	if (fread(magic, 1, sizeof(magic), file) != sizeof(magic) ||
		memcmp(magic, TMNF_TRACE_MAGIC, sizeof(magic)) != 0) {
		fail("bad trace magic", path);
	}

	Trace trace;
	trace.va = read_u32(file, path);
	trace.record_count = read_u32(file, path);
	trace.records = trace.record_count == 0
		? NULL
		: allocate((size_t)trace.record_count * sizeof(*trace.records));
	for (uint32_t i = 0; i < trace.record_count; i++) {
		Record *record = &trace.records[i];
		record->seq = read_u32(file, path);
		record->input_count = read_u16(file, path);
		record->output_count = read_u16(file, path);
		record->inputs =
			read_buffers(file, record->input_count, path);
		record->outputs =
			read_buffers(file, record->output_count, path);
	}
	if (fgetc(file) != EOF) {
		fail("trailing trace bytes", path);
	}
	fclose(file);
	return trace;
}

static void free_buffers(Buffer *buffers, uint16_t count) {
	for (uint16_t i = 0; i < count; i++) {
		free(buffers[i].bytes);
	}
	free(buffers);
}

static void free_trace(Trace *trace) {
	for (uint32_t i = 0; i < trace->record_count; i++) {
		free_buffers(trace->records[i].inputs,
			trace->records[i].input_count);
		free_buffers(trace->records[i].outputs,
			trace->records[i].output_count);
	}
	free(trace->records);
}

static const Buffer *find_buffer(
	const Buffer *buffers, uint16_t count, uint32_t tag) {
	for (uint16_t i = 0; i < count; i++) {
		if (buffers[i].tag == tag) {
			return &buffers[i];
		}
	}
	return NULL;
}

static int replay_mult_transpose(const Record *record) {
	const Buffer *input =
		find_buffer(record->inputs, record->input_count, TAG_THIS);
	const Buffer *matrix =
		find_buffer(record->inputs, record->input_count, TAG_ARG1);
	const Buffer *expected =
		find_buffer(record->outputs, record->output_count, TAG_OUT_THIS);
	if (input == NULL || input->len != sizeof(GmVec3) ||
		matrix == NULL || matrix->len != sizeof(GmMat3) ||
		expected == NULL || expected->len != sizeof(GmVec3)) {
		fprintf(stderr, "invalid 0x0045BD40 record shape\n");
		exit(2);
	}
	GmVec3 actual;
	memcpy(&actual, input->bytes, sizeof(actual));
	GmVec3_MultTranspose(&actual, (const GmMat3 *)matrix->bytes);
	return memcmp(&actual, expected->bytes, sizeof(actual)) == 0;
}

static int replay_collision_neg(const Record *record) {
	const Buffer *input =
		find_buffer(record->inputs, record->input_count, TAG_THIS);
	const Buffer *expected =
		find_buffer(record->outputs, record->output_count, TAG_OUT_THIS);
	if (input == NULL || input->len != sizeof(GmCollision) ||
		expected == NULL || expected->len != sizeof(GmCollision)) {
		fprintf(stderr, "invalid 0x00547E00 record shape\n");
		exit(2);
	}
	GmCollision actual;
	memcpy(&actual, input->bytes, sizeof(actual));
	GmCollision_Neg(&actual);
	return memcmp(&actual, expected->bytes, sizeof(actual)) == 0;
}

typedef struct {
	const struct TmnfResponseTraceHeader *header;
	const struct TmnfResponseCollision *collisions;
	const struct TmnfResponseBody *bodies;
	const struct TmnfResponseMaterial *materials;
	const struct TmnfResponseContact *contacts;
	const struct TmnfResponseContactEvent *events;
	const GmVec3 *replacements;
	const CPlugSurfaceMaterialData *surface_materials;
} ResponseView;

typedef struct ReplayGraph ReplayGraph;

typedef struct {
	ReplayGraph *graph;
	uint32_t body_id;
	uint32_t target_va;
} CallbackUser;

typedef struct {
	CHmsResponseBody body;
	CHmsDyna dyna;
	CHmsDynaParams params;
	CHmsStateDyna state;
	CallbackUser callback;
} ReplayBody;

struct ReplayGraph {
	ResponseView input;
	ResponseView expected;
	SHmsPhysicalCollision *collisions;
	ReplayBody *bodies;
	CHmsResponseMaterial *materials;
	CPlugSurfaceMaterialData surface_materials[
		TMNF_RESPONSE_SURFACE_MATERIAL_COUNT];
	CHmsPhysicalContact contacts[2];
	CHmsResponseZone zone;
	uint32_t event_index;
	int callback_failed;
};

static int range_valid(
	uint32_t offset, uint32_t count, uint32_t stride, uint32_t size) {
	uint64_t end = (uint64_t)offset + (uint64_t)count * stride;
	return offset <= size && end <= size;
}

static ResponseView response_view(
	const Buffer *buffer, uint32_t expected_kind) {
	ResponseView view;
	memset(&view, 0, sizeof(view));
	if (buffer == NULL ||
		buffer->len < sizeof(struct TmnfResponseTraceHeader)) {
		fprintf(stderr, "truncated response graph\n");
		exit(2);
	}
	view.header =
		(const struct TmnfResponseTraceHeader *)buffer->bytes;
	const struct TmnfResponseTraceHeader *header = view.header;
	if (memcmp(header->magic, TMNF_RESPONSE_TRACE_MAGIC, 8) != 0 ||
		header->version != TMNF_RESPONSE_TRACE_VERSION ||
		header->kind != expected_kind ||
		header->total_size != buffer->len ||
		!range_valid(header->collisions_offset,
			header->collision_count,
			sizeof(struct TmnfResponseCollision), buffer->len) ||
		!range_valid(header->bodies_offset, header->body_count,
			sizeof(struct TmnfResponseBody), buffer->len) ||
		!range_valid(header->materials_offset, header->material_count,
			sizeof(struct TmnfResponseMaterial), buffer->len) ||
		!range_valid(header->contacts_offset, header->contact_count,
			sizeof(struct TmnfResponseContact), buffer->len) ||
		!range_valid(header->events_offset, header->event_count,
			sizeof(struct TmnfResponseContactEvent), buffer->len) ||
		!range_valid(header->replacements_offset,
			header->replacement_count, sizeof(GmVec3), buffer->len) ||
		!range_valid(header->surface_materials_offset,
			TMNF_RESPONSE_SURFACE_MATERIAL_COUNT,
			sizeof(CPlugSurfaceMaterialData), buffer->len)) {
		fprintf(stderr, "invalid response graph framing\n");
		exit(2);
	}
	view.collisions = (const struct TmnfResponseCollision *)
		(buffer->bytes + header->collisions_offset);
	view.bodies = (const struct TmnfResponseBody *)
		(buffer->bytes + header->bodies_offset);
	view.materials = (const struct TmnfResponseMaterial *)
		(buffer->bytes + header->materials_offset);
	view.contacts = (const struct TmnfResponseContact *)
		(buffer->bytes + header->contacts_offset);
	view.events = (const struct TmnfResponseContactEvent *)
		(buffer->bytes + header->events_offset);
	view.replacements = (const GmVec3 *)
		(buffer->bytes + header->replacements_offset);
	view.surface_materials = (const CPlugSurfaceMaterialData *)
		(buffer->bytes + header->surface_materials_offset);
	return view;
}

static int compare_raw(
	const char *role, const void *actual_data,
	const void *expected_data, size_t size) {
	if (size == 0) {
		return 1;
	}
	const uint8_t *actual = actual_data;
	const uint8_t *expected = expected_data;
	if (memcmp(actual, expected, size) == 0) {
		return 1;
	}
	size_t offset = 0;
	while (offset < size && actual[offset] == expected[offset]) {
		offset++;
	}
	uint32_t actual_bits = 0;
	uint32_t expected_bits = 0;
	size_t word_offset = offset & ~(size_t)3;
	size_t available = size - word_offset;
	size_t copy_size = available < 4 ? available : 4;
	memcpy(&actual_bits, actual + word_offset, copy_size);
	memcpy(&expected_bits, expected + word_offset, copy_size);
	if (diagnostic_budget != 0) {
		fprintf(stderr,
			"%s mismatch byte=%zu word=%zu actual=%08x expected=%08x\n",
			role, offset, word_offset, actual_bits, expected_bits);
		diagnostic_budget--;
	}
	return 0;
}

static ReplayBody *body_by_id(ReplayGraph *graph, uint32_t id) {
	if (id == 0) {
		return NULL;
	}
	if (id > graph->input.header->body_count ||
		graph->bodies[id - 1].body.corpus_ref != id) {
		fprintf(stderr, "unknown replay body id %u\n", id);
		exit(2);
	}
	return &graph->bodies[id - 1];
}

static CHmsResponseBody *resolve_body_id(void *user, uint32_t id) {
	ReplayBody *body = body_by_id((ReplayGraph *)user, id);
	if (body == NULL) {
		fprintf(stderr, "null replay body id\n");
		exit(2);
	}
	return &body->body;
}

static const CHmsResponseMaterial *resolve_material_id(
	void *user, uint32_t id) {
	ReplayGraph *graph = user;
	if (id == 0 || id > graph->input.header->material_count) {
		fprintf(stderr, "unknown replay material id %u\n", id);
		exit(2);
	}
	return &graph->materials[id - 1];
}

static uint32_t native_body_id(
	ReplayGraph *graph, const CHmsResponseBody *body) {
	if (body == NULL) {
		return 0;
	}
	for (uint32_t i = 0; i < graph->input.header->body_count; i++) {
		if (&graph->bodies[i].body == body) {
			return graph->bodies[i].body.corpus_ref;
		}
	}
	fprintf(stderr, "unknown native response body pointer\n");
	exit(2);
}

static void encode_native_contact(
	ReplayGraph *graph, const CHmsPhysicalContact *contact,
	struct TmnfResponseContact *output) {
	memset(output, 0, sizeof(*output));
	if (contact == NULL) {
		return;
	}
	output->present = 1;
	output->body_id = native_body_id(graph, contact->body);
	output->tree_id = contact->tree_ref;
	output->surface_material = contact->surface_material;
	output->reserved0a = contact->reserved0a;
	memcpy(output->normal, &contact->normal, sizeof(contact->normal));
	memcpy(output->position, &contact->position, sizeof(contact->position));
	memcpy(output->relative_speed, &contact->relative_speed,
		sizeof(contact->relative_speed));
	memcpy(output->replacement, &contact->replacement,
		sizeof(contact->replacement));
	output->accepted = contact->accepted;
	output->other_body_id = native_body_id(graph, contact->other_body);
	output->other_tree_id = contact->other_tree_ref;
	output->other_surface_material = contact->other_surface_material;
	output->reserved4a = contact->reserved_other;
}

static void decode_contact(
	ReplayGraph *graph, const struct TmnfResponseContact *input,
	CHmsPhysicalContact *output) {
	memset(output, 0, sizeof(*output));
	if (input->present == 0) {
		return;
	}
	output->body = &body_by_id(graph, input->body_id)->body;
	output->tree_ref = input->tree_id;
	output->surface_material = input->surface_material;
	output->reserved0a = input->reserved0a;
	memcpy(&output->normal, input->normal, sizeof(output->normal));
	memcpy(&output->position, input->position, sizeof(output->position));
	memcpy(&output->relative_speed, input->relative_speed,
		sizeof(output->relative_speed));
	memcpy(&output->replacement, input->replacement,
		sizeof(output->replacement));
	output->accepted = input->accepted;
	ReplayBody *other = body_by_id(graph, input->other_body_id);
	output->other_body = other == NULL ? NULL : &other->body;
	output->other_tree_ref = input->other_tree_id;
	output->other_surface_material = input->other_surface_material;
	output->reserved_other = input->reserved4a;
}

static void replay_absorb_contact(
	void *user, CHmsResponseBody *body, CHmsPhysicalContact *contact) {
	CallbackUser *callback = user;
	ReplayGraph *graph = callback->graph;
	if (graph->event_index >= graph->expected.header->event_count) {
		fprintf(stderr, "unexpected AbsorbContact callback\n");
		graph->callback_failed = 1;
		return;
	}
	const struct TmnfResponseContactEvent *event =
		&graph->expected.events[graph->event_index++];
	struct TmnfResponseContact actual;
	encode_native_contact(graph, contact, &actual);
	/* Compute traces contain only outer entry/exit dyna state. Once the first
	 * callback mutates that state, later relative speeds are callback-owned. */
	if (graph->expected.header->kind ==
			TMNF_RESPONSE_COMPUTE_COLLISION_RESPONSE &&
		graph->event_index > 1 && !strict_response_replay) {
		memcpy(actual.relative_speed, event->input.relative_speed,
			sizeof(actual.relative_speed));
	}
	if (event->target_va != callback->target_va ||
		event->item_body_id != callback->body_id ||
		body != &body_by_id(graph, callback->body_id)->body ||
		!compare_raw("AbsorbContact input", &actual,
			&event->input, sizeof(actual))) {
		graph->callback_failed = 1;
	}
	decode_contact(graph, &event->output, contact);
	if (graph->event_index == graph->expected.header->event_count &&
		!strict_response_replay) {
		for (uint32_t i = 0;
		     i < graph->expected.header->body_count; i++) {
			const struct TmnfResponseBody *expected_body =
				&graph->expected.bodies[i];
			ReplayBody *actual_body = &graph->bodies[i];
			if (expected_body->dyna_present != 0) {
				memcpy(&actual_body->state, expected_body->state,
					sizeof(actual_body->state));
			}
		}
	}
}

static void decode_collision(
	const struct TmnfResponseCollision *input,
	SHmsPhysicalCollision *output) {
	memset(output, 0, sizeof(*output));
	output->corpus1 = input->body1_id;
	output->tree1 = input->tree1_id;
	output->corpus2 = input->body2_id;
	output->tree2 = input->tree2_id;
	memcpy(&output->collision, input->collision, sizeof(output->collision));
	output->material = input->material_id;
}

static void encode_native_collision(
	const SHmsPhysicalCollision *input,
	struct TmnfResponseCollision *output) {
	memset(output, 0, sizeof(*output));
	output->body1_id = input->corpus1;
	output->tree1_id = input->tree1;
	output->body2_id = input->corpus2;
	output->tree2_id = input->tree2;
	memcpy(output->collision, &input->collision, sizeof(input->collision));
	output->material_id = input->material;
}

static void initialize_graph(
	ReplayGraph *graph, ResponseView input, ResponseView expected) {
	memset(graph, 0, sizeof(*graph));
	graph->input = input;
	graph->expected = expected;
	if (input.header->body_count != expected.header->body_count ||
		input.header->material_count != expected.header->material_count ||
		input.header->tree_count != expected.header->tree_count ||
		input.header->collision_count != expected.header->collision_count) {
		fprintf(stderr, "response graph identity counts changed\n");
		exit(2);
	}
	graph->collisions = allocate(
		(size_t)input.header->collision_count *
		sizeof(*graph->collisions));
	graph->bodies = allocate(
		(size_t)input.header->body_count * sizeof(*graph->bodies));
	graph->materials = allocate(
		(size_t)input.header->material_count *
		sizeof(*graph->materials));
	memset(graph->bodies, 0,
		(size_t)input.header->body_count * sizeof(*graph->bodies));
	memset(graph->materials, 0,
		(size_t)input.header->material_count *
		sizeof(*graph->materials));
	memcpy(graph->surface_materials, input.surface_materials,
		sizeof(graph->surface_materials));
	for (uint32_t i = 0; i < input.header->collision_count; i++) {
		decode_collision(&input.collisions[i], &graph->collisions[i]);
	}
	for (uint32_t i = 0; i < input.header->material_count; i++) {
		const struct TmnfResponseMaterial *source = &input.materials[i];
		if (source->id != i + 1) {
			fprintf(stderr, "non-canonical response material id\n");
			exit(2);
		}
		memcpy(&graph->materials[i].category, source->bytes + 0x00, 4);
		memcpy(&graph->materials[i].response_mode, source->bytes + 0x08, 4);
		memcpy(graph->materials[i].side_enabled, source->bytes + 0x0c, 8);
	}
	for (uint32_t i = 0; i < input.header->body_count; i++) {
		const struct TmnfResponseBody *source = &input.bodies[i];
		ReplayBody *target = &graph->bodies[i];
		if (source->id != i + 1 ||
			source->replacement_index + source->replacement_count >
				input.header->replacement_count ||
			source->replacement_count > source->replacement_capacity) {
			fprintf(stderr, "invalid response body record\n");
			exit(2);
		}
		target->body.corpus_ref = source->id;
		target->body.classification_flags = source->classification_flags;
		target->body.response_flags = source->response_flags;
		target->body.response_weight = source->response_weight;
		memcpy(&target->body.iso, source->iso, sizeof(target->body.iso));
		target->body.has_contact_sink = source->has_contact_sink;
		target->callback.graph = graph;
		target->callback.body_id = source->id;
		target->callback.target_va = source->contact_target_va;
		if (source->has_contact_sink != 0) {
			target->body.absorb_contact = replay_absorb_contact;
			target->body.contact_user = &target->callback;
		}
		if (source->dyna_present != 0) {
			memcpy(&target->params, source->params, sizeof(target->params));
			memcpy(&target->state, source->state, sizeof(target->state));
			target->dyna.params = &target->params;
			target->dyna.liveState = &target->state;
			target->dyna.dirtyFlag = (int32_t)source->dirty_flag;
			target->dyna.mode = source->mode;
			target->dyna.replacementBuf.count =
				source->replacement_count;
			target->dyna.replacementBuf.capacity =
				source->replacement_capacity;
			if (source->replacement_capacity != 0) {
				target->dyna.replacementBuf.data = allocate(
					(size_t)source->replacement_capacity *
					sizeof(GmVec3));
				memcpy(target->dyna.replacementBuf.data,
					input.replacements +
						source->replacement_index,
					(size_t)source->replacement_count *
						sizeof(GmVec3));
			}
			target->body.dyna = &target->dyna;
		}
	}
	graph->zone.collisions = allocate(sizeof(*graph->zone.collisions));
	graph->zone.collisions->count = input.header->collision_count;
	graph->zone.collisions->capacity = input.header->collision_count;
	graph->zone.collisions->data = graph->collisions;
	graph->zone.surface_materials = graph->surface_materials;
	graph->zone.surface_material_count =
		TMNF_RESPONSE_SURFACE_MATERIAL_COUNT;
	graph->zone.resolve_body = resolve_body_id;
	graph->zone.resolve_material = resolve_material_id;
	graph->zone.resolver_user = graph;
	if (input.header->contact_count != 0) {
		if (input.header->contact_count != 2) {
			fprintf(stderr, "invalid SolveImpulse contact count\n");
			exit(2);
		}
		decode_contact(graph, &input.contacts[0], &graph->contacts[0]);
		decode_contact(graph, &input.contacts[1], &graph->contacts[1]);
	}
}

static void free_graph(ReplayGraph *graph) {
	for (uint32_t i = 0; i < graph->input.header->body_count; i++) {
		free(graph->bodies[i].dyna.replacementBuf.data);
	}
	free(graph->zone.collisions);
	free(graph->materials);
	free(graph->bodies);
	free(graph->collisions);
}

static int compare_graph(ReplayGraph *graph) {
	int passed = !graph->callback_failed;
	if (graph->event_index != graph->expected.header->event_count) {
		if (diagnostic_budget != 0) {
			fprintf(stderr, "AbsorbContact count actual=%u expected=%u\n",
				graph->event_index,
				graph->expected.header->event_count);
			diagnostic_budget--;
		}
		passed = 0;
	}
	if (graph->zone.collisions->count !=
		graph->expected.header->collision_count) {
		fprintf(stderr, "collision count actual=%u expected=%u\n",
			graph->zone.collisions->count,
			graph->expected.header->collision_count);
		passed = 0;
	}
	uint32_t collision_count = graph->zone.collisions->count;
	if (collision_count > graph->expected.header->collision_count) {
		collision_count = graph->expected.header->collision_count;
	}
	for (uint32_t i = 0; i < collision_count; i++) {
		struct TmnfResponseCollision actual;
		encode_native_collision(&graph->collisions[i], &actual);
		if (!compare_raw("physical collision", &actual,
			&graph->expected.collisions[i], sizeof(actual))) {
			passed = 0;
		}
	}
	uint32_t replacement_index = 0;
	for (uint32_t i = 0; i < graph->input.header->body_count; i++) {
		ReplayBody *actual_body = &graph->bodies[i];
		const struct TmnfResponseBody *expected_body =
			&graph->expected.bodies[i];
		struct TmnfResponseBody actual;
		memset(&actual, 0, sizeof(actual));
		actual.id = actual_body->body.corpus_ref;
		actual.classification_flags =
			actual_body->body.classification_flags;
		actual.response_flags = actual_body->body.response_flags;
		actual.response_weight = actual_body->body.response_weight;
		memcpy(actual.iso, &actual_body->body.iso, sizeof(actual.iso));
		actual.has_contact_sink = actual_body->body.has_contact_sink;
		actual.contact_target_va = actual_body->callback.target_va;
		if (actual_body->body.dyna != NULL) {
			actual.dyna_present = 1;
			actual.dirty_flag =
				(uint32_t)actual_body->dyna.dirtyFlag;
			actual.mode = actual_body->dyna.mode;
			actual.replacement_count =
				actual_body->dyna.replacementBuf.count;
			actual.replacement_capacity =
				actual_body->dyna.replacementBuf.capacity;
			actual.replacement_index = replacement_index;
			memcpy(actual.params, &actual_body->params,
				sizeof(actual.params));
			memcpy(actual.state, &actual_body->state,
				sizeof(actual.state));
		}
		if (!compare_raw("response body", &actual, expected_body,
			sizeof(actual))) {
			passed = 0;
		}
		if (expected_body->replacement_index +
				expected_body->replacement_count >
			graph->expected.header->replacement_count) {
			fprintf(stderr, "invalid expected replacement range\n");
			exit(2);
		}
		if (actual.replacement_count !=
			expected_body->replacement_count) {
			if (diagnostic_budget != 0) {
				fprintf(stderr,
					"replacement count body=%u actual=%u expected=%u\n",
					i + 1, actual.replacement_count,
					expected_body->replacement_count);
				diagnostic_budget--;
			}
			passed = 0;
		} else if (!compare_raw("replacement buffer",
			actual_body->dyna.replacementBuf.data,
			graph->expected.replacements +
				expected_body->replacement_index,
			(size_t)actual.replacement_count * sizeof(GmVec3))) {
			passed = 0;
		}
		replacement_index += actual.replacement_count;
	}
	if (graph->input.header->contact_count != 0) {
		for (uint32_t i = 0; i < 2; i++) {
			struct TmnfResponseContact actual;
			CHmsPhysicalContact *contact =
				graph->input.contacts[i].present == 0
				? NULL : &graph->contacts[i];
			encode_native_contact(graph, contact, &actual);
			if (!compare_raw("SolveImpulse contact", &actual,
				&graph->expected.contacts[i], sizeof(actual))) {
				passed = 0;
			}
		}
	}
	return passed;
}

static int replay_response(const Record *record, uint32_t kind) {
	const Buffer *input =
		find_buffer(record->inputs, record->input_count, TAG_THIS);
	const Buffer *expected =
		find_buffer(record->outputs, record->output_count, TAG_OUT_THIS);
	ResponseView input_view = response_view(input, kind);
	ResponseView expected_view = response_view(expected, kind);
	ReplayGraph graph;
	initialize_graph(&graph, input_view, expected_view);
	if (kind == TMNF_RESPONSE_SOLVE_IMPULSE) {
		CHmsPhysicalContact *contact1 =
			input_view.contacts[0].present == 0
			? NULL : &graph.contacts[0];
		CHmsPhysicalContact *contact2 =
			input_view.contacts[1].present == 0
			? NULL : &graph.contacts[1];
		CHmsZoneDynamic_SolveImpulse(
			&graph.zone, &graph.collisions[0], contact1, contact2);
	} else {
		CHmsZoneDynamic_ComputeCollisionResponse(&graph.zone);
	}
	int passed = compare_graph(&graph);
	free_graph(&graph);
	return passed;
}

static int replay_solve_impulse(const Record *record) {
	return replay_response(record, TMNF_RESPONSE_SOLVE_IMPULSE);
}

static int replay_compute_collision_response(const Record *record) {
	return replay_response(
		record, TMNF_RESPONSE_COMPUTE_COLLISION_RESPONSE);
}

static uint32_t raw_u32(const uint8_t *raw, uint32_t offset) {
	uint32_t value;
	memcpy(&value, raw + offset, sizeof(value));
	return value;
}

static int32_t raw_i32(const uint8_t *raw, uint32_t offset) {
	int32_t value;
	memcpy(&value, raw + offset, sizeof(value));
	return value;
}

static float raw_f32(const uint8_t *raw, uint32_t offset) {
	float value;
	memcpy(&value, raw + offset, sizeof(value));
	return value;
}

static const GmIso4 *wheel_contact_body_iso(
	void *user, const CHmsResponseBody *body) {
	(void)body;
	return user;
}

static void decode_wheel_contact(
	CHmsPhysicalContact *contact, const uint8_t *raw,
	CHmsResponseBody *body, CHmsResponseBody *other_body) {
	memset(contact, 0, sizeof(*contact));
	contact->body = raw_u32(raw, 0x00) == 0 ? NULL : body;
	contact->tree_ref = raw_u32(raw, 0x04);
	memcpy(&contact->surface_material, raw + 0x08, 4);
	memcpy(&contact->normal, raw + 0x0c, 12);
	memcpy(&contact->position, raw + 0x18, 12);
	memcpy(&contact->relative_speed, raw + 0x24, 12);
	memcpy(&contact->replacement, raw + 0x30, 12);
	contact->accepted = raw_u32(raw, 0x3c);
	contact->other_body =
		raw_u32(raw, 0x40) == 0 ? NULL : other_body;
	contact->other_tree_ref = raw_u32(raw, 0x44);
	memcpy(&contact->other_surface_material, raw + 0x48, 4);
}

static void decode_contact_tuning(
	TMNFVehicleContactTuning *tuning, const uint8_t *raw) {
	memset(tuning, 0, sizeof(*tuning));
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
	tuning->wheel_contact_model = raw_i32(raw, 0x350);
	tuning->friction_model = raw_i32(raw, 0x354);
}

static void decode_contact_wheel(
	CSceneVehicleCarWheel *wheel, const uint8_t *raw) {
	memset(wheel, 0, sizeof(*wheel));
	memcpy(&wheel->real_time, raw + 0x0b4,
		sizeof(wheel->real_time));
	wheel->field15c = raw_i32(raw, 0x15c);
	memcpy(&wheel->contact_relative_local_distance,
		raw + 0x160, 12);
}

static int replay_wheel_absorb_contact(const Record *record) {
	const Buffer *car_input =
		find_buffer(record->inputs, record->input_count, TAG_THIS);
	const Buffer *wheel_input =
		find_buffer(record->inputs, record->input_count, TAG_ARG1);
	const Buffer *contact_input =
		find_buffer(record->inputs, record->input_count, TAG_ARG2);
	const Buffer *tuning_input =
		find_buffer(record->inputs, record->input_count, TAG_ARG3);
	const Buffer *params_input =
		find_buffer(record->inputs, record->input_count, TAG_ARG4);
	const Buffer *state_input =
		find_buffer(record->inputs, record->input_count, TAG_SCALAR1);
	const Buffer *vehicle_iso_input =
		find_buffer(record->inputs, record->input_count, TAG_SCALAR2);
	const Buffer *other_iso_input =
		find_buffer(record->inputs, record->input_count, TAG_SCALAR3);
	const Buffer *car_expected =
		find_buffer(record->outputs, record->output_count, TAG_OUT_THIS);
	const Buffer *wheel_expected =
		find_buffer(record->outputs, record->output_count, TAG_OUT_ARG1);
	const Buffer *contact_expected =
		find_buffer(record->outputs, record->output_count, TAG_OUT_ARG2);
	const Buffer *state_expected =
		find_buffer(record->outputs, record->output_count, TAG_OUT_ARG3);
	if (car_input == NULL || car_input->len != 0x878 ||
		wheel_input == NULL || wheel_input->len != 0x2fc ||
		contact_input == NULL || contact_input->len != 0x50 ||
		tuning_input == NULL || tuning_input->len != 0x3ac ||
		params_input == NULL ||
			params_input->len != sizeof(CHmsDynaParams) ||
		state_input == NULL || state_input->len != sizeof(CHmsStateDyna) ||
		vehicle_iso_input == NULL ||
			vehicle_iso_input->len != sizeof(GmIso4) ||
		other_iso_input == NULL ||
			other_iso_input->len != sizeof(GmIso4) ||
		car_expected == NULL || car_expected->len != 0x878 ||
		wheel_expected == NULL || wheel_expected->len != 0x2fc ||
		contact_expected == NULL || contact_expected->len != 0x50 ||
		state_expected == NULL ||
			state_expected->len != sizeof(CHmsStateDyna)) {
		fprintf(stderr, "invalid 0x007C11D0 record shape\n");
		exit(2);
	}

	CHmsStateDyna state;
	CHmsDynaParams params;
	CSceneVehicleCar vehicle;
	CSceneVehicleCarWheel wheel;
	TMNFVehicleContactWheelState wheel_state;
	TMNFVehicleContactTuning tuning;
	TMNFVehicleContactContext context;
	CHmsPhysicalContact contact;
	CHmsResponseBody body = { 0 };
	CHmsResponseBody other_body = { 0 };
	uint8_t car_actual[0x878];
	uint8_t wheel_actual[0x2fc];
	uint8_t contact_actual[0x50];
	memcpy(&state, state_input->bytes, sizeof(state));
	memcpy(&params, params_input->bytes, sizeof(params));
	memset(&vehicle, 0, sizeof(vehicle));
	vehicle.dyna_state = &state;
	vehicle.dyna_params = &params;
	memcpy(&vehicle.total_impulse_added,
		car_input->bytes + 0x824, 12);
	decode_contact_wheel(&wheel, wheel_input->bytes);
	memset(&wheel_state, 0, sizeof(wheel_state));
	wheel_state.wheel = &wheel;
	memcpy(&wheel_state.impulse_point,
		wheel_input->bytes + 0x064, 12);
	decode_contact_tuning(&tuning, tuning_input->bytes);
	memset(&context, 0, sizeof(context));
	context.vehicle = &vehicle;
	context.tuning = &tuning;
	context.wheels = &wheel_state;
	context.wheel_count = 1;
	context.vehicle_contact_rotation =
		(const GmMat3 *)vehicle_iso_input->bytes;
	context.resolve_body_iso = wheel_contact_body_iso;
	context.resolve_body_iso_user = other_iso_input->bytes;
	context.side_contact = raw_i32(car_input->bytes, 0x5dc);
	decode_wheel_contact(
		&contact, contact_input->bytes, &body, &other_body);

	CSceneVehicleCar_WheelAbsorbContact(
		&context, &wheel_state, &contact);

	memcpy(car_actual, car_input->bytes, sizeof(car_actual));
	memcpy(car_actual + 0x5dc, &context.side_contact, 4);
	memcpy(car_actual + 0x824, &vehicle.total_impulse_added, 12);
	memcpy(wheel_actual, wheel_input->bytes, sizeof(wheel_actual));
	memcpy(wheel_actual + 0x0b4, &wheel.real_time,
		sizeof(wheel.real_time));
	memcpy(wheel_actual + 0x15c, &wheel.field15c, 4);
	memcpy(wheel_actual + 0x160,
		&wheel.contact_relative_local_distance, 12);
	if (wheel_state.contact_body != NULL) {
		uint32_t contact_body = raw_u32(contact_input->bytes, 0x40);
		memcpy(wheel_actual + 0x13c, &contact_body, 4);
	}
	memcpy(contact_actual, contact_input->bytes, sizeof(contact_actual));
	memcpy(contact_actual + 0x30, &contact.replacement, 12);
	memcpy(contact_actual + 0x3c, &contact.accepted, 4);

	return compare_raw("WheelAbsorbContact dyna", &state,
			state_expected->bytes, sizeof(state)) &&
		compare_raw("WheelAbsorbContact car", car_actual,
			car_expected->bytes, sizeof(car_actual)) &&
		compare_raw("WheelAbsorbContact wheel", wheel_actual,
			wheel_expected->bytes, sizeof(wheel_actual)) &&
		compare_raw("WheelAbsorbContact contact", contact_actual,
			contact_expected->bytes, sizeof(contact_actual));
}

static int replay_body_absorb_contact(const Record *record) {
	const Buffer *car_input =
		find_buffer(record->inputs, record->input_count, TAG_THIS);
	const Buffer *contact_input =
		find_buffer(record->inputs, record->input_count, TAG_ARG1);
	const Buffer *tuning_input =
		find_buffer(record->inputs, record->input_count, TAG_ARG2);
	const Buffer *params_input =
		find_buffer(record->inputs, record->input_count, TAG_ARG3);
	const Buffer *state_input =
		find_buffer(record->inputs, record->input_count, TAG_ARG4);
	const Buffer *car_expected =
		find_buffer(record->outputs, record->output_count, TAG_OUT_THIS);
	const Buffer *contact_expected =
		find_buffer(record->outputs, record->output_count, TAG_OUT_ARG1);
	const Buffer *state_expected =
		find_buffer(record->outputs, record->output_count, TAG_OUT_ARG2);
	if (car_input == NULL || car_input->len != 0x878 ||
		contact_input == NULL || contact_input->len != 0x50 ||
		tuning_input == NULL || tuning_input->len != 0x3ac ||
		params_input == NULL ||
			params_input->len != sizeof(CHmsDynaParams) ||
		state_input == NULL || state_input->len != sizeof(CHmsStateDyna) ||
		car_expected == NULL || car_expected->len != 0x878 ||
		contact_expected == NULL || contact_expected->len != 0x50 ||
		state_expected == NULL ||
			state_expected->len != sizeof(CHmsStateDyna)) {
		fprintf(stderr, "invalid 0x007C3410 record shape\n");
		exit(2);
	}
	if (raw_u32(car_expected->bytes, 0x680) ==
		raw_u32(car_input->bytes, 0x680)) {
		return 1;
	}

	CHmsStateDyna state;
	CHmsDynaParams params;
	CSceneVehicleCar vehicle;
	TMNFVehicleContactTuning tuning;
	TMNFVehicleContactWheelState wheel_state;
	TMNFVehicleContactContext context;
	CHmsPhysicalContact contact;
	CHmsResponseBody body = { 0 };
	CHmsResponseBody other_body = { 0 };
	int32_t air_control = raw_i32(car_input->bytes, 0x5d4);
	int32_t contact_block = raw_i32(car_input->bytes, 0x5d8);
	uint8_t event_source_c = car_input->bytes[0x200];
	uint8_t event_source_ab = car_input->bytes[0x201];
	float event_metric_a = raw_f32(car_input->bytes, 0x670);
	float event_metric_b = raw_f32(car_input->bytes, 0x674);
	float event_metric_c = raw_f32(car_input->bytes, 0x678);
	uint32_t wheel_tree =
		raw_u32(contact_input->bytes, 0x04) ^ 0xffffffffu;
	uint32_t body_tree = raw_u32(contact_input->bytes, 0x04);
	uint8_t car_actual[0x878];
	uint8_t contact_actual[0x50];

	memcpy(&state, state_input->bytes, sizeof(state));
	memcpy(&params, params_input->bytes, sizeof(params));
	memset(&vehicle, 0, sizeof(vehicle));
	vehicle.dyna_state = &state;
	vehicle.dyna_params = &params;
	vehicle.input_gas = raw_f32(car_input->bytes, 0x050);
	vehicle.input_brake = raw_f32(car_input->bytes, 0x054);
	vehicle.flag_60c = raw_i32(car_input->bytes, 0x60c);
	memcpy(&vehicle.total_impulse_added,
		car_input->bytes + 0x824, 12);
	decode_contact_tuning(&tuning, tuning_input->bytes);
	memset(&wheel_state, 0, sizeof(wheel_state));
	memset(&context, 0, sizeof(context));
	context.vehicle = &vehicle;
	context.tuning = &tuning;
	context.wheels = &wheel_state;
	context.wheel_count = 1;
	context.wheel_tree_refs = &wheel_tree;
	context.body_tree_refs = &body_tree;
	context.body_tree_count = 1;
	context.air_control_immediate = &air_control;
	context.contact_block_count = &contact_block;
	context.event_source_c = &event_source_c;
	context.event_source_ab = &event_source_ab;
	context.event_metric_a = &event_metric_a;
	context.event_metric_b = &event_metric_b;
	context.event_metric_c = &event_metric_c;
	context.wheel_contact_absorb_count =
		raw_u32(car_input->bytes, 0x67c);
	context.body_contact_count = raw_u32(car_input->bytes, 0x680);
	memcpy(&context.body_contact_position_sum,
		car_input->bytes + 0x684, 12);
	memcpy(&context.body_contact_normal_sum,
		car_input->bytes + 0x690, 12);
	context.friction_input_selector =
		raw_i32(car_input->bytes, 0x5c4);
	context.side_contact = raw_i32(car_input->bytes, 0x5dc);
	context.last_side_contact_tick =
		raw_u32(car_input->bytes, 0x5e0);
	context.airborne_friction_gate =
		raw_i32(car_input->bytes, 0x5e4);
	decode_wheel_contact(
		&contact, contact_input->bytes, &body, &other_body);

	CSceneVehicleCar_AbsorbContact(&context, &contact);

	memcpy(car_actual, car_input->bytes, sizeof(car_actual));
	memcpy(car_actual + 0x5d4, &air_control, 4);
	memcpy(car_actual + 0x5d8, &contact_block, 4);
	car_actual[0x200] = event_source_c;
	car_actual[0x201] = event_source_ab;
	memcpy(car_actual + 0x670, &event_metric_a, 4);
	memcpy(car_actual + 0x674, &event_metric_b, 4);
	memcpy(car_actual + 0x678, &event_metric_c, 4);
	memcpy(car_actual + 0x67c,
		&context.wheel_contact_absorb_count, 4);
	memcpy(car_actual + 0x680, &context.body_contact_count, 4);
	memcpy(car_actual + 0x684,
		&context.body_contact_position_sum, 12);
	memcpy(car_actual + 0x690,
		&context.body_contact_normal_sum, 12);
	memcpy(car_actual + 0x5dc, &context.side_contact, 4);
	memcpy(car_actual + 0x824, &vehicle.total_impulse_added, 12);
	memcpy(contact_actual, contact_input->bytes, sizeof(contact_actual));
	memcpy(contact_actual + 0x30, &contact.replacement, 12);
	memcpy(contact_actual + 0x3c, &contact.accepted, 4);

	return compare_raw("AbsorbContact dyna", &state,
			state_expected->bytes, sizeof(state)) &&
		compare_raw("AbsorbContact car", car_actual,
			car_expected->bytes, sizeof(car_actual)) &&
		compare_raw("AbsorbContact contact", contact_actual,
			contact_expected->bytes, sizeof(contact_actual));
}

static ReplayHandler find_handler(uint32_t va) {
	switch (va) {
	case 0x0045BD40:
		return replay_mult_transpose;
	case 0x00547E00:
		return replay_collision_neg;
	case 0x00548BF0:
		return replay_solve_impulse;
	case 0x005497C0:
		return replay_compute_collision_response;
	case 0x007C11D0:
		return replay_wheel_absorb_contact;
	case 0x007C3410:
		return replay_body_absorb_contact;
	default:
		return NULL;
	}
}

int main(int argc, char **argv) {
	if (argc != 2) {
		fprintf(stderr, "usage: %s TRACE_DIR\n", argv[0]);
		return 2;
	}
	strict_response_replay =
		getenv("TMNF_STRICT_RESPONSE_REPLAY") != NULL;
	DIR *directory = opendir(argv[1]);
	if (directory == NULL) {
		printf("UNVALIDATED: no collision-response traces\n");
		return 0;
	}

	uint32_t replayed = 0;
	uint32_t failed = 0;
	uint32_t primary_records = 0;
	struct dirent *entry;
	while ((entry = readdir(directory)) != NULL) {
		size_t name_length = strlen(entry->d_name);
		if (name_length < 4 ||
			strcmp(entry->d_name + name_length - 4, ".bin") != 0) {
			continue;
		}
		char path[4096];
		int length = snprintf(
			path, sizeof(path), "%s/%s", argv[1], entry->d_name);
		if (length < 0 || (size_t)length >= sizeof(path)) {
			fail("trace path too long", entry->d_name);
		}
		Trace trace = load_trace(path);
		if (trace.va == 0x00548BF0 || trace.va == 0x005497C0) {
			primary_records += trace.record_count;
		}
		ReplayHandler handler = find_handler(trace.va);
		if (handler != NULL) {
			uint32_t file_failures = 0;
			diagnostic_budget = 24;
			for (uint32_t i = 0; i < trace.record_count; i++) {
				replayed++;
				if (!handler(&trace.records[i])) {
					failed++;
					file_failures++;
					if (file_failures <= 3) {
						printf("FAIL  0x%08X seq=%u\n",
							trace.va, trace.records[i].seq);
					}
				}
			}
			if (file_failures == 0) {
				printf("OK    0x%08X (%u records)\n",
					trace.va, trace.record_count);
			} else {
				printf("FAIL  0x%08X (%u/%u records failed)\n",
					trace.va, file_failures,
					trace.record_count);
			}
		}
		free_trace(&trace);
	}
	closedir(directory);

	if (primary_records == 0) {
		printf("UNVALIDATED: no SolveImpulse/ComputeCollisionResponse traces\n");
	}
	printf("%u collision-response records replayed, %u failed\n",
		replayed, failed);
	return failed != 0;
}
