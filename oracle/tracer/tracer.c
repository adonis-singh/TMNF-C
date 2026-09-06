#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "targets.h"
#include "broadphase_capture.h"
#include "detect_capture.h"
#include "model6_capture.h"
#include "response_capture.h"
#include "response_trace.h"
#include "../../src/trace_format.h"

#define IMAGE_BASE 0x00400000u
#define RECORD_LIMIT 2000u
#define LARGE_GRAPH_RECORD_LIMIT 128u

struct SavedBuffer {
	uint32_t tag;
	uint32_t addr;
	uint32_t len;
	uint8_t *bytes;
};

struct OutputBuffer {
	uint32_t tag;
	uint32_t addr;
	uint32_t len;
};

struct CallContext {
	struct CallContext *prev;
	uint32_t target_index;
	uint32_t seq;
	uint32_t return_addr;
	uint32_t return_value;
	uint16_t n_in;
	uint16_t n_out;
	struct SavedBuffer in[MAX_DIRECTIVES];
	struct OutputBuffer out[MAX_DIRECTIVES];
	struct ResponseCapture *response;
	struct Model6Capture *model6;
	struct DetectCapture *detect;
	struct BroadphaseCapture *broadphase;
	uint32_t callback_contact;
	uint32_t callback_ref_count;
	struct {
		struct ResponseCapture *capture;
		uint32_t event_index;
	} callback_refs[4];
};

struct RuntimeTarget {
	uint8_t *target;
	uint8_t *trampoline;
	uint8_t *entry_stub;
	uint8_t *return_stub;
	HANDLE file;
	CRITICAL_SECTION file_lock;
	volatile LONG started;
	uint32_t written;
};

struct VirtualCallSite {
	uint32_t va;
	const char *name;
	uint8_t patch_len;
	uint8_t target_load_len;
	uint8_t target_reg;
	uint8_t expected[16];
};

struct RuntimeVirtualCallSite {
	uint8_t *site;
	uint8_t *stub;
	volatile LONG logged;
};

static const struct VirtualCallSite g_virtual_call_sites[] = {
	{
		0x0054825Bu, "force-field", 8, 6,
		0,
		{ 0x8B, 0x82, 0x8C, 0x00, 0x00, 0x00, 0xFF, 0xD0 }
	},
	{
		0x005483B7u, "vehicle-force", 5, 3,
		0,
		{ 0x8B, 0x42, 0x0C, 0xFF, 0xD0 }
	},
	{
		0x007C12AAu, "wheel-contact-surface-iso", 7, 5, 0,
		{ 0x8B, 0x11, 0x8B, 0x42, 0x78, 0xFF, 0xD0 }
	},
	{
		0x007C12FCu, "wheel-contact-car-iso", 7, 5, 2,
		{ 0x8B, 0x01, 0x8B, 0x50, 0x78, 0xFF, 0xD2 }
	},
	{
		0x007C2928u, "water-car-iso", 7, 5, 2,
		{ 0x8B, 0x01, 0x8B, 0x50, 0x78, 0xFF, 0xD2 }
	},
	{
		0x007C298Bu, "water-zone-environment", 12, 10, 0,
		{ 0x8B, 0x10, 0x8B, 0xC8, 0x8B, 0x82, 0xA8, 0x00,
		  0x00, 0x00, 0xFF, 0xD0 }
	},
	{
		0x007C3BE4u, "integrate-vehicle-callback", 12, 10, 0,
		{ 0x8B, 0x11, 0x8B, 0x82, 0xBC, 0x00, 0x00, 0x00,
		  0x6A, 0x00, 0xFF, 0xD0 }
	},
	{
		0x007C3CB5u, "fake-contact-car-iso", 8, 6, 2,
		{ 0x8B, 0x50, 0x78, 0x8D, 0x6B, 0x34, 0xFF, 0xD2 }
	},
	{
		0x007C3E9Bu, "model6-car-iso", 7, 5, 2,
		{ 0x8B, 0x01, 0x8B, 0x50, 0x78, 0xFF, 0xD2 }
	},
	{
		0x007C73FEu, "compute-forces-post", 10, 8, 0,
		{ 0x8B, 0x16, 0x8B, 0x82, 0xA4, 0x01, 0x00, 0x00,
		  0xFF, 0xD0 }
	},
	{
		0x007C97B5u, "water-splash-zombie", 12, 10, 2,
		{ 0x8B, 0x06, 0x8B, 0x90, 0x08, 0x01, 0x00, 0x00,
		  0x8B, 0xCE, 0xFF, 0xD2 }
	},
	{
		0x00844E1Au, "bitmap-mipmap-image", 10, 8, 0,
		{ 0x8B, 0x11, 0x8B, 0x82, 0x80, 0x00, 0x00, 0x00,
		  0xFF, 0xD0 }
	},
	{
		0x00845861u, "bitmap-regenerate-image", 10, 8, 2,
		{ 0x8B, 0x01, 0x8B, 0x90, 0x80, 0x00, 0x00, 0x00,
		  0xFF, 0xD2 }
	},
	{
		0x0092494Du, "mw-release-child", 8, 6, 2,
		{ 0x8B, 0x01, 0x8B, 0x50, 0x20, 0x53, 0xFF, 0xD2 }
	},
	{
		0x0092497Du, "mw-release-self", 11, 9, 2,
		{ 0x8B, 0x03, 0x8B, 0x50, 0x04, 0x6A, 0x01, 0x8B,
		  0xCB, 0xFF, 0xD2 }
	},
	{
		0x0093E5D2u, "command-buffer-node-iso", 7, 5, 0,
		{ 0x8B, 0x11, 0x8B, 0x42, 0x78, 0xFF, 0xD0 }
	}
};

static struct RuntimeTarget g_runtime[40];
static struct RuntimeVirtualCallSite g_virtual_runtime[16];
static DWORD g_context_tls = TLS_OUT_OF_INDEXES;
static HANDLE g_heap;
static HANDLE g_log = INVALID_HANDLE_VALUE;
static volatile LONG g_ready;
static uint32_t g_module_base;
static int g_filter_active;
static uint32_t g_record_start;
static uint8_t g_target_enabled[32];
static const float g_identity_iso[12] = {
	1.0f, 0.0f, 0.0f,
	0.0f, 1.0f, 0.0f,
	0.0f, 0.0f, 1.0f,
	0.0f, 0.0f, 0.0f,
};

static void fatal(const char *text);

static int is_response_target(uint32_t va)
{
	return va == 0x00548BF0u || va == 0x005497C0u;
}

/* The four CSceneVehicleCar::ComputeForcesModelN entries share one ABI and
 * one vehicle-graph capture (model6_capture.c). */
static int is_model6_target(uint32_t va)
{
	return va == 0x007C3E80u || va == 0x007FA770u
		|| va == 0x007FB5F0u || va == 0x007FC170u;
}

static int is_internal_target(uint32_t va)
{
	return va == 0x007B39F0u || va == 0x0047CBA0u;
}

static int filter_contains(const char *filter, uint32_t va)
{
	const char *cursor = filter;
	while (*cursor != '\0') {
		char *end;
		unsigned long value = strtoul(cursor, &end, 0);
		if (end == cursor || value > UINT32_MAX)
			fatal("TMNF_TRACE_TARGETS is invalid");
		if ((uint32_t)value == va)
			return 1;
		if (*end == '\0')
			return 0;
		if (*end != ',')
			fatal("TMNF_TRACE_TARGETS is invalid");
		cursor = end + 1;
	}
	return 0;
}

static uint32_t record_limit(uint32_t va)
{
	if (va == 0x005341C0u || va == 0x00548BF0u || va == 0x007BE390u ||
	    va == 0x007C11D0u || va == 0x007C3410u)
		return 12000u;
	if ((va == 0x00537150u || va == 0x008EADC0u
	     || va == 0x008EA2D0u)
	    && g_filter_active)
		return 1024u;
	if (va == 0x00537150u || va == 0x008E8890u
	    || va == 0x008EADC0u || va == 0x008EA2D0u)
		return LARGE_GRAPH_RECORD_LIMIT;
	if (va == 0x0053A120u)
		return 19u;
	if (va == 0x0053B1C0u)
		return 7u;
	return RECORD_LIMIT;
}

static int has_race_target_ancestor(void)
{
	struct CallContext *context =
		(struct CallContext *)TlsGetValue(g_context_tls);
	for (; context != NULL; context = context->prev) {
		if (detect_capture_is_race_target(
			    g_targets[context->target_index].va))
			return 1;
	}
	return 0;
}

static int has_target_ancestor(uint32_t va)
{
	struct CallContext *context =
		(struct CallContext *)TlsGetValue(g_context_tls);
	for (; context != NULL; context = context->prev) {
		if (g_targets[context->target_index].va == va)
			return 1;
	}
	return 0;
}

static void write_all(HANDLE file, const void *data, DWORD len)
{
	const uint8_t *cursor = (const uint8_t *)data;
	while (len != 0) {
		DWORD written = 0;
		if (!WriteFile(file, cursor, len, &written, NULL) || written == 0)
			TerminateProcess(GetCurrentProcess(), 120);
		cursor += written;
		len -= written;
	}
}

static void log_line(const char *text)
{
	DWORD len;
	if (g_log == INVALID_HANDLE_VALUE)
		return;
	len = (DWORD)strlen(text);
	write_all(g_log, text, len);
	write_all(g_log, "\r\n", 2);
	FlushFileBuffers(g_log);
}

static void fatal(const char *text)
{
	OutputDebugStringA(text);
	log_line(text);
	TerminateProcess(GetCurrentProcess(), 121);
}

static void __cdecl log_virtual_target(uint32_t site_index, uint32_t target)
{
	const struct VirtualCallSite *site;
	struct RuntimeVirtualCallSite *runtime;
	char line[160];
	uint32_t static_va;

	if (site_index >= sizeof(g_virtual_call_sites) / sizeof(g_virtual_call_sites[0]))
		fatal("invalid virtual call-site index");
	site = &g_virtual_call_sites[site_index];
	runtime = &g_virtual_runtime[site_index];
	if (InterlockedCompareExchange(&runtime->logged, 1, 0) != 0)
		return;
	static_va = target - g_module_base + IMAGE_BASE;
	_snprintf(line, sizeof(line),
	          "virtual %s site=%08X runtime=%08X static=%08X",
	          site->name, site->va, target, static_va);
	log_line(line);
}

static uint32_t resolve_address(
	const struct CaptureDirective *directive,
	uint32_t this_ptr,
	uint32_t entry_esp)
{
	switch (directive->kind) {
	case CAP_THIS:
		return this_ptr + directive->add;
	case CAP_THIS_OFFSET:
		return this_ptr + directive->offset + directive->add;
	case CAP_STACK_POINTER:
		return *(const uint32_t *)(uintptr_t)(entry_esp + directive->offset)
			+ directive->add;
	case CAP_STACK_SCALAR:
		return entry_esp + directive->offset;
	case CAP_CURRENT_TICK: {
		uint32_t root = *(const uint32_t *)(uintptr_t)(
			g_module_base + 0x00D731E0u - IMAGE_BASE);
		uint32_t timer = *(const uint32_t *)(uintptr_t)(root + 0x14);

		if (timer == 0)
			timer = root + 0xA0;
		return timer + 0x1C + directive->add;
	}
	case CAP_DEREF_THIS:
		return *(const uint32_t *)(uintptr_t)(this_ptr + directive->offset)
			+ directive->add;
	case CAP_DYNA_REPLACEMENTS:
		return *(const uint32_t *)(uintptr_t)(this_ptr + directive->offset)
			+ directive->add;
	case CAP_VEHICLE_WHEEL_SPEED: {
		uint32_t vehicle_info =
			*(const uint32_t *)(uintptr_t)(this_ptr + 0x64);
		uint32_t data =
			*(const uint32_t *)(uintptr_t)(vehicle_info + 0x18);
		uint32_t index =
			*(const uint32_t *)(uintptr_t)(vehicle_info + 0x24);
		uint32_t tuning =
			*(const uint32_t *)(uintptr_t)(data + index * 4);

		return tuning + 0x2bc;
	}
	case CAP_VEHICLE_DYNA_STATE: {
		uint32_t item = *(const uint32_t *)(uintptr_t)(this_ptr + 0x28);
		uint32_t data = *(const uint32_t *)(uintptr_t)(item + 0x38);
		uint32_t relation = *(const uint32_t *)(uintptr_t)data;
		uint32_t dyna = *(const uint32_t *)(uintptr_t)(relation + 0x58);
		uint32_t state = *(const uint32_t *)(uintptr_t)(dyna + 0x32c);

		return state + directive->add;
	}
	case CAP_VEHICLE_DYNA_TEMP_STATE: {
		uint32_t item = *(const uint32_t *)(uintptr_t)(this_ptr + 0x28);
		uint32_t data = *(const uint32_t *)(uintptr_t)(item + 0x38);
		uint32_t relation = *(const uint32_t *)(uintptr_t)data;
		uint32_t dyna = *(const uint32_t *)(uintptr_t)(relation + 0x58);

		return dyna + 0x274 + directive->add;
	}
	case CAP_VEHICLE_CORPUS_ISO: {
		typedef uint32_t (__attribute__((thiscall)) *GetIsoFn)(
			uint32_t);
		uint32_t item = *(const uint32_t *)(uintptr_t)(this_ptr + 0x28);
		uint32_t data = *(const uint32_t *)(uintptr_t)(item + 0x38);
		uint32_t corpus = *(const uint32_t *)(uintptr_t)data;
		uint32_t vtable = *(const uint32_t *)(uintptr_t)corpus;
		GetIsoFn get_iso = (GetIsoFn)(uintptr_t)
			*(const uint32_t *)(uintptr_t)(vtable + 0x78);

		return get_iso(corpus) + directive->add;
	}
	case CAP_VEHICLE_DYNA_PARAMS: {
		uint32_t item = *(const uint32_t *)(uintptr_t)(this_ptr + 0x28);
		uint32_t data = *(const uint32_t *)(uintptr_t)(item + 0x38);
		uint32_t relation = *(const uint32_t *)(uintptr_t)data;
		uint32_t dyna = *(const uint32_t *)(uintptr_t)(relation + 0x58);
		uint32_t params = *(const uint32_t *)(uintptr_t)(dyna + 0x108);

		return params + directive->add;
	}
	case CAP_VEHICLE_TUNING: {
		uint32_t vehicle_info =
			*(const uint32_t *)(uintptr_t)(this_ptr + 0x64);
		uint32_t data =
			*(const uint32_t *)(uintptr_t)(vehicle_info + 0x18);
		uint32_t index =
			*(const uint32_t *)(uintptr_t)(vehicle_info + 0x24);

		return *(const uint32_t *)(uintptr_t)(data + index * 4)
			+ directive->add;
	}
	case CAP_VEHICLE_WHEELS:
		return *(const uint32_t *)(uintptr_t)(this_ptr + 0x2ec)
			+ directive->add;
	case CAP_VEHICLE_TUNING_BUFFER: {
		uint32_t vehicle_info =
			*(const uint32_t *)(uintptr_t)(this_ptr + 0x64);
		uint32_t data =
			*(const uint32_t *)(uintptr_t)(vehicle_info + 0x18);
		uint32_t index =
			*(const uint32_t *)(uintptr_t)(vehicle_info + 0x24);
		uint32_t tuning =
			*(const uint32_t *)(uintptr_t)(data + index * 4);

		return *(const uint32_t *)(uintptr_t)(
			tuning + directive->offset + 4) + directive->add;
	}
	case CAP_CONTACT_OTHER_ISO: {
		uint32_t contact =
			*(const uint32_t *)(uintptr_t)(entry_esp + directive->offset);
		uint32_t corpus =
			*(const uint32_t *)(uintptr_t)(contact + 0x40);
		if (corpus == 0)
			return (uint32_t)(uintptr_t)g_identity_iso;
		return corpus + 0x18 + directive->add;
	}
	}
	fatal("unknown capture directive");
	return 0;
}

static void free_context(struct CallContext *context)
{
	uint32_t i;
	for (i = 0; i < context->n_in; ++i)
		HeapFree(g_heap, 0, context->in[i].bytes);
	response_capture_destroy(context->response);
	model6_capture_destroy(context->model6);
	detect_capture_destroy(context->detect);
	broadphase_capture_destroy(context->broadphase);
	HeapFree(g_heap, 0, context);
}

static void update_record_count(struct RuntimeTarget *runtime)
{
	DWORD end;
	if (SetFilePointer(runtime->file, 12, NULL, FILE_BEGIN) == INVALID_SET_FILE_POINTER
	    && GetLastError() != NO_ERROR)
		fatal("failed to seek trace header");
	write_all(runtime->file, &runtime->written, sizeof(runtime->written));
	end = SetFilePointer(runtime->file, 0, NULL, FILE_END);
	if (end == INVALID_SET_FILE_POINTER && GetLastError() != NO_ERROR)
		fatal("failed to seek trace end");
	FlushFileBuffers(runtime->file);
}

static void write_record(struct RuntimeTarget *runtime, const struct CallContext *context)
{
	uint32_t i;
	write_all(runtime->file, &context->seq, sizeof(context->seq));
	write_all(runtime->file, &context->n_in, sizeof(context->n_in));
	write_all(runtime->file, &context->n_out, sizeof(context->n_out));
	for (i = 0; i < context->n_in; ++i) {
		write_all(runtime->file, &context->in[i].tag, 4);
		write_all(runtime->file, &context->in[i].addr, 4);
		write_all(runtime->file, &context->in[i].len, 4);
		write_all(runtime->file, context->in[i].bytes, context->in[i].len);
	}
	for (i = 0; i < context->n_out; ++i) {
		const struct OutputBuffer *out = &context->out[i];
		write_all(runtime->file, &out->tag, 4);
		write_all(runtime->file, &out->addr, 4);
		write_all(runtime->file, &out->len, 4);
		write_all(runtime->file, (const void *)(uintptr_t)out->addr, out->len);
	}
	++runtime->written;
	update_record_count(runtime);
}

static void write_response_record(
	struct RuntimeTarget *runtime, const struct CallContext *context)
{
	const uint8_t *input;
	uint8_t *output;
	uint32_t input_size;
	uint32_t output_size;
	uint16_t input_count = (uint16_t)(1 + context->n_in);
	uint16_t one = 1;
	uint32_t tag;
	uint32_t address = 0;
	uint32_t i;
	input = response_capture_input(context->response, &input_size);
	output = response_capture_output(context->response, &output_size);
	write_all(runtime->file, &context->seq, sizeof(context->seq));
	write_all(runtime->file, &input_count, sizeof(input_count));
	write_all(runtime->file, &one, sizeof(one));
	tag = TAG_THIS;
	write_all(runtime->file, &tag, 4);
	write_all(runtime->file, &address, 4);
	write_all(runtime->file, &input_size, 4);
	write_all(runtime->file, input, input_size);
	for (i = 0; i < context->n_in; ++i) {
		write_all(runtime->file, &context->in[i].tag, 4);
		write_all(runtime->file, &context->in[i].addr, 4);
		write_all(runtime->file, &context->in[i].len, 4);
		write_all(runtime->file,
			context->in[i].bytes, context->in[i].len);
	}
	tag = TAG_OUT_THIS;
	write_all(runtime->file, &tag, 4);
	write_all(runtime->file, &address, 4);
	write_all(runtime->file, &output_size, 4);
	write_all(runtime->file, output, output_size);
	response_capture_free_blob(output);
	++runtime->written;
	update_record_count(runtime);
}

static void write_model6_record(
	struct RuntimeTarget *runtime, const struct CallContext *context)
{
	const uint8_t *input;
	uint8_t *output;
	uint32_t input_size;
	uint32_t output_size;
	uint16_t one = 1;
	uint32_t tag;
	uint32_t address = 0;
	input = model6_capture_input(context->model6, &input_size);
	output = model6_capture_output(context->model6, &output_size);
	write_all(runtime->file, &context->seq, sizeof(context->seq));
	write_all(runtime->file, &one, sizeof(one));
	write_all(runtime->file, &one, sizeof(one));
	tag = TAG_THIS;
	write_all(runtime->file, &tag, 4);
	write_all(runtime->file, &address, 4);
	write_all(runtime->file, &input_size, 4);
	write_all(runtime->file, input, input_size);
	tag = TAG_OUT_THIS;
	write_all(runtime->file, &tag, 4);
	write_all(runtime->file, &address, 4);
	write_all(runtime->file, &output_size, 4);
	write_all(runtime->file, output, output_size);
	model6_capture_free_blob(output);
	++runtime->written;
	update_record_count(runtime);
}

static void write_detect_record(
	struct RuntimeTarget *runtime, const struct CallContext *context)
{
	const uint8_t *input;
	uint8_t *output;
	uint32_t input_size;
	uint32_t output_size;
	uint16_t one = 1;
	uint32_t tag;
	uint32_t address = 0;
	input = detect_capture_input(context->detect, &input_size);
	output = detect_capture_output(
		context->detect, context->return_value, &output_size);
	write_all(runtime->file, &context->seq, sizeof(context->seq));
	write_all(runtime->file, &one, sizeof(one));
	write_all(runtime->file, &one, sizeof(one));
	tag = TAG_THIS;
	write_all(runtime->file, &tag, 4);
	write_all(runtime->file, &address, 4);
	write_all(runtime->file, &input_size, 4);
	write_all(runtime->file, input, input_size);
	tag = TAG_OUT_THIS;
	write_all(runtime->file, &tag, 4);
	write_all(runtime->file, &address, 4);
	write_all(runtime->file, &output_size, 4);
	write_all(runtime->file, output, output_size);
	detect_capture_free_blob(output);
	++runtime->written;
	update_record_count(runtime);
}

static void write_broadphase_record(
	struct RuntimeTarget *runtime, const struct CallContext *context)
{
	const uint8_t *input;
	uint8_t *output;
	uint32_t input_size;
	uint32_t output_size;
	uint16_t one = 1;
	uint32_t tag;
	uint32_t address = 0;
	input = broadphase_capture_input(context->broadphase, &input_size);
	output = broadphase_capture_output(
		context->broadphase, context->return_value, &output_size);
	write_all(runtime->file, &context->seq, sizeof(context->seq));
	write_all(runtime->file, &one, sizeof(one));
	write_all(runtime->file, &one, sizeof(one));
	tag = TAG_THIS;
	write_all(runtime->file, &tag, 4);
	write_all(runtime->file, &address, 4);
	write_all(runtime->file, &input_size, 4);
	write_all(runtime->file, input, input_size);
	tag = TAG_OUT_THIS;
	write_all(runtime->file, &tag, 4);
	write_all(runtime->file, &address, 4);
	write_all(runtime->file, &output_size, 4);
	write_all(runtime->file, output, output_size);
	broadphase_capture_free_blob(output);
	++runtime->written;
	update_record_count(runtime);
}

void __cdecl trace_enter(uint32_t index, uint32_t this_ptr, uint32_t entry_esp)
{
	const struct TargetSpec *target;
	struct RuntimeTarget *runtime;
	struct CallContext *context;
	uint32_t i;
	LONG sequence;

	if (!g_ready || index >= g_target_count)
		fatal("entry hook ran before tracer initialization");
	target = &g_targets[index];
	runtime = &g_runtime[index];
	if (is_model6_target(target->va)
	    && (*(const uint32_t *)(uintptr_t)(this_ptr + 0x2E8) != 4
	        || *(const uint32_t *)(uintptr_t)(this_ptr + 0x2EC) == 0))
		return;
	if (detect_capture_is_race_target(target->va)
	    && !detect_capture_race_ready())
		return;
	if (detect_capture_is_target(target->va)
	    && target->va != 0x00547DE0u
	    && !g_filter_active
	    && !has_race_target_ancestor())
		return;
	sequence = InterlockedIncrement(&runtime->started) - 1;
	if (!is_internal_target(target->va)
	    && !((target->va == 0x008E2570u
	          || target->va == 0x008E2970u)
	        && has_target_ancestor(0x008EADC0u))
	    && ((uint32_t)sequence < g_record_start
	        || (uint64_t)(uint32_t)sequence >=
	            (uint64_t)g_record_start + record_limit(target->va)))
		return;

	context = (struct CallContext *)HeapAlloc(g_heap, HEAP_ZERO_MEMORY, sizeof(*context));
	if (context == NULL)
		fatal("failed to allocate call context");
	context->prev = (struct CallContext *)TlsGetValue(g_context_tls);
	context->target_index = index;
	context->seq = (uint32_t)sequence;
	context->return_addr = *(const uint32_t *)(uintptr_t)entry_esp;
	context->n_in = target->n_in;
	context->n_out = target->n_out;

	if (is_response_target(target->va)) {
		uint32_t kind = target->va == 0x00548BF0u
			? TMNF_RESPONSE_SOLVE_IMPULSE
			: TMNF_RESPONSE_COMPUTE_COLLISION_RESPONSE;
		context->response =
			response_capture_create(kind, this_ptr, entry_esp);
	} else if (is_model6_target(target->va)) {
		context->model6 = model6_capture_create(this_ptr, entry_esp);
	} else if (detect_capture_is_target(target->va)) {
		context->detect =
			detect_capture_create(target->va, this_ptr, entry_esp);
	} else if (broadphase_capture_is_target(target->va)) {
		context->broadphase =
			broadphase_capture_create(target->va, this_ptr, entry_esp);
	} else if (is_internal_target(target->va)) {
		struct CallContext *ancestor;
		context->callback_contact =
			*(const uint32_t *)(uintptr_t)(entry_esp + 8);
		for (ancestor = context->prev; ancestor != NULL;
		     ancestor = ancestor->prev) {
			uint32_t ref;
			if (ancestor->response == NULL)
				continue;
			if (context->callback_ref_count
			    >= sizeof(context->callback_refs)
				    / sizeof(context->callback_refs[0]))
				fatal("too many nested response captures");
			ref = context->callback_ref_count++;
			context->callback_refs[ref].capture = ancestor->response;
			context->callback_refs[ref].event_index =
				response_capture_event_enter(
					ancestor->response, target->va,
					*(const uint32_t *)(uintptr_t)(entry_esp + 4),
					context->callback_contact);
		}
	}

	for (i = 0; i < context->n_in; ++i) {
		const struct CaptureDirective *directive = &target->in[i];
		uint32_t source = resolve_address(directive, this_ptr, entry_esp);
		struct SavedBuffer *saved = &context->in[i];
		saved->tag = directive->tag;
		saved->addr = directive->kind == CAP_STACK_SCALAR ? 0 : source;
		saved->len = directive->kind == CAP_DYNA_REPLACEMENTS
			? *(const uint32_t *)(uintptr_t)(
				this_ptr + directive->offset - 4) * 12
			: directive->len;
		saved->bytes = (uint8_t *)HeapAlloc(g_heap, 0, saved->len);
		if (saved->bytes == NULL)
			fatal("failed to allocate input snapshot");
		memcpy(saved->bytes, (const void *)(uintptr_t)source, saved->len);
	}
	for (i = 0; i < target->n_out; ++i) {
		const struct CaptureDirective *directive = &target->out[i];
		struct OutputBuffer *out = &context->out[i];
		out->tag = directive->tag;
		out->addr = resolve_address(directive, this_ptr, entry_esp);
		out->len = directive->len;
	}

	if (!TlsSetValue(g_context_tls, context))
		fatal("failed to set hook context");
	*(uint32_t *)(uintptr_t)entry_esp = (uint32_t)(uintptr_t)runtime->return_stub;
}

uint32_t __cdecl trace_exit(uint32_t index, uint32_t return_value)
{
	struct CallContext *context = (struct CallContext *)TlsGetValue(g_context_tls);
	struct RuntimeTarget *runtime;
	uint32_t return_addr;
	uint32_t i;

	if (context == NULL || context->target_index != index)
		fatal("hook return stack mismatch");
	if (!TlsSetValue(g_context_tls, context->prev))
		fatal("failed to pop hook context");
	context->return_value = return_value;
	runtime = &g_runtime[index];
	if (is_internal_target(g_targets[index].va)) {
		for (i = 0; i < context->callback_ref_count; ++i) {
			response_capture_event_exit(
				context->callback_refs[i].capture,
				context->callback_refs[i].event_index,
				context->callback_contact);
		}
	} else {
		EnterCriticalSection(&runtime->file_lock);
		if (context->response != NULL)
			write_response_record(runtime, context);
		else if (context->model6 != NULL)
			write_model6_record(runtime, context);
		else if (context->detect != NULL)
			write_detect_record(runtime, context);
		else if (context->broadphase != NULL)
			write_broadphase_record(runtime, context);
		else
			write_record(runtime, context);
		LeaveCriticalSection(&runtime->file_lock);
	}
	return_addr = context->return_addr;
	free_context(context);
	return return_addr;
}

static void emit_u8(uint8_t **cursor, uint8_t value)
{
	*(*cursor)++ = value;
}

static void emit_u32(uint8_t **cursor, uint32_t value)
{
	memcpy(*cursor, &value, sizeof(value));
	*cursor += sizeof(value);
}

static void emit_rel32(uint8_t **cursor, uint8_t opcode, const void *destination)
{
	uintptr_t next;
	int32_t displacement;
	emit_u8(cursor, opcode);
	next = (uintptr_t)*cursor + 4;
	displacement = (int32_t)((uintptr_t)destination - next);
	emit_u32(cursor, (uint32_t)displacement);
}

static uint8_t *allocate_code(void)
{
	uint8_t *memory = (uint8_t *)VirtualAlloc(
		NULL, 128, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE);
	if (memory == NULL)
		fatal("failed to allocate executable hook memory");
	return memory;
}

static void build_runtime_code(uint32_t index)
{
	const struct TargetSpec *target = &g_targets[index];
	struct RuntimeTarget *runtime = &g_runtime[index];
	uint8_t *cursor;

	runtime->trampoline = allocate_code();
	cursor = runtime->trampoline;
	if (target->expected[0] == 0xE9) {
		int32_t displacement;
		uint8_t *destination;
		memcpy(&displacement, runtime->target + 1, sizeof(displacement));
		destination = runtime->target + 5 + displacement;
		emit_rel32(&cursor, 0xE9, destination);
	} else {
		memcpy(runtime->trampoline, runtime->target, target->patch_len);
		cursor += target->patch_len;
		emit_rel32(&cursor, 0xE9, runtime->target + target->patch_len);
	}

	runtime->return_stub = allocate_code();
	cursor = runtime->return_stub;
	emit_u8(&cursor, 0x9C);
	emit_u8(&cursor, 0x60);
	emit_u8(&cursor, 0x8B);
	emit_u8(&cursor, 0x44);
	emit_u8(&cursor, 0x24);
	emit_u8(&cursor, 0x1C);
	emit_u8(&cursor, 0x50);
	emit_u8(&cursor, 0x68);
	emit_u32(&cursor, index);
	emit_rel32(&cursor, 0xE8, trace_exit);
	emit_u8(&cursor, 0x83);
	emit_u8(&cursor, 0xC4);
	emit_u8(&cursor, 0x08);
	emit_u8(&cursor, 0x89);
	emit_u8(&cursor, 0x44);
	emit_u8(&cursor, 0x24);
	emit_u8(&cursor, 0x0C);
	emit_u8(&cursor, 0x61);
	emit_u8(&cursor, 0x9D);
	emit_u8(&cursor, 0xFF);
	emit_u8(&cursor, 0x64);
	emit_u8(&cursor, 0x24);
	emit_u8(&cursor, 0xE8);

	runtime->entry_stub = allocate_code();
	cursor = runtime->entry_stub;
	emit_u8(&cursor, 0x9C);
	emit_u8(&cursor, 0x60);
	emit_u8(&cursor, 0x8B);
	emit_u8(&cursor, 0x44);
	emit_u8(&cursor, 0x24);
	emit_u8(&cursor, 0x0C);
	emit_u8(&cursor, 0x83);
	emit_u8(&cursor, 0xC0);
	emit_u8(&cursor, 0x04);
	emit_u8(&cursor, 0x8B);
	emit_u8(&cursor, 0x54);
	emit_u8(&cursor, 0x24);
	emit_u8(&cursor, 0x18);
	emit_u8(&cursor, 0x50);
	emit_u8(&cursor, 0x52);
	emit_u8(&cursor, 0x68);
	emit_u32(&cursor, index);
	emit_rel32(&cursor, 0xE8, trace_enter);
	emit_u8(&cursor, 0x83);
	emit_u8(&cursor, 0xC4);
	emit_u8(&cursor, 0x0C);
	emit_u8(&cursor, 0x61);
	emit_u8(&cursor, 0x9D);
	emit_rel32(&cursor, 0xE9, runtime->trampoline);
}

static void build_virtual_call_site(uint32_t index)
{
	const struct VirtualCallSite *site = &g_virtual_call_sites[index];
	struct RuntimeVirtualCallSite *runtime = &g_virtual_runtime[index];
	uint8_t *cursor;

	runtime->stub = allocate_code();
	cursor = runtime->stub;
	memcpy(cursor, site->expected, site->target_load_len);
	cursor += site->target_load_len;
	emit_u8(&cursor, 0x9C);
	emit_u8(&cursor, 0x60);
	emit_u8(&cursor, (uint8_t)(0x50 + site->target_reg));
	emit_u8(&cursor, 0x68);
	emit_u32(&cursor, index);
	emit_rel32(&cursor, 0xE8, log_virtual_target);
	emit_u8(&cursor, 0x83);
	emit_u8(&cursor, 0xC4);
	emit_u8(&cursor, 0x08);
	emit_u8(&cursor, 0x61);
	emit_u8(&cursor, 0x9D);
	emit_u8(&cursor, 0xFF);
	emit_u8(&cursor, (uint8_t)(0xD0 + site->target_reg));
	emit_rel32(&cursor, 0xE9, runtime->site + site->patch_len);
}

static void install_patch(uint32_t index)
{
	const struct TargetSpec *target = &g_targets[index];
	struct RuntimeTarget *runtime = &g_runtime[index];
	DWORD old_protect;
	DWORD ignored;
	uint8_t *cursor;
	uint32_t i;
	char message[128];

	if ((target->expected[0] == 0xE9 && runtime->target[0] != 0xE9)
	    || (target->expected[0] != 0xE9
	        && memcmp(runtime->target, target->expected, target->patch_len) != 0)) {
		_snprintf(message, sizeof(message),
		          "target mismatch %08X (%s): %02X %02X %02X %02X %02X %02X",
		          target->va, target->name,
		          runtime->target[0], runtime->target[1], runtime->target[2],
		          runtime->target[3], runtime->target[4], runtime->target[5]);
		fatal(message);
	}
	if (!VirtualProtect(runtime->target, target->patch_len, PAGE_EXECUTE_READWRITE,
	                    &old_protect))
		fatal("failed to make target writable");
	cursor = runtime->target;
	emit_rel32(&cursor, 0xE9, runtime->entry_stub);
	for (i = 5; i < target->patch_len; ++i)
		runtime->target[i] = 0x90;
	if (!VirtualProtect(runtime->target, target->patch_len, old_protect, &ignored))
		fatal("failed to restore target protection");
	FlushInstructionCache(GetCurrentProcess(), runtime->target, target->patch_len);
}

static void install_virtual_call_site(uint32_t index)
{
	const struct VirtualCallSite *site = &g_virtual_call_sites[index];
	struct RuntimeVirtualCallSite *runtime = &g_virtual_runtime[index];
	DWORD old_protect;
	DWORD ignored;
	uint8_t *cursor;
	uint32_t i;
	char message[128];

	runtime->site = (uint8_t *)(uintptr_t)(
		g_module_base + (site->va - IMAGE_BASE));
	if (memcmp(runtime->site, site->expected, site->patch_len) != 0) {
		_snprintf(message, sizeof(message),
		          "virtual call-site mismatch %08X (%s)",
		          site->va, site->name);
		fatal(message);
	}
	build_virtual_call_site(index);
	if (!VirtualProtect(runtime->site, site->patch_len, PAGE_EXECUTE_READWRITE,
	                    &old_protect))
		fatal("failed to make virtual call-site writable");
	cursor = runtime->site;
	emit_rel32(&cursor, 0xE9, runtime->stub);
	for (i = 5; i < site->patch_len; ++i)
		runtime->site[i] = 0x90;
	if (!VirtualProtect(runtime->site, site->patch_len, old_protect, &ignored))
		fatal("failed to restore virtual call-site protection");
	FlushInstructionCache(
		GetCurrentProcess(), runtime->site, site->patch_len);
}

static void open_trace_file(
	uint32_t index,
	const char *trace_directory,
	uint32_t module_base)
{
	const struct TargetSpec *target = &g_targets[index];
	struct RuntimeTarget *runtime = &g_runtime[index];
	char path[MAX_PATH];
	char line[192];
	uint32_t zero = 0;

	runtime->target = (uint8_t *)(uintptr_t)(
		module_base + (target->va - IMAGE_BASE));
	if (is_internal_target(target->va)) {
		_snprintf(line, sizeof(line), "hook internal %08X runtime=%p name=%s",
		          target->va, runtime->target, target->name);
		log_line(line);
		return;
	}
	if (_snprintf(path, sizeof(path), "%s\\%08X_%s.bin", trace_directory,
	              target->va, target->name) < 0)
		fatal("trace path is too long");
	runtime->file = CreateFileA(path, GENERIC_WRITE, FILE_SHARE_READ, NULL,
	                            CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
	if (runtime->file == INVALID_HANDLE_VALUE)
		fatal("failed to create trace file");
	write_all(runtime->file, "TMNFTRC1", 8);
	write_all(runtime->file, &target->va, 4);
	write_all(runtime->file, &zero, 4);
	InitializeCriticalSection(&runtime->file_lock);
	_snprintf(line, sizeof(line), "hook %08X runtime=%p name=%s",
	          target->va, runtime->target, target->name);
	log_line(line);
}

static void probe_update_turbo(void)
{
	typedef void (__attribute__((thiscall)) *UpdateTurboFn)(
		void *car, uint32_t tick);
	static const struct {
		uint32_t progress;
		uint32_t start;
		uint32_t end;
		uint32_t type;
		uint32_t tick;
	} cases[] = {
		{ 0x3F800000u, 0u, 0u, 0u, 10u },
		{ 0u, 100u, 200u, 1u, 150u },
		{ 0u, 100u, 107u, 2u, 107u },
		{ 0x3F800000u, 100u, 107u, 1u, 108u },
		{ 0u, 0u, 0xFFFFFFFFu, 1u, 0x80000000u },
		{ 0u, 200u, 300u, 2u, 100u },
	};
	uint8_t *car;
	uint32_t i;
	uint32_t target_index = UINT32_MAX;

	for (i = 0; i < g_target_count; ++i) {
		if (g_targets[i].va == 0x007BC8B0u) {
			target_index = i;
			break;
		}
	}
	if (target_index == UINT32_MAX)
		return;
	car = (uint8_t *)HeapAlloc(g_heap, HEAP_ZERO_MEMORY, 0x878);
	if (car == NULL)
		fatal("UpdateTurbo probe allocation failed");
	for (i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
		memset(car, 0, 0x878);
		memcpy(car + 0x5F0, &cases[i].progress, 4);
		memcpy(car + 0x5F8, &cases[i].start, 4);
		memcpy(car + 0x5FC, &cases[i].end, 4);
		memcpy(car + 0x600, &cases[i].type, 4);
		((UpdateTurboFn)(uintptr_t)g_runtime[target_index].target)(
			car, cases[i].tick);
	}
	HeapFree(g_heap, 0, car);
}

static DWORD WINAPI initialize_tracer(void *unused)
{
	char trace_directory[MAX_PATH];
	char log_path[MAX_PATH];
	char record_start_text[32];
	char target_filter[512];
	char *record_start_end;
	HMODULE executable;
	uint32_t module_base;
	uint32_t i;
	DWORD attributes;
	DWORD filter_length;
	DWORD record_start_length;
	unsigned long record_start;
	(void)unused;

	if (GetEnvironmentVariableA(
		    "TMNF_TRACE_DIR", trace_directory, sizeof(trace_directory)) == 0)
		fatal("TMNF_TRACE_DIR is required");
	filter_length = GetEnvironmentVariableA(
		"TMNF_TRACE_TARGETS", target_filter, sizeof(target_filter));
	if (filter_length >= sizeof(target_filter))
		fatal("TMNF_TRACE_TARGETS is too long");
	g_filter_active = filter_length != 0;
	record_start_length = GetEnvironmentVariableA(
		"TMNF_TRACE_RECORD_START",
		record_start_text, sizeof(record_start_text));
	if (record_start_length != 0) {
		if (record_start_length >= sizeof(record_start_text))
			fatal("TMNF_TRACE_RECORD_START is invalid");
		record_start = strtoul(
			record_start_text, &record_start_end, 10);
		if (record_start_end == record_start_text
		    || *record_start_end != '\0'
		    || record_start > UINT32_MAX)
			fatal("TMNF_TRACE_RECORD_START is invalid");
		g_record_start = (uint32_t)record_start;
	}
	attributes = GetFileAttributesA(trace_directory);
	if (attributes == INVALID_FILE_ATTRIBUTES
	    || !(attributes & FILE_ATTRIBUTE_DIRECTORY))
		fatal("TMNF_TRACE_DIR is not a directory");
	_snprintf(log_path, sizeof(log_path), "%s\\tracer.log", trace_directory);
	g_log = CreateFileA(log_path, GENERIC_WRITE, FILE_SHARE_READ, NULL,
	                    CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
	if (g_log == INVALID_HANDLE_VALUE)
		fatal("failed to create tracer log");

	executable = GetModuleHandleA(NULL);
	if (executable == NULL)
		fatal("failed to locate TmForever module");
	module_base = (uint32_t)(uintptr_t)executable;
	g_module_base = module_base;
	g_heap = GetProcessHeap();
	response_capture_initialize(g_heap, module_base, fatal);
	model6_capture_initialize(g_heap, module_base, fatal, log_line);
	detect_capture_initialize(g_heap, module_base, fatal);
	broadphase_capture_initialize(g_heap, fatal);
	g_context_tls = TlsAlloc();
	if (g_context_tls == TLS_OUT_OF_INDEXES)
		fatal("failed to allocate hook TLS");
	if (g_target_count > sizeof(g_runtime) / sizeof(g_runtime[0]))
		fatal("too many tracer targets");

	for (i = 0; i < g_target_count; ++i) {
		uint32_t va = g_targets[i].va;
		g_target_enabled[i] = !g_filter_active
			|| filter_contains(target_filter, va)
			|| (is_internal_target(va)
			    && (filter_contains(target_filter, 0x00548BF0u)
			        || filter_contains(target_filter, 0x005497C0u)));
		if (!g_target_enabled[i])
			continue;
		open_trace_file(i, trace_directory, module_base);
		build_runtime_code(i);
	}
	InterlockedExchange(&g_ready, 1);
	for (i = 0; i < g_target_count; ++i) {
		if (g_target_enabled[i])
			install_patch(i);
	}
	if (!g_filter_active) {
		for (i = 0;
		     i < sizeof(g_virtual_call_sites)
		        / sizeof(g_virtual_call_sites[0]);
		     ++i)
			install_virtual_call_site(i);
		probe_update_turbo();
	}
	log_line("all hooks installed");
	return 0;
}

BOOL WINAPI DllMain(HINSTANCE instance, DWORD reason, void *reserved)
{
	uint32_t i;
	(void)reserved;
	if (reason == DLL_PROCESS_ATTACH) {
		HANDLE thread;
		DisableThreadLibraryCalls(instance);
		thread = CreateThread(NULL, 0, initialize_tracer, NULL, 0, NULL);
		if (thread == NULL)
			return FALSE;
		CloseHandle(thread);
	} else if (reason == DLL_PROCESS_DETACH) {
		for (i = 0; i < g_target_count; ++i) {
			if (g_runtime[i].file != NULL
			    && g_runtime[i].file != INVALID_HANDLE_VALUE)
				CloseHandle(g_runtime[i].file);
		}
		if (g_log != INVALID_HANDLE_VALUE)
			CloseHandle(g_log);
	}
	return TRUE;
}
