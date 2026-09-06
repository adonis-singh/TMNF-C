#ifndef TMNF_MODEL6_CAPTURE_H
#define TMNF_MODEL6_CAPTURE_H

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <stdint.h>

struct Model6Capture;
typedef void (*Model6CaptureFatalFn)(const char *message);

void model6_capture_initialize(
	HANDLE heap, uint32_t module_base, Model6CaptureFatalFn fatal_fn,
	Model6CaptureFatalFn log_fn);
struct Model6Capture *model6_capture_create(
	uint32_t this_ptr, uint32_t entry_esp);
const uint8_t *model6_capture_input(
	const struct Model6Capture *capture, uint32_t *size);
uint8_t *model6_capture_output(
	struct Model6Capture *capture, uint32_t *size);
void model6_capture_free_blob(uint8_t *blob);
void model6_capture_destroy(struct Model6Capture *capture);

#endif
