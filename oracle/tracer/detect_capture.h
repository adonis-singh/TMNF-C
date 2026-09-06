#ifndef TMNF_DETECT_CAPTURE_H
#define TMNF_DETECT_CAPTURE_H

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <stdint.h>

struct DetectCapture;

typedef void (*DetectCaptureFatalFn)(const char *message);

void detect_capture_initialize(
	HANDLE heap, uint32_t module_base, DetectCaptureFatalFn fatal_fn);
int detect_capture_is_target(uint32_t va);
int detect_capture_is_race_target(uint32_t va);
int detect_capture_race_ready(void);
struct DetectCapture *detect_capture_create(
	uint32_t va, uint32_t this_ptr, uint32_t entry_esp);
const uint8_t *detect_capture_input(
	const struct DetectCapture *capture, uint32_t *size);
uint8_t *detect_capture_output(
	struct DetectCapture *capture, uint32_t return_value, uint32_t *size);
void detect_capture_free_blob(uint8_t *blob);
void detect_capture_destroy(struct DetectCapture *capture);

#endif
