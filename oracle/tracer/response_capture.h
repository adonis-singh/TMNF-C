#ifndef TMNF_RESPONSE_CAPTURE_H
#define TMNF_RESPONSE_CAPTURE_H

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <stdint.h>

struct ResponseCapture;

typedef void (*ResponseCaptureFatalFn)(const char *message);

void response_capture_initialize(
	HANDLE heap, uint32_t module_base, ResponseCaptureFatalFn fatal_fn);

struct ResponseCapture *response_capture_create(
	uint32_t kind, uint32_t this_ptr, uint32_t entry_esp);

const uint8_t *response_capture_input(
	const struct ResponseCapture *capture, uint32_t *size);

uint8_t *response_capture_output(
	struct ResponseCapture *capture, uint32_t *size);

uint32_t response_capture_event_enter(
	struct ResponseCapture *capture, uint32_t target_va,
	uint32_t item, uint32_t contact);

void response_capture_event_exit(
	struct ResponseCapture *capture, uint32_t event_index,
	uint32_t contact);

void response_capture_free_blob(uint8_t *blob);
void response_capture_destroy(struct ResponseCapture *capture);

#endif
