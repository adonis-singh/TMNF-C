#ifndef TMNF_BROADPHASE_CAPTURE_H
#define TMNF_BROADPHASE_CAPTURE_H

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <stdint.h>

struct BroadphaseCapture;

typedef void (*BroadphaseCaptureFatalFn)(const char *message);

void broadphase_capture_initialize(
	HANDLE heap, BroadphaseCaptureFatalFn fatal_fn);
int broadphase_capture_is_target(uint32_t va);
struct BroadphaseCapture *broadphase_capture_create(
	uint32_t va, uint32_t this_ptr, uint32_t entry_esp);
const uint8_t *broadphase_capture_input(
	const struct BroadphaseCapture *capture, uint32_t *size);
uint8_t *broadphase_capture_output(
	struct BroadphaseCapture *capture, uint32_t return_value, uint32_t *size);
void broadphase_capture_free_blob(uint8_t *blob);
void broadphase_capture_destroy(struct BroadphaseCapture *capture);

#endif
