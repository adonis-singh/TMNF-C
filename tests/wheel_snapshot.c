#include <stdio.h>
#include <string.h>

#include "vehicle.h"

/* Every byte of the v6 record, including padding and reset's retained half
 * words, must survive the smaller runtime representation. */
int main(void)
{
	uint32_t random = 0x18273ab1u;
	CSceneVehicleCarWheelHistory history[2];
	CSceneVehicleCarWheel wheel[2] = {0};
	int surface[2] = {0};
	for (unsigned i = 0; i < 2; ++i) {
		wheel[i].history = &history[i];
		wheel[i].surface_handler = &surface[i];
	}
	for (unsigned trial = 0; trial < 1024; ++trial) {
		CSceneVehicleCarWheelSnapshot input, output;
		unsigned char *bytes = (unsigned char *)&input;
		for (size_t i = 0; i < sizeof(input); ++i) {
			random ^= random << 13;
			random ^= random >> 17;
			random ^= random << 5;
			bytes[i] = (unsigned char)random;
		}
		input.surface_handler = NULL;
		unsigned destination = trial % 2;
		CSceneVehicleCarWheel_Restore(&wheel[destination], &input);
		CSceneVehicleCarWheel_Capture(&wheel[destination], &output);
		if (memcmp(&input, &output, sizeof(input)) != 0 ||
			wheel[destination].history != &history[destination] ||
			wheel[destination].surface_handler != &surface[destination]) {
			fprintf(stderr, "wheel snapshot: roundtrip failed at %u\n", trial);
			return 1;
		}
	}
	return 0;
}
