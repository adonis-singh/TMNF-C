#ifndef TMNF_COLLECTION_FFI_H
#define TMNF_COLLECTION_FFI_H

#include "vehicle.h"
#include "checkpoint_set.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TmnfCollection TmnfCollection;

/* Fixed-size, pointer-free export; time labels the state entering a tick,
 * matching the oracle capture convention (the initial state is 10 ms). */
typedef struct {
	uint32_t version; /* 2: unpacked <=64 flags; 3: packed <=2048 flags. Same byte layout. */
	uint32_t time_ms;
	uint32_t finished;
	uint32_t finish_time_ms;
	uint32_t checkpoint_count;
	uint32_t checkpoint_states[64]; /* v3: flag i is bit (i%32) of word (i/32). */
	uint32_t completed_laps;
	uint32_t respawn_count;
	CHmsStateDyna dyna;
	uint8_t car[2168];
	uint8_t wheels[3056];
} TmnfCollectionFrame;

_Static_assert(sizeof(((TmnfCollectionFrame *)0)->checkpoint_states) * 8 == TMNF_CHECKPOINT_CAPACITY,
	"packed collection flags must fit the complete checkpoint set");

const char *tmnf_collection_error(void);
size_t tmnf_collection_frame_size(void);
TmnfCollection *tmnf_collection_create(const char *track, const char *vehicle,
	const char *route, const uint8_t sha256[32], int cuda);
TmnfCollection *tmnf_collection_create_graph(const char *track, const char *vehicle,
	const char *route, const uint8_t sha256[32]);
TmnfCollection *tmnf_collection_create_config(const char *track, const char *vehicle,
	const char *route, const uint8_t sha256[32], int cuda, int initialize_from_graph,
	int build_grid, uint32_t validation_seed);
int tmnf_collection_reset_seed(TmnfCollection *env, uint32_t validation_seed);
/* Recorded ghost playback permits a respawn at the retained start pose even
 * before a respawnable checkpoint. Live racing still requires a restart. */
int tmnf_collection_set_replay_mode(TmnfCollection *env, int enabled);
/* Race-specific lap count, independent of shared map assets. Zero restores
 * the map default. Set before stepping; resets and clones retain the choice. */
int tmnf_collection_set_lap_count(TmnfCollection *env, uint32_t lap_count);
/* Replay-specific settled position; NULL restores the authored start. The
 * route's checkpoint/start respawn poses remain unchanged. */
int tmnf_collection_reset_position(TmnfCollection *env, uint32_t validation_seed,
	const float position[3]);
/* Borrowed race checkpoint times, valid until the next step/reset/destroy.
 * Includes lap finishes and the final finish, in acceptance order. */
uint32_t tmnf_collection_checkpoint_times(TmnfCollection *env, const uint32_t **times);
TmnfCollection *tmnf_collection_create_geometry(const char *track, const char *vehicle,
	const char *route, const char *triggers, const uint8_t sha256[32], int cuda,
	int initialize_from_graph, int build_grid, uint32_t validation_seed);
/* New independent CPU world sharing immutable track/route assets. The source
 * may be destroyed first; shared assets live until the final world is closed. */
TmnfCollection *tmnf_collection_clone(TmnfCollection *source);
void tmnf_collection_destroy(TmnfCollection *env);
int tmnf_collection_reset(TmnfCollection *env);
int tmnf_collection_observe(TmnfCollection *env, TmnfCollectionFrame *frame);
/* Returns the input-applied pre-step frame, then advances exactly one tick.
 * No autoreset, timeout, off-track termination, or observation normalization. */
int tmnf_collection_tick(TmnfCollection *env, const TMNFRaceInputs *inputs,
	TmnfCollectionFrame *frame);
/* Replay a bounded input chunk, returning selected input-applied pre-step
 * observations and their input indices. Sampling matches tick collection:
 * first state, applied-control changes, 100 ms, and terminal/last input.
 * Pass last_chunk only for the final source chunk. Reset clears sampling state.
 * frames/indices each have capacity >= count; final is the post-step state. */
int tmnf_collection_replay(TmnfCollection *env, const TMNFRaceInputs *inputs,
	uint32_t count, int every_tick, int last_chunk, TmnfCollectionFrame *frames,
	uint32_t *indices, uint32_t capacity, uint32_t *selected, uint32_t *executed,
	TmnfCollectionFrame *final);
/* Additionally retain any input whose corresponding sample_mask byte is
 * nonzero. The optional mask has count bytes; physics and ordinary sampling
 * are unchanged. The original replay entry point remains ABI-compatible. */
int tmnf_collection_replay_samples(TmnfCollection *env, const TMNFRaceInputs *inputs,
	uint32_t count, int every_tick, int last_chunk, const uint8_t *sample_mask,
	TmnfCollectionFrame *frames, uint32_t *indices, uint32_t capacity,
	uint32_t *selected, uint32_t *executed, TmnfCollectionFrame *final);
/* Advance an action's records, stopping at finish/timeout, and export only the
 * final observation. Validation can continue using per-tick pre-state export. */
int tmnf_collection_advance(TmnfCollection *env, const TMNFRaceInputs *inputs,
	uint32_t count, uint32_t timeout_ms, TmnfCollectionFrame *frame, uint32_t *executed);
/* Digital action bits match policy IDs: brake, accelerate, right, left. Write
 * executed input records for replay/debugging without per-tick Python calls. */
int tmnf_collection_action(TmnfCollection *env, uint32_t controls,
	uint32_t count, uint32_t timeout_ms, TmnfCollectionFrame *frame,
	TMNFRaceInputs records[10], uint32_t *executed);
/* Independent CPU worlds, evaluated in parallel. Records has count*10 entries;
 * frames/executed each have count entries. World handles must be distinct. */
int tmnf_collection_batch_action(TmnfCollection *const *envs,
	const uint32_t *controls, const uint32_t *ticks, uint32_t count,
	uint32_t threads, uint32_t timeout_ms, TmnfCollectionFrame *frames,
	TMNFRaceInputs *records, uint32_t *executed);

#ifdef __cplusplus
}
#endif

#endif
