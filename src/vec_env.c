/* The RL layer (observation, reward, termination, autoreset, action
 * decoding, capture/restore) is mirrored by hand in src/cuda/tmnf_cuda_env.cu. Any
 * change here must be made there too;
 * tests/cuda_lockstep.c compares every result and snapshot byte. */
#define _GNU_SOURCE

#include "vec_env.h"

#include <errno.h>
#include <limits.h>
#include <linux/futex.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "vehicle_respawn.h"

const float TMNF_OBSERVATION_LOOKAHEAD_METERS[
	TMNF_OBSERVATION_LOOKAHEAD_COUNT] = {
	5.0f, 10.0f, 20.0f, 35.0f, 55.0f, 80.0f, 110.0f, 150.0f,
};

static void env_fail(const char *message)
{
	fprintf(stderr, "tmnf vec env: %s\n", message);
	abort();
}

typedef enum {
	TMNF_VEC_OPERATION_NONE = 0,
	TMNF_VEC_OPERATION_RESET,
	TMNF_VEC_OPERATION_STEP,
	TMNF_VEC_OPERATION_CAPTURE,
	TMNF_VEC_OPERATION_RESTORE,
	TMNF_VEC_OPERATION_SET_THREAD_COUNT,
	TMNF_VEC_OPERATION_DISPATCH_EMPTY,
	TMNF_VEC_OPERATION_DESTROY,
} TmnfVecOperation;

static const char *operation_name(uint32_t operation)
{
	switch (operation) {
	case TMNF_VEC_OPERATION_RESET:
		return "reset";
	case TMNF_VEC_OPERATION_STEP:
		return "step";
	case TMNF_VEC_OPERATION_CAPTURE:
		return "capture";
	case TMNF_VEC_OPERATION_RESTORE:
		return "restore";
	case TMNF_VEC_OPERATION_SET_THREAD_COUNT:
		return "set-thread-count";
	case TMNF_VEC_OPERATION_DISPATCH_EMPTY:
		return "empty-dispatch";
	case TMNF_VEC_OPERATION_DESTROY:
		return "destroy";
	default:
		return "unknown";
	}
}

static void begin_operation(
	TmnfVecEnv *env, TmnfVecOperation operation)
{
	uint32_t expected = TMNF_VEC_OPERATION_NONE;
	if (!atomic_compare_exchange_strong_explicit(
			&env->active_operation, &expected, operation,
			memory_order_acquire, memory_order_relaxed)) {
		fprintf(
			stderr,
			"tmnf vec env: %s overlaps active %s operation\n",
			operation_name(operation), operation_name(expected));
		abort();
	}
}

static void end_operation(TmnfVecEnv *env)
{
	atomic_store_explicit(
		&env->active_operation, TMNF_VEC_OPERATION_NONE,
		memory_order_release);
}

static void *env_allocate(size_t count, size_t size)
{
	if (count == 0 || size == 0 || count > SIZE_MAX / size)
		env_fail("invalid allocation size");
	void *memory = calloc(count, size);
	if (memory == NULL)
		env_fail("out of memory");
	return memory;
}

typedef void (*TmnfVecJob)(
	TmnfVecEnv *env, uint32_t begin, uint32_t end, void *context);

typedef struct TmnfVecWorkerPool TmnfVecWorkerPool;

typedef struct {
	TmnfVecWorkerPool *pool;
	uint32_t index;
} TmnfVecWorker;

struct TmnfVecWorkerPool {
	TmnfVecEnv *env;
	uint32_t thread_count;
	pthread_t *threads;
	TmnfVecWorker *workers;
	cpu_set_t *cpu_sets;
	cpu_set_t caller_original_cpu_set;
	int caller_was_pinned;
	TmnfVecJob job;
	void *context;
	_Alignas(64) _Atomic uint32_t generation;
	_Alignas(64) _Atomic uint32_t completed;
};

static cpu_set_t *parse_cpu_list(uint32_t thread_count)
{
	const char *text = getenv("TMNF_PIN_CPUS");
	if (text == NULL)
		return NULL;
	if (*text == '\0')
		env_fail("TMNF_PIN_CPUS is empty");

	cpu_set_t *sets = env_allocate(thread_count, sizeof(*sets));
	const char *cursor = text;
	for (uint32_t index = 0; index < thread_count; ++index) {
		if (*cursor < '0' || *cursor > '9')
			env_fail("TMNF_PIN_CPUS must be comma-separated CPU numbers");
		errno = 0;
		char *end = NULL;
		unsigned long cpu = strtoul(cursor, &end, 10);
		if (errno != 0 || end == cursor || cpu >= CPU_SETSIZE)
			env_fail("TMNF_PIN_CPUS contains an invalid CPU number");
		for (uint32_t previous = 0; previous < index; ++previous) {
			if (CPU_ISSET((int)cpu, &sets[previous]))
				env_fail("TMNF_PIN_CPUS contains a duplicate CPU");
		}
		CPU_ZERO(&sets[index]);
		CPU_SET((int)cpu, &sets[index]);
		if (index + 1 == thread_count) {
			if (*end != '\0')
				env_fail("TMNF_PIN_CPUS count differs from thread count");
		} else if (*end != ',') {
			env_fail("TMNF_PIN_CPUS count differs from thread count");
		}
		cursor = end + 1;
	}
	return sets;
}

static void pin_thread(const cpu_set_t *set)
{
	if (pthread_setaffinity_np(pthread_self(), sizeof(*set), set) != 0)
		env_fail("cannot pin worker thread");
}

static void futex_wait_for_change(
	_Atomic uint32_t *address, uint32_t expected)
{
	for (;;) {
		int result = (int)syscall(
			SYS_futex, (uint32_t *)address,
			FUTEX_WAIT_PRIVATE, expected, NULL, NULL, 0);
		if (result == 0 || errno == EAGAIN)
			return;
		if (errno != EINTR)
			env_fail("futex wait failed");
	}
}

static void futex_wake(_Atomic uint32_t *address, int count)
{
	if (syscall(
			SYS_futex, (uint32_t *)address,
			FUTEX_WAKE_PRIVATE, count, NULL, NULL, 0) < 0) {
		env_fail("futex wake failed");
	}
}

static void worker_range(
	const TmnfVecWorkerPool *pool, uint32_t worker_index,
	uint32_t *begin, uint32_t *end)
{
	uint64_t count = pool->env->count;
	*begin = (uint32_t)(
		count * worker_index / pool->thread_count);
	*end = (uint32_t)(
		count * (worker_index + 1) / pool->thread_count);
}

static void *worker_main(void *argument)
{
	TmnfVecWorker *worker = argument;
	TmnfVecWorkerPool *pool = worker->pool;
	if (pool->cpu_sets != NULL)
		pin_thread(&pool->cpu_sets[worker->index]);
	uint32_t seen_generation = 0;
	for (;;) {
		uint32_t generation = atomic_load_explicit(
			&pool->generation, memory_order_acquire);
		while (generation == seen_generation) {
			futex_wait_for_change(
				&pool->generation, seen_generation);
			generation = atomic_load_explicit(
				&pool->generation, memory_order_acquire);
		}
		seen_generation = generation;
		TmnfVecJob job = pool->job;
		if (job == NULL)
			return NULL;

		uint32_t begin;
		uint32_t end;
		worker_range(pool, worker->index, &begin, &end);
		job(pool->env, begin, end, pool->context);
		uint32_t completed = atomic_fetch_add_explicit(
			&pool->completed, 1, memory_order_release) + 1;
		if (completed == pool->thread_count - 1)
			futex_wake(&pool->completed, 1);
	}
}

static TmnfVecWorkerPool *worker_pool_create(
	TmnfVecEnv *env, uint32_t thread_count)
{
	if (thread_count == 0)
		env_fail("thread count must be positive");
	TmnfVecWorkerPool *pool =
		env_allocate(1, sizeof(*pool));
	pool->env = env;
	pool->thread_count = thread_count;
	atomic_init(&pool->generation, 0);
	atomic_init(&pool->completed, 0);
	pool->cpu_sets = parse_cpu_list(thread_count);
	if (pool->cpu_sets != NULL) {
		if (pthread_getaffinity_np(
				pthread_self(), sizeof(pool->caller_original_cpu_set),
				&pool->caller_original_cpu_set) != 0) {
			env_fail("cannot read caller thread affinity");
		}
		pin_thread(&pool->cpu_sets[0]);
		pool->caller_was_pinned = 1;
	}
	if (thread_count == 1)
		return pool;

	uint32_t background_count = thread_count - 1;
	pool->threads = env_allocate(
		background_count, sizeof(*pool->threads));
	pool->workers = env_allocate(
		background_count, sizeof(*pool->workers));
	for (uint32_t i = 0; i < background_count; ++i) {
		pool->workers[i].pool = pool;
		pool->workers[i].index = i + 1;
		if (pthread_create(
				&pool->threads[i], NULL, worker_main,
				&pool->workers[i]) != 0) {
			env_fail("cannot create worker thread");
		}
	}
	return pool;
}

static void worker_pool_destroy(TmnfVecWorkerPool *pool)
{
	if (pool == NULL)
		return;
	uint32_t background_count = pool->thread_count - 1;
	if (background_count != 0) {
		pool->job = NULL;
		atomic_fetch_add_explicit(
			&pool->generation, 1, memory_order_release);
		futex_wake(&pool->generation, INT_MAX);
		for (uint32_t i = 0; i < background_count; ++i) {
			if (pthread_join(pool->threads[i], NULL) != 0)
				env_fail("cannot join worker thread");
		}
	}
	free(pool->workers);
	free(pool->threads);
	if (pool->caller_was_pinned != 0 &&
		pthread_setaffinity_np(
			pthread_self(), sizeof(pool->caller_original_cpu_set),
			&pool->caller_original_cpu_set) != 0) {
		env_fail("cannot restore caller thread affinity");
	}
	free(pool->cpu_sets);
	free(pool);
}

static void dispatch_job(
	TmnfVecEnv *env, TmnfVecJob job, void *context)
{
	TmnfVecWorkerPool *pool = env->worker_pool;
	if (pool == NULL || job == NULL)
		env_fail("worker pool is not initialized");
	pool->job = job;
	pool->context = context;
	atomic_store_explicit(
		&pool->completed, 0, memory_order_relaxed);
	atomic_fetch_add_explicit(
		&pool->generation, 1, memory_order_release);
	if (pool->thread_count > 1)
		futex_wake(&pool->generation, INT_MAX);

	uint32_t begin;
	uint32_t end;
	worker_range(pool, 0, &begin, &end);
	job(env, begin, end, context);

	uint32_t expected = pool->thread_count - 1;
	uint32_t completed = atomic_load_explicit(
		&pool->completed, memory_order_acquire);
	while (completed != expected) {
		futex_wait_for_change(&pool->completed, completed);
		completed = atomic_load_explicit(
			&pool->completed, memory_order_acquire);
	}
}

static TmnfPhysicsCorpus *player_corpus(
	const TmnfVecEnv *env, uint32_t index)
{
	return &env->worlds[index]->corpora[
		env->player_corpus_indices[index]];
}

static void write_physics_observation(
	const TmnfPhysicsCorpus *corpus, TmnfObservation *out)
{
	const CHmsStateDyna *state = corpus->dyna->liveState;
	const CSceneVehicleCar *car = corpus->vehicle;

	memset(out, 0, sizeof(*out));
	out->position = state->pos;
	out->rotation = state->quat;
	out->linear_speed = state->linVel;
	out->angular_speed = state->angVel;
	for (uint32_t i = 0; i < TMNF_STADIUM_WHEEL_COUNT; ++i) {
		out->wheel_speed[i] = car->wheels[i].real_time.field6c;
		out->wheel_damper[i] =
			car->wheels[i].real_time.damper_absorb;
		out->wheel_contact[i] = (float)
			car->wheels[i].real_time.has_ground_contact;
		out->wheel_sliding[i] = (float)
			car->wheels[i].real_time.is_sliding;
		/* The game stores the material as a uint16 in a 32-bit word whose
		 * upper half keeps unrelated bytes; every port reader masks it. */
		out->wheel_material[i] = (float)(uint16_t)
			car->wheels[i].real_time.contact_material_id;
	}
	out->engine_rpm = car->engine.rpm;
	out->gear = car->engine.gear;
	out->input_steer = car->input_steer;
	out->input_gas = car->input_gas;
	out->input_brake = car->input_brake;
}

static TmnfCenterlineSample centerline_sample_at(
	const TmnfRoute *route, float arc_length, uint32_t minimum_segment)
{
	const TmnfRouteReferencePoint *points =
		TmnfRoute_GetReferencePoints(route);
	uint32_t count = TmnfRoute_GetReferencePointCount(route);
	if (count < 2 || minimum_segment + 1 >= count)
		env_fail("centerline sample cursor is invalid");
	if (arc_length < points[minimum_segment].arc_length)
		env_fail("centerline sample precedes projection cursor");
	if (arc_length >= points[count - 1].arc_length) {
		return (TmnfCenterlineSample){
			.position = points[count - 1].position,
			.half_width = points[count - 1].half_width,
		};
	}

	uint32_t lower = minimum_segment;
	uint32_t upper = count - 1;
	while (lower + 1 < upper) {
		uint32_t middle = lower + (upper - lower) / 2;
		if (points[middle].arc_length <= arc_length)
			lower = middle;
		else
			upper = middle;
	}
	const TmnfRouteReferencePoint *a = &points[lower];
	const TmnfRouteReferencePoint *b = &points[lower + 1];
	float t = (arc_length - a->arc_length) /
		(b->arc_length - a->arc_length);
	return (TmnfCenterlineSample){
		.position = {
			a->position.x + t * (b->position.x - a->position.x),
			a->position.y + t * (b->position.y - a->position.y),
			a->position.z + t * (b->position.z - a->position.z),
		},
		.half_width = a->half_width +
			t * (b->half_width - a->half_width),
	};
}

static void write_rl_observation(
	const TmnfVecEnv *env, uint32_t index, TmnfObservation *out)
{
	const TmnfPhysicsCorpus *corpus = player_corpus(env, index);
	const TmnfRaceState *race = &env->race_states[index];
	const CHmsStateDyna *state = corpus->dyna->liveState;
	const CSceneVehicleCarAuxContext *aux =
		corpus->vehicle_compute->aux;
	write_physics_observation(corpus, out);
	if (aux->turbo_type != TMNF_TURBO_NONE) {
		out->turbo_active = 1.0f;
		out->turbo_type = (float)aux->turbo_type;
		out->turbo_remaining_progress =
			1.0f - aux->turbo_progress;
	}

	float route_length = TmnfRoute_GetReferenceLength(env->route);
	float total_length = route_length * env->route->metadata->lap_count;
	out->arc_length = race->arc_length;
	out->unwrapped_progress = race->unwrapped_progress;
	out->lateral_offset = race->lateral_offset;
	out->track_half_width = race->half_width;
	out->remaining_distance = total_length - race->unwrapped_progress;
	out->elapsed_fraction = (float)race->elapsed_ticks /
		(float)env->config.max_race_ticks;
	/* With zero intermediate checkpoints the fraction is defined as 1.0:
	 * every required checkpoint has been passed. Dividing would yield NaN
	 * and silently poison training observations. */
	out->next_checkpoint_fraction =
		env->route->metadata->checkpoint_count == 0
			? 1.0f
			: (float)race->visited_count /
				(float)env->route->metadata->checkpoint_count;
	out->completed_lap_fraction = (float)race->completed_laps /
		(float)env->route->metadata->lap_count;

	GmIso4 inverse_car;
	GmIso4_SetInverse(
		&inverse_car, (const GmIso4 *)&state->rot);
	for (uint32_t i = 0;
		i < TMNF_OBSERVATION_LOOKAHEAD_COUNT; ++i) {
		float sample_arc = race->arc_length +
			TMNF_OBSERVATION_LOOKAHEAD_METERS[i];
		TmnfCenterlineSample sample = centerline_sample_at(
			env->route, sample_arc, race->centerline_segment);
		GmVec3_SetMult_Iso4(
			&out->centerline_lookahead[i].position,
			&sample.position, &inverse_car);
		out->centerline_lookahead[i].half_width =
			sample.half_width;
	}
}

static TMNFVehicleComputeContext *vehicle_context(
	const TmnfPhysicsCorpus *corpus)
{
	TMNFVehicleComputeContext *compute = corpus->vehicle_compute;
	if (compute == NULL || compute->aux == NULL ||
		compute->contact == NULL || compute->model6 == NULL ||
		compute->state == NULL || compute->aux->wheels == NULL ||
		compute->contact->wheels == NULL ||
		compute->model6->state == NULL ||
		compute->contact->timer == NULL ||
		compute->contact->wheel_count != TMNF_STADIUM_WHEEL_COUNT) {
		env_fail("vehicle context graph is incomplete");
	}
	return compute;
}

static CHmsResponseBody *resolve_contact_body(
	const TmnfVecEnv *env, uint32_t index, uint32_t body_ref)
{
	const CHmsResponseZone *zone = &env->worlds[index]->response_zone;
	if (zone->resolve_body == NULL)
		env_fail("world has no response body resolver");
	return zone->resolve_body(zone->resolver_user, body_ref);
}

/* Wheel contact bodies are the only state pointers in the vehicle graph. They
 * travel as corpus_ref tokens, which every world resolves to its own body. */
static uint32_t contact_body_ref(
	const TmnfVecEnv *env, uint32_t index, const CHmsResponseBody *body)
{
	if (body == NULL)
		return TMNF_ENV_NO_CONTACT_BODY;
	if (resolve_contact_body(env, index, body->corpus_ref) != body)
		env_fail("wheel contact body is not addressable by its token");
	return body->corpus_ref;
}

/* Every pointer inside the copied structs is topology owned by the world;
 * none of it may leave the environment inside a snapshot. */
static void scrub_car_pointers(CSceneVehicleCar *car)
{
	car->hms_item = NULL;
	car->dyna_state = NULL;
	car->dyna_params = NULL;
	car->tuning = NULL;
	car->wheels = NULL;
}

static void scrub_aux_pointers(CSceneVehicleCarAuxContext *aux)
{
	aux->vehicle = NULL;
	aux->tuning = NULL;
	aux->wheels = NULL;
	aux->runtime = NULL;
	aux->play_turbo_sound = NULL;
	aux->set_surface_location = NULL;
	aux->finish_integration = NULL;
}

static void scrub_contact_pointers(TMNFVehicleContactContext *contact)
{
	contact->vehicle = NULL;
	contact->tuning = NULL;
	contact->wheels = NULL;
	contact->ground_material_indices = NULL;
	contact->ground_materials = NULL;
	contact->timer = NULL;
	contact->vehicle_contact_rotation = NULL;
	contact->resolve_body_iso = NULL;
	contact->resolve_body_iso_user = NULL;
	contact->wheel_tree_refs = NULL;
	contact->body_tree_refs = NULL;
	contact->air_control_immediate = NULL;
	contact->contact_block_count = NULL;
	contact->event_source_c = NULL;
	contact->event_source_ab = NULL;
	contact->event_metric_a = NULL;
	contact->event_metric_b = NULL;
	contact->event_metric_c = NULL;
}

static void capture_physics(
	const TmnfVecEnv *env, uint32_t index, TmnfEnvSnapshot *snapshot)
{
	const TmnfPhysicsCorpus *corpus = player_corpus(env, index);
	const CHmsDyna *dyna = corpus->dyna;
	const CSceneVehicleCar *car = corpus->vehicle;
	const TMNFVehicleComputeContext *compute = vehicle_context(corpus);

	memset(snapshot, 0, sizeof(*snapshot));
	snapshot->version = TMNF_ENV_SNAPSHOT_VERSION;
	snapshot->live_state = *dyna->liveState;
	snapshot->committed_state = *dyna->stateB;
	snapshot->temp_state = dyna->tempState;
	snapshot->dyna_params = *dyna->params;
	snapshot->dyna_dirty = dyna->dirtyFlag;

	snapshot->car = *car;
	scrub_car_pointers(&snapshot->car);
	for (uint32_t i = 0; i < TMNF_STADIUM_WHEEL_COUNT; ++i)
		CSceneVehicleCarWheel_Capture(&car->wheels[i], &snapshot->wheels[i]);

	snapshot->aux = *compute->aux;
	scrub_aux_pointers(&snapshot->aux);
	memcpy(snapshot->aux_wheels, compute->aux->wheels,
		sizeof(snapshot->aux_wheels));

	snapshot->contact = *compute->contact;
	scrub_contact_pointers(&snapshot->contact);
	for (uint32_t i = 0; i < TMNF_STADIUM_WHEEL_COUNT; ++i) {
		const TMNFVehicleContactWheelState *wheel =
			&compute->contact->wheels[i];
		snapshot->contact_wheels[i] = *wheel;
		snapshot->contact_wheels[i].wheel = NULL;
		snapshot->contact_wheels[i].contact_body = NULL;
		snapshot->contact_body_refs[i] =
			contact_body_ref(env, index, wheel->contact_body);
	}

	snapshot->model6_state = *compute->model6->state;
	snapshot->compute_state = *compute->state;
	snapshot->contact_timer_tick = compute->contact->timer->tick_time;
	snapshot->physics_tick_time = corpus->tick_time;
	snapshot->vehicle_context_valid = 1;
}

static void restore_physics(
	TmnfVecEnv *env, uint32_t index, const TmnfEnvSnapshot *snapshot)
{
	if (snapshot->version != TMNF_ENV_SNAPSHOT_VERSION)
		env_fail("snapshot version is unsupported");
	if (snapshot->vehicle_context_valid == 0)
		env_fail("snapshot has no vehicle context state");
	TmnfPhysicsCorpus *corpus = player_corpus(env, index);
	CHmsDyna *dyna = corpus->dyna;
	CSceneVehicleCar *car = corpus->vehicle;
	TMNFVehicleComputeContext *compute = vehicle_context(corpus);
	CSceneVehicleCarAuxContext *aux = compute->aux;
	TMNFVehicleContactContext *contact = compute->contact;

	/* The destination world's topology is the only source of pointers. */
	CSceneVehicleCar car_topology = *car;
	CSceneVehicleCarAuxContext aux_topology = *aux;
	TMNFVehicleContactContext contact_topology = *contact;
	void *surface_handlers[TMNF_STADIUM_WHEEL_COUNT];
	CSceneVehicleCarWheel *contact_wheel_refs[TMNF_STADIUM_WHEEL_COUNT];
	for (uint32_t i = 0; i < TMNF_STADIUM_WHEEL_COUNT; ++i) {
		surface_handlers[i] = car->wheels[i].surface_handler;
		contact_wheel_refs[i] = contact->wheels[i].wheel;
	}

	*dyna->liveState = snapshot->live_state;
	*dyna->stateB = snapshot->committed_state;
	dyna->tempState = snapshot->temp_state;
	*dyna->params = snapshot->dyna_params;
	dyna->replacementBuf.count = 0;
	dyna->dirtyFlag = snapshot->dyna_dirty;

	*car = snapshot->car;
	car->hms_item = car_topology.hms_item;
	car->dyna_state = car_topology.dyna_state;
	car->dyna_params = car_topology.dyna_params;
	car->tuning = car_topology.tuning;
	car->wheels = car_topology.wheels;
	for (uint32_t i = 0; i < TMNF_STADIUM_WHEEL_COUNT; ++i) {
		CSceneVehicleCarWheel_Restore(&car->wheels[i], &snapshot->wheels[i]);
		car->wheels[i].surface_handler = surface_handlers[i];
	}

	*aux = snapshot->aux;
	aux->vehicle = aux_topology.vehicle;
	aux->tuning = aux_topology.tuning;
	aux->wheels = aux_topology.wheels;
	aux->runtime = aux_topology.runtime;
	aux->play_turbo_sound = aux_topology.play_turbo_sound;
	aux->set_surface_location = aux_topology.set_surface_location;
	aux->finish_integration = aux_topology.finish_integration;
	memcpy(aux->wheels, snapshot->aux_wheels, sizeof(snapshot->aux_wheels));

	*contact = snapshot->contact;
	contact->vehicle = contact_topology.vehicle;
	contact->tuning = contact_topology.tuning;
	contact->wheels = contact_topology.wheels;
	contact->ground_material_indices =
		contact_topology.ground_material_indices;
	contact->ground_materials = contact_topology.ground_materials;
	contact->timer = contact_topology.timer;
	contact->vehicle_contact_rotation =
		contact_topology.vehicle_contact_rotation;
	contact->resolve_body_iso = contact_topology.resolve_body_iso;
	contact->resolve_body_iso_user = contact_topology.resolve_body_iso_user;
	contact->wheel_tree_refs = contact_topology.wheel_tree_refs;
	contact->body_tree_refs = contact_topology.body_tree_refs;
	contact->air_control_immediate = contact_topology.air_control_immediate;
	contact->contact_block_count = contact_topology.contact_block_count;
	contact->event_source_c = contact_topology.event_source_c;
	contact->event_source_ab = contact_topology.event_source_ab;
	contact->event_metric_a = contact_topology.event_metric_a;
	contact->event_metric_b = contact_topology.event_metric_b;
	contact->event_metric_c = contact_topology.event_metric_c;
	for (uint32_t i = 0; i < TMNF_STADIUM_WHEEL_COUNT; ++i) {
		TMNFVehicleContactWheelState *wheel = &contact->wheels[i];
		*wheel = snapshot->contact_wheels[i];
		wheel->wheel = contact_wheel_refs[i];
		uint32_t body_ref = snapshot->contact_body_refs[i];
		if (body_ref == TMNF_ENV_NO_CONTACT_BODY) {
			wheel->contact_body = NULL;
			continue;
		}
		wheel->contact_body = resolve_contact_body(env, index, body_ref);
		if (wheel->contact_body == NULL)
			env_fail("snapshot contact body has no counterpart here");
	}

	*compute->model6->state = snapshot->model6_state;
	*compute->state = snapshot->compute_state;
	contact->timer->tick_time = snapshot->contact_timer_tick;
	corpus->tick_time = snapshot->physics_tick_time;

	for (uint32_t i = 0; i < TMNF_STADIUM_WHEEL_COUNT; ++i) {
		CPlugTree *tree = car->wheels[i].surface_handler;
		if (tree == NULL)
			env_fail("wheel surface tree is null");
		tree->local_iso = aux->wheels[i].surface_location;
		tree->box.center = (GmVec3){
			aux->wheels[i].surface_location.t[0],
			aux->wheels[i].surface_location.t[1],
			aux->wheels[i].surface_location.t[2],
		};
	}
}

static void validate_config(const TmnfVecEnvConfig *config)
{
	if (config == NULL ||
		config->off_track_grace_ticks == 0 ||
		config->stuck_grace_ticks == 0 ||
		config->thread_count == 0 ||
		!(config->discount_per_tick > 0.0f) ||
		!(config->discount_per_tick <= 1.0f) ||
		!(config->reference_speed > 0.0f) ||
		!(config->stuck_progress_epsilon > 0.0f) ||
		!isfinite(config->discount_per_tick) ||
		!isfinite(config->reference_speed) ||
		!isfinite(config->stuck_progress_epsilon) ||
		(config->autoreset_mode != TMNF_AUTORESET_SAME_STEP &&
		 config->autoreset_mode != TMNF_AUTORESET_NEXT_STEP) ||
		(config->action_space != TMNF_ACTION_SPACE_DISCRETE &&
		 config->action_space != TMNF_ACTION_SPACE_ANALOG)) {
		env_fail("invalid configuration");
	}
}

/*
 * A race that cannot finish inside a tick budget is a configuration error.
 * The game's rigid-body speed cap is 277.777... m/s (1,000 km/h,
 * scene_mobil.max_linear_speed in every capture); no lap can cover the route
 * faster than that.
 */
static uint32_t minimum_finish_ticks(const TmnfRoute *route)
{
	double total_length = (double)TmnfRoute_GetReferenceLength(route) *
		route->metadata->lap_count;
	double ticks = ceil(total_length / TMNF_MAX_LINEAR_SPEED_MPS /
		(TMNF_RACE_TICK_MS / 1000.0));
	if (ticks < 1.0)
		ticks = 1.0;
	return (uint32_t)ticks;
}

uint32_t TmnfVecEnv_DerivePaceTicks(
	const TmnfRoute *route, float reference_speed)
{
	if (route == NULL || !(reference_speed > 0.0f) || !isfinite(reference_speed))
		env_fail("invalid budget reference speed");
	double total_length = (double)TmnfRoute_GetReferenceLength(route) *
		route->metadata->lap_count;
	double ticks = ceil(
		TMNF_HORIZON_PACE_FACTOR * total_length /
		(double)reference_speed /
		(TMNF_RACE_TICK_MS / 1000.0));
	if (ticks > (double)UINT32_MAX)
		env_fail("derived race budget overflows");
	return (uint32_t)ticks;
}

/* Lowest centerline sample. A car more than one block below it has left the
 * track for good (A04's pool beside the start swallows a floating car for the
 * whole race budget otherwise). */
static float route_floor_y(const TmnfRoute *route)
{
	uint32_t count = TmnfRoute_GetReferencePointCount(route);
	const TmnfRouteReferencePoint *points =
		TmnfRoute_GetReferencePoints(route);
	if (count == 0)
		env_fail("route has no centerline samples");
	float floor_y = points[0].position.y;
	for (uint32_t i = 1; i < count; ++i) {
		if (points[i].position.y < floor_y)
			floor_y = points[i].position.y;
	}
	if (!isfinite(floor_y))
		env_fail("route centerline height is not finite");
	return floor_y;
}

static void resolve_race_budget(TmnfVecEnv *env)
{
	TmnfVecEnvConfig *config = &env->config;
	uint32_t minimum = minimum_finish_ticks(env->route);
	uint32_t pace = TmnfVecEnv_DerivePaceTicks(env->route, config->reference_speed);
	if (config->max_race_ticks == 0)
		config->max_race_ticks = pace;
	if (config->horizon_ticks == 0)
		config->horizon_ticks = pace;
	if (config->max_race_ticks < minimum) {
		fprintf(stderr,
			"tmnf vec env: max_race_ticks %u cannot finish a %.0f m race "
			"(needs at least %u ticks at the %.1f m/s speed cap)\n",
			config->max_race_ticks,
			(double)TmnfRoute_GetReferenceLength(env->route) *
				env->route->metadata->lap_count,
			minimum, TMNF_MAX_LINEAR_SPEED_MPS);
		env_fail("race timeout makes a finish impossible");
	}
	if (config->horizon_ticks < minimum) {
		fprintf(stderr,
			"tmnf vec env: horizon_ticks %u cannot finish a %.0f m race "
			"(needs at least %u ticks at the %.1f m/s speed cap)\n",
			config->horizon_ticks,
			(double)TmnfRoute_GetReferenceLength(env->route) *
				env->route->metadata->lap_count,
			minimum, TMNF_MAX_LINEAR_SPEED_MPS);
		env_fail("collection horizon makes a finish impossible");
	}
}

TmnfVecEnvConfig TmnfVecEnv_DefaultConfig(void)
{
	return (TmnfVecEnvConfig){
		.max_race_ticks = 0,
		.horizon_ticks = 0,
		.off_track_grace_ticks = 100,
		.stuck_grace_ticks = 500,
		.thread_count = 1,
		.discount_per_tick = 0.9999f,
		.reference_speed = 50.0f,
		.stuck_progress_epsilon = 0.001f,
		.autoreset_mode = TMNF_AUTORESET_SAME_STEP,
		.action_space = TMNF_ACTION_SPACE_DISCRETE,
		.respawn_action = 0,
	};
}

static void validate_worlds(TmnfVecEnv *env)
{
	for (uint32_t i = 0; i < env->count; ++i) {
		if (env->worlds[i] == NULL ||
			env->player_corpus_indices[i] >=
				env->worlds[i]->corpus_count) {
			env_fail("invalid player world");
		}
		TmnfPhysicsCorpus *corpus = player_corpus(env, i);
		if (corpus->dyna == NULL || corpus->dyna->liveState == NULL ||
			corpus->dyna->stateB == NULL ||
			corpus->dyna->params == NULL || corpus->vehicle == NULL ||
			corpus->vehicle->wheels == NULL ||
			corpus->vehicle->wheel_count !=
				TMNF_STADIUM_WHEEL_COUNT ||
			corpus->collision_corpus == NULL ||
			corpus->collision_corpus->tree == NULL ||
			corpus->vehicle_compute == NULL) {
			env_fail("player world lacks race geometry or state");
		}
		/* The physics step tests the route's triggers in its detection
		 * passes (TmnfPhysicsWorld.route, TmnfPhysicsWorld.trigger_contacts). */
		env->worlds[i]->route = env->route;
	}
}

/*
 * The reset state is the world's spawn state as captured at Init: the game's
 * state after the spawn tick (record 0 of every oracle capture, race time
 * 10 ms), which is where replay_tick starts stepping too. The route's
 * initial_state is the pre-tick spawn with zero velocities; stepping from it
 * would re-run the spawn tick and put the env one tick behind the game.
 *
 * The spawn tick leaves the car airborne for the compute stage, so the
 * CHmsDynaParams it writes for the next tick (forceFieldScale, dragLinear)
 * hold the airborne values; every later tick writes the grounded ones. They
 * are part of the snapshot so a reset does not inherit the previous episode's
 * gravity scale and drag (F2b, 0.5 mm at tick 2).
 */
static void validate_spawn(const TmnfVecEnv *env, uint32_t index)
{
	const CHmsStateDyna *spawn = &env->reset_snapshots[index].live_state;
	const TmnfRouteInitialState *initial =
		&env->route->start->initial_state;
	if (memcmp(&spawn->pos, &initial->pos, sizeof(initial->pos)) != 0 ||
		memcmp(&spawn->rot, &initial->rot, sizeof(initial->rot)) != 0) {
		env_fail("world spawn does not match the route start");
	}
}

static void reset_one(TmnfVecEnv *env, uint32_t index)
{
	TmnfPhysicsCorpus *corpus = player_corpus(env, index);
	restore_physics(env, index, &env->reset_snapshots[index]);

	env->episode_ids[index]++;
	env->episode_returns[index] = 0.0f;
	env->reset_pending[index] = 0;
	TmnfRace_Reset(
		env->route, &env->race_states[index],
		corpus->collision_corpus->live_iso);
}

void TmnfVecEnv_Init(
	TmnfVecEnv *env,
	TmnfPhysicsWorld **worlds,
	uint32_t *player_corpus_indices,
	uint32_t count,
	const TmnfRoute *route,
	const TmnfVecEnvConfig *config)
{
	if (env == NULL || worlds == NULL ||
		player_corpus_indices == NULL || count == 0 || route == NULL) {
		env_fail("initialization argument is null or empty");
	}
	validate_config(config);
	memset(env, 0, sizeof(*env));
	atomic_init(&env->active_operation, TMNF_VEC_OPERATION_NONE);
	env->worlds = worlds;
	env->player_corpus_indices = player_corpus_indices;
	env->count = count;
	env->route = route;
	env->config = *config;
	resolve_race_budget(env);
	env->fell_floor_y = route_floor_y(route) - TMNF_FELL_MARGIN_METERS;
	validate_worlds(env);

	env->reset_snapshots =
		env_allocate(count, sizeof(*env->reset_snapshots));
	for (uint32_t i = 0; i < count; ++i) {
		capture_physics(env, i, &env->reset_snapshots[i]);
		validate_spawn(env, i);
	}
	env->race_states = env_allocate(count, sizeof(*env->race_states));
	env->episode_ids = env_allocate(count, sizeof(*env->episode_ids));
	env->episode_returns =
		env_allocate(count, sizeof(*env->episode_returns));
	env->reset_pending = env_allocate(count, sizeof(*env->reset_pending));
	TmnfVecEnv_Reset(env, NULL, NULL);
	env->worker_pool = worker_pool_create(env, config->thread_count);
}

void TmnfVecEnv_Destroy(TmnfVecEnv *env)
{
	if (env == NULL)
		return;
	begin_operation(env, TMNF_VEC_OPERATION_DESTROY);
	worker_pool_destroy(env->worker_pool);
	free(env->reset_pending);
	free(env->episode_returns);
	free(env->episode_ids);
	free(env->race_states);
	free(env->reset_snapshots);
	env->worker_pool = NULL;
	env->reset_pending = NULL;
	env->episode_returns = NULL;
	env->episode_ids = NULL;
	env->race_states = NULL;
	env->reset_snapshots = NULL;
	env->worlds = NULL;
	env->player_corpus_indices = NULL;
	env->route = NULL;
	env->count = 0;
	memset(&env->config, 0, sizeof(env->config));
	end_operation(env);
}

void TmnfVecEnv_SetThreadCount(TmnfVecEnv *env, uint32_t thread_count)
{
	if (env == NULL || thread_count == 0)
		env_fail("cannot set thread count");
	begin_operation(env, TMNF_VEC_OPERATION_SET_THREAD_COUNT);
	if (env->worker_pool == NULL) {
		env->worker_pool = worker_pool_create(env, thread_count);
		env->config.thread_count = thread_count;
		end_operation(env);
		return;
	}
	TmnfVecWorkerPool *old_pool = env->worker_pool;
	if (old_pool->thread_count == thread_count) {
		end_operation(env);
		return;
	}
	env->worker_pool = NULL;
	worker_pool_destroy(old_pool);
	env->worker_pool = worker_pool_create(env, thread_count);
	env->config.thread_count = thread_count;
	end_operation(env);
}

void TmnfVecEnv_Reset(
	TmnfVecEnv *env,
	const uint8_t *mask,
	TmnfObservation *observations)
{
	if (env == NULL)
		env_fail("environment is not initialized");
	begin_operation(env, TMNF_VEC_OPERATION_RESET);
	if (env->route == NULL || env->race_states == NULL)
		env_fail("environment is not initialized");
	for (uint32_t i = 0; i < env->count; ++i) {
		if (mask == NULL || mask[i] != 0)
			reset_one(env, i);
		if (observations != NULL)
			write_rl_observation(env, i, &observations[i]);
	}
	end_operation(env);
}

static void capture_env_state(
	TmnfVecEnv *env, uint32_t index, TmnfEnvSnapshot *snapshot)
{
	capture_physics(env, index, snapshot);
	if (env->race_states != NULL) {
		snapshot->race_state = env->race_states[index];
		snapshot->episode_id = env->episode_ids[index];
		snapshot->episode_return = env->episode_returns[index];
		snapshot->reset_pending = env->reset_pending[index];
		snapshot->race_state_valid = 1;
	}
}

static void restore_env_state(
	TmnfVecEnv *env, uint32_t index, const TmnfEnvSnapshot *snapshot)
{
	restore_physics(env, index, snapshot);
	if (env->race_states != NULL) {
		if (snapshot->race_state_valid == 0)
			env_fail("snapshot has no race state");
		env->race_states[index] = snapshot->race_state;
		env->episode_ids[index] = snapshot->episode_id;
		env->episode_returns[index] = snapshot->episode_return;
		env->reset_pending[index] = snapshot->reset_pending;
	}
}

static void validate_snapshot_indices(
	const TmnfVecEnv *env, const uint32_t *indices, uint32_t count)
{
	if (indices == NULL || count == 0)
		env_fail("snapshot indices are null or empty");
	for (uint32_t i = 0; i < count; ++i) {
		if (indices[i] >= env->count)
			env_fail("snapshot index is out of range");
		for (uint32_t j = 0; j < i; ++j) {
			if (indices[i] == indices[j])
				env_fail("snapshot indices contain a duplicate");
		}
	}
}

void TmnfVecEnv_Capture(
	TmnfVecEnv *env, TmnfEnvSnapshot *snapshots)
{
	if (env == NULL || snapshots == NULL)
		env_fail("capture argument is null");
	begin_operation(env, TMNF_VEC_OPERATION_CAPTURE);
	for (uint32_t i = 0; i < env->count; ++i)
		capture_env_state(env, i, &snapshots[i]);
	end_operation(env);
}

void TmnfVecEnv_Restore(
	TmnfVecEnv *env, const TmnfEnvSnapshot *snapshots)
{
	if (env == NULL || snapshots == NULL)
		env_fail("restore argument is null");
	begin_operation(env, TMNF_VEC_OPERATION_RESTORE);
	for (uint32_t i = 0; i < env->count; ++i)
		restore_env_state(env, i, &snapshots[i]);
	end_operation(env);
}

void TmnfVecEnv_CaptureIndices(
	TmnfVecEnv *env, const uint32_t *indices, uint32_t count,
	TmnfEnvSnapshot *snapshots)
{
	if (env == NULL || snapshots == NULL)
		env_fail("indexed capture argument is null");
	begin_operation(env, TMNF_VEC_OPERATION_CAPTURE);
	validate_snapshot_indices(env, indices, count);
	for (uint32_t i = 0; i < count; ++i)
		capture_env_state(env, indices[i], &snapshots[i]);
	end_operation(env);
}

void TmnfVecEnv_RestoreIndices(
	TmnfVecEnv *env, const uint32_t *indices, uint32_t count,
	const TmnfEnvSnapshot *snapshots, TmnfObservation *observations)
{
	if (env == NULL || snapshots == NULL)
		env_fail("indexed restore argument is null");
	begin_operation(env, TMNF_VEC_OPERATION_RESTORE);
	validate_snapshot_indices(env, indices, count);
	for (uint32_t i = 0; i < count; ++i) {
		restore_env_state(env, indices[i], &snapshots[i]);
		if (observations != NULL)
			write_rl_observation(env, indices[i], &observations[i]);
	}
	end_operation(env);
}

void TmnfPhysicsCorpus_AdvanceTimer(
	TmnfPhysicsCorpus *corpus, uint32_t tick_ms)
{
	if (corpus == NULL || corpus->vehicle_compute == NULL ||
		corpus->vehicle_compute->contact == NULL ||
		corpus->vehicle_compute->contact->timer == NULL) {
		env_fail("corpus has no game timer");
	}
	TMNFVehicleContactTimer *timer =
		corpus->vehicle_compute->contact->timer;
	timer->tick_time += tick_ms;
	corpus->tick_time = timer->tick_time;
}

/* One game tick exactly as tests/replay_tick.c drives the World: a respawn
 * press is applied first (0x0047DCD0 OnInputEvent runs before the tick's
 * control mapping), inputs are applied to the vehicle, the timer advances,
 * then the zone steps. */
static void physics_tick(
	TmnfVecEnv *env, uint32_t index, const TMNFRaceInputs *input,
	uint32_t tick_ms)
{
	TmnfPhysicsCorpus *corpus = player_corpus(env, index);
	if (input->respawn != 0) {
		if (env->race_states == NULL)
			env_fail("respawn needs the race layer");
		const GmIso4 *spawn =
			TmnfRace_RespawnLocation(&env->race_states[index]);
		if (spawn == NULL)
			env_fail("respawn before any checkpoint restarts the race");
		TmnfVehicle_Respawn(corpus, spawn);
	}
	CTrackManiaControlPlayerInput_UpdateVehicleStateFromInputs(
		input, corpus->vehicle);
	TmnfPhysicsCorpus_AdvanceTimer(corpus, tick_ms);
	CHmsZoneDynamic_PhysicsStep2(env->worlds[index], tick_ms);
}

typedef struct {
	const TMNFRaceInputs *inputs;
	uint32_t tick_ms;
	TmnfObservation *observations;
} RawStepContext;

static void step_raw_range(
	TmnfVecEnv *env, uint32_t begin, uint32_t end, void *context_pointer)
{
	RawStepContext *context = context_pointer;
	for (uint32_t i = begin; i < end; ++i) {
		physics_tick(env, i, &context->inputs[i], context->tick_ms);
		write_physics_observation(
			player_corpus(env, i), &context->observations[i]);
	}
}

void TmnfVecEnv_Step(
	TmnfVecEnv *env, const TMNFRaceInputs *inputs,
	uint32_t tick_ms, TmnfObservation *observations)
{
	if (env == NULL || inputs == NULL || observations == NULL ||
		tick_ms == 0) {
		env_fail("physics step argument is invalid");
	}
	begin_operation(env, TMNF_VEC_OPERATION_STEP);
	RawStepContext context = {
		.inputs = inputs,
		.tick_ms = tick_ms,
		.observations = observations,
	};
	dispatch_job(env, step_raw_range, &context);
	end_operation(env);
}

static TMNFRaceInputs discrete_input(
	const TmnfVecEnv *env, uint8_t action, uint32_t timestamp)
{
	uint32_t respawn = (action & TMNF_DISCRETE_RESPAWN_FLAG) != 0;
	if (respawn && env->config.respawn_action == 0)
		env_fail("discrete respawn used without respawn_action");
	action &= (uint8_t)~TMNF_DISCRETE_RESPAWN_FLAG;
	if (action >= TMNF_DISCRETE_ACTION_COUNT)
		env_fail("discrete action is out of range");
	uint32_t longitudinal = action / 3;
	uint32_t steering = action % 3;
	TMNFRaceInputs input = {
		.steer_left_time = timestamp,
		.steer_right_time = timestamp,
		.accelerate_time = timestamp,
		.brake_time = timestamp,
		.respawn = respawn,
	};
	input.steer_left = steering == 0;
	input.steer_right = steering == 2;
	input.accelerate = longitudinal == 1 || longitudinal == 3;
	input.brake = longitudinal == 2 || longitudinal == 3;
	return input;
}

int32_t TmnfVecEnv_QuantizeAnalogSteer(float steer)
{
	if (!isfinite(steer) || steer < -1.0f || steer > 1.0f)
		env_fail("analog steer is outside [-1, 1]");
	return (int32_t)roundf(steer * 65536.0f);
}

static TMNFRaceInputs analog_input(
	const TmnfVecEnv *env, const TmnfAnalogAction *action,
	uint32_t timestamp)
{
	if (action->respawn != 0 && env->config.respawn_action == 0)
		env_fail("analog respawn used without respawn_action");
	int32_t steer = TmnfVecEnv_QuantizeAnalogSteer(action->steer);
	/* TMInterface writes (float)(-v) / 65536: a zero steer is +0.0 in the
	 * packet and the game's negation makes input_steer -0.0, as every analog
	 * record capture shows. Negating the float instead would flip that sign. */
	return (TMNFRaceInputs){
		.steer_analog_time = timestamp,
		.steer_analog = (float)(-steer) / 65536.0f,
		.accelerate_time = timestamp,
		.accelerate = action->gas,
		.brake_time = timestamp,
		.brake = action->brake,
		.respawn = action->respawn,
	};
}

static float race_potential(const TmnfVecEnv *env, uint32_t index)
{
	float total_distance = TmnfRoute_GetReferenceLength(env->route) *
		env->route->metadata->lap_count;
	return -(total_distance -
		env->race_states[index].unwrapped_progress) /
		env->config.reference_speed;
}

static float failure_base_reward(
	const TmnfVecEnv *env, uint32_t elapsed_before_failure)
{
	uint32_t remaining = env->config.max_race_ticks -
		elapsed_before_failure;
	float gamma = env->config.discount_per_tick;
	if (gamma == 1.0f)
		return -0.01f * (float)remaining;
	return -0.01f *
		(1.0f - powf(gamma, (float)remaining)) /
		(1.0f - gamma);
}

static TmnfTerminationReason failure_reason(
	TmnfVecEnv *env, uint32_t index)
{
	TmnfRaceState *race = &env->race_states[index];
	const TmnfPhysicsCorpus *corpus = player_corpus(env, index);
	const CHmsStateDyna *state = corpus->dyna->liveState;
	uint32_t wheels_in_contact = 0;
	uint32_t wheels_on_ground_plane = 0;
	for (uint32_t i = 0; i < TMNF_STADIUM_WHEEL_COUNT; ++i) {
		const CSceneVehicleCarWheelRealTimeState *wheel =
			&corpus->vehicle->wheels[i].real_time;
		if (wheel->has_ground_contact == 0)
			continue;
		wheels_in_contact++;
		if (TmnfRace_IsGroundPlaneMaterial(
				(uint16_t)wheel->contact_material_id))
			wheels_on_ground_plane++;
	}
	int off_track = TmnfRace_UpdateOffTrack(
		race, env->config.off_track_grace_ticks,
		wheels_in_contact, wheels_on_ground_plane);

	/* Progress only: a car that moves without advancing along the route
	 * (spinning in place, floating in water, rolling along a wall, or
	 * outside the corridor where TmnfRace_Step freezes progress) is as
	 * stuck as one standing still. No committed game capture holds
	 * |delta| <= 1 mm for more than a handful of ticks inside the
	 * corridor, airborne phases included. */
	float delta = fabsf(
		race->unwrapped_progress - race->previous_progress);
	if (delta <= env->config.stuck_progress_epsilon)
		race->stuck_ticks++;
	else
		race->stuck_ticks = 0;

	if (race->elapsed_ticks >= env->config.max_race_ticks)
		return TMNF_TERMINATION_TIMEOUT;
	if (state->pos.y < env->fell_floor_y)
		return TMNF_TERMINATION_FELL;
	if (off_track)
		return TMNF_TERMINATION_OFF_TRACK;
	if (race->stuck_ticks >= env->config.stuck_grace_ticks)
		return TMNF_TERMINATION_STUCK;
	return TMNF_TERMINATION_NONE;
}

/*
 * Reward of the tick that just ran: base + gamma * phi(s') - phi(s), with
 * phi(s) = -(remaining distance) / reference_speed and base = -0.01.
 *
 * FINISH: phi(s') = 0 exactly (TmnfRace_Step pins progress to laps * length;
 * asserted below to within one centerline sample). Over an episode of T
 * ticks the shaping telescopes to -phi(s_0) and the base cost sums to
 * -0.01 (1 - gamma^T) / (1 - gamma):
 *
 *   G_finish  = -phi(s_0) - 0.01 (1 - gamma^T) / (1 - gamma)
 *
 * FAILURE (timeout, off-track, stuck, fell): the terminal potential is kept
 * and deferred to the end of the budget, phi(s') = gamma^(Tmax - T) phi(s_T),
 * and base becomes the discounted cost of every tick left in the race
 * budget, -0.01 (1 - gamma^(Tmax - T + 1)) / (1 - gamma). The T-1 ordinary
 * ticks plus that lump sum to -0.01 (1 - gamma^Tmax) / (1 - gamma) whatever
 * T is, and gamma^T gamma^(Tmax - T) = gamma^Tmax, so:
 *
 *   G_fail    = gamma^Tmax phi(s_T) - phi(s_0) - 0.01 (1 - gamma^Tmax) / (1 - gamma)
 *
 * A failure is worth exactly what running out the budget at s_T would be:
 * the failure tick T does not enter, so failing early buys nothing and
 * surviving buys nothing. G_finish - G_fail =
 * -gamma^Tmax phi(s_T') + 0.01 (gamma^T - gamma^Tmax) / (1 - gamma) >= 0
 * because phi <= 0 and T <= Tmax: a finish beats every failure, with
 * equality only for a failure on the finish line at the last budget tick.
 * Between two failures under one budget, d G_fail / d progress(s_T) =
 * gamma^Tmax / reference_speed > 0: failing further along the route is
 * never worse than failing earlier. Progress is unwrapped_progress, the
 * dense projection's arc length plus completed laps, so a car that reverses
 * lowers phi(s_T) and pays for the ground it gave back; a car that stops
 * paying attention (A04's floating car) is ranked by where it stopped.
 *
 * Two earlier forms failed. Zeroing the terminal potential on failure made
 * G_fail = -phi(s_0) - 0.01 (1 - gamma^Tmax) / (1 - gamma), one constant
 * for every non-finishing episode. Keeping
 * phi(s_T) undeferred, G_fail = gamma^T phi(s_T) - ..., paid (1 - gamma^dT)
 * |phi(s_T)| for failing dT ticks later at the same progress, up to
 * |phi(s_0)| (1 - gamma^Tmax) (B05 19.5): on B05 a car surviving the 7,444
 * tick budget at 30 m out-earned one driving 263 m in 8 s and failing by 12,
 * and PPO learned to survive (0 finishes in 70,423 episodes,
 * f33_b05_ppo_baseline_20m_s1); the deferred form finished 72% of its
 * episodes on the same seed (f33_b05_tmaxscaled_20m_s1).
 */
static float tick_reward(
	const TmnfVecEnv *env, uint32_t index, TmnfTerminationReason reason,
	uint32_t elapsed_before, float previous_potential)
{
	float base = -0.01f;
	float next_potential = race_potential(env, index);
	if (reason == TMNF_TERMINATION_FINISH) {
		float sample_spacing =
			TmnfRoute_GetReferenceLength(env->route) /
			(float)(TmnfRoute_GetReferencePointCount(env->route) - 1);
		if (-next_potential * env->config.reference_speed >
			sample_spacing) {
			env_fail("finish left remaining distance");
		}
		next_potential = 0.0f;
	} else if (reason != TMNF_TERMINATION_NONE) {
		base = failure_base_reward(env, elapsed_before);
		/* Defer phi(s_T) to the budget's end: gamma^(Tmax - T). */
		next_potential *= powf(env->config.discount_per_tick,
			(float)(env->config.max_race_ticks - (elapsed_before + 1)));
	}
	return base + env->config.discount_per_tick * next_potential -
		previous_potential;
}

/*
 * One RL tick over a prepared packet. `first` marks the first tick of an
 * action repeat: a respawn is a press edge, so only that tick respawns and
 * the rest of the repeat coasts with the same controls. A press before any
 * respawnable checkpoint is the game's race restart: the tick still runs
 * (the car coasts), then the episode ends with TMNF_TERMINATION_RESTART and
 * the failure reward.
 */
static int step_one_tick_input(
	TmnfVecEnv *env, uint32_t index, TMNFRaceInputs input, int first,
	float reward_weight, TmnfStepResult *result)
{
	TmnfPhysicsCorpus *corpus = player_corpus(env, index);
	TmnfRaceState *race = &env->race_states[index];
	float previous_potential = race_potential(env, index);
	uint32_t elapsed_before = race->elapsed_ticks;
	int restart = 0;
	if (input.respawn != 0) {
		if (!first || TmnfRace_RespawnLocation(race) == NULL) {
			restart = first;
			input.respawn = 0;
		}
	}
	physics_tick(env, index, &input, TMNF_RACE_TICK_MS);

	TmnfRaceStepResult race_result = TmnfRace_Step(
		env->route, race, env->worlds[index]->trigger_contacts,
		corpus->collision_corpus->live_iso);
	TmnfTerminationReason reason = race_result.finished
		? TMNF_TERMINATION_FINISH
		: failure_reason(env, index);
	if (reason == TMNF_TERMINATION_NONE && restart)
		reason = TMNF_TERMINATION_RESTART;
	result->terminated = reason != TMNF_TERMINATION_NONE;
	result->truncated =
		race->elapsed_ticks >= env->config.horizon_ticks;
	result->termination_reason = reason;
	result->race_time_ms = race_result.race_time_ms;
	result->reward += reward_weight * tick_reward(
		env, index, reason, elapsed_before, previous_potential);
	return result->terminated || result->truncated;
}

static int step_one_tick(
	TmnfVecEnv *env, uint32_t index, uint8_t action, int first,
	float reward_weight, TmnfStepResult *result)
{
	TMNFRaceInputs input = discrete_input(env, action,
		(env->race_states[index].elapsed_ticks + 1) * TMNF_RACE_TICK_MS);
	return step_one_tick_input(
		env, index, input, first, reward_weight, result);
}

static int step_one_tick_analog(
	TmnfVecEnv *env, uint32_t index, const TmnfAnalogAction *action,
	int first, float reward_weight, TmnfStepResult *result)
{
	TMNFRaceInputs input = analog_input(env, action,
		(env->race_states[index].elapsed_ticks + 1) * TMNF_RACE_TICK_MS);
	return step_one_tick_input(
		env, index, input, first, reward_weight, result);
}

static void write_gate_observation(const TmnfVecEnv *env, uint32_t index,
 TmnfGateObservations *out)
{
 const TmnfPhysicsCorpus *corpus = player_corpus(env, index);
 TmnfRace_ObserveGates(env->route, &env->race_states[index],
  corpus->collision_corpus->live_iso, out);
}

void TmnfVecEnv_ObserveGates(TmnfVecEnv *env, TmnfGateObservations *gates)
{
 if (!env || !gates) env_fail("gate observation argument is null");
 begin_operation(env, TMNF_VEC_OPERATION_CAPTURE);
 if (!env->route || !env->race_states) env_fail("environment is not initialized");
 for (uint32_t i = 0; i < env->count; ++i)
  write_gate_observation(env, i, &gates[i]);
 end_operation(env);
}

typedef struct {
	const uint8_t *actions;
	uint32_t action_repeat;
	TmnfStepResult *results;
	TmnfGateObservations *gates, *final_gates;
} DiscreteStepContext;

static void step_discrete_range(
	TmnfVecEnv *env, uint32_t begin, uint32_t end, void *context_pointer)
{
	DiscreteStepContext *context = context_pointer;
	for (uint32_t i = begin; i < end; ++i) {
		TmnfStepResult *result = &context->results[i];
		memset(result, 0, sizeof(*result));
		if (context->final_gates) memset(&context->final_gates[i], 0, sizeof(context->final_gates[i]));
		result->episode_id = env->episode_ids[i];
		if (env->reset_pending[i]) {
			env->reset_pending[i] = 0;
			result->reset_only = 1;
			result->transition_discount = 1.0f;
			write_rl_observation(env, i, &result->observation);
			if (context->gates) write_gate_observation(env, i, &context->gates[i]);
			continue;
		}

		int ended = 0;
		float reward_weight = 1.0f;
		for (uint32_t repeat = 0;
			repeat < context->action_repeat; ++repeat) {
			result->executed_ticks++;
			if (step_one_tick(
				env, i, context->actions[i], repeat == 0,
				reward_weight, result)) {
				reward_weight *= env->config.discount_per_tick;
				ended = 1;
				break;
			}
			reward_weight *= env->config.discount_per_tick;
		}
		env->episode_returns[i] += result->reward;
		result->transition_discount = reward_weight;
		if (!ended) {
			write_rl_observation(env, i, &result->observation);
			if (context->gates) write_gate_observation(env, i, &context->gates[i]);
			continue;
		}

		write_rl_observation(
			env, i, &result->final_observation);
		if (context->final_gates) write_gate_observation(env, i, &context->final_gates[i]);
		result->final_observation_valid = 1;
		result->completed_episode_return =
			env->episode_returns[i];
		result->completed_episode_ticks =
			env->race_states[i].elapsed_ticks;
		TmnfObservation terminal = result->final_observation;
		reset_one(env, i);
		if (env->config.autoreset_mode ==
			TMNF_AUTORESET_SAME_STEP) {
			write_rl_observation(
				env, i, &result->observation);
			if (context->gates) write_gate_observation(env, i, &context->gates[i]);
		} else {
			result->observation = terminal;
			if (context->gates) context->gates[i] = context->final_gates[i];
			env->reset_pending[i] = 1;
		}
	}
}

static void step_discrete_with_gates(
	TmnfVecEnv *env,
	const uint8_t *actions,
	uint32_t action_repeat,
	TmnfStepResult *results,
	TmnfGateObservations *gates, TmnfGateObservations *final_gates)
{
	if (env == NULL || actions == NULL || results == NULL ||
		action_repeat == 0) {
		env_fail("RL step argument is invalid");
	}
	if (env->config.action_space != TMNF_ACTION_SPACE_DISCRETE)
		env_fail("discrete step used with analog action space");
	begin_operation(env, TMNF_VEC_OPERATION_STEP);
	if (env->route == NULL || env->race_states == NULL)
		env_fail("environment is not initialized");
	DiscreteStepContext context = {
		.actions = actions,
		.action_repeat = action_repeat,
		.results = results,
		.gates = gates, .final_gates = final_gates,
	};
	dispatch_job(env, step_discrete_range, &context);
	end_operation(env);
}


void TmnfVecEnv_StepDiscrete(TmnfVecEnv *env, const uint8_t *actions,
 uint32_t repeat, TmnfStepResult *results)
{
 step_discrete_with_gates(env, actions, repeat, results, NULL, NULL);
}
void TmnfVecEnv_StepDiscreteWithGates(TmnfVecEnv *env, const uint8_t *actions,
 uint32_t repeat, TmnfStepResult *results,
 TmnfGateObservations *gates, TmnfGateObservations *final_gates)
{
 if (!gates || !final_gates || gates == final_gates)
  env_fail("gate sidecars must be nonnull distinct arrays");
 step_discrete_with_gates(env, actions, repeat, results, gates, final_gates);
}
typedef struct {
	const TmnfAnalogAction *actions;
	uint32_t action_repeat;
	TmnfStepResult *results;
	TmnfGateObservations *gates, *final_gates;
} AnalogStepContext;

static void step_analog_range(
	TmnfVecEnv *env, uint32_t begin, uint32_t end, void *context_pointer)
{
	AnalogStepContext *context = context_pointer;
	for (uint32_t i = begin; i < end; ++i) {
		TmnfStepResult *result = &context->results[i];
		memset(result, 0, sizeof(*result));
		if (context->final_gates) memset(&context->final_gates[i], 0, sizeof(context->final_gates[i]));
		result->episode_id = env->episode_ids[i];
		if (env->reset_pending[i]) {
			env->reset_pending[i] = 0;
			result->reset_only = 1;
			result->transition_discount = 1.0f;
			write_rl_observation(env, i, &result->observation);
			if (context->gates) write_gate_observation(env, i, &context->gates[i]);
			continue;
		}

		int ended = 0;
		float reward_weight = 1.0f;
		for (uint32_t repeat = 0;
			repeat < context->action_repeat; ++repeat) {
			result->executed_ticks++;
			if (step_one_tick_analog(
				env, i, &context->actions[i], repeat == 0,
				reward_weight, result)) {
				reward_weight *= env->config.discount_per_tick;
				ended = 1;
				break;
			}
			reward_weight *= env->config.discount_per_tick;
		}
		env->episode_returns[i] += result->reward;
		result->transition_discount = reward_weight;
		if (!ended) {
			write_rl_observation(env, i, &result->observation);
			if (context->gates) write_gate_observation(env, i, &context->gates[i]);
			continue;
		}

		write_rl_observation(
			env, i, &result->final_observation);
		if (context->final_gates) write_gate_observation(env, i, &context->final_gates[i]);
		result->final_observation_valid = 1;
		result->completed_episode_return =
			env->episode_returns[i];
		result->completed_episode_ticks =
			env->race_states[i].elapsed_ticks;
		TmnfObservation terminal = result->final_observation;
		reset_one(env, i);
		if (env->config.autoreset_mode ==
			TMNF_AUTORESET_SAME_STEP) {
			write_rl_observation(
				env, i, &result->observation);
			if (context->gates) write_gate_observation(env, i, &context->gates[i]);
		} else {
			result->observation = terminal;
			if (context->gates) context->gates[i] = context->final_gates[i];
			env->reset_pending[i] = 1;
		}
	}
}

static void step_analog_with_gates(
	TmnfVecEnv *env,
	const TmnfAnalogAction *actions,
	uint32_t action_repeat,
	TmnfStepResult *results,
	TmnfGateObservations *gates, TmnfGateObservations *final_gates)
{
	if (env == NULL || actions == NULL || results == NULL ||
		action_repeat == 0) {
		env_fail("analog step argument is invalid");
	}
	if (env->config.action_space != TMNF_ACTION_SPACE_ANALOG)
		env_fail("analog step used with discrete action space");
	for (uint32_t i = 0; i < env->count; ++i) {
		(void)TmnfVecEnv_QuantizeAnalogSteer(actions[i].steer);
		if (actions[i].gas > 1 || actions[i].brake > 1)
			env_fail("analog gas and brake must be binary");
	}
	begin_operation(env, TMNF_VEC_OPERATION_STEP);
	if (env->route == NULL || env->race_states == NULL)
		env_fail("environment is not initialized");
	AnalogStepContext context = {
		.actions = actions,
		.action_repeat = action_repeat,
		.results = results,
		.gates = gates, .final_gates = final_gates,
	};
	dispatch_job(env, step_analog_range, &context);
	end_operation(env);
}

void TmnfVecEnv_StepAnalog(TmnfVecEnv *env, const TmnfAnalogAction *actions,
 uint32_t repeat, TmnfStepResult *results)
{
 step_analog_with_gates(env, actions, repeat, results, NULL, NULL);
}
void TmnfVecEnv_StepAnalogWithGates(TmnfVecEnv *env, const TmnfAnalogAction *actions,
 uint32_t repeat, TmnfStepResult *results,
 TmnfGateObservations *gates, TmnfGateObservations *final_gates)
{
 if (!gates || !final_gates || gates == final_gates)
  env_fail("gate sidecars must be nonnull distinct arrays");
 step_analog_with_gates(env, actions, repeat, results, gates, final_gates);
}

static void empty_range(
	TmnfVecEnv *env, uint32_t begin, uint32_t end, void *context)
{
	(void)env;
	(void)begin;
	(void)end;
	(void)context;
}

void TmnfVecEnv_DispatchEmpty(TmnfVecEnv *env)
{
	if (env == NULL)
		env_fail("empty dispatch environment is null");
	begin_operation(env, TMNF_VEC_OPERATION_DISPATCH_EMPTY);
	dispatch_job(env, empty_range, NULL);
	end_operation(env);
}

static void flatten_observation(
	const TmnfObservation *observation, float *flat)
{
	memcpy(flat, observation, 34 * sizeof(*flat));
	flat[34] = (float)observation->gear;
	memcpy(
		flat + 35, &observation->input_steer,
		11 * sizeof(*flat));
	memcpy(
		flat + 46, observation->centerline_lookahead,
		TMNF_OBSERVATION_LOOKAHEAD_COUNT *
			sizeof(TmnfCenterlineSample));
	memcpy(
		flat + 78, &observation->turbo_active,
		3 * sizeof(*flat));
}

void TmnfVecEnv_FlattenStepResults(
	const TmnfStepResult *results,
	uint32_t count,
	float *observations,
	float *final_observations,
	float *transitions)
{
	if (results == NULL || count == 0 || observations == NULL ||
		final_observations == NULL || transitions == NULL) {
		env_fail("flatten results argument is null or empty");
	}
	for (uint32_t i = 0; i < count; ++i) {
		flatten_observation(
			&results[i].observation,
			observations +
				(size_t)i * TMNF_POLICY_OBSERVATION_WIDTH);
		flatten_observation(
			&results[i].final_observation,
			final_observations +
				(size_t)i * TMNF_POLICY_OBSERVATION_WIDTH);
		float *transition = transitions +
			(size_t)i * TMNF_POLICY_TRANSITION_WIDTH;
		transition[0] = results[i].reward;
		transition[1] = results[i].transition_discount;
		transition[2] = (float)results[i].terminated;
		transition[3] = (float)results[i].truncated;
		transition[4] = (float)results[i].executed_ticks;
	}
}
