/* Replay harness: loads golden trace files, feeds recorded inputs through the
 * ported C, and compares outputs byte-for-byte against the recorded game
 * outputs. Reports the first divergent byte per function.
 *
 * Add a handler for each ported function in the registry below, keyed by VA. */
#include <dirent.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gm.h"
#include "hms_state.h"
#include "hms_dyna.h"
#include "trace_format.h"

typedef struct {
	uint32_t tag;
	uint32_t addr;
	uint32_t len;
	uint8_t *bytes;
} Buffer;

typedef struct {
	uint32_t seq;
	int n_in, n_out;
	Buffer *in;
	Buffer *out;
} Record;

typedef struct {
	uint32_t va;
	Record *records;
	int count;
} Trace;

static void free_trace(Trace *trace)
{
	for (int i = 0; i < trace->count; ++i) {
		Record *record = &trace->records[i];
		for (int j = 0; j < record->n_in; ++j)
			free(record->in[j].bytes);
		for (int j = 0; j < record->n_out; ++j)
			free(record->out[j].bytes);
		free(record->in);
		free(record->out);
	}
	free(trace->records);
}


static Buffer *find(Buffer *arr, int n, uint32_t tag) {
	for (int i = 0; i < n; i++) {
		if (arr[i].tag == tag) {
			return &arr[i];
		}
	}
	return NULL;
}

/* A handler runs the C port on one record's inputs and writes its outputs into
 * caller-provided scratch, returning them for comparison. It returns 0 on a
 * shape it does not recognise. */
typedef int (*Handler)(const Record *rec, uint8_t *scratch, uint32_t *out_len,
	uint32_t *out_tag);

/* ---- Handlers ------------------------------------------------------------ */

/* GmQuat::Normalize (0x008E3120): this is a 16-byte quat, mutated in place. */
static int h_quat_normalize(const Record *rec, uint8_t *scratch,
	uint32_t *out_len, uint32_t *out_tag) {
	const Buffer *in = find(rec->in, rec->n_in, TAG_THIS);
	if (!in || in->len < (uint32_t)sizeof(GmQuat)) {
		return 0;
	}
	GmQuat q;
	memcpy(&q, in->bytes, sizeof(q));
	GmQuat_Normalize(&q);
	memcpy(scratch, &q, sizeof(q));
	*out_len = sizeof(q);
	*out_tag = TAG_OUT_THIS;
	return 1;
}

/* GmVec3::SetMult(Iso4) (0x0045BBA0): out=this, in ARG1=v(12), ARG2=A(48). */
static int h_setmult_iso4(const Record *rec, uint8_t *scratch,
	uint32_t *out_len, uint32_t *out_tag) {
	const Buffer *v = find(rec->in, rec->n_in, TAG_ARG1);
	const Buffer *a = find(rec->in, rec->n_in, TAG_ARG2);
	if (!v || !a || v->len < 12 || a->len < 48) {
		return 0;
	}
	GmVec3 out;
	GmVec3_SetMult_Iso4(&out, (const GmVec3 *)v->bytes, (const GmIso4 *)a->bytes);
	memcpy(scratch, &out, sizeof(out));
	*out_len = sizeof(out);
	*out_tag = TAG_OUT_THIS;
	return 1;
}

/* GmVec3::SetMult(Mat3) (0x004574A0): out=this, in ARG1=v(12), ARG2=M(36). */
static int h_setmult_mat3(const Record *rec, uint8_t *scratch,
	uint32_t *out_len, uint32_t *out_tag) {
	const Buffer *v = find(rec->in, rec->n_in, TAG_ARG1);
	const Buffer *m = find(rec->in, rec->n_in, TAG_ARG2);
	if (!v || !m || v->len < 12 || m->len < 36) {
		return 0;
	}
	GmVec3 out;
	GmVec3_SetMult_Mat3(&out, (const GmVec3 *)v->bytes, (const GmMat3 *)m->bytes);
	memcpy(scratch, &out, sizeof(out));
	*out_len = sizeof(out);
	*out_tag = TAG_OUT_THIS;
	return 1;
}

/* GmVec3::MultTranspose (0x0045BD40): this(12) in place, ARG1=M(36). */
static int h_mult_transpose(const Record *rec, uint8_t *scratch,
	uint32_t *out_len, uint32_t *out_tag) {
	const Buffer *t = find(rec->in, rec->n_in, TAG_THIS);
	const Buffer *m = find(rec->in, rec->n_in, TAG_ARG1);
	if (!t || !m || t->len < 12 || m->len < 36) {
		return 0;
	}
	GmVec3 v;
	memcpy(&v, t->bytes, sizeof(v));
	GmVec3_MultTranspose(&v, (const GmMat3 *)m->bytes);
	memcpy(scratch, &v, sizeof(v));
	*out_len = sizeof(v);
	*out_tag = TAG_OUT_THIS;
	return 1;
}

/* GmMat3::Mult (0x008E09F0): this(36) in place, ARG1=B(36). */
static int h_mat3_mult(const Record *rec, uint8_t *scratch,
	uint32_t *out_len, uint32_t *out_tag) {
	const Buffer *t = find(rec->in, rec->n_in, TAG_THIS);
	const Buffer *b = find(rec->in, rec->n_in, TAG_ARG1);
	if (!t || !b || t->len < 36 || b->len < 36) {
		return 0;
	}
	GmMat3 m;
	memcpy(&m, t->bytes, sizeof(m));
	GmMat3_Mult(&m, (const GmMat3 *)b->bytes);
	memcpy(scratch, &m, sizeof(m));
	*out_len = sizeof(m);
	*out_tag = TAG_OUT_THIS;
	return 1;
}

/* GmMat3::SetTranspose (0x008E0C60): this=out, ARG1=source matrix. */
static int h_mat3_set_transpose(const Record *rec, uint8_t *scratch,
	uint32_t *out_len, uint32_t *out_tag) {
	const Buffer *src = find(rec->in, rec->n_in, TAG_ARG1);
	if (!src || src->len < (uint32_t)sizeof(GmMat3)) {
		return 0;
	}
	GmMat3 out;
	GmMat3_SetTranspose(&out, (const GmMat3 *)src->bytes);
	memcpy(scratch, &out, sizeof(out));
	*out_len = sizeof(out);
	*out_tag = TAG_OUT_THIS;
	return 1;
}

/* CHmsDyna::CopyStateToTemp (0x00532D40): tracer records the live state as
 * ARG1 and the embedded temp-state window as THIS. */
static int h_copy_state_to_temp(const Record *rec, uint8_t *scratch,
	uint32_t *out_len, uint32_t *out_tag) {
	const Buffer *live = find(rec->in, rec->n_in, TAG_ARG1);
	const Buffer *temp = find(rec->in, rec->n_in, TAG_THIS);
	if (!live || !temp || live->len < (uint32_t)sizeof(CHmsStateDyna) ||
		temp->len < (uint32_t)sizeof(CHmsStateDyna)) {
		return 0;
	}
	CHmsStateDyna live_state;
	memcpy(&live_state, live->bytes, sizeof(live_state));
	CHmsDyna dyn;
	memset(&dyn, 0, sizeof(dyn));
	memcpy(&dyn.tempState, temp->bytes, sizeof(dyn.tempState));
	dyn.liveState = &live_state;
	CHmsDyna_CopyStateToTemp(&dyn);
	memcpy(scratch, &dyn.tempState, sizeof(dyn.tempState));
	*out_len = sizeof(dyn.tempState);
	*out_tag = TAG_OUT_THIS;
	return 1;
}

/* CHmsDyna::IntegrateStep (0x00533510): ARG1=stateIn(180), SCALAR1=dt(4),
 * ARG3=params(0x44 window), THIS=CHmsDyna window (>=0x344); out OUT_ARG2=stateOut(180). */
static int h_integrate_step(const Record *rec, uint8_t *scratch,
	uint32_t *out_len, uint32_t *out_tag) {
	const Buffer *sin = find(rec->in, rec->n_in, TAG_ARG1);
	const Buffer *dt = find(rec->in, rec->n_in, TAG_SCALAR1);
	const Buffer *par = find(rec->in, rec->n_in, TAG_ARG3);
	const Buffer *th = find(rec->in, rec->n_in, TAG_THIS);
	if (!sin || !dt || !par || !th || sin->len < 180 || par->len < 0x44 ||
		th->len < 0x344) {
		return 0;
	}
	CHmsDyna dyn;
	memset(&dyn, 0, sizeof(dyn));
	CHmsDynaParams params;
	memcpy(&params, par->bytes, sizeof(params));
	dyn.params = &params;
	memcpy(&dyn.clampAngular, th->bytes + 0xC0, 4);
	memcpy(&dyn.maxAngularSpeed, th->bytes + 0xC4, 4);
	memcpy(&dyn.mode, th->bytes + 0x340, 4);
	CHmsStateDyna in, out;
	memcpy(&in, sin->bytes, 180);
	/* IntegrateStep leaves 0xA0..0xB3 untouched; in DoPreCollisionDynamic the
	 * out buffer is the live state whose pre-call tail equals in's tail. */
	memcpy(&out, sin->bytes, 180);
	float dtv;
	memcpy(&dtv, dt->bytes, 4);
	CHmsDyna_IntegrateStep(&dyn, &in, &out, dtv);
	memcpy(scratch, &out, 180);
	*out_len = 180;
	*out_tag = TAG_OUT_ARG2;
	return 1;
}

static const struct {
	uint32_t va;
	Handler fn;
	const char *name;
} REGISTRY[] = {
	{ 0x008E3120, h_quat_normalize, "GmQuat::Normalize" },
	{ 0x0045BBA0, h_setmult_iso4,   "GmVec3::SetMult(Iso4)" },
	{ 0x004574A0, h_setmult_mat3,   "GmVec3::SetMult(Mat3)" },
	{ 0x0045BD40, h_mult_transpose, "GmVec3::MultTranspose" },
	{ 0x008E09F0, h_mat3_mult,      "GmMat3::Mult" },
	{ 0x008E0C60, h_mat3_set_transpose, "GmMat3::SetTranspose" },
	{ 0x00532D40, h_copy_state_to_temp, "CHmsDyna::CopyStateToTemp" },
	{ 0x00533510, h_integrate_step, "CHmsDyna::IntegrateStep" },
};
#define REGISTRY_N ((int)(sizeof(REGISTRY) / sizeof(REGISTRY[0])))

/* ---- Loader -------------------------------------------------------------- */

static uint32_t rd_u32(FILE *f) {
	uint8_t b[4];
	if (fread(b, 1, 4, f) != 4) {
		fprintf(stderr, "unexpected EOF reading u32\n");
		exit(2);
	}
	return (uint32_t)b[0] | (uint32_t)b[1] << 8 | (uint32_t)b[2] << 16 |
		(uint32_t)b[3] << 24;
}

static uint16_t rd_u16(FILE *f) {
	uint8_t b[2];
	if (fread(b, 1, 2, f) != 2) {
		fprintf(stderr, "unexpected EOF reading u16\n");
		exit(2);
	}
	return (uint16_t)(b[0] | b[1] << 8);
}

static void rd_buffers(FILE *f, int n, Buffer *arr) {
	for (int i = 0; i < n; i++) {
		arr[i].tag = rd_u32(f);
		arr[i].addr = rd_u32(f);
		arr[i].len = rd_u32(f);
		arr[i].bytes = malloc(arr[i].len);
		if (fread(arr[i].bytes, 1, arr[i].len, f) != arr[i].len) {
			fprintf(stderr, "unexpected EOF reading buffer bytes\n");
			exit(2);
		}
	}
}

static Trace load(const char *path) {
	FILE *f = fopen(path, "rb");
	if (!f) {
		fprintf(stderr, "cannot open %s\n", path);
		exit(2);
	}
	char magic[8];
	if (fread(magic, 1, 8, f) != 8 ||
		memcmp(magic, TMNF_TRACE_MAGIC, 8) != 0) {
		fprintf(stderr, "bad magic in %s\n", path);
		exit(2);
	}
	Trace t;
	t.va = rd_u32(f);
	t.count = (int)rd_u32(f);
	t.records = calloc(t.count, sizeof(Record));
	for (int i = 0; i < t.count; i++) {
		Record *r = &t.records[i];
		r->seq = rd_u32(f);
		r->n_in = rd_u16(f);
		r->n_out = rd_u16(f);
		r->in = calloc(r->n_in, sizeof(Buffer));
		r->out = calloc(r->n_out, sizeof(Buffer));
		rd_buffers(f, r->n_in, r->in);
		rd_buffers(f, r->n_out, r->out);
	}
	fclose(f);
	return t;
}

static Handler lookup(uint32_t va, const char **name) {
	for (int i = 0; i < REGISTRY_N; i++) {
		if (REGISTRY[i].va == va) {
			*name = REGISTRY[i].name;
			return REGISTRY[i].fn;
		}
	}
	return NULL;
}

int main(int argc, char **argv) {
	if (argc != 2) {
		fprintf(stderr, "usage: %s TRACE_DIR\n", argv[0]);
		return 2;
	}
	DIR *d = opendir(argv[1]);
	if (!d) {
		fprintf(stderr, "no trace dir %s (nothing to replay)\n", argv[1]);
		return 0;
	}
	int total = 0, matched = 0, failed = 0, skipped = 0;
	struct dirent *e;
	uint8_t scratch[4096];
	while ((e = readdir(d)) != NULL) {
		size_t n = strlen(e->d_name);
		if (n < 4 || strcmp(e->d_name + n - 4, ".bin") != 0) {
			continue;
		}
		char path[4096];
		snprintf(path, sizeof(path), "%s/%s", argv[1], e->d_name);
		Trace t = load(path);
		const char *name = NULL;
		Handler h = lookup(t.va, &name);
		if (!h) {
			printf("SKIP  0x%08X %s (no handler)\n", t.va, e->d_name);
			skipped++;
			free_trace(&t);
			continue;
		}
		int file_fail = 0;
		for (int i = 0; i < t.count; i++) {
			uint32_t out_len = 0, out_tag = 0;
			if (!h(&t.records[i], scratch, &out_len, &out_tag)) {
				continue;
			}
			const Buffer *exp = find(t.records[i].out, t.records[i].n_out, out_tag);
			if (!exp) {
				continue;
			}
			total++;
			if (exp->len == out_len && memcmp(exp->bytes, scratch, out_len) == 0) {
				matched++;
			} else {
				failed++;
				file_fail++;
				if (file_fail <= 3) {
					int off = -1;
					for (uint32_t b = 0; b < out_len && b < exp->len; b++) {
						if (scratch[b] != exp->bytes[b]) { off = (int)b; break; }
					}
					printf("FAIL  %s seq=%u first diff at byte %d\n",
						name, t.records[i].seq, off);
				}
			}
		}
		if (!file_fail) {
			printf("OK    0x%08X %s\n", t.va, name);
		}
		free_trace(&t);
	}
	closedir(d);
	printf("\n%d records: %d matched, %d failed, %d files skipped\n",
		total, matched, failed, skipped);
	return failed ? 1 : 0;
}
