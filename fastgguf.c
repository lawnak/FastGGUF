// fastgguf.c — GGUF inference engine in C with AVX2 kernels
// Supported architectures: Gemma (gemma*) / Qwen (qwen2, qwen3, qwen3moe) / Mistral and Llama (llama)
// Gemma path ported from the verified Go version; architecture matches llama.cpp (src/models/gemma4.cpp)
//
// Build (Windows with w64devkit or MSYS2):
//   gcc -O3 -march=native -fopenmp -o fastgguf.exe fastgguf.c -lm -lws2_32
// Build (Linux):
//   gcc -O3 -march=native -fopenmp -o fastgguf fastgguf.c -lm -ldl
//
// Usage:
//   fastgguf chat    model.gguf [ctx] [--temp 1.0] [--top-k 64] [--top-p 0.95]
//                    [--repeat-penalty 1.0] [--repeat-last-n 64] [--max-tokens 512] [--seed N]
//   fastgguf serve   model.gguf [model2.gguf ...] [port] [ctx]
//   fastgguf inspect model.gguf
//   fastgguf dump    model.gguf
//   fastgguf gpu
//   fastgguf selftest

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#else
static int omp_get_thread_num(void) { return 0; }
static int omp_get_max_threads(void) { return 1; }
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(__GNUC__)
#define MAYBE_UNUSED __attribute__((unused))
#else
#define MAYBE_UNUSED
#endif

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <shellapi.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

// =========================================================================
// fp16 / bf16 — full conversion table (65536 entries) for speed
// =========================================================================

static float g_fp16_tbl[65536];

static float fp16_to_float_sw(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x03ff;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) { f = sign; }
        else {
            exp = 1;
            while (!(mant & 0x0400)) { mant <<= 1; exp--; }
            mant &= 0x03ff;
            exp += 127 - 15;
            f = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7f800000u | (mant << 13);
    } else {
        exp += 127 - 15;
        f = sign | (exp << 23) | (mant << 13);
    }
    float out; memcpy(&out, &f, 4); return out;
}

static void fp16_init(void) {
    for (int i = 0; i < 65536; i++) g_fp16_tbl[i] = fp16_to_float_sw((uint16_t)i);
}
#define FP16(h) g_fp16_tbl[(uint16_t)(h)]

static inline float bf16_to_float(uint16_t b) {
    uint32_t f = (uint32_t)b << 16;
    float out; memcpy(&out, &f, 4); return out;
}

static inline uint16_t rd_u16(const uint8_t *p) { uint16_t v; memcpy(&v, p, 2); return v; }
static inline uint32_t rd_u32(const uint8_t *p) { uint32_t v; memcpy(&v, p, 4); return v; }
static inline uint64_t rd_u64(const uint8_t *p) { uint64_t v; memcpy(&v, p, 8); return v; }
static inline float    rd_f32(const uint8_t *p) { float v;    memcpy(&v, p, 4); return v; }

// =========================================================================
// mmap (Windows + POSIX)
// =========================================================================

typedef struct {
    uint8_t *data;
    uint64_t size;
#ifdef _WIN32
    HANDLE file, mapping;
#else
    int fd;
#endif
} mmap_t;

static int mmap_open(mmap_t *m, const char *path) {
    memset(m, 0, sizeof(*m));
#ifdef _WIN32
    // path is UTF-8 (see utf8_argv) — CreateFileW accepts non-ASCII (e.g. Arabic) names, unlike CreateFileA
    int wn = MultiByteToWideChar(CP_UTF8, 0, path, -1, NULL, 0);
    if (wn <= 0) return -1;
    wchar_t *wpath = (wchar_t *)malloc((size_t)wn * sizeof(wchar_t));
    MultiByteToWideChar(CP_UTF8, 0, path, -1, wpath, wn);
    m->file = CreateFileW(wpath, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
                          FILE_ATTRIBUTE_NORMAL, NULL);
    free(wpath);
    if (m->file == INVALID_HANDLE_VALUE) return -1;
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(m->file, &sz)) { CloseHandle(m->file); return -1; }
    m->size = (uint64_t)sz.QuadPart;
    m->mapping = CreateFileMappingA(m->file, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!m->mapping) { CloseHandle(m->file); return -1; }
    m->data = (uint8_t *)MapViewOfFile(m->mapping, FILE_MAP_READ, 0, 0, 0);
    if (!m->data) { CloseHandle(m->mapping); CloseHandle(m->file); return -1; }
#else
    m->fd = open(path, O_RDONLY);
    if (m->fd < 0) return -1;
    struct stat st;
    if (fstat(m->fd, &st) < 0) { close(m->fd); return -1; }
    m->size = (uint64_t)st.st_size;
    m->data = (uint8_t *)mmap(NULL, m->size, PROT_READ, MAP_SHARED, m->fd, 0);
    if (m->data == MAP_FAILED) { close(m->fd); return -1; }
#endif
    return 0;
}

MAYBE_UNUSED static void mmap_close(mmap_t *m) {
#ifdef _WIN32
    if (m->data) UnmapViewOfFile(m->data);
    if (m->mapping) CloseHandle(m->mapping);
    if (m->file && m->file != INVALID_HANDLE_VALUE) CloseHandle(m->file);
#else
    if (m->data && m->data != MAP_FAILED) munmap(m->data, m->size);
    if (m->fd >= 0) close(m->fd);
#endif
    memset(m, 0, sizeof(*m));
}

// =========================================================================
// GGML types
// =========================================================================

enum {
    T_F32 = 0, T_F16 = 1, T_Q4_0 = 2, T_Q4_1 = 3,
    T_Q5_0 = 6, T_Q5_1 = 7, T_Q8_0 = 8,
    T_Q2_K = 10, T_Q3_K = 11, T_Q4_K = 12, T_Q5_K = 13, T_Q6_K = 14,
    T_IQ4_NL = 20, T_IQ4_XS = 23,
    T_BF16 = 30,
};

static const char *type_name(uint32_t t) {
    switch (t) {
    case T_F32: return "F32";   case T_F16: return "F16";  case T_BF16: return "BF16";
    case T_Q4_0: return "Q4_0"; case T_Q4_1: return "Q4_1";
    case T_Q5_0: return "Q5_0"; case T_Q5_1: return "Q5_1"; case T_Q8_0: return "Q8_0";
    case T_Q2_K: return "Q2_K"; case T_Q3_K: return "Q3_K"; case T_Q4_K: return "Q4_K";
    case T_Q5_K: return "Q5_K"; case T_Q6_K: return "Q6_K";
    case T_IQ4_NL: return "IQ4_NL"; case T_IQ4_XS: return "IQ4_XS";
    default: { static char b[32]; snprintf(b, 32, "TYPE_%u", t); return b; }
    }
}

static int block_size_of(uint32_t t, uint64_t *elems, uint64_t *bytes) {
    switch (t) {
    case T_F32:  *elems = 1;   *bytes = 4;   return 0;
    case T_F16: case T_BF16: *elems = 1; *bytes = 2; return 0;
    case T_Q4_0: *elems = 32;  *bytes = 18;  return 0;
    case T_Q4_1: *elems = 32;  *bytes = 20;  return 0;
    case T_Q5_0: *elems = 32;  *bytes = 22;  return 0;
    case T_Q5_1: *elems = 32;  *bytes = 24;  return 0;
    case T_Q8_0: *elems = 32;  *bytes = 34;  return 0;
    case T_Q2_K: *elems = 256; *bytes = 84;  return 0;
    case T_Q3_K: *elems = 256; *bytes = 110; return 0;
    case T_Q4_K: *elems = 256; *bytes = 144; return 0;
    case T_Q5_K: *elems = 256; *bytes = 176; return 0;
    case T_Q6_K: *elems = 256; *bytes = 210; return 0;
    case T_IQ4_NL: *elems = 32;  *bytes = 18;  return 0;
    case T_IQ4_XS: *elems = 256; *bytes = 136; return 0;
    default: return -1;
    }
}

static uint64_t tensor_nbytes(uint32_t t, uint64_t elems) {
    uint64_t be, bb;
    if (block_size_of(t, &be, &bb) != 0) return 0;
    return ((elems + be - 1) / be) * bb;
}

static uint64_t row_stride(uint32_t t, uint64_t cols) { return tensor_nbytes(t, cols); }

// =========================================================================
// GGUF reader
// =========================================================================

typedef struct {
    char name[128];
    uint32_t type;
    int n_dims;
    uint64_t ne[4];
    const uint8_t *data;
    uint64_t nbytes;
    void *gpu;          // cl_mem — copy of the weights on the GPU (NULL = on the CPU)
} tensor_t;

enum { KV_STR, KV_NUM, KV_INT_ARR, KV_STR_ARR };

typedef struct {
    char key[96];
    int kind;
    char *str;          // KV_STR
    double num;         // KV_NUM
    int *iarr;          // KV_INT_ARR
    size_t iarr_n;
    char **sarr;        // KV_STR_ARR
    size_t sarr_n;
} kv_t;

typedef struct {
    mmap_t map;
    uint32_t version;
    uint64_t alignment;
    kv_t *kvs;
    size_t n_kvs;
    tensor_t *tensors;
    size_t n_tensors;
} gguf_t;

typedef struct { const uint8_t *b; uint64_t pos, size; int err; } rdr_t;

static int rneed(rdr_t *r, uint64_t n) {
    if (r->err || r->pos + n > r->size) { r->err = 1; return -1; }
    return 0;
}
static uint8_t  ru8(rdr_t *r)  { if (rneed(r,1)) return 0; return r->b[r->pos++]; }
static uint16_t ru16(rdr_t *r) { if (rneed(r,2)) return 0; uint16_t v=rd_u16(r->b+r->pos); r->pos+=2; return v; }
static uint32_t ru32(rdr_t *r) { if (rneed(r,4)) return 0; uint32_t v=rd_u32(r->b+r->pos); r->pos+=4; return v; }
static uint64_t ru64(rdr_t *r) { if (rneed(r,8)) return 0; uint64_t v=rd_u64(r->b+r->pos); r->pos+=8; return v; }
static float    rf32(rdr_t *r) { uint32_t v=ru32(r); float f; memcpy(&f,&v,4); return f; }
static double   rf64(rdr_t *r) { uint64_t v=ru64(r); double f; memcpy(&f,&v,8); return f; }

// reads a GGUF string into newly allocated NUL-terminated memory
static char *rstr(rdr_t *r) {
    uint64_t n = ru64(r);
    if (rneed(r, n)) return NULL;
    char *s = (char *)malloc(n + 1);
    memcpy(s, r->b + r->pos, n);
    s[n] = 0;
    r->pos += n;
    return s;
}

static double read_scalar(rdr_t *r, uint32_t vt) {
    switch (vt) {
    case 0:  return (double)ru8(r);            // u8
    case 1:  return (double)(int8_t)ru8(r);    // i8
    case 2:  return (double)ru16(r);           // u16
    case 3:  return (double)(int16_t)ru16(r);  // i16
    case 4:  return (double)ru32(r);           // u32
    case 5:  return (double)(int32_t)ru32(r);  // i32
    case 6:  return (double)rf32(r);           // f32
    case 7:  return (double)ru8(r);            // bool
    case 10: return (double)ru64(r);           // u64
    case 11: return (double)(int64_t)ru64(r);  // i64
    case 12: return rf64(r);                   // f64
    default: r->err = 1; return 0;
    }
}

static int gguf_open(gguf_t *g, const char *path) {
    memset(g, 0, sizeof(*g));
    g->alignment = 32;
    if (mmap_open(&g->map, path) != 0) {
        fprintf(stderr, "cannot open %s\n", path);
        return -1;
    }
    if (g->map.size < 16 || memcmp(g->map.data, "GGUF", 4) != 0) {
        fprintf(stderr, "not a GGUF file\n");
        return -1;
    }
    rdr_t r = { g->map.data, 4, g->map.size, 0 };
    g->version = ru32(&r);
    uint64_t n_tensors = ru64(&r);
    uint64_t n_kvs = ru64(&r);

    g->kvs = (kv_t *)calloc(n_kvs, sizeof(kv_t));
    g->n_kvs = n_kvs;

    for (uint64_t i = 0; i < n_kvs && !r.err; i++) {
        char *key = rstr(&r);
        if (!key) return -1;
        kv_t *kv = &g->kvs[i];
        snprintf(kv->key, sizeof(kv->key), "%s", key);
        free(key);
        uint32_t vt = ru32(&r);
        if (vt == 8) { // string
            kv->kind = KV_STR;
            kv->str = rstr(&r);
        } else if (vt == 9) { // array
            uint32_t et = ru32(&r);
            uint64_t n = ru64(&r);
            if (et == 8) {
                kv->kind = KV_STR_ARR;
                kv->sarr = (char **)malloc(n * sizeof(char *));
                kv->sarr_n = n;
                for (uint64_t j = 0; j < n && !r.err; j++) kv->sarr[j] = rstr(&r);
            } else if (et == 4 || et == 5 || et == 7 || et == 0 || et == 1) {
                kv->kind = KV_INT_ARR;
                kv->iarr = (int *)malloc(n * sizeof(int));
                kv->iarr_n = n;
                for (uint64_t j = 0; j < n && !r.err; j++)
                    kv->iarr[j] = (int)read_scalar(&r, et);
            } else { // array of another type: skip it
                kv->kind = KV_NUM;
                kv->num = (double)n;
                for (uint64_t j = 0; j < n && !r.err; j++) read_scalar(&r, et);
            }
        } else {
            kv->kind = KV_NUM;
            kv->num = read_scalar(&r, vt);
        }
    }
    if (r.err) { fprintf(stderr, "metadata parse error\n"); return -1; }

    // alignment, if present
    for (size_t i = 0; i < g->n_kvs; i++)
        if (strcmp(g->kvs[i].key, "general.alignment") == 0 && g->kvs[i].kind == KV_NUM && g->kvs[i].num > 0)
            g->alignment = (uint64_t)g->kvs[i].num;

    g->tensors = (tensor_t *)calloc(n_tensors, sizeof(tensor_t));
    g->n_tensors = n_tensors;
    uint64_t *offsets = (uint64_t *)malloc(n_tensors * sizeof(uint64_t));

    for (uint64_t i = 0; i < n_tensors && !r.err; i++) {
        char *name = rstr(&r);
        if (!name) { free(offsets); return -1; }
        tensor_t *t = &g->tensors[i];
        snprintf(t->name, sizeof(t->name), "%s", name);
        free(name);
        t->n_dims = (int)ru32(&r);
        uint64_t elems = 1;
        for (int j = 0; j < t->n_dims && j < 4; j++) {
            t->ne[j] = ru64(&r);
            elems *= t->ne[j];
        }
        t->type = ru32(&r);
        offsets[i] = ru64(&r);
        t->nbytes = tensor_nbytes(t->type, elems);
        if (t->nbytes == 0) {
            fprintf(stderr, "tensor %s: unsupported type %s\n", t->name, type_name(t->type));
            free(offsets);
            return -1;
        }
    }
    if (r.err) { fprintf(stderr, "tensor header parse error\n"); free(offsets); return -1; }

    uint64_t data_start = ((r.pos + g->alignment - 1) / g->alignment) * g->alignment;
    for (uint64_t i = 0; i < n_tensors; i++) {
        g->tensors[i].data = g->map.data + data_start + offsets[i];
        if (data_start + offsets[i] + g->tensors[i].nbytes > g->map.size) {
            fprintf(stderr, "tensor %s extends past EOF\n", g->tensors[i].name);
            free(offsets);
            return -1;
        }
    }
    free(offsets);
    return 0;
}

static kv_t *kv_find(gguf_t *g, const char *key) {
    for (size_t i = 0; i < g->n_kvs; i++)
        if (strcmp(g->kvs[i].key, key) == 0) return &g->kvs[i];
    return NULL;
}

static double kv_num(gguf_t *g, const char *key, double def) {
    kv_t *kv = kv_find(g, key);
    if (!kv) return def;
    if (kv->kind == KV_NUM) return kv->num;
    if (kv->kind == KV_INT_ARR && kv->iarr_n > 0) return (double)kv->iarr[0];
    return def;
}

static tensor_t *tensor_find(gguf_t *g, const char *name) {
    for (size_t i = 0; i < g->n_tensors; i++)
        if (strcmp(g->tensors[i].name, name) == 0) return &g->tensors[i];
    return NULL;
}

// =========================================================================
// Dequantization — ported 1:1 from the verified Go version
// =========================================================================

static void scale_min_k4(int j, const uint8_t *q, uint8_t *sc, uint8_t *m) {
    if (j < 4) { *sc = q[j] & 63; *m = q[j + 4] & 63; }
    else {
        *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m  = (q[j + 4] >> 4)  | ((q[j] >> 6) << 4);
    }
}

// non-linear value table for the IQ4 family (same as llama.cpp kvalues_iq4nl)
static const int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

static int dequantize_row(uint32_t type, const uint8_t *raw, float *out, int n) {
    switch (type) {
    case T_F32:
        for (int i = 0; i < n; i++) out[i] = rd_f32(raw + i * 4);
        return 0;
    case T_F16:
        for (int i = 0; i < n; i++) out[i] = FP16(rd_u16(raw + i * 2));
        return 0;
    case T_BF16:
        for (int i = 0; i < n; i++) out[i] = bf16_to_float(rd_u16(raw + i * 2));
        return 0;
    case T_Q4_0: {
        int nb = (n + 31) / 32;
        for (int b = 0; b < nb; b++) {
            const uint8_t *p = raw + b * 18;
            float d = FP16(rd_u16(p));
            const uint8_t *qs = p + 2;
            int base = b * 32;
            for (int j = 0; j < 16; j++) {
                if (base + j < n)      out[base + j]      = d * (float)((int)(qs[j] & 0x0f) - 8);
                if (base + j + 16 < n) out[base + j + 16] = d * (float)((int)(qs[j] >> 4) - 8);
            }
        }
        return 0;
    }
    case T_Q4_1: {
        int nb = (n + 31) / 32;
        for (int b = 0; b < nb; b++) {
            const uint8_t *p = raw + b * 20;
            float d = FP16(rd_u16(p)), m = FP16(rd_u16(p + 2));
            const uint8_t *qs = p + 4;
            int base = b * 32;
            for (int j = 0; j < 16; j++) {
                if (base + j < n)      out[base + j]      = d * (float)(qs[j] & 0x0f) + m;
                if (base + j + 16 < n) out[base + j + 16] = d * (float)(qs[j] >> 4) + m;
            }
        }
        return 0;
    }
    case T_Q5_0: {
        int nb = (n + 31) / 32;
        for (int b = 0; b < nb; b++) {
            const uint8_t *p = raw + b * 22;
            float d = FP16(rd_u16(p));
            uint32_t qh = rd_u32(p + 2);
            const uint8_t *qs = p + 6;
            int base = b * 32;
            for (int j = 0; j < 16; j++) {
                uint8_t xh0 = (uint8_t)((qh >> j) << 4) & 0x10;
                uint8_t xh1 = (uint8_t)(qh >> (j + 12)) & 0x10;
                if (base + j < n)      out[base + j]      = d * (float)((int)((qs[j] & 0x0f) | xh0) - 16);
                if (base + j + 16 < n) out[base + j + 16] = d * (float)((int)((qs[j] >> 4) | xh1) - 16);
            }
        }
        return 0;
    }
    case T_Q5_1: {
        int nb = (n + 31) / 32;
        for (int b = 0; b < nb; b++) {
            const uint8_t *p = raw + b * 24;
            float d = FP16(rd_u16(p)), m = FP16(rd_u16(p + 2));
            uint32_t qh = rd_u32(p + 4);
            const uint8_t *qs = p + 8;
            int base = b * 32;
            for (int j = 0; j < 16; j++) {
                uint8_t xh0 = (uint8_t)((qh >> j) << 4) & 0x10;
                uint8_t xh1 = (uint8_t)(qh >> (j + 12)) & 0x10;
                if (base + j < n)      out[base + j]      = d * (float)((qs[j] & 0x0f) | xh0) + m;
                if (base + j + 16 < n) out[base + j + 16] = d * (float)((qs[j] >> 4) | xh1) + m;
            }
        }
        return 0;
    }
    case T_Q8_0: {
        int nb = (n + 31) / 32;
        for (int b = 0; b < nb; b++) {
            const uint8_t *p = raw + b * 34;
            float d = FP16(rd_u16(p));
            const int8_t *qs = (const int8_t *)(p + 2);
            int base = b * 32;
            for (int j = 0; j < 32 && base + j < n; j++) out[base + j] = d * (float)qs[j];
        }
        return 0;
    }
    case T_Q2_K: {
        // block: scales[16] (4-bit scale + 4-bit min per 16 elements) + qs[64] + d + dmin
        int nb = (n + 255) / 256;
        for (int b = 0; b < nb; b++) {
            const uint8_t *p = raw + b * 84;
            const uint8_t *sc = p, *qs = p + 16;
            float d = FP16(rd_u16(p + 80)), dmin = FP16(rd_u16(p + 82));
            int is = 0;
            for (int half = 0; half < 2; half++) {
                const uint8_t *q = qs + half * 32;
                for (int j = 0; j < 4; j++) {           // shift = 0,2,4,6
                    for (int part = 0; part < 2; part++) {
                        uint8_t s = sc[is++];
                        float dl = d * (float)(s & 0xF), ml = dmin * (float)(s >> 4);
                        int base = b * 256 + half * 128 + j * 32 + part * 16;
                        for (int l = 0; l < 16; l++)
                            if (base + l < n)
                                out[base + l] = dl * (float)((q[part * 16 + l] >> (2 * j)) & 3) - ml;
                    }
                }
            }
        }
        return 0;
    }
    case T_Q3_K: {
        // block: hmask[32] (high 5th bit) + qs[64] (2 bits) + scales[12] (16×6-bit packed) + d
        int nb = (n + 255) / 256;
        for (int b = 0; b < nb; b++) {
            const uint8_t *p = raw + b * 110;
            const uint8_t *hm = p, *qs = p + 32;
            float d = FP16(rd_u16(p + 108));
            uint32_t a0 = rd_u32(p + 96), a1 = rd_u32(p + 100), tmp = rd_u32(p + 104);
            uint32_t aux[4];
            aux[0] = (a0 & 0x0f0f0f0fu)        | (((tmp >> 0) & 0x03030303u) << 4);
            aux[1] = (a1 & 0x0f0f0f0fu)        | (((tmp >> 2) & 0x03030303u) << 4);
            aux[2] = ((a0 >> 4) & 0x0f0f0f0fu) | (((tmp >> 4) & 0x03030303u) << 4);
            aux[3] = ((a1 >> 4) & 0x0f0f0f0fu) | (((tmp >> 6) & 0x03030303u) << 4);
            int8_t scs[16];
            memcpy(scs, aux, 16);
            int is = 0;
            for (int half = 0; half < 2; half++) {
                const uint8_t *q = qs + half * 32;
                for (int j = 0; j < 4; j++) {
                    uint8_t m = (uint8_t)(1u << (half * 4 + j));
                    for (int part = 0; part < 2; part++) {
                        float dl = d * (float)((int)scs[is++] - 32);
                        int base = b * 256 + half * 128 + j * 32 + part * 16;
                        for (int l = 0; l < 16; l++) {
                            if (base + l >= n) continue;
                            int lo = (q[part * 16 + l] >> (2 * j)) & 3;
                            int hi = (hm[part * 16 + l] & m) ? 0 : 4;
                            out[base + l] = dl * (float)(lo - hi);
                        }
                    }
                }
            }
        }
        return 0;
    }
    case T_Q4_K: {
        int nb = (n + 255) / 256;
        for (int b = 0; b < nb; b++) {
            const uint8_t *p = raw + b * 144;
            float d = FP16(rd_u16(p)), dmin = FP16(rd_u16(p + 2));
            const uint8_t *sc = p + 4, *qs = p + 16;
            int y = b * 256, is = 0, qi = 0;
            for (int j = 0; j < 256; j += 64) {
                uint8_t s1, m1, s2, m2;
                scale_min_k4(is, sc, &s1, &m1);
                scale_min_k4(is + 1, sc, &s2, &m2);
                float d1 = d * s1, mm1 = dmin * m1;
                float d2 = d * s2, mm2 = dmin * m2;
                for (int l = 0; l < 32; l++) {
                    if (y + j + l < n)      out[y + j + l]      = d1 * (float)(qs[qi + l] & 0xF) - mm1;
                    if (y + j + l + 32 < n) out[y + j + l + 32] = d2 * (float)(qs[qi + l] >> 4)  - mm2;
                }
                qi += 32; is += 2;
            }
        }
        return 0;
    }
    case T_Q5_K: {
        int nb = (n + 255) / 256;
        for (int b = 0; b < nb; b++) {
            const uint8_t *p = raw + b * 176;
            float d = FP16(rd_u16(p)), dmin = FP16(rd_u16(p + 2));
            const uint8_t *sc = p + 4, *qh = p + 16, *ql = p + 48;
            int y = b * 256, is = 0, qi = 0;
            uint8_t u1 = 1, u2 = 2;
            for (int j = 0; j < 256; j += 64) {
                uint8_t s1, m1, s2, m2;
                scale_min_k4(is, sc, &s1, &m1);
                scale_min_k4(is + 1, sc, &s2, &m2);
                float d1 = d * s1, mm1 = dmin * m1;
                float d2 = d * s2, mm2 = dmin * m2;
                for (int l = 0; l < 32; l++) {
                    float h1 = (qh[l] & u1) ? 16.0f : 0.0f;
                    float h2 = (qh[l] & u2) ? 16.0f : 0.0f;
                    if (y + j + l < n)      out[y + j + l]      = d1 * ((float)(ql[qi + l] & 0xF) + h1) - mm1;
                    if (y + j + l + 32 < n) out[y + j + l + 32] = d2 * ((float)(ql[qi + l] >> 4)  + h2) - mm2;
                }
                qi += 32; is += 2; u1 <<= 2; u2 <<= 2;
            }
        }
        return 0;
    }
    case T_Q6_K: {
        int nb = (n + 255) / 256;
        for (int b = 0; b < nb; b++) {
            const uint8_t *p = raw + b * 210;
            const uint8_t *ql = p, *qh = p + 128;
            const int8_t *sc = (const int8_t *)(p + 192);
            float d = FP16(rd_u16(p + 208));
            int y = b * 256;
            for (int half = 0; half < 2; half++) {
                int yo = y + half * 128, qlo = half * 64, qho = half * 32, so = half * 8;
                for (int l = 0; l < 32; l++) {
                    int is = l / 16;
                    int q1 = (int)(ql[qlo + l] & 0xF)      | (((int)(qh[qho + l] >> 0) & 3) << 4);
                    int q2 = (int)(ql[qlo + l + 32] & 0xF) | (((int)(qh[qho + l] >> 2) & 3) << 4);
                    int q3 = (int)(ql[qlo + l] >> 4)       | (((int)(qh[qho + l] >> 4) & 3) << 4);
                    int q4 = (int)(ql[qlo + l + 32] >> 4)  | (((int)(qh[qho + l] >> 6) & 3) << 4);
                    if (yo + l < n)      out[yo + l]      = d * (float)sc[so + is]     * (float)(q1 - 32);
                    if (yo + l + 32 < n) out[yo + l + 32] = d * (float)sc[so + is + 2] * (float)(q2 - 32);
                    if (yo + l + 64 < n) out[yo + l + 64] = d * (float)sc[so + is + 4] * (float)(q3 - 32);
                    if (yo + l + 96 < n) out[yo + l + 96] = d * (float)sc[so + is + 6] * (float)(q4 - 32);
                }
            }
        }
        return 0;
    }
    case T_IQ4_NL: {
        // block: d (fp16) + 16 bytes of nibbles — value = d * kvalues[q]
        int nb = (n + 31) / 32;
        for (int b = 0; b < nb; b++) {
            const uint8_t *p = raw + b * 18;
            float d = FP16(rd_u16(p));
            const uint8_t *qs = p + 2;
            int base = b * 32;
            for (int j = 0; j < 16; j++) {
                if (base + j < n)      out[base + j]      = d * (float)kvalues_iq4nl[qs[j] & 0xF];
                if (base + j + 16 < n) out[base + j + 16] = d * (float)kvalues_iq4nl[qs[j] >> 4];
            }
        }
        return 0;
    }
    case T_IQ4_XS: {
        // block: d (fp16) + scales_h (u16) + scales_l[4] + qs[128]
        // 6-bit scale per 32 elements: low 4 bits from scales_l + high 2 from scales_h, then -32
        int nb = (n + 255) / 256;
        for (int b = 0; b < nb; b++) {
            const uint8_t *p = raw + b * 136;
            float d = FP16(rd_u16(p));
            uint16_t sh = rd_u16(p + 2);
            const uint8_t *sl = p + 4, *qs = p + 8;
            for (int ib = 0; ib < 8; ib++) {
                int ls = ((sl[ib / 2] >> (4 * (ib % 2))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4);
                float dl = d * (float)(ls - 32);
                const uint8_t *q = qs + ib * 16;
                int base = b * 256 + ib * 32;
                for (int j = 0; j < 16; j++) {
                    if (base + j < n)      out[base + j]      = dl * (float)kvalues_iq4nl[q[j] & 0xF];
                    if (base + j + 16 < n) out[base + j + 16] = dl * (float)kvalues_iq4nl[q[j] >> 4];
                }
            }
        }
        return 0;
    }
    default:
        return -1;
    }
}

// =========================================================================
// Fast kernel: quantize activations to Q8 + AVX2 integer dot product
// this is the secret of llama.cpp's speed — integer math, no float conversion per element
// =========================================================================

// Q8 activation block: scale + 32 int8 values + sum of every 16 values
// (the sums are needed by K-quants: the min term in Q2_K/Q4_K/Q5_K and the zero offset in Q3_K/Q6_K)
typedef struct { float d; int8_t qs[32]; int16_t s16[2]; } q8_blk_t;

// quantize a float vector into Q8 blocks (once per MatVec, then every row reuses it)
static void quantize_q8(const float *x, q8_blk_t *out, int n) {
    int nb = n / 32;
    for (int b = 0; b < nb; b++) {
        const float *xb = x + b * 32;
        float amax = 0.0f;
        for (int j = 0; j < 32; j++) {
            float a = fabsf(xb[j]);
            if (a > amax) amax = a;
        }
        float d = amax / 127.0f;
        float id = d > 0.0f ? 1.0f / d : 0.0f;
        out[b].d = d;
        int s0 = 0, s1 = 0;
        for (int j = 0; j < 32; j++) {
            float v = xb[j] * id;
            int8_t qv = (int8_t)lrintf(v);
            out[b].qs[j] = qv;
            if (j < 16) s0 += qv; else s1 += qv;
        }
        out[b].s16[0] = (int16_t)s0;
        out[b].s16[1] = (int16_t)s1;
    }
}

// Q8_K activation block for K-quants: one scale per 256 values + sum of every 16 values
typedef struct { float d; int8_t qs[256]; int16_t bsums[16]; } q8k_blk_t;

static void quantize_q8k(const float *x, q8k_blk_t *out, int n) {
    for (int b = 0; b < n / 256; b++) {
        const float *xb = x + b * 256;
        float amax = 0.0f;
        for (int j = 0; j < 256; j++) {
            float a = fabsf(xb[j]);
            if (a > amax) amax = a;
        }
        float d = amax / 127.0f, id = d > 0.0f ? 1.0f / d : 0.0f;
        out[b].d = d;
        for (int g = 0; g < 16; g++) {
            int s = 0;
            for (int j = 0; j < 16; j++) {
                int v = (int)lrintf(xb[16 * g + j] * id);
                out[b].qs[16 * g + j] = (int8_t)v;
                s += v;
            }
            out[b].bsums[g] = (int16_t)s;
        }
    }
}

#if defined(__AVX2__)
// horizontal sum of a __m256 vector
static inline float hsum_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 sh = _mm_movehdup_ps(lo);
    __m128 s  = _mm_add_ps(lo, sh);
    sh = _mm_movehl_ps(sh, s);
    s  = _mm_add_ss(s, sh);
    return _mm_cvtss_f32(s);
}

// unpack 16 nibble bytes into 32 bytes: [lo×16 | hi×16] — matches the Q4_0 layout exactly
static inline __m256i nibbles32(const uint8_t *q) {
    __m128i t  = _mm_loadu_si128((const __m128i *)q);
    __m128i lo = _mm_and_si128(t, _mm_set1_epi8(0x0F));
    __m128i hi = _mm_and_si128(_mm_srli_epi16(t, 4), _mm_set1_epi8(0x0F));
    return _mm256_insertf128_si256(_mm256_castsi128_si256(lo), hi, 1);
}

// sum of products of int8 pairs (with the sign trick so maddubs can be used)
static inline __m256i mul_sum_i8(__m256i a, __m256i b) {
    __m256i ax = _mm256_sign_epi8(a, a);   // |a|
    __m256i sy = _mm256_sign_epi8(b, a);   // b with the sign of a
#if defined(__AVXVNNI__)
    return _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), ax, sy);
#elif defined(__AVX512VNNI__) && defined(__AVX512VL__)
    // AVX-512 form of the same instruction on 256-bit registers (CPUs without standalone AVX-VNNI)
    return _mm256_dpbusd_epi32(_mm256_setzero_si256(), ax, sy);
#else
    __m256i dt = _mm256_maddubs_epi16(ax, sy);
    return _mm256_madd_epi16(dt, _mm256_set1_epi16(1));
#endif
}
#endif

// dot(Q4_0 row, Q8-quantized x)
static float dot_q4_0_q8(const uint8_t *row, const q8_blk_t *xq, int nb) {
#if defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    for (int b = 0; b < nb; b++) {
        const uint8_t *p = row + b * 18;
        float d = FP16(rd_u16(p)) * xq[b].d;
        __m256i bx = _mm256_sub_epi8(nibbles32(p + 2), _mm256_set1_epi8(8));
        __m256i by = _mm256_loadu_si256((const __m256i *)xq[b].qs);
        __m256i s32 = mul_sum_i8(bx, by);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(s32), acc);
    }
    return hsum_ps(acc);
#else
    float sum = 0.0f;
    for (int b = 0; b < nb; b++) {
        const uint8_t *p = row + b * 18;
        float d = FP16(rd_u16(p)) * xq[b].d;
        const uint8_t *qs = p + 2;
        int si = 0;
        for (int j = 0; j < 16; j++) {
            si += ((int)(qs[j] & 0x0F) - 8) * xq[b].qs[j];
            si += ((int)(qs[j] >> 4)   - 8) * xq[b].qs[j + 16];
        }
        sum += d * (float)si;
    }
    return sum;
#endif
}


// two rows at once: load xq once + two independent dependency chains (better ILP)
static void dot_q4_0_q8_x2(const uint8_t *r0, const uint8_t *r1,
                           const q8_blk_t *xq, int nb, float *o0, float *o1) {
#if defined(__AVX2__)
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    const __m256i off8 = _mm256_set1_epi8(8);
    for (int b = 0; b < nb; b++) {
        const uint8_t *p0 = r0 + b * 18, *p1 = r1 + b * 18;
        _mm_prefetch((const char *)(p0 + 18 * 8), _MM_HINT_T0);
        _mm_prefetch((const char *)(p1 + 18 * 8), _MM_HINT_T0);
        float xd = xq[b].d;
        __m256i by = _mm256_loadu_si256((const __m256i *)xq[b].qs); // shared
        __m256i b0 = _mm256_sub_epi8(nibbles32(p0 + 2), off8);
        __m256i b1 = _mm256_sub_epi8(nibbles32(p1 + 2), off8);
        __m256i s0 = mul_sum_i8(b0, by);
        __m256i s1 = mul_sum_i8(b1, by);
        a0 = _mm256_fmadd_ps(_mm256_set1_ps(FP16(rd_u16(p0)) * xd), _mm256_cvtepi32_ps(s0), a0);
        a1 = _mm256_fmadd_ps(_mm256_set1_ps(FP16(rd_u16(p1)) * xd), _mm256_cvtepi32_ps(s1), a1);
    }
    *o0 = hsum_ps(a0);
    *o1 = hsum_ps(a1);
#else
    *o0 = dot_q4_0_q8(r0, xq, nb);
    *o1 = dot_q4_0_q8(r1, xq, nb);
#endif
}

// dot(Q8_0 row, Q8-quantized x)
static float dot_q8_0_q8(const uint8_t *row, const q8_blk_t *xq, int nb) {
#if defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    for (int b = 0; b < nb; b++) {
        const uint8_t *p = row + b * 34;
        float d = FP16(rd_u16(p)) * xq[b].d;
        __m256i bx = _mm256_loadu_si256((const __m256i *)(p + 2));
        __m256i by = _mm256_loadu_si256((const __m256i *)xq[b].qs);
        __m256i s32 = mul_sum_i8(bx, by);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(s32), acc);
    }
    return hsum_ps(acc);
#else
    float sum = 0.0f;
    for (int b = 0; b < nb; b++) {
        const uint8_t *p = row + b * 34;
        float d = FP16(rd_u16(p)) * xq[b].d;
        const int8_t *qs = (const int8_t *)(p + 2);
        int si = 0;
        for (int j = 0; j < 32; j++) si += (int)qs[j] * xq[b].qs[j];
        sum += d * (float)si;
    }
    return sum;
#endif
}

// -------------------------------------------------------------------------
// Q4_1 / Q5_0 / Q5_1 against Q8-quantized x. Element j of a 32-block is nibble j (low
// nibbles first, then high nibbles — the nibbles32 layout) plus, for Q5, bit j of qh as bit 4.
//   Q5_0: d*(q5-16)      → d*xd*Σ((q5-16)·y)
//   Q4_1: d*q4 + m       → d*xd*Σ(q4·y) + m*xd*Σy
//   Q5_1: d*q5 + m       → d*xd*Σ(q5·y) + m*xd*Σy     (Σy = s16[0] + s16[1])
// -------------------------------------------------------------------------

#if defined(__AVX2__)
// 32 bits → 32 bytes: 0xFF where the bit is set, 0x00 otherwise (byte j = bit j)
static inline __m256i bytes_from_bits_32(const uint8_t *p) {
    uint32_t x32;
    memcpy(&x32, p, 4);
    const __m256i shuf = _mm256_set_epi64x(0x0303030303030303LL, 0x0202020202020202LL,
                                           0x0101010101010101LL, 0x0000000000000000LL);
    __m256i bytes = _mm256_shuffle_epi8(_mm256_set1_epi32((int)x32), shuf);
    bytes = _mm256_or_si256(bytes, _mm256_set1_epi64x(0x7fbfdfeff7fbfdfeLL));
    return _mm256_cmpeq_epi8(bytes, _mm256_set1_epi64x(-1));
}
#endif

// value of element j (0..31) of a Q5 block before the offset: nibble | (qh bit j) << 4
static inline int q5_elem(const uint8_t *qs, uint32_t qh, int j) {
    int nib = j < 16 ? (qs[j] & 0x0F) : (qs[j - 16] >> 4);
    return nib | (int)(((qh >> j) & 1u) << 4);
}

// dot(Q5_0 row, Q8-quantized x)
static float dot_q5_0_q8(const uint8_t *row, const q8_blk_t *xq, int nb) {
#if defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    const __m256i f0 = _mm256_set1_epi8((char)0xF0);
    for (int b = 0; b < nb; b++) {
        const uint8_t *p = row + b * 22;
        float d = FP16(rd_u16(p)) * xq[b].d;
        // q5 - 16 as int8: high bit set → nibble, high bit clear → nibble | 0xF0 (= nibble - 16)
        __m256i hb = _mm256_andnot_si256(bytes_from_bits_32(p + 2), f0);
        __m256i bx = _mm256_or_si256(nibbles32(p + 6), hb);
        __m256i by = _mm256_loadu_si256((const __m256i *)xq[b].qs);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(mul_sum_i8(bx, by)), acc);
    }
    return hsum_ps(acc);
#else
    float sum = 0.0f;
    for (int b = 0; b < nb; b++) {
        const uint8_t *p = row + b * 22;
        uint32_t qh = rd_u32(p + 2);
        int si = 0;
        for (int j = 0; j < 32; j++) si += (q5_elem(p + 6, qh, j) - 16) * xq[b].qs[j];
        sum += FP16(rd_u16(p)) * xq[b].d * (float)si;
    }
    return sum;
#endif
}

// dot(Q5_1 row, Q8-quantized x)
static float dot_q5_1_q8(const uint8_t *row, const q8_blk_t *xq, int nb) {
#if defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    float macc = 0.0f;
    const __m256i x10 = _mm256_set1_epi8(0x10);
    for (int b = 0; b < nb; b++) {
        const uint8_t *p = row + b * 24;
        float xd = xq[b].d;
        __m256i hb = _mm256_and_si256(bytes_from_bits_32(p + 4), x10);
        __m256i bx = _mm256_or_si256(nibbles32(p + 8), hb);          // q5 in 0..31
        __m256i by = _mm256_loadu_si256((const __m256i *)xq[b].qs);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(FP16(rd_u16(p)) * xd),
                              _mm256_cvtepi32_ps(mul_sum_i8(bx, by)), acc);
        macc += FP16(rd_u16(p + 2)) * xd * (float)(xq[b].s16[0] + xq[b].s16[1]);
    }
    return hsum_ps(acc) + macc;
#else
    float sum = 0.0f;
    for (int b = 0; b < nb; b++) {
        const uint8_t *p = row + b * 24;
        uint32_t qh = rd_u32(p + 4);
        int si = 0;
        for (int j = 0; j < 32; j++) si += q5_elem(p + 8, qh, j) * xq[b].qs[j];
        sum += FP16(rd_u16(p)) * xq[b].d * (float)si +
               FP16(rd_u16(p + 2)) * xq[b].d * (float)(xq[b].s16[0] + xq[b].s16[1]);
    }
    return sum;
#endif
}

// dot(Q4_1 row, Q8-quantized x)
static float dot_q4_1_q8(const uint8_t *row, const q8_blk_t *xq, int nb) {
#if defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    float macc = 0.0f;
    for (int b = 0; b < nb; b++) {
        const uint8_t *p = row + b * 20;
        float xd = xq[b].d;
        __m256i bx = nibbles32(p + 4);                                 // q4 in 0..15
        __m256i by = _mm256_loadu_si256((const __m256i *)xq[b].qs);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(FP16(rd_u16(p)) * xd),
                              _mm256_cvtepi32_ps(mul_sum_i8(bx, by)), acc);
        macc += FP16(rd_u16(p + 2)) * xd * (float)(xq[b].s16[0] + xq[b].s16[1]);
    }
    return hsum_ps(acc) + macc;
#else
    float sum = 0.0f;
    for (int b = 0; b < nb; b++) {
        const uint8_t *p = row + b * 20;
        int si = 0;
        for (int j = 0; j < 16; j++) {
            si += (int)(p[4 + j] & 0x0F) * xq[b].qs[j];
            si += (int)(p[4 + j] >> 4)   * xq[b].qs[j + 16];
        }
        sum += FP16(rd_u16(p)) * xq[b].d * (float)si +
               FP16(rd_u16(p + 2)) * xq[b].d * (float)(xq[b].s16[0] + xq[b].s16[1]);
    }
    return sum;
#endif
}


#if defined(__AVX2__)
// horizontal sum of 8 i32 values
static inline int hsum_i32(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    lo = _mm_add_epi32(lo, hi);
    lo = _mm_hadd_epi32(lo, lo);
    lo = _mm_hadd_epi32(lo, lo);
    return _mm_cvtsi128_si32(lo);
}
#endif

// -------------------------------------------------------------------------
// K-quants (Q4_K / Q5_K / Q6_K) against Q8_K-quantized x: one float scale per 256 elements
// (like llama.cpp), so a whole 256 block is summed in integers and converted to float once.
//   Q4_K/Q5_K: element = d*sc*q - dmin*m  →  Σ = d*yd*Σ_s sc_s·Σ(q·y) - dmin*yd*Σ_s m_s·Σy
//   Q6_K:      element = d*sc*(q-32), sc per 16  →  Σ = d*yd*(Σ_g sc_g·Σ(q·y) - 32·Σ_g sc_g·Σy)
// Σy per 16 elements is precomputed in bsums. q is unsigned (≤ 63) so it feeds maddubs directly.
// nb = number of 256-element blocks
// -------------------------------------------------------------------------

// the 12 packed scale bytes of Q4_K/Q5_K → out[0..7] = six-bit scales, out[8..15] = six-bit mins
static inline void unpack_scales_k4(const uint8_t *s12, uint8_t out[16]) {
    uint32_t u[4];
    memcpy(u, s12, 12);
    const uint32_t k1 = 0x3f3f3f3f, k2 = 0x0f0f0f0f, k3 = 0x03030303;
    u[3] = ((u[2] >> 4) & k2) | (((u[1] >> 6) & k3) << 4);
    uint32_t aux = u[1] & k1;
    u[1] = (u[2] & k2) | (((u[0] >> 6) & k3) << 4);
    u[2] = aux;
    u[0] &= k1;
    memcpy(out, u, 16);
}

// Σ_s m_s · (Σy over the 32 elements of sub-block s)
static inline int k4_min_sum(const uint8_t sm[16], const q8k_blk_t *y) {
    int s = 0;
    for (int j = 0; j < 8; j++) s += sm[8 + j] * (y->bsums[2 * j] + y->bsums[2 * j + 1]);
    return s;
}

static float dot_q4_K_q8k(const uint8_t *row, const q8k_blk_t *y, int nb) {
    float sum = 0.0f;
#if defined(__AVX2__)
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    __m256 acc = _mm256_setzero_ps();
#endif
    for (int b = 0; b < nb; b++) {
        const uint8_t *p = row + b * 144, *qs = p + 16;
        const int8_t *q8 = y[b].qs;
        uint8_t sm[16];
        unpack_scales_k4(p + 4, sm);
        float yd = y[b].d;
#if defined(__AVX2__)
        __m256i sumi = _mm256_setzero_si256();
        for (int j = 0; j < 4; j++) {     // 64 elements: low nibbles = sub-block 2j, high = 2j+1
            __m256i q4 = _mm256_loadu_si256((const __m256i *)(qs + 32 * j));
            __m256i lo = _mm256_and_si256(q4, m4);
            __m256i hi = _mm256_and_si256(_mm256_srli_epi16(q4, 4), m4);
            __m256i pl = _mm256_maddubs_epi16(lo, _mm256_loadu_si256((const __m256i *)(q8 + 64 * j)));
            __m256i ph = _mm256_maddubs_epi16(hi, _mm256_loadu_si256((const __m256i *)(q8 + 64 * j + 32)));
            pl = _mm256_madd_epi16(pl, _mm256_set1_epi16(sm[2 * j]));
            ph = _mm256_madd_epi16(ph, _mm256_set1_epi16(sm[2 * j + 1]));
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(pl, ph));
        }
        acc = _mm256_fmadd_ps(_mm256_set1_ps(FP16(rd_u16(p)) * yd), _mm256_cvtepi32_ps(sumi), acc);
#else
        int sumi = 0;
        for (int j = 0; j < 4; j++) {
            int i1 = 0, i2 = 0;
            for (int l = 0; l < 32; l++) {
                i1 += (qs[32 * j + l] & 0xF) * q8[64 * j + l];
                i2 += (qs[32 * j + l] >> 4)  * q8[64 * j + 32 + l];
            }
            sumi += sm[2 * j] * i1 + sm[2 * j + 1] * i2;
        }
        sum += FP16(rd_u16(p)) * yd * (float)sumi;
#endif
        sum -= FP16(rd_u16(p + 2)) * yd * (float)k4_min_sum(sm, &y[b]);
    }
#if defined(__AVX2__)
    sum += hsum_ps(acc);
#endif
    return sum;
}

static float dot_q5_K_q8k(const uint8_t *row, const q8k_blk_t *y, int nb) {
    float sum = 0.0f;
#if defined(__AVX2__)
    const __m256i m4 = _mm256_set1_epi8(0x0F), one8 = _mm256_set1_epi8(1);
    __m256 acc = _mm256_setzero_ps();
#endif
    for (int b = 0; b < nb; b++) {
        const uint8_t *p = row + b * 176, *qh = p + 16, *ql = p + 48;
        const int8_t *q8 = y[b].qs;
        uint8_t sm[16];
        unpack_scales_k4(p + 4, sm);
        float yd = y[b].d;
#if defined(__AVX2__)
        // 5th bit: sub-block s takes bit s of every qh byte — shift by 2 per pair
        __m256i hb = _mm256_loadu_si256((const __m256i *)qh);
        __m256i sumi = _mm256_setzero_si256();
        for (int j = 0; j < 4; j++) {
            __m256i q = _mm256_loadu_si256((const __m256i *)(ql + 32 * j));
            __m256i lo = _mm256_or_si256(_mm256_and_si256(q, m4),
                                         _mm256_slli_epi16(_mm256_and_si256(hb, one8), 4));
            __m256i hi = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q, 4), m4),
                                         _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(hb, 1), one8), 4));
            hb = _mm256_srli_epi16(hb, 2);
            __m256i pl = _mm256_maddubs_epi16(lo, _mm256_loadu_si256((const __m256i *)(q8 + 64 * j)));
            __m256i ph = _mm256_maddubs_epi16(hi, _mm256_loadu_si256((const __m256i *)(q8 + 64 * j + 32)));
            pl = _mm256_madd_epi16(pl, _mm256_set1_epi16(sm[2 * j]));
            ph = _mm256_madd_epi16(ph, _mm256_set1_epi16(sm[2 * j + 1]));
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(pl, ph));
        }
        acc = _mm256_fmadd_ps(_mm256_set1_ps(FP16(rd_u16(p)) * yd), _mm256_cvtepi32_ps(sumi), acc);
#else
        int sumi = 0;
        for (int j = 0; j < 4; j++) {
            int i1 = 0, i2 = 0;
            for (int l = 0; l < 32; l++) {
                int h = qh[l] >> (2 * j);
                i1 += ((ql[32 * j + l] & 0xF) | ((h & 1) << 4)) * q8[64 * j + l];
                i2 += ((ql[32 * j + l] >> 4)  | ((h & 2) << 3)) * q8[64 * j + 32 + l];
            }
            sumi += sm[2 * j] * i1 + sm[2 * j + 1] * i2;
        }
        sum += FP16(rd_u16(p)) * yd * (float)sumi;
#endif
        sum -= FP16(rd_u16(p + 2)) * yd * (float)k4_min_sum(sm, &y[b]);
    }
#if defined(__AVX2__)
    sum += hsum_ps(acc);
#endif
    return sum;
}

static float dot_q6_K_q8k(const uint8_t *row, const q8k_blk_t *y, int nb) {
    float sum = 0.0f;
#if defined(__AVX2__)
    const __m256i m4 = _mm256_set1_epi8(0x0F), m3 = _mm256_set1_epi8(3);
    __m256 acc = _mm256_setzero_ps();
#endif
    for (int b = 0; b < nb; b++) {
        const uint8_t *p = row + b * 210;
        const int8_t *sc = (const int8_t *)(p + 192);
        float dd = FP16(rd_u16(p + 208)) * y[b].d;
        int corr = 0;                      // Σ_g sc_g · Σy_g  (the -32 offset)
        for (int g = 0; g < 16; g++) corr += sc[g] * y[b].bsums[g];
#if defined(__AVX2__)
        __m256i sumi = _mm256_setzero_si256();
        for (int half = 0; half < 2; half++) {
            const uint8_t *ql = p + half * 64, *qh = p + 128 + half * 32;
            const int8_t *s = sc + half * 8, *q8 = y[b].qs + half * 128;
            __m256i a0 = _mm256_loadu_si256((const __m256i *)ql);
            __m256i a1 = _mm256_loadu_si256((const __m256i *)(ql + 32));
            __m256i hh = _mm256_loadu_si256((const __m256i *)qh);
            __m256i q[4];
            q[0] = _mm256_or_si256(_mm256_and_si256(a0, m4),
                                   _mm256_slli_epi16(_mm256_and_si256(hh, m3), 4));
            q[1] = _mm256_or_si256(_mm256_and_si256(a1, m4),
                                   _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(hh, 2), m3), 4));
            q[2] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(a0, 4), m4),
                                   _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(hh, 4), m3), 4));
            q[3] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(a1, 4), m4),
                                   _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(hh, 6), m3), 4));
            for (int k = 0; k < 4; k++) {
                // scale per 16 elements: low half of the register uses s[2k], high half s[2k+1]
                __m256i sv = _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_set1_epi16(s[2 * k])),
                                                     _mm_set1_epi16(s[2 * k + 1]), 1);
                __m256i dt = _mm256_maddubs_epi16(q[k], _mm256_loadu_si256((const __m256i *)(q8 + 32 * k)));
                sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(dt, sv));
            }
        }
        acc = _mm256_fmadd_ps(_mm256_set1_ps(dd), _mm256_cvtepi32_ps(sumi), acc);
        sum -= dd * 32.0f * (float)corr;
#else
        int sumi = 0;
        for (int half = 0; half < 2; half++) {
            const uint8_t *ql = p + half * 64, *qh = p + 128 + half * 32;
            const int8_t *s = sc + half * 8, *q8 = y[b].qs + half * 128;
            for (int k = 0; k < 4; k++) {
                // k: 0/1 low nibble of ql[l]/ql[l+32], 2/3 high nibble — and the qh bits at 2k
                const uint8_t *lq = ql + (k & 1) * 32;
                int lsh = (k >> 1) * 4, hsh = 2 * k, i0 = 0, i1 = 0;
                for (int l = 0; l < 32; l++) {
                    int q = ((lq[l] >> lsh) & 0xF) | (((qh[l] >> hsh) & 3) << 4);
                    if (l < 16) i0 += q * q8[32 * k + l]; else i1 += q * q8[32 * k + l];
                }
                sumi += s[2 * k] * i0 + s[2 * k + 1] * i1;
            }
        }
        sum += dd * (float)(sumi - 32 * corr);
#endif
    }
#if defined(__AVX2__)
    sum += hsum_ps(acc);
#endif
    return sum;
}

// dot(Q2_K row, Q8-quantized x) — value d*(s&0xF)*q - dmin*(s>>4) per 16 elements:
//   Σ = d*sc*(q·xq) - dmin*m*Σxq with two scales per 32-vector (same trick as Q6_K)
static float dot_q2_K_q8(const uint8_t *row, const q8_blk_t *xq, int nb256) {
    float sum = 0.0f;
    for (int b = 0; b < nb256; b++) {
        const uint8_t *p = row + b * 84;
        const uint8_t *sc = p, *qs = p + 16;
        float d = FP16(rd_u16(p + 80)), dmin = FP16(rd_u16(p + 82));
        const q8_blk_t *xa = xq + b * 8;
#if defined(__AVX2__)
        const __m256i m3 = _mm256_set1_epi8(0x03);
        for (int half = 0; half < 2; half++) {
            __m256i bits = _mm256_loadu_si256((const __m256i *)(qs + half * 32));
            for (int j = 0; j < 4; j++) {
                __m256i qv = _mm256_and_si256(_mm256_srli_epi16(bits, 2 * j), m3);
                uint8_t sA = sc[half * 8 + 2 * j], sB = sc[half * 8 + 2 * j + 1];
                int scA = sA & 0xF, mA = sA >> 4;
                int scB = sB & 0xF, mB = sB >> 4;
                const q8_blk_t *xk = &xa[half * 4 + j];
                __m256i p16 = _mm256_maddubs_epi16(qv,
                                  _mm256_loadu_si256((const __m256i *)xk->qs));
                __m256i scv = _mm256_setr_epi16(
                    (short)scA, (short)scA, (short)scA, (short)scA,
                    (short)scA, (short)scA, (short)scA, (short)scA,
                    (short)scB, (short)scB, (short)scB, (short)scB,
                    (short)scB, (short)scB, (short)scB, (short)scB);
                int S = hsum_i32(_mm256_madd_epi16(p16, scv));
                int M = mA * (int)xk->s16[0] + mB * (int)xk->s16[1];
                sum += xk->d * (d * (float)S - dmin * (float)M);
            }
        }
#else
        for (int half = 0; half < 2; half++) {
            const uint8_t *q = qs + half * 32;
            for (int j = 0; j < 4; j++) {
                uint8_t sA = sc[half * 8 + 2 * j], sB = sc[half * 8 + 2 * j + 1];
                int scA = sA & 0xF, mA = sA >> 4;
                int scB = sB & 0xF, mB = sB >> 4;
                const q8_blk_t *xk = &xa[half * 4 + j];
                int S = 0;
                for (int l = 0; l < 32; l++) {
                    int qv = (q[l] >> (2 * j)) & 3;
                    S += qv * (l < 16 ? scA : scB) * xk->qs[l];
                }
                int M = mA * (int)xk->s16[0] + mB * (int)xk->s16[1];
                sum += xk->d * (d * (float)S - dmin * (float)M);
            }
        }
#endif
    }
    return sum;
}

// dot(Q3_K row, Q8-quantized x) — value d*(sc-32)*(q-4h̄) where h̄=0 if the high bit is set:
//   build qu = q | (hbit<<2) ∈ [0,7] so the value is (qu-4) and the sum is sc*(qu·xq) - 4*sc*Σxq
static float dot_q3_K_q8(const uint8_t *row, const q8_blk_t *xq, int nb256) {
    float sum = 0.0f;
    for (int b = 0; b < nb256; b++) {
        const uint8_t *p = row + b * 110;
        const uint8_t *hm = p, *qs = p + 32;
        float d = FP16(rd_u16(p + 108));
        uint32_t a0 = rd_u32(p + 96), a1 = rd_u32(p + 100), tmp = rd_u32(p + 104);
        uint32_t aux[4];
        aux[0] = (a0 & 0x0f0f0f0fu)        | (((tmp >> 0) & 0x03030303u) << 4);
        aux[1] = (a1 & 0x0f0f0f0fu)        | (((tmp >> 2) & 0x03030303u) << 4);
        aux[2] = ((a0 >> 4) & 0x0f0f0f0fu) | (((tmp >> 4) & 0x03030303u) << 4);
        aux[3] = ((a1 >> 4) & 0x0f0f0f0fu) | (((tmp >> 6) & 0x03030303u) << 4);
        int8_t scs[16];
        memcpy(scs, aux, 16);
        const q8_blk_t *xa = xq + b * 8;
#if defined(__AVX2__)
        const __m256i m3 = _mm256_set1_epi8(0x03);
        const __m256i four = _mm256_set1_epi8(0x04);
        __m256i hbits = _mm256_loadu_si256((const __m256i *)hm);
        for (int half = 0; half < 2; half++) {
            __m256i bits = _mm256_loadu_si256((const __m256i *)(qs + half * 32));
            for (int j = 0; j < 4; j++) {
                __m256i mv = _mm256_set1_epi8((char)(1u << (half * 4 + j)));
                __m256i hv = _mm256_and_si256(
                    _mm256_cmpeq_epi8(_mm256_and_si256(hbits, mv), mv), four);
                __m256i qu = _mm256_or_si256(
                    _mm256_and_si256(_mm256_srli_epi16(bits, 2 * j), m3), hv);
                int scA = (int)scs[half * 8 + 2 * j] - 32;
                int scB = (int)scs[half * 8 + 2 * j + 1] - 32;
                const q8_blk_t *xk = &xa[half * 4 + j];
                __m256i p16 = _mm256_maddubs_epi16(qu,
                                  _mm256_loadu_si256((const __m256i *)xk->qs));
                __m256i scv = _mm256_setr_epi16(
                    (short)scA, (short)scA, (short)scA, (short)scA,
                    (short)scA, (short)scA, (short)scA, (short)scA,
                    (short)scB, (short)scB, (short)scB, (short)scB,
                    (short)scB, (short)scB, (short)scB, (short)scB);
                int S = hsum_i32(_mm256_madd_epi16(p16, scv));
                int off = 4 * (scA * (int)xk->s16[0] + scB * (int)xk->s16[1]);
                sum += d * xk->d * (float)(S - off);
            }
        }
#else
        for (int half = 0; half < 2; half++) {
            const uint8_t *q = qs + half * 32;
            for (int j = 0; j < 4; j++) {
                uint8_t m = (uint8_t)(1u << (half * 4 + j));
                int scA = (int)scs[half * 8 + 2 * j] - 32;
                int scB = (int)scs[half * 8 + 2 * j + 1] - 32;
                const q8_blk_t *xk = &xa[half * 4 + j];
                int S = 0;
                for (int l = 0; l < 32; l++) {
                    int qu = ((q[l] >> (2 * j)) & 3) | ((hm[l] & m) ? 4 : 0);
                    S += qu * (l < 16 ? scA : scB) * xk->qs[l];
                }
                int off = 4 * (scA * (int)xk->s16[0] + scB * (int)xk->s16[1]);
                sum += d * xk->d * (float)(S - off);
            }
        }
#endif
    }
    return sum;
}

// dot(IQ4_NL row, Q8-quantized x) — 32-element block: value d * kvalues[nibble]
// AVX2: the kvalues table is applied with shuffle_epi8 (16-entry lookup per lane)
static float dot_iq4_nl_q8(const uint8_t *row, const q8_blk_t *xq, int nb) {
#if defined(__AVX2__)
    const __m256i ktbl = _mm256_broadcastsi128_si256(
        _mm_loadu_si128((const __m128i *)kvalues_iq4nl));
    __m256 acc = _mm256_setzero_ps();
    for (int b = 0; b < nb; b++) {
        const uint8_t *p = row + b * 18;
        float d = FP16(rd_u16(p)) * xq[b].d;
        __m256i bx = _mm256_shuffle_epi8(ktbl, nibbles32(p + 2));
        __m256i by = _mm256_loadu_si256((const __m256i *)xq[b].qs);
        __m256i s32 = mul_sum_i8(bx, by);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(s32), acc);
    }
    return hsum_ps(acc);
#else
    float sum = 0.0f;
    for (int b = 0; b < nb; b++) {
        const uint8_t *p = row + b * 18;
        float d = FP16(rd_u16(p)) * xq[b].d;
        const uint8_t *qs = p + 2;
        int si = 0;
        for (int j = 0; j < 16; j++) {
            si += (int)kvalues_iq4nl[qs[j] & 0xF] * xq[b].qs[j];
            si += (int)kvalues_iq4nl[qs[j] >> 4]  * xq[b].qs[j + 16];
        }
        sum += d * (float)si;
    }
    return sum;
#endif
}

// dot(IQ4_XS row, Q8-quantized x) — 256 block = 8 activation blocks
// value d*(ls-32)*kvalues[nibble] per 32-element sub-block
static float dot_iq4_xs_q8(const uint8_t *row, const q8_blk_t *xq, int nb256) {
    float sum = 0.0f;
    for (int b = 0; b < nb256; b++) {
        const uint8_t *p = row + b * 136;
        float d = FP16(rd_u16(p));
        uint16_t sh = rd_u16(p + 2);
        const uint8_t *sl = p + 4, *qs = p + 8;
        const q8_blk_t *xa = xq + b * 8;
#if defined(__AVX2__)
        const __m256i ktbl = _mm256_broadcastsi128_si256(
            _mm_loadu_si128((const __m128i *)kvalues_iq4nl));
        for (int ib = 0; ib < 8; ib++) {
            int ls = ((sl[ib / 2] >> (4 * (ib % 2))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4);
            __m256i bx = _mm256_shuffle_epi8(ktbl, nibbles32(qs + ib * 16));
            __m256i by = _mm256_loadu_si256((const __m256i *)xa[ib].qs);
            int s32 = hsum_i32(mul_sum_i8(bx, by));
            sum += d * (float)(ls - 32) * xa[ib].d * (float)s32;
        }
#else
        for (int ib = 0; ib < 8; ib++) {
            int ls = ((sl[ib / 2] >> (4 * (ib % 2))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4);
            const uint8_t *q = qs + ib * 16;
            int si = 0;
            for (int j = 0; j < 16; j++) {
                si += (int)kvalues_iq4nl[q[j] & 0xF] * xa[ib].qs[j];
                si += (int)kvalues_iq4nl[q[j] >> 4]  * xa[ib].qs[j + 16];
            }
            sum += d * (float)(ls - 32) * xa[ib].d * (float)si;
        }
#endif
    }
    return sum;
}

// plain float dot product (the compiler auto-vectorizes it with -O3 -march=native)
static float dot_f32v(const float *a, const float *b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

static float dot_f16_row(const uint8_t *row, const float *x, int n) {
#if defined(__F16C__) && defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 w = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(row + i * 2)));
        acc = _mm256_fmadd_ps(w, _mm256_loadu_ps(x + i), acc);
    }
    float s = hsum_ps(acc);
    for (; i < n; i++) s += FP16(rd_u16(row + i * 2)) * x[i];
    return s;
#else
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += FP16(rd_u16(row + i * 2)) * x[i];
    return s;
#endif
}

// =========================================================================
// GPU (OpenCL) — the same code runs on NVIDIA, AMD and Intel
// loaded dynamically from OpenCL.dll (ships with every GPU driver) — no SDK and no build-time
// dependency; without a GPU or driver everything simply keeps running on the CPU.
// Control: FASTGGUF_GPU = auto (default: discrete GPU only) | on (integrated too) | off
//          FASTGGUF_GPU_DEVICE = device index from the [gpu] list
// =========================================================================

typedef int32_t  cl_int;
typedef uint32_t cl_uint;
typedef uint64_t cl_ulong;
typedef cl_ulong cl_bitfield;
typedef void *cl_platform_id, *cl_device_id, *cl_context, *cl_command_queue,
             *cl_program, *cl_kernel, *cl_mem;

#define CL_SUCCESS                    0
#define CL_DEVICE_TYPE_GPU            (1ULL << 2)
#define CL_DEVICE_NAME                0x102B
#define CL_DEVICE_GLOBAL_MEM_SIZE     0x101F
#define CL_DEVICE_MAX_MEM_ALLOC_SIZE  0x1010
#define CL_DEVICE_HOST_UNIFIED_MEMORY 0x1035
#define CL_MEM_READ_ONLY              (1ULL << 2)
#define CL_MEM_WRITE_ONLY             (1ULL << 1)
#define CL_PROGRAM_BUILD_LOG          0x1183

static struct {
    cl_int (*GetPlatformIDs)(cl_uint, cl_platform_id *, cl_uint *);
    cl_int (*GetDeviceIDs)(cl_platform_id, cl_bitfield, cl_uint, cl_device_id *, cl_uint *);
    cl_int (*GetDeviceInfo)(cl_device_id, cl_uint, size_t, void *, size_t *);
    cl_context (*CreateContext)(const intptr_t *, cl_uint, const cl_device_id *, void *, void *, cl_int *);
    cl_command_queue (*CreateCommandQueue)(cl_context, cl_device_id, cl_bitfield, cl_int *);
    cl_program (*CreateProgramWithSource)(cl_context, cl_uint, const char **, const size_t *, cl_int *);
    cl_int (*BuildProgram)(cl_program, cl_uint, const cl_device_id *, const char *, void *, void *);
    cl_int (*GetProgramBuildInfo)(cl_program, cl_device_id, cl_uint, size_t, void *, size_t *);
    cl_kernel (*CreateKernel)(cl_program, const char *, cl_int *);
    cl_int (*SetKernelArg)(cl_kernel, cl_uint, size_t, const void *);
    cl_mem (*CreateBuffer)(cl_context, cl_bitfield, size_t, void *, cl_int *);
    cl_int (*EnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_uint, size_t, size_t, const void *, cl_uint, const void *, void *);
    cl_int (*EnqueueReadBuffer)(cl_command_queue, cl_mem, cl_uint, size_t, size_t, void *, cl_uint, const void *, void *);
    cl_int (*EnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t *, const size_t *, const size_t *, cl_uint, const void *, void *);
    cl_int (*Finish)(cl_command_queue);
    cl_int (*ReleaseMemObject)(cl_mem);
} cl;

#ifndef _WIN32
#include <dlfcn.h>
#endif

static int cl_load_runtime(void) {
#ifdef _WIN32
    HMODULE h = LoadLibraryA("OpenCL.dll");
    if (!h) return -1;
#define CLSYM(n) (*(void **)&cl.n = (void *)GetProcAddress(h, "cl" #n))
#else
    void *h = dlopen("libOpenCL.so.1", RTLD_NOW);
    if (!h) h = dlopen("libOpenCL.so", RTLD_NOW);
    if (!h) return -1;
#define CLSYM(n) (*(void **)&cl.n = dlsym(h, "cl" #n))
#endif
    if (!CLSYM(GetPlatformIDs) || !CLSYM(GetDeviceIDs) || !CLSYM(GetDeviceInfo) ||
        !CLSYM(CreateContext) || !CLSYM(CreateCommandQueue) ||
        !CLSYM(CreateProgramWithSource) || !CLSYM(BuildProgram) ||
        !CLSYM(GetProgramBuildInfo) || !CLSYM(CreateKernel) || !CLSYM(SetKernelArg) ||
        !CLSYM(CreateBuffer) || !CLSYM(EnqueueWriteBuffer) || !CLSYM(EnqueueReadBuffer) ||
        !CLSYM(EnqueueNDRangeKernel) || !CLSYM(Finish) || !CLSYM(ReleaseMemObject))
        return -1;
#undef CLSYM
    return 0;
}

// one kernel per quant type: out[row] = dot(W[row], x) — one work-group per row
#define GPU_WGS 64
enum { GK_F32, GK_F16, GK_BF16, GK_Q4_0, GK_Q4_1, GK_Q5_0, GK_Q5_1, GK_Q8_0,
       GK_Q4_K, GK_Q5_K, GK_Q6_K, GK_IQ4_NL, GK_COUNT };

static const char *GPU_KERNEL_NAMES[GK_COUNT] = {
    "mv_f32", "mv_f16", "mv_bf16", "mv_q4_0", "mv_q4_1", "mv_q5_0", "mv_q5_1",
    "mv_q8_0", "mv_q4_K", "mv_q5_K", "mv_q6_K", "mv_iq4_nl",
};

static int gpu_kernel_for(uint32_t type) {
    switch (type) {
    case T_F32:  return GK_F32;   case T_F16:  return GK_F16;  case T_BF16: return GK_BF16;
    case T_Q4_0: return GK_Q4_0;  case T_Q4_1: return GK_Q4_1;
    case T_Q5_0: return GK_Q5_0;  case T_Q5_1: return GK_Q5_1; case T_Q8_0: return GK_Q8_0;
    case T_Q4_K: return GK_Q4_K;  case T_Q5_K: return GK_Q5_K; case T_Q6_K: return GK_Q6_K;
    case T_IQ4_NL: return GK_IQ4_NL;
    default: return -1;   // Q2_K/Q3_K/IQ4_XS stay on the CPU
    }
}

// kernel argument n: number of blocks (or number of elements for float types)
static int gpu_nblocks(uint32_t type, int cols) {
    switch (type) {
    case T_F32: case T_F16: case T_BF16: return cols;
    case T_Q4_0: case T_Q4_1: case T_Q5_0: case T_Q5_1: case T_Q8_0: case T_IQ4_NL:
        return cols % 32 == 0 ? cols / 32 : -1;
    case T_Q4_K: case T_Q5_K: case T_Q6_K:
        return cols % 256 == 0 ? cols / 256 : -1;
    default: return -1;
    }
}

// kernel source — dequantization copied verbatim from dequantize_row above
static const char *GPU_SRC =
"inline float h2f(__global const uchar *p){ return vload_half(0,(__global const half*)p); }\n"
"void sm4(int j, __global const uchar *q, uchar *sc, uchar *m){\n"
"  if(j<4){ *sc=q[j]&63; *m=q[j+4]&63; }\n"
"  else { *sc=(uchar)((q[j+4]&0xF)|((q[j-4]>>6)<<4)); *m=(uchar)((q[j+4]>>4)|((q[j]>>6)<<4)); }\n"
"}\n"
"__constant float IQ4NL[16]={-127.f,-104.f,-83.f,-65.f,-49.f,-35.f,-22.f,-10.f,1.f,13.f,25.f,38.f,53.f,69.f,89.f,113.f};\n"
"#define ARGS __global const uchar *wv, ulong st, __global const float *x, __global float *y, int n\n"
"#define HEAD int row=get_group_id(0); int lid=get_local_id(0); __global const uchar *w=wv+(ulong)row*st; float acc=0.f;\n"
"#define TAIL __local float red[WGS]; red[lid]=acc; barrier(CLK_LOCAL_MEM_FENCE); for(int s=WGS/2;s>0;s>>=1){ if(lid<s) red[lid]+=red[lid+s]; barrier(CLK_LOCAL_MEM_FENCE); } if(lid==0) y[row]=red[0];\n"
"__kernel void mv_f32(ARGS){ HEAD __global const float *f=(__global const float*)w;\n"
"  for(int i=lid;i<n;i+=WGS) acc+=f[i]*x[i]; TAIL }\n"
"__kernel void mv_f16(ARGS){ HEAD __global const half *hh=(__global const half*)w;\n"
"  for(int i=lid;i<n;i+=WGS) acc+=vload_half(i,hh)*x[i]; TAIL }\n"
"__kernel void mv_bf16(ARGS){ HEAD __global const ushort *u=(__global const ushort*)w;\n"
"  for(int i=lid;i<n;i+=WGS) acc+=as_float((uint)u[i]<<16)*x[i]; TAIL }\n"
"__kernel void mv_q4_0(ARGS){ HEAD\n"
"  for(int b=lid;b<n;b+=WGS){ __global const uchar *p=w+b*18; float d=h2f(p); float s=0.f;\n"
"    for(int j=0;j<16;j++){ int q=p[2+j];\n"
"      s+=(float)((q&15)-8)*x[b*32+j]+(float)((q>>4)-8)*x[b*32+j+16]; }\n"
"    acc+=d*s; } TAIL }\n"
"__kernel void mv_q4_1(ARGS){ HEAD\n"
"  for(int b=lid;b<n;b+=WGS){ __global const uchar *p=w+b*20; float d=h2f(p), m=h2f(p+2); float s=0.f, sx=0.f;\n"
"    for(int j=0;j<16;j++){ int q=p[4+j]; float x0=x[b*32+j], x1=x[b*32+j+16];\n"
"      s+=(float)(q&15)*x0+(float)(q>>4)*x1; sx+=x0+x1; }\n"
"    acc+=d*s+m*sx; } TAIL }\n"
"__kernel void mv_q5_0(ARGS){ HEAD\n"
"  for(int b=lid;b<n;b+=WGS){ __global const uchar *p=w+b*22; float d=h2f(p);\n"
"    uint qh=(uint)p[2]|((uint)p[3]<<8)|((uint)p[4]<<16)|((uint)p[5]<<24); float s=0.f;\n"
"    for(int j=0;j<16;j++){ int q=p[6+j];\n"
"      int v0=(int)((q&15)|(((qh>>j)<<4)&16))-16; int v1=(int)((q>>4)|((qh>>(j+12))&16))-16;\n"
"      s+=(float)v0*x[b*32+j]+(float)v1*x[b*32+j+16]; }\n"
"    acc+=d*s; } TAIL }\n"
"__kernel void mv_q5_1(ARGS){ HEAD\n"
"  for(int b=lid;b<n;b+=WGS){ __global const uchar *p=w+b*24; float d=h2f(p), m=h2f(p+2);\n"
"    uint qh=(uint)p[4]|((uint)p[5]<<8)|((uint)p[6]<<16)|((uint)p[7]<<24); float s=0.f, sx=0.f;\n"
"    for(int j=0;j<16;j++){ int q=p[8+j]; float x0=x[b*32+j], x1=x[b*32+j+16];\n"
"      s+=(float)((q&15)|(((qh>>j)<<4)&16))*x0+(float)((q>>4)|((qh>>(j+12))&16))*x1; sx+=x0+x1; }\n"
"    acc+=d*s+m*sx; } TAIL }\n"
"__kernel void mv_q8_0(ARGS){ HEAD\n"
"  for(int b=lid;b<n;b+=WGS){ __global const uchar *p=w+b*34; float d=h2f(p);\n"
"    __global const char *q=(__global const char*)(p+2); float s=0.f;\n"
"    for(int j=0;j<32;j++) s+=(float)q[j]*x[b*32+j];\n"
"    acc+=d*s; } TAIL }\n"
"__kernel void mv_q4_K(ARGS){ HEAD\n"
"  for(int b=lid;b<n;b+=WGS){ __global const uchar *p=w+b*144; float d=h2f(p), dmin=h2f(p+2);\n"
"    __global const uchar *sc=p+4, *qs=p+16; int yb=b*256, is=0, qi=0;\n"
"    for(int j=0;j<256;j+=64){ uchar s1,m1,s2,m2; sm4(is,sc,&s1,&m1); sm4(is+1,sc,&s2,&m2);\n"
"      float t1=0.f,u1=0.f,t2=0.f,u2=0.f;\n"
"      for(int l=0;l<32;l++){ int q=qs[qi+l]; float x0=x[yb+j+l], x1=x[yb+j+l+32];\n"
"        t1+=(float)(q&15)*x0; u1+=x0; t2+=(float)(q>>4)*x1; u2+=x1; }\n"
"      acc+=d*(float)s1*t1-dmin*(float)m1*u1+d*(float)s2*t2-dmin*(float)m2*u2;\n"
"      qi+=32; is+=2; } } TAIL }\n"
"__kernel void mv_q5_K(ARGS){ HEAD\n"
"  for(int b=lid;b<n;b+=WGS){ __global const uchar *p=w+b*176; float d=h2f(p), dmin=h2f(p+2);\n"
"    __global const uchar *sc=p+4, *qh=p+16, *ql=p+48; int yb=b*256, is=0, qi=0; int hb1=1, hb2=2;\n"
"    for(int j=0;j<256;j+=64){ uchar s1,m1,s2,m2; sm4(is,sc,&s1,&m1); sm4(is+1,sc,&s2,&m2);\n"
"      float t1=0.f,u1=0.f,t2=0.f,u2=0.f;\n"
"      for(int l=0;l<32;l++){ int q=ql[qi+l]; float x0=x[yb+j+l], x1=x[yb+j+l+32];\n"
"        float h1=(qh[l]&hb1)?16.f:0.f, hx=(qh[l]&hb2)?16.f:0.f;\n"
"        t1+=((float)(q&15)+h1)*x0; u1+=x0; t2+=((float)(q>>4)+hx)*x1; u2+=x1; }\n"
"      acc+=d*(float)s1*t1-dmin*(float)m1*u1+d*(float)s2*t2-dmin*(float)m2*u2;\n"
"      qi+=32; is+=2; hb1<<=2; hb2<<=2; } } TAIL }\n"
"__kernel void mv_q6_K(ARGS){ HEAD\n"
"  for(int b=lid;b<n;b+=WGS){ __global const uchar *p=w+b*210;\n"
"    __global const uchar *ql=p, *qh=p+128; __global const char *sc=(__global const char*)(p+192);\n"
"    float d=h2f(p+208); int yb=b*256;\n"
"    for(int hf=0;hf<2;hf++){ int yo=yb+hf*128, qlo=hf*64, qho=hf*32, so=hf*8;\n"
"      for(int l=0;l<32;l++){ int is=l/16; int hv=qh[qho+l];\n"
"        int q1=(ql[qlo+l]&15)|(((hv>>0)&3)<<4); int q2=(ql[qlo+l+32]&15)|(((hv>>2)&3)<<4);\n"
"        int q3=(ql[qlo+l]>>4)|(((hv>>4)&3)<<4);  int q4=(ql[qlo+l+32]>>4)|(((hv>>6)&3)<<4);\n"
"        acc+=d*((float)sc[so+is]*(float)(q1-32)*x[yo+l]+(float)sc[so+is+2]*(float)(q2-32)*x[yo+l+32]\n"
"               +(float)sc[so+is+4]*(float)(q3-32)*x[yo+l+64]+(float)sc[so+is+6]*(float)(q4-32)*x[yo+l+96]); } } } TAIL }\n"
"__kernel void mv_iq4_nl(ARGS){ HEAD\n"
"  for(int b=lid;b<n;b+=WGS){ __global const uchar *p=w+b*18; float d=h2f(p); float s=0.f;\n"
"    for(int j=0;j<16;j++){ int q=p[2+j]; s+=IQ4NL[q&15]*x[b*32+j]+IQ4NL[q>>4]*x[b*32+j+16]; }\n"
"    acc+=d*s; } TAIL }\n";

static struct {
    int state;              // 0 not initialized yet, 1 ready, -1 disabled/unavailable
    cl_context ctx;
    cl_command_queue q;
    cl_kernel k[GK_COUNT];
    cl_mem xbuf, ybuf;
    size_t xcap, ycap;      // in bytes
    cl_ulong vram, max_alloc, budget;
    uint64_t used;
    int n_offloaded, n_skip_type, n_skip_mem;
    char name[100];
} g_gpu;

typedef struct { cl_device_id dev; char name[100]; cl_ulong mem; int unified; } gpudev_t;

static int gpu_list_devices(gpudev_t *out, int cap) {
    cl_platform_id plats[8];
    cl_uint np = 0;
    if (cl.GetPlatformIDs(8, plats, &np) != CL_SUCCESS) return 0;
    int n = 0;
    for (cl_uint p = 0; p < np && n < cap; p++) {
        cl_device_id devs[8];
        cl_uint nd = 0;
        if (cl.GetDeviceIDs(plats[p], CL_DEVICE_TYPE_GPU, 8, devs, &nd) != CL_SUCCESS) continue;
        for (cl_uint d = 0; d < nd && n < cap; d++) {
            gpudev_t *g = &out[n];
            g->dev = devs[d];
            g->name[0] = 0; g->mem = 0; g->unified = 0;
            cl.GetDeviceInfo(devs[d], CL_DEVICE_NAME, sizeof(g->name) - 1, g->name, NULL);
            cl.GetDeviceInfo(devs[d], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(g->mem), &g->mem, NULL);
            cl_uint uni = 0;
            cl.GetDeviceInfo(devs[d], CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(uni), &uni, NULL);
            g->unified = (int)uni;
            n++;
        }
    }
    return n;
}

// float → fp16 (self-check only)
static uint16_t gpu_f16(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | ((x & 0x7FFFFF) >> 13));
}

static int fg_gpu_matvec_raw(cl_mem w, uint32_t type, uint64_t stride,
                             const float *x, float *out, int cols, int64_t rows);

// self-check: random rows for every type, GPU result compared against the CPU
static int gpu_selfcheck(void) {
    enum { COLS = 512, ROWS = 3 };
    static const uint32_t types[GK_COUNT] = {
        T_F32, T_F16, T_BF16, T_Q4_0, T_Q4_1, T_Q5_0, T_Q5_1, T_Q8_0,
        T_Q4_K, T_Q5_K, T_Q6_K, T_IQ4_NL,
    };
    float *x = (float *)malloc(COLS * sizeof(float));
    float *ref = (float *)malloc(COLS * sizeof(float));
    float got[ROWS];
    uint64_t seed = 0x12345678ABCDEFULL;
    for (int i = 0; i < COLS; i++) {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        x[i] = (float)((seed >> 33) & 0xFFFF) / 32768.0f - 1.0f;
    }
    int fail = 0;
    for (int ti = 0; ti < GK_COUNT && !fail; ti++) {
        uint32_t type = types[ti];
        uint64_t stride = row_stride(type, COLS);
        uint8_t *w = (uint8_t *)malloc((size_t)stride * ROWS);
        for (size_t i = 0; i < (size_t)stride * ROWS; i++) {
            seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
            w[i] = (uint8_t)(seed >> 33);
        }
        // random-byte fp16 scales may be inf/NaN — write small sane values
        uint64_t be, bb;
        block_size_of(type, &be, &bb);
        for (int r = 0; r < ROWS; r++) {
            uint8_t *rb = w + (size_t)r * stride;
            for (uint64_t b = 0; b < COLS / be; b++) {
                uint8_t *p = rb + b * bb;
                seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
                float d = 0.01f + (float)((seed >> 33) & 0xFF) / 2560.0f;
                uint16_t h = gpu_f16(d);
                switch (type) {
                case T_F32:  memcpy(p, &d, 4); break;
                case T_F16:  memcpy(p, &h, 2); break;
                case T_BF16: { uint32_t u; memcpy(&u, &d, 4); uint16_t bf = (uint16_t)(u >> 16); memcpy(p, &bf, 2); break; }
                case T_Q4_1: case T_Q5_1: case T_Q4_K: case T_Q5_K:
                    memcpy(p, &h, 2); memcpy(p + 2, &h, 2); break;
                case T_Q6_K: memcpy(p + 208, &h, 2); break;
                default:     memcpy(p, &h, 2); break;
                }
            }
        }
        // GPU
        cl_int err = 0;
        cl_mem wb = cl.CreateBuffer(g_gpu.ctx, CL_MEM_READ_ONLY, (size_t)stride * ROWS, NULL, &err);
        if (!wb || err != CL_SUCCESS ||
            cl.EnqueueWriteBuffer(g_gpu.q, wb, 1, 0, (size_t)stride * ROWS, w, 0, NULL, NULL) != CL_SUCCESS ||
            fg_gpu_matvec_raw(wb, type, stride, x, got, COLS, ROWS) != 0)
            fail = 1;
        // CPU
        for (int r = 0; r < ROWS && !fail; r++) {
            if (dequantize_row(type, w + (size_t)r * stride, ref, COLS) != 0) { fail = 1; break; }
            float want = dot_f32v(ref, x, COLS);
            float diff = fabsf(want - got[r]);
            if (diff > 1e-2f + 1e-3f * fabsf(want)) {
                fprintf(stderr, "[gpu] selfcheck %s row %d: cpu=%g gpu=%g\n",
                        type_name(type), r, want, got[r]);
                fail = 1;
            }
        }
        if (wb) cl.ReleaseMemObject(wb);
        free(w);
    }
    free(x); free(ref);
    return fail ? -1 : 0;
}

// mode 0: normal (respects FASTGGUF_GPU) — mode 1: probe (ignores off and accepts integrated)
static void fg_gpu_init_ex(int mode) {
    if (g_gpu.state != 0) return;
    g_gpu.state = -1;

    const char *env = getenv("FASTGGUF_GPU");
    int force_on = mode == 1;
    if (mode == 0 && env) {
        if (!strcmp(env, "off") || !strcmp(env, "0") || !strcmp(env, "cpu") || !strcmp(env, "none"))
            return;
        if (!strcmp(env, "on") || !strcmp(env, "1") || !strcmp(env, "force") || !strcmp(env, "gpu"))
            force_on = 1;
    }
    const char *devenv = getenv("FASTGGUF_GPU_DEVICE");
    int want_idx = devenv ? atoi(devenv) : -1;
    if (want_idx >= 0) force_on = 1;

    if (cl_load_runtime() != 0) {
        if (force_on) fprintf(stderr, "[gpu] OpenCL runtime not found - staying on CPU\n");
        return;
    }
    gpudev_t devs[16];
    int n = gpu_list_devices(devs, 16);
    if (n == 0) {
        if (force_on) fprintf(stderr, "[gpu] no OpenCL GPU found - staying on CPU\n");
        return;
    }
    for (int i = 0; i < n; i++)
        printf("[gpu] device %d: %s (%s, %llu MB)\n", i, devs[i].name,
               devs[i].unified ? "integrated" : "discrete",
               (unsigned long long)(devs[i].mem >> 20));

    int pick = -1;
    if (want_idx >= 0 && want_idx < n) pick = want_idx;
    else {
        for (int i = 0; i < n; i++)     // preference: discrete GPU with the most memory
            if (!devs[i].unified && (pick < 0 || devs[i].mem > devs[pick].mem)) pick = i;
        if (pick < 0 && force_on)       // no discrete GPU — integrated only when explicitly requested
            for (int i = 0; i < n; i++)
                if (pick < 0 || devs[i].mem > devs[pick].mem) pick = i;
        if (pick < 0) {
            printf("[gpu] only integrated GPU found - staying on CPU (set FASTGGUF_GPU=on to use it)\n");
            return;
        }
    }

    cl_int err = 0;
    g_gpu.ctx = cl.CreateContext(NULL, 1, &devs[pick].dev, NULL, NULL, &err);
    if (!g_gpu.ctx || err != CL_SUCCESS) { fprintf(stderr, "[gpu] context failed (%d)\n", err); return; }
    g_gpu.q = cl.CreateCommandQueue(g_gpu.ctx, devs[pick].dev, 0, &err);
    if (!g_gpu.q || err != CL_SUCCESS) { fprintf(stderr, "[gpu] queue failed (%d)\n", err); return; }

    cl_program prog = cl.CreateProgramWithSource(g_gpu.ctx, 1, &GPU_SRC, NULL, &err);
    if (!prog || err != CL_SUCCESS) { fprintf(stderr, "[gpu] program failed (%d)\n", err); return; }
    char opts[64];
    snprintf(opts, sizeof(opts), "-cl-mad-enable -DWGS=%d", GPU_WGS);
    if (cl.BuildProgram(prog, 1, &devs[pick].dev, opts, NULL, NULL) != CL_SUCCESS) {
        char log[4096] = {0};
        cl.GetProgramBuildInfo(prog, devs[pick].dev, CL_PROGRAM_BUILD_LOG, sizeof(log) - 1, log, NULL);
        fprintf(stderr, "[gpu] kernel build failed:\n%s\n", log);
        return;
    }
    for (int i = 0; i < GK_COUNT; i++) {
        g_gpu.k[i] = cl.CreateKernel(prog, GPU_KERNEL_NAMES[i], &err);
        if (!g_gpu.k[i] || err != CL_SUCCESS) {
            fprintf(stderr, "[gpu] kernel %s failed (%d)\n", GPU_KERNEL_NAMES[i], err);
            return;
        }
    }
    g_gpu.vram = devs[pick].mem;
    g_gpu.max_alloc = 0;
    cl.GetDeviceInfo(devs[pick].dev, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(g_gpu.max_alloc), &g_gpu.max_alloc, NULL);
    snprintf(g_gpu.name, sizeof(g_gpu.name), "%s", devs[pick].name);

    // weight memory budget: ~87% of VRAM by default, or FASTGGUF_GPU_MEM (GB)
    // for models larger than VRAM (e.g. 40B on a 16GB card) what fits is uploaded, the rest stays on the CPU
    const char *menv = getenv("FASTGGUF_GPU_MEM");
    double mem_gb = menv ? atof(menv) : 0.0;
    g_gpu.budget = mem_gb > 0 ? (cl_ulong)(mem_gb * 1073741824.0) : g_gpu.vram - g_gpu.vram / 8;

    g_gpu.state = 1;    // temporary — the self-check needs fg_gpu_matvec_raw
    if (gpu_selfcheck() != 0) {
        fprintf(stderr, "[gpu] selfcheck FAILED on %s - staying on CPU\n", g_gpu.name);
        g_gpu.state = -1;
        return;
    }
    printf("[gpu] using %s (OpenCL, %llu MB vram, %s memory)\n", g_gpu.name,
           (unsigned long long)(g_gpu.vram >> 20), devs[pick].unified ? "shared" : "dedicated");
}

static void fg_gpu_init(void) { fg_gpu_init_ex(0); }

// upload one tensor's weights to VRAM (silently skipped when full or for an unsupported type)
static void gpu_upload_tensor(tensor_t *t) {
    if (g_gpu.state != 1 || !t || t->gpu) return;
    if (gpu_kernel_for(t->type) < 0 || gpu_nblocks(t->type, (int)t->ne[0]) < 0) {
        g_gpu.n_skip_type++;
        return;
    }
    if (t->nbytes > g_gpu.max_alloc || g_gpu.used + t->nbytes > g_gpu.budget) {
        g_gpu.n_skip_mem++;
        return;
    }
    cl_int err = 0;
    cl_mem m = cl.CreateBuffer(g_gpu.ctx, CL_MEM_READ_ONLY, (size_t)t->nbytes, NULL, &err);
    if (!m || err != CL_SUCCESS) { g_gpu.n_skip_mem++; return; }
    if (cl.EnqueueWriteBuffer(g_gpu.q, m, 1, 0, (size_t)t->nbytes, t->data, 0, NULL, NULL) != CL_SUCCESS) {
        cl.ReleaseMemObject(m);
        g_gpu.n_skip_mem++;
        return;
    }
    t->gpu = m;
    g_gpu.used += t->nbytes;
    g_gpu.n_offloaded++;
}

static int gpu_grow_buf(cl_mem *buf, size_t *cap, size_t need, cl_bitfield flags) {
    if (need <= *cap) return 0;
    if (*buf) cl.ReleaseMemObject(*buf);
    cl_int err = 0;
    *buf = cl.CreateBuffer(g_gpu.ctx, flags, need, NULL, &err);
    if (!*buf || err != CL_SUCCESS) { *cap = 0; return -1; }
    *cap = need;
    return 0;
}

static int fg_gpu_matvec_raw(cl_mem w, uint32_t type, uint64_t stride,
                             const float *x, float *out, int cols, int64_t rows) {
    int ki = gpu_kernel_for(type);
    int nb = gpu_nblocks(type, cols);
    if (ki < 0 || nb < 0) return -1;
    if (gpu_grow_buf(&g_gpu.xbuf, &g_gpu.xcap, (size_t)cols * 4, CL_MEM_READ_ONLY) != 0) return -1;
    if (gpu_grow_buf(&g_gpu.ybuf, &g_gpu.ycap, (size_t)rows * 4, CL_MEM_WRITE_ONLY) != 0) return -1;
    cl_kernel k = g_gpu.k[ki];
    cl_ulong st = stride;
    if (cl.EnqueueWriteBuffer(g_gpu.q, g_gpu.xbuf, 0, 0, (size_t)cols * 4, x, 0, NULL, NULL) != CL_SUCCESS) return -1;
    if (cl.SetKernelArg(k, 0, sizeof(cl_mem), &w) != CL_SUCCESS ||
        cl.SetKernelArg(k, 1, sizeof(cl_ulong), &st) != CL_SUCCESS ||
        cl.SetKernelArg(k, 2, sizeof(cl_mem), &g_gpu.xbuf) != CL_SUCCESS ||
        cl.SetKernelArg(k, 3, sizeof(cl_mem), &g_gpu.ybuf) != CL_SUCCESS ||
        cl.SetKernelArg(k, 4, sizeof(int), &nb) != CL_SUCCESS) return -1;
    size_t gws = (size_t)rows * GPU_WGS, lws = GPU_WGS;
    if (cl.EnqueueNDRangeKernel(g_gpu.q, k, 1, NULL, &gws, &lws, 0, NULL, NULL) != CL_SUCCESS) return -1;
    if (cl.EnqueueReadBuffer(g_gpu.q, g_gpu.ybuf, 1, 0, (size_t)rows * 4, out, 0, NULL, NULL) != CL_SUCCESS) return -1;
    return 0;
}

static int fg_gpu_matvec(tensor_t *t, const float *x, float *out) {
    int cols = (int)t->ne[0];
    int64_t rows = 1;
    for (int i = 1; i < t->n_dims; i++) rows *= (int64_t)t->ne[i];
    return fg_gpu_matvec_raw((cl_mem)t->gpu, t->type, row_stride(t->type, (uint64_t)cols),
                             x, out, cols, rows);
}

// console probe: fastgguf-console gpu — returns 0 if the GPU is actually ready
static int cmd_gpuinfo(void) {
    fg_gpu_init_ex(1);
    if (g_gpu.state == 1) { printf("[gpu] READY: %s\n", g_gpu.name); return 0; }
    printf("[gpu] no usable GPU - CPU only\n");
    return 1;
}

// =========================================================================
// Multi-threaded MatVec — out[r] = dot(W[r], x)
// =========================================================================

static float **g_scratch;      // [thread][max_cols] for dequantizing K-quants
static q8_blk_t *g_xq8;        // shared buffer for the quantized x (32-element blocks)
static q8k_blk_t *g_xq8k;      // the same x as 256-element blocks, for K-quants
static int g_max_cols;

static void matvec_init_scratch(int max_cols) {
    if (max_cols <= g_max_cols) return;  // grow only — several models of different sizes may be loaded
    int nt = omp_get_max_threads();
    if (g_scratch) {
        for (int i = 0; i < nt; i++) free(g_scratch[i]);
        free(g_scratch);
        free(g_xq8);
        free(g_xq8k);
    }
    g_max_cols = max_cols;
    g_scratch = (float **)malloc(nt * sizeof(float *));
    for (int i = 0; i < nt; i++) g_scratch[i] = (float *)malloc((size_t)max_cols * sizeof(float));
    g_xq8 = (q8_blk_t *)malloc(((size_t)max_cols / 32 + 1) * sizeof(q8_blk_t));
    g_xq8k = (q8k_blk_t *)malloc(((size_t)max_cols / 256 + 1) * sizeof(q8k_blk_t));
}

// how x must be prepared for this weight type: ACT_Q8K (256-element blocks, K-quants),
// ACT_Q8 (32-element blocks) or ACT_F32 (no integer kernel: generic float path)
enum { ACT_F32 = 0, ACT_Q8 = 1, ACT_Q8K = 2 };
static int type_uses_q8(uint32_t type, int cols) {
    if ((type == T_Q4_K || type == T_Q5_K || type == T_Q6_K) && cols % 256 == 0) return ACT_Q8K;
    if ((type == T_Q4_0 || type == T_Q8_0 || type == T_IQ4_NL || type == T_Q4_1 ||
         type == T_Q5_0 || type == T_Q5_1) && cols % 32 == 0) return ACT_Q8;
    if ((type == T_Q2_K || type == T_Q3_K || type == T_IQ4_XS) && cols % 256 == 0) return ACT_Q8;
    return ACT_F32;
}

// quantize x for the given kind into the matching buffer
static void quantize_act(int kind, const float *x, q8_blk_t *xq, q8k_blk_t *xk, int n) {
    if (kind == ACT_Q8) quantize_q8(x, xq, n);
    else if (kind == ACT_Q8K) quantize_q8k(x, xk, n);
}

// one row: dot(W[r], x) — kind (ACT_*) says which pre-quantized copy of x to use: xq or xk
// unsupported type: returns 0 and sets *fail
static inline float dot_row(uint32_t type, int kind, const uint8_t *rb, const q8_blk_t *xq,
                            const q8k_blk_t *xk, const float *x, int cols, int *fail) {
    int nb32 = cols / 32, nb256 = cols / 256;
    if (kind == ACT_Q8K) {
        switch (type) {
        case T_Q4_K: return dot_q4_K_q8k(rb, xk, nb256);
        case T_Q5_K: return dot_q5_K_q8k(rb, xk, nb256);
        case T_Q6_K: return dot_q6_K_q8k(rb, xk, nb256);
        default: break;
        }
    }
    switch (kind == ACT_Q8 ? type : 0xFFFF) {
    case T_Q4_0: return dot_q4_0_q8(rb, xq, nb32);
    case T_Q8_0: return dot_q8_0_q8(rb, xq, nb32);
    case T_Q4_1: return dot_q4_1_q8(rb, xq, nb32);
    case T_Q5_0: return dot_q5_0_q8(rb, xq, nb32);
    case T_Q5_1: return dot_q5_1_q8(rb, xq, nb32);
    case T_Q2_K: return dot_q2_K_q8(rb, xq, nb256);
    case T_Q3_K: return dot_q3_K_q8(rb, xq, nb256);
    case T_IQ4_NL: return dot_iq4_nl_q8(rb, xq, nb32);
    case T_IQ4_XS: return dot_iq4_xs_q8(rb, xq, nb256);
    default: break;
    }
    switch (type) {
    case T_F32:  return dot_f32v((const float *)rb, x, cols);
    case T_F16:  return dot_f16_row(rb, x, cols);
    case T_BF16: {
        float s = 0.0f;
        for (int i = 0; i < cols; i++) s += bf16_to_float(rd_u16(rb + i * 2)) * x[i];
        return s;
    }
    default: {
        float *scr = g_scratch[omp_get_thread_num()];
        if (dequantize_row(type, rb, scr, cols) != 0) { *fail = 1; return 0.0f; }
        return dot_f32v(scr, x, cols);
    }
    }
}

static int matvec(tensor_t *t, const float *x, float *out) {
    // weights uploaded to the GPU are multiplied there — any failure disables the GPU and continues on the CPU
    if (t->gpu && g_gpu.state == 1) {
        if (fg_gpu_matvec(t, x, out) == 0) return 0;
        fprintf(stderr, "[gpu] matvec failed on %s - falling back to CPU\n", t->name);
        g_gpu.state = -1;
    }
    int cols = (int)t->ne[0];
    int64_t rows = 1;
    for (int i = 1; i < t->n_dims; i++) rows *= (int64_t)t->ne[i];
    uint64_t stride = row_stride(t->type, (uint64_t)cols);
    const uint8_t *raw = t->data;

    int use_q8 = type_uses_q8(t->type, cols);
    quantize_act(use_q8, x, g_xq8, g_xq8k, cols);
    int nb32 = cols / 32;

    int fail = 0;

    // special path: Q4_0 two rows at a time
    if (use_q8 == ACT_Q8 && t->type == T_Q4_0) {
        int64_t pairs = rows / 2;
        #pragma omp parallel for schedule(static)
        for (int64_t pr = 0; pr < pairs; pr++) {
            int64_t r = pr * 2;
            dot_q4_0_q8_x2(raw + (uint64_t)r * stride, raw + (uint64_t)(r + 1) * stride,
                           g_xq8, nb32, &out[r], &out[r + 1]);
        }
        if (rows & 1) out[rows - 1] = dot_q4_0_q8(raw + (uint64_t)(rows - 1) * stride, g_xq8, nb32);
        return 0;
    }

    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; r++)
        out[r] = dot_row(t->type, use_q8, raw + (uint64_t)r * stride, g_xq8, g_xq8k, x, cols, &fail);
    if (fail) {
        fprintf(stderr, "matvec: unsupported type %s in %s\n", type_name(t->type), t->name);
        return -1;
    }
    return 0;
}

// =========================================================================
// Basic ops: RMSNorm / GELU / RoPE / Softmax
// =========================================================================

// critical note: Gemma norm weights in GGUF already include the (+1) — multiply directly
static void rmsnorm_w(float *x, const float *w, int n, float eps) {
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float inv = (float)(1.0 / sqrt(ss / n + (double)eps));
    for (int i = 0; i < n; i++) x[i] = x[i] * inv * w[i];
}

static void rmsnorm_plain(float *x, int n, float eps) {
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float inv = (float)(1.0 / sqrt(ss / n + (double)eps));
    for (int i = 0; i < n; i++) x[i] *= inv;
}

static inline float gelu_f(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

static inline float silu_f(float x) { return x / (1.0f + expf(-x)); }

// RoPE on the first n_rot dims of each head
//   neox=1: halves (Gemma/Qwen)   neox=0: adjacent pairs (Llama/Mistral)
// freq_scale: linear position scaling (1/factor) — large Gemma 3 models use it on global layers
static void apply_rope(float *x, int total, int pos, int head_dim, int n_rot,
                       double base, double freq_scale, const float *freq_factors, int neox) {
    int half = n_rot / 2;
    for (int ho = 0; ho + head_dim <= total; ho += head_dim) {
        for (int j = 0; j < half; j++) {
            double theta = (double)pos * freq_scale / pow(base, (double)(2 * j) / (double)n_rot);
            if (freq_factors) theta /= (double)freq_factors[j];
            float c = (float)cos(theta), s = (float)sin(theta);
            int i0 = neox ? ho + j : ho + 2 * j;
            int i1 = neox ? ho + j + half : ho + 2 * j + 1;
            float v0 = x[i0], v1 = x[i1];
            x[i0] = v0 * c - v1 * s;
            x[i1] = v0 * s + v1 * c;
        }
    }
}

static void softmax_inplace(float *s, int n) {
    float mx = -INFINITY;
    for (int i = 0; i < n; i++) if (s[i] > mx) mx = s[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { s[i] = expf(s[i] - mx); sum += s[i]; }
    if (sum > 0.0f) {
        float inv = 1.0f / sum;
        for (int i = 0; i < n; i++) s[i] *= inv;
    }
}

static int argmax_f(const float *v, int n) {
    int best = 0; float bv = -INFINITY;
    for (int i = 0; i < n; i++) if (v[i] > bv) { bv = v[i]; best = i; }
    return best;
}

static uint64_t rng_next(uint64_t *s) { // xorshift64*
    uint64_t x = *s;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    *s = x;
    return x * 0x2545F4914F6CDD1DULL;
}

// =========================================================================
// TurboQuant — quantized KV cache (Google Research 2025, arXiv 2504.19874):
// a random Hadamard rotation (orthogonal — preserves dot products) makes the coordinate
// distribution near-Gaussian and concentrated, then an optimal Lloyd-Max quantizer per coordinate.
// keys K = 4-bit and values V = 2-bit + one f32 scale per head slice
// → cache memory ≈ one tenth of f32, no training and negligible accuracy loss.
// Enable: FASTGGUF_KV=turbo — requires head_dim to be a power of 2 (otherwise stays f32 automatically)
// =========================================================================

#define KVQ_MAX_HD 1024
static float g_kvq_sign[KVQ_MAX_HD]; // rotation signs ±1 — fixed seed, always the same rotation
static int g_kvq_ready = 0;

// optimal Lloyd-Max centroids for a Gaussian N(0,1)
static const float kvq_c4[16] = {
    -2.7326f, -2.0690f, -1.6181f, -1.2562f, -0.9424f, -0.6568f, -0.3881f, -0.1284f,
     0.1284f,  0.3881f,  0.6568f,  0.9424f,  1.2562f,  1.6181f,  2.0690f,  2.7326f };
static const float kvq_c2[4] = { -1.5104f, -0.4528f, 0.4528f, 1.5104f };

// record size of one head slice in the cache: f32 scale + packed codes
#define KVQ_REC(hd, bits) (4 + (hd) * (bits) / 8)

static int g_kvq_norot = 0; // debug: FASTGGUF_KV_NOROT=1 disables the rotation (raw quantization)

static void kvq_init(void) {
    if (g_kvq_ready) return;
    uint64_t s = 0x243F6A8885A308D3ULL;
    for (int i = 0; i < KVQ_MAX_HD; i++)
        g_kvq_sign[i] = (rng_next(&s) >> 63) ? 1.0f : -1.0f;
    const char *nr = getenv("FASTGGUF_KV_NOROT");
    g_kvq_norot = nr && *nr == '1';
    g_kvq_ready = 1;
}

// fast in-place Walsh-Hadamard transform (n power of 2), divided by sqrt(n) — orthogonal
static void kvq_fwht(float *x, int n) {
    for (int len = 1; len < n; len <<= 1)
        for (int i = 0; i < n; i += len << 1)
            for (int j = i; j < i + len; j++) {
                float a = x[j], b = x[j + len];
                x[j] = a + b;
                x[j + len] = a - b;
            }
    float inv = 1.0f / sqrtf((float)n);
    for (int i = 0; i < n; i++) x[i] *= inv;
}

// R·x = H(D·x)/sqrt(n): random sign flip then Hadamard
static void kvq_rotate(float *x, int n) {
    if (g_kvq_norot) return;
    for (int i = 0; i < n; i++) x[i] *= g_kvq_sign[i];
    kvq_fwht(x, n);
}

// inverse: R⁻¹·y = D·(H(y)/sqrt(n)) — since H is symmetric and H² = n·I
static void kvq_unrotate(float *x, int n) {
    if (g_kvq_norot) return;
    kvq_fwht(x, n);
    for (int i = 0; i < n; i++) x[i] *= g_kvq_sign[i];
}

// nearest centroid — boundaries are midpoints between neighbouring centroids
static inline int kvq_enc4(float u) {
    int c = 0;
    while (c < 15 && u > (kvq_c4[c] + kvq_c4[c + 1]) * 0.5f) c++;
    return c;
}
static inline int kvq_enc2(float u) {
    int c = 0;
    while (c < 3 && u > (kvq_c2[c] + kvq_c2[c + 1]) * 0.5f) c++;
    return c;
}

// quantize a head slice (after rotation):
// bits=8: linear int8 with absmax scale (~0.4% error — for models with huge attention scores like Qwen)
// bits=4/2: scale = RMS then nearest Lloyd-Max centroid per value
static void kvq_quant_slice(const float *x, int hd, uint8_t *rec, int bits) {
    if (bits == 8) {
        float amax = 0.0f;
        for (int i = 0; i < hd; i++) if (fabsf(x[i]) > amax) amax = fabsf(x[i]);
        float s = amax / 127.0f;
        memcpy(rec, &s, 4);
        int8_t *q = (int8_t *)(rec + 4);
        float inv = s > 0.0f ? 1.0f / s : 0.0f;
        for (int i = 0; i < hd; i++) q[i] = (int8_t)lrintf(x[i] * inv);
        return;
    }
    float ss = 0.0f;
    for (int i = 0; i < hd; i++) ss += x[i] * x[i];
    float s = sqrtf(ss / (float)hd);
    memcpy(rec, &s, 4);
    uint8_t *q = rec + 4;
    float inv = s > 0.0f ? 1.0f / s : 0.0f;
    if (bits == 4) {
        for (int i = 0; i < hd; i += 2)
            q[i >> 1] = (uint8_t)(kvq_enc4(x[i] * inv) | (kvq_enc4(x[i + 1] * inv) << 4));
    } else {
        for (int i = 0; i < hd; i += 4)
            q[i >> 2] = (uint8_t)( kvq_enc2(x[i] * inv)
                                 | (kvq_enc2(x[i + 1] * inv) << 2)
                                 | (kvq_enc2(x[i + 2] * inv) << 4)
                                 | (kvq_enc2(x[i + 3] * inv) << 6));
    }
}

// q·k̂ for a quantized key slice (bits = 8 or 4)
static float kvq_dot_k(const float *qh, const uint8_t *rec, int hd, int bits) {
    float s;
    memcpy(&s, rec, 4);
    float acc = 0.0f;
    if (bits == 8) {
        const int8_t *q = (const int8_t *)(rec + 4);
        for (int i = 0; i < hd; i++) acc += qh[i] * (float)q[i];
    } else {
        const uint8_t *q = rec + 4;
        for (int i = 0; i < hd; i += 2) {
            uint8_t b = q[i >> 1];
            acc += qh[i] * kvq_c4[b & 0xF] + qh[i + 1] * kvq_c4[b >> 4];
        }
    }
    return acc * s;
}

// oh += w·v̂ for a quantized value slice (bits = 8, 4 or 2)
static void kvq_axpy_v(float *oh, float w, const uint8_t *rec, int hd, int bits) {
    float s;
    memcpy(&s, rec, 4);
    const uint8_t *q = rec + 4;
    float ws = w * s;
    if (bits == 8) {
        const int8_t *q8 = (const int8_t *)q;
        for (int i = 0; i < hd; i++) oh[i] += ws * (float)q8[i];
    } else if (bits == 4) {
        for (int i = 0; i < hd; i += 2) {
            uint8_t b = q[i >> 1];
            oh[i]     += ws * kvq_c4[b & 0xF];
            oh[i + 1] += ws * kvq_c4[b >> 4];
        }
    } else {
        for (int i = 0; i < hd; i += 4) {
            uint8_t b = q[i >> 2];
            oh[i]     += ws * kvq_c2[b & 3];
            oh[i + 1] += ws * kvq_c2[(b >> 2) & 3];
            oh[i + 2] += ws * kvq_c2[(b >> 4) & 3];
            oh[i + 3] += ws * kvq_c2[b >> 6];
        }
    }
}

// decode a whole slice (selftest)
static void kvq_dequant_slice(const uint8_t *rec, float *out, int hd, int bits) {
    float s;
    memcpy(&s, rec, 4);
    const uint8_t *q = rec + 4;
    if (bits == 8) {
        const int8_t *q8 = (const int8_t *)q;
        for (int i = 0; i < hd; i++) out[i] = s * (float)q8[i];
    } else if (bits == 4) {
        for (int i = 0; i < hd; i += 2) {
            uint8_t b = q[i >> 1];
            out[i] = s * kvq_c4[b & 0xF];
            out[i + 1] = s * kvq_c4[b >> 4];
        }
    } else {
        for (int i = 0; i < hd; i += 4) {
            uint8_t b = q[i >> 2];
            out[i]     = s * kvq_c2[b & 3];
            out[i + 1] = s * kvq_c2[(b >> 2) & 3];
            out[i + 2] = s * kvq_c2[(b >> 4) & 3];
            out[i + 3] = s * kvq_c2[b >> 6];
        }
    }
}

// =========================================================================
// Tokenizer — SentencePiece (SPM) or GPT-2 BPE, greedy longest-prefix with a hash map
// =========================================================================

// supported chat templates
enum { TMPL_GEMMA, TMPL_CHATML, TMPL_MISTRAL, TMPL_LLAMA3, TMPL_GEMMA4 };
static const char *tmpl_names[] = { "gemma", "chatml", "mistral", "llama3", "gemma4" };

typedef struct {
    char **tokens;
    int n_tokens;
    uint32_t *map;       // open addressing: map[i] = token_id + 1 (0 = empty)
    size_t map_cap;      // power of 2
    int bos_id;
    int stop_ids[8];
    int n_stops;
    int byte_tok[256];   // SPM: <0xXX> → id (or -1)
    int is_bpe;          // 1 = GPT-2 BPE (Qwen), 0 = SentencePiece (Gemma/Mistral)
    int add_bos;
    int tmpl;            // TMPL_*
    const int *ttype;    // tokenizer.ggml.token_type (may be NULL)
    size_t ttype_n;
    uint16_t byte_cp[256];   // BPE: byte → codepoint (GPT-2 map)
    int16_t  cp_byte[512];   // BPE: codepoint → byte (or -1)
} tok_t;

// GPT-2 map: printable bytes stay as-is, the rest are shifted to 256+
static void bpe_byte_maps(tok_t *t) {
    int n = 0;
    for (int i = 0; i < 512; i++) t->cp_byte[i] = -1;
    for (int b = 0; b < 256; b++) {
        int keep = (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
        int cp = keep ? b : 256 + n++;
        t->byte_cp[b] = (uint16_t)cp;
        t->cp_byte[cp] = (int16_t)b;
    }
}

// decodes the first UTF-8 codepoint — returns the number of bytes consumed
static int utf8_decode(const char *s, unsigned *cp) {
    uint8_t c = (uint8_t)s[0];
    if (c < 0x80) { *cp = c; return 1; }
    if ((c >> 5) == 6 && (s[1] & 0xC0) == 0x80) {
        *cp = ((unsigned)(c & 0x1F) << 6) | (s[1] & 0x3F);
        return 2;
    }
    if ((c >> 4) == 14 && (s[1] & 0xC0) == 0x80 && (s[2] & 0xC0) == 0x80) {
        *cp = ((unsigned)(c & 0x0F) << 12) | ((unsigned)(s[1] & 0x3F) << 6) | (s[2] & 0x3F);
        return 3;
    }
    if ((c >> 3) == 30 && (s[1] & 0xC0) == 0x80 && (s[2] & 0xC0) == 0x80 && (s[3] & 0xC0) == 0x80) {
        *cp = ((unsigned)(c & 0x07) << 18) | ((unsigned)(s[1] & 0x3F) << 12)
            | ((unsigned)(s[2] & 0x3F) << 6) | (s[3] & 0x3F);
        return 4;
    }
    *cp = c; return 1;
}

static uint64_t fnv1a(const char *s, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) { h ^= (uint8_t)s[i]; h *= 1099511628211ULL; }
    return h;
}

static void tok_map_put(tok_t *t, int id) {
    const char *s = t->tokens[id];
    size_t n = strlen(s);
    uint64_t h = fnv1a(s, n);
    size_t i = h & (t->map_cap - 1);
    while (t->map[i]) i = (i + 1) & (t->map_cap - 1);
    t->map[i] = (uint32_t)id + 1;
}

static int tok_lookup(tok_t *t, const char *s, size_t n) {
    uint64_t h = fnv1a(s, n);
    size_t i = h & (t->map_cap - 1);
    while (t->map[i]) {
        int id = (int)t->map[i] - 1;
        const char *ts = t->tokens[id];
        if (strlen(ts) == n && memcmp(ts, s, n) == 0) return id;
        i = (i + 1) & (t->map_cap - 1);
    }
    return -1;
}

#define TOKEN_TYPE_CONTROL 3

static int tok_is_control(const tok_t *t, int id) {
    return t->ttype && id >= 0 && id < (int)t->ttype_n && t->ttype[id] == TOKEN_TYPE_CONTROL;
}

static void add_stop(tok_t *t, int id) {
    if (id < 0) return;
    // a token found by text lookup is accepted only if it is CONTROL — otherwise it may be ordinary text in the vocab
    if (t->ttype && id < (int)t->ttype_n && t->ttype[id] != TOKEN_TYPE_CONTROL) return;
    for (int i = 0; i < t->n_stops; i++)
        if (t->stop_ids[i] == id) return;
    if (t->n_stops < (int)(sizeof(t->stop_ids) / sizeof(t->stop_ids[0])))
        t->stop_ids[t->n_stops++] = id;
}

static int tok_load(tok_t *t, gguf_t *g) {
    memset(t, 0, sizeof(*t));
    kv_t *kv = kv_find(g, "tokenizer.ggml.tokens");
    if (!kv || kv->kind != KV_STR_ARR) {
        fprintf(stderr, "tokenizer.ggml.tokens not found\n");
        return -1;
    }
    t->tokens = kv->sarr;
    t->n_tokens = (int)kv->sarr_n;

    // tokenizer type: "gpt2" = BPE (Qwen), "llama" = SentencePiece (Gemma/Mistral)
    kv_t *tm = kv_find(g, "tokenizer.ggml.model");
    t->is_bpe = tm && tm->kind == KV_STR && strcmp(tm->str, "gpt2") == 0;
    if (t->is_bpe) bpe_byte_maps(t);

    kv_t *tt = kv_find(g, "tokenizer.ggml.token_type");
    if (tt && tt->kind == KV_INT_ARR) { t->ttype = tt->iarr; t->ttype_n = tt->iarr_n; }

    t->map_cap = 1;
    while (t->map_cap < (size_t)t->n_tokens * 2) t->map_cap <<= 1;
    t->map = (uint32_t *)calloc(t->map_cap, sizeof(uint32_t));
    for (int i = 0; i < t->n_tokens; i++) tok_map_put(t, i);

    t->bos_id = (int)kv_num(g, "tokenizer.ggml.bos_token_id", -1);
    if (t->bos_id < 0) t->bos_id = tok_lookup(t, "<bos>", 5);
    t->add_bos = (int)kv_num(g, "tokenizer.ggml.add_bos_token", t->is_bpe ? 0 : 1);
    if (t->bos_id < 0) t->add_bos = 0;

    int eos = (int)kv_num(g, "tokenizer.ggml.eos_token_id", -1);
    if (eos >= 0) t->stop_ids[t->n_stops++] = eos; // from metadata: always accepted
    add_stop(t, tok_lookup(t, "<eos>", 5));
    add_stop(t, tok_lookup(t, "<end_of_turn>", 13));
    add_stop(t, tok_lookup(t, "<turn|>", 7));        // Gemma 4
    add_stop(t, tok_lookup(t, "<|im_end|>", 10));
    add_stop(t, tok_lookup(t, "<|eot_id|>", 10));
    add_stop(t, tok_lookup(t, "<|endoftext|>", 13));
    add_stop(t, tok_lookup(t, "</s>", 4));

    for (int i = 0; i < 256; i++) t->byte_tok[i] = -1;
    if (!t->is_bpe) {
        for (int i = 0; i < t->n_tokens; i++) {
            const char *s = t->tokens[i];
            unsigned int b;
            if (strlen(s) == 6 && sscanf(s, "<0x%02X>", &b) == 1) t->byte_tok[b & 0xFF] = i;
        }
    }

    // choose the chat template: from tokenizer.chat_template first, then from vocab tokens
    kv_t *ct = kv_find(g, "tokenizer.chat_template");
    const char *cts = (ct && ct->kind == KV_STR) ? ct->str : NULL;
    if (cts && strstr(cts, "<|turn>"))                  t->tmpl = TMPL_GEMMA4;
    else if (cts && strstr(cts, "<start_of_turn>"))     t->tmpl = TMPL_GEMMA;
    else if (cts && strstr(cts, "<|im_start|>"))        t->tmpl = TMPL_CHATML;
    else if (cts && strstr(cts, "<|start_header_id|>")) t->tmpl = TMPL_LLAMA3;
    else if (cts && strstr(cts, "[INST]"))              t->tmpl = TMPL_MISTRAL;
    else if (tok_lookup(t, "<|turn>", 7) >= 0)                   t->tmpl = TMPL_GEMMA4;
    else if (tok_lookup(t, "<start_of_turn>", 15) >= 0)          t->tmpl = TMPL_GEMMA;
    else if (tok_lookup(t, "<|im_start|>", 12) >= 0)             t->tmpl = TMPL_CHATML;
    else if (tok_lookup(t, "<|start_header_id|>", 19) >= 0)      t->tmpl = TMPL_LLAMA3;
    else if (tok_lookup(t, "[INST]", 6) >= 0)                    t->tmpl = TMPL_MISTRAL;
    else {
        kv_t *ak = kv_find(g, "general.architecture");
        const char *arch = (ak && ak->kind == KV_STR) ? ak->str : "";
        if (strncmp(arch, "gemma", 5) == 0)      t->tmpl = TMPL_GEMMA;
        else if (strncmp(arch, "qwen", 4) == 0)  t->tmpl = TMPL_CHATML;
        else                                     t->tmpl = TMPL_MISTRAL;
    }
    // the Llama 3 template always starts with <|begin_of_text|> — many files do not set add_bos_token
    // so the default (0 for BPE) would be wrong and the model would drift off context
    if (t->tmpl == TMPL_LLAMA3 && t->bos_id >= 0 && !kv_find(g, "tokenizer.ggml.add_bos_token"))
        t->add_bos = 1;

    printf("[Tokenizer] %d tokens (%s), template=%s, bos=%d add_bos=%d, stops=",
           t->n_tokens, t->is_bpe ? "BPE" : "SPM", tmpl_names[t->tmpl], t->bos_id, t->add_bos);
    for (int i = 0; i < t->n_stops; i++) printf("%d ", t->stop_ids[i]);
    printf("\n");
    return 0;
}

static int tok_is_stop(tok_t *t, int id) {
    for (int i = 0; i < t->n_stops; i++)
        if (t->stop_ids[i] == id) return 1;
    return 0;
}

typedef struct { char role[16]; char *content; } msg_t;

// Encode — SPM: replace space with U+2581 | BPE: GPT-2 byte→codepoint map
// then greedy longest match on bytes in both cases
static int *tok_encode(tok_t *t, const char *text, int add_bos, int *out_n) {
    size_t src_n = strlen(text);
    char *buf;
    size_t bn = 0;
    if (t->is_bpe) {
        buf = (char *)malloc(src_n * 2 + 1);
        for (size_t i = 0; i < src_n; i++) {
            unsigned cp = t->byte_cp[(uint8_t)text[i]];
            if (cp < 0x80) buf[bn++] = (char)cp;
            else {
                buf[bn++] = (char)(0xC0 | (cp >> 6));
                buf[bn++] = (char)(0x80 | (cp & 0x3F));
            }
        }
    } else {
        buf = (char *)malloc(src_n * 3 + 1);
        for (size_t i = 0; i < src_n; i++) {
            if (text[i] == ' ') { buf[bn++] = (char)0xE2; buf[bn++] = (char)0x96; buf[bn++] = (char)0x81; }
            else buf[bn++] = text[i];
        }
    }
    buf[bn] = 0;

    int cap = (int)bn + 8;
    int *ids = (int *)malloc(cap * sizeof(int));
    int n = 0;
    if (add_bos && t->bos_id >= 0) ids[n++] = t->bos_id;

    size_t i = 0;
    while (i < bn) {
        size_t maxL = bn - i;
        if (maxL > 64) maxL = 64;
        int match = -1;
        size_t mlen = 0;
        for (size_t L = maxL; L >= 1; L--) {
            int id = tok_lookup(t, buf + i, L);
            // control tokens (<bos>, <end_of_turn>, <|im_end|>...) never come from text — the template
            // adds them by id, otherwise the user could break the template by typing them
            if (id >= 0 && !tok_is_control(t, id)) { match = id; mlen = L; break; }
        }
        if (match >= 0) {
            ids[n++] = match;
            i += mlen;
        } else if (t->is_bpe) {
            i += ((uint8_t)buf[i] >= 0xC0) ? 2 : 1; // skip the mapped character (usually every byte has a token)
        } else {
            int bid = t->byte_tok[(uint8_t)buf[i]];
            if (bid >= 0) ids[n++] = bid;
            i++;
        }
        if (n + 2 > cap) { cap *= 2; ids = (int *)realloc(ids, cap * sizeof(int)); }
    }
    free(buf);
    *out_n = n;
    return ids;
}

// --- building the chat sequence: role tokens are added by id, not as text ---
// encoding "<end_of_turn>" as text fails after punctuation because longest-match grabs ".<" as an
// ordinary token, so the model never sees the end of turn; control tokens are excluded from text matching (tok_encode)

typedef struct { int *ids; int n, cap; } idbuf_t;

static void ids_push(idbuf_t *b, int id) {
    if (b->n == b->cap) { b->cap = b->cap * 2 + 16; b->ids = (int *)realloc(b->ids, (size_t)b->cap * sizeof(int)); }
    b->ids[b->n++] = id;
}

// plain text = a then b (b may be NULL)
static void ids_push_text(tok_t *t, idbuf_t *b, const char *a, const char *c) {
    size_t na = strlen(a), nc = c ? strlen(c) : 0;
    if (na + nc == 0) return;
    char *s = (char *)malloc(na + nc + 1);
    memcpy(s, a, na);
    if (nc) memcpy(s + na, c, nc);
    s[na + nc] = 0;
    int k = 0;
    int *v = tok_encode(t, s, 0, &k);
    for (int i = 0; i < k; i++) ids_push(b, v[i]);
    free(v);
    free(s);
}

// special token by name — by id if present in the vocab, otherwise as text (old vocabs without it)
static void ids_push_special(tok_t *t, idbuf_t *b, const char *name) {
    int id = tok_lookup(t, name, strlen(name));
    if (id >= 0) ids_push(b, id);
    else ids_push_text(t, b, name, NULL);
}

// encodes the conversation with the model's template and ends by opening the assistant turn.
// close_prev: closes the model's previous reply first (console mode: everything before is in the cache,
// and the stop token that ended the reply was never passed through forward)
static int *encode_chat(tok_t *t, msg_t *msgs, int n_msgs, int add_bos, int close_prev, int *out_n) {
    idbuf_t b = { NULL, 0, 0 };
    if (add_bos && t->bos_id >= 0) ids_push(&b, t->bos_id);

    switch (t->tmpl) {
    case TMPL_CHATML:
        if (close_prev) { ids_push_special(t, &b, "<|im_end|>"); ids_push_text(t, &b, "\n", NULL); }
        for (int i = 0; i < n_msgs; i++) {
            char hdr[24];
            snprintf(hdr, sizeof(hdr), "%.15s\n", msgs[i].role);
            ids_push_special(t, &b, "<|im_start|>");
            ids_push_text(t, &b, hdr, msgs[i].content);
            ids_push_special(t, &b, "<|im_end|>");
            ids_push_text(t, &b, "\n", NULL);
        }
        ids_push_special(t, &b, "<|im_start|>");
        ids_push_text(t, &b, "assistant\n", NULL);
        break;
    case TMPL_LLAMA3:
        if (close_prev) ids_push_special(t, &b, "<|eot_id|>");
        for (int i = 0; i < n_msgs; i++) {
            ids_push_special(t, &b, "<|start_header_id|>");
            ids_push_text(t, &b, msgs[i].role, NULL);
            ids_push_special(t, &b, "<|end_header_id|>");
            ids_push_text(t, &b, "\n\n", msgs[i].content);
            ids_push_special(t, &b, "<|eot_id|>");
        }
        ids_push_special(t, &b, "<|start_header_id|>");
        ids_push_text(t, &b, "assistant", NULL);
        ids_push_special(t, &b, "<|end_header_id|>");
        ids_push_text(t, &b, "\n\n", NULL);
        break;
    case TMPL_MISTRAL: {
        if (close_prev) ids_push_special(t, &b, "</s>");
        // system is merged into the first user message that follows
        const char *sys = NULL;
        for (int i = 0; i < n_msgs; i++) {
            if (strcmp(msgs[i].role, "system") == 0) { sys = msgs[i].content; continue; }
            if (strcmp(msgs[i].role, "assistant") == 0) {
                ids_push_text(t, &b, " ", msgs[i].content);
                ids_push_special(t, &b, "</s>");
            } else {
                ids_push_special(t, &b, "[INST]");
                if (sys) {
                    ids_push_text(t, &b, " ", sys);
                    ids_push_text(t, &b, "\n\n", msgs[i].content);
                    sys = NULL;
                } else ids_push_text(t, &b, " ", msgs[i].content);
                ids_push_text(t, &b, " ", NULL);
                ids_push_special(t, &b, "[/INST]");
            }
        }
        break;
    }
    case TMPL_GEMMA4: // <|turn>role\n...<turn|>\n — with a separate system role, the reply ends with <turn|>
        if (close_prev) { ids_push_special(t, &b, "<turn|>"); ids_push_text(t, &b, "\n", NULL); }
        for (int i = 0; i < n_msgs; i++) {
            const char *role = strcmp(msgs[i].role, "assistant") == 0 ? "model\n"
                             : strcmp(msgs[i].role, "system") == 0    ? "system\n" : "user\n";
            ids_push_special(t, &b, "<|turn>");
            ids_push_text(t, &b, role, msgs[i].content);
            ids_push_special(t, &b, "<turn|>");
            ids_push_text(t, &b, "\n", NULL);
        }
        ids_push_special(t, &b, "<|turn>");
        ids_push_text(t, &b, "model\n", NULL);
        break;
    default: // TMPL_GEMMA — no separate system role: treat it as a user turn
        if (close_prev) { ids_push_special(t, &b, "<end_of_turn>"); ids_push_text(t, &b, "\n", NULL); }
        for (int i = 0; i < n_msgs; i++) {
            const char *role = strcmp(msgs[i].role, "assistant") == 0 ? "model\n" : "user\n";
            ids_push_special(t, &b, "<start_of_turn>");
            ids_push_text(t, &b, role, msgs[i].content);
            ids_push_special(t, &b, "<end_of_turn>");
            ids_push_text(t, &b, "\n", NULL);
        }
        ids_push_special(t, &b, "<start_of_turn>");
        ids_push_text(t, &b, "model\n", NULL);
        break;
    }
    *out_n = b.n;
    return b.ids;
}

// next user turn in console mode; the first turn (pos == 0) starts with BOS, later ones close the previous reply
static int *encode_turn(tok_t *t, const char *msg, int pos, int *out_n) {
    msg_t one;
    snprintf(one.role, sizeof(one.role), "user");
    one.content = (char *)msg;
    return encode_chat(t, &one, 1, pos == 0 && t->add_bos, pos > 0, out_n);
}

// is this token a non-printing control token? (token_type=3 CONTROL or special-token shape)
static int tok_is_special(tok_t *t, int id) {
    if (tok_is_control(t, id)) return 1;
    const char *w = t->tokens[id];
    size_t n = strlen(w);
    if (t->is_bpe)
        return n >= 4 && w[0] == '<' && w[1] == '|' && w[n - 2] == '|' && w[n - 1] == '>';
    return n >= 2 && w[0] == '<' && w[n - 1] == '>';
}

// decode a token into a text buffer
static int tok_text(tok_t *t, int id, char *buf, int cap) {
    if (id < 0 || id >= t->n_tokens || cap < 8) return 0;
    const char *w = t->tokens[id];
    size_t n = strlen(w);
    unsigned int b;
    if (!t->is_bpe && n == 6 && sscanf(w, "<0x%02X>", &b) == 1) {
        buf[0] = (char)(b & 0xFF);
        return 1;
    }
    if (tok_is_special(t, id)) return 0;
    int o = 0;
    if (t->is_bpe) {
        // every mapped codepoint goes back to its original byte
        for (size_t i = 0; i < n && o < cap - 4; ) {
            unsigned cp;
            int l = utf8_decode(w + i, &cp);
            if (cp < 512 && t->cp_byte[cp] >= 0) buf[o++] = (char)t->cp_byte[cp];
            else for (int k = 0; k < l && o < cap - 4; k++) buf[o++] = w[i + k];
            i += l;
        }
        return o;
    }
    for (size_t i = 0; i < n && o < cap - 4; i++) {
        // U+2581 (E2 96 81) → space
        if (i + 2 < n && (uint8_t)w[i] == 0xE2 && (uint8_t)w[i+1] == 0x96 && (uint8_t)w[i+2] == 0x81) {
            buf[o++] = ' ';
            i += 2;
        } else buf[o++] = w[i];
    }
    return o;
}

// <0xXX> bytes (and BPE tokens) may split one character (e.g. Arabic) across several tokens, and streaming
// output fflushes after every token — so we hold back an incomplete sequence until it completes, to avoid garbage on the console
static uint8_t g_u8_pend[4];
static int g_u8_have, g_u8_need;

static void out_flush_pending(void) {
    if (g_u8_have) fwrite(g_u8_pend, 1, (size_t)g_u8_have, stdout);
    g_u8_have = g_u8_need = 0;
}

static void out_byte(uint8_t c) {
    if (g_u8_need) {
        if ((c & 0xC0) == 0x80) {
            g_u8_pend[g_u8_have++] = c;
            if (g_u8_have == g_u8_need) out_flush_pending();
            return;
        }
        out_flush_pending(); // broken sequence: print it as-is and continue
    }
    int need = (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : (c & 0xF8) == 0xF0 ? 4 : 0;
    if (need) { g_u8_pend[0] = c; g_u8_have = 1; g_u8_need = need; }
    else putchar(c);
}

// decode one token for streaming output
static void tok_print(tok_t *t, int id) {
    char buf[512];
    int n = tok_text(t, id, buf, sizeof(buf));
    for (int i = 0; i < n; i++) out_byte((uint8_t)buf[i]);
}

// =========================================================================
// Next-token selection (sampling): repeat penalty → temperature → top-k → top-p → draw
// temp <= 0 means greedy (argmax) — deterministic, useful for tests and comparison with llama.cpp
// =========================================================================

typedef struct { float v; int id; } cand_t;

typedef struct {
    float temp, top_p, repeat_penalty;
    int top_k, repeat_last_n;  // top_k <= 0 = unlimited
    uint64_t rng;              // RNG state (the seed)
    cand_t *cand;              // workspace: n_vocab candidates
} sampler_t;

// xorshift64* → [0,1) with 24-bit precision
static float rng_uniform(uint64_t *s) {
    return (float)(rng_next(s) >> 40) / 16777216.0f;
}

static int cand_cmp_desc(const void *a, const void *b) {
    float x = ((const cand_t *)a)->v, y = ((const cand_t *)b)->v;
    return (x < y) - (x > y);
}

// fills out with the top k values sorted descending and returns their count
static int top_k_select(const float *logits, int n, int k, cand_t *out) {
    if (k <= 0 || k >= n) {
        for (int i = 0; i < n; i++) { out[i].v = logits[i]; out[i].id = i; }
        qsort(out, (size_t)n, sizeof(cand_t), cand_cmp_desc);
        return n;
    }
    // insertion into a small sorted array — most of the vocab is rejected with one comparison against the smallest
    int m = 0;
    for (int i = 0; i < n; i++) {
        float v = logits[i];
        if (m == k && !(v > out[k - 1].v)) continue;
        int j = m < k ? m++ : k - 1;
        while (j > 0 && out[j - 1].v < v) { out[j] = out[j - 1]; j--; }
        out[j].v = v; out[j].id = i;
    }
    return m;
}

// hist = conversation tokens so far (for the repeat penalty). Modifies logits in place
static int sample_next(sampler_t *s, tok_t *tk, float *logits, int n_vocab,
                       const int *hist, int n_hist) {
    if (s->repeat_penalty > 0.0f && s->repeat_penalty != 1.0f && s->repeat_last_n > 0) {
        int start = n_hist > s->repeat_last_n ? n_hist - s->repeat_last_n : 0;
        for (int i = start; i < n_hist; i++) {
            int id = hist[i], seen = 0;
            // stop tokens (<end_of_turn> in the template) are not penalized, otherwise the model hesitates to end its reply
            if (id < 0 || id >= n_vocab || tok_is_stop(tk, id)) continue;
            for (int j = start; j < i; j++) if (hist[j] == id) { seen = 1; break; }
            if (seen) continue;
            logits[id] = logits[id] > 0.0f ? logits[id] / s->repeat_penalty
                                           : logits[id] * s->repeat_penalty;
        }
    }
    if (s->temp <= 0.0f) return argmax_f(logits, n_vocab);

    cand_t *c = s->cand;
    int n;
    if (s->top_k > 0) n = top_k_select(logits, n_vocab, s->top_k, c);
    else {
        // without top-k: ignore anything with probability < e^-20 of the top (total < 0.06% even with 262k tokens)
        // instead of sorting the whole vocab for every token
        float cut = logits[argmax_f(logits, n_vocab)] - 20.0f * s->temp;
        n = 0;
        for (int i = 0; i < n_vocab; i++)
            if (logits[i] > cut) { c[n].v = logits[i]; c[n].id = i; n++; }
        qsort(c, (size_t)n, sizeof(cand_t), cand_cmp_desc);
    }
    float mx = c[0].v, sum = 0.0f;
    for (int i = 0; i < n; i++) { c[i].v = expf((c[i].v - mx) / s->temp); sum += c[i].v; }

    // top-p: smallest set of most-probable tokens whose sum reaches top_p — at least one
    int keep = n;
    if (s->top_p > 0.0f && s->top_p < 1.0f) {
        float cum = 0.0f, lim = s->top_p * sum;
        for (int i = 0; i < n; i++) {
            cum += c[i].v;
            if (cum >= lim) { keep = i + 1; break; }
        }
    }
    float tot = 0.0f;
    for (int i = 0; i < keep; i++) tot += c[i].v;
    float r = rng_uniform(&s->rng) * tot;
    for (int i = 0; i < keep; i++) {
        r -= c[i].v;
        if (r < 0.0f) return c[i].id;
    }
    return c[keep - 1].id;
}

// =========================================================================
// The model: config + weights + preallocated buffers
// =========================================================================

#define MAX_LAYERS 128

typedef struct {
    float *attn_norm;
    tensor_t *wq, *wk, *wv, *wo;
    float *bq, *bk, *bv;           // optional biases (Qwen2)
    float *q_norm, *k_norm;        // size = head dim (Gemma/Qwen3, may be NULL)
    float *attn_post_norm;         // optional (Gemma only)
    float *ffn_norm;
    tensor_t *ffn_gate, *ffn_up, *ffn_down;
    // MoE (qwen3moe): router + experts stacked 3D [cols, rows, n_expert] — pointers into the mmap
    tensor_t *ffn_gate_inp;        // NULL = dense FFN layer
    tensor_t *ffn_gate_exps, *ffn_up_exps, *ffn_down_exps;
    float *ffn_post_norm;
    tensor_t *ple_inp_gate, *ple_proj;
    float *ple_post_norm;
    float out_scale;
    int head_dim, n_head, n_head_kv, kv_dim, q_dim, ff_dim;
} layer_t;

typedef struct {
    gguf_t *g;
    // config
    char arch_name[32];
    int is_gemma;        // Gemma behaviour: embedding scaling, mandatory norms, GELU
    // attention differences: Gemma 4 = V norm and scale 1.0; Gemma 3 no V norm, scale 1/√qpas
    // and RoPE scaling on global layers; others no V norm and scale 1/√head_dim (attn_scale = 0)
    int v_norm;
    float attn_scale;
    double rope_scale;   // freq_scale for global layers only
    int act_gelu;        // 1 = GELU (Gemma), 0 = SiLU (Qwen/Mistral/Llama)
    int rope_neox;       // 1 = halves (Gemma/Qwen), 0 = pairs (Llama/Mistral)
    float embd_scale;
    int n_layer, n_embd, n_ple, n_swa, n_kv_from_start;
    int n_rot, n_rot_swa;
    double rope_base, rope_base_swa;
    float norm_eps, final_softcap;
    int is_swa[MAX_LAYERS];
    int kv_donor[MAX_LAYERS];
    int n_ctx, n_vocab;
    int n_expert, n_expert_used, ff_exp;   // MoE — 0 = dense model

    tensor_t *tok_embd, *output;
    float *output_norm;
    tensor_t *ple_tok_embd, *ple_model_proj;
    float *ple_proj_norm;
    float *rope_freqs;          // may be NULL
    int rope_freqs_n;

    layer_t layers[MAX_LAYERS];

    // KV cache: per owning layer — ring buffer for SWA to save memory
    float *cache_k[MAX_LAYERS], *cache_v[MAX_LAYERS];
    int cache_cap[MAX_LAYERS]; // number of stored positions

    // TurboQuant (FASTGGUF_KV=turbo): quantized cache instead of f32
    int kvq, kvq_kbits, kvq_vbits;  // kvq==2: debug mode — computes both paths and prints the error
    uint8_t *cache_kq[MAX_LAYERS], *cache_vq[MAX_LAYERS];
    int kq_stride[MAX_LAYERS], vq_stride[MAX_LAYERS]; // bytes per position (all heads)
    float *dbg_q, *dbg_o, *dbg_scores; // debug mode only

    // track the tokens resident in the cache (to reuse the prefix between requests)
    int *seq_ids;  // tokens at positions 0..seq_len-1
    int seq_len;

    // buffers allocated once (zero allocations inside forward)
    float *x, *cur, *q, *k, *v, *attn_out, *ao, *f;
    float *gate, *up, *ffn_out, *scores;
    float *ple_in, *ple_t, *ple_t2, *logits;
    float *moe_probs, *moe_w, *moe_h;  // router probabilities, weights of the selected experts, SiLU(gate)·up per expert
    int *moe_sel;                      // indices of the selected experts (n_expert_used)
    q8_blk_t *moe_xq;                  // down input quantized for every selected expert
    q8k_blk_t *moe_xk;                 // the same, as 256-element blocks (K-quant experts)
    cand_t *cand;   // sampling workspace (n_vocab)
} model_t;

// loads a whole tensor into a new float array
static float *load_f32(gguf_t *g, const char *name, int *out_n) {
    tensor_t *t = tensor_find(g, name);
    if (!t) return NULL;
    uint64_t n = 1;
    for (int i = 0; i < t->n_dims; i++) n *= t->ne[i];
    float *out = (float *)malloc(n * sizeof(float));
    if (dequantize_row(t->type, t->data, out, (int)n) != 0) { free(out); return NULL; }
    if (out_n) *out_n = (int)n;
    return out;
}

static float *must_f32(gguf_t *g, const char *name, int *out_n, int *err) {
    float *p = load_f32(g, name, out_n);
    if (!p) { fprintf(stderr, "missing tensor: %s\n", name); *err = 1; }
    return p;
}

static tensor_t *must_tensor(gguf_t *g, const char *name, int *err) {
    tensor_t *t = tensor_find(g, name);
    if (!t) { fprintf(stderr, "missing tensor: %s\n", name); *err = 1; }
    return t;
}

static int64_t tensor_rows(tensor_t *t) {
    int64_t r = 1;
    for (int i = 1; i < t->n_dims; i++) r *= (int64_t)t->ne[i];
    return r;
}

// stacked expert tensor [cols, rows, n]: one 2D matrix per expert, contiguous in the file
static int exps_shape_ok(const tensor_t *t, int cols, int rows, int n) {
    return t->n_dims == 3 && t->ne[0] == (uint64_t)cols && t->ne[1] == (uint64_t)rows &&
           t->ne[2] == (uint64_t)n;
}

static int model_load(model_t *m, gguf_t *g, int n_ctx) {
    memset(m, 0, sizeof(*m));
    m->g = g;
    m->n_ctx = n_ctx;

    kv_t *arch_kv = kv_find(g, "general.architecture");
    if (!arch_kv || arch_kv->kind != KV_STR) { fprintf(stderr, "no architecture\n"); return -1; }
    const char *arch = arch_kv->str;
    char key[160];
#define K(fmt) (snprintf(key, sizeof(key), "%s." fmt, arch), key)

    // per-architecture settings — gemma* / qwen2 / qwen3 / qwen3moe / llama (includes Mistral)
    snprintf(m->arch_name, sizeof(m->arch_name), "%s", arch);
    m->is_gemma = strncmp(arch, "gemma", 5) == 0;
    if (!m->is_gemma && strncmp(arch, "qwen", 4) != 0 &&
        strcmp(arch, "llama") != 0 && strcmp(arch, "mistral") != 0)
        fprintf(stderr, "[warn] arch '%s' is untested — continuing with the standard llama path\n", arch);
    m->act_gelu = m->is_gemma;
    m->rope_neox = !(strcmp(arch, "llama") == 0 || strcmp(arch, "mistral") == 0);

    // Gemma 3 — the differences as implemented by llama.cpp (LLM_ARCH_GEMMA3); the Gemma 4 path is unchanged
    int is_g3 = strcmp(arch, "gemma3") == 0;
    m->v_norm = m->is_gemma && !is_g3;
    m->attn_scale = m->is_gemma && !is_g3 ? 1.0f : 0.0f; // Gemma 3: computed once the head dim is known
    m->rope_scale = 1.0;
    if (is_g3) {
        // linear scaling (4B/12B/27B: factor 8) applies to global layers only
        kv_t *st = kv_find(g, K("rope.scaling.type"));
        double f = kv_num(g, K("rope.scaling.factor"), 0.0);
        if (f > 0.0 && (!st || (st->kind == KV_STR && strcmp(st->str, "linear") == 0)))
            m->rope_scale = 1.0 / f;
    }

    m->n_layer = (int)kv_num(g, K("block_count"), 0);
    m->n_embd  = (int)kv_num(g, K("embedding_length"), 0);
    if (m->n_layer <= 0 || m->n_embd <= 0 || m->n_layer > MAX_LAYERS) {
        fprintf(stderr, "bad config: layers=%d embd=%d\n", m->n_layer, m->n_embd);
        return -1;
    }
    m->embd_scale = m->is_gemma ? sqrtf((float)m->n_embd) : 1.0f;
    m->n_ple = (int)kv_num(g, K("embedding_length_per_layer_input"), 0);
    m->n_swa = (int)kv_num(g, K("attention.sliding_window"), 0);
    int shared_kv = (int)kv_num(g, K("attention.shared_kv_layers"), 0);
    m->n_kv_from_start = m->n_layer - shared_kv;
    m->rope_base     = kv_num(g, K("rope.freq_base"), m->is_gemma ? 1e6 : 10000.0);
    m->rope_base_swa = kv_num(g, K("rope.freq_base_swa"), 10000.0);
    m->n_rot     = (int)kv_num(g, K("rope.dimension_count"), 0);
    m->n_rot_swa = (int)kv_num(g, K("rope.dimension_count_swa"), 0);
    m->norm_eps  = (float)kv_num(g, K("attention.layer_norm_rms_epsilon"), 1e-6);
    m->final_softcap = (float)kv_num(g, K("final_logit_softcapping"), 0.0);

    // MoE (qwen3moe): number of experts, experts used per token, and FFN width of one expert
    m->n_expert      = (int)kv_num(g, K("expert_count"), 0);
    m->n_expert_used = (int)kv_num(g, K("expert_used_count"), 0);
    m->ff_exp        = (int)kv_num(g, K("expert_feed_forward_length"), 0);
    if (m->n_expert > 0 && (m->n_expert_used <= 0 || m->n_expert_used > m->n_expert)) {
        fprintf(stderr, "bad MoE config: experts=%d used=%d\n", m->n_expert, m->n_expert_used);
        return -1;
    }

    // SWA pattern
    snprintf(key, sizeof(key), "%s.attention.sliding_window_pattern", arch);
    kv_t *pat = kv_find(g, key);
    if (pat && pat->kind == KV_INT_ARR && (int)pat->iarr_n >= m->n_layer) {
        for (int i = 0; i < m->n_layer; i++) m->is_swa[i] = pat->iarr[i] != 0;
    } else if (pat && pat->kind == KV_NUM && pat->num > 0) {
        int p = (int)pat->num;
        for (int i = 0; i < m->n_layer; i++) m->is_swa[i] = ((i + 1) % p) != 0;
    } else if (is_g3 && m->n_swa > 0) {
        // Gemma 3 GGUF does not store the pattern: 5 SWA layers then one global layer
        for (int i = 0; i < m->n_layer; i++) m->is_swa[i] = ((i + 1) % 6) != 0;
    } // otherwise: all layers are global

    // donor map for the shared cache
    int last_swa = -1, last_full = -1;
    for (int i = 0; i < m->n_layer; i++) {
        if (i < m->n_kv_from_start) {
            if (m->is_swa[i]) last_swa = i; else last_full = i;
            m->kv_donor[i] = i;
        } else {
            m->kv_donor[i] = m->is_swa[i] ? last_swa : last_full;
            if (m->kv_donor[i] < 0) { fprintf(stderr, "layer %d: no KV donor\n", i); return -1; }
        }
    }

    int err = 0;
    m->tok_embd = must_tensor(g, "token_embd.weight", &err);
    if (err) return -1;
    m->n_vocab = (int)tensor_rows(m->tok_embd);
    m->output = tensor_find(g, "output.weight");
    if (!m->output) m->output = m->tok_embd;
    m->output_norm = must_f32(g, "output_norm.weight", NULL, &err);

    if (m->n_ple > 0) {
        m->ple_tok_embd   = must_tensor(g, "per_layer_token_embd.weight", &err);
        m->ple_model_proj = must_tensor(g, "per_layer_model_proj.weight", &err);
        m->ple_proj_norm  = must_f32(g, "per_layer_proj_norm.weight", NULL, &err);
    }
    m->rope_freqs = load_f32(g, "rope_freqs.weight", &m->rope_freqs_n);

    // head dim from metadata (fallback when there is no q_norm)
    int meta_n_head = (int)kv_num(g, K("attention.head_count"), 0);
    int meta_head_dim = (int)kv_num(g, K("attention.key_length"), 0);
    if (meta_head_dim <= 0 && meta_n_head > 0) meta_head_dim = m->n_embd / meta_n_head;

    int max_cols = m->n_embd, max_q = 0, max_kv = 0, max_ff = 0;

    for (int i = 0; i < m->n_layer && !err; i++) {
        layer_t *L = &m->layers[i];
        char nm[160];
#define BLK(fmt) (snprintf(nm, sizeof(nm), "blk.%d." fmt, i), nm)

        L->attn_norm = must_f32(g, BLK("attn_norm.weight"), NULL, &err);
        L->wq = must_tensor(g, BLK("attn_q.weight"), &err);
        L->wo = must_tensor(g, BLK("attn_output.weight"), &err);
        int qn = 0;
        L->q_norm = load_f32(g, BLK("attn_q_norm.weight"), &qn); // Gemma/Qwen3
        if (m->is_gemma && !L->q_norm) { fprintf(stderr, "missing tensor: %s\n", nm); return -1; }
        L->attn_post_norm = load_f32(g, BLK("post_attention_norm.weight"), NULL);
        if (m->is_gemma && !L->attn_post_norm) { fprintf(stderr, "missing tensor: %s\n", nm); return -1; }
        L->bq = load_f32(g, BLK("attn_q.bias"), NULL); // Qwen2
        if (err) break;

        // true head dim = length of q_norm if present — reliable even when metadata holds per-layer arrays
        L->head_dim = L->q_norm ? qn : meta_head_dim;
        if (L->head_dim <= 0) {
            fprintf(stderr, "L%d: cannot determine head_dim\n", i);
            return -1;
        }
        L->q_dim = (int)tensor_rows(L->wq);
        L->n_head = L->q_dim / L->head_dim;
        if (L->n_head * L->head_dim != L->q_dim) {
            fprintf(stderr, "L%d: q_dim %d %% head_dim %d != 0\n", i, L->q_dim, L->head_dim);
            return -1;
        }

        if (i < m->n_kv_from_start) {
            L->wk = must_tensor(g, BLK("attn_k.weight"), &err);
            L->wv = tensor_find(g, BLK("attn_v.weight")); // may be NULL in Gemma (V = K)
            if (!L->wv && !m->is_gemma) { fprintf(stderr, "missing tensor: %s\n", nm); return -1; }
            int kn = 0;
            L->k_norm = load_f32(g, BLK("attn_k_norm.weight"), &kn);
            if (m->is_gemma && !L->k_norm) { fprintf(stderr, "missing tensor: %s\n", nm); return -1; }
            L->bk = load_f32(g, BLK("attn_k.bias"), NULL);
            L->bv = load_f32(g, BLK("attn_v.bias"), NULL);
            if (err) break;
            if (L->k_norm && L->q_norm && kn != qn) {
                fprintf(stderr, "L%d: k_norm %d != q_norm %d\n", i, kn, qn);
                return -1;
            }
            L->kv_dim = (int)tensor_rows(L->wk);
            L->n_head_kv = L->kv_dim / L->head_dim;
            if (L->n_head_kv * L->head_dim != L->kv_dim) {
                fprintf(stderr, "L%d: kv_dim %d %% head_dim %d != 0\n", i, L->kv_dim, L->head_dim);
                return -1;
            }
        }

        L->ffn_norm = must_f32(g, BLK("ffn_norm.weight"), NULL, &err);
        L->ffn_gate_inp = m->n_expert > 0 ? tensor_find(g, BLK("ffn_gate_inp.weight")) : NULL;
        if (L->ffn_gate_inp) {
            // MoE: experts are not copied — they stay as pointers into the mmap so the OS reads from disk
            // only the pages of the experts chosen by the router (a model larger than RAM still runs)
            if (tensor_find(g, BLK("ffn_gate_inp_shexp.weight"))) {
                fprintf(stderr, "L%d: shared experts (qwen2moe) not supported\n", i);
                return -1;
            }
            L->ffn_gate_exps = must_tensor(g, BLK("ffn_gate_exps.weight"), &err);
            L->ffn_up_exps   = must_tensor(g, BLK("ffn_up_exps.weight"), &err);
            L->ffn_down_exps = must_tensor(g, BLK("ffn_down_exps.weight"), &err);
            if (err) break;
            if (m->ff_exp <= 0) m->ff_exp = (int)L->ffn_gate_exps->ne[1];
            tensor_t *gi = L->ffn_gate_inp;
            if (gi->n_dims != 2 || gi->ne[0] != (uint64_t)m->n_embd || gi->ne[1] != (uint64_t)m->n_expert ||
                !exps_shape_ok(L->ffn_gate_exps, m->n_embd, m->ff_exp, m->n_expert) ||
                !exps_shape_ok(L->ffn_up_exps,   m->n_embd, m->ff_exp, m->n_expert) ||
                !exps_shape_ok(L->ffn_down_exps, m->ff_exp, m->n_embd, m->n_expert)) {
                fprintf(stderr, "L%d: MoE tensor shapes do not match embd=%d ff_exp=%d experts=%d\n",
                        i, m->n_embd, m->ff_exp, m->n_expert);
                return -1;
            }
            L->ff_dim = m->ff_exp;
        } else {
            L->ffn_gate = must_tensor(g, BLK("ffn_gate.weight"), &err);
            L->ffn_up   = must_tensor(g, BLK("ffn_up.weight"), &err);
            L->ffn_down = must_tensor(g, BLK("ffn_down.weight"), &err);
            if (err) break;
            L->ff_dim = (int)tensor_rows(L->ffn_gate);
        }
        L->ffn_post_norm = load_f32(g, BLK("post_ffw_norm.weight"), NULL);
        if (m->is_gemma && !L->ffn_post_norm) { fprintf(stderr, "missing tensor: %s\n", nm); return -1; }
        if (err) break;

        if (m->n_ple > 0) {
            L->ple_inp_gate = must_tensor(g, BLK("inp_gate.weight"), &err);
            L->ple_proj     = must_tensor(g, BLK("proj.weight"), &err);
            L->ple_post_norm = must_f32(g, BLK("post_norm.weight"), NULL, &err);
        }

        L->out_scale = 1.0f;
        float *osc = load_f32(g, BLK("layer_output_scale.weight"), NULL);
        if (osc) { L->out_scale = osc[0]; free(osc); }

        if (L->q_dim > max_q) max_q = L->q_dim;
        if (L->kv_dim > max_kv) max_kv = L->kv_dim;
        if (L->ff_dim > max_ff) max_ff = L->ff_dim;
    }
    if (err) return -1;

    if (is_g3) {
        // query_pre_attn_scalar in gemma_pytorch: n_embd/n_head for 27B (62 layers), head dim for the others
        const layer_t *L0 = &m->layers[0];
        float qpas = m->n_layer == 62 ? (float)(m->n_embd / L0->n_head) : (float)L0->head_dim;
        m->attn_scale = 1.0f / sqrtf(qpas);
    }

    // largest column count of any matvec (to size the scratch)
    for (size_t i = 0; i < g->n_tensors; i++)
        if ((int)g->tensors[i].ne[0] > max_cols) max_cols = (int)g->tensors[i].ne[0];
    matvec_init_scratch(max_cols);

    // TurboQuant — requires head_dim to be a power of 2:
    //   FASTGGUF_KV=turbo   K4+V4 (~7.5x smaller) — Qwen automatically upgrades K to 8-bit
    //                       because its attention scores are huge (|logit| in the hundreds) and 4-bit breaks softmax
    //   FASTGGUF_KV=turbo4  force K4 even on Qwen
    //   FASTGGUF_KV=turbo8  force K8+V4 (~5x) — the safe choice for any sensitive model
    //   FASTGGUF_KV=turbo2  K4+V2 (~9.5x) — maximum compression, noticeable loss
    const char *kvenv = getenv("FASTGGUF_KV");
    m->kvq = kvenv && (strncmp(kvenv, "turbo", 5) == 0 || strcmp(kvenv, "on") == 0);
    if (kvenv && strcmp(kvenv, "debug") == 0) m->kvq = 2; // compare both paths
    m->kvq_kbits = (kvenv && strcmp(kvenv, "turbo8") == 0) ? 8 : 4;
    m->kvq_vbits = (kvenv && strcmp(kvenv, "turbo2") == 0) ? 2 : 4;
    if (m->kvq && m->kvq_kbits == 4 && strncmp(arch, "qwen", 4) == 0
        && !(kvenv && strcmp(kvenv, "turbo4") == 0)) {
        m->kvq_kbits = 8;
        printf("[kv] %s attention logits are huge - keys auto-upgraded to 8-bit"
               " (force 4-bit with FASTGGUF_KV=turbo4)\n", arch);
    }
    if (m->kvq) {
        for (int i = 0; i < m->n_kv_from_start; i++) {
            int hd = m->layers[i].head_dim;
            if (hd < 8 || hd > KVQ_MAX_HD || (hd & (hd - 1))) { m->kvq = 0; break; }
        }
        if (!m->kvq) printf("[kv] TurboQuant needs power-of-2 head_dim - using f32 cache\n");
        else kvq_init();
    }

    // allocate the KV cache — SWA only needs a window of n_swa positions (ring buffer)
    size_t kv_bytes = 0, kv_f32_bytes = 0;
    for (int i = 0; i < m->n_kv_from_start; i++) {
        layer_t *L = &m->layers[i];
        int cap = m->is_swa[i] && m->n_swa > 0 ? (m->n_swa < n_ctx ? m->n_swa : n_ctx) : n_ctx;
        m->cache_cap[i] = cap;
        kv_f32_bytes += 2ULL * cap * L->kv_dim * sizeof(float);
        if (m->kvq) {
            int hd = L->head_dim;
            m->kq_stride[i] = L->n_head_kv * KVQ_REC(hd, m->kvq_kbits);
            m->vq_stride[i] = L->n_head_kv * KVQ_REC(hd, m->kvq_vbits);
            m->cache_kq[i] = (uint8_t *)malloc((size_t)cap * m->kq_stride[i]);
            m->cache_vq[i] = (uint8_t *)malloc((size_t)cap * m->vq_stride[i]);
            kv_bytes += (size_t)cap * (m->kq_stride[i] + m->vq_stride[i]);
            if (m->kvq == 2) { // debug mode also needs the f32 reference cache
                m->cache_k[i] = (float *)malloc((size_t)cap * L->kv_dim * sizeof(float));
                m->cache_v[i] = (float *)malloc((size_t)cap * L->kv_dim * sizeof(float));
            }
        } else {
            m->cache_k[i] = (float *)malloc((size_t)cap * L->kv_dim * sizeof(float));
            m->cache_v[i] = (float *)malloc((size_t)cap * L->kv_dim * sizeof(float));
            kv_bytes += 2ULL * cap * L->kv_dim * sizeof(float);
        }
    }
    if (m->kvq)
        printf("[kv] TurboQuant cache: K %d-bit + V %d-bit, %.0f MB instead of %.0f MB\n",
               m->kvq_kbits, m->kvq_vbits, kv_bytes / 1048576.0, kv_f32_bytes / 1048576.0);
    m->seq_ids = (int *)malloc((size_t)n_ctx * sizeof(int));
    m->seq_len = 0;

    // buffers
    m->x        = (float *)malloc(m->n_embd * sizeof(float));
    m->cur      = (float *)malloc(m->n_embd * sizeof(float));
    m->q        = (float *)malloc(max_q * sizeof(float));
    m->k        = (float *)malloc((max_kv > 0 ? max_kv : 1) * sizeof(float));
    m->v        = (float *)malloc((max_kv > 0 ? max_kv : 1) * sizeof(float));
    m->attn_out = (float *)malloc(max_q * sizeof(float));
    m->ao       = (float *)malloc(m->n_embd * sizeof(float));
    m->f        = (float *)malloc(m->n_embd * sizeof(float));
    m->gate     = (float *)malloc(max_ff * sizeof(float));
    m->up       = (float *)malloc(max_ff * sizeof(float));
    m->ffn_out  = (float *)malloc(m->n_embd * sizeof(float));
    m->scores   = (float *)malloc(n_ctx * sizeof(float));
    if (m->kvq == 2) { // debug mode: unrotated copy of q + reference output + reference scores
        m->dbg_q      = (float *)malloc(max_q * sizeof(float));
        m->dbg_o      = (float *)malloc(max_q * sizeof(float));
        m->dbg_scores = (float *)malloc(n_ctx * sizeof(float));
    }
    m->logits   = (float *)malloc(m->n_vocab * sizeof(float));
    m->cand     = (cand_t *)malloc((size_t)m->n_vocab * sizeof(cand_t));
    if (m->n_expert > 0) {
        int k = m->n_expert_used;
        m->moe_probs = (float *)malloc((size_t)m->n_expert * sizeof(float));
        m->moe_w     = (float *)malloc((size_t)k * sizeof(float));
        m->moe_sel   = (int *)malloc((size_t)k * sizeof(int));
        m->moe_h     = (float *)malloc((size_t)k * (m->ff_exp > 0 ? m->ff_exp : 1) * sizeof(float));
        m->moe_xq    = (q8_blk_t *)malloc(((size_t)k * (m->ff_exp / 32) + 1) * sizeof(q8_blk_t));
        m->moe_xk    = (q8k_blk_t *)malloc(((size_t)k * (m->ff_exp / 256) + 1) * sizeof(q8k_blk_t));
    }
    if (m->n_ple > 0) {
        m->ple_in = (float *)malloc((size_t)m->n_ple * m->n_layer * sizeof(float));
        m->ple_t  = (float *)malloc(m->n_ple * sizeof(float));
        m->ple_t2 = (float *)malloc(m->n_embd * sizeof(float));
    }

    // GPU: upload the weights of the large matrices (everything that goes through matvec) to VRAM.
    // models larger than VRAM (e.g. 40B on a 16GB card) are split automatically:
    // the first layers that fit go on the GPU and the rest stay on the CPU.
    // FASTGGUF_GPU_LAYERS=N sets the layer count manually (N > layer count = include the output head)
    fg_gpu_init();
    if (g_gpu.state == 1) {
        uint64_t used0 = g_gpu.used;
        int n0 = g_gpu.n_offloaded, st0 = g_gpu.n_skip_type, sm0 = g_gpu.n_skip_mem;
        const char *lenv = getenv("FASTGGUF_GPU_LAYERS");
        int max_layers = lenv ? atoi(lenv) : -1;   // -1 = as many as memory allows
        int full_layers = 0, part_layers = 0;
        for (int i = 0; i < m->n_layer; i++) {
            if (max_layers >= 0 && i >= max_layers) break;
            layer_t *L = &m->layers[i];
            tensor_t *ts[9] = { L->wq, L->wk, L->wv, L->wo, L->ffn_gate, L->ffn_up,
                                L->ffn_down, L->ple_inp_gate, L->ple_proj };
            int on = 0, present = 0;
            for (int j = 0; j < 9; j++) {
                if (!ts[j]) continue;
                present++;
                gpu_upload_tensor(ts[j]);
                on += ts[j]->gpu != NULL;
            }
            if (on == present) full_layers++;
            else if (on > 0) part_layers++;
            if (on == 0) break;   // memory is full — no point trying the following layers
        }
        if (max_layers < 0 || max_layers > m->n_layer) {
            gpu_upload_tensor(m->ple_model_proj);
            gpu_upload_tensor(m->output);
        }
        printf("[gpu] offloaded %d/%d layers%s%s%s: %d tensors, %.0f MB on %s (budget %.0f MB)\n",
               full_layers, m->n_layer,
               part_layers ? " (+1 partial)" : "",
               m->output && m->output->gpu ? " + output head" : "",
               m->n_expert > 0 ? " (MoE experts stay on CPU)" : "",
               g_gpu.n_offloaded - n0, (g_gpu.used - used0) / 1048576.0,
               g_gpu.name, g_gpu.budget / 1048576.0);
        int st = g_gpu.n_skip_type - st0, sm = g_gpu.n_skip_mem - sm0;
        if (st + sm > 0)
            printf("[gpu] %d tensors stay on CPU (%d unsupported type, %d vram full)\n",
                   st + sm, st, sm);
    }

    int nswa = 0;
    for (int i = 0; i < m->n_layer; i++) nswa += m->is_swa[i];
    printf("[Config] %s: layers=%d (swa=%d, kv_own=%d), embd=%d, ple=%d, ctx=%d, kv_mem=%.0fMB\n",
           arch, m->n_layer, nswa, m->n_kv_from_start, m->n_embd, m->n_ple, n_ctx,
           kv_bytes / 1048576.0);
    printf("[Model] %d tensors, output head: %s (%s), vocab=%d\n",
           (int)g->n_tensors, m->output->name, type_name(m->output->type), m->n_vocab);
    if (m->n_expert > 0)
        printf("[MoE] experts=%d, used/token=%d, ff_exp=%d - experts read from mmap on demand\n",
               m->n_expert, m->n_expert_used, m->ff_exp);
    if (is_g3)
        printf("[Gemma3] attn_scale=%.4f v_norm=off rope_scale_global=%.4f swa_window=%d\n",
               m->attn_scale, m->rope_scale, m->n_swa);
    return 0;
}

// read an embedding row + multiply by sqrt(dim)
static int read_embedding(tensor_t *t, int row, float *out, int n, float scale) {
    uint64_t stride = row_stride(t->type, t->ne[0]);
    if (dequantize_row(t->type, t->data + (uint64_t)row * stride, out, n) != 0) return -1;
    for (int i = 0; i < n; i++) out[i] *= scale;
    return 0;
}

// FFN with experts (MoE) — matches llama.cpp build_moe_ffn for qwen3moe:
// router → softmax → top-k experts → renormalize their weights to sum to 1 (norm_topk_prob)
// → for each expert down(SiLU(gate·x) * up·x) → weighted sum.
// row r of expert e starts at (e·rows + r)·stride inside the 3D tensor — no copies, so the OS reads
// only the pages of the selected experts from disk. Parallel over all experts at once: only two regions per layer
static int moe_ffn(model_t *m, layer_t *L, const float *x, float *out) {
    int n_embd = m->n_embd, ff = m->ff_exp, k = m->n_expert_used;
    float *p = m->moe_probs, *w = m->moe_w, *h = m->moe_h;
    int *sel = m->moe_sel;

    if (matvec(L->ffn_gate_inp, x, p) != 0) return -1;
    softmax_inplace(p, m->n_expert);
    float wsum = 0.0f;
    for (int j = 0; j < k; j++) {
        int e = argmax_f(p, m->n_expert);
        sel[j] = e;
        w[j] = p[e];
        wsum += p[e];
        p[e] = -1.0f;   // softmax probabilities are ≥ 0 — it will not be picked again
    }
    if (wsum < 6.103515625e-5f) wsum = 6.103515625e-5f; // same limit as llama.cpp
    for (int j = 0; j < k; j++) w[j] /= wsum;

    // gate/up: same input for every expert — one quantization, and the rows of all selected experts in one loop
    tensor_t *tg = L->ffn_gate_exps, *tu = L->ffn_up_exps, *td = L->ffn_down_exps;
    uint64_t sg = row_stride(tg->type, (uint64_t)n_embd), su = row_stride(tu->type, (uint64_t)n_embd);
    int qg = type_uses_q8(tg->type, n_embd), qu = type_uses_q8(tu->type, n_embd);
    quantize_act(qg, x, g_xq8, g_xq8k, n_embd);
    if (qu != qg) quantize_act(qu, x, g_xq8, g_xq8k, n_embd);
    int fail = 0;
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < k * ff; r++) {
        uint64_t row = (uint64_t)sel[r / ff] * ff + r % ff;
        float gv = dot_row(tg->type, qg, tg->data + row * sg, g_xq8, g_xq8k, x, n_embd, &fail);
        float uv = dot_row(tu->type, qu, tu->data + row * su, g_xq8, g_xq8k, x, n_embd, &fail);
        h[r] = silu_f(gv) * uv;
    }

    // down: different input per expert — each output row sums k weighted experts
    uint64_t sd = row_stride(td->type, (uint64_t)ff);
    int qd = type_uses_q8(td->type, ff), nbd = ff / 32, nbk = ff / 256;
    for (int j = 0; j < k; j++)
        quantize_act(qd, h + j * ff, m->moe_xq + j * nbd, m->moe_xk + j * nbk, ff);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_embd; i++) {
        float s = 0.0f;
        for (int j = 0; j < k; j++) {
            uint64_t row = (uint64_t)sel[j] * n_embd + i;
            s += w[j] * dot_row(td->type, qd, td->data + row * sd, m->moe_xq + j * nbd, m->moe_xk + j * nbk,
                                h + j * ff, ff, &fail);
        }
        out[i] = s;
    }
    if (fail) {
        fprintf(stderr, "moe: unsupported expert type %s/%s/%s\n",
                type_name(tg->type), type_name(tu->type), type_name(td->type));
        return -1;
    }
    return 0;
}

// =========================================================================
// Forward pass — matches llama.cpp gemma4.cpp and the working Go version
// =========================================================================

static float *forward(model_t *m, int token, int pos) {
    int n_embd = m->n_embd, n_ple = m->n_ple;

    // 1. embedding (× sqrt(n_embd) for Gemma only)
    if (read_embedding(m->tok_embd, token, m->x, n_embd, m->embd_scale) != 0)
        return NULL;

    // 2. per-layer PLE inputs
    if (n_ple > 0) {
        int tot = n_ple * m->n_layer;
        // the token-specific part (stored temporarily in ple_in)
        if (read_embedding(m->ple_tok_embd, token, m->ple_in, tot, sqrtf((float)n_ple)) != 0)
            return NULL;
        // projecting x needs a second buffer — use thread 0's scratch (safe: matvec has not started)
        float *proj = (float *)malloc((size_t)tot * sizeof(float));
        if (matvec(m->ple_model_proj, m->x, proj) != 0) { free(proj); return NULL; }
        float ps = 1.0f / sqrtf((float)n_embd);
        for (int i = 0; i < tot; i++) proj[i] *= ps;
        for (int il = 0; il < m->n_layer; il++)
            rmsnorm_w(proj + il * n_ple, m->ple_proj_norm, n_ple, m->norm_eps);
        float inv2 = 1.0f / sqrtf(2.0f);
        for (int i = 0; i < tot; i++) m->ple_in[i] = (m->ple_in[i] + proj[i]) * inv2;
        free(proj);
    }

    // 3. the layers
    for (int il = 0; il < m->n_layer; il++) {
        layer_t *L = &m->layers[il];
        int swa = m->is_swa[il];
        int hd = L->head_dim;
        int n_rot = swa ? m->n_rot_swa : m->n_rot;
        if (n_rot <= 0 || n_rot > hd) n_rot = hd;
        double base = swa ? m->rope_base_swa : m->rope_base;
        double fscale = swa ? 1.0 : m->rope_scale;
        const float *ff = (!swa && m->rope_freqs && m->rope_freqs_n >= n_rot / 2)
                          ? m->rope_freqs : NULL;

        // attn_norm
        memcpy(m->cur, m->x, n_embd * sizeof(float));
        rmsnorm_w(m->cur, L->attn_norm, n_embd, m->norm_eps);

        // Q → (bias) → per-head norm if present → RoPE
        if (matvec(L->wq, m->cur, m->q) != 0) return NULL;
        if (L->bq) for (int i = 0; i < L->q_dim; i++) m->q[i] += L->bq[i];
        if (L->q_norm)
            for (int h = 0; h < L->n_head; h++)
                rmsnorm_w(m->q + h * hd, L->q_norm, hd, m->norm_eps);
        apply_rope(m->q, L->q_dim, pos, hd, n_rot, base, fscale, ff, m->rope_neox);

        // K/V for owning layers
        if (il < m->n_kv_from_start) {
            if (matvec(L->wk, m->cur, m->k) != 0) return NULL;
            if (L->bk) for (int i = 0; i < L->kv_dim; i++) m->k[i] += L->bk[i];
            if (L->wv) {
                if (matvec(L->wv, m->cur, m->v) != 0) return NULL;
                if (L->bv) for (int i = 0; i < L->kv_dim; i++) m->v[i] += L->bv[i];
            } else {
                memcpy(m->v, m->k, L->kv_dim * sizeof(float)); // V = raw K projection
            }
            for (int h = 0; h < L->n_head_kv; h++) {
                if (L->k_norm) rmsnorm_w(m->k + h * hd, L->k_norm, hd, m->norm_eps);
                if (m->v_norm) rmsnorm_plain(m->v + h * hd, hd, m->norm_eps);
            }
            apply_rope(m->k, L->kv_dim, pos, hd, n_rot, base, fscale, ff, m->rope_neox);

            int slot = pos % m->cache_cap[il];
            if (m->kvq == 2) { // debug: the reference cache takes k/v before rotation
                memcpy(m->cache_k[il] + (size_t)slot * L->kv_dim, m->k, L->kv_dim * sizeof(float));
                memcpy(m->cache_v[il] + (size_t)slot * L->kv_dim, m->v, L->kv_dim * sizeof(float));
            }
            if (m->kvq) {
                // TurboQuant: rotate each head slice then quantize it into the cache record
                for (int h = 0; h < L->n_head_kv; h++) {
                    kvq_rotate(m->k + h * hd, hd);
                    kvq_rotate(m->v + h * hd, hd);
                    kvq_quant_slice(m->k + h * hd, hd,
                        m->cache_kq[il] + (size_t)slot * m->kq_stride[il] + h * KVQ_REC(hd, m->kvq_kbits),
                        m->kvq_kbits);
                    kvq_quant_slice(m->v + h * hd, hd,
                        m->cache_vq[il] + (size_t)slot * m->vq_stride[il] + h * KVQ_REC(hd, m->kvq_vbits),
                        m->kvq_vbits);
                }
            } else {
                memcpy(m->cache_k[il] + (size_t)slot * L->kv_dim, m->k, L->kv_dim * sizeof(float));
                memcpy(m->cache_v[il] + (size_t)slot * L->kv_dim, m->v, L->kv_dim * sizeof(float));
            }
        }

        // attention — Gemma 4: scale = 1.0 (q/k normalized), Gemma 3: 1/√qpas, others: 1/sqrt(head_dim)
        float qk_scale = m->attn_scale > 0.0f ? m->attn_scale : 1.0f / sqrtf((float)hd);
        int dl = m->kv_donor[il];
        layer_t *DL = &m->layers[dl];
        int kv_dim = DL->kv_dim;
        int n_head_kv = DL->n_head_kv;
        int cache_cap = m->cache_cap[dl];
        const float *ck = m->cache_k[dl], *cv = m->cache_v[dl];
        const uint8_t *ckq = m->cache_kq[dl], *cvq = m->cache_vq[dl];
        int kqs = m->kq_stride[dl], vqs = m->vq_stride[dl];

        int start = 0;
        if (swa && m->n_swa > 0 && pos + 1 > m->n_swa) start = pos + 1 - m->n_swa;
        // do not exceed the donor cache capacity (safety)
        if (pos + 1 - start > cache_cap) start = pos + 1 - cache_cap;
        int np = pos + 1 - start;

        // TurboQuant: rotate q with the same rotation as the keys — the dot product is preserved (orthogonal rotation)
        if (m->kvq == 2) memcpy(m->dbg_q, m->q, (size_t)L->q_dim * sizeof(float));
        if (m->kvq)
            for (int h = 0; h < L->n_head; h++) kvq_rotate(m->q + h * hd, hd);

        float dbg_serr = 0.0f, dbg_smax = 0.0f; // debug: largest score difference and largest reference value
        if (m->kvq == 2) memset(m->dbg_o, 0, (size_t)L->q_dim * sizeof(float));

        memset(m->attn_out, 0, (size_t)L->q_dim * sizeof(float));
        for (int h = 0; h < L->n_head; h++) {
            int kvh = h * n_head_kv / L->n_head;
            const float *qh = m->q + h * hd;
            int kvo = kvh * hd;
            if (m->kvq) {
                for (int p = start; p <= pos; p++) {
                    const uint8_t *rk = ckq + (size_t)(p % cache_cap) * kqs
                                      + kvh * KVQ_REC(hd, m->kvq_kbits);
                    m->scores[p - start] = kvq_dot_k(qh, rk, hd, m->kvq_kbits) * qk_scale;
                }
            } else {
                for (int p = start; p <= pos; p++) {
                    const float *pk = ck + (size_t)(p % cache_cap) * kv_dim + kvo;
                    m->scores[p - start] = dot_f32v(qh, pk, hd) * qk_scale;
                }
            }
            if (m->kvq == 2) { // reference: the same computation with unrotated q and the f32 cache
                const float *qf = m->dbg_q + h * hd;
                for (int p = start; p <= pos; p++) {
                    const float *pk = ck + (size_t)(p % cache_cap) * kv_dim + kvo;
                    float sf = dot_f32v(qf, pk, hd) * qk_scale;
                    m->dbg_scores[p - start] = sf;
                    float d = fabsf(m->scores[p - start] - sf);
                    if (d > dbg_serr) dbg_serr = d;
                    if (fabsf(sf) > dbg_smax) dbg_smax = fabsf(sf);
                }
                softmax_inplace(m->dbg_scores, np);
                float *od = m->dbg_o + h * hd;
                for (int p = start; p <= pos; p++) {
                    float w = m->dbg_scores[p - start];
                    if (w == 0.0f) continue;
                    const float *pv = cv + (size_t)(p % cache_cap) * kv_dim + kvo;
                    for (int i = 0; i < hd; i++) od[i] += w * pv[i];
                }
            }
            softmax_inplace(m->scores, np);
            float *oh = m->attn_out + h * hd;
            for (int p = start; p <= pos; p++) {
                float w = m->scores[p - start];
                if (w == 0.0f) continue;
                if (m->kvq) {
                    kvq_axpy_v(oh, w, cvq + (size_t)(p % cache_cap) * vqs
                                    + kvh * KVQ_REC(hd, m->kvq_vbits), hd, m->kvq_vbits);
                } else {
                    const float *pv = cv + (size_t)(p % cache_cap) * kv_dim + kvo;
                    for (int i = 0; i < hd; i++) oh[i] += w * pv[i];
                }
            }
        }

        // TurboQuant: the value sum accumulated in rotated space — rotate it back before the output projection
        if (m->kvq)
            for (int h = 0; h < L->n_head; h++) kvq_unrotate(m->attn_out + h * hd, hd);

        if (m->kvq == 2 && (pos < 3 || il < 2)) { // debug: relative output error
            float oe = 0.0f, on = 0.0f;
            for (int i = 0; i < L->q_dim; i++) {
                float d = m->attn_out[i] - m->dbg_o[i];
                oe += d * d;
                on += m->dbg_o[i] * m->dbg_o[i];
            }
            printf("[kvdbg] pos=%d il=%d score_err=%.4f (max=%.3f) out_relerr=%.4f\n",
                   pos, il, dbg_serr, dbg_smax, sqrtf(oe / (on + 1e-9f)));
        }

        // output projection + post_attention_norm (if present) + residual
        if (matvec(L->wo, m->attn_out, m->ao) != 0) return NULL;
        if (L->attn_post_norm) rmsnorm_w(m->ao, L->attn_post_norm, n_embd, m->norm_eps);
        for (int i = 0; i < n_embd; i++) m->ao[i] += m->x[i];

        // FFN — GELU for Gemma, SiLU for the rest, or MoE experts
        memcpy(m->f, m->ao, n_embd * sizeof(float));
        rmsnorm_w(m->f, L->ffn_norm, n_embd, m->norm_eps);
        if (L->ffn_gate_inp) {
            if (moe_ffn(m, L, m->f, m->ffn_out) != 0) return NULL;
        } else {
            if (matvec(L->ffn_gate, m->f, m->gate) != 0) return NULL;
            if (matvec(L->ffn_up, m->f, m->up) != 0) return NULL;
            if (m->act_gelu)
                for (int i = 0; i < L->ff_dim; i++) m->gate[i] = gelu_f(m->gate[i]) * m->up[i];
            else
                for (int i = 0; i < L->ff_dim; i++) m->gate[i] = silu_f(m->gate[i]) * m->up[i];
            if (matvec(L->ffn_down, m->gate, m->ffn_out) != 0) return NULL;
        }
        if (L->ffn_post_norm) rmsnorm_w(m->ffn_out, L->ffn_post_norm, n_embd, m->norm_eps);
        for (int i = 0; i < n_embd; i++) m->ffn_out[i] += m->ao[i];

        // PLE
        if (n_ple > 0) {
            if (matvec(L->ple_inp_gate, m->ffn_out, m->ple_t) != 0) return NULL;
            const float *pl = m->ple_in + il * n_ple;
            for (int i = 0; i < n_ple; i++) m->ple_t[i] = gelu_f(m->ple_t[i]) * pl[i];
            if (matvec(L->ple_proj, m->ple_t, m->ple_t2) != 0) return NULL;
            rmsnorm_w(m->ple_t2, L->ple_post_norm, n_embd, m->norm_eps);
            for (int i = 0; i < n_embd; i++) m->ffn_out[i] += m->ple_t2[i];
        }

        if (L->out_scale != 1.0f)
            for (int i = 0; i < n_embd; i++) m->ffn_out[i] *= L->out_scale;

        memcpy(m->x, m->ffn_out, n_embd * sizeof(float));
    }

    // 4. final norm + LM head + softcap
    rmsnorm_w(m->x, m->output_norm, n_embd, m->norm_eps);
    if (matvec(m->output, m->x, m->logits) != 0) return NULL;
    if (m->final_softcap > 0.0f) {
        float cp = m->final_softcap;
        for (int i = 0; i < m->n_vocab; i++)
            m->logits[i] = cp * tanhf(m->logits[i] / cp);
    }
    return m->logits;
}

// =========================================================================
// Commands: chat / inspect / dump / selftest
// =========================================================================

static double now_sec(void) {
#ifdef _WIN32
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
#endif
}

// reads a whole line of any length (without \r\n) as UTF-8 into new memory, or NULL at end of input.
// on the Windows console fgets returns the OEM code page, not UTF-8, so non-ASCII text arrives broken — so we read
// UTF-16 directly with ReadConsoleW and convert it. Redirected input (pipe/file) is read as-is.
static char *read_line(void) {
#ifdef _WIN32
    HANDLE h = GetStdHandle(STD_INPUT_HANDLE);
    DWORD mode;
    if (h != INVALID_HANDLE_VALUE && GetConsoleMode(h, &mode)) {
        size_t cap = 1024, n = 0;
        wchar_t *w = (wchar_t *)malloc(cap * sizeof(wchar_t));
        for (;;) {
            if (cap - n < 512) { cap *= 2; w = (wchar_t *)realloc(w, cap * sizeof(wchar_t)); }
            DWORD got = 0;
            if (!ReadConsoleW(h, w + n, 512, &got, NULL) || got == 0) break;
            n += got;
            if (w[n - 1] == L'\n') break;
        }
        if (n == 0 || w[0] == 0x1A) { free(w); return NULL; } // Ctrl+Z = end of input
        while (n && (w[n - 1] == L'\n' || w[n - 1] == L'\r')) n--;
        int len = n ? WideCharToMultiByte(CP_UTF8, 0, w, (int)n, NULL, 0, NULL, NULL) : 0;
        char *s = (char *)malloc((size_t)len + 1);
        if (len) WideCharToMultiByte(CP_UTF8, 0, w, (int)n, s, len, NULL, NULL);
        s[len] = 0;
        free(w);
        return s;
    }
#endif
    size_t cap = 1024, n = 0;
    char *s = (char *)malloc(cap);
    while (fgets(s + n, (int)(cap - n), stdin)) {
        n += strlen(s + n);
        if (n && s[n - 1] == '\n') break;
        if (cap - n < 2) { cap *= 2; s = (char *)realloc(s, cap); }
    }
    if (n == 0) { free(s); return NULL; }
    while (n && (s[n - 1] == '\n' || s[n - 1] == '\r')) n--;
    s[n] = 0;
    return s;
}

#ifdef _WIN32
// CRT argv is ANSI and replaces unrepresentable characters with '?' — build a UTF-8 copy from the original command line
static char **utf8_argv(int *argc) {
    int n = 0;
    wchar_t **wv = CommandLineToArgvW(GetCommandLineW(), &n);
    if (!wv) return NULL;
    char **v = (char **)malloc((size_t)(n + 1) * sizeof(char *));
    for (int i = 0; i < n; i++) {
        int len = WideCharToMultiByte(CP_UTF8, 0, wv[i], -1, NULL, 0, NULL, NULL);
        v[i] = (char *)malloc((size_t)(len > 0 ? len : 1));
        if (len <= 0 || !WideCharToMultiByte(CP_UTF8, 0, wv[i], -1, v[i], len, NULL, NULL)) v[i][0] = 0;
    }
    v[n] = NULL;
    LocalFree(wv);
    *argc = n;
    return v;
}
#endif

// max_tokens <= 0 = unlimited (until a stop token or the context is full)
static int cmd_chat(const char *path, int n_ctx, int max_tokens, sampler_t *smp) {
    gguf_t g;
    if (gguf_open(&g, path) != 0) return 2;
    tok_t tk;
    if (tok_load(&tk, &g) != 0) return 2;
    model_t *m = (model_t *)calloc(1, sizeof(model_t));
    if (model_load(m, &g, n_ctx) != 0) return 2;

    printf("\n=== fastgguf-c | AVX2 %s | OpenMP %d threads ===\n",
#if defined(__AVX2__)
           "ON",
#else
           "OFF",
#endif
           omp_get_max_threads());
    // without --seed take a seed from the clock, and print it so the same reply can be reproduced with --seed
    if (smp->rng == 0) smp->rng = ((uint64_t)time(NULL) * 0x9E3779B97F4A7C15ULL) ^ (uint64_t)(now_sec() * 1e6);
    if (smp->rng == 0) smp->rng = 0x9E3779B97F4A7C15ULL;
    smp->cand = m->cand;
    if (smp->temp <= 0.0f)
        printf("[Sampler] greedy");
    else
        printf("[Sampler] temp=%.2f top_k=%d top_p=%.2f", smp->temp, smp->top_k, smp->top_p);
    printf(" repeat_penalty=%.2f (last %d) max_tokens=%d seed=%llu\n", smp->repeat_penalty,
           smp->repeat_last_n, max_tokens, (unsigned long long)smp->rng);
    printf("Type your message, /reset for a new conversation, or /exit to quit\n");

    int pos = 0; // conversation length in the KV cache — persists between messages until /reset
    int *hist = (int *)malloc((size_t)n_ctx * sizeof(int)); // hist[p] = token at position p
    for (;;) {
        printf("\nyou> ");
        fflush(stdout);
        char *line = read_line();
        if (!line) break;
        if (line[0] == 0) { free(line); continue; }
        if (strcmp(line, "/exit") == 0) { free(line); break; }
        if (strcmp(line, "/reset") == 0) {
            free(line);
            pos = 0;
            printf("[new conversation]\n");
            continue;
        }

        int n_ids = 0;
        int *ids = encode_turn(&tk, line, pos, &n_ids);
        // the turn + one reply token must fit; otherwise start a new conversation with this message alone —
        // unless it does not fit even alone, in which case reject it and keep the conversation as is
        if (pos > 0 && pos + n_ids >= n_ctx) {
            int n_fresh = 0;
            int *fresh = encode_turn(&tk, line, 0, &n_fresh);
            if (n_fresh < n_ctx) {
                printf("[context full (%d/%d) - starting a new conversation]\n", pos, n_ctx);
                free(ids);
                ids = fresh;
                n_ids = n_fresh;
                pos = 0;
            } else free(fresh);
        }
        free(line);
        if (pos + n_ids >= n_ctx) {
            printf("[message too long: %d tokens, context is %d - run with a larger ctx]\n",
                   n_ids, n_ctx);
            free(ids);
            continue;
        }
        printf("[%d tokens | ctx %d/%d]\n%s> ", n_ids, pos + n_ids, n_ctx, m->arch_name);
        fflush(stdout);

        double t0 = now_sec();
        float *logits = NULL;
        for (int i = 0; i < n_ids; i++) {
            logits = forward(m, ids[i], pos);
            if (!logits) break;
            hist[pos++] = ids[i];
        }
        free(ids);
        // the cache is now in a partial state — it cannot be built upon
        if (!logits) { printf("\n[ERROR in prompt - conversation reset]\n"); pos = 0; continue; }
        double t_prompt = now_sec() - t0;

        double t1 = now_sec();
        int n_gen = 0;
        for (int i = 0; max_tokens <= 0 || i < max_tokens; i++) {
            int next = sample_next(smp, &tk, logits, m->n_vocab, hist, pos);
            if (tok_is_stop(&tk, next)) break;
            tok_print(&tk, next);
            fflush(stdout);
            n_gen++;
            if (pos >= n_ctx) { out_flush_pending(); printf("\n[context full]"); break; }
            logits = forward(m, next, pos);
            if (!logits) {
                out_flush_pending();
                printf("\n[ERROR in gen - conversation reset]\n");
                pos = 0;
                break;
            }
            hist[pos++] = next;
        }
        out_flush_pending();
        double t_gen = now_sec() - t1;
        printf("\n[prompt: %.1fs (%.2f tok/s) | gen: %d tok @ %.2f tok/s]\n",
               t_prompt, n_ids / (t_prompt > 0 ? t_prompt : 1),
               n_gen, n_gen / (t_gen > 0 ? t_gen : 1));
    }
    return 0;
}

static int cmd_inspect(const char *path) {
    gguf_t g;
    if (gguf_open(&g, path) != 0) return 2;
    printf("GGUF v%u | tensors=%zu | meta=%zu | align=%llu\n",
           g.version, g.n_tensors, g.n_kvs, (unsigned long long)g.alignment);
    kv_t *arch_kv = kv_find(&g, "general.architecture");
    if (arch_kv && arch_kv->kind == KV_STR) {
        const char *arch = arch_kv->str;
        char key[160];
        printf("arch: %s\n", arch);
        int n_exp = (int)kv_num(&g, K("expert_count"), 0);
        if (n_exp > 0) {
            // what is read per token: all dense weights + only used/count of the experts
            uint64_t exp_b = 0, all_b = 0;
            for (size_t i = 0; i < g.n_tensors; i++) {
                all_b += g.tensors[i].nbytes;
                if (strstr(g.tensors[i].name, "_exps.")) exp_b += g.tensors[i].nbytes;
            }
            int n_used = (int)kv_num(&g, K("expert_used_count"), 0);
            printf("MoE: experts=%d, used/token=%d, ff_exp=%d\n", n_exp, n_used,
                   (int)kv_num(&g, K("expert_feed_forward_length"), 0));
            printf("     dense weights %.2f GB (always read) + experts %.2f GB"
                   " (~%.2f GB read per token)\n",
                   (all_b - exp_b) / 1e9, exp_b / 1e9, exp_b / 1e9 * n_used / n_exp);
        }
    }
    int counts[64] = {0};
    for (size_t i = 0; i < g.n_tensors; i++)
        if (g.tensors[i].type < 64) counts[g.tensors[i].type]++;
    printf("\ntensor types:\n");
    for (int t = 0; t < 64; t++)
        if (counts[t]) printf("  %-6s %d\n", type_name((uint32_t)t), counts[t]);
    printf("\nfirst tensors:\n");
    for (size_t i = 0; i < g.n_tensors && i < 15; i++) {
        tensor_t *t = &g.tensors[i];
        printf("  %-44s %-5s [", t->name, type_name(t->type));
        for (int j = 0; j < t->n_dims; j++)
            printf("%llu%s", (unsigned long long)t->ne[j], j + 1 < t->n_dims ? "," : "");
        printf("]\n");
    }
    return 0;
}

static int cmd_dump(const char *path) {
    gguf_t g;
    if (gguf_open(&g, path) != 0) return 2;
    for (size_t i = 0; i < g.n_kvs; i++) {
        kv_t *kv = &g.kvs[i];
        switch (kv->kind) {
        case KV_STR:     printf("%s = %s\n", kv->key, kv->str); break;
        case KV_NUM:     printf("%s = %g\n", kv->key, kv->num); break;
        case KV_INT_ARR: printf("%s = [%zu ints]\n", kv->key, kv->iarr_n); break;
        case KV_STR_ARR: printf("%s = [%zu strings]\n", kv->key, kv->sarr_n); break;
        }
    }
    return 0;
}

// K-quant kernel against an independent float reference: dequantized row · dequantized quantized x
// the error is measured relative to Σ|term|, not the sum, because the min term can cancel most of the value
static int test_kquant(const char *name, uint32_t type, int blk_bytes,
                       float (*dot)(const uint8_t *, const q8k_blk_t *, int)) {
    int n = 1024, nb = n / 256;
    uint8_t *row = (uint8_t *)malloc((size_t)nb * blk_bytes);
    for (int i = 0; i < nb * blk_bytes; i++) row[i] = (uint8_t)(rand() & 0xFF);
    for (int b = 0; b < nb; b++) {
        uint8_t *p = row + b * blk_bytes;
        uint16_t d = (uint16_t)(0x2C00 + (rand() & 0x3FF));    // ~0.06..0.12
        uint16_t dmin = (uint16_t)(0x2400 + (rand() & 0x3FF)); // ~0.016..0.03
        if (type == T_Q6_K) memcpy(p + 208, &d, 2);
        else { memcpy(p, &d, 2); memcpy(p + 2, &dmin, 2); }
    }
    float *x = (float *)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) x[i] = (float)(rand() % 2000 - 1000) / 500.0f;
    q8k_blk_t *xk = (q8k_blk_t *)malloc((n / 256) * sizeof(q8k_blk_t));
    quantize_q8k(x, xk, n);

    float fast = dot(row, xk, nb);

    float *dq = (float *)malloc(n * sizeof(float));
    dequantize_row(type, row, dq, n);
    double ref = 0.0, mag = 0.0;
    for (int i = 0; i < n; i++) {
        double t = (double)dq[i] * (double)(xk[i / 256].d * (float)xk[i / 256].qs[i % 256]);
        ref += t;
        mag += fabs(t);
    }
    double rel = fabs((double)fast - ref) / (mag + 1e-9);
    printf("%-5s fast=%.6f float_ref=%.6f rel_err=%.2e %s\n", name, fast, ref, rel,
           rel < 1e-4 ? "OK" : "FAIL");
    free(row); free(x); free(xk); free(dq);
    return rel < 1e-4 ? 0 : 1;
}

// sampling matches the probabilities, top-p cuts the tail, greedy/top-k=1 = argmax, the penalty flips the order
static int test_sampler(void) {
    int fails = 0, N = 200000;
    tok_t tk0;
    memset(&tk0, 0, sizeof(tk0)); // no stop tokens
    cand_t cand[1000];
    sampler_t s = { 1.0f, 1.0f, 1.0f, 0, 64, 12345, cand };
    float base[8] = { logf(0.5f), logf(0.3f), logf(0.2f),
                      -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY };
    float lg[8];
    int cnt[8];

    memset(cnt, 0, sizeof(cnt));
    for (int i = 0; i < N; i++) { memcpy(lg, base, sizeof(lg)); cnt[sample_next(&s, &tk0, lg, 8, NULL, 0)]++; }
    float f0 = cnt[0] / (float)N, f1 = cnt[1] / (float)N, f2 = cnt[2] / (float)N;
    int ok = fabsf(f0 - 0.5f) < 0.01f && fabsf(f1 - 0.3f) < 0.01f && fabsf(f2 - 0.2f) < 0.01f
             && cnt[0] + cnt[1] + cnt[2] == N;
    printf("sampler dist   %.3f %.3f %.3f (want .5 .3 .2) %s\n", f0, f1, f2, ok ? "OK" : "FAIL");
    fails += !ok;

    // top_p=0.6: 0.5 < 0.6 so the second is added (0.8) and it stops — the third never appears
    s.top_p = 0.6f;
    memset(cnt, 0, sizeof(cnt));
    for (int i = 0; i < N; i++) { memcpy(lg, base, sizeof(lg)); cnt[sample_next(&s, &tk0, lg, 8, NULL, 0)]++; }
    f0 = cnt[0] / (float)N;
    ok = cnt[2] == 0 && fabsf(f0 - 0.625f) < 0.01f;
    printf("sampler top-p  %.3f %.3f %.3f (want .625 .375 0) %s\n", f0, cnt[1] / (float)N,
           cnt[2] / (float)N, ok ? "OK" : "FAIL");
    fails += !ok;

    // temp=0 and top_k=1 are both always argmax
    float tie[8] = { 0.1f, 2.0f, 1.9f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    s.top_p = 1.0f; s.temp = 0.0f;
    memcpy(lg, tie, sizeof(lg));
    int g0 = sample_next(&s, &tk0, lg, 8, NULL, 0);
    s.temp = 1.0f; s.top_k = 1;
    int g1 = 1;
    for (int i = 0; i < 1000; i++) { memcpy(lg, tie, sizeof(lg)); g1 &= sample_next(&s, &tk0, lg, 8, NULL, 0) == 1; }
    ok = g0 == 1 && g1;
    printf("sampler greedy temp0=%d top_k1=%s %s\n", g0, g1 ? "always 1" : "varies", ok ? "OK" : "FAIL");
    fails += !ok;

    // penalty 1.5 on token 1 (repeated twice in history, applied once): 2.0/1.5 < 1.9 → token 2 wins
    int hist[3] = { 1, 5, 1 };
    s.temp = 0.0f; s.repeat_penalty = 1.5f;
    memcpy(lg, tie, sizeof(lg));
    int pw = sample_next(&s, &tk0, lg, 8, hist, 3);
    ok = pw == 2 && fabsf(lg[1] - 2.0f / 1.5f) < 1e-6f;
    printf("sampler penalty winner=%d logit=%.4f %s\n", pw, lg[1], ok ? "OK" : "FAIL");
    fails += !ok;

    // insertion-based top_k_select = top 10 of the full sort
    float big[1000];
    for (int i = 0; i < 1000; i++) big[i] = (float)(rand() % 100000) / 7.0f;
    cand_t ten[10];
    top_k_select(big, 1000, 10, ten);
    top_k_select(big, 1000, 0, cand);
    ok = 1;
    for (int i = 0; i < 10; i++) ok &= ten[i].v == cand[i].v;
    printf("top-k select   10 of 1000 %s\n", ok ? "OK" : "FAIL");
    fails += !ok;
    return fails;
}

// Self-test: compare the AVX2 kernel against the scalar path and against dequantization
static int cmd_selftest(void) {
    srand(42);
    int n = 1024, nb = n / 32;
    int fails = 0;

    // --- Q4_0 ---
    {
        uint8_t *row = (uint8_t *)malloc(nb * 18);
        for (int i = 0; i < nb * 18; i++) row[i] = (uint8_t)(rand() & 0xFF);
        // make the fp16 scales sensible (avoid NaN/Inf)
        for (int b = 0; b < nb; b++) {
            uint16_t h = (uint16_t)(0x2C00 + (rand() & 0x3FF)); // ~0.01..0.06
            memcpy(row + b * 18, &h, 2);
        }
        float *x = (float *)malloc(n * sizeof(float));
        for (int i = 0; i < n; i++) x[i] = (float)(rand() % 2000 - 1000) / 500.0f;

        q8_blk_t *xq = (q8_blk_t *)malloc(nb * sizeof(q8_blk_t));
        quantize_q8(x, xq, n);

        float fast = dot_q4_0_q8(row, xq, nb);

        // reference: explicit scalar on the same quantized blocks
        float ref = 0.0f;
        for (int b = 0; b < nb; b++) {
            const uint8_t *p = row + b * 18;
            float d = FP16(rd_u16(p)) * xq[b].d;
            int si = 0;
            for (int j = 0; j < 16; j++) {
                si += ((int)(p[2 + j] & 0x0F) - 8) * xq[b].qs[j];
                si += ((int)(p[2 + j] >> 4)   - 8) * xq[b].qs[j + 16];
            }
            ref += d * (float)si;
        }
        float rel = fabsf(fast - ref) / (fabsf(ref) + 1e-6f);
        printf("Q4_0  avx2=%.6f ref=%.6f rel_err=%.2e %s\n", fast, ref, rel,
               rel < 1e-4 ? "OK" : "FAIL");
        if (rel >= 1e-4) fails++;

        // float reference: dequantize both sides (the row + the quantized x) — must match the integer path
        float *dq = (float *)malloc(n * sizeof(float));
        float *xdq = (float *)malloc(n * sizeof(float));
        dequantize_row(T_Q4_0, row, dq, n);
        for (int b = 0; b < nb; b++)
            for (int j = 0; j < 32; j++) xdq[b * 32 + j] = xq[b].d * (float)xq[b].qs[j];
        float fref = dot_f32v(dq, xdq, n);
        float rel2 = fabsf(fast - fref) / (fabsf(fref) + 1e-3f);
        printf("Q4_0  float_ref=%.6f rel_err=%.2e %s\n", fref, rel2,
               rel2 < 1e-3 ? "OK" : "FAIL");
        if (rel2 >= 1e-3) fails++;
        free(row); free(x); free(xq); free(dq); free(xdq);
    }

    // --- Q8_0 ---
    {
        uint8_t *row = (uint8_t *)malloc(nb * 34);
        for (int i = 0; i < nb * 34; i++) row[i] = (uint8_t)(rand() & 0xFF);
        for (int b = 0; b < nb; b++) {
            uint16_t h = (uint16_t)(0x2C00 + (rand() & 0x3FF));
            memcpy(row + b * 34, &h, 2);
        }
        float *x = (float *)malloc(n * sizeof(float));
        for (int i = 0; i < n; i++) x[i] = (float)(rand() % 2000 - 1000) / 500.0f;
        q8_blk_t *xq = (q8_blk_t *)malloc(nb * sizeof(q8_blk_t));
        quantize_q8(x, xq, n);
        float fast = dot_q8_0_q8(row, xq, nb);
        float ref = 0.0f;
        for (int b = 0; b < nb; b++) {
            const uint8_t *p = row + b * 34;
            float d = FP16(rd_u16(p)) * xq[b].d;
            const int8_t *qs = (const int8_t *)(p + 2);
            int si = 0;
            for (int j = 0; j < 32; j++) si += (int)qs[j] * xq[b].qs[j];
            ref += d * (float)si;
        }
        float rel = fabsf(fast - ref) / (fabsf(ref) + 1e-6f);
        printf("Q8_0  avx2=%.6f ref=%.6f rel_err=%.2e %s\n", fast, ref, rel,
               rel < 1e-4 ? "OK" : "FAIL");
        if (rel >= 1e-4) fails++;
        free(row); free(x); free(xq);
    }

    // --- Q4_1 / Q5_0 / Q5_1: integer kernel against dequantized row · dequantized x ---
    // error relative to Σ|term|, since the min term of Q4_1/Q5_1 can cancel most of the sum
    {
        static const struct { uint32_t type; const char *name; int bytes, n_scales; } qt[3] = {
            { T_Q4_1, "Q4_1", 20, 2 }, { T_Q5_0, "Q5_0", 22, 1 }, { T_Q5_1, "Q5_1", 24, 2 },
        };
        float *x = (float *)malloc(n * sizeof(float));
        for (int i = 0; i < n; i++) x[i] = (float)(rand() % 2000 - 1000) / 500.0f;
        q8_blk_t *xq = (q8_blk_t *)malloc(nb * sizeof(q8_blk_t));
        quantize_q8(x, xq, n);
        float *xdq = (float *)malloc(n * sizeof(float));
        for (int b = 0; b < nb; b++)
            for (int j = 0; j < 32; j++) xdq[b * 32 + j] = xq[b].d * (float)xq[b].qs[j];
        float *dq = (float *)malloc(n * sizeof(float));
        for (int t = 0; t < 3; t++) {
            int bb = qt[t].bytes;
            uint8_t *row = (uint8_t *)malloc((size_t)nb * bb);
            for (int i = 0; i < nb * bb; i++) row[i] = (uint8_t)(rand() & 0xFF);
            for (int b = 0; b < nb; b++)        // sane fp16 d (and m, which may be negative)
                for (int k = 0; k < qt[t].n_scales; k++) {
                    uint16_t h = (uint16_t)(0x2C00 + (rand() & 0x3FF)) | (k && (rand() & 1) ? 0x8000 : 0);
                    memcpy(row + b * bb + 2 * k, &h, 2);
                }
            float fast = qt[t].type == T_Q4_1 ? dot_q4_1_q8(row, xq, nb)
                       : qt[t].type == T_Q5_0 ? dot_q5_0_q8(row, xq, nb)
                       :                        dot_q5_1_q8(row, xq, nb);
            dequantize_row(qt[t].type, row, dq, n);
            float fref = 0.0f, mag = 0.0f;
            for (int i = 0; i < n; i++) { fref += dq[i] * xdq[i]; mag += fabsf(dq[i] * xdq[i]); }
            float rel = fabsf(fast - fref) / (mag + 1e-6f);
            printf("%s  fast=%.6f float_ref=%.6f rel_err=%.2e %s\n", qt[t].name, fast, fref, rel,
                   rel < 1e-4 ? "OK" : "FAIL");
            if (rel >= 1e-4) fails++;
            free(row);
        }
        free(x); free(xq); free(xdq); free(dq);
    }


    // --- Q2_K / Q3_K / IQ4: integer kernel against full dequantization of both sides ---
    {
        int n2 = 1024, nb2 = n2 / 256;
        float *x = (float *)malloc(n2 * sizeof(float));
        for (int i = 0; i < n2; i++) x[i] = (float)(rand() % 2000 - 1000) / 500.0f;
        q8_blk_t *xq = (q8_blk_t *)malloc((n2 / 32) * sizeof(q8_blk_t));
        quantize_q8(x, xq, n2);
        float *xdq = (float *)malloc(n2 * sizeof(float));
        for (int b = 0; b < n2 / 32; b++)
            for (int j = 0; j < 32; j++) xdq[b * 32 + j] = xq[b].d * (float)xq[b].qs[j];
        float *dq = (float *)malloc(n2 * sizeof(float));

        // Q2_K
        uint8_t *r2 = (uint8_t *)malloc(nb2 * 84);
        for (int i = 0; i < nb2 * 84; i++) r2[i] = (uint8_t)(rand() & 0xFF);
        for (int b = 0; b < nb2; b++) {
            uint16_t h = (uint16_t)(0x2C00 + (rand() & 0x3FF));
            memcpy(r2 + b * 84 + 80, &h, 2);
            h = (uint16_t)(0x2800 + (rand() & 0x3FF));
            memcpy(r2 + b * 84 + 82, &h, 2);
        }
        float fast2 = dot_q2_K_q8(r2, xq, nb2);
        dequantize_row(T_Q2_K, r2, dq, n2);
        float ref2 = dot_f32v(dq, xdq, n2);
        float relq2 = fabsf(fast2 - ref2) / (fabsf(ref2) + 1e-3f);
        printf("Q2_K  fast=%.6f ref=%.6f rel_err=%.2e %s\n", fast2, ref2, relq2,
               relq2 < 1e-3 ? "OK" : "FAIL");
        if (relq2 >= 1e-3) fails++;
        free(r2);

        // Q3_K
        uint8_t *r3 = (uint8_t *)malloc(nb2 * 110);
        for (int i = 0; i < nb2 * 110; i++) r3[i] = (uint8_t)(rand() & 0xFF);
        for (int b = 0; b < nb2; b++) {
            uint16_t h = (uint16_t)(0x2C00 + (rand() & 0x3FF));
            memcpy(r3 + b * 110 + 108, &h, 2);
        }
        float fast3 = dot_q3_K_q8(r3, xq, nb2);
        dequantize_row(T_Q3_K, r3, dq, n2);
        float ref3 = dot_f32v(dq, xdq, n2);
        float relq3 = fabsf(fast3 - ref3) / (fabsf(ref3) + 1e-3f);
        printf("Q3_K  fast=%.6f ref=%.6f rel_err=%.2e %s\n", fast3, ref3, relq3,
               relq3 < 1e-3 ? "OK" : "FAIL");
        if (relq3 >= 1e-3) fails++;
        free(r3);

        // IQ4_NL (32-element blocks — reuse the same x with n2/32 blocks)
        {
            int nbn = n2 / 32;
            uint8_t *rn = (uint8_t *)malloc(nbn * 18);
            for (int i = 0; i < nbn * 18; i++) rn[i] = (uint8_t)(rand() & 0xFF);
            for (int b = 0; b < nbn; b++) {
                uint16_t h = (uint16_t)(0x2C00 + (rand() & 0x3FF));
                memcpy(rn + b * 18, &h, 2);
            }
            float fastn = dot_iq4_nl_q8(rn, xq, nbn);
            dequantize_row(T_IQ4_NL, rn, dq, n2);
            float refn = dot_f32v(dq, xdq, n2);
            float reln = fabsf(fastn - refn) / (fabsf(refn) + 1e-3f);
            printf("IQ4_NL fast=%.6f ref=%.6f rel_err=%.2e %s\n", fastn, refn, reln,
                   reln < 1e-3 ? "OK" : "FAIL");
            if (reln >= 1e-3) fails++;
            free(rn);
        }

        // IQ4_XS
        {
            uint8_t *rx = (uint8_t *)malloc(nb2 * 136);
            for (int i = 0; i < nb2 * 136; i++) rx[i] = (uint8_t)(rand() & 0xFF);
            for (int b = 0; b < nb2; b++) {
                uint16_t h = (uint16_t)(0x2C00 + (rand() & 0x3FF));
                memcpy(rx + b * 136, &h, 2);
            }
            float fastx = dot_iq4_xs_q8(rx, xq, nb2);
            dequantize_row(T_IQ4_XS, rx, dq, n2);
            float refx = dot_f32v(dq, xdq, n2);
            float relx = fabsf(fastx - refx) / (fabsf(refx) + 1e-3f);
            printf("IQ4_XS fast=%.6f ref=%.6f rel_err=%.2e %s\n", fastx, refx, relx,
                   relx < 1e-3 ? "OK" : "FAIL");
            if (relx >= 1e-3) fails++;
            free(rx);
        }
        free(x); free(xq); free(xdq); free(dq);
    }

    // --- Q4_K / Q5_K / Q6_K: independent reference, error relative to Σ|term| ---
    fails += test_kquant("Q4_K", T_Q4_K, 144, dot_q4_K_q8k);
    fails += test_kquant("Q5_K", T_Q5_K, 176, dot_q5_K_q8k);
    fails += test_kquant("Q6_K", T_Q6_K, 210, dot_q6_K_q8k);

    // --- sampler ---
    fails += test_sampler();

    // --- RoPE: linear scaling 0.5 at position 2p = no scaling at p, and really differs from no scaling ---
    {
        float a[512], b[512], c[512];
        for (int i = 0; i < 512; i++) a[i] = b[i] = c[i] = (float)(rand() % 2000 - 1000) / 500.0f;
        apply_rope(a, 512, 1234, 256, 256, 1e6, 0.5, NULL, 1);
        apply_rope(b, 512, 617, 256, 256, 1e6, 1.0, NULL, 1);
        apply_rope(c, 512, 1234, 256, 256, 1e6, 1.0, NULL, 1);
        float same = 0.0f, diff = 0.0f;
        for (int i = 0; i < 512; i++) {
            same = fmaxf(same, fabsf(a[i] - b[i]));
            diff = fmaxf(diff, fabsf(a[i] - c[i]));
        }
        int ok = same < 1e-5f && diff > 0.1f;
        printf("rope linear scale  same=%.1e changed=%.2f %s\n", same, diff, ok ? "OK" : "FAIL");
        fails += !ok;
    }

    // --- MoE: top-k + renormalization + per-expert offset inside the 3D tensor, against a float reference ---
    {
        enum { NE = 8, NK = 3, D = 64, F = 64 };
        matvec_init_scratch(D);
        model_t *mm = (model_t *)calloc(1, sizeof(model_t));
        mm->n_embd = D; mm->ff_exp = F; mm->n_expert = NE; mm->n_expert_used = NK;
        mm->moe_probs = (float *)malloc(NE * sizeof(float));
        mm->moe_w = (float *)malloc(NK * sizeof(float));
        mm->moe_sel = (int *)malloc(NK * sizeof(int));
        mm->moe_h = (float *)malloc(NK * F * sizeof(float));
        mm->moe_xq = (q8_blk_t *)malloc((NK * F / 32 + 1) * sizeof(q8_blk_t));
        mm->moe_xk = (q8k_blk_t *)malloc((NK * F / 256 + 1) * sizeof(q8k_blk_t));

        static float router[NE * D], x[D], out[D], ref[D];
        for (int i = 0; i < NE * D; i++) router[i] = (float)(rand() % 2000 - 1000) / 2000.0f;
        for (int i = 0; i < D; i++) x[i] = (float)(rand() % 2000 - 1000) / 500.0f;
        // Q8_0 experts: gate/up [D, F, NE] and down [F, D, NE]
        int nblk = NE * F * D / 32;
        uint8_t *qe[3];
        float *fe[3];
        for (int t = 0; t < 3; t++) {
            qe[t] = (uint8_t *)malloc((size_t)nblk * 34);
            for (int b = 0; b < nblk; b++) {
                uint16_t h = (uint16_t)(0x2000 + (rand() & 0x3FF)); // ~0.008..0.016
                memcpy(qe[t] + b * 34, &h, 2);
                for (int j = 0; j < 32; j++) qe[t][b * 34 + 2 + j] = (uint8_t)(rand() & 0xFF);
            }
            fe[t] = (float *)malloc((size_t)nblk * 32 * sizeof(float));
            dequantize_row(T_Q8_0, qe[t], fe[t], nblk * 32);
        }
        tensor_t ti, tg, tu, td;
        memset(&ti, 0, sizeof(ti));
        ti.type = T_F32; ti.n_dims = 2; ti.ne[0] = D; ti.ne[1] = NE; ti.data = (const uint8_t *)router;
        tg = tu = td = ti;
        tg.type = tu.type = td.type = T_Q8_0;
        tg.n_dims = tu.n_dims = td.n_dims = 3;
        tg.ne[0] = D; tg.ne[1] = F; tg.ne[2] = NE; tu.ne[0] = D; tu.ne[1] = F; tu.ne[2] = NE;
        td.ne[0] = F; td.ne[1] = D; td.ne[2] = NE;
        tg.data = qe[0]; tu.data = qe[1]; td.data = qe[2];
        layer_t LL;
        memset(&LL, 0, sizeof(LL));
        LL.ffn_gate_inp = &ti; LL.ffn_gate_exps = &tg; LL.ffn_up_exps = &tu; LL.ffn_down_exps = &td;
        int rc = moe_ffn(mm, &LL, x, out);

        // reference: full softmax, selection by simple sort, and float computation per expert from dequantized rows
        float pr[NE], pmax = -INFINITY, psum = 0.0f;
        for (int e = 0; e < NE; e++) {
            pr[e] = dot_f32v(router + e * D, x, D);
            if (pr[e] > pmax) pmax = pr[e];
        }
        for (int e = 0; e < NE; e++) { pr[e] = expf(pr[e] - pmax); psum += pr[e]; }
        int taken[NE] = {0}, sel[NK];
        float wsel[NK], wsum = 0.0f;
        for (int j = 0; j < NK; j++) {
            int best = -1;
            for (int e = 0; e < NE; e++)
                if (!taken[e] && (best < 0 || pr[e] > pr[best])) best = e;
            taken[best] = 1; sel[j] = best; wsel[j] = pr[best] / psum; wsum += wsel[j];
        }
        memset(ref, 0, sizeof(ref));
        for (int j = 0; j < NK; j++) {
            float hh[F];
            for (int r = 0; r < F; r++) {
                const float *gr = fe[0] + ((size_t)sel[j] * F + r) * D;
                const float *ur = fe[1] + ((size_t)sel[j] * F + r) * D;
                hh[r] = silu_f(dot_f32v(gr, x, D)) * dot_f32v(ur, x, D);
            }
            for (int i = 0; i < D; i++)
                ref[i] += wsel[j] / wsum * dot_f32v(fe[2] + ((size_t)sel[j] * D + i) * F, hh, F);
        }
        float en = 0.0f, rn = 0.0f;
        for (int i = 0; i < D; i++) { en += (out[i] - ref[i]) * (out[i] - ref[i]); rn += ref[i] * ref[i]; }
        float rel = sqrtf(en / (rn + 1e-12f));
        int same_sel = 1;
        for (int j = 0; j < NK; j++) same_sel &= mm->moe_sel[j] == sel[j];
        int ok = rc == 0 && same_sel && rel < 3e-2f;
        printf("moe top-%d of %d experts=[%d,%d,%d] rel_err=%.2e %s\n", NK, NE,
               sel[0], sel[1], sel[2], rel, ok ? "OK" : "FAIL");
        fails += !ok;
        for (int t = 0; t < 3; t++) { free(qe[t]); free(fe[t]); }
        free(mm->moe_probs); free(mm->moe_w); free(mm->moe_sel); free(mm->moe_h); free(mm->moe_xq); free(mm->moe_xk);
        free(mm);
    }

    // --- fp16 ---
    {
        int bad = 0;
        for (int i = 0; i < 65536; i++) {
            float a = g_fp16_tbl[i], b = fp16_to_float_sw((uint16_t)i);
            if (memcmp(&a, &b, 4) != 0) bad++;
        }
        printf("fp16 table: %d mismatches %s\n", bad, bad == 0 ? "OK" : "FAIL");
        if (bad) fails++;
    }

    // --- TurboQuant (quantized KV cache) ---
    {
        kvq_init();
        int hd = 128;
        float x[128], y[128], k[128], q[128], dq[128];
        uint8_t rec[4 + 128 / 2];
        uint64_t rs = 12345;
        for (int i = 0; i < hd; i++) {
            x[i] = (float)((int)(rng_next(&rs) % 2001) - 1000) / 300.0f;
            q[i] = (float)((int)(rng_next(&rs) % 2001) - 1000) / 300.0f;
        }

        // rotation and its inverse do not change the vector
        memcpy(y, x, sizeof(x));
        kvq_rotate(y, hd);
        kvq_unrotate(y, hd);
        float rerr = 0.0f, xn = 0.0f;
        for (int i = 0; i < hd; i++) {
            rerr += (y[i] - x[i]) * (y[i] - x[i]);
            xn += x[i] * x[i];
        }
        rerr = sqrtf(rerr / xn);
        printf("kvq rotate roundtrip rel_err=%.2e %s\n", rerr, rerr < 1e-5f ? "OK" : "FAIL");
        if (rerr >= 1e-5f) fails++;

        // K 4-bit: relative quantization error and preservation of q·k (after rotating both sides)
        memcpy(k, x, sizeof(x));
        kvq_rotate(k, hd);
        kvq_quant_slice(k, hd, rec, 4);
        kvq_dequant_slice(rec, dq, hd, 4);
        float qerr = 0.0f, kn = 0.0f;
        for (int i = 0; i < hd; i++) {
            qerr += (dq[i] - k[i]) * (dq[i] - k[i]);
            kn += k[i] * k[i];
        }
        qerr = sqrtf(qerr / kn);
        printf("kvq K4 quant rel_err=%.3f %s\n", qerr, qerr < 0.15f ? "OK" : "FAIL");
        if (qerr >= 0.15f) fails++;

        float qr[128];
        memcpy(qr, q, sizeof(q));
        kvq_rotate(qr, hd);
        float dot_ref = dot_f32v(q, x, hd);              // original q·k
        float dot_q = kvq_dot_k(qr, rec, hd, 4);         // on the quantized cache
        float qn = sqrtf(dot_f32v(q, q, hd)), xnn = sqrtf(xn);
        float derr = fabsf(dot_q - dot_ref) / (qn * xnn + 1e-6f);
        printf("kvq K4 dot ref=%.4f quant=%.4f rel_err=%.3f %s\n",
               dot_ref, dot_q, derr, derr < 0.05f ? "OK" : "FAIL");
        if (derr >= 0.05f) fails++;

        // V 2-bit: inherently higher quantization error but bounded
        uint8_t recv2[4 + 128 / 4];
        kvq_quant_slice(k, hd, recv2, 2);
        kvq_dequant_slice(recv2, dq, hd, 2);
        float verr = 0.0f;
        for (int i = 0; i < hd; i++) verr += (dq[i] - k[i]) * (dq[i] - k[i]);
        verr = sqrtf(verr / kn);
        printf("kvq V2 quant rel_err=%.3f %s\n", verr, verr < 0.45f ? "OK" : "FAIL");
        if (verr >= 0.45f) fails++;

        // K 8-bit (turbo8): very small int8 error
        uint8_t rec8[4 + 128];
        kvq_quant_slice(k, hd, rec8, 8);
        kvq_dequant_slice(rec8, dq, hd, 8);
        float e8 = 0.0f;
        for (int i = 0; i < hd; i++) e8 += (dq[i] - k[i]) * (dq[i] - k[i]);
        e8 = sqrtf(e8 / kn);
        float dot8 = kvq_dot_k(qr, rec8, hd, 8);
        float d8err = fabsf(dot8 - dot_ref) / (qn * xnn + 1e-6f);
        printf("kvq K8 quant rel_err=%.4f dot rel_err=%.4f %s\n", e8, d8err,
               (e8 < 0.01f && d8err < 0.005f) ? "OK" : "FAIL");
        if (e8 >= 0.01f || d8err >= 0.005f) fails++;
    }

    printf(fails == 0 ? "\nALL TESTS PASSED\n" : "\n%d TESTS FAILED\n", fails);
    return fails ? 1 : 0;
}


// =========================================================================
// Server: HTTP + OpenAI-compatible API + built-in web UI
// =========================================================================

#ifdef _WIN32
#include <winsock2.h>
typedef SOCKET sock_t;
#define sock_close closesocket
#define SOCK_INVALID INVALID_SOCKET
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
typedef int sock_t;
#define sock_close close
#define SOCK_INVALID (-1)
#endif

// --- mini JSON: extract fields from an OpenAI request ---

static const char *json_skip_ws(const char *p) {
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    return p;
}

// decode a JSON string (with escapes and \uXXXX) — returns the pointer after the closing quote, or NULL
static const char *json_string(const char *p, char *out, int cap) {
    p = json_skip_ws(p);
    if (*p != '"') return NULL;
    p++;
    int o = 0;
    while (*p && *p != '"') {
        if (o >= cap - 5) { // skip the rest safely
            if (*p == '\\' && p[1]) p++;
            p++;
            continue;
        }
        if (*p == '\\') {
            p++;
            switch (*p) {
            case 'n': out[o++] = '\n'; break;
            case 't': out[o++] = '\t'; break;
            case 'r': out[o++] = '\r'; break;
            case 'b': out[o++] = '\b'; break;
            case 'f': out[o++] = '\f'; break;
            case '"': out[o++] = '"'; break;
            case '\\': out[o++] = '\\'; break;
            case '/': out[o++] = '/'; break;
            case 'u': {
                unsigned int cp = 0;
                if (sscanf(p + 1, "%4x", &cp) == 1) {
                    p += 4;
                    // surrogate pair
                    if (cp >= 0xD800 && cp <= 0xDBFF && p[1] == '\\' && p[2] == 'u') {
                        unsigned int lo = 0;
                        if (sscanf(p + 3, "%4x", &lo) == 1 && lo >= 0xDC00 && lo <= 0xDFFF) {
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            p += 6;
                        }
                    }
                    if (cp < 0x80) out[o++] = (char)cp;
                    else if (cp < 0x800) {
                        out[o++] = (char)(0xC0 | (cp >> 6));
                        out[o++] = (char)(0x80 | (cp & 0x3F));
                    } else if (cp < 0x10000) {
                        out[o++] = (char)(0xE0 | (cp >> 12));
                        out[o++] = (char)(0x80 | ((cp >> 6) & 0x3F));
                        out[o++] = (char)(0x80 | (cp & 0x3F));
                    } else {
                        out[o++] = (char)(0xF0 | (cp >> 18));
                        out[o++] = (char)(0x80 | ((cp >> 12) & 0x3F));
                        out[o++] = (char)(0x80 | ((cp >> 6) & 0x3F));
                        out[o++] = (char)(0x80 | (cp & 0x3F));
                    }
                }
                break;
            }
            default: out[o++] = *p; break;
            }
            p++;
        } else out[o++] = *p++;
    }
    out[o] = 0;
    return *p == '"' ? p + 1 : NULL;
}

// finds the end of a JSON value (object/array/string/number) starting at p
static const char *json_value_end(const char *p) {
    p = json_skip_ws(p);
    if (*p == '"') {
        p++;
        while (*p && *p != '"') { if (*p == '\\' && p[1]) p++; p++; }
        return *p ? p + 1 : p;
    }
    if (*p == '{' || *p == '[') {
        char open = *p, close = (open == '{') ? '}' : ']';
        int depth = 0, instr = 0;
        while (*p) {
            if (instr) {
                if (*p == '\\' && p[1]) p++;
                else if (*p == '"') instr = 0;
            } else {
                if (*p == '"') instr = 1;
                else if (*p == open) depth++;
                else if (*p == close) { depth--; if (depth == 0) return p + 1; }
            }
            p++;
        }
        return p;
    }
    while (*p && *p != ',' && *p != '}' && *p != ']') p++;
    return p;
}

// looks for "field": at the current object level (without descending into sub-objects)
static const char *json_field(const char *obj, const char *field) {
    const char *p = json_skip_ws(obj);
    if (*p != '{') return NULL;
    p++;
    size_t flen = strlen(field);
    while (*p) {
        p = json_skip_ws(p);
        if (*p == '}') return NULL;
        if (*p != '"') return NULL;
        const char *key = p + 1;
        const char *ke = key;
        while (*ke && *ke != '"') { if (*ke == '\\' && ke[1]) ke++; ke++; }
        p = json_skip_ws(ke + 1);
        if (*p != ':') return NULL;
        p++;
        if ((size_t)(ke - key) == flen && strncmp(key, field, flen) == 0)
            return json_skip_ws(p);
        p = json_value_end(p);
        p = json_skip_ws(p);
        if (*p == ',') p++;
    }
    return NULL;
}

// parses the messages array — returns the message count, content is allocated dynamically
static int parse_messages(const char *body, msg_t *msgs, int max_msgs) {
    const char *arr = json_field(body, "messages");
    if (!arr || *arr != '[') return -1;
    const char *p = arr + 1;
    int n = 0;
    while (*p && n < max_msgs) {
        p = json_skip_ws(p);
        if (*p == ']') break;
        if (*p != '{') return -1;
        const char *obj_end = json_value_end(p);
        // copy the object to isolate it
        size_t olen = (size_t)(obj_end - p);
        char *obj = (char *)malloc(olen + 1);
        memcpy(obj, p, olen);
        obj[olen] = 0;

        const char *rv = json_field(obj, "role");
        const char *cv = json_field(obj, "content");
        if (rv && cv) {
            json_string(rv, msgs[n].role, sizeof(msgs[n].role));
            size_t clen = strlen(obj);
            msgs[n].content = (char *)malloc(clen + 1);
            if (!json_string(cv, msgs[n].content, (int)clen + 1))
                msgs[n].content[0] = 0;
            n++;
        }
        free(obj);
        p = json_skip_ws(obj_end);
        if (*p == ',') p++;
    }
    return n;
}

static double json_number(const char *body, const char *field, double def) {
    const char *p = json_field(body, field);
    if (!p) return def;
    return atof(p);
}

static int json_bool(const char *body, const char *field, int def) {
    const char *p = json_field(body, field);
    if (!p) return def;
    return strncmp(p, "true", 4) == 0;
}

// JSON-encode outgoing strings
static int json_escape(const char *in, int in_n, char *out, int cap) {
    int o = 0;
    for (int i = 0; i < in_n && o < cap - 8; i++) {
        unsigned char c = (unsigned char)in[i];
        switch (c) {
        case '"':  out[o++] = '\\'; out[o++] = '"'; break;
        case '\\': out[o++] = '\\'; out[o++] = '\\'; break;
        case '\n': out[o++] = '\\'; out[o++] = 'n'; break;
        case '\r': out[o++] = '\\'; out[o++] = 'r'; break;
        case '\t': out[o++] = '\\'; out[o++] = 't'; break;
        default:
            if (c < 0x20) o += snprintf(out + o, cap - o, "\\u%04x", c);
            else out[o++] = (char)c;
        }
    }
    out[o] = 0;
    return o;
}

// --- HTTP send ---

static int send_all(sock_t s, const char *buf, int n) {
    int sent = 0;
    while (sent < n) {
        int r = (int)send(s, buf + sent, n - sent, 0);
        if (r <= 0) return -1;
        sent += r;
    }
    return 0;
}

static void http_reply(sock_t s, int code, const char *ctype, const char *body, int blen) {
    char hdr[512];
    const char *st = code == 200 ? "OK" : code == 400 ? "Bad Request"
                   : code == 401 ? "Unauthorized" : code == 404 ? "Not Found" : "Error";
    int hn = snprintf(hdr, sizeof(hdr),
        "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %d\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
        "Connection: close\r\n\r\n", code, st, ctype, blen);
    if (send_all(s, hdr, hn) == 0 && blen > 0) send_all(s, body, blen);
}

// --- built-in UI (v2: model and mode selection) ---
static const char INDEX_HTML[] =
"<!DOCTYPE html><html><head><meta charset=utf-8>"
"<meta name=viewport content='width=device-width,initial-scale=1'>"
"<title>fastgguf</title><style>"
"*{box-sizing:border-box;margin:0}body{font-family:system-ui,'Segoe UI',sans-serif;"
"background:#0f1117;color:#e2e4ea;height:100vh;display:flex;flex-direction:column}"
"header{padding:12px 20px;background:#161a24;border-bottom:1px solid #262b38;"
"font-weight:600;display:flex;justify-content:space-between;align-items:center}"
"header span{color:#7a8399;font-size:13px;font-weight:400}"
"select{background:#0f1117;color:#e2e4ea;border:1px solid #2c3242;border-radius:8px;"
"padding:5px 8px;font-size:13px;outline:none}select:disabled{opacity:.45}"
"#ctl{display:flex;gap:8px;align-items:center}"
"#log{flex:1;overflow-y:auto;padding:20px;display:flex;flex-direction:column;gap:12px}"
".m{max-width:78%;padding:10px 14px;border-radius:14px;white-space:pre-wrap;"
"word-break:break-word;line-height:1.55;unicode-bidi:plaintext}"
".u{align-self:flex-end;background:#2b5cd9;border-bottom-right-radius:4px}"
".a{align-self:flex-start;background:#1c212e;border-bottom-left-radius:4px}"
"footer{padding:14px 20px;background:#161a24;border-top:1px solid #262b38;display:flex;gap:10px}"
"#in{flex:1;background:#0f1117;border:1px solid #2c3242;border-radius:10px;color:#e2e4ea;"
"padding:11px 14px;font-size:15px;outline:none}#in:focus{border-color:#2b5cd9}"
"button{background:#2b5cd9;border:none;border-radius:10px;color:#fff;padding:0 22px;"
"font-size:15px;cursor:pointer}button:disabled{opacity:.45}"
"</style></head><body>"
"<header>fastgguf<div id=ctl>"
"<select id=md><option value=single>single</option>"
"<option value=parallel>parallel</option><option value=chain>chain</option>"
"<option value=loop>loop</option></select>"
"<input id=rn type=number value=2 min=1 max=8 title=rounds "
"style='width:52px;display:none;background:#0f1117;color:#e2e4ea;"
"border:1px solid #2c3242;border-radius:8px;padding:5px 8px;font-size:13px'>"
"<select id=mo></select><span id=st>ready</span></div></header>"
"<div id=log></div>"
"<footer><input id=in placeholder='اكتب رسالتك... / Type a message' autofocus>"
"<button id=btn>&#10148;</button></footer><script>"
"const log=document.getElementById('log'),inp=document.getElementById('in'),"
"btn=document.getElementById('btn'),st=document.getElementById('st'),"
"mo=document.getElementById('mo'),md=document.getElementById('md');let hist=[];"
"fetch('/v1/models').then(r=>r.json()).then(j=>{j.data.forEach(m=>{"
"const o=document.createElement('option');o.value=m.id;o.textContent=m.id;mo.appendChild(o)});"
"if(j.data.length<2){mo.style.display='none';"
"for(const o of md.options)if(o.value==='parallel'||o.value==='chain')o.disabled=true}}).catch(()=>{});"
"md.onchange=()=>{mo.disabled=md.value!=='single';"
"document.getElementById('rn').style.display=md.value==='loop'?'':'none'};"
"function add(cls,txt){const d=document.createElement('div');d.className='m '+cls;"
"d.textContent=txt;log.appendChild(d);log.scrollTop=log.scrollHeight;return d}"
"async function go(){const q=inp.value.trim();if(!q)return;inp.value='';btn.disabled=true;"
"add('u',q);hist.push({role:'user',content:q});const el=add('a','');st.textContent='thinking...';"
"try{const r=await fetch('/v1/chat/completions',{method:'POST',"
"headers:{'Content-Type':'application/json'},"
"body:JSON.stringify({messages:hist,stream:true,max_tokens:1024,"
"model:mo.value,mode:md.value,rounds:+document.getElementById('rn').value||2})});"
"const rd=r.body.getReader(),dec=new TextDecoder();let buf='',out='';"
"for(;;){const{done,value}=await rd.read();if(done)break;buf+=dec.decode(value,{stream:true});"
"let i;while((i=buf.indexOf('\\n\\n'))>=0){const ln=buf.slice(0,i).trim();buf=buf.slice(i+2);"
"if(!ln.startsWith('data:'))continue;const d=ln.slice(5).trim();if(d==='[DONE]')continue;"
"try{const j=JSON.parse(d);const c=j.choices[0].delta.content;"
"if(c){out+=c;el.textContent=out;log.scrollTop=log.scrollHeight}}catch(e){}}}"
"hist.push({role:'assistant',content:out});st.textContent='ready'}"
"catch(e){el.textContent='[error] '+e;st.textContent='error'}btn.disabled=false;inp.focus()}"
"btn.onclick=go;inp.onkeydown=e=>{if(e.key==='Enter')go()};"
"</script></body></html>";

// =========================================================================
// v2: several models at once — an independent engine per model (gguf + vocab + own cache)
// =========================================================================

#define MAX_MODELS 4
#define MAX_STAGES (MAX_MODELS * 8)   // limit on chain/loop stages

typedef struct {
    gguf_t g;
    tok_t  tk;
    model_t *m;
    char name[128];
} engine_t;

static char *dupstr(const char *s) {
    size_t n = strlen(s);
    char *d = (char *)malloc(n + 1);
    memcpy(d, s, n + 1);
    return d;
}

// model name = file name without the path and without .gguf
static void path_model_name(const char *path, char *out, int cap) {
    const char *b = path;
    for (const char *p = path; *p; p++)
        if (*p == '/' || *p == '\\') b = p + 1;
    int n = (int)strlen(b);
    if (n > 5 && strcmp(b + n - 5, ".gguf") == 0) n -= 5;
    if (n >= cap) n = cap - 1;
    memcpy(out, b, n);
    out[n] = 0;
}

static int engine_load(engine_t *e, const char *path, int n_ctx) {
    if (gguf_open(&e->g, path) != 0) return 1;
    if (tok_load(&e->tk, &e->g) != 0) return 1;
    e->m = (model_t *)calloc(1, sizeof(model_t));
    if (model_load(e->m, &e->g, n_ctx) != 0) return 1;
    path_model_name(path, e->name, sizeof(e->name));
    return 0;
}

// --- run one engine on a conversation (shared by the single/parallel/chain modes) ---
// builds the prompt (dropping the oldest turns when the context overflows), reuses the prefix cache,
// generates, and returns the text (malloc — the caller frees it). Does not free msgs.
// valid stream_sock: streams SSE chunks; non-NULL label: streams a section header before the tokens.
// status: 0 success | 1 context does not fit | 2 forward failed | 3 client disconnected
static char *run_engine(engine_t *e, msg_t *msgs, int n_msgs, int max_tokens,
                        sock_t stream_sock, const char *label, sampler_t *sp,
                        int *status, int *out_prompt_toks, int *out_gen_toks) {
    model_t *m = e->m;
    tok_t *tk = &e->tk;
    int stream = (stream_sock != SOCK_INVALID);
    *status = 0; *out_prompt_toks = 0; *out_gen_toks = 0;
    if (max_tokens < 1) max_tokens = 1;
    sp->cand = m->cand;

    // build the prompt — when the context overflows, drop the oldest turns progressively
    // (always keep the first system message and the last message, and reserve room for generation)
    int keep_sys = strcmp(msgs[0].role, "system") == 0;
    int first = keep_sys ? 1 : 0;
    int dropped = 0;
    int reserve = max_tokens < 256 ? max_tokens : 256;
    int *ids = NULL, n_ids = 0;
    for (;;) {
        msg_t tmp[66];
        int nt = 0;
        if (keep_sys) tmp[nt++] = msgs[0];
        for (int i = first; i < n_msgs; i++) tmp[nt++] = msgs[i];
        ids = encode_chat(tk, tmp, nt, tk->add_bos, 0, &n_ids);
        if (n_ids + reserve <= m->n_ctx - 8) break;
        if (first < n_msgs - 1) {
            free(ids); ids = NULL;
            int step = (n_msgs - 1 - first) >= 2 ? 2 : 1;
            first += step; dropped += step;
            continue;
        }
        if (n_ids < m->n_ctx - 8) break;
        free(ids); ids = NULL;
        break;
    }
    if (!ids) { *status = 1; return NULL; }
    if (dropped)
        printf("[%s] context overflow: dropped %d oldest msgs\n", e->name, dropped);
    if (n_ids + max_tokens > m->n_ctx) max_tokens = m->n_ctx - n_ids;

    // cache reuse: longest common prefix with the previous request's sequence
    int keep = 0;
    {
        int lim = n_ids - 1;
        if (lim > m->seq_len) lim = m->seq_len;
        while (keep < lim && ids[keep] == m->seq_ids[keep]) keep++;
    }
    // SWA layers are ring buffers: the window may have moved — if it is no longer resident, re-ingest from scratch
    if (keep > 0 && m->n_swa > 0) {
        for (int il = 0; il < m->n_kv_from_start; il++) {
            if (!m->is_swa[il]) continue;
            int oldest_needed = keep + 1 - m->n_swa;
            if (oldest_needed < 0) oldest_needed = 0;
            int oldest_resident = m->seq_len - m->cache_cap[il];
            if (oldest_resident < 0) oldest_resident = 0;
            if (oldest_needed < oldest_resident) keep = 0;
            break;
        }
    }

    printf("[%s] %d msgs, %d prompt tokens (%d cached), max_gen=%d, stream=%d\n",
           e->name, n_msgs, n_ids, keep, max_tokens, stream);

    char tbuf[512], ebuf[1200], chunk[1600];

    // section header (parallel/chain) — streamed immediately so the user knows who is answering
    if (stream && label) {
        char hdr[192];
        int hn = snprintf(hdr, sizeof(hdr), "\n### %s\n", label);
        json_escape(hdr, hn, ebuf, sizeof(ebuf));
        int cn = snprintf(chunk, sizeof(chunk),
            "data: {\"id\":\"chatcmpl-fg\",\"object\":\"chat.completion.chunk\",\"model\":\"%s\","
            "\"choices\":[{\"index\":0,\"delta\":{\"content\":\"%s\"},"
            "\"finish_reason\":null}]}\n\n", e->name, ebuf);
        if (send_all(stream_sock, chunk, cn) != 0) { free(ids); *status = 3; return NULL; }
    }

    // ingest the prompt from the point of divergence
    int pos = keep;
    float *logits = NULL;
    m->seq_len = keep;
    double t0 = now_sec();
    for (int i = keep; i < n_ids; i++) {
        logits = forward(m, ids[i], pos);
        if (!logits) break;
        m->seq_ids[m->seq_len++] = ids[i];
        pos++;
    }
    free(ids);
    if (!logits) { *status = 2; return NULL; }
    double t_prompt = now_sec() - t0;

    size_t out_cap = 65536, out_n = 0;
    char *out_txt = (char *)malloc(out_cap);
    int n_gen = 0;
    double t1 = now_sec();

    for (int i = 0; i < max_tokens; i++) {
        int next = sample_next(sp, tk, logits, m->n_vocab, m->seq_ids, m->seq_len);
        if (tok_is_stop(tk, next)) break;
        int tn = tok_text(tk, next, tbuf, sizeof(tbuf));
        if (tn > 0) {
            if (out_n + tn + 1 > out_cap) { out_cap *= 2; out_txt = (char *)realloc(out_txt, out_cap); }
            memcpy(out_txt + out_n, tbuf, tn);
            out_n += tn;
            if (stream) {
                json_escape(tbuf, tn, ebuf, sizeof(ebuf));
                int cn = snprintf(chunk, sizeof(chunk),
                    "data: {\"id\":\"chatcmpl-fg\",\"object\":\"chat.completion.chunk\",\"model\":\"%s\","
                    "\"choices\":[{\"index\":0,\"delta\":{\"content\":\"%s\"},"
                    "\"finish_reason\":null}]}\n\n", e->name, ebuf);
                if (send_all(stream_sock, chunk, cn) != 0) { // client disconnected
                    free(out_txt); *status = 3; return NULL;
                }
            }
        }
        n_gen++;
        logits = forward(m, next, pos);
        if (!logits) break;
        m->seq_ids[m->seq_len++] = next;
        pos++;
    }
    double t_gen = now_sec() - t1;
    printf("[%s] done: prompt %.1fs (%.1f t/s, %d reused) | gen %d tok @ %.2f t/s\n",
           e->name, t_prompt, (n_ids - keep) / (t_prompt > 0 ? t_prompt : 1), keep,
           n_gen, n_gen / (t_gen > 0 ? t_gen : 1));

    out_txt[out_n] = 0;
    *out_prompt_toks = n_ids;
    *out_gen_toks = n_gen;
    return out_txt;
}

// --- /v1/chat/completions: single mode, parallel (all models) or chain (layers) ---

// refinement message between chain stages
static const char CHAIN_REFINE_MSG[] =
    "The assistant message above is a draft answer to my request. "
    "Improve it: fix any mistakes, fill in anything missing, and reply with only the final answer.";

static void handle_chat_multi(sock_t s, engine_t *engs, int n_eng, const char *body) {
    msg_t msgs[64];
    int n_msgs = parse_messages(body, msgs, 64);
    if (n_msgs <= 0) {
        http_reply(s, 400, "application/json",
                   "{\"error\":\"missing or invalid messages\"}", 39);
        return;
    }
    int max_tokens = (int)json_number(body, "max_tokens", 1024);
    if (max_tokens < 1) max_tokens = 1;
    int stream = json_bool(body, "stream", 0);

    // sampling: temperature<=0 (the default) means greedy — any request that does not pass temperature
    // gets exactly the same deterministic behaviour as before
    sampler_t smp;
    smp.temp    = (float)json_number(body, "temperature", 0.0);
    smp.top_p   = (float)json_number(body, "top_p", 0.95);
    smp.top_k   = (int)json_number(body, "top_k", 40);
    smp.repeat_penalty = (float)json_number(body, "repeat_penalty", 1.1);
    smp.repeat_last_n  = (int)json_number(body, "repeat_last_n", 64);
    smp.cand    = NULL;   // run_engine sets the selected model's workspace
    if (smp.temp < 0.0f) smp.temp = 0.0f;
    double seed = json_number(body, "seed", -1.0);
    if (seed >= 0.0) {
        smp.rng = (uint64_t)seed;
    } else {
#ifdef _WIN32
        LARGE_INTEGER ctr; QueryPerformanceCounter(&ctr);
        smp.rng = (uint64_t)ctr.QuadPart;
#else
        struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
        smp.rng = (uint64_t)ts.tv_nsec ^ ((uint64_t)ts.tv_sec << 32);
#endif
    }
    if (smp.rng == 0) smp.rng = 0x9E3779B97F4A7C15ULL;

    char want[128] = {0}, mode[32] = {0};
    const char *f;
    if ((f = json_field(body, "model")) != NULL && *f == '"') json_string(f, want, sizeof(want));
    if ((f = json_field(body, "mode"))  != NULL && *f == '"') json_string(f, mode, sizeof(mode));
    // mode via the model field too — for OpenAI clients that cannot pass extra fields
    if (!mode[0] && (strcmp(want, "parallel") == 0 || strcmp(want, "chain") == 0 ||
                     strcmp(want, "layers") == 0 || strcmp(want, "loop") == 0)) {
        strcpy(mode, want);
        want[0] = 0;
    }
    int m_par = (strcmp(mode, "parallel") == 0) && n_eng > 1;
    int m_chn = (strcmp(mode, "chain") == 0 || strcmp(mode, "layers") == 0 ||
                 strcmp(mode, "pipeline") == 0) && n_eng > 1;
    // loop: the refinement chain repeated for several rounds — works even with one model (self-refinement)
    int m_loop = (strcmp(mode, "loop") == 0) && n_eng >= 1;
    int rounds = (int)json_number(body, "rounds", 2);
    if (rounds < 1) rounds = 1;
    if (rounds > 8) rounds = 8;

    // engine selection for single mode: match the name, otherwise the first one (for OpenAI client compatibility)
    engine_t *sel = &engs[0];
    if (want[0]) {
        for (int i = 0; i < n_eng; i++)
            if (strcmp(engs[i].name, want) == 0 || strstr(engs[i].name, want) != NULL) {
                sel = &engs[i];
                break;
            }
    }

    char *texts[MAX_STAGES] = {0};
    const char *tnames[MAX_STAGES] = {0};
    int n_texts = 0, sum_p = 0, sum_g = 0;
    int status = 0, pt = 0, gt = 0;
    const char *resp_model = "fastgguf";
    sock_t ss = SOCK_INVALID;

    if (stream) {
        const char *hdr =
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
            "Cache-Control: no-cache\r\nAccess-Control-Allow-Origin: *\r\n"
            "Connection: close\r\n\r\n";
        if (send_all(s, hdr, (int)strlen(hdr)) != 0) goto done_free;
        ss = s;
    }

    if (m_par) {
        // every model answers the same question — run sequentially (the CPU is fully busy for each model)
        resp_model = "parallel";
        for (int i = 0; i < n_eng; i++) {
            char *t = run_engine(&engs[i], msgs, n_msgs, max_tokens, ss, engs[i].name, &smp,
                                 &status, &pt, &gt);
            if (!t) {
                if (status == 3) goto done_free;
                t = dupstr("[error: prompt too long or generation failed]");
            }
            texts[n_texts] = t;
            tnames[n_texts] = engs[i].name;
            n_texts++;
            if (pt > 0) sum_p += pt;
            if (gt > 0) sum_g += gt;
        }
    } else if (m_chn || m_loop) {
        // layers: a draft then sequential refinement — loop repeats the chain for several rounds
        resp_model = m_loop ? "loop" : "chain";
        int n_stages = m_loop ? n_eng * rounds : n_eng;
        if (n_stages > MAX_STAGES) n_stages = MAX_STAGES;
        char *draft = NULL;
        for (int i = 0; i < n_stages; i++) {
            engine_t *e = &engs[i % n_eng];
            msg_t cm[66];
            int cn = n_msgs;
            memcpy(cm, msgs, (size_t)n_msgs * sizeof(msg_t));
            if (draft) {
                strcpy(cm[cn].role, "assistant"); cm[cn].content = draft; cn++;
                strcpy(cm[cn].role, "user");      cm[cn].content = (char *)CHAIN_REFINE_MSG; cn++;
            }
            char label[192];
            snprintf(label, sizeof(label), "%s%s", e->name,
                     i == n_stages - 1 ? " (final)" : (i ? " (refine)" : " (draft)"));
            char *t = run_engine(e, cm, cn, max_tokens, ss, label, &smp, &status, &pt, &gt);
            if (!t) {
                if (status == 3) goto done_free;
                break; // keep the last successful draft
            }
            texts[n_texts] = t;
            tnames[n_texts] = e->name;
            n_texts++;
            sum_p += pt; sum_g += gt;
            // an empty draft does not replace the previous one — the next stage refines the last real text
            {
                const char *q = t;
                while (*q == ' ' || *q == '\n' || *q == '\r' || *q == '\t') q++;
                if (*q) draft = t;
            }
        }
        if (!n_texts) {
            if (!stream)
                http_reply(s, 500, "application/json", "{\"error\":\"generation failed\"}", 29);
            goto done_free;
        }
    } else {
        char *t = run_engine(sel, msgs, n_msgs, max_tokens, ss,
                             NULL, &smp, &status, &pt, &gt);
        if (!t) {
            if (!stream) {
                if (status == 1)
                    http_reply(s, 400, "application/json",
                               "{\"error\":\"prompt too long for context\"}", 40);
                else
                    http_reply(s, 500, "application/json", "{\"error\":\"forward failed\"}", 26);
            }
            goto done_free;
        }
        texts[n_texts] = t;
        tnames[n_texts] = sel->name;
        n_texts++;
        sum_p = pt; sum_g = gt;
        resp_model = sel->name;
    }

    if (stream) {
        const char *done =
            "data: {\"id\":\"chatcmpl-fg\",\"object\":\"chat.completion.chunk\","
            "\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"
            "data: [DONE]\n\n";
        send_all(s, done, (int)strlen(done));
    } else {
        size_t rcap = 1024;
        for (int i = 0; i < n_texts; i++) rcap += strlen(texts[i]) * 6 + 512;
        char *resp = (char *)malloc(rcap);
        char *esc  = (char *)malloc(rcap);
        int rn = 0;
        if (m_par) {
            rn += snprintf(resp + rn, rcap - rn,
                "{\"id\":\"chatcmpl-fg\",\"object\":\"chat.completion\",\"model\":\"parallel\",\"choices\":[");
            for (int i = 0; i < n_texts; i++) {
                json_escape(texts[i], (int)strlen(texts[i]), esc, (int)rcap);
                rn += snprintf(resp + rn, rcap - rn,
                    "%s{\"index\":%d,\"model\":\"%s\",\"message\":{\"role\":\"assistant\","
                    "\"content\":\"%s\"},\"finish_reason\":\"stop\"}",
                    i ? "," : "", i, tnames[i], esc);
            }
            rn += snprintf(resp + rn, rcap - rn,
                "],\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d}}",
                sum_p, sum_g, sum_p + sum_g);
        } else if (m_chn || m_loop) {
            rn += snprintf(resp + rn, rcap - rn,
                "{\"id\":\"chatcmpl-fg\",\"object\":\"chat.completion\",\"model\":\"%s\",\"stages\":[",
                resp_model);
            for (int i = 0; i < n_texts; i++) {
                json_escape(texts[i], (int)strlen(texts[i]), esc, (int)rcap);
                rn += snprintf(resp + rn, rcap - rn,
                    "%s{\"model\":\"%s\",\"content\":\"%s\"}", i ? "," : "", tnames[i], esc);
            }
            // final answer = the last non-empty stage (a weak model may return empty text)
            int fin = n_texts - 1;
            while (fin > 0) {
                const char *q = texts[fin];
                while (*q == ' ' || *q == '\n' || *q == '\r' || *q == '\t') q++;
                if (*q) break;
                fin--;
            }
            json_escape(texts[fin], (int)strlen(texts[fin]), esc, (int)rcap);
            rn += snprintf(resp + rn, rcap - rn,
                "],\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"%s\"},"
                "\"finish_reason\":\"stop\"}],"
                "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d}}",
                esc, sum_p, sum_g, sum_p + sum_g);
        } else {
            json_escape(texts[0], (int)strlen(texts[0]), esc, (int)rcap);
            rn = snprintf(resp, rcap,
                "{\"id\":\"chatcmpl-fg\",\"object\":\"chat.completion\",\"model\":\"%s\","
                "\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"%s\"},"
                "\"finish_reason\":\"stop\"}],"
                "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d}}",
                resp_model, esc, sum_p, sum_g, sum_p + sum_g);
        }
        http_reply(s, 200, "application/json", resp, rn);
        free(resp);
        free(esc);
    }

done_free:
    for (int i = 0; i < n_texts; i++) free(texts[i]);
    for (int i = 0; i < n_msgs; i++) free(msgs[i].content);
}

// --- server loop (multiple models) ---
static int cmd_serve_multi(char **paths, int n_models, int port, int n_ctx) {
#ifdef _WIN32
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) { fprintf(stderr, "WSAStartup failed\n"); return 2; }
#endif
    static engine_t engs[MAX_MODELS];
    if (n_models > MAX_MODELS) n_models = MAX_MODELS;
    for (int i = 0; i < n_models; i++) {
        printf("loading model %d/%d: %s\n", i + 1, n_models, paths[i]);
        if (engine_load(&engs[i], paths[i], n_ctx) != 0) return 2;
        // duplicate names — add a numeric suffix
        for (int j = 0; j < i; j++)
            if (strcmp(engs[j].name, engs[i].name) == 0) {
                size_t L = strlen(engs[i].name);
                snprintf(engs[i].name + L, sizeof(engs[i].name) - L, "-%d", i + 1);
                break;
            }
    }

    setvbuf(stdout, NULL, _IONBF, 0); // immediate logs
    const char *api_key = getenv("FASTGGUF_API_KEY"); // optional

    sock_t srv = socket(AF_INET, SOCK_STREAM, 0);
    if (srv == SOCK_INVALID) { fprintf(stderr, "socket failed\n"); return 2; }
    int opt = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, (const char *)&opt, sizeof(opt));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;   // 0.0.0.0 — reachable from the whole network
    addr.sin_port = htons((uint16_t)port);
    if (bind(srv, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        fprintf(stderr, "bind failed on port %d (already in use?)\n", port);
        return 2;
    }
    listen(srv, 8);

    printf("\n=== fastgguf server (multi-model) ===\n");
    printf("  models :");
    for (int i = 0; i < n_models; i++) printf(" [%s]", engs[i].name);
    printf("\n");
    printf("  web UI : http://localhost:%d/\n", port);
    printf("  API    : http://<your-ip>:%d/v1/chat/completions (OpenAI-compatible)\n", port);
    printf("  modes  : \"model\":\"<name>\" | \"mode\":\"parallel\" | \"mode\":\"chain\""
           " | \"mode\":\"loop\" (+\"rounds\":N)\n");
    printf("  auth   : %s\n", api_key ? "API key required (FASTGGUF_API_KEY)" : "open (set FASTGGUF_API_KEY to protect)");
    printf("  ctx    : %d tokens/model | Ctrl+C to stop\n\n", n_ctx);

    // model list JSON (constant after loading)
    char models_json[1024];
    int mj_n = snprintf(models_json, sizeof(models_json), "{\"object\":\"list\",\"data\":[");
    for (int i = 0; i < n_models; i++)
        mj_n += snprintf(models_json + mj_n, sizeof(models_json) - mj_n,
                         "%s{\"id\":\"%s\",\"object\":\"model\"}", i ? "," : "", engs[i].name);
    mj_n += snprintf(models_json + mj_n, sizeof(models_json) - mj_n, "]}");

    size_t req_cap = 1 << 20; // 1MB request size limit
    char *req = (char *)malloc(req_cap);

    for (;;) {
        sock_t c = accept(srv, NULL, NULL);
        if (c == SOCK_INVALID) continue;
        opt = 1;
        setsockopt(c, IPPROTO_TCP, TCP_NODELAY, (const char *)&opt, sizeof(opt));

        // read the headers then the body according to Content-Length
        size_t got = 0;
        char *body = NULL;
        size_t need_total = 0;
        while (got < req_cap - 1) {
            int r = (int)recv(c, req + got, (int)(req_cap - 1 - got), 0);
            if (r <= 0) break;
            got += r;
            req[got] = 0;
            if (!body) {
                char *hdr_end = strstr(req, "\r\n\r\n");
                if (hdr_end) {
                    body = hdr_end + 4;
                    size_t clen = 0;
                    char *cl = strstr(req, "Content-Length:");
                    if (!cl) cl = strstr(req, "content-length:");
                    if (cl) clen = (size_t)atol(cl + 15);
                    need_total = (size_t)(body - req) + clen;
                    if (need_total >= req_cap) break;
                }
            }
            if (body && got >= need_total) break;
        }
        if (got == 0) { sock_close(c); continue; }
        req[got] = 0;

        // request line
        char method[8] = {0}, url[256] = {0};
        sscanf(req, "%7s %255s", method, url);

        if (strcmp(method, "OPTIONS") == 0) {
            http_reply(c, 200, "text/plain", "", 0);
        } else if (strcmp(method, "GET") == 0 && strcmp(url, "/") == 0) {
            http_reply(c, 200, "text/html; charset=utf-8", INDEX_HTML, (int)sizeof(INDEX_HTML) - 1);
        } else if (strcmp(method, "GET") == 0 && strcmp(url, "/v1/models") == 0) {
            http_reply(c, 200, "application/json", models_json, mj_n);
        } else if (strcmp(method, "POST") == 0 && strcmp(url, "/v1/chat/completions") == 0 && body) {
            // check the key if enabled
            if (api_key) {
                char expect[256];
                snprintf(expect, sizeof(expect), "Bearer %s", api_key);
                if (!strstr(req, expect)) {
                    http_reply(c, 401, "application/json", "{\"error\":\"invalid api key\"}", 27);
                    sock_close(c);
                    continue;
                }
            }
            handle_chat_multi(c, engs, n_models, body);
        } else {
            http_reply(c, 404, "application/json", "{\"error\":\"not found\"}", 21);
        }
        sock_close(c);
    }
    return 0;
}

static void common_init(int *argc, char ***argv) {
#ifdef _WIN32
#ifdef PROCESS_POWER_THROTTLING_EXECUTION_SPEED
    // Windows 11 puts a process that waits a long time (the server in accept) into efficiency mode
    // (EcoQoS), especially on battery — slowing compute ~3x. Explicitly ask not to be throttled
    PROCESS_POWER_THROTTLING_STATE pts;
    memset(&pts, 0, sizeof(pts));
    pts.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
    pts.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
    pts.StateMask = 0;
    SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &pts, sizeof(pts));
#endif
    SetConsoleOutputCP(CP_UTF8); // print UTF-8 (e.g. Arabic) correctly
    char **u8argv = utf8_argv(argc);
    if (u8argv) *argv = u8argv;
#else
    (void)argc; (void)argv;
#endif
    fp16_init();
}

// chat options after the path: positional [ctx] then --temp/--top-k/... — returns 0, or 1 on error
MAYBE_UNUSED static int parse_chat_args(char **argv, int from, int argc, int *ctx, int *max_tokens, sampler_t *smp) {
    for (int i = from; i < argc; i++) {
        const char *a = argv[i];
        if (a[0] != '-') { *ctx = atoi(a); continue; } // legacy positional [ctx]
        if (i + 1 >= argc) { printf("missing value for %s\n", a); return 1; }
        const char *v = argv[++i];
        if      (strcmp(a, "--temp") == 0)           smp->temp = (float)atof(v);
        else if (strcmp(a, "--top-k") == 0)          smp->top_k = atoi(v);
        else if (strcmp(a, "--top-p") == 0)          smp->top_p = (float)atof(v);
        else if (strcmp(a, "--repeat-penalty") == 0) smp->repeat_penalty = (float)atof(v);
        else if (strcmp(a, "--repeat-last-n") == 0)  smp->repeat_last_n = atoi(v);
        else if (strcmp(a, "--max-tokens") == 0)     *max_tokens = atoi(v);
        else if (strcmp(a, "--seed") == 0)           smp->rng = strtoull(v, NULL, 10);
        else { printf("unknown option %s\n", a); return 1; }
    }
    if (*ctx < 64) *ctx = 64;
    return 0;
}

#define CHAT_OPTIONS_HELP \
    "chat options: --temp 1.0 (0 = greedy)  --top-k 64 (0 = off)  --top-p 0.95\n" \
    "              --repeat-penalty 1.0 (1 = off)  --repeat-last-n 64\n" \
    "              --max-tokens 512 (0 = until stop)  --seed N\n"

// defaults = Google's recommendation for Gemma (temp 1.0, top_k 64, top_p 0.95)
#define CHAT_SAMPLER_DEFAULTS { 1.0f, 0.95f, 1.0f, 64, 64, 0, NULL }

// sorts the arguments: numbers = port then context, everything else = model paths (up to MAX_MODELS)
static int parse_serve_args(char **argv, int from, int argc,
                            char **paths, int *port, int *ctx) {
    int n_models = 0, n_num = 0;
    for (int i = from; i < argc; i++) {
        char *end;
        long v = strtol(argv[i], &end, 10);
        if (argv[i][0] && *end == 0) {
            if (n_num++ == 0) *port = (int)v; else *ctx = (int)v;
        } else if (n_models < MAX_MODELS) {
            paths[n_models++] = argv[i];
        }
    }
    if (*ctx < 64) *ctx = 64;
    return n_models;
}

#if defined(FG_SERVER_MAIN)

// server-only build: fastgguf-server.exe <model.gguf> [model2.gguf ...] [port] [ctx]
int main(int argc, char **argv) {
    common_init(&argc, &argv);
    char *paths[MAX_MODELS];
    int port = 8080, ctx = 4096;
    if (argc == 2 && strcmp(argv[1], "gpu") == 0) return cmd_gpuinfo();
    int n_models = parse_serve_args(argv, 1, argc, paths, &port, &ctx);
    if (n_models == 0) {
        printf("Usage: fastgguf-server <model.gguf> [model2.gguf ... up to %d] [port] [ctx]\n"
               "       fastgguf-server gpu            list GPUs / check OpenCL support\n"
               "  defaults: port 8080, ctx 4096\n"
               "  GPU: uses a discrete NVIDIA/AMD GPU automatically when present (OpenCL).\n"
               "       models bigger than VRAM are split: first layers on GPU, rest on CPU.\n"
               "       FASTGGUF_GPU=on|off|auto  FASTGGUF_GPU_DEVICE=<n>\n"
               "       FASTGGUF_GPU_MEM=<GB>     cap GPU weight memory (e.g. 14)\n"
               "       FASTGGUF_GPU_LAYERS=<n>   offload exactly n layers\n"
               "  KV:  FASTGGUF_KV=turbo   TurboQuant KV cache ~7x smaller (K4+V4; Qwen auto K8)\n"
               "       FASTGGUF_KV=turbo8|turbo4|turbo2   force K8+V4 / K4+V4 / K4+V2 (max squeeze)\n"
               "  multi-model request fields:\n"
               "    \"model\":\"<name>\"    route to one model\n"
               "    \"mode\":\"parallel\"   every model answers the same prompt\n"
               "    \"mode\":\"chain\"      layered: each model refines the previous answer\n"
               "    \"mode\":\"loop\"       chain repeated for N rounds (\"rounds\":2, works with 1 model)\n",
               MAX_MODELS);
        return 1;
    }
    return cmd_serve_multi(paths, n_models, port, ctx);
}

#elif defined(FG_CONSOLE_MAIN)

// console build: fastgguf-console.exe <model.gguf> [ctx]  (or inspect/dump/selftest)
int main(int argc, char **argv) {
    common_init(&argc, &argv);
    if (argc < 2) {
        printf("Usage: fastgguf-console <model.gguf> [ctx] [options]   terminal chat\n"
               "       fastgguf-console inspect <model.gguf>\n"
               "       fastgguf-console dump    <model.gguf>\n"
               "       fastgguf-console selftest\n"
               "       fastgguf-console gpu                    list GPUs / check OpenCL support\n"
               "  GPU: uses a discrete NVIDIA/AMD GPU automatically when present (OpenCL).\n"
               "       models bigger than VRAM are split: first layers on GPU, rest on CPU.\n"
               "       FASTGGUF_GPU=on|off|auto  FASTGGUF_GPU_DEVICE=<n>\n"
               "       FASTGGUF_GPU_MEM=<GB>     cap GPU weight memory (e.g. 14)\n"
               "       FASTGGUF_GPU_LAYERS=<n>   offload exactly n layers\n"
               "  KV:  FASTGGUF_KV=turbo   TurboQuant KV cache ~7x smaller (K4+V4; Qwen auto K8)\n"
               "       FASTGGUF_KV=turbo8|turbo4|turbo2   force K8+V4 / K4+V4 / K4+V2 (max squeeze)\n"
               CHAT_OPTIONS_HELP);
        return 1;
    }
    if (strcmp(argv[1], "selftest") == 0) return cmd_selftest();
    if (strcmp(argv[1], "gpu") == 0) return cmd_gpuinfo();
    if (strcmp(argv[1], "inspect") == 0 && argc > 2) return cmd_inspect(argv[2]);
    if (strcmp(argv[1], "dump") == 0 && argc > 2) return cmd_dump(argv[2]);
    int ctx = 2048, max_tokens = 512;
    sampler_t smp = CHAT_SAMPLER_DEFAULTS;
    if (parse_chat_args(argv, 2, argc, &ctx, &max_tokens, &smp) != 0) return 1;
    return cmd_chat(argv[1], ctx, max_tokens, &smp);
}

#else

int main(int argc, char **argv) {
    common_init(&argc, &argv);

    if (argc < 2) {
        printf("Usage: fastgguf <chat|serve|inspect|dump|gpu|selftest> <model.gguf> [...]\n"
               "  chat:  fastgguf chat model.gguf [ctx] [options]\n"
               "  serve: fastgguf serve model1.gguf [model2.gguf ...] [port 8080] [ctx 4096]\n"
               CHAT_OPTIONS_HELP
               "  GPU: FASTGGUF_GPU=on|off|auto  FASTGGUF_GPU_DEVICE=<n>  FASTGGUF_GPU_MEM=<GB>\n"
               "       FASTGGUF_GPU_LAYERS=<n>\n"
               "  KV:  FASTGGUF_KV=turbo|turbo8|turbo4|turbo2   TurboQuant KV cache\n");
        return 1;
    }
    if (strcmp(argv[1], "selftest") == 0) return cmd_selftest();
    if (strcmp(argv[1], "gpu") == 0) return cmd_gpuinfo();
    if (argc < 3) { printf("missing model path\n"); return 1; }

    if (strcmp(argv[1], "chat") == 0) {
        int ctx = 2048, max_tokens = 512;
        sampler_t smp = CHAT_SAMPLER_DEFAULTS;
        if (parse_chat_args(argv, 3, argc, &ctx, &max_tokens, &smp) != 0) return 1;
        return cmd_chat(argv[2], ctx, max_tokens, &smp);
    }
    if (strcmp(argv[1], "serve") == 0) {
        char *paths[MAX_MODELS];
        int port = 8080, ctx = 4096;
        int n_models = parse_serve_args(argv, 2, argc, paths, &port, &ctx);
        if (n_models == 0) { printf("missing model path\n"); return 1; }
        return cmd_serve_multi(paths, n_models, port, ctx);
    }
    if (strcmp(argv[1], "inspect") == 0) return cmd_inspect(argv[2]);
    if (strcmp(argv[1], "dump") == 0) return cmd_dump(argv[2]);
    printf("unknown command %s\n", argv[1]);
    return 1;
}

#endif
