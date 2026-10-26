// Start-up dispatcher for the portable builds (make portable).
//
// fastgguf.c is compiled four times, once per instruction set, and linked with this file:
//   avx512   x86-64-v4 + AVX-512 VNNI   Intel Ice Lake / Sapphire Rapids and newer, AMD Zen 4 / 5
//   avxvnni  x86-64-v3 + AVX-VNNI       Intel 12th gen (Alder Lake) and newer desktop / laptop
//   avx2     x86-64-v3                  Intel Haswell (2013) and newer, AMD Zen 1 and newer
//   sse2     x86-64                     any 64-bit x86 CPU (slow)
// and main() runs the fastest one this CPU supports. FASTGGUF_CPU=avx512|avxvnni|avx2|sse2
// picks one by hand (for testing; refused when the CPU cannot run it), FASTGGUF_CPU=auto is the default.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DECLARE(v) int fg_main_full_##v(int, char **); int fg_main_server_##v(int, char **); \
                   int fg_main_console_##v(int, char **);
DECLARE(avx512) DECLARE(avxvnni) DECLARE(avx2) DECLARE(sse2)

#if defined(FG_SERVER_MAIN)
#define ENTRY(v) fg_main_server_##v
#elif defined(FG_CONSOLE_MAIN)
#define ENTRY(v) fg_main_console_##v
#else
#define ENTRY(v) fg_main_full_##v
#endif

static int has_avx2(void) {
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("bmi2") &&
           __builtin_cpu_supports("f16c") && __builtin_cpu_supports("movbe") && __builtin_cpu_supports("lzcnt");
}
static int has_avxvnni(void) { return has_avx2() && __builtin_cpu_supports("avxvnni"); }
static int has_avx512(void) {
    return has_avx2() && __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512vl") &&
           __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512dq") &&
           __builtin_cpu_supports("avx512cd") && __builtin_cpu_supports("avx512vnni");
}
static int has_sse2(void) { return 1; }

static const struct { const char *name; int (*ok)(void); int (*run)(int, char **); } builds[] = {
    { "avx512",  has_avx512,  ENTRY(avx512) },
    { "avxvnni", has_avxvnni, ENTRY(avxvnni) },
    { "avx2",    has_avx2,    ENTRY(avx2) },
    { "sse2",    has_sse2,    ENTRY(sse2) },
};

int main(int argc, char **argv) {
    __builtin_cpu_init();
    int n = (int)(sizeof(builds) / sizeof(builds[0]));
    const char *want = getenv("FASTGGUF_CPU");
    if (want && *want && strcmp(want, "auto") != 0) {
        int i = 0;
        while (i < n && strcmp(want, builds[i].name) != 0) i++;
        if (i < n && builds[i].ok()) return builds[i].run(argc, argv);
        fprintf(stderr, i < n ? "FASTGGUF_CPU=%s: this CPU cannot run that build, choosing automatically\n"
                              : "FASTGGUF_CPU=%s: unknown (avx512, avxvnni, avx2 or sse2), choosing automatically\n", want);
    }
    for (int i = 0; i < n; i++)
        if (builds[i].ok()) return builds[i].run(argc, argv);
    return 1;
}
