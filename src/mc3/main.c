/* mc3 — PortMaster static-recompilation host for Midnight Club: L.A. Remix (USA).
 *
 * Per-game host that psprecomp expects a game repo to provide:
 *   1. load the decrypted ELF module (segments at their link addresses)
 *   2. map it, reset the CPU, register every recompiled function
 *   3. initialise the HLE firmware surface and a render backend
 *   4. run module_start as the boot thread (PREAL_THREADS)
 *   5. pump a host frame loop: yield to guest threads, present the framebuffer
 *
 * Nothing here touches game files: the decrypted EBOOT.ELF is supplied by the
 * user at run time (-e). Generated C (mc3_funcs_*.c, mc3_register.c,
 * mc3_imports.c) is produced by `allegrexrecomp emit` and compiled in.
 *
 * Build: see src/mc3/CMakeLists.txt  (build psprecomp with -DPSPRECOMP_REAL_THREADS=ON)
 *
 * THREAD MODEL (host side):
 *   - Main thread: runs module_start (boot), creates and starts guest threads,
 *     then pumps the host frame loop (psp_threadman_yield + present).
 *   - Present thread: calls sdl2_present_loop() at a fixed ~60 FPS, reading the
 *     shared framebuffer state set by guest execution. Independent of guest
 *     execution so the window stays alive even if the game stalls.
 *
 * The guest threads (created via sceKernelCreateThread / sceKernelStartThread)
 * run inside psp_threadman_yield() calls during the frame loop — they borrow
 * the CPU from the main thread through the baton handover in threadman.c.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define _DEFAULT_SOURCE
#define _POSIX_C_SOURCE 200809L
#include <psprecomp/cpu.h>
#include <psprecomp/mem.h>
#include <psprecomp/dispatch.h>
#include <psprecomp/hle.h>
#include <psprecomp/render.h>
#include <psprecomp/ctors.h>
#include "mc3_telemetry.h"
#include <psprecomp/recomp_rt.h>

#include <signal.h>
#include <unistd.h>
#include <dlfcn.h>
#include <link.h>
#include <sys/time.h>
#include <ucontext.h>

#include "render_gles.h"
#include "render_sdl.h"
#include "render_terminal.h"

#if defined(PSPRECOMP_HAVE_GLES)
extern const psp_render_backend psp_render_gles32;
#endif
#if defined(PSPRECOMP_HAVE_SDL2)
extern const psp_render_backend psp_render_sdl2;
#endif
extern const psp_render_backend psp_render_terminal;

/* ---- SIGSEGV handler: report last guest function before crashing --------- */
static void on_segv(int sig) {
    (void)sig;
    const char msg[] = "\n=== SIGSEGV in guest execution ===\n";
    ssize_t unused = write(STDERR_FILENO, msg, sizeof(msg) - 1);
    (void)unused;
    _exit(128 + SIGSEGV);
}

#include <elf.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/fb.h>
#include <errno.h>

#include "mc3_funcs.h"

extern void psp_recomp_register(void);   /* defined by mc3_register.c */

/* ---- runtime state ------------------------------------------------------ */

static uint8_t *g_module_image;
static uint32_t  g_module_size;
static volatile sig_atomic_t g_watchdog_fired = 0;

/* Direct framebuffer presentation via /dev/fb0 (no SDL dependency). */
static int   g_fb_fd = -1;
static void *g_fb_mem = NULL;
static uint32_t g_fb_w = 0, g_fb_h = 0;
static size_t   g_fb_size = 0;

/* ---- optional MC3_PROF=1 SIGPROF PC sampler (zero overhead when unset) --- */
#define MC3_PROF_RING 16384u
static int       g_prof_enabled = 0;
static uintptr_t g_prof_base = 0;
static uintptr_t g_prof_end  = 0;
static uint32_t  g_prof_ring[MC3_PROF_RING];
static uint32_t  g_prof_head = 0;
static uint64_t  g_prof_total = 0;

static int prof_phdr_cb(struct dl_phdr_info *info, size_t size, void *data) {
    (void)size; (void)data;
    if (info->dlpi_name && info->dlpi_name[0] != '\0') return 0;
    g_prof_base = (uintptr_t)info->dlpi_addr;
    uintptr_t hi = g_prof_base;
    for (ElfW(Half) i = 0; i < info->dlpi_phnum; i++) {
        if (info->dlpi_phdr[i].p_type == PT_LOAD) {
            uintptr_t seg_end = g_prof_base + info->dlpi_phdr[i].p_vaddr + info->dlpi_phdr[i].p_memsz;
            if (seg_end > hi) hi = seg_end;
        }
    }
    g_prof_end = hi;
    return 1;
}

static void on_prof(int sig, siginfo_t *si, void *uctx) {
    (void)sig; (void)si;
    ucontext_t *uc = (ucontext_t *)uctx;
#if defined(__aarch64__)
    uintptr_t pc = (uintptr_t)uc->uc_mcontext.pc;
#elif defined(__x86_64__)
    uintptr_t pc = (uintptr_t)uc->uc_mcontext.gregs[REG_RIP];
#else
    (void)uc;
    uintptr_t pc = 0;
#endif
    uint32_t off = (pc >= g_prof_base && pc < g_prof_end) ? (uint32_t)(pc - g_prof_base) : 0u;
    uint32_t idx = __atomic_fetch_add(&g_prof_head, 1u, __ATOMIC_RELAXED) & (MC3_PROF_RING - 1u);
    g_prof_ring[idx] = off;
    __atomic_fetch_add(&g_prof_total, 1u, __ATOMIC_RELAXED);
}

static void mc3_prof_init(void) {
    const char *env = getenv("MC3_PROF");
    if (!env || !*env || strcmp(env, "0") == 0) return;
    dl_iterate_phdr(prof_phdr_cb, NULL);
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = on_prof;
    sa.sa_flags = SA_SIGINFO | SA_RESTART;
    sigemptyset(&sa.sa_mask);
    if (sigaction(SIGPROF, &sa, NULL) != 0) return;
    struct itimerval tv;
    memset(&tv, 0, sizeof tv);
    tv.it_interval.tv_usec = 2000; /* 500 Hz */
    tv.it_value.tv_usec = 2000;
    setitimer(ITIMER_PROF, &tv, NULL);
    g_prof_enabled = 1;
    fprintf(stderr, "mc3: MC3_PROF=1 enabled (500 Hz, base=0x%lx end=0x%lx)\n",
            (unsigned long)g_prof_base, (unsigned long)g_prof_end);
}

typedef struct { uint32_t off; uint32_t count; } prof_pair_t;
static int prof_cmp_off(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}
static int prof_cmp_cnt_desc(const void *a, const void *b) {
    const prof_pair_t *x = (const prof_pair_t *)a, *y = (const prof_pair_t *)b;
    return (y->count > x->count) - (y->count < x->count);
}

static void mc3_prof_dump(void) {
    if (!g_prof_enabled) return;
    struct itimerval tv;
    memset(&tv, 0, sizeof tv);
    setitimer(ITIMER_PROF, &tv, NULL);
    uint64_t total = __atomic_load_n(&g_prof_total, __ATOMIC_RELAXED);
    uint32_t n = total < MC3_PROF_RING ? (uint32_t)total : MC3_PROF_RING;
    if (n == 0) return;
    uint32_t tmp[MC3_PROF_RING];
    memcpy(tmp, g_prof_ring, n * sizeof(uint32_t));
    qsort(tmp, n, sizeof(uint32_t), prof_cmp_off);
    prof_pair_t *pairs = (prof_pair_t *)malloc(n * sizeof(prof_pair_t));
    if (!pairs) return;
    uint32_t np = 0;
    for (uint32_t i = 0; i < n; ) {
        uint32_t j = i + 1;
        while (j < n && tmp[j] == tmp[i]) j++;
        pairs[np].off = tmp[i];
        pairs[np].count = j - i;
        np++;
        i = j;
    }
    qsort(pairs, np, sizeof(prof_pair_t), prof_cmp_cnt_desc);
    uint32_t top = np < 35 ? np : 35;
    uint32_t file_top = np < 4096 ? np : 4096;
    FILE *fp = fopen("mc3_prof.txt", "w");
    fprintf(stderr, "=== MC3_PROF top %u/%u PCs (last %u of %llu samples) ===\n",
            top, np, n, (unsigned long long)total);
    if (fp) fprintf(fp, "=== MC3_PROF top %u/%u PCs (last %u of %llu samples) ===\n",
                    file_top, np, n, (unsigned long long)total);
    for (uint32_t i = 0; i < file_top; i++) {
        Dl_info dli;
        const char *sym = "<ext/libc>";
        if (pairs[i].off != 0) {
            if (dladdr((void *)(g_prof_base + pairs[i].off), &dli) && dli.dli_sname)
                sym = dli.dli_sname;
            else
                sym = "<mc3>";
        }
        double pct = 100.0 * (double)pairs[i].count / (double)n;
        if (i < top)
            fprintf(stderr, "  0x%08x  %5u  (%5.2f%%)  %s\n", pairs[i].off, pairs[i].count, pct, sym);
        if (fp) fprintf(fp, "  0x%08x  %5u  (%5.2f%%)  %s\n", pairs[i].off, pairs[i].count, pct, sym);
    }
    if (fp) fclose(fp);
    free(pairs);
}

/* ---- small ELF32 loader ------------------------------------------------- */

static int read_exact(int fd, void *buf, size_t len) {
    size_t done = 0;
    while (done < len) {
        ssize_t r = read(fd, (char *)buf + done, len - done);
        if (r <= 0) return -1;
        done += (size_t)r;
    }
    return 0;
}

static uint8_t *load_segments(const char *path, uint32_t *out_size) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return NULL; }

    Elf32_Ehdr eh;
    if (read_exact(fd, &eh, sizeof eh) != 0) { perror("read ehdr"); close(fd); return NULL; }
    if (eh.e_ident[EI_MAG0] != ELFMAG0 || eh.e_ident[EI_MAG1] != ELFMAG1 ||
        eh.e_ident[EI_MAG2] != ELFMAG2 || eh.e_ident[EI_MAG3] != ELFMAG3) {
        fprintf(stderr, "mc3: not an ELF\n"); close(fd); return NULL;
    }
    if (eh.e_ident[EI_CLASS] != ELFCLASS32 || eh.e_ident[EI_DATA] != ELFDATA2LSB) {
        fprintf(stderr, "mc3: not ELF32 little-endian\n"); close(fd); return NULL;
    }
    if (eh.e_machine != EM_MIPS) {
        fprintf(stderr, "mc3: not MIPS (machine 0x%x)\n", eh.e_machine); close(fd); return NULL;
    }

    Elf32_Phdr *ph = calloc(eh.e_phnum, sizeof *ph);
    if (!ph) { perror("calloc"); close(fd); return NULL; }
    if (lseek(fd, eh.e_phoff, SEEK_SET) == (off_t)-1 ||
        read_exact(fd, ph, (size_t)eh.e_phnum * sizeof *ph) != 0) {
        perror("read phdr"); free(ph); close(fd); return NULL;
    }

    uint32_t total = 0;
    for (int i = 0; i < eh.e_phnum; i++)
        if (ph[i].p_type == PT_LOAD && ph[i].p_vaddr + ph[i].p_memsz > total)
            total = ph[i].p_vaddr + ph[i].p_memsz;

    uint8_t *image = (uint8_t *)calloc(1, total ? total : 1);
    if (!image) { perror("calloc image"); free(ph); close(fd); return NULL; }

    for (int i = 0; i < eh.e_phnum; i++) {
        if (ph[i].p_type != PT_LOAD) continue;
        if (lseek(fd, ph[i].p_offset, SEEK_SET) == (off_t)-1) {
            perror("lseek seg"); free(image); free(ph); close(fd); return NULL;
        }
        uint8_t *dst = image + ph[i].p_vaddr;   /* module links at 0: vaddr == offset */
        if (read_exact(fd, dst, ph[i].p_filesz) != 0) {
            perror("read seg"); free(image); free(ph); close(fd); return NULL;
        }
        fprintf(stderr, "mc3: loaded PT_LOAD @ vaddr 0x%08x filesz 0x%x memsz 0x%x\n",
                ph[i].p_vaddr, ph[i].p_filesz, ph[i].p_memsz);
    }

    free(ph);
    close(fd);
    *out_size = total;
    fprintf(stderr, "mc3: module image 0x%08x bytes\n", total);
    return image;
}

/* ---- RSS ---------------------------------------------------------------- */

static uint64_t peak_rss_kb(void) {
    FILE *f = fopen("/proc/self/status", "r");
    if (!f) return 0;
    char line[256];
    uint64_t hwm = 0;
    while (fgets(line, sizeof line, f)) {
        if (strncmp(line, "VmHWM:", 6) == 0) {
            unsigned long v = 0;
            if (sscanf(line + 6, " %lu", &v) == 1) hwm = v;
            break;
        }
    }
    fclose(f);
    return hwm;
}

/* ---- watchdog ----------------------------------------------------------- */

static void on_alarm(int sig) {
    (void)sig;
    g_watchdog_fired = 1;
    psp_exit_requested_set(1);
    const char msg[] = "\n=== WATCHDOG: timeout reached ===\n";
    ssize_t unused = write(STDERR_FILENO, msg, sizeof(msg) - 1);
    (void)unused;
}

/* ---- teclado como controle PSP (para testes via QEMU) ------------------- */

/* PSP_CTRL_* bit assignments from the official controller API (pspctrl.h). */
enum {
    PSP_CTRL_SELECT   = 0x0001u,
    PSP_CTRL_START    = 0x0008u,
    PSP_CTRL_UP       = 0x0010u,
    PSP_CTRL_RIGHT    = 0x0020u,
    PSP_CTRL_DOWN     = 0x0040u,
    PSP_CTRL_LEFT     = 0x0080u,
    PSP_CTRL_LTRIGGER = 0x0100u,
    PSP_CTRL_RTRIGGER = 0x0200u,
    PSP_CTRL_TRIANGLE = 0x1000u,
    PSP_CTRL_CIRCLE   = 0x2000u,
    PSP_CTRL_CROSS    = 0x4000u,
    PSP_CTRL_SQUARE   = 0x8000u
};

/* ---- present thread ------------------------------------------------------
 *
 * Calls sdl2_draw_frame() at a fixed rate (~60 FPS). Reads the shared
 * framebuffer state set by guest execution and presents it to the window.
 * Independent of guest execution so the window stays alive even if the game
 * stalls. */
typedef struct { int watchdog_secs; int watchdog_loop_enabled; uint64_t max_frames; uint64_t frames; } mc3_guest_ctx;

static void *guest_thread_main(void *arg) {
    mc3_guest_ctx *ctx = (mc3_guest_ctx *)arg;
    psp_threadman_reset();
    alarm((unsigned)ctx->watchdog_secs);
psp_cpu.r[PSP_REG_A0] = 4u;
psp_cpu.r[PSP_REG_A1] = 0u;
fprintf(stderr, "mc3: entering module_start @ 0x000005A4 (boot thread)\n");
psp_func_000005A4();
fprintf(stderr, "mc3: module_start returned\n");
alarm(0);
uint64_t frames = 0;
uint64_t max_frames = ctx->max_frames;
fprintf(stderr, "mc3: entering host frame loop\n");
fflush(stderr);
if (ctx->watchdog_loop_enabled) alarm((unsigned)ctx->watchdog_secs);
const char *ac_env = getenv("MC3_AUTO_CROSS");
int auto_cross = (ac_env && strcmp(ac_env, "1") == 0);
    const char *tge_env = getenv("MC3_TRACE_GE");
    int trace_ge = (tge_env && strcmp(tge_env, "1") == 0);
while (!psp_exit_requested() && !g_watchdog_fired && frames < max_frames) {
psp_threadman_yield();
if ((psp_ctrl_get_buttons() & 0x0009u) == 0x0009u) {
fprintf(stderr, "mc3: EXIT requested via SELECT+START\n");
break;
}
if (psp_exit_requested()) {
fprintf(stderr, "mc3: EXIT requested\n");
break;
}
if (auto_cross) {
static int auto_cross_held = 0;
if (frames > 120 && (frames % 60) == 0) {
psp_ctrl_set(psp_ctrl_get_buttons() | PSP_CTRL_CROSS, 128, 128);
auto_cross_held = 4;
fprintf(stderr, "mc3: auto-ctrl CROSS held at frame %llu\n", (unsigned long long)frames);
} else if (auto_cross_held > 0 && --auto_cross_held == 0) {
psp_ctrl_set(psp_ctrl_get_buttons() & ~PSP_CTRL_CROSS, 128, 128);
}
}
if (trace_ge) {
static uint64_t last_ge_count = 0;
uint64_t current_ge_count = psp_ge_command_count();
if (current_ge_count != last_ge_count) {
fprintf(stderr, "mc3: GE commands %llu -> %llu at frame %llu\n",
(unsigned long long)last_ge_count, (unsigned long long)current_ge_count, (unsigned long long)frames);
last_ge_count = current_ge_count;
}
}
frames++;
if ((frames % 300) == 0) {
fprintf(stderr, "mc3: frame %llu, GE commands=%llu, GE vertices=%llu, GE pixels=%llu\n",
(unsigned long long)frames, (unsigned long long)psp_ge_command_count(),
(unsigned long long)psp_ge_vertex_count(), (unsigned long long)psp_render_pixels());
}
}
ctx->frames = frames;
alarm(0);
psp_exit_requested_set(1);
return NULL;
}


/* ---- entry point -------------------------------------------------------- */

int main(int argc, char **argv) {
    const char *elf_path = NULL;
    const char *render_name = "software";
    const char *data_dir = ".";
    const char *log_path = NULL;
    int opt;
    while ((opt = getopt(argc, argv, "e:r:d:l:")) != -1) {
        if (opt == 'e') elf_path = optarg;
        else if (opt == 'r') render_name = optarg;
        else if (opt == 'd') data_dir = optarg;
        else if (opt == 'l') log_path = optarg;
        else {
            fprintf(stderr, "usage: mc3 -e <EBOOT.ELF> [-r software|gles32|null] [-d <data_dir>] [-l <log_path>]\n");
            return 2;
        }
    }
    if (!elf_path) {
        fprintf(stderr, "mc3: -e <EBOOT.ELF> required\n");
        return 2;
    }

    /* 1. memory */
    if (psp_mem_init() != 0) { fprintf(stderr, "mc3: mem init failed\n"); return 1; }

    uint8_t *image = load_segments(elf_path, &g_module_size);
    if (!image) return 1;
    g_module_image = image;
    if (psp_mem_map_module(0, g_module_size) != 0) {
        fprintf(stderr, "mc3: map module failed\n"); return 1;
    }
    /* psp_mem_map_module() creates the address range for a PRX linked at
     * zero. Install the loaded segment bytes into that mapping; otherwise
     * guest data reads -- including the boot path's PCODE.TXT filename --
     * see a zero-filled image even though the code was loaded successfully. */
    if (psp_mem_write_block(0, image, g_module_size) != 0) {
        fprintf(stderr, "mc3: copy module image failed\n"); return 1;
    }

    /* 2. HLE firmware surface */
    psp_hle_init();
    psp_io_set_root(data_dir);

    /* 3. register every recompiled function (+ interior labels) */
    psp_recomp_register();

    /* 4. CPU — set initial stack pointer into user RAM. */
    psp_cpu_reset();
    psp_cpu.r[PSP_REG_SP] = 0x09FFF000u;   /* initial stack, in user RAM */
    psp_cpu.r[PSP_REG_GP] = 0x0042CEB0u;   /* .sdata (0x00424EC0) + 0x7FF0 */

    /* 5. render backend & telemetry. */
    mc3_telemetry_init(log_path, render_name);
#if defined(PSPRECOMP_HAVE_GLES)
    psp_render_register_backend(&psp_render_gles32);
#endif
#if defined(PSPRECOMP_HAVE_SDL2)
    psp_render_register_backend(&psp_render_sdl2);
#endif
    psp_render_register_backend(&psp_render_terminal);
    if (psp_render_select(render_name) != 0) {
        fprintf(stderr, "mc3: renderer '%s' unavailable; using software renderer\n",
                render_name);
        render_name = "software";
        if (psp_render_select(render_name) != 0) {
            fprintf(stderr, "mc3: no usable render backend\n");
            return 1;
        }
    }
    if (psp_render_current()->init(640, 480) != 0) {
        fprintf(stderr, "mc3: render init failed (%s)\n", render_name); return 1;
    }
    fprintf(stderr, "mc3: render backend: %s\n", psp_render_current()->name);

    /* Open /dev/fb0 for direct framebuffer presentation (software backend only). */
    if (strcmp(psp_render_current()->name, "software") == 0) {
        g_fb_fd = open("/dev/fb0", O_RDWR);
        if (g_fb_fd < 0) {
#if defined(PSPRECOMP_HAVE_SDL2)
            /* The software rasterizer targets guest VRAM. On a desktop host
             * there is no /dev/fb0, so use the SDL2 presenter to show that
             * framebuffer in a window instead of aborting before boot. */
            fprintf(stderr, "mc3: /dev/fb0 unavailable (%s); falling back to SDL2 window\n",
                    strerror(errno));
    psp_render_current()->shutdown();
            if (psp_render_select("sdl2") != 0 ||
                psp_render_current()->init(640, 480) != 0) {
                fprintf(stderr, "mc3: SDL2 fallback could not initialize\n");
                return 1;
            }
            render_name = "sdl2";
#else
            fprintf(stderr, "mc3: cannot open /dev/fb0: %s\n", strerror(errno));
            return 1;
#endif
        }
        if (g_fb_fd >= 0) {
            struct fb_var_screeninfo vinfo;
            if (ioctl(g_fb_fd, FBIOGET_VSCREENINFO, &vinfo) != 0) {
                fprintf(stderr, "mc3: FBIOGET_VSCREENINFO failed: %s\n", strerror(errno));
                return 1;
            }
            g_fb_w = vinfo.xres;
            g_fb_h = vinfo.yres;
            g_fb_size = (size_t)vinfo.xres * (size_t)vinfo.yres * (size_t)(vinfo.bits_per_pixel / 8);
            g_fb_mem = mmap(NULL, g_fb_size, PROT_READ | PROT_WRITE, MAP_SHARED, g_fb_fd, 0);
            if (g_fb_mem == MAP_FAILED) {
                fprintf(stderr, "mc3: mmap /dev/fb0 failed: %s\n", strerror(errno));
                return 1;
            }
            fprintf(stderr, "mc3: fb0 %ux%u @ %p\n", g_fb_w, g_fb_h, g_fb_mem);
        }
    }

    /* 6. threading. psp_threadman_reset() runs inside guest_thread_main so
     * thread-local boot state (g_tls_me) binds to the guest execution thread. */
    signal(SIGSEGV, on_segv);

    /* Watchdog: 30 s of guest time, then longjmp back. */
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = on_alarm;
    sigaction(SIGALRM, &sa, NULL);
    int watchdog_secs = 30;
    const char *wd_env = getenv("MC3_WATCHDOG");
    if (wd_env && *wd_env) {
        int v = atoi(wd_env);
        if (v > 0) watchdog_secs = v;
    }
    uint64_t max_frames = 100000000;
    const char *frames_env = getenv("MC3_FRAMES");
    if (frames_env && *frames_env) {
        long v = atol(frames_env);
        if (v > 0) max_frames = (uint64_t)v;
    }
mc3_guest_ctx gctx;
gctx.watchdog_secs = watchdog_secs;
gctx.watchdog_loop_enabled = (wd_env && *wd_env) ? 1 : 0;
gctx.max_frames = max_frames;
gctx.frames = 0;

    mc3_prof_init();

#if defined(PSPRECOMP_HAVE_SDL2)
if (strcmp(render_name, "sdl2") == 0) {
pthread_t guest_tid;
if (pthread_create(&guest_tid, NULL, guest_thread_main, &gctx) != 0) {
fprintf(stderr, "mc3: failed to create guest thread\n");
return 1;
}
sdl2_present_loop();
psp_exit_requested_set(1);
pthread_join(guest_tid, NULL);
} else
#endif
#if defined(PSPRECOMP_HAVE_GLES)
if (strcmp(render_name, "gles32") == 0) {
pthread_t guest_tid;
if (pthread_create(&guest_tid, NULL, guest_thread_main, &gctx) != 0) {
fprintf(stderr, "mc3: failed to create guest thread\n");
return 1;
}
gles32_present_loop();
psp_exit_requested_set(1);
pthread_join(guest_tid, NULL);
} else
#endif
{
guest_thread_main(&gctx);
}

uint64_t frames = gctx.frames;
    

    /* ---- report --------------------------------------------------------- */
    fprintf(stderr, "\n=== mc3 report ===\n");
    fprintf(stderr, "GE commands:   %llu\n", (unsigned long long)psp_ge_command_count());
    fprintf(stderr, "GE vertices:   %llu\n", (unsigned long long)psp_ge_vertex_count());
    fprintf(stderr, "GE pixels:     %llu  <-- >0 means the GE drew\n",
            (unsigned long long)psp_render_pixels());
    fprintf(stderr, "host frames:   %llu\n", (unsigned long long)frames);
    fprintf(stderr, "io bytes read: %llu\n", (unsigned long long)psp_io_bytes_read());
    fprintf(stderr, "dispatch:      %u registered, %llu misses\n",
            psp_dispatch_count(), (unsigned long long)psp_dispatch_misses());
    fprintf(stderr, "HLE unimplemented calls: %llu (logged %llu)\n",
            (unsigned long long)psp_hle_unimplemented_calls(),
            (unsigned long long)psp_hle_unimplemented_logged());
    fprintf(stderr, "bad memory accesses: %llu\n",
            (unsigned long long)psp_mem_bad_access);
    fprintf(stderr, "threadman: switches=%llu parks=%llu created=%llu started=%llu\n",
            (unsigned long long)psp_threadman_switches(),
            (unsigned long long)psp_threadman_parks(),
            (unsigned long long)psp_threadman_created(),
            (unsigned long long)psp_threadman_started());
    fprintf(stderr, "loop back-edge: addr=0x%08x hits=%llu\n",
            psp_trace_loop_addr(), (unsigned long long)psp_trace_loop_hits());
    fprintf(stderr, "peak RSS:      %llu KiB\n",
            (unsigned long long)peak_rss_kb());
    fprintf(stderr, "=================\n");
    psp_threadman_dump(stderr);
    psp_ge_dump_stats(stderr);
    psp_trace_dump();
    mc3_prof_dump();

    /* Framebuffer cleanup. */
    if (g_fb_mem && g_fb_size) munmap(g_fb_mem, g_fb_size);
    if (g_fb_fd >= 0) close(g_fb_fd);

    mc3_telemetry_shutdown();
    psp_render_current()->shutdown();
    psp_mem_free();
    free(image);
    return g_watchdog_fired ? 1 : 0;
}
