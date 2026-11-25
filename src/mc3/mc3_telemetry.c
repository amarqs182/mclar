/* mc3 -- real-time performance telemetry & diagnostic log generator. */

#define _POSIX_C_SOURCE 200809L

#include "mc3_telemetry.h"

#include <psprecomp/hle.h>

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

#define MAX_SPIKE_LOG 5
#define FT_RING_CAP   4096

typedef struct {
    double   t_sec;
    char     kind[16];
    double   ft_ms;
    double   ge_ms;
    double   gpu_ms;
    uint32_t draws;
    uint32_t verts;
} spike_record;

typedef struct {
    uint64_t sink_calls;
    uint64_t underruns;
    uint64_t overruns;
    uint64_t last_sink_us;
    double   last_gap_ms;
    double   max_gap_ms;
    uint32_t last_buf_frames;
} audio_ch_state;

static pthread_mutex_t g_tel_mtx = PTHREAD_MUTEX_INITIALIZER;
static FILE *g_tel_fp = NULL;
static int g_tel_inited = 0;
static uint64_t g_start_us = 0;
static uint64_t g_last_sec_flush_us = 0;
static char g_log_path[256] = {0};
static char g_backend_name[64] = {0};

/* Rolling 1-second & overall guest stats */
static uint64_t g_guest_frames_total = 0;
static uint64_t g_guest_frames_sec = 0;
static double   g_ft_sum_sec = 0.0;
static double   g_ft_max_sec = 0.0;
static double   g_ft_sum_total = 0.0;
static double   g_ft_max_total = 0.0;
static double   g_ge_sum_sec = 0.0;
static double   g_pace_sum_sec = 0.0;
static uint64_t g_draws_sum_sec = 0;
static uint64_t g_verts_sum_sec = 0;
static uint64_t g_lists_sec = 0;
static uint64_t g_bbox_tested_sec = 0;
static uint64_t g_bbox_culled_sec = 0;
static uint64_t g_bbox_tested_total = 0;
static uint64_t g_bbox_culled_total = 0;

/* Rolling 120-frame history for HUD */
static float    g_ft_hist[MC3_TEL_HISTORY];
static int      g_ft_hist_head = 0;
static float    g_ft_ring[FT_RING_CAP];
static uint32_t g_ft_ring_cnt = 0;
static uint32_t g_ft_ring_pos = 0;

/* Last published snapshot values */
static double   g_snap_guest_fps = 0.0;
static double   g_snap_host_fps = 0.0;
static double   g_snap_ft_avg = 0.0;
static double   g_snap_ft_max = 0.0;
static double   g_snap_ft_p95 = 0.0;
static double   g_snap_ge_ms = 0.0;
static double   g_snap_gpu_ms = 0.0;
static double   g_snap_tex_ms = 0.0;
static double   g_snap_pace_ms = 0.0;
static uint32_t g_snap_draws = 0;
static uint32_t g_snap_verts = 0;
static uint64_t g_snap_lists_sec = 0;
static uint64_t g_snap_bbox_tested_sec = 0;
static uint64_t g_snap_bbox_culled_sec = 0;

/* GPU stats */
static uint64_t g_gpu_frames_total = 0;
static uint64_t g_gpu_frames_sec = 0;
static double   g_gpu_exec_sum_sec = 0.0;
static double   g_gpu_exec_sum_total = 0.0;
static double   g_gpu_exec_max_total = 0.0;
static double   g_tex_up_sum_sec = 0.0;
static uint64_t g_tex_hits_sec = 0;
static uint64_t g_tex_misses_sec = 0;
static uint64_t g_tex_hits_total = 0;
static uint64_t g_tex_misses_total = 0;
static uint32_t g_last_tex_hits = 0;
static uint32_t g_last_tex_misses = 0;
static uint32_t g_tex_slots_used = 0;
static uint32_t g_tex_slots_cap = 1024;

/* Audio stats */
static audio_ch_state g_aud_ch[MC3_TEL_AUDIO_CH];
static int      g_aud_active_chs = 0;
static uint32_t g_aud_min_buf_frames = 0;
static uint32_t g_aud_min_buf_sec = UINT32_MAX;
static uint64_t g_aud_underruns_total = 0;
static uint64_t g_aud_underruns_sec = 0;
static uint64_t g_aud_underruns_recent = 0;
static uint64_t g_aud_overruns_total = 0;
static double   g_aud_max_gap_sec = 0.0;
static double   g_aud_max_gap_snap = 0.0;
static uint64_t g_last_ur_log_us = 0;
static uint32_t g_ur_log_burst = 0;
static uint64_t g_last_ov_log_us = 0;

/* Spike rate-limiting & worst spikes */
static uint64_t g_spike_sec_epoch = 0;
static uint32_t g_spike_sec_count = 0;
static spike_record g_worst_spikes[MAX_SPIKE_LOG];

static uint64_t tel_now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000ULL;
}

static int cmp_float(const void *a, const void *b) {
    float fa = *(const float *)a;
    float fb = *(const float *)b;
    return (fa > fb) - (fa < fb);
}

static void compute_percentiles(double *p50, double *p95, double *p99) {
    uint32_t n = g_ft_ring_cnt;
    if (n == 0) {
        *p50 = *p95 = *p99 = 0.0;
        return;
    }
    if (n > FT_RING_CAP) n = FT_RING_CAP;
    float tmp[FT_RING_CAP];
    memcpy(tmp, g_ft_ring, (size_t)n * sizeof(float));
    qsort(tmp, n, sizeof(float), cmp_float);
    uint32_t i50 = (uint32_t)(0.50 * (double)(n - 1));
    uint32_t i95 = (uint32_t)(0.95 * (double)(n - 1));
    uint32_t i99 = (uint32_t)(0.99 * (double)(n - 1));
    *p50 = tmp[i50];
    *p95 = tmp[i95];
    *p99 = tmp[i99];
}

static double read_rss_mib(void) {
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
        return (double)ru.ru_maxrss / 1024.0;
    }
    return 0.0;
}

static void record_worst_spike(double t_sec, const char *kind, double ft_ms,
                               double ge_ms, double gpu_ms, uint32_t draws, uint32_t verts) {
    int worst_idx = -1;
    double min_val = ft_ms;
    for (int i = 0; i < MAX_SPIKE_LOG; i++) {
        if (g_worst_spikes[i].ft_ms == 0.0) {
            worst_idx = i;
            break;
        }
        if (g_worst_spikes[i].ft_ms < min_val) {
            min_val = g_worst_spikes[i].ft_ms;
            worst_idx = i;
        }
    }
    if (worst_idx >= 0) {
        g_worst_spikes[worst_idx].t_sec = t_sec;
        snprintf(g_worst_spikes[worst_idx].kind, sizeof(g_worst_spikes[worst_idx].kind), "%s", kind);
        g_worst_spikes[worst_idx].ft_ms = ft_ms;
        g_worst_spikes[worst_idx].ge_ms = ge_ms;
        g_worst_spikes[worst_idx].gpu_ms = gpu_ms;
        g_worst_spikes[worst_idx].draws = draws;
        g_worst_spikes[worst_idx].verts = verts;
    }
}

void mc3_telemetry_init(const char *log_path, const char *backend) {
    pthread_mutex_lock(&g_tel_mtx);
    memset(g_ft_hist, 0, sizeof(g_ft_hist));
    memset(g_ft_ring, 0, sizeof(g_ft_ring));
    memset(g_aud_ch, 0, sizeof(g_aud_ch));
    memset(g_worst_spikes, 0, sizeof(g_worst_spikes));

    g_start_us = tel_now_us();
    g_last_sec_flush_us = g_start_us;
    g_spike_sec_epoch = g_start_us / 1000000ULL;
    snprintf(g_backend_name, sizeof(g_backend_name), "%s", backend ? backend : "unknown");

    const char *path = log_path;
    if (!path || !path[0]) {
        const char *env = getenv("MC3_LOG");
        if (env && env[0]) path = env;
        else path = "mc3_telemetry.log";
    }

    if (strcmp(path, "0") == 0 || strcmp(path, "off") == 0 || strcmp(path, "none") == 0) {
        g_log_path[0] = '\0';
        g_tel_fp = NULL;
    } else {
        snprintf(g_log_path, sizeof(g_log_path), "%s", path);
        g_tel_fp = fopen(g_log_path, "w");
        if (g_tel_fp) {
            time_t now_t = time(NULL);
            struct tm tm_now;
            localtime_r(&now_t, &tm_now);
            char tbuf[64];
            strftime(tbuf, sizeof(tbuf), "%Y-%m-%d %H:%M:%S", &tm_now);
            fprintf(g_tel_fp, "[INIT] time=%s backend=%s\n", tbuf, g_backend_name);
            fflush(g_tel_fp);
            fprintf(stderr, "mc3: telemetry log -> %s (F3 toggles HUD)\n", g_log_path);
        } else {
            fprintf(stderr, "mc3: warning: failed to open telemetry log %s\n", g_log_path);
            g_log_path[0] = '\0';
        }
    }
    g_tel_inited = 1;
    pthread_mutex_unlock(&g_tel_mtx);
}

void mc3_telemetry_log_video(int w, int h, int vw, int vh, int vsync,
                             const char *gl_renderer, const char *gl_version) {
    pthread_mutex_lock(&g_tel_mtx);
    if (g_tel_fp) {
        fprintf(g_tel_fp, "[VIDEO] win=%dx%d view=%dx%d vsync=%d renderer=\"%s\" version=\"%s\"\n",
                w, h, vw, vh, vsync,
                gl_renderer ? gl_renderer : "unknown",
                gl_version ? gl_version : "unknown");
        fflush(g_tel_fp);
    }
    pthread_mutex_unlock(&g_tel_mtx);
}

void mc3_telemetry_log_audio_spec(int want_freq, int want_samples, int want_ch,
                                  int got_freq, int got_samples, int got_ch,
                                  uint32_t got_size) {
    pthread_mutex_lock(&g_tel_mtx);
    if (g_tel_fp) {
        fprintf(g_tel_fp,
                "[AUDIO_SPEC] want=%dHz/%dsmps/%dch got=%dHz/%dsmps/%dch buf_bytes=%u\n",
                want_freq, want_samples, want_ch,
                got_freq, got_samples, got_ch, got_size);
        fflush(g_tel_fp);
    }
    pthread_mutex_unlock(&g_tel_mtx);
}

void mc3_telemetry_on_guest_frame(double ft_ms, double ge_cpu_ms, double pace_wait_ms,
                                  uint32_t draws, uint32_t verts,
                                  uint64_t ge_lists_delta,
                                  uint64_t bbox_tested_delta,
                                  uint64_t bbox_culled_delta) {
    if (ft_ms < 0.0) ft_ms = 0.0;
    uint64_t now = tel_now_us();

    pthread_mutex_lock(&g_tel_mtx);
    if (!g_tel_inited) {
        pthread_mutex_unlock(&g_tel_mtx);
        return;
    }

    g_guest_frames_total++;
    g_guest_frames_sec++;
    g_ft_sum_sec += ft_ms;
    if (ft_ms > g_ft_max_sec) g_ft_max_sec = ft_ms;
    g_ft_sum_total += ft_ms;
    if (ft_ms > g_ft_max_total) g_ft_max_total = ft_ms;

    g_ge_sum_sec += ge_cpu_ms;
    g_pace_sum_sec += pace_wait_ms;
    g_draws_sum_sec += draws;
    g_verts_sum_sec += verts;
    g_lists_sec += ge_lists_delta;
    g_bbox_tested_sec += bbox_tested_delta;
    g_bbox_culled_sec += bbox_culled_delta;
    g_bbox_tested_total += bbox_tested_delta;
    g_bbox_culled_total += bbox_culled_delta;

    g_snap_draws = draws;
    g_snap_verts = verts;

    g_ft_hist[g_ft_hist_head] = (float)ft_ms;
    g_ft_hist_head = (g_ft_hist_head + 1) % MC3_TEL_HISTORY;

    g_ft_ring[g_ft_ring_pos] = (float)ft_ms;
    g_ft_ring_pos = (g_ft_ring_pos + 1) % FT_RING_CAP;
    if (g_ft_ring_cnt < FT_RING_CAP) g_ft_ring_cnt++;

    if (ft_ms > 45.0) {
        double t_sec = (double)(now - g_start_us) / 1000000.0;
        record_worst_spike(t_sec, "GUEST", ft_ms, ge_cpu_ms, g_snap_gpu_ms, draws, verts);
        if (g_tel_fp) {
            uint64_t ep = now / 1000000ULL;
            if (ep != g_spike_sec_epoch) {
                g_spike_sec_epoch = ep;
                g_spike_sec_count = 0;
            }
            if (g_spike_sec_count < 4u) {
                g_spike_sec_count++;
                fprintf(g_tel_fp,
                        "[SPIKE] t=%.2fs kind=GUEST ft=%.1fms ge_cpu=%.1fms pace=%.1fms draws=%u verts=%u\n",
                        t_sec, ft_ms, ge_cpu_ms, pace_wait_ms, draws, verts);
            }
        }
    }

    uint64_t elapsed_us = now - g_last_sec_flush_us;
    if (elapsed_us >= 1000000ULL) {
        double dt = (double)elapsed_us / 1000000.0;
        double gfps = (double)g_guest_frames_sec / dt;
        double hfps = (double)g_gpu_frames_sec / dt;
        double avg_ft = g_guest_frames_sec ? (g_ft_sum_sec / (double)g_guest_frames_sec) : 0.0;
        double avg_ge = g_guest_frames_sec ? (g_ge_sum_sec / (double)g_guest_frames_sec) : 0.0;
        double avg_pace = g_guest_frames_sec ? (g_pace_sum_sec / (double)g_guest_frames_sec) : 0.0;
        double avg_gpu = g_gpu_frames_sec ? (g_gpu_exec_sum_sec / (double)g_gpu_frames_sec) : 0.0;
        double avg_tex = g_gpu_frames_sec ? (g_tex_up_sum_sec / (double)g_gpu_frames_sec) : 0.0;
        double p50 = 0.0, p95 = 0.0, p99 = 0.0;
        compute_percentiles(&p50, &p95, &p99);

        g_snap_guest_fps = gfps;
        g_snap_host_fps = hfps;
        g_snap_ft_avg = avg_ft;
        g_snap_ft_max = g_ft_max_sec;
        g_snap_ft_p95 = p95;
        g_snap_ge_ms = avg_ge;
        g_snap_gpu_ms = avg_gpu;
        g_snap_tex_ms = avg_tex;
        g_snap_pace_ms = avg_pace;
        g_snap_lists_sec = g_lists_sec;
        g_snap_bbox_tested_sec = g_bbox_tested_sec;
        g_snap_bbox_culled_sec = g_bbox_culled_sec;
        g_aud_underruns_recent = g_aud_underruns_sec;
        g_aud_max_gap_snap = g_aud_max_gap_sec;

        if (g_tel_fp) {
            uint64_t tex_tot = g_tex_hits_sec + g_tex_misses_sec;
            double hit_pct = tex_tot ? (100.0 * (double)g_tex_hits_sec / (double)tex_tot) : 100.0;
            uint32_t avg_draws = g_guest_frames_sec ? (uint32_t)(g_draws_sum_sec / g_guest_frames_sec) : 0u;
            uint32_t avg_verts = g_guest_frames_sec ? (uint32_t)(g_verts_sum_sec / g_guest_frames_sec) : 0u;
            double buf_ms = (double)g_aud_min_buf_frames * 1000.0 / 44100.0;
            double min_buf_ms = (g_aud_min_buf_sec == UINT32_MAX) ? 0.0 : ((double)g_aud_min_buf_sec * 1000.0 / 44100.0);
            fprintf(g_tel_fp,
                    "[SEC] t=%.2fs gfps=%.1f hfps=%.1f ft_avg=%.1fms ft_p95=%.1fms ft_max=%.1fms "
                    "ge=%.1fms gpu=%.1fms tex=%.1fms wait=%.1fms draws=%u verts=%u "
                    "tex_hit=%.1f%% tex_miss=%llu bbox=%llu/%llu aud_ch=%d aud_buf=%.0fms "
                    "aud_min=%.0fms ur=%llu ov=%llu hle_ur=%llu aud_gap=%.1fms rss=%.1fMiB\n",
                    (double)(now - g_start_us) / 1000000.0,
                    gfps, hfps, avg_ft, p95, g_ft_max_sec,
                    avg_ge, avg_gpu, avg_tex, avg_pace, avg_draws, avg_verts,
                    hit_pct, (unsigned long long)g_tex_misses_sec,
                    (unsigned long long)g_bbox_culled_sec, (unsigned long long)g_bbox_tested_sec,
                    g_aud_active_chs, buf_ms, min_buf_ms,
                    (unsigned long long)g_aud_underruns_sec,
                    (unsigned long long)g_aud_overruns_total,
                    (unsigned long long)psp_audio_underruns_hle(),
                    g_aud_max_gap_sec, read_rss_mib());
            fflush(g_tel_fp);
        }

        g_last_sec_flush_us = now;
        g_guest_frames_sec = 0;
        g_ft_sum_sec = 0.0;
        g_ft_max_sec = 0.0;
        g_ge_sum_sec = 0.0;
        g_pace_sum_sec = 0.0;
        g_draws_sum_sec = 0;
        g_verts_sum_sec = 0;
        g_lists_sec = 0;
        g_bbox_tested_sec = 0;
        g_bbox_culled_sec = 0;
        g_gpu_frames_sec = 0;
        g_gpu_exec_sum_sec = 0.0;
        g_tex_up_sum_sec = 0.0;
        g_tex_hits_sec = 0;
        g_tex_misses_sec = 0;
        g_aud_underruns_sec = 0;
        g_aud_min_buf_sec = UINT32_MAX;
        g_aud_max_gap_sec = 0.0;
    }

    pthread_mutex_unlock(&g_tel_mtx);
}

void mc3_telemetry_on_gpu_frame(double host_ft_ms, double gpu_exec_ms,
                                double tex_up_ms, double swap_ms,
                                uint32_t tex_hits, uint32_t tex_misses,
                                uint32_t tex_slots_used, uint32_t tex_slots_cap) {
    (void)host_ft_ms;
    (void)swap_ms;
    uint64_t now = tel_now_us();

    pthread_mutex_lock(&g_tel_mtx);
    if (!g_tel_inited) {
        pthread_mutex_unlock(&g_tel_mtx);
        return;
    }

    g_gpu_frames_total++;
    g_gpu_frames_sec++;
    g_gpu_exec_sum_sec += gpu_exec_ms;
    g_gpu_exec_sum_total += gpu_exec_ms;
    if (gpu_exec_ms > g_gpu_exec_max_total) g_gpu_exec_max_total = gpu_exec_ms;
    g_tex_up_sum_sec += tex_up_ms;
    g_tex_hits_sec += tex_hits;
    g_tex_misses_sec += tex_misses;
    g_tex_hits_total += tex_hits;
    g_tex_misses_total += tex_misses;
    g_last_tex_hits = tex_hits;
    g_last_tex_misses = tex_misses;
    g_tex_slots_used = tex_slots_used;
    g_tex_slots_cap = tex_slots_cap;

    if (gpu_exec_ms > 20.0) {
        double t_sec = (double)(now - g_start_us) / 1000000.0;
        record_worst_spike(t_sec, "GPU", gpu_exec_ms, g_snap_ge_ms, gpu_exec_ms, g_snap_draws, g_snap_verts);
        if (g_tel_fp) {
            uint64_t ep = now / 1000000ULL;
            if (ep != g_spike_sec_epoch) {
                g_spike_sec_epoch = ep;
                g_spike_sec_count = 0;
            }
            if (g_spike_sec_count < 4u) {
                g_spike_sec_count++;
                fprintf(g_tel_fp,
                        "[SPIKE] t=%.2fs kind=GPU gpu=%.1fms tex_up=%.1fms tex_miss=%u draws=%u verts=%u\n",
                        t_sec, gpu_exec_ms, tex_up_ms, tex_misses, g_snap_draws, g_snap_verts);
            }
        }
    }

    pthread_mutex_unlock(&g_tel_mtx);
}

void mc3_telemetry_on_audio_sink(int ch, int samples, int freq, int channels,
                                 uint32_t buf_frames_after, int dropped_frames) {
    (void)freq;
    (void)channels;
    if (ch < 0 || ch >= MC3_TEL_AUDIO_CH) ch = 0;
    uint64_t now = tel_now_us();

    pthread_mutex_lock(&g_tel_mtx);
    if (!g_tel_inited) {
        pthread_mutex_unlock(&g_tel_mtx);
        return;
    }

    audio_ch_state *cs = &g_aud_ch[ch];
    cs->sink_calls++;
    cs->last_buf_frames = buf_frames_after;
    if (cs->last_sink_us > 0) {
        double gap = (double)(now - cs->last_sink_us) / 1000.0;
        cs->last_gap_ms = gap;
        if (gap > cs->max_gap_ms) cs->max_gap_ms = gap;
        if (gap > g_aud_max_gap_sec) g_aud_max_gap_sec = gap;
    }
    cs->last_sink_us = now;

    if (dropped_frames > 0) {
        cs->overruns++;
        g_aud_overruns_total++;
        if (g_tel_fp && (now - g_last_ov_log_us >= 1000000ULL)) {
            g_last_ov_log_us = now;
            fprintf(g_tel_fp,
                    "[AUDIO_OVERRUN] t=%.2fs ch=%d samples=%d dropped=%d buf=%u\n",
                    (double)(now - g_start_us) / 1000000.0,
                    ch, samples, dropped_frames, buf_frames_after);
        }
    }
    pthread_mutex_unlock(&g_tel_mtx);
}

void mc3_telemetry_on_audio_cb(double cb_gap_ms, int active_chs,
                               uint32_t min_buf_frames, uint32_t max_buf_frames,
                               int underrun_ch, uint32_t underrun_avail,
                               uint32_t underrun_need) {
    (void)max_buf_frames;
    uint64_t now = tel_now_us();

    pthread_mutex_lock(&g_tel_mtx);
    if (!g_tel_inited) {
        pthread_mutex_unlock(&g_tel_mtx);
        return;
    }

    g_aud_active_chs = active_chs;
    g_aud_min_buf_frames = min_buf_frames;
    if (active_chs > 0 && min_buf_frames < g_aud_min_buf_sec) {
        g_aud_min_buf_sec = min_buf_frames;
    }
    if (cb_gap_ms > g_aud_max_gap_sec) g_aud_max_gap_sec = cb_gap_ms;

    if (underrun_ch >= 0 && underrun_ch < MC3_TEL_AUDIO_CH) {
        g_aud_ch[underrun_ch].underruns++;
        g_aud_underruns_total++;
        g_aud_underruns_sec++;
        if (g_tel_fp && (g_ur_log_burst < 25u || (now - g_last_ur_log_us) >= 1000000ULL)) {
            g_ur_log_burst++;
            g_last_ur_log_us = now;
            fprintf(g_tel_fp,
                    "[AUDIO_UNDERRUN] t=%.2fs ch=%d avail=%u need=%u cb_gap=%.1fms "
                    "sink_gap=%.1fms ge=%.1fms gpu=%.1fms hle_ur=%llu\n",
                    (double)(now - g_start_us) / 1000000.0,
                    underrun_ch, underrun_avail, underrun_need, cb_gap_ms,
                    g_aud_ch[underrun_ch].last_gap_ms,
                    g_snap_ge_ms, g_snap_gpu_ms,
                    (unsigned long long)psp_audio_underruns_hle());
        }
    }

    pthread_mutex_unlock(&g_tel_mtx);
}

void mc3_telemetry_get_snapshot(mc3_tel_snapshot *out) {
    if (!out) return;
    pthread_mutex_lock(&g_tel_mtx);
    out->guest_fps = g_snap_guest_fps;
    out->host_fps = g_snap_host_fps;
    out->ft_avg_ms = g_snap_ft_avg;
    out->ft_max_ms = g_snap_ft_max;
    out->ft_p95_ms = g_snap_ft_p95;
    out->ge_cpu_ms = g_snap_ge_ms;
    out->gpu_exec_ms = g_snap_gpu_ms;
    out->tex_up_ms = g_snap_tex_ms;
    out->pace_wait_ms = g_snap_pace_ms;
    out->last_draws = g_snap_draws;
    out->last_verts = g_snap_verts;
    out->last_tex_hits = g_last_tex_hits;
    out->last_tex_misses = g_last_tex_misses;
    out->tex_slots_used = g_tex_slots_used;
    out->tex_slots_cap = g_tex_slots_cap;
    out->bbox_tested_sec = g_snap_bbox_tested_sec;
    out->bbox_culled_sec = g_snap_bbox_culled_sec;
    out->ge_lists_sec = g_snap_lists_sec;
    out->aud_active_chs = g_aud_active_chs;
    out->aud_buf_ms = (double)g_aud_min_buf_frames * 1000.0 / 44100.0;
    out->aud_buf_min_ms = (g_aud_min_buf_sec == UINT32_MAX) ? 0.0 : ((double)g_aud_min_buf_sec * 1000.0 / 44100.0);
    out->aud_max_gap_ms = g_aud_max_gap_snap;
    out->aud_underruns_total = g_aud_underruns_total;
    out->aud_underruns_recent = g_aud_underruns_recent;
    out->aud_overruns_total = g_aud_overruns_total;
    out->aud_hle_underruns = psp_audio_underruns_hle();
    memcpy(out->ft_hist_ms, g_ft_hist, sizeof(g_ft_hist));
    out->ft_hist_head = g_ft_hist_head;
    snprintf(out->log_path, sizeof(out->log_path), "%s", g_log_path);
    pthread_mutex_unlock(&g_tel_mtx);
}

void mc3_telemetry_shutdown(void) {
    pthread_mutex_lock(&g_tel_mtx);
    if (!g_tel_inited) {
        pthread_mutex_unlock(&g_tel_mtx);
        return;
    }

    if (g_tel_fp) {
        double runtime_s = (double)(tel_now_us() - g_start_us) / 1000000.0;
        double avg_ft = g_guest_frames_total ? (g_ft_sum_total / (double)g_guest_frames_total) : 0.0;
        double avg_gpu = g_gpu_frames_total ? (g_gpu_exec_sum_total / (double)g_gpu_frames_total) : 0.0;
        double p50 = 0.0, p95 = 0.0, p99 = 0.0;
        compute_percentiles(&p50, &p95, &p99);
        uint64_t tex_tot = g_tex_hits_total + g_tex_misses_total;
        double hit_pct = tex_tot ? (100.0 * (double)g_tex_hits_total / (double)tex_tot) : 100.0;
        double cull_pct = g_bbox_tested_total ? (100.0 * (double)g_bbox_culled_total / (double)g_bbox_tested_total) : 0.0;

        fprintf(g_tel_fp, "\n[SUMMARY]\n");
        fprintf(g_tel_fp, "runtime_s=%.2f backend=%s guest_frames=%llu host_frames=%llu\n",
                runtime_s, g_backend_name,
                (unsigned long long)g_guest_frames_total,
                (unsigned long long)g_gpu_frames_total);
        fprintf(g_tel_fp, "ft_avg=%.2fms ft_p50=%.2fms ft_p95=%.2fms ft_p99=%.2fms ft_max=%.2fms\n",
                avg_ft, p50, p95, p99, g_ft_max_total);
        fprintf(g_tel_fp, "gpu_exec_avg=%.2fms gpu_exec_max=%.2fms tex_hit_rate=%.2f%% (%llu hits / %llu misses)\n",
                avg_gpu, g_gpu_exec_max_total, hit_pct,
                (unsigned long long)g_tex_hits_total, (unsigned long long)g_tex_misses_total);
        fprintf(g_tel_fp, "bbox_tested=%llu bbox_culled=%llu (%.1f%%)\n",
                (unsigned long long)g_bbox_tested_total,
                (unsigned long long)g_bbox_culled_total, cull_pct);
        fprintf(g_tel_fp, "audio_underruns_cb=%llu audio_overruns=%llu audio_underruns_hle=%llu\n",
                (unsigned long long)g_aud_underruns_total,
                (unsigned long long)g_aud_overruns_total,
                (unsigned long long)psp_audio_underruns_hle());
        for (int c = 0; c < MC3_TEL_AUDIO_CH; c++) {
            if (g_aud_ch[c].sink_calls > 0 || g_aud_ch[c].underruns > 0) {
                fprintf(g_tel_fp, "  ch[%d]: sink_calls=%llu underruns=%llu overruns=%llu max_gap=%.1fms\n",
                        c,
                        (unsigned long long)g_aud_ch[c].sink_calls,
                        (unsigned long long)g_aud_ch[c].underruns,
                        (unsigned long long)g_aud_ch[c].overruns,
                        g_aud_ch[c].max_gap_ms);
            }
        }
        for (int i = 0; i < MAX_SPIKE_LOG; i++) {
            if (g_worst_spikes[i].ft_ms > 0.0) {
                fprintf(g_tel_fp, "  worst_spike[%d]: t=%.2fs kind=%s ft=%.1fms ge=%.1fms gpu=%.1fms draws=%u verts=%u\n",
                        i, g_worst_spikes[i].t_sec, g_worst_spikes[i].kind,
                        g_worst_spikes[i].ft_ms, g_worst_spikes[i].ge_ms,
                        g_worst_spikes[i].gpu_ms, g_worst_spikes[i].draws,
                        g_worst_spikes[i].verts);
            }
        }
        fclose(g_tel_fp);
        g_tel_fp = NULL;
        fprintf(stderr, "mc3: telemetry log saved to %s\n", g_log_path);
    }

    g_tel_inited = 0;
    pthread_mutex_unlock(&g_tel_mtx);
}
