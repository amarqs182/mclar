/* mc3 -- real-time performance telemetry & diagnostic log generator.
 *
 * Collects thread-safe rolling metrics from the guest GE/pacing path,
 * the GLES32 render/present thread, and the SDL audio callback/sink,
 * feeds the on-screen Performance HUD, and writes a structured log file
 * (`mc3_telemetry.log` by default, `-l <path>` or `MC3_LOG=<path>`,
 * `MC3_LOG=0` to disable).
 */
#ifndef MC3_TELEMETRY_H
#define MC3_TELEMETRY_H

#include <stdint.h>

#define MC3_TEL_HISTORY  120
#define MC3_TEL_AUDIO_CH 9

typedef struct {
    double   guest_fps;
    double   host_fps;
    double   ft_avg_ms;
    double   ft_max_ms;
    double   ft_p95_ms;
    double   ge_cpu_ms;
    double   gpu_exec_ms;
    double   tex_up_ms;
    double   pace_wait_ms;

    uint32_t last_draws;
    uint32_t last_verts;
    uint32_t last_tex_hits;
    uint32_t last_tex_misses;
    uint32_t tex_slots_used;
    uint32_t tex_slots_cap;

    uint64_t bbox_tested_sec;
    uint64_t bbox_culled_sec;
    uint64_t ge_lists_sec;

    int      aud_active_chs;
    double   aud_buf_ms;
    double   aud_buf_min_ms;
    double   aud_max_gap_ms;
    uint64_t aud_underruns_total;
    uint64_t aud_underruns_recent;
    uint64_t aud_overruns_total;
    uint64_t aud_hle_underruns;

    float    ft_hist_ms[MC3_TEL_HISTORY];
    int      ft_hist_head;

    char     log_path[256];
} mc3_tel_snapshot;

void mc3_telemetry_init(const char *log_path, const char *backend);
void mc3_telemetry_log_video(int w, int h, int vw, int vh, int vsync,
                             const char *gl_renderer, const char *gl_version);
void mc3_telemetry_log_audio_spec(int want_freq, int want_samples, int want_ch,
                                  int got_freq, int got_samples, int got_ch,
                                  uint32_t got_size);

void mc3_telemetry_on_guest_frame(double ft_ms, double ge_cpu_ms, double pace_wait_ms,
                                  uint32_t draws, uint32_t verts,
                                  uint64_t ge_lists_delta,
                                  uint64_t bbox_tested_delta,
                                  uint64_t bbox_culled_delta);

void mc3_telemetry_on_gpu_frame(double host_ft_ms, double gpu_exec_ms,
                                double tex_up_ms, double swap_ms,
                                uint32_t tex_hits, uint32_t tex_misses,
                                uint32_t tex_slots_used, uint32_t tex_slots_cap);

void mc3_telemetry_on_audio_sink(int ch, int samples, int freq, int channels,
                                 uint32_t buf_frames_after, int dropped_frames);

void mc3_telemetry_on_audio_cb(double cb_gap_ms, int active_chs,
                               uint32_t min_buf_frames, uint32_t max_buf_frames,
                               int underrun_ch, uint32_t underrun_avail,
                               uint32_t underrun_need);

void mc3_telemetry_get_snapshot(mc3_tel_snapshot *out);
void mc3_telemetry_shutdown(void);

#endif /* MC3_TELEMETRY_H */
