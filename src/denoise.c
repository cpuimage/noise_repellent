/*
 * noise-repellent -- Noise Reduction LV2
 *
 * Copyright 2016 Luciano Dato <lucianodato@gmail.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/
 */


#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"
#include "noise_repellent.c"
#include <stdint.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
/* STFT default values */
#define FFT_SIZE    2048                    /* Size of the fft transform and frame */
#define INPUT_WINDOW    3                       /* 0 HANN 1 HAMMING 2 BLACKMAN 3 VORBIS Input windows for STFT algorithm */
#define OUTPUT_WINDOW    3                       /* 0 HANN 1 HAMMING 2 BLACKMAN 3 VORBIS Output windows for STFT algorithm */
#define OVERLAP_FACTOR    4                       /* 4 is 75% overlap Values bigger than 4 will rescale correctly (if Vorbis windows is not used) */

#define NOISE_ARRAY_STATE_MAX_SIZE 8192         /* max alloc size of the noise_thresholds to save with the session. This will consider upto fs of 192 kHz with aprox 23hz resolution */


typedef struct _Denoise Denoise;

struct _Denoise {
    /* here you can add additional per-instance
     * data such as properties */
    const float *input;                 /* input of samples from host (changing size) */
    float *output;                /* output of samples to host (changing size) */
    float samp_rate;              /* Sample rate received from the host */

    /* Parameters for the algorithm (user input) */
    float amount_of_reduction;            /* Amount of noise to reduce in dB */
    float noise_thresholds_offset;        /* This is to scale the noise profile (over subtraction factor) in dB */
    float release;                        /* Release time */
    float masking;                        /* Masking scaling */
    float whitening_factor_pc;            /* Whitening amount of the reduction percentage */
    bool noise_learn_state;              /* Learn Noise state (Manual-Off-Auto) */
    bool adaptive_state;                 /* Autocapture switch */
    // float reset_profile;                  /* Reset Noise switch */
    bool residual_listen;                /* For noise only listening */
    float transient_protection;           /* Multiplier for thresholding onsets with rolling mean */
    float enable;                         /* For soft bypass (click free bypass) */
//    float *report_latency;                /* Latency necessary */
    char *file_location;

    /* Parameters values and arrays for the STFT */
    int fft_size;                       /* FFTW input size */
    int fft_size_2;                     /* FFTW half input size */
    int window_option_input;            /* Type of input Window for the STFT */
    int window_option_output;           /* Type of output Window for the STFT */
    float overlap_factor;                 /* oversampling factor for overlap calculations */
    float overlap_scale_factor;           /* Scaling factor for conserving the final amplitude */
    int hop;                            /* Hop size for the STFT */
    float *input_window;                  /* Input Window values */
    float *output_window;                 /* Output Window values */

    /* Algorithm exta variables */
    float tau;                            /* time constant for soft bypass */
    float wet_dry_target;                 /* softbypass target for softbypass */
    float wet_dry;                        /* softbypass coeff */
    //  float reduction_coeff;                /* Gain to apply to the residual noise */
    float release_coeff;                  /* Release coefficient for Envelopes */
    float amount_of_reduction_linear;     /* Reduction amount linear value */
    float thresholds_offset_linear;       /* Threshold offset linear value */
    float whitening_factor;               /* Whitening amount of the reduction */

    /* Buffers for processing and outputting */
    int input_latency;
    float *in_fifo;                       /* internal input buffer */
    float *out_fifo;                      /* internal output buffer */
    float *output_accum;                  /* FFT output accumulator */
    int read_ptr;                       /* buffers read pointer */

    /* FFTW related arrays */
    float *input_fft_buffer;
    float *output_fft_buffer;
    fftwf_plan forward;
    fftwf_plan backward;

    /* Arrays and variables for getting bins info */
    float *fft_p2;                        /* power spectrum */
    float *fft_magnitude;                 /* magnitude spectrum */
    float *fft_phase;                     /* phase spectrum */

    /* noise related */
    float *noise_thresholds_p2;           /* captured noise profile power spectrum */
    float *noise_thresholds_scaled;       /* captured noise profile power spectrum scaled by oversubtraction */
    bool noise_thresholds_availables;    /* indicate whether a noise profile is available or no */
    float noise_window_count;             /* Count windows for mean computing */

    /* smoothing related */
    float *smoothed_spectrum;             /* power spectrum to be smoothed */
    float *smoothed_spectrum_prev;        /* previous frame smoothed power spectrum for envelopes */

    /* Transient preservation related */
    float *transient_preserv_prev;        /* previous frame smoothed power spectrum for envelopes */
    float tp_r_mean;
    bool transient_present;
    float tp_window_count;

    /* Reduction gains */
    float *Gk;                              /* definitive gain */

    /* Ensemble related */
    float *residual_spectrum;
    float *denoised_spectrum;
    float *final_spectrum;

    /* whitening related */
    float *residual_max_spectrum;
    float max_decay_rate;
    float whitening_window_count;

    /* Loizou algorithm */
    float *auto_thresholds; /* Reference threshold for louizou algorithm */
    float *prev_noise_thresholds;
    float *s_pow_spec;
    float *prev_s_pow_spec;
    float *p_min;
    float *prev_p_min;
    float *speech_p_p;
    float *prev_speech_p_p;

    /* masking */
    float *bark_z;
    float *absolute_thresholds; /* absolute threshold of hearing */
    float *SSF;
    float *spl_reference_values;
    float *unity_gain_bark_spectrum;
    float *spreaded_unity_gain_bark_spectrum;
    float *alpha_masking;
    float *beta_masking;
    float *input_fft_buffer_at;
    float *output_fft_buffer_at;
    fftwf_plan forward_at;
};

static void
denoise_init(Denoise *filter, float intensity, float noise_offset, float release_time, float whitening_factor,
             float masking, float transient_protection, bool residual_listen, bool adaptive_state,
             bool noise_learn_state);

static bool
denoise_setup(Denoise *filter,
              float rate);


static int
denoise_filter_inplace(Denoise *base_transform,
                       float *buf, int n_samples);


static bool
denoise_stop(Denoise *trans);


#define DEF_RELEASE_TIME        0
#define DEF_INTENSITY            20.0f
#define DEF_NOISE_OFFSET        2
#define DEF_WHITENING_FACTOR        0
#define DEF_MASKING            1.0f
#define DEF_TRANSIENT_PROTECTION    0
#define DEF_RESIDUAL            false
#define DEF_AUTO_LEARN            true
#define DEF_BUILD_NOISE_PROFILE        false
#define DEF_NOISE_FILE            "noise.fft"

static void
denoise_init(Denoise *filter, float intensity, float noise_offset, float release_time, float whitening_factor,
             float masking, float transient_protection, bool residual_listen, bool adaptive_state,
             bool noise_learn_state) {
    Denoise *self;
    self = (filter);

    self->release = release_time;  /* DEF_RELEASE_TIME; */ ;
    self->amount_of_reduction = intensity;    /* DEF_INTENSITY; */
    self->noise_thresholds_offset = noise_offset; /* DEF_NOISE_OFFSET; */
    self->whitening_factor = whitening_factor; /* DEF_WHITENING_FACTOR; */
    self->masking = masking;/* DEF_MASKING; */
    self->transient_protection = transient_protection;/* DEF_TRANSIENT_PROTECTION; */
    self->residual_listen = residual_listen; /* DEF_RESIDUAL; */
    self->adaptive_state = adaptive_state; /* DEF_AUTO_LEARN; */
    self->noise_learn_state = noise_learn_state;  /* DEF_BUILD_NOISE_PROFILE; */

    self->file_location = strdup(DEF_NOISE_FILE);
}


static bool
denoise_setup(Denoise *filter,
              float rate) {
    Denoise *self = filter;
    /* Sampling related */
    self->samp_rate = rate;

    /* FFT related */
    self->fft_size = FFT_SIZE;
    self->fft_size_2 = self->fft_size / 2;
    self->input_fft_buffer = (float *) calloc(self->fft_size, sizeof(float));
    self->output_fft_buffer = (float *) calloc(self->fft_size, sizeof(float));
    self->forward = fftwf_plan_r2r_1d(self->fft_size, self->input_fft_buffer, self->output_fft_buffer, FFTW_R2HC,
                                      FFTW_ESTIMATE);
    self->backward = fftwf_plan_r2r_1d(self->fft_size, self->output_fft_buffer, self->input_fft_buffer, FFTW_HC2R,
                                       FFTW_ESTIMATE);

    /* STFT window related */
    self->window_option_input = INPUT_WINDOW;
    self->window_option_output = OUTPUT_WINDOW;
    self->input_window = (float *) calloc(self->fft_size, sizeof(float));
    self->output_window = (float *) calloc(self->fft_size, sizeof(float));

    /* buffers for OLA */
    self->in_fifo = (float *) calloc(self->fft_size, sizeof(float));
    self->out_fifo = (float *) calloc(self->fft_size, sizeof(float));
    self->output_accum = (float *) calloc(self->fft_size * 2, sizeof(float));
    self->overlap_factor = OVERLAP_FACTOR;
    self->hop = (int) (self->fft_size / self->overlap_factor);
    self->input_latency = self->fft_size - self->hop;
    self->read_ptr = self->input_latency; /* the initial position because we are that many samples ahead */

    /* soft bypass */
    self->tau = (1.f - expf(-2.f * M_PI * 25.f * 64.f / self->samp_rate));
    self->wet_dry = 0.f;

    /* Arrays for getting bins info */
    self->fft_p2 = (float *) calloc((self->fft_size_2 + 1), sizeof(float));
    self->fft_magnitude = (float *) calloc((self->fft_size_2 + 1), sizeof(float));
    self->fft_phase = (float *) calloc((self->fft_size_2 + 1), sizeof(float));

    /* noise threshold related */
    self->noise_thresholds_p2 = (float *) calloc((self->fft_size_2 + 1), sizeof(float));
    self->noise_thresholds_scaled = (float *) calloc((self->fft_size_2 + 1), sizeof(float));
    self->noise_window_count = 0.f;
    self->noise_thresholds_availables = false;

    /* noise adaptive estimation related */
    self->auto_thresholds = (float *) calloc((self->fft_size_2 + 1), sizeof(float));
    self->prev_noise_thresholds = (float *) calloc((self->fft_size_2 + 1), sizeof(float));
    self->s_pow_spec = (float *) calloc((self->fft_size_2 + 1), sizeof(float));
    self->prev_s_pow_spec = (float *) calloc((self->fft_size_2 + 1), sizeof(float));
    self->p_min = (float *) calloc((self->fft_size_2 + 1), sizeof(float));
    self->prev_p_min = (float *) calloc((self->fft_size_2 + 1), sizeof(float));
    self->speech_p_p = (float *) calloc((self->fft_size_2 + 1), sizeof(float));
    self->prev_speech_p_p = (float *) calloc((self->fft_size_2 + 1), sizeof(float));

    /* smoothing related */
    self->smoothed_spectrum = (float *) calloc((self->fft_size_2 + 1), sizeof(float));
    self->smoothed_spectrum_prev = (float *) calloc((self->fft_size_2 + 1), sizeof(float));

    /* transient preservation */
    self->transient_preserv_prev = (float *) calloc((self->fft_size_2 + 1), sizeof(float));
    self->tp_window_count = 0.f;
    self->tp_r_mean = 0.f;
    self->transient_present = false;

    /* masking related */
    self->bark_z = (float *) calloc((self->fft_size_2 + 1), sizeof(float));
    self->absolute_thresholds = (float *) calloc((self->fft_size_2 + 1), sizeof(float));
    self->unity_gain_bark_spectrum = (float *) calloc(N_BARK_BANDS, sizeof(float));
    self->spreaded_unity_gain_bark_spectrum = (float *) calloc(N_BARK_BANDS, sizeof(float));
    self->spl_reference_values = (float *) calloc((self->fft_size_2 + 1), sizeof(float));
    self->alpha_masking = (float *) calloc((self->fft_size_2 + 1), sizeof(float));
    self->beta_masking = (float *) calloc((self->fft_size_2 + 1), sizeof(float));
    self->SSF = (float *) calloc((N_BARK_BANDS * N_BARK_BANDS), sizeof(float));
    self->input_fft_buffer_at = (float *) calloc(self->fft_size, sizeof(float));
    self->output_fft_buffer_at = (float *) calloc(self->fft_size, sizeof(float));
    self->forward_at = fftwf_plan_r2r_1d(self->fft_size, self->input_fft_buffer_at, self->output_fft_buffer_at,
                                         FFTW_R2HC, FFTW_ESTIMATE);

    /* reduction gains related */
    self->Gk = (float *) calloc((self->fft_size), sizeof(float));

    /* whitening related */
    self->residual_max_spectrum = (float *) calloc((self->fft_size), sizeof(float));
    self->max_decay_rate = expf(-1000.f / (((WHITENING_DECAY_RATE) * self->samp_rate) / self->hop));
    self->whitening_window_count = 0.f;

    /* final ensemble related */
    self->residual_spectrum = (float *) calloc((self->fft_size), sizeof(float));
    self->denoised_spectrum = (float *) calloc((self->fft_size), sizeof(float));
    self->final_spectrum = (float *) calloc((self->fft_size), sizeof(float));

    /* Window combination initialization (pre processing window post processing window) */
    fft_pre_and_post_window(self->input_window, self->output_window,
                            self->fft_size, self->window_option_input,
                            self->window_option_output, &self->overlap_scale_factor);

    /* Set initial gain as unity for the positive part */
    initialize_array(self->Gk, 1.f, self->fft_size);

    /* Compute adaptive initial thresholds */
    compute_auto_thresholds(self->auto_thresholds, (float) self->fft_size, (float) self->fft_size_2,
                            self->samp_rate);

    /* MASKING initializations */
    compute_bark_mapping(self->bark_z, self->fft_size_2, (int) self->samp_rate);
    compute_absolute_thresholds(self->absolute_thresholds, self->fft_size_2,
                                (int) self->samp_rate);
    spl_reference(self->spl_reference_values, self->fft_size_2, (int) self->samp_rate,
                  self->input_fft_buffer_at, self->output_fft_buffer_at,
                  &self->forward_at);
    compute_SSF(self->SSF);

    /* Initializing unity gain values for offset normalization */
    initialize_array(self->unity_gain_bark_spectrum, 1.f, N_BARK_BANDS);
    /* Convolve unitary energy bark spectrum with SSF */
    convolve_with_SSF(self->SSF, self->unity_gain_bark_spectrum,
                      self->spreaded_unity_gain_bark_spectrum);

    initialize_array(self->alpha_masking, 1.f, self->fft_size_2 + 1);
    initialize_array(self->beta_masking, 0.f, self->fft_size_2 + 1);

    /* Added */
    self->enable = 1.f;

    /* If configured, load a noise profile from a file */
    if (!self->noise_learn_state && !self->adaptive_state) {
        printf("Will load noise data from file: %s \n", self->file_location);
        FILE *save_file = fopen(self->file_location, "r");
        if (0 != save_file) {
            printf("Failed loading noise file\n");
            const int BUFF_SIZE = 32;
            char str_buff[BUFF_SIZE];
            for (int i = 0; i < self->fft_size_2 + 1; i++) {
                fgets(str_buff, BUFF_SIZE, save_file);
                self->noise_thresholds_p2[i] = (float) atof(str_buff);
                memset(str_buff, 0, sizeof(str_buff));
            }
            self->noise_thresholds_availables = true;
            printf("Noise print fully loaded from file\n");
        }
    }

    return (true);
}


char *
strdup_printf(const char *fmt, ...) {
    /* Guess we need no more than 100 bytes. */
    int n, size = 100;
    char *p;
    va_list ap;

    if ((p = malloc(size)) == NULL)
        return (NULL);
    while (1) {
        /* Try to print in the allocated space. */
        va_start(ap, fmt);
#ifdef WIN32
        n = _vsnprintf(p, size, fmt, ap);
#else
        n = vsnprintf( p, size, fmt, ap );
#endif
        va_end(ap);
        /* If that worked, return the string. */
        if (n > -1 && n < size)
            return (p);
        /* Else try again with more space. */
        if (n > -1)           /* glibc 2.1 */
            size = n + 1;   /* precisely what is needed */
        else                    /* glibc 2.0 */
            size *= 2;      /* twice the old size */
        if ((p = realloc(p, size)) == NULL)
            return (NULL);
    }
}


static bool
denoise_stop(Denoise *trans) {
    printf("Stopping!\n");
    Denoise *self = (trans);

    /* If we have a noise profile, save it to a file */
    if (self->noise_learn_state && self->noise_thresholds_availables) {
        /*
         * get_noise_statistics(self->fft_p2, self->fft_size_2,
         * self->noise_thresholds_p2, self->noise_window_count);
         */
        printf("Will save noise data to file\n");
        FILE *save_file = fopen(self->file_location, "w");
        if (0 != save_file) {
            printf("Failed saving noise file\n");
        } else {
            for (int i = 0; i < self->fft_size_2 + 1; i++) {
                char *str = strdup_printf("%f\n", self->noise_thresholds_p2[i]);
                fputs(str, save_file);
                free(str);
            }
            fclose(save_file);
            printf("Noise print saved\n");
        }
    }

    /* Cleanup memory */
    free(self->file_location);

    free(self->input_fft_buffer);
    free(self->output_fft_buffer);
    fftwf_destroy_plan(self->forward);
    fftwf_destroy_plan(self->backward);

    /* STFT window related */
    free(self->input_window);
    free(self->output_window);

    /* buffers for OLA */
    free(self->in_fifo);
    free(self->out_fifo);
    free(self->output_accum);

    /* Arrays for getting bins info */
    free(self->fft_p2);
    free(self->fft_magnitude);
    free(self->fft_phase);

    /* noise threshold related */
    free(self->noise_thresholds_p2);
    free(self->noise_thresholds_scaled);

    /* noise adaptive estimation related */
    free(self->auto_thresholds);
    free(self->prev_noise_thresholds);
    free(self->s_pow_spec);
    free(self->prev_s_pow_spec);
    free(self->p_min);
    free(self->prev_p_min);
    free(self->speech_p_p);
    free(self->prev_speech_p_p);

    /* smoothing related */
    free(self->smoothed_spectrum);
    free(self->smoothed_spectrum_prev);

    /* transient preservation */
    free(self->transient_preserv_prev);

    /* masking related */
    free(self->bark_z);
    free(self->absolute_thresholds);
    free(self->unity_gain_bark_spectrum);
    free(self->spreaded_unity_gain_bark_spectrum);
    free(self->spl_reference_values);
    free(self->alpha_masking);
    free(self->beta_masking);
    free(self->SSF);
    free(self->input_fft_buffer_at);
    free(self->output_fft_buffer_at);
    fftwf_destroy_plan(self->forward_at);

    /* reduction gains related */
    free(self->Gk);

    /* whitening related */
    free(self->residual_max_spectrum);

    /* final ensemble related */
    free(self->residual_spectrum);
    free(self->denoised_spectrum);
    free(self->final_spectrum);

    return (true);
}


static int
denoise_filter_inplace(Denoise *base_transform,
                       float *buf, int n_samples) {
    Denoise *self = (base_transform);
    int flow = 1;
    /* Softbypass targets in case of disabled or enabled */
    if (self->enable == 0.f)      /* if disabled */
    {
        self->wet_dry_target = 0.f;
    } else {                        /* if enabled */
        self->wet_dry_target = 1.f;
    }
    /* Interpolate parameters over time softly to bypass without clicks or pops */
    self->wet_dry += self->tau * (self->wet_dry_target - self->wet_dry) + FLT_MIN;

    /* Parameters values */


    /*exponential decay coefficients for envelopes and adaptive noise profiling
     *  These must take into account the hop size as explained in the following paper
     *  FFT-BASED DYNAMIC RANGE COMPRESSION*/
    if (self->release != 0.f)             /* This allows to turn off smoothing with 0 ms in order to use masking only */
    {
        self->release_coeff = expf(-1000.f / (((self->release) * self->samp_rate) / self->hop));
    } else {
        self->release_coeff = 0.f;      /* This avoids incorrect results when moving sliders rapidly */
    }

    self->amount_of_reduction_linear = from_dB(-1.f * self->amount_of_reduction);
    self->thresholds_offset_linear = from_dB(self->noise_thresholds_offset);
    self->whitening_factor = self->whitening_factor_pc / 100.f;


    int k = 0;
    float *samples = (float *) buf;

    /* main loop for processing */
    for (int pos = 0; pos < n_samples; pos++) {
        /*
         * Store samples int the input buffer
         * self->in_fifo[self->read_ptr] = self->input[pos];
         */
        self->in_fifo[self->read_ptr] = samples[pos];
        /*
         * Output samples in the output buffer (even zeros introduced by latency)
         * self->output[pos] = self->out_fifo[self->read_ptr - self->input_latency];
         */
        samples[pos] = self->out_fifo[self->read_ptr - self->input_latency];
        /* Now move the read pointer */
        self->read_ptr++;

        /* Once the buffer is full we can do stuff */
        if (self->read_ptr < self->fft_size)
            continue;
        /*
         * Reset the input buffer position
         */
        self->read_ptr = self->input_latency;

        /* ----------STFT Analysis------------ */

        /* Adding and windowing the frame input values in the center (zero-phasing) */
        for (k = 0; k < self->fft_size; k++) {
            self->input_fft_buffer[k] = self->in_fifo[k] * self->input_window[k];
        }

        /* ----------FFT Analysis------------ */

        /* Do transform */
        fftwf_execute(self->forward);

        /* -----------GET INFO FROM BINS-------------- */

        get_info_from_bins(self->fft_p2, self->fft_magnitude, self->fft_phase,
                           self->fft_size_2, self->fft_size,
                           self->output_fft_buffer);

        /* ///////////////////SPECTRAL PROCESSING////////////////////////// */


        /*This section countains the specific noise reduction processing blocks
         *  but it could be replaced with any spectral processing (I'm looking at you future tinkerer)
         *  Parameters for the STFT transform can be changed at the top of this file
         */

        /* If the spectrum is not silence */
        if (!is_empty(self->fft_p2, self->fft_size_2)) {
            /* If adaptive noise is selected the noise is adapted in time */
            if (self->adaptive_state) {
                /* This has to be revised(issue 8 on github) */
                adapt_noise(self->fft_p2, self->fft_size_2, self->noise_thresholds_p2,
                            self->auto_thresholds, self->prev_noise_thresholds,
                            self->s_pow_spec, self->prev_s_pow_spec, self->p_min,
                            self->prev_p_min, self->speech_p_p, self->prev_speech_p_p);

                self->noise_thresholds_availables = true;
            }


            /*If selected estimate noise spectrum is based on selected portion of signal
             * *do not process the signal
             */
            if (self->noise_learn_state) /* MANUAL */

            { /* Increase window count for rolling mean */
                self->noise_window_count++;

                get_noise_statistics(self->fft_p2, self->fft_size_2,
                                     self->noise_thresholds_p2, self->noise_window_count);

                self->noise_thresholds_availables = true;
            } else {
                /* If there is a noise profile reduce noise */
                if (self->noise_thresholds_availables == true) {
                    /* Detector smoothing and oversubtraction */
                    preprocessing(self->thresholds_offset_linear, self->fft_p2,
                                  self->noise_thresholds_p2, self->noise_thresholds_scaled,
                                  self->smoothed_spectrum, self->smoothed_spectrum_prev,
                                  self->fft_size_2, self->bark_z, self->absolute_thresholds,
                                  self->SSF, self->release_coeff,
                                  self->spreaded_unity_gain_bark_spectrum,
                                  self->spl_reference_values, self->alpha_masking,
                                  self->beta_masking, self->masking, self->adaptive_state,
                                  self->amount_of_reduction_linear, self->transient_preserv_prev,
                                  &self->tp_window_count, &self->tp_r_mean,
                                  &self->transient_present, self->transient_protection);

                    /* Supression rule */
                    spectral_gain(self->fft_p2, self->noise_thresholds_p2,
                                  self->noise_thresholds_scaled, self->smoothed_spectrum,
                                  self->fft_size_2, self->adaptive_state, self->Gk,
                                  self->transient_protection, self->transient_present);

                    /* apply gains */
                    denoised_calulation(self->fft_size, self->output_fft_buffer,
                                        self->denoised_spectrum, self->Gk);

                    /* residual signal */
                    residual_calulation(self->fft_size, self->output_fft_buffer,
                                        self->residual_spectrum, self->denoised_spectrum,
                                        self->whitening_factor, self->residual_max_spectrum,
                                        &self->whitening_window_count, self->max_decay_rate);

                    /* Ensemble the final spectrum using residual and denoised */
                    final_spectrum_ensemble(self->fft_size, self->final_spectrum,
                                            self->residual_spectrum,
                                            self->denoised_spectrum,
                                            self->amount_of_reduction_linear,
                                            self->residual_listen);

                    soft_bypass(self->final_spectrum, self->output_fft_buffer,
                                self->wet_dry, self->fft_size);
                }
            }
        }

        /* ///////////////////////////////////////////////////////// */

        /* ----------STFT Synthesis------------ */

        /* ------------FFT Synthesis------------- */

        /* Do inverse transform */
        fftwf_execute(self->backward);

        /* Normalizing value */
        for (k = 0; k < self->fft_size; k++) {
            self->input_fft_buffer[k] = self->input_fft_buffer[k] / self->fft_size;
        }

        /* ------------OVERLAPADD------------- */

        /* Windowing scaling and accumulation */
        for (k = 0; k < self->fft_size; k++) {
            self->output_accum[k] += (self->output_window[k] * self->input_fft_buffer[k]) /
                                     (self->overlap_scale_factor * self->overlap_factor);
        }

        /* Output samples up to the hop size */
        for (k = 0; k < self->hop; k++) {
            self->out_fifo[k] = self->output_accum[k];
        }

        /* shift FFT accumulator the hop size */
        memmove(self->output_accum, self->output_accum + self->hop,
                self->fft_size * sizeof(float));

        /* move input FIFO */
        for (k = 0; k < self->input_latency; k++) {
            self->in_fifo[k] = self->in_fifo[k + self->hop];
        }
        /* ------------------------------- */
    }
    return (1);
}


void wavWrite_f32(char *filename, float *buffer, int sampleRate, uint32_t totalSampleCount) {
    drwav_data_format format;
    format.container = drwav_container_riff;
    format.format = DR_WAVE_FORMAT_IEEE_FLOAT;
    format.channels = 1;
    format.sampleRate = (drwav_uint32) sampleRate;
    format.bitsPerSample = 32;
    drwav *pWav = drwav_open_file_write(filename, &format);
    if (pWav) {
        drwav_uint64 samplesWritten = drwav_write(pWav, totalSampleCount, buffer);
        drwav_uninit(pWav);
        if (samplesWritten != totalSampleCount) {
            fprintf(stderr, "ERROR\n");
            exit(1);
        }
    }
}


float *wavRead_f32(char *filename, uint32_t *sampleRate, uint64_t *totalSampleCount) {
    unsigned int channels;
    float *buffer = drwav_open_and_read_file_f32(filename, &channels, sampleRate, totalSampleCount);
    if (buffer == NULL) {
        fprintf(stderr, "ERROR\n");
        exit(1);
    }
    if (channels != 1) {
        drwav_free(buffer);
        buffer = NULL;
        *sampleRate = 0;
        *totalSampleCount = 0;
    }
    return (buffer);
}


void splitpath(const char *path, char *drv, char *dir, char *name, char *ext) {
    const char *end;
    const char *p;
    const char *s;
    if (path[0] && path[1] == ':') {
        if (drv) {
            *drv++ = *path++;
            *drv++ = *path++;
            *drv = '\0';
        }
    } else if (drv)
        *drv = '\0';
    for (end = path; *end && *end != ':';)
        end++;
    for (p = end; p > path && *--p != '\\' && *p != '/';)
        if (*p == '.') {
            end = p;
            break;
        }
    if (ext)
        for (s = end; (*ext = *s++);)
            ext++;
    for (p = end; p > path;)
        if (*--p == '\\' || *p == '/') {
            p++;
            break;
        }
    if (name) {
        for (s = p; s < end;)
            *name++ = *s++;
        *name = '\0';
    }
    if (dir) {
        for (s = path; s < p;)
            *dir++ = *s++;
        *dir = '\0';
    }
}


void denoise_proc(float sampleRate, float *input, uint32_t buffen_len) {
    Denoise filter;
    float intensity = DEF_INTENSITY;
    float noise_offset = DEF_NOISE_OFFSET;
    float release_time = DEF_RELEASE_TIME;
    float whitening_factor = DEF_WHITENING_FACTOR;
    float masking = DEF_MASKING;
    float transient_protection = DEF_TRANSIENT_PROTECTION;
    bool residual_listen = DEF_RESIDUAL;
    bool adaptive_state = DEF_AUTO_LEARN;
    bool noise_learn_state = DEF_BUILD_NOISE_PROFILE;
    denoise_init(&filter, intensity, noise_offset, release_time, whitening_factor, masking, transient_protection,
                 residual_listen, adaptive_state, noise_learn_state);
    denoise_setup(&filter, sampleRate);
    denoise_filter_inplace(&filter, input, buffen_len);
    denoise_stop(&filter);
}

void DeNoise(char *in_file, char *out_file) {
    uint32_t in_sampleRate = 0;
    uint64_t in_size = 0;
    float *data_in = wavRead_f32(in_file, &in_sampleRate, &in_size);
    if (data_in != NULL) {
        denoise_proc(in_sampleRate, data_in, in_size);
        wavWrite_f32(out_file, data_in, in_sampleRate, (uint32_t) in_size);
        free(data_in);
    }
}


int main(int argc, char **argv) {
    printf("Noise Repellent \n");
    printf("blog:http://cpuimage.cnblogs.com/\n");
    printf("e-mail:gaozhihan@vip.qq.com\n");
    if (argc < 2)
        return (-1);

    char *in_file = argv[1];
    char drive[3];
    char dir[256];
    char fname[256];
    char ext[256];
    char out_file[1024];
    splitpath(in_file, drive, dir, fname, ext);
    sprintf(out_file, "%s%s%s_out%s", drive, dir, fname, ext);
    DeNoise(in_file, out_file);
    printf("press any key to exit.\n");
    getchar();
    return (0);
}


