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


/**
 * \file spectral_processing.c
 * \author Luciano Dato
 * \brief All methods related to spectral processing the spectrum
 */
#include <fftw3.h>
#include <float.h>
#include <math.h>


/**
 * \file estimate_noise_spectrum.c
 * \author Luciano Dato
 * \brief Methods for noise spectrum estimation
 */


/**
 * \file extra_functions.c
 * \author Luciano Dato
 * \brief Extra methods used by others. This keeps clean other files.
 */

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdio.h>

#define MAX( a, b )	( ( (a) > (b) ) ? (a) : (b) )
#define MIN( a, b )	( ( (a) < (b) ) ? (a) : (b) )

/* Window types */
#define HANN_WINDOW	0
#define HAMMING_WINDOW	1
#define BLACKMAN_WINDOW 2
#define VORBIS_WINDOW	3

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#define SP_MAX_NUM	100             /* Max number of spectral peaks to find */
#define SP_THRESH	0.1f            /* Threshold to discriminate peaks (high value to discard noise) Linear 0<>1 */
#define SP_USE_P_INTER	true            /* Use parabolic interpolation */
#define SP_MAX_FREQ	16000.f         /* Highest frequency to search for peaks */
#define SP_MIN_FREQ	40.f            /* Lowest frequency to search for peaks */

#define SE_RESOLUTION 100.f             /* Spectral envelope resolution */

#define WHITENING_DECAY_RATE	1000.f  /* Deacay in ms for max spectrum for whitening */
#define WHITENING_FLOOR		0.02f   /* Minumum max value posible */

#define TP_UPPER_LIMIT 5.f              /* This correspond to the upper limit of the adaptive threshold multiplier. Should be the same as the ttl configured one */


/**
 * Method to force already-denormal float value to zero.
 * \param value to sanitize
 */
float sanitize_denormal( float value )
{
    if ( isnan( value ) )
    {
        return(FLT_MIN); /* to avoid log errors */
    } else {
        return(value);
    }
}


/* /sign function. */
int sign( float x )
{
    return(x >= 0.f ? 1.f : -1.f);
}


/* /gets the next power of two of a number x. */
int next_pow_two( int x )
{
    int power = 2;
    while ( x >>= 1 )
        power <<= 1;
    return(power);
}


/* /gets the nearest odd number of a number x. */
int nearest_odd( int x )
{
    if ( x % 2 == 0 )
        return(x + 1);
    else
        return(x);
}


/* /gets the nearest even number of a number x. */
int nearest_even( int x )
{
    if ( x % 2 == 0 )
        return(x);
    else
        return(x - 1);
}


/* /converts a db value to linear scale. */
float from_dB( float gdb )
{
    return(expf( gdb / 10.f * logf( 10.f ) ) );
}


/* /converts a linear value to db scale. */
float to_dB( float g )
{
    return(10.f * log10f( g ) );
}


/*Maps a bin number to a frequency
 * \param i bin number
 * \param samp_rate current sample rate of the host
 * \param N size of the fftw
 */
float bin_to_freq( int i, float samp_rate, int N )
{
    return( (float) i * (samp_rate / N / 2.f) );
}


/*Maps a frequency to a bin number
 * \param freq frequency
 * \param samp_rate current sample rate of the host
 * \param N size of the fftw
 */
int freq_to_bin( float freq, float samp_rate, int N )
{
    return( (int) (freq / (samp_rate / N / 2.f) ) );
}


/* ---------SPECTRAL METHODS------------- */


/**
 * FFT peak struct. To describe spectral peaks.
 */
typedef struct {
    float	magnitude;
    int	position;
} FFTPeak;


/**
 * Parabolic interpolation as explained in  https://ccrma.stanford.edu/~jos/parshl/Peak_Detection_Steps_3.html.
 * This is used for more precise spectral peak detection.
 * \param left_val value at the left of the point to interpolate
 * \param middle_val value at the middle of the point to interpolate
 * \param right_val value at the right of the point to interpolate
 * \param current_bin current bin value before interpolation
 * \param result_val interpolation value result
 * \param result_val interpolation bin result
 */
void parabolic_interpolation( float left_val, float middle_val, float right_val,
                              int current_bin, float *result_val, int *result_bin )
{
    float delta_x = 0.5 * ( (left_val - right_val) / (left_val - 2.f * middle_val + right_val) );
    *result_bin	= current_bin + (int) delta_x;
    *result_val	= middle_val - 0.25 * (left_val - right_val) * delta_x;
}


/**
 * To initialize an array to a single value in all positions.
 * \param array the array to initialize
 * \param value the value to copy to every position in the array
 * \param size the size of the array
 */
void initialize_array( float *array, float value, int size )
{
    for ( int k = 0; k < size; k++ )
    {
        array[k] = value;
    }
}


/**
 * Verifies if the spectrum is full of zeros.
 * \param spectrum the array to check
 * \param N the size of the array (half the fft size plus 1)
 */
bool is_empty( float *spectrum, int N )
{
    int k;
    for ( k = 0; k <= N; k++ )
    {
        if ( spectrum[k] > FLT_MIN )
        {
            return(false);
        }
    }
    return(true);
}


/**
 * Finds the max value of the spectrum.
 * \param spectrum the array to check
 * \param N the size of the array (half the fft size plus 1)
 */
float max_spectral_value( float *spectrum, int N )
{
    int	k;
    float	max = spectrum[0];
    for ( k = 0; k <= N; k++ )
    {
        max = MAX( spectrum[k], max );
    }
    return(max);
}


/**
 * Finds the min value of the spectrum.
 * \param spectrum the array to check
 * \param N the size of the array (half the fft size plus 1)
 */
float min_spectral_value( float *spectrum, int N )
{
    int	k;
    float	min = spectrum[0];
    for ( k = 0; k <= N; k++ )
    {
        min = MIN( spectrum[k], min );
    }
    return(min);
}


/**
 * Finds the mean value of the spectrum.
 * \param a the array to check
 * \param m the size of the array (half the fft size plus 1)
 */
float spectral_mean( float *a, int m )
{
    float sum = 0.f;
    for ( int i = 0; i <= m; i++ )
        sum += a[i];
    return(sum / (float) (m + 1) );
}


/**
 * Sums of all values of a spectrum.
 * \param a the array to sum
 * \param m the size of the array (half the fft size plus 1)
 */
float spectral_addition( float *a, int m )
{
    float sum = 0.f;
    for ( int i = 0; i <= m; i++ )
        sum += a[i];
    return(sum);
}


/**
 * Finds the median value of the spectrum.
 * \param x the array to check
 * \param n the size of the array (half the fft size plus 1)
 */
float spectral_median( float *x, int n )
{
    float	temp;
    int	i, j;
    float	tmp[n + 1];
    memcpy( tmp, x, sizeof(float) * (n + 1) );
    /* the following two loops sort the array x in ascending order */
    for ( i = 0; i < n; i++ )
    {
        for ( j = i + 1; j <= n; j++ )
        {
            if ( tmp[j] < tmp[i] )
            {
                /* swap elements */
                temp	= tmp[i];
                tmp[i]	= tmp[j];
                tmp[j]	= temp;
            }
        }
    }

    if ( n % 2 == 0 )
    {
        /* if there is an even number of elements, return mean of the two elements in the middle */
        return( (tmp[n / 2] + tmp[n / 2 - 1]) / 2.f);
    } else {
        /* else return the element in the middle */
        return(tmp[n / 2]);
    }
}


/**
 * Finds the moda value of the spectrum.
 * \param x the array to check
 * \param n the size of the array (half the fft size plus 1)
 */
float spectral_moda( float *x, int n )
{
    float	temp[n];
    int	i, j, pos_max;
    float	max;

    for ( i = 0; i < n; i++ )
    {
        temp[i] = 0.f;
    }

    for ( i = 0; i < n; i++ )
    {
        for ( j = i; j < n; j++ )
        {
            if ( x[j] == x[i] )
                temp[i]++;
        }
    }

    max	= temp[0];
    pos_max = 0;
    for ( i = 0; i < n; i++ )
    {
        if ( temp[i] > max )
        {
            pos_max = i;
            max	= temp[i];
        }
    }
    return(x[pos_max]);
}


/**
 * Normalizes spectral values.
 * \param spectrum the spectrum to normalize
 * \param N the size of the spectrum (half the fft size plus 1)
 */
void get_normalized_spectum( float *spectrum, int N )
{
    int	k;
    float	max_value	= max_spectral_value( spectrum, N );
    float	min_value	= min_spectral_value( spectrum, N );

    /* Normalizing the noise print */
    for ( k = 0; k <= N; k++ )
    {
        spectrum[k] = (spectrum[k] - min_value) / (max_value - min_value);
    }
}


/**
 * Outputs the spectral flux between two spectrums.
 * \param spectrum the current power spectrum
 * \param spectrum_prev the previous power spectrum
 * \param N the size of the spectrum (half the fft size plus 1)
 */
float spectral_flux( float *spectrum, float *spectrum_prev, float N )
{
    int	i;
    float	spectral_flux = 0.f;
    float	temp;

    for ( i = 0; i <= N; i++ )
    {
        temp		= sqrtf( spectrum[i] ) - sqrtf( spectrum_prev[i] ); /* Recieves power spectrum uses magnitude */
        spectral_flux	+= (temp + fabs( temp ) ) / 2.f;
    }
    return(spectral_flux);
}


/**
 * Outputs the high frequency content of the spectrum.
 * \param spectrum the current power spectrum
 * \param N the size of the spectrum (half the fft size plus 1)
 */
float high_frequency_content( float *spectrum, float N )
{
    int	i;
    float	sum = 0.f;

    for ( i = 0; i <= N; i++ )
    {
        sum += i * spectrum[i];
    }
    return(sum / (float) (N + 1) );
}


/**
 * Computes the spectral envelope like Robel 'Efficient Spectral Envelope Estimation and its
 * application to pitch shifting and envelope preservation' indicates.
 * \param fft_size_2 half of the fft size
 * \param fft_p2 the current power spectrum
 * \param samp_rate current sample rate of the host
 * \param spectral_envelope_values array that holds the spectral envelope values
 */
void spectral_envelope( int fft_size_2, float *fft_p2, int samp_rate, float *spectral_envelope_values )
{
    int k;

    /* compute envelope */
    int	spec_size	= fft_size_2 + 1;
    float	spectral_range	= bin_to_freq( spec_size, samp_rate, fft_size_2 * 2 );
    int	hop		= (int) freq_to_bin( SE_RESOLUTION, samp_rate, fft_size_2 * 2 );        /* Experimental */

    for ( k = 0; k <= fft_size_2; k += hop )
    {
        float freq = bin_to_freq( k, samp_rate, fft_size_2 * 2 );

        float	bf	= freq - MAX( 50.0, freq * 0.34 );                                      /* 0.66 */
        float	ef	= freq + MAX( 50.0, freq * 0.58 );                                      /* 1.58 */
        int	b	= (int) (bf / spectral_range * (spec_size - 1.0) + 0.5);
        int	e	= (int) (ef / spectral_range * (spec_size - 1.0) + 0.5);
        b	= MAX( b, 0 );
        b	= MIN( spec_size - 1, b );
        e	= MAX( e, b + 1 );
        e	= MIN( spec_size, e );
        float	c			= b / 2.0 + e / 2.0;
        float	half_window_length	= e - c;

        float	n	= 0.0;
        float	wavg	= 0.0;

        for ( int i = b; i < e; ++i )
        {
            float weight = 1.0 - fabs( (float) (i) - c ) / half_window_length;
            weight	*= weight;
            weight	*= weight;
            float spectrum_energy_val = fft_p2[i]; /* * fft_p2[i]; */
            weight	*= spectrum_energy_val;
            wavg	+= spectrum_energy_val * weight;
            n	+= weight;
        }
        if ( n != 0.0 )
            wavg /= n;

        /* final value */
        spectral_envelope_values[k] = wavg; /* sqrtf(wavg); */
    }
}


/**
 * Finds the spectral peaks of a spectrum. (Not used in current version)
 * \param fft_size_2 half of the fft size
 * \param fft_p2 the current power spectrum
 * \param spectral_peaks array of resulting spectral peaks founded
 * \param peak_pos array of positions of resulting spectral peaks
 * \param peaks_count counter of peaks founded
 * \param samp_rate current sample rate of the host
 */
void spectral_peaks( int fft_size_2, float *fft_p2, FFTPeak *spectral_peaks, int *peak_pos,
                     int *peaks_count, int samp_rate )
{
    int	k;
    float	fft_magnitude_db[fft_size_2 + 1];
    float	peak_threshold_db	= to_dB( SP_THRESH );
    int	max_bin			= MIN( freq_to_bin( SP_MAX_FREQ, samp_rate, fft_size_2 * 2 ), fft_size_2 + 1 );
    int	min_bin			= MAX( freq_to_bin( SP_MIN_FREQ, samp_rate, fft_size_2 * 2 ), 0 );
    int	result_bin;
    float	result_val;

    /* Get the magnitude spectrum in dB scale (twise as precise than using linear scale) */
    for ( k = 0; k <= fft_size_2; k++ )
    {
        fft_magnitude_db[k] = to_dB( sqrtf( fft_p2[k] ) );
    }

    /* index for the magnitude array */
    int i = min_bin;

    /* Index for peak array */
    k = 0;

    /* The zero bin could be a peak */
    if ( i + 1 < fft_size_2 + 1 && fft_magnitude_db[i] > fft_magnitude_db[i + 1] )
    {
        if ( fft_magnitude_db[i] > peak_threshold_db )
        {
            spectral_peaks[k].position	= i;
            spectral_peaks[k].magnitude	= sqrtf( from_dB( fft_magnitude_db[i] ) );
            peak_pos[i]			= 1;
            k++;
        }
    }

    /* Peak finding loop */
    while ( k < SP_MAX_NUM || i < max_bin )
    {
        /* descending a peak */
        while ( i + 1 < fft_size_2 && fft_magnitude_db[i] >= fft_magnitude_db[i + 1] )
        {
            i++;
        }
        /* ascending a peak */
        while ( i + 1 < fft_size_2 && fft_magnitude_db[i] < fft_magnitude_db[i + 1] )
        {
            i++;
        }

        /* when reaching a peak verify that is one value peak or multiple values peak */
        int j = i;
        while ( j + 1 < fft_size_2 && (fft_magnitude_db[j] == fft_magnitude_db[j + 1]) )
        {
            j++;
        }

        /* end of the flat peak if the peak decreases is really a peak otherwise it is not */
        if ( j + 1 < fft_size_2 && fft_magnitude_db[j + 1] < fft_magnitude_db[j] &&
             fft_magnitude_db[j] > peak_threshold_db )
        {
            result_bin	= 0.0;
            result_val	= 0.0;

            if ( j != i )                                   /* peak between i and j */
            {
                if ( SP_USE_P_INTER )
                {
                    result_bin = (i + j) * 0.5;     /* center bin of the flat peak */
                } else {
                    result_bin = i;
                }
                result_val = fft_magnitude_db[i];
            } else {                                        /* interpolate peak at i-1, i and i+1 */
                if ( SP_USE_P_INTER )
                {
                    parabolic_interpolation( fft_magnitude_db[j - 1], fft_magnitude_db[j], fft_magnitude_db[j + 1], j,
                                             &result_val, &result_bin );
                } else {
                    result_bin	= j;
                    result_val	= fft_magnitude_db[j];
                }
            }

            spectral_peaks[k].position	= result_bin;
            spectral_peaks[k].magnitude	= sqrtf( from_dB( result_val ) );
            peak_pos[i]			= 1;
            k++;
        }

        /* if turned out not to be a peak advance i */
        i = j;

        /* If it's the last position of the array */
        if ( i + 1 >= fft_size_2 )
        {
            if ( i == fft_size_2 - 1 && fft_magnitude_db[i - 1] < fft_magnitude_db[i] &&
                 fft_magnitude_db[i + 1] < fft_magnitude_db[i] &&
                 fft_magnitude_db[i] > peak_threshold_db )
            {
                result_bin	= 0.0;
                result_val	= 0.0;
                if ( SP_USE_P_INTER )
                {
                    parabolic_interpolation( fft_magnitude_db[i - 1], fft_magnitude_db[i], fft_magnitude_db[i + 1], j,
                                             &result_val, &result_bin );
                } else {
                    result_bin	= i;
                    result_val	= fft_magnitude_db[i];
                }
                spectral_peaks[k].position	= result_bin;
                spectral_peaks[k].magnitude	= sqrtf( from_dB( result_val ) );
                peak_pos[i]			= 1;
                k++;
            }
            break;
        }
    }
    *peaks_count = k;
    /* printf("%i\n",k ); */
}


/**
 * Outputs the p norm of a spectrum.
 * \param spectrum the power spectrum to get the norm of
 * \param N the size of the array
 * \param p the norm number
 */
float spectrum_p_norm( float *spectrum, float N, float p )
{
    float sum = 0.f;

    for ( int k = 0; k < N; k++ )
    {
        sum += powf( spectrum[k], p );
    }

    return(powf( sum, 1.f / p ) );
}


/* ---------------WHITENING-------------- */


/**
 * Whitens the spectrum adaptively as proposed in 'Adaptive whitening for improved
 * real-time audio onset detection' by Stowell and Plumbley. The idea here is that when
 * residual noise resembles white noise the ear is able to precieve it as not so annoying.
 * It uses a temporal max value for each bin and a decay factor as the memory regulator of
 * that maximun value.
 * \param spectrum the power spectrum of the residue to be whitened
 * \param b the mixing coefficient
 * \param N the size of the array (fft size)
 * \param max_spectrum array of temporal maximums of the residual signal
 * \param max_decay_rate amount of ms of decay for temporal maximums
 */
void spectral_whitening( float *spectrum, float b, int N, float *max_spectrum,
                         float *whitening_window_count, float max_decay_rate )
{
    float whitened_spectrum[N];

    *(whitening_window_count) += 1.f;

    for ( int k = 0; k < N; k++ )
    {
        if ( *(whitening_window_count) > 1.f )
        {
            max_spectrum[k] = MAX( MAX( spectrum[k], WHITENING_FLOOR ), max_spectrum[k] * max_decay_rate );
        } else {
            max_spectrum[k] = MAX( spectrum[k], WHITENING_FLOOR );
        }
    }

    for ( int k = 0; k < N; k++ )
    {
        if ( spectrum[k] > FLT_MIN )
        {
            /* Get whitened spectrum */
            whitened_spectrum[k] = spectrum[k] / max_spectrum[k];

            /* Interpolate between whitened and non whitened residual */
            spectrum[k] = (1.f - b) * spectrum[k] + b * whitened_spectrum[k];
        }
    }
}


/* ---------------TIME SMOOTHING-------------- */


/**
 * Spectral smoothing proposed in 'Spectral subtraction with adaptive averaging of
 * the gain function' but is not used yet.
 * \param fft_size_2 half of the fft size
 * \param spectrum the current power spectrum
 * \param spectrum_prev the previous power spectrum
 * \param noise_thresholds the noise thresholds estimated
 * \param prev_beta beta corresponded to previos frame
 * \param coeff reference smoothing value
 */
void spectrum_adaptive_time_smoothing( int fft_size_2, float *spectrum_prev, float *spectrum,
                                       float *noise_thresholds, float *prev_beta, float coeff )
{
    int	k;
    float	discrepancy, numerator = 0.f, denominator = 0.f;
    float	beta_ts;
    float	beta_smooth;
    float	gamma_ts;

    for ( k = 0; k <= fft_size_2; k++ )
    {
        /* These has to be magnitude spectrums */
        numerator	+= fabs( spectrum[k] - noise_thresholds[k] );
        denominator	+= noise_thresholds[k];
    }
    /* this is the discrepancy of the spectum */
    discrepancy = numerator / denominator;
    /* beta is the adaptive coefficient */
    beta_ts = MIN( discrepancy, 1.f );

    /* Gamma is the smoothing coefficient of the adaptive factor beta */
    if ( *prev_beta < beta_ts )
    {
        gamma_ts = 0.f;
    } else {
        gamma_ts = coeff;
    }

    /* Smoothing beta */
    beta_smooth = gamma_ts * *(prev_beta) + (1.f - gamma_ts) * beta_ts;

    /* copy current value to previous */
    *prev_beta = beta_smooth;

    /* Apply the adaptive smoothed beta over the signal */
    for ( k = 0; k <= fft_size_2; k++ )
    {
        spectrum[k] = (1.f - beta_smooth) * spectrum_prev[k] + beta_smooth * spectrum[k];
    }
}


/**
 * Spectral time smoothing by applying a release envelope. This seems to work better than * using time smoothing directly or McAulay & Malpass modification.
 * \param spectrum the current power spectrum
 * \param spectrum_prev the previous power spectrum
 * \param N half of the fft size
 * \param release_coeff release coefficient
 */
void apply_time_envelope( float *spectrum, float *spectrum_prev, float N, float release_coeff )
{
    int k;

    for ( k = 0; k <= N; k++ )
    {
        /* It doesn't make much sense to have an attack slider when there is time smoothing */
        if ( spectrum[k] > spectrum_prev[k] )
        {
            /* Release (when signal is incrementing in amplitude) */
            spectrum[k] = release_coeff * spectrum_prev[k] + (1.f - release_coeff) * spectrum[k];
        }
    }
}


/* ------TRANSIENT PROTECTION------ */


/**
 * Transient detection using a rolling mean thresholding over the spectral flux of
 * the signal. Using more heuristics like high frequency content and others like the ones
 * anylised by Dixon in 'Simple Spectrum-Based Onset Detection' would be better. Onset
 * detection is explained thoroughly in 'A tutorial on onset detection in music signals' * by Bello.
 * \param fft_p2 the current power spectrum
 * \param transient_preserv_prev the previous power spectrum
 * \param fft_size_2 half of the fft size
 * \param tp_window_count tp_window_count counter for the rolling mean thresholding
 * \param tp_r_mean rolling mean value
 * \param transient_protection the manual scaling of the mean thresholding setted by the user
 */
bool transient_detection( float *fft_p2, float *transient_preserv_prev, float fft_size_2,
                          float *tp_window_count, float *tp_r_mean, float transient_protection )
{
    float adapted_threshold, reduction_function;

    /* Transient protection by forcing wiener filtering when an onset is detected */
    reduction_function = spectral_flux( fft_p2, transient_preserv_prev, fft_size_2 );
    /* reduction_function = high_frequency_content(fft_p2, fft_size_2); */

    *(tp_window_count) += 1.f;

    if ( *(tp_window_count) > 1.f )
    {
        *(tp_r_mean) += ( (reduction_function - *(tp_r_mean) ) / *(tp_window_count) );
    } else {
        *(tp_r_mean) = reduction_function;
    }

    adapted_threshold = (TP_UPPER_LIMIT - transient_protection) * *(tp_r_mean);

    memcpy( transient_preserv_prev, fft_p2, sizeof(float) * (fft_size_2 + 1) );

    if ( reduction_function > adapted_threshold )
    {
        return(true);
    } else {
        return(false);
    }
}


/* -----------WINDOW--------------- */


/**
 * blackman window values computing.
 * \param k bin number
 * \param N fft size
 */
float blackman( int k, int N )
{
    float p = ( (float) (k) ) / ( (float) (N) );
    return(0.42 - 0.5 * cosf( 2.f * M_PI * p ) + 0.08 * cosf( 4.f * M_PI * p ) );
}


/**
 * hanning window values computing.
 * \param k bin number
 * \param N fft size
 */
float hanning( int k, int N )
{
    float p = ( (float) (k) ) / ( (float) (N) );
    return(0.5 - 0.5 * cosf( 2.f * M_PI * p ) );
}


/**
 * hamming window values computing.
 * \param k bin number
 * \param N fft size
 */
float hamming( int k, int N )
{
    float p = ( (float) (k) ) / ( (float) (N) );
    return(0.54 - 0.46 * cosf( 2.f * M_PI * p ) );
}


/**
 * Vorbis window values computing. It satisfies Princen-Bradley criterion so perfect
 * reconstruction could be achieved with 50% overlap when used both in Analysis and
 * Synthesis
 * \param k bin number
 * \param N fft size
 */
float vorbis( int k, int N )
{
    float p = ( (float) (k) ) / ( (float) (N) );
    return(sinf( M_PI / 2.f * powf( sinf( M_PI * p ), 2.f ) ) );
}


/**
 * Wrapper to compute windows values.
 * \param window array for window values
 * \param N fft size
 * \param window_type type of window
 */
void fft_window( float *window, int N, int window_type )
{
    int k;
    for ( k = 0; k < N; k++ )
    {
        switch ( window_type )
        {
            case BLACKMAN_WINDOW:
                window[k] = blackman( k, N );
                break;
            case HANN_WINDOW:
                window[k] = hanning( k, N );
                break;
            case HAMMING_WINDOW:
                window[k] = hamming( k, N );
                break;
            case VORBIS_WINDOW:
                window[k] = vorbis( k, N );
                break;
        }
    }
}


/**
 * Outputs the scaling needed by the configured STFT transform to apply in OLA method.
 * \param input_window array of the input window values
 * \param output_window array of the output window values
 * \param frame_size size of the window arrays
 */
float get_window_scale_factor( float *input_window, float *output_window, int frame_size )
{
    float sum = 0.f;
    for ( int i = 0; i < frame_size; i++ )
        sum += input_window[i] * output_window[i];
    return(sum / (float) (frame_size) );
}


/**
 * Wrapper for getting the pre and post processing windows.
 * \param input_window array of the input window values
 * \param output_window array of the output window values
 * \param frame_size size of the window arrays
 * \param window_option_input input window option
 * \param window_option_output output window option
 * \param overlap_scale_factor scaling factor for the OLA for configured window options
 */
void fft_pre_and_post_window( float *input_window, float *output_window, int frame_size,
                              int window_option_input, int window_option_output,
                              float *overlap_scale_factor )
{
    /* Input window */
    switch ( window_option_input )
    {
        case 0:                                                 /* HANN */
            fft_window( input_window, frame_size, 0 );      /* STFT input window */
            break;
        case 1:                                                 /* HAMMING */
            fft_window( input_window, frame_size, 1 );      /* STFT input window */
            break;
        case 2:                                                 /* BLACKMAN */
            fft_window( input_window, frame_size, 2 );      /* STFT input window */
            break;
        case 3:                                                 /* VORBIS */
            fft_window( input_window, frame_size, 3 );      /* STFT input window */
            break;
    }

    /* Output window */
    switch ( window_option_output )
    {
        case 0:                                                 /* HANN */
            fft_window( output_window, frame_size, 0 );     /* STFT input window */
            break;
        case 1:                                                 /* HAMMING */
            fft_window( output_window, frame_size, 1 );     /* STFT input window */
            break;
        case 2:                                                 /* BLACKMAN */
            fft_window( output_window, frame_size, 2 );     /* STFT input window */
            break;
        case 3:                                                 /* VORBIS */
            fft_window( output_window, frame_size, 3 );     /* STFT input window */
            break;
    }

    /* Scaling necessary for perfect reconstruction using Overlapp Add */
    *(overlap_scale_factor) = get_window_scale_factor( input_window, output_window, frame_size );
}


/**
 * Gets the magnitude and phase spectrum of the complex spectrum. Takimg into account that
 * the half complex fft was used half of the spectrum contains the real part the other
 * the imaginary. Look at http://www.fftw.org/doc/The-Halfcomplex_002dformat-DFT.html for
 * more info. DC bin was treated as suggested in http://www.fftw.org/fftw2_doc/fftw_2.html
 * \param fft_p2 the current power spectrum
 * \param fft_magnitude the current magnitude spectrum
 * \param fft_phase the current phase spectrum
 * \param fft_size_2 half of the fft size
 * \param fft_size size of the fft
 * \param fft_buffer buffer with the complex spectrum of the fft transform
 */
void get_info_from_bins( float *fft_p2, float *fft_magnitude, float *fft_phase,
                         int fft_size_2, int fft_size, float *fft_buffer )
{
    int	k;
    float	real_p, imag_n, mag, p2, phase;

    /* DC bin */
    real_p	= fft_buffer[0];
    imag_n	= 0.f;

    fft_p2[0]		= real_p * real_p;
    fft_magnitude[0]	= real_p;
    fft_phase[0]		= atan2f( real_p, 0.f ); /* Phase is 0 for DC and nyquist */

    /* Get the rest of positive spectrum and compute the magnitude */
    for ( k = 1; k <= fft_size_2; k++ )
    {
        /* Get the half complex spectrum reals and complex */
        real_p	= fft_buffer[k];
        imag_n	= fft_buffer[fft_size - k];

        /* Get the magnitude, phase and power spectrum */
        if ( k < fft_size_2 )
        {
            p2	= (real_p * real_p + imag_n * imag_n);
            mag	= sqrtf( p2 );                  /* sqrt(real^2+imag^2) */
            phase	= atan2f( real_p, imag_n );
        } else {
            /* Nyquist - this is due to half complex transform */
            p2	= real_p * real_p;
            mag	= real_p;
            phase	= atan2f( real_p, 0.f );        /* Phase is 0 for DC and nyquist */
        }
        /* Store values in magnitude and power arrays (this stores the positive spectrum only) */
        fft_p2[k]		= p2;
        fft_magnitude[k]	= mag;                  /* This is not used but part of the STFT transform for generic use */
        fft_phase[k]		= phase;                /* This is not used but part of the STFT transform for generic use */
    }
}


/* For louizou algorith */
#define N_SMOOTH	0.7f                                    /* Smoothing over the power spectrum [0.9 - previous / 0.7 - actual] */
#define BETA_AT		0.8f                                    /* Adaption time of the local minimun [1 - slower / 0 - faster] */
#define GAMMA		0.998f                                  /* Smoothing factor over local minimun [1 - previous / 0 - actual] */
#define ALPHA_P		0.2f                                    /* smoothing constant over speech presence [1 - previous / 0 - actual] */
#define ALPHA_D		0.95f                                   /* time鈥揻requency dependent smoothing [0-1] [1 - previous / 0 - actual] */

/* for auto_thresholds initialization */
#define CROSSOVER_POINT1	1000.f                          /* crossover point for loizou reference thresholds */
#define CROSSOVER_POINT2	3000.f                          /* crossover point for loizou reference thresholds */
#define BAND_1_GAIN		2.0f                            /* gain for the band */
#define BAND_2_GAIN		2.0f                            /* gain for the band */
#define BAND_3_GAIN		7.0f                            /* gain for the band */


/**
 * Outputs the thresholds used by louizou method to discriminate between noise and signal.
 * 3 bands are used to perform more or less disctintion. Previous macros defines it
 * configuration.
 * \param auto_thresholds Reference threshold for louizou algorithm (same as thresh)
 * \param fft_size is the fft size
 * \param fft_size_2 is half of the fft size
 * \param samp_rate current sample rate of the host
 */
void compute_auto_thresholds( float *auto_thresholds, float fft_size, float fft_size_2,
                              float samp_rate )
{
    /* This was experimentally obteined in louizou paper */
    int	LF	= freq_to_bin( CROSSOVER_POINT1, samp_rate, fft_size ); /* 1kHz */
    int	MF	= freq_to_bin( CROSSOVER_POINT2, samp_rate, fft_size ); /* 3kHz */
    for ( int k = 0; k <= fft_size_2; k++ )
    {
        if ( k <= LF )
        {
            auto_thresholds[k] = BAND_1_GAIN;
        }
        if ( k > LF && k < MF )
        {
            auto_thresholds[k] = BAND_2_GAIN;
        }
        if ( k >= MF )
        {
            auto_thresholds[k] = BAND_3_GAIN;
        }
    }
}


/**
 * Loizou noise-estimation algorithm for highly non-stationary environments.
 * \param thresh Reference threshold for louizou algorithm
 * \param fft_size_2 is half of the fft size
 * \param p2 the power spectrum of current frame
 * \param s_pow_spec current smoothed power spectrum
 * \param prev_s_pow_spec previous smoothed power spectrum
 * \param noise_thresholds_p2 the noise thresholds for each bin estimated
 * \param prev_noise noise thresholds estimated for previous frame
 * \param p_min spectrum of the local minimun values
 * \param prev_p_min spectrum of the previous local minimun values
 * \param speech_p_p speech presence probability spectrum
 * \param prev_speech_p_p speech presence probability spectrum of previous frame
 */
static void
estimate_noise_loizou( float *thresh, int fft_size_2, float *p2, float *s_pow_spec,
                       float *prev_s_pow_spec, float *noise_thresholds_p2,
                       float *prev_noise, float *p_min, float *prev_p_min,
                       float *speech_p_p, float *prev_speech_p_p )
{
    int	k;
    float	ratio_ns = 0.f;
    float	freq_s[fft_size_2 + 1];
    float	speech_p_d[fft_size_2 + 1];

    for ( k = 0; k <= fft_size_2; k++ )
    {
        /* 1- Smooth between current and past noisy speech power spectrum */
        s_pow_spec[k] = N_SMOOTH * prev_s_pow_spec[k] + (1.f - N_SMOOTH) * p2[k]; /* interpolation between */

        /* 2- Compute the local minimum of noisy speech */
        if ( prev_p_min[k] < s_pow_spec[k] )
        {
            p_min[k] = GAMMA * prev_p_min[k] +
                       ( (1.f - GAMMA) / (1.f - BETA_AT) ) * (s_pow_spec[k] - BETA_AT * prev_s_pow_spec[k]);
        } else {
            p_min[k] = s_pow_spec[k];
        }

        /* 3- Compute ratio of noisy speech power spectrum to its local minimum */
        ratio_ns = s_pow_spec[k] / p_min[k];

        /* 4- Compute the indicator function I for speech present/absent detection */
        if ( ratio_ns > thresh[k] )     /* thresh could be freq dependant */
        {
            speech_p_d[k] = 1.f;    /* present */
        } else {
            speech_p_d[k] = 0.f;    /* absent */
        }

        /* 5- Calculate speech presence probability using first-order recursion */
        speech_p_p[k] = ALPHA_P * prev_speech_p_p[k] + (1.f - ALPHA_P) * speech_p_d[k];

        /* 6- Compute time-frequency dependent smoothing constant */
        freq_s[k] = ALPHA_D + (1.f - ALPHA_D) * speech_p_p[k];

        /* 7- Update noise estimate D using time-frequency dependent smoothing factor 伪 s (位,k). */
        noise_thresholds_p2[k] = freq_s[k] * prev_noise[k] + (1.f - freq_s[k]) * p2[k];
    }
}


/**
 * Wrapper for adaptive noise estimation.
 * \param p2 the power spectrum of current frame
 * \param fft_size_2 is half of the fft size
 * \param noise_thresholds_p2 the noise thresholds for each bin estimated
 * \param thresh Reference threshold for louizou algorithm
 * \param prev_noise_thresholds noise thresholds estimated for previous frame
 * \param s_pow_spec current smoothed power spectrum
 * \param prev_s_pow_spec previous smoothed power spectrum
 * \param p_min spectrum of the local minimun values
 * \param prev_p_min spectrum of the previous local minimun values
 * \param speech_p_p speech presence probability spectrum
 * \param prev_speech_p_p speech presence probability spectrum of previous frame
 */
void adapt_noise( float *p2, int fft_size_2, float *noise_thresholds_p2, float *thresh,
                  float *prev_noise_thresholds, float *s_pow_spec, float *prev_s_pow_spec, float *p_min,
                  float *prev_p_min, float *speech_p_p, float *prev_speech_p_p )
{
    estimate_noise_loizou( thresh, fft_size_2, p2, s_pow_spec, prev_s_pow_spec,
                           noise_thresholds_p2, prev_noise_thresholds, p_min, prev_p_min, speech_p_p, prev_speech_p_p );

    /* Update previous variables */
    memcpy( prev_noise_thresholds, noise_thresholds_p2, sizeof(float) * (fft_size_2 + 1) );
    memcpy( prev_s_pow_spec, s_pow_spec, sizeof(float) * (fft_size_2 + 1) );
    memcpy( prev_p_min, p_min, sizeof(float) * (fft_size_2 + 1) );
    memcpy( prev_speech_p_p, speech_p_p, sizeof(float) * (fft_size_2 + 1) );
}


/**
 * Noise estimation using a rolling mean over user selected noise section.
 * \param fft_p2 the power spectrum of current frame
 * \param fft_size_2 is half of the fft size
 * \param noise_thresholds_p2 the noise thresholds for each bin estimated
 * \param window_count is the frame counter for the rolling mean estimation
 */
void get_noise_statistics( float *fft_p2, int fft_size_2, float *noise_thresholds_p2,
                           float window_count )
{
    int k;

    /* Get noise thresholds based on averageing the input noise signal between frames */
    for ( k = 0; k <= fft_size_2; k++ )
    {
        if ( window_count <= 1.f )
        {
            noise_thresholds_p2[k] = fft_p2[k];
        } else {
            noise_thresholds_p2[k] += ( (fft_p2[k] - noise_thresholds_p2[k]) / window_count);
        }
    }
}


/**
 * \file denoise_gain.c
 * \author Luciano Dato
 * \brief All supression rules and filter computing related methods
 */

#define MAX( a, b )	( ( (a) > (b) ) ? (a) : (b) )
#define MIN( a, b )	( ( (a) < (b) ) ? (a) : (b) )
/* General spectral subtraction configuration */
#define GAMMA1	2.f
#define GAMMA2	0.5f


/**
 * Wiener substraction supression rule. Outputs the filter mirrored around nyquist.
 * \param fft_size_2 is half of the fft size
 * \param noise_thresholds is the threshold for each corresponding power spectum value
 * /param spectrum is the power spectum array
 * \param Gk is the filter computed by the supression rule for each bin of the spectrum
 */
void wiener_subtraction( int fft_size_2, float *spectrum, float *noise_thresholds, float *Gk )
{
    int k;

    for ( k = 0; k <= fft_size_2; k++ )
    {
        if ( noise_thresholds[k] > FLT_MIN )
        {
            if ( spectrum[k] > noise_thresholds[k] )
            {
                Gk[k] = (spectrum[k] - noise_thresholds[k]) / spectrum[k];
            } else {
                Gk[k] = 0.f;
            }
        } else {
            /* Otherwise we keep everything as is */
            Gk[k] = 1.f;
        }
    }

    /* mirrored gain array */
    for ( k = 1; k < fft_size_2; k++ )
    {
        Gk[(2 * fft_size_2) - k] = Gk[k];
    }
}


/**
 * Power substraction supression rule. Outputs the filter mirrored around nyquist.
 * \param fft_size_2 is half of the fft size
 * \param spectrum is the power spectum array
 * \param noise_thresholds is the threshold for each corresponding power spectum value
 * \param Gk is the filter computed by the supression rule for each bin of the spectrum
 */
void power_subtraction( int fft_size_2, float *spectrum, float *noise_thresholds, float *Gk )
{
    int k;

    for ( k = 0; k <= fft_size_2; k++ )
    {
        if ( noise_thresholds[k] > FLT_MIN )
        {
            if ( spectrum[k] > noise_thresholds[k] )
            {
                Gk[k] = sqrtf( (spectrum[k] - noise_thresholds[k]) / spectrum[k] );
            } else {
                Gk[k] = 0.f;
            }
        } else {
            /* Otherwise we keep everything as is */
            Gk[k] = 1.f;
        }
    }

    /* mirrored gain array */
    for ( k = 1; k < fft_size_2; k++ )
    {
        Gk[(2 * fft_size_2) - k] = Gk[k];
    }
}


/**
 * Magnitude substraction supression rule. Outputs the filter mirrored around nyquist.
 * \param fft_size_2 is half of the fft size
 * \param spectrum is the power spectum array
 * \param noise_thresholds is the threshold for each corresponding power spectum value
 * \param Gk is the filter computed by the supression rule for each bin of the spectrum
 */
void magnitude_subtraction( int fft_size_2, float *spectrum, float *noise_thresholds, float *Gk )
{
    int k;

    for ( k = 0; k <= fft_size_2; k++ )
    {
        if ( noise_thresholds[k] > FLT_MIN )
        {
            if ( spectrum[k] > noise_thresholds[k] )
            {
                Gk[k] = (sqrtf( spectrum[k] ) - sqrtf( noise_thresholds[k] ) ) / sqrtf( spectrum[k] );
            } else {
                Gk[k] = 0.f;
            }
        } else {
            /* Otherwise we keep everything as is */
            Gk[k] = 1.f;
        }
    }

    /* mirrored gain array */
    for ( k = 1; k < fft_size_2; k++ )
    {
        Gk[(2 * fft_size_2) - k] = Gk[k];
    }
}


/**
 * Gating with hard knee supression rule. Outputs the filter mirrored around nyquist.
 * \param fft_size_2 is half of the fft size
 * \param spectrum is the power spectum array
 * \param noise_thresholds is the threshold for each corresponding power spectum value
 * \param Gk is the filter computed by the supression rule for each bin of the spectrum
 */
void spectral_gating( int fft_size_2, float *spectrum, float *noise_thresholds, float *Gk )
{
    int k;

    for ( k = 0; k <= fft_size_2; k++ )
    {
        if ( noise_thresholds[k] > FLT_MIN )
        {
            /* //Without knee */
            if ( spectrum[k] >= noise_thresholds[k] )
            {
                /* over the threshold */
                Gk[k] = 1.f;
            } else {
                /* under the threshold */
                Gk[k] = 0.f;
            }
        } else {
            /* Otherwise we keep everything as is */
            Gk[k] = 1.f;
        }
    }

    /* mirrored gain array */
    for ( k = 1; k < fft_size_2; k++ )
    {
        Gk[(2 * fft_size_2) - k] = Gk[k];
    }
}


/**
 * Generalized spectral subtraction supression rule. This version uses an array of alphas and betas. Outputs the filter mirrored around nyquist. GAMMA defines what type of spectral Subtraction is used. GAMMA1=GAMMA2=1 is magnitude substaction. GAMMA1=2 GAMMA2=0.5 is power Subtraction. GAMMA1=2 GAMMA2=1 is wiener filtering.
 * \param fft_size_2 is half of the fft size
 * \param alpha is the array of oversubtraction factors for each bin
 * \param beta is the array of the spectral flooring factors for each bin
 * \param spectrum is the power spectum array
 * \param noise_thresholds is the threshold for each corresponding power spectum value
 * \param Gk is the filter computed by the supression rule for each bin of the spectrum
 */
void denoise_gain_gss( int fft_size_2, float *alpha, float *beta, float *spectrum,
                       float *noise_thresholds, float *Gk )
{
    int k;

    for ( k = 0; k <= fft_size_2; k++ )
    {
        if ( spectrum[k] > FLT_MIN )
        {
            if ( powf( (noise_thresholds[k] / spectrum[k]), GAMMA1 ) < (1.f / (alpha[k] + beta[k]) ) )
            {
                Gk[k] = MAX( powf( 1.f - (alpha[k] * powf( (noise_thresholds[k] / spectrum[k]), GAMMA1 ) ), GAMMA2 ), 0.f );
            } else {
                Gk[k] = MAX( powf( beta[k] * powf( (noise_thresholds[k] / spectrum[k]), GAMMA1 ), GAMMA2 ), 0.f );
            }
        } else {
            /* Otherwise we keep everything as is */
            Gk[k] = 1.f;
        }
    }

    /* mirrored gain array */
    for ( k = 1; k < fft_size_2; k++ )
    {
        Gk[(2 * fft_size_2) - k] = Gk[k];
    }
}


/*
 *
 * /**
 * \file masking.c
 * \author Luciano Dato
 * \brief Methods for masking threshold estimation
 */


/* masking thresholds values recomended by virag */
#define ALPHA_MAX	6.f
#define ALPHA_MIN	1.f
#define BETA_MAX	0.02f
#define BETA_MIN	0.0f

/* extra values */
#define N_BARK_BANDS		25
#define AT_SINE_WAVE_FREQ	1000.f
#define REFERENCE_LEVEL		90.f                            /* dbSPL level of reproduction */

#define BIAS		0
#define HIGH_FREQ_BIAS	20.f
#define S_AMP		1.f

#define ARRAYACCESS( a, i, j ) ( (a)[(i) * N_BARK_BANDS + (j)]) /* This is for SSF Matrix recall */

/* Proposed by Sinha and Tewfik and explained by Virag */
const float relative_thresholds[N_BARK_BANDS] = { -16.f, -17.f, -18.f, -19.f, -20.f, -21.f, -22.f, -23.f, -24.f, -25.f,
                                                  -25.f, -25.f, -25.f, -25.f, -25.f, -24.f, -23.f, -22.f, -19.f, -18.f,
                                                  -18.f, -18.f, -18.f, -18.f, -18.f };


/**
 * Fft to bark bilinear scale transform. This computes the corresponding bark band for
 * each fft bin and generates an array that
 * inicates this mapping.
 * \param bark_z defines the bark to linear mapping for current spectrum config
 * \param fft_size_2 is half of the fft size
 * \param srate current sample rate of the host
 */
void compute_bark_mapping( float *bark_z, int fft_size_2, int srate )
{
    int	k;
    float	freq;

    for ( k = 0; k <= fft_size_2; k++ )
    {
        freq		= (float) srate / (2.f * (float) (fft_size_2) * (float) k); /* bin to freq */
        bark_z[k]	= 1.f + 13.f * atanf( 0.00076f * freq ) + 3.5f * atanf( powf( freq / 7500.f, 2.f ) );
    }
}


/**
 * Computes the spectral spreading function of Schroeder as a matrix using bark scale.
 * This is to perform a convolution between this function and a bark spectrum. The
 * complete explanation for this is in Robinsons master thesis 'Perceptual model for
 * assessment of coded audio'.
 * \param SSF defines the spreading function matrix
 */
void compute_SSF( float *SSF )
{
    int	i, j;
    float	y;
    for ( i = 0; i < N_BARK_BANDS; i++ )
    {
        for ( j = 0; j < N_BARK_BANDS; j++ )
        {
            y = (i + 1) - (j + 1);
            /* Spreading function (Schroeder) */
            ARRAYACCESS( SSF, i, j ) =
                    15.81f + 7.5f * (y + 0.474f) - 17.5f * sqrtf( 1.f + (y + 0.474f) * (y + 0.474f) ); /* dB scale */
            /* db to Linear */
            ARRAYACCESS( SSF, i, j ) = powf( 10.f, ARRAYACCESS( SSF, i, j ) / 10.f );
        }
    }
}


/**
 * Convolution between the spreading function by multiplication of a Toepliz matrix
 * to a bark spectrum. The complete explanation for this is in Robinsons master thesis
 * 'Perceptual model for assessment of coded audio'.
 * \param SSF defines the spreading function matrix
 * \param bark_spectrum the bark spectrum values of current power spectrum
 * \param spreaded_spectrum result of the convolution bewtween SSF and the bark spectrum
 */
void convolve_with_SSF( float *SSF, float *bark_spectrum, float *spreaded_spectrum )
{
    int i, j;
    for ( i = 0; i < N_BARK_BANDS; i++ )
    {
        spreaded_spectrum[i] = 0.f;
        for ( j = 0; j < N_BARK_BANDS; j++ )
        {
            spreaded_spectrum[i] += ARRAYACCESS( SSF, i, j ) * bark_spectrum[j];
        }
    }
}


/**
 * Computes the energy of each bark band taking a power or magnitude spectrum. It performs
 * the mapping from linear fft scale to the bark scale. Often called critical band Analysis
 * \param bark_z defines the bark to linear mapping for current spectrum config
 * \param bark_spectrum the bark spectrum values of current power spectrum
 * \param spectrum is the power spectum array
 * \param intermediate_band_bins holds the bin numbers that are limits of each band
 * \param n_bins_per_band holds the the number of bins in each band
 */
void compute_bark_spectrum( float *bark_z, float *bark_spectrum, float *spectrum,
                            float *intermediate_band_bins, float *n_bins_per_band )
{
    int	j;
    int	last_position = 0;

    for ( j = 0; j < N_BARK_BANDS; j++ )
    {
        int cont = 0;
        if ( j == 0 )
            cont = 1;                                               /* Do not take into account the DC component */

        bark_spectrum[j] = 0.f;
        /* If we are on the same band for the bin */
        while ( floor( bark_z[last_position + cont] ) == (j + 1) )      /* First bark band is 1 */
        {
            bark_spectrum[j] += spectrum[last_position + cont];
            cont++;
        }
        /* Move the position to the next group of bins from the upper bark band */
        last_position += cont;

        /* store bin information */
        n_bins_per_band[j]		= cont;
        intermediate_band_bins[j]	= last_position;
    }
}


/**
 * Computes the reference spectrum to perform the db to dbSPL conversion. It uses
 * a full scale 1khz sine wave as the reference signal and performs an fft transform over
 * it to get the magnitude spectrum and then scales it using a reference dbSPL level. This
 * is used because the absolute thresholds of hearing are obtained in SPL scale so it's
 * necessary to compare this thresholds with obtained masking thresholds using the same
 * scale.
 * \param spl_reference_values defines the reference values for each bin to convert from db to db SPL
 * \param fft_size_2 is half of the fft size
 * \param srate current sample rate of the host
 * \param input_fft_buffer_at input buffer for the reference sinewave fft transform
 * \param output_fft_buffer_at output buffer for the reference sinewave fft transform
 * \param forward_at fftw plan for the reference sinewave fft transform
 */
void spl_reference( float *spl_reference_values, int fft_size_2, int srate,
                    float *input_fft_buffer_at, float *output_fft_buffer_at,
                    fftwf_plan *forward_at )
{
    int	k;
    float	sinewave[2 * fft_size_2];
    float	window[2 * fft_size_2];
    float	fft_p2_at[fft_size_2 + 1];
    float	fft_magnitude_at[fft_size_2 + 1];
    float	fft_phase_at[fft_size_2 + 1];
    float	fft_p2_at_dbspl[fft_size_2 + 1];

    /* Generate a fullscale sine wave of 1 kHz */
    for ( k = 0; k < 2 * fft_size_2; k++ )
    {
        sinewave[k] = S_AMP * sinf( (2.f * M_PI * k * AT_SINE_WAVE_FREQ) / (float) srate );
    }

    /* Windowing the sinewave */
    fft_window( window, 2 * fft_size_2, 0 ); /* von-Hann window */
    for ( k = 0; k < 2 * fft_size_2; k++ )
    {
        input_fft_buffer_at[k] = sinewave[k] * window[k];
    }

    /* Do FFT */
    fftwf_execute( *forward_at );

    /* Get the magnitude */
    get_info_from_bins( fft_p2_at, fft_magnitude_at, fft_phase_at, fft_size_2, 2 * fft_size_2,
                        output_fft_buffer_at );

    /* Convert to db and taking into account 90dbfs of reproduction loudness */
    for ( k = 0; k <= fft_size_2; k++ )
    {
        fft_p2_at_dbspl[k] = REFERENCE_LEVEL - 10.f * log10f( fft_p2_at[k] );
    }

    memcpy( spl_reference_values, fft_p2_at_dbspl, sizeof(float) * (fft_size_2 + 1) );
}


/**
 * dB scale to dBSPL conversion. This is to convert masking thresholds from db to dbSPL
 * scale to then compare them to absolute threshold of hearing.
 * \param spl_reference_values defines the reference values for each bin to convert from db to db SPL
 * \param masking_thresholds the masking thresholds obtained in db scale
 * \param fft_size_2 is half of the fft size
 */
void convert_to_dbspl( float *spl_reference_values, float *masking_thresholds, int fft_size_2 )
{
    for ( int k = 0; k <= fft_size_2; k++ )
    {
        masking_thresholds[k] += spl_reference_values[k];
    }
}


/**
 * Computes the absolute thresholds of hearing to contrast with the masking thresholds.
 * This formula is explained in Thiemann thesis 'Acoustic Noise Suppression for Speech
 * Signals using Auditory Masking Effects'
 * \param absolute_thresholds defines the absolute thresholds of hearing for current spectrum config
 * \param fft_size_2 is half of the fft size
 * \param srate current sample rate of the host
 */
void compute_absolute_thresholds( float *absolute_thresholds, int fft_size_2, int srate )
{
    int	k;
    float	freq;

    for ( k = 1; k <=
                 fft_size_2; k++ )                                                 /* As explained by thiemann */
    {
        freq = bin_to_freq( k, srate,
                            fft_size_2 );                               /* bin to freq */
        absolute_thresholds[k] =
                3.64f * powf( (freq / 1000.f), -0.8f ) - 6.5f * exp( -0.6f * powf( (freq / 1000.f - 3.3f), 2.f ) ) +
                powf( 10.f, -3.f ) * powf( (freq / 1000.f), 4.f );      /* dBSPL scale */
    }
}


/**
 * Computes the tonality factor using the spectral flatness for a given bark band
 * spectrum. Values are in db scale. An inferior limit of -60 db is imposed. To avoid zero
 * logs some trickery is used as explained in https://en.wikipedia.org/wiki/Spectral_flatness
 * Robinsons thesis explains this further too.
 * \param spectrum is the power spectum array
 * \param intermediate_band_bins holds the bin numbers that are limits of each band
 * \param n_bins_per_band holds the the number of bins in each band
 * \param band the bark band given
 */
float compute_tonality_factor( float *spectrum, float *intermediate_band_bins,
                               float *n_bins_per_band, int band )
{
    int	k;
    float	SFM, tonality_factor;
    float	sum_p = 0.f, sum_log_p = 0.f;
    int	start_pos, end_pos = 0;

    /* Mapping to bark bands */
    if ( band == 0 )
    {
        start_pos	= band;
        end_pos		= n_bins_per_band[band];
    } else {
        start_pos	= intermediate_band_bins[band - 1];
        end_pos		= intermediate_band_bins[band - 1] + n_bins_per_band[band];
    }

    /* Using power spectrum to compute the tonality factor */
    for ( k = start_pos; k < end_pos; k++ )
    {
        /* For spectral flatness measures */
        sum_p		+= spectrum[k];
        sum_log_p	+= log10f( spectrum[k] );
    }
    /* spectral flatness measure using Geometric and Arithmetic means of the spectrum */
    SFM = 10.f * (sum_log_p / (float) (n_bins_per_band[band]) -
                  log10f( sum_p / (float) (n_bins_per_band[band]) ) ); /* this value is in db scale */

    /* Tonality factor in db scale */
    tonality_factor = MIN( SFM / -60.f, 1.f );

    return(tonality_factor);
}


/**
 * Johnston Masking threshold calculation. This are greatly explained in Robinsons thesis
 * and Thiemann thesis too and in Virags paper 'Single Channel Speech Enhancement Based on
 * Masking Properties of the Human Auditory System'. Some optimizations suggested in
 * Virags work are implemented but not used as they seem to not be necessary in modern
 * age computers.
 * \param bark_z defines the bark to linear mapping for current spectrum config
 * \param absolute_thresholds defines the absolute thresholds of hearing for current spectrum config
 * \param SSF defines the spreading function matrix
 * \param spectrum is the power spectum array
 * \param fft_size_2 is half of the fft size
 * \param masking_thresholds the masking thresholds obtained in db scale
 * \param spreaded_unity_gain_bark_spectrum correction to be applied to SSF convolution
 * \param spl_reference_values defines the reference values for each bin to convert from db to db SPL
 */
void compute_masking_thresholds( float *bark_z, float *absolute_thresholds, float *SSF,
                                 float *spectrum, int fft_size_2, float *masking_thresholds,
                                 float *spreaded_unity_gain_bark_spectrum,
                                 float *spl_reference_values )
{
    int	k, j, start_pos, end_pos;
    float	intermediate_band_bins[N_BARK_BANDS];
    float	n_bins_per_band[N_BARK_BANDS];
    float	bark_spectrum[N_BARK_BANDS];
    float	threshold_j[N_BARK_BANDS];
    float	masking_offset[N_BARK_BANDS];
    float	spreaded_spectrum[N_BARK_BANDS];
    float	tonality_factor;

    /* First we get the energy in each bark band */
    compute_bark_spectrum( bark_z, bark_spectrum, spectrum, intermediate_band_bins,
                           n_bins_per_band );

    /*
     * Now that we have the bark spectrum
     * Convolution bewtween the bark spectrum and SSF (Toepliz matrix multiplication)
     */
    convolve_with_SSF( SSF, bark_spectrum, spreaded_spectrum );

    for ( j = 0; j < N_BARK_BANDS; j++ )
    {
        /* Then we compute the tonality_factor for each band (1 tone like 0 noise like) */
        tonality_factor = compute_tonality_factor( spectrum, intermediate_band_bins, n_bins_per_band,
                                                   j ); /* Uses power spectrum */

        /* Masking offset */
        masking_offset[j] = (tonality_factor * (14.5f + (float) (j + 1) ) + 5.5f * (1.f - tonality_factor) );

#if BIAS
        /* Using offset proposed by Virag (an optimization not needed) */
		masking_offset[j] = relative_thresholds[j];
		/* Consider tonal noise in upper bands (j>15) due to musical noise of the power Sustraction that was used at First */
		if ( j > 15 )
			masking_offset[j] += HIGH_FREQ_BIAS;
#endif

        /* spread Masking threshold */
        threshold_j[j] = powf( 10.f, log10f( spreaded_spectrum[j] ) - (masking_offset[j] / 10.f) );

        /* Renormalization */
        threshold_j[j] -= 10.f * log10f( spreaded_unity_gain_bark_spectrum[j] );

        /*
         * Relating the spread masking threshold to the critical band masking thresholds
         * Border case
         */
        if ( j == 0 )
        {
            start_pos = 0;
        } else {
            start_pos = intermediate_band_bins[j - 1];
        }
        end_pos = intermediate_band_bins[j];

        for ( k = start_pos; k < end_pos; k++ )
        {
            masking_thresholds[k] = threshold_j[j];
        }
    }

    /*
     * Masking thresholds need to be converted to db spl scale in order to be compared with
     * absolute threshold of hearing
     */
    convert_to_dbspl( spl_reference_values, masking_thresholds, fft_size_2 );

    /* Take into account the absolute_thresholds of hearing */
    for ( k = 0; k <= fft_size_2; k++ )
    {
        masking_thresholds[k] = MAX( masking_thresholds[k], absolute_thresholds[k] );
    }
}


/**
 * alpha and beta computation according to Virags paper. Alphas refers to the oversubtraction
 * factor for each fft bin and beta to the spectral flooring. What the oversubtraction
 * factor for each bin really does is scaling the noise profile in order reduce more or
 * less noise in the supression rule. Spectral flooring limits the amount of reduction in
 * each bin. Using masking thresholds for means of adapting this two parameters correlate
 * much more with human hearing and results in smoother results than using non linear
 * subtraction or others methods of adapting them. Spectral flooring is not used since
 * users decide the amount of noise reduccion themselves and spectral flooring is tied to
 * that parameter instead of being setted automatically.
 * \param fft_p2 the power spectrum of current frame
 * \param noise_thresholds_p2 the noise thresholds for each bin estimated previously
 * \param fft_size_2 is half of the fft size
 * \param alpha_masking is the array of oversubtraction factors for each bin
 * \param beta_masking is the array of the spectral flooring factors for each bin
 * \param bark_z defines the bark to linear mapping for current spectrum config
 * \param absolute_thresholds defines the absolute thresholds of hearing for current spectrum config
 * \param SSF defines the spreading function matrix
 * \param spreaded_unity_gain_bark_spectrum correction to be applied to SSF convolution
 * \param spl_reference_values defines the reference values for each bin to convert from db to db SPL
 * \param masking_value is the limit max oversubtraction to be computed
 * \param reduction_value is the limit max the spectral flooring to be computed
 */
void compute_alpha_and_beta( float *fft_p2, float *noise_thresholds_p2, int fft_size_2,
                             float *alpha_masking, float *beta_masking, float *bark_z,
                             float *absolute_thresholds, float *SSF,
                             float *spreaded_unity_gain_bark_spectrum,
                             float *spl_reference_values, float masking_value,
                             float reduction_value )
{
    int	k;
    float	masking_thresholds[fft_size_2 + 1];
    float	estimated_clean[fft_size_2 + 1];
    float	normalized_value;

    /*
     * Noise masking threshold must be computed from a clean signal
     * therefor we aproximate a clean signal using a power Sustraction over
     * the original noisy one
     */

    /* basic Power Sustraction to estimate clean signal */
    for ( k = 0; k <= fft_size_2; k++ )
    {
        estimated_clean[k] = MAX( fft_p2[k] - noise_thresholds_p2[k], FLT_MIN );
    }

    /* Now we can compute noise masking threshold from this clean signal */
    compute_masking_thresholds( bark_z, absolute_thresholds, SSF, estimated_clean,
                                fft_size_2, masking_thresholds,
                                spreaded_unity_gain_bark_spectrum, spl_reference_values );

    /* First we need the maximun and the minimun value of the masking threshold */
    float	max_masked_tmp	= max_spectral_value( masking_thresholds, fft_size_2 );
    float	min_masked_tmp	= min_spectral_value( masking_thresholds, fft_size_2 );

    for ( k = 0; k <= fft_size_2; k++ )
    {
        /* new alpha and beta vector */
        if ( masking_thresholds[k] == max_masked_tmp )
        {
            alpha_masking[k]	= ALPHA_MIN;
            beta_masking[k]		= BETA_MIN;
        }
        if ( masking_thresholds[k] == min_masked_tmp )
        {
            alpha_masking[k]	= masking_value;
            beta_masking[k]		= reduction_value;
        }
        if ( masking_thresholds[k] < max_masked_tmp && masking_thresholds[k] > min_masked_tmp )
        {
            /* Linear interpolation of the value between max and min masked threshold values */
            normalized_value = (masking_thresholds[k] - min_masked_tmp) / (max_masked_tmp - min_masked_tmp);

            alpha_masking[k]	= (1.f - normalized_value) * ALPHA_MIN + normalized_value * masking_value;
            beta_masking[k]		= (1.f - normalized_value) * BETA_MIN + normalized_value * reduction_value;
        }
    }
}


/* ------------GAIN AND THRESHOLD CALCULATION--------------- */


/**
 * Includes every preprocessing or precomputing before the supression rule.
 * \param noise_thresholds_offset the scaling of the thresholds setted by the user
 * \param fft_p2 the power spectrum of current frame
 * \param noise_thresholds_p2 the noise thresholds for each bin estimated previously
 * \param noise_thresholds_scaled the noise thresholds for each bin estimated previously scaled by the user
 * \param smoothed_spectrum current power specturm with time smoothing applied
 * \param smoothed_spectrum_prev the power specturm with time smoothing applied of previous frame
 * \param fft_size_2 is half of the fft size
 * \param prev_beta beta of previous frame for adaptive smoothing (not used yet)
 * \param bark_z defines the bark to linear mapping for current spectrum config
 * \param absolute_thresholds defines the absolute thresholds of hearing for current spectrum config
 * \param SSF defines the spreading function matrix
 * \param release_coeff release coefficient for time smoothing
 * \param spreaded_unity_gain_bark_spectrum correction to be applied to SSF convolution
 * \param spl_reference_values defines the reference values for each bin to convert from db to db SPL
 * \param alpha_masking is the array of oversubtraction factors for each bin
 * \param beta_masking is the array of the spectral flooring factors for each bin
 * \param masking_value is the limit max oversubtraction to be computed
 * \param adaptive flag that indicates if the noise is being estimated adaptively
 * \param reduction_value is the limit max the spectral flooring to be computed
 * \param transient_preserv_prev is the previous frame for spectral flux computing
 * \param tp_window_count is the frame counter for the rolling mean thresholding for onset detection
 * \param tp_r_mean is the rolling mean value for onset detection
 * \param transient_present indicates if current frame is an onset or not (contains a transient)
 * \param transient_protection is the flag that indicates whether transient protection is active or not
 */
void preprocessing( float noise_thresholds_offset, float *fft_p2,
                    float *noise_thresholds_p2,
                    float *noise_thresholds_scaled, float *smoothed_spectrum,
                    float *smoothed_spectrum_prev, int fft_size_2,
                    float *bark_z, float *absolute_thresholds, float *SSF,
                    float release_coeff, float *spreaded_unity_gain_bark_spectrum,
                    float *spl_reference_values, float *alpha_masking, float *beta_masking,
                    float masking_value, float adaptive_state, float reduction_value,
                    float *transient_preserv_prev, float *tp_window_count, float *tp_r_mean,
                    bool *transient_present, float transient_protection )
{
    int k;

    /* PREPROCESSING - PRECALCULATIONS */

    /* ------TRANSIENT DETECTION------ */

    if ( transient_protection > 1.f )
    {
        *(transient_present) = transient_detection( fft_p2, transient_preserv_prev,
                                                    fft_size_2, tp_window_count, tp_r_mean,
                                                    transient_protection );
    }

    /* CALCULATION OF ALPHA WITH MASKING THRESHOLDS USING VIRAGS METHOD */

    if ( masking_value > 1.f && adaptive_state == 0.f ) /* Only when adaptive is off */
    {
        compute_alpha_and_beta( fft_p2, noise_thresholds_p2, fft_size_2,
                                alpha_masking, beta_masking, bark_z, absolute_thresholds,
                                SSF, spreaded_unity_gain_bark_spectrum, spl_reference_values,
                                masking_value, reduction_value );
    } else {
        initialize_array( alpha_masking, 1.f,
                          fft_size_2 + 1 ); /* This avoids incorrect results when moving sliders rapidly */
    }

    /* ------OVERSUBTRACTION------ */

    /* Scale noise thresholds (equals applying an oversubtraction factor in spectral subtraction) */
    for ( k = 0; k <= fft_size_2; k++ )
    {
        if ( adaptive_state == 0.f )
        {
            noise_thresholds_scaled[k] = noise_thresholds_p2[k] * noise_thresholds_offset * alpha_masking[k];
        } else {
            noise_thresholds_scaled[k] = noise_thresholds_p2[k] * noise_thresholds_offset;
        }
    }

    /* ------SMOOTHING DETECTOR------ */

    if ( adaptive_state == 0.f ) /* Only when adaptive is off */
    {
        memcpy( smoothed_spectrum, fft_p2, sizeof(float) * (fft_size_2 + 1) );

        apply_time_envelope( smoothed_spectrum, smoothed_spectrum_prev, fft_size_2, release_coeff );

        memcpy( smoothed_spectrum_prev, smoothed_spectrum, sizeof(float) * (fft_size_2 + 1) );
    }
}


/**
 * Computes the supression filter based on pre-processing data.
 * \param fft_p2 the power spectrum of current frame
 * \param noise_thresholds_p2 the noise thresholds for each bin estimated previously
 * \param noise_thresholds_scaled the noise thresholds for each bin estimated previously scaled by the user
 * \param smoothed_spectrum current power specturm with time smoothing applied
 * \param fft_size_2 is half of the fft size
 * \param adaptive flag that indicates if the noise is being estimated adaptively
 * \param Gk is the filter computed by the supression rule for each bin of the spectrum
 * \param transient_protection is the flag that indicates whether transient protection is active or not
 * \param transient_present indicates if current frame is an onset or not (contains a transient)
 */
void spectral_gain( float *fft_p2, float *noise_thresholds_p2, float *noise_thresholds_scaled,
                    float *smoothed_spectrum, int fft_size_2, float adaptive, float *Gk,
                    float transient_protection, bool transient_present )
{
    /* ------REDUCTION GAINS------ */

    /* Get reduction to apply */
    if ( adaptive == 1.f )
    {
        power_subtraction( fft_size_2, fft_p2, noise_thresholds_scaled, Gk );
    } else {
        /* Protect transient by avoiding smoothing if present */
        if ( transient_present && transient_protection > 1.f )
        {
            wiener_subtraction( fft_size_2, fft_p2, noise_thresholds_scaled, Gk );
        } else {
            spectral_gating( fft_size_2, smoothed_spectrum, noise_thresholds_scaled, Gk );
        }
    }
}


/**
 * Applies the filter to the complex spectrum and gets the clean signal.
 * \param fft_size size of the fft
 * \param output_fft_buffer the unprocessed spectrum remaining in the fft buffer
 * \param denoised_spectrum the spectrum of the cleaned signal
 * \param Gk is the filter computed by the supression rule for each bin of the spectrum
 */
void denoised_calulation( int fft_size, float *output_fft_buffer,
                          float *denoised_spectrum, float *Gk )
{
    int k;

    /* Apply the computed gain to the signal and store it in denoised array */
    for ( k = 0; k < fft_size; k++ )
    {
        denoised_spectrum[k] = output_fft_buffer[k] * Gk[k];
    }
}


/**
 * Gets the residual signal of the reduction.
 * \param fft_size size of the fft
 * \param output_fft_buffer the unprocessed spectrum remaining in the fft buffer
 * \param denoised_spectrum the spectrum of the cleaned signal
 * \param whitening_factor the mix coefficient between whitened and not whitened residual spectrum
 * \param residual_max_spectrum contains the maximun temporal value in each residual bin
 * \param whitening_window_count counts frames to distinguish the first from the others
 * \param max_decay_rate coefficient that sets the memory for each temporal maximun
 */
void residual_calulation( int fft_size, float *output_fft_buffer,
                          float *residual_spectrum, float *denoised_spectrum,
                          float whitening_factor, float *residual_max_spectrum,
                          float *whitening_window_count, float max_decay_rate )
{
    int k;

    /* Residual signal */
    for ( k = 0; k < fft_size; k++ )
    {
        residual_spectrum[k] = output_fft_buffer[k] - denoised_spectrum[k];
    }

    /*
     * //////////POSTPROCESSING RESIDUAL
     * Whitening (residual spectrum more similar to white noise)
     */
    if ( whitening_factor > 0.f )
    {
        spectral_whitening( residual_spectrum, whitening_factor, fft_size,
                            residual_max_spectrum, whitening_window_count, max_decay_rate );
    }
    /* ////////// */
}


/**
 * Mixes the cleaned signal with the residual taking into account the reduction configured
 * by the user. Outputs the final signal or the residual only.
 * \param fft_size size of the fft
 * \param final_spectrum the spectrum to output from the plugin
 * \param residual_spectrum the spectrum of the reduction residual
 * \param denoised_spectrum the spectrum of the cleaned signal
 * \param reduction_amount the amount of dB power to reduce setted by the user
 * \param noise_listen control variable that decides whether to output the mixed noise reduced signal or the residual only
 */
void final_spectrum_ensemble( int fft_size, float *final_spectrum,
                              float *residual_spectrum, float *denoised_spectrum,
                              float reduction_amount, float noise_listen )
{
    int k;

    /* OUTPUT RESULTS using smooth bypass and parametric subtraction */
    if ( noise_listen == 0.f )
    {
        /* Mix residual and processed (Parametric way of noise reduction) */
        for ( k = 0; k < fft_size; k++ )
        {
            final_spectrum[k] = denoised_spectrum[k] + residual_spectrum[k] * reduction_amount;
        }
    } else {
        /* Output noise only */
        for ( k = 0; k < fft_size; k++ )
        {
            final_spectrum[k] = residual_spectrum[k];
        }
    }
}


/**
 * Mixes unprocessed and processed signal to bypass softly.
 * \param final_spectrum the spectrum to output from the plugin
 * \param output_fft_buffer the unprocessed spectrum remaining in the fft buffer
 * \param wet_dry mixing coefficient
 * \param fft_size size of the fft
 */
void soft_bypass( float *final_spectrum, float *output_fft_buffer, float wet_dry,
                  int fft_size )
{
    int k;

    for ( k = 0; k < fft_size; k++ )
    {
        output_fft_buffer[k] = (1.f - wet_dry) * output_fft_buffer[k] + final_spectrum[k] * wet_dry;
    }
}


