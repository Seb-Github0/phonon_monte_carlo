//! Calculate final energy fraction and rise time from the raw simulation output.

use crate::config::Config;
use crate::data_structures::MinMax;
use crate::simulate::EnergyResults;
use interp::{InterpMode, interp};
use sci_rs::signal::convolve::{ConvolveMode, fftconvolve};
use std::io::BufRead;

/// Load the decay template from the CSV file specified in the config.
/// Match the sampling rate to the simulation data.
///
/// Return vector of signal values only. Times are implicitly given by index * cfg.time_bin_size.
pub fn load_decay_template(cfg: &Config) -> Vec<f64> {
    let dt = cfg.time_bin_size;
    let n = cfg.number_of_timesteps;
    let max_time = (n - 1) as f64 * dt;

    let mut times = Vec::new();
    let mut signal = Vec::new();

    // like pd.read_csv(file_decay_template_LAMPv4) (column 0 is time, column 1 is value)
    let file = std::fs::File::open(&cfg.post_processing.file_decay_template)
        .expect("Could not open decay template file");
    let reader = std::io::BufReader::new(file);
    for line in reader.lines().skip(1) {
        let line = line.expect("Could not read line from decay template file");
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() != 2 {
            continue; // skip lines that don't have exactly 2 columns
        }
        let time_us: f64 = parts[0]
            .trim()
            .parse()
            .expect("Could not parse time from decay template file");
        let value: f64 = parts[1]
            .trim()
            .parse()
            .expect("Could not parse value from decay template file");
        times.push(time_us * 1e-6);
        signal.push(value);

        if time_us * 1e-6 > max_time * 1.1 {
            break;
        }
    }

    // match sampling rate to simulation data
    let time_decay_fine: Vec<f64> = (0..n).map(|i| i as f64 * dt).collect();
    let signal: Vec<f64> = time_decay_fine
        .iter()
        .map(|t| interp(&times, &signal, *t, &InterpMode::FirstLast))
        .collect();

    // take only values where times <= max_time
    let mut signal_masked: Vec<f64> = Vec::with_capacity(n);
    for (t, v) in time_decay_fine.iter().zip(signal.iter()) {
        if *t <= max_time {
            signal_masked.push(*v);
        } else {
            break;
        }
    }

    signal
}

/// Calculate the threshold-to-threshold rise time from the energy results,
/// optionally convolving with a decay template first. Uses the thresholds
/// specified in the config.
///
/// Return the rise time in seconds.
pub fn calculate_rise_time(
    energy_results: &[EnergyResults],
    decay_template: &Option<Vec<f64>>,
    cfg: &Config,
) -> f64 {
    let mut signal: Vec<f64> = energy_results
        .iter()
        .map(|res| res.e_absorbed_total)
        .collect();

    if let Some(signal_decay) = decay_template {
        let signal_convolved: Vec<f64> =
            fftconvolve(&signal.diff(), signal_decay, ConvolveMode::Full);

        // take only [:len(signal)-1]
        signal = signal_convolved
            .into_iter()
            .take(signal.len() - 1)
            .collect();
    }

    let signal_max = signal.max();

    let time_to_threshold = |threshold: f64| -> f64 {
        // find idx where signal >= threshold * signal_max for the first time
        let idx = signal
            .iter()
            .position(|&v| v >= threshold * signal_max)
            .unwrap_or(0);
        // interpolate linearly between idx-1 and idx
        if idx == 0 {
            0.0
        } else {
            let v1 = signal[idx - 1];
            let v2 = signal[idx];
            let t1 = (idx - 1) as f64 * cfg.time_bin_size;
            let t2 = idx as f64 * cfg.time_bin_size;

            // linear interpolation
            t1 + (t2 - t1) * (threshold * signal_max - v1) / (v2 - v1)
        }
    };

    let t_start = time_to_threshold(cfg.post_processing.rise_time_threshold_start);
    let t_end = time_to_threshold(cfg.post_processing.rise_time_threshold_end);

    t_end - t_start
}

// utilities
trait Diff {
    fn diff(&self) -> Vec<f64>;
}

impl Diff for Vec<f64> {
    /// Equivalent to np.diff in python
    fn diff(&self) -> Vec<f64> {
        self.windows(2).map(|w| w[1] - w[0]).collect()
    }
}

/// Post-processing to get temperature response.
/// Integrate ODEs for sensor and absorber temperatures using explict Euler steps.
pub fn get_temperature_response(e_results_vec: &[EnergyResults], cfg: &Config) -> Option<Vec<f64>> {
    if cfg.post_processing.calculate_sensor_temperature {
        let mut temp_sens_vec = Vec::with_capacity(cfg.number_of_timesteps);
        let mut temp_sens = 0.0;
        let mut temp_abs = 0.0;
        let dt = cfg.time_bin_size;

        // constant prefactors
        let a = cfg.post_processing.g_abs_sens / cfg.post_processing.c_sens * dt;
        let b = -cfg.post_processing.g_abs_sens / cfg.post_processing.c_abs * dt;
        let c = 1.0 / cfg.post_processing.c_abs;

        temp_sens_vec.push(temp_sens);
        for t in 1..cfg.number_of_timesteps {
            let delta_temp = temp_abs - temp_sens;
            let delta_energy =
                e_results_vec[t].e_absorbed_total - e_results_vec[t - 1].e_absorbed_total;

            temp_sens += a * delta_temp;
            temp_abs += b * delta_temp + c * delta_energy;

            temp_sens_vec.push(temp_sens);
        }

        return Some(temp_sens_vec);
    }

    None
}
