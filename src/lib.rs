//! Version 0.1.0
//! # Phonon Monte Carlo
//! A Monte Carlo simulator for phonon transport.
//! The goal is simulating the phonons after an event in the P2 Photon Detector, to:  
//! 1. Estimate the fraction of energy lost through the bridges.  
//! 2. Estimate the timeline of energy absorbed in the absorber regions, including rise time.  
//!
//! The simulation models phonons as particles moving through a silicon, with either specular or diffuse scattering at the boundaries. Absorption occurs when phonons hit absorber regions at the top.
//!
//! This work started originally from the python project FreePATHS by Roman Anufriev (GPL3 license), which can be found at:  
//! <https://anufrievroman.gitbook.io/freepaths>
//! None of this code still remains any more.
//!
//! ## Assumptions
//! The current model makes these simplifying assumptions:
//!  - Each photon absorption events is a **point source** of many athermal phonons.
//!  - all phonons have the **same speed**, independent of frequency, branch, or direction
//!  - top and bottom surfaces have a **fixed probability of specular vs diffuse scattering**, independent of angle of incidence or wavevector
//!  - absorbers **absorb a fixed fraction** of phonon energy when hit, independent of angle of incidence or wavevector
//!  - the phonons transmitted to the absorber **thermalize instantly in the absorber**
//!  - bridge hits **lose entirely** the hitting phonon. There is no chance of the phonon coming back.
//!
//! ## Not implemented
//!  - wavevectors, branches, frequencies
//!  - Internal elastic scattering
//!  - down conversion through 3-phonon processes
//!  - wavevector-dependent scattering
//!  - materials other than silicon
//!  - non-convex polygons for outside wall
//!
//! ## Getting started
//! ### Installation and Usage
//! #### At KIP with milkyway
//! ```python
//! pip install --proxy=proxy.kip.uni-heidelberg.de:8080 --find-links="//twix3/bolo/programs/python/phonon_monte_carlo" phonon_monte_carlo
//! ```
//! Run from python as
//! ```python
//! import phonon_monte_carlo
//! ```
//!
//! #### At KIP with Jupyter
//! The Jupyter server doesn't allow for pip installations. Instead of pip installing, you'll need to import it directly every time:
//! ```python
//! import sys
//! sys.path.insert(0, "//twix3/bolometer/programs/python")
//! import phonon_monte_carlo
//! ```
//!
//! ### Running the simulation
//! To run the simulation, have a look at the `config_base.toml` file for an example configuration and
//! at `Simulate.ipynb`, where different designs have already been simulated.
//! Look at `2025_11_11_Phonon Monte Carlo.pdf` to get a sense of what can be done.
//!
//! The simulation can be run from python as
//! ```python
//! import phonon_monte_carlo
//! phonon_monte_carlo.run("/path/to/config.toml", "/path/to/sources.parquet")
//! ```
//! To run the simulation, you don't need any Rust at all, only a TOML configuration file, like `config_base.toml`, has to be created.
//! TOML is like JSON but without so many brackets. This can be also automated from python using the `toml` python package.
//!
//! ### Modifying the simulation code
//! To modify the code itself, you need to have Rust installed.
//! A good starting point is:
//!  - Install Rust from <https://www.rust-lang.org/tools/install>
//!  - use VS Code with the rust-analyzer extension. This will help with syntax and type checking. Essentially, if rust-analyzer doesn't show errors, it will compile.
//!  - run `cargo build --release` to compile. The --release flag enables compiler optimizations, the corresponding file will be in `target/release/phonon_monte_carlo.exe`.
//!
//! On Rust: Rust is similar to C/C++ (it's fast and low level),
//! except that it helps avoid many common bugs, such as memory leaks or type mismatches.
//! Python bindings can be created using the python module `maturin`.
//!
//! ## Limitations
//! The current model is one of the simplest possible. It is probably not quantitatively accurate.
//! Still, it gives a first estimate of the expected energy loss and timing characteristics due to different geometries.
//!  
//! As a next step, wavevectors, branches and frequencies could implemented.
//! The most important unconsidered effect is likely wavevector-dependent scattering at rough surfaces,
//! as the distribution (e.g. Lambertian vs. uniform) after diffuse scattering strongly affects the results.  
//!
//! Internal scattering and down-conversion add some diffuseness as well, but are likely less important than surface scattering.
//! However, since for wavevector-dependent surface scattering the theoretical models are debatable and incomplete (to my knowledge),
//! and the necessary parameters (roughness, roughness autocorrelation length) are hard to measure accurately for the entire wafer, we stop at this simple model for now.
//! We instead go back to an experimental approach.
//!
//! Provided under a GPL3 license, see license file.  
//! Sebastian Hilscher, June 2026
//!
//! ## Example
//! ```python
//! import os
//! import numpy as np
//! import pandas as pd
//! import matplotlib.pyplot as plt
//!
//! # On Jupyter:
//! # import sys
//! # sys.path.insert(0, "//twix3/bolometer/programs/python")
//! import phonon_monte_carlo
//! import toml
//!
//! # Assume that a file old_config.toml has previously been created
//! # Let's say you want to make a change to it
//! config = toml.load("old_config.toml")
//! config["number_of_particles"] = 10000
//! config["output_folder"] = "cool_example"
//! with open("config.toml", "w") as f:
//!     toml.dump(config, f)
//! os.makedirs("cool_example", exist_ok=True)
//!
//! # Set up the particle source positions
//! sources_X = np.linspace(-0.005, 0.005, 10)
//! sources_Y = np.zeros(10)
//! sources_Z = np.zeros(10)
//! sources = pd.DataFrame({"Source X": sources_X, "Source Y": sources_Y, "Source Z": sources_Z})
//! sources.to_parquet("sources.parquet")
//!
//! # Run simulation for all sources
//! phonon_monte_carlo.run("config.toml", "sources.parquet")
//!
//! # Evaluate output
//! df = pd.read_parquet("cool_example/output.parquet")
//! df.head()
//! ```
//!

use crate::config::{Config, load_particle_sources};
use crate::data_structures::PointXYZ;
use crate::postprocessing::{calculate_rise_time, get_temperature_response, load_decay_template};
use crate::simulate::{EnergyResults, get_zero_results, simulate_parallel, simulate_single_thread};
use arrow_array::builder::{FixedSizeListBuilder, Float32Builder};
use arrow_array::{
    ArrayRef, FixedSizeListArray, Float32Array, Float64Array, RecordBatch, StringArray, UInt32Array,
};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::arrow_writer::ArrowWriter;
use parquet::basic::GzipLevel;
use parquet::file::properties::WriterProperties;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rayon::ThreadPoolBuilder;
use std::fs::File;
use std::sync::Arc;

pub mod config;
pub mod data_structures;
mod materials;
mod phonon;
mod postprocessing;
mod reflection_models;
mod simulate;
mod simulate_particle;
mod top_bottom_scattering;
mod wall_scattering;

// Main function.
//
// Parses command line arguments to get input file, loads configuration file,
// checks parameter validity, and starts the simulation.
// See module 'simulate' for simulation functions.
// pub fn main() {
//     // Get input file from command line arguments
//     let args: Vec<String> = std::env::args().collect();
//     if args.len() < 2 {
//         eprintln!(
//             "Error: No input file provided.\n
//             Usage: phonon_monte_carlo.exe -input=\"path/to/config.toml\""
//         );
//         std::process::exit(1);
//     }
//     let mut input_file = "";
//     let mut sources_file = "";
//     let mut verbose = true;
//     for arg in args.iter().skip(1) {
//         if arg.starts_with("-input=") {
//             input_file = arg.trim_start_matches("-input=").trim_matches('"');
//         }
//         if arg.starts_with("-sources=") {
//             sources_file = arg.trim_start_matches("-sources=").trim_matches('"');
//         }
//         if arg.starts_with("--quiet") {
//             verbose = false;
//         }
//     }

//     if input_file.is_empty() {
//         eprintln!("Error: No input file provided with -input=\"...\"");
//         std::process::exit(1);
//     }

//     match run_main(input_file, sources_file, verbose) {
//         Ok(stdout) => println!("{}", stdout),
//         Err(e) => eprintln!("{}", e),
//     };
// }

#[pyfunction]
#[pyo3(signature = (config, sources=None, quiet=false))]
fn run(py: Python<'_>, config: &str, sources: Option<&str>, quiet: bool) -> PyResult<()> {
    match run_main(config, sources, !quiet) {
        Ok(output) => {
            if !quiet {
                let sys = py.import("sys")?;
                let stdout = sys.getattr("stdout")?;
                stdout.call_method1("write", (output,))?;
                stdout.call_method1("flush", ())?;
            }
            Ok(())
        }
        Err(e) => Err(PyRuntimeError::new_err(e)),
    }
}

#[pymodule]
fn phonon_monte_carlo(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run, m)?)?;
    Ok(())
}

fn run_main(input_file: &str, sources_file: Option<&str>, verbose: bool) -> Result<String, String> {
    // Load configuration from input file
    let mut cfg = match Config::from_toml_file(input_file) {
        Ok(cfg) => cfg,
        Err(e) => {
            eprintln!("Configuration error: {}", e);
            return Err(format!("Configuration error: {}", e));
        }
    };
    match cfg.check_parameter_validity() {
        Ok(_) => {}
        Err(e) => {
            eprintln!("Configuration error: {}", e);
            return Err(format!("Configuration error: {}", e));
        }
    }

    // Load particle sources from file
    let sources: Vec<PointXYZ> = match sources_file {
        Some(file) => match load_particle_sources(file) {
            Ok(sources) => sources,
            Err(e) => {
                eprintln!("Error loading particle sources: {}", e);
                return Err(format!("Error loading particle sources: {}", e));
            }
        },
        None => {
            vec![PointXYZ {
                x: cfg.particle_source.x,
                y: cfg.particle_source.y,
                z: cfg.particle_source.z,
            }]
        }
    };

    // Create structures for output data
    let num_sources = sources.len();
    let (schema, mut writer) = setup_parquet_writer(&cfg);
    let output_folder = &cfg.output_folder;
    let wall = wall_scattering::Wall::new(&cfg.outside_wall, &cfg.inside_walls);

    // Load decay template for rise time calculation, if needed. Needed only once
    let decay_template = if cfg.post_processing.convolve_with_decay_template {
        Some(load_decay_template(&cfg))
    } else {
        None
    };

    let mut num_ignored_sources = 0;
    for (i, source) in sources.iter().enumerate() {
        if !(wall.is_strictly_inside(source.x, source.y) && source.z.abs() <= cfg.thickness / 2.0) {
            num_ignored_sources += 1;
            let (e_results, sensor_temperature) = get_zero_results(&cfg);
            let rise_time = if cfg.post_processing.calculate_rise_time {
                Some(0.0)
            } else {
                None
            };

            write_results(
                i,
                source.x,
                source.y,
                source.z,
                &e_results,
                sensor_temperature,
                rise_time,
                &cfg,
                schema.clone(),
                &mut writer,
            );
            continue;
        }

        cfg.particle_source.x = source.x;
        cfg.particle_source.y = source.y;
        cfg.particle_source.z = source.z;
        cfg.output_file_points =
            format!("{}/scattering_points_source_{}.parquet", output_folder, i);

        // Run simulation
        let results = if cfg.use_multiprocessing {
            if cfg.num_workers > 0 {
                ThreadPoolBuilder::new()
                    .num_threads(cfg.num_workers)
                    .build()
                    .unwrap()
                    .install(|| simulate_parallel(&cfg))
            } else {
                simulate_parallel(&cfg)
            }
        } else {
            simulate_single_thread(&cfg)
        };

        let (scattering_points_all, e_results) = results?;

        // Do post-processing that would otherwise be done in python
        let sensor_temperature = get_temperature_response(&e_results, &cfg);
        let rise_time = match cfg.post_processing.calculate_rise_time {
            true => Some(calculate_rise_time(&e_results, &decay_template, &cfg)),
            false => None,
        };

        // Save results to files
        if cfg.write_scattering_points {
            write_scattering_points(scattering_points_all, &cfg);
        }
        write_results(
            i,
            source.x,
            source.y,
            source.z,
            &e_results,
            sensor_temperature,
            rise_time,
            &cfg,
            schema.clone(),
            &mut writer,
        );
    }

    match writer.close() {
        Ok(_) => {}
        Err(e) => {
            eprintln!("Unable to close Parquet writer: {}", e);
            return Err(format!("Unable to close Parquet writer: {}", e));
        }
    }

    let mut stdout = "".to_string();
    if verbose {
        stdout.push_str(&format!(
            "Completed {} simulations",
            num_sources - num_ignored_sources
        ));

        if num_ignored_sources > 0 {
            stdout.push_str(&format!(
                "\nWrote zeros for {} sources that were outside the domain",
                num_ignored_sources
            ));
        }
    }
    Ok(stdout)
}

/// Write scattering points to a Parquet file.
fn write_scattering_points(
    scattering_points_all: Vec<Vec<data_structures::ScatteringPoint>>,
    cfg: &Config,
) {
    // Set up columns
    let fields = vec![
        Field::new("Particle", DataType::UInt32, false),
        Field::new("X", DataType::Float64, false),
        Field::new("Y", DataType::Float64, false),
        Field::new("Z", DataType::Float64, false),
        Field::new("Time", DataType::Float64, false),
        Field::new("Location", DataType::Utf8, false),
        #[cfg(debug_assertions)]
        Field::new("is_inside", DataType::Boolean, false),
    ];
    let schema = Arc::new(Schema::new(fields));

    let compression = parquet::basic::Compression::GZIP(GzipLevel::try_new(9).unwrap());
    let props = WriterProperties::builder()
        .set_compression(compression)
        .build();
    let file = File::create(&cfg.output_file_points).expect("Unable to create file");
    let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))
        .expect("Unable to create Parquet writer");

    // write points
    for (i, scattering_points) in scattering_points_all.iter().enumerate() {
        let mut x_values = Vec::with_capacity(scattering_points.len());
        let mut y_values = Vec::with_capacity(scattering_points.len());
        let mut z_values = Vec::with_capacity(scattering_points.len());
        let mut time_values = Vec::with_capacity(scattering_points.len());
        let mut location_values = Vec::with_capacity(scattering_points.len());
        #[cfg(debug_assertions)]
        let mut is_inside_values = Vec::with_capacity(scattering_points.len());

        for point in scattering_points {
            x_values.push(point.x);
            y_values.push(point.y);
            z_values.push(point.z);
            time_values.push(point.time);
            location_values.push(point.location.as_str().to_string());
            #[cfg(debug_assertions)]
            is_inside_values.push(point.is_inside);
        }

        #[cfg(debug_assertions)]
        {
            let any_point_is_outside = is_inside_values.iter().any(|&val| !val);
            if !any_point_is_outside {
                continue;
            }
        }

        let columns: Vec<ArrayRef> = vec![
            Arc::new(UInt32Array::from(vec![i as u32; scattering_points.len()])),
            Arc::new(Float64Array::from(x_values)),
            Arc::new(Float64Array::from(y_values)),
            Arc::new(Float64Array::from(z_values)),
            Arc::new(Float64Array::from(time_values)),
            Arc::new(StringArray::from(location_values)),
            #[cfg(debug_assertions)]
            Arc::new(arrow_array::BooleanArray::from(is_inside_values)),
        ];

        let batch =
            RecordBatch::try_new(schema.clone(), columns).expect("Unable to create record batch");
        writer.write(&batch).expect("Unable to write record batch");
    }

    writer
        .close()
        .expect("Unable to close Parquet writer for scattering points");
}

/// Set up Parquet writer for the main output file, and return the schema and writer.
fn setup_parquet_writer(cfg: &Config) -> (Arc<Schema>, ArrowWriter<File>) {
    let n = cfg.number_of_timesteps;

    // Helper function to create a fixed-size list field for the schema
    let list = |name: &str, dtype: DataType| {
        Field::new(
            name,
            DataType::FixedSizeList(Arc::new(Field::new("item", dtype, false)), n as i32),
            false,
        )
    };

    let mut fields = vec![
        Field::new("Source Index", DataType::UInt32, false),
        Field::new("Source X", DataType::Float32, false),
        Field::new("Source Y", DataType::Float32, false),
        Field::new("Source Z", DataType::Float32, false),
        Field::new("Time Bin Size", DataType::Float32, false),
        Field::new("Final Energy Absorbed", DataType::Float32, false),
    ];

    if cfg.write_absorbed_energy {
        fields.push(list("Energy Absorbed", DataType::Float32));
    }

    if cfg.write_lost_energy {
        fields.push(list("Energy Lost", DataType::Float32));
    }
    if cfg.post_processing.calculate_sensor_temperature {
        fields.push(list("Sensor Temperature", DataType::Float32));
    }

    if cfg.post_processing.calculate_rise_time {
        fields.push(Field::new("Rise Time", DataType::Float32, false));
    }

    let schema = Arc::new(Schema::new(fields));

    let compression = parquet::basic::Compression::GZIP(GzipLevel::try_new(9).unwrap());
    let props = WriterProperties::builder()
        .set_compression(compression)
        .build();

    let output_file_name = format!("{}/output.parquet", cfg.output_folder);
    let file = File::create(&output_file_name).expect("Unable to create file");
    let writer = ArrowWriter::try_new(file, schema.clone(), Some(props))
        .expect("Unable to create Parquet writer");

    (schema, writer)
}

/// Append results for a single particle source to the main Parquet output file.
/// The columns of the file depend on the configuration.
fn write_results(
    i: usize,
    source_x: f64,
    source_y: f64,
    source_z: f64,
    e_results: &Vec<EnergyResults>,
    sensor_temperature: Option<Vec<f64>>,
    rise_time: Option<f64>,
    cfg: &Config,
    schema: Arc<Schema>,
    writer: &mut ArrowWriter<File>,
) {
    let n = cfg.number_of_timesteps;
    let mut energy_absorbed: Vec<f32> = Vec::with_capacity(n);
    let mut energy_lost: Vec<f32> = Vec::with_capacity(n);

    // convert to f32
    for record in e_results {
        energy_absorbed.push(record.e_absorbed_total as f32);
        energy_lost.push(record.e_loss as f32);
    }

    let mut columns: Vec<ArrayRef> = vec![
        Arc::new(UInt32Array::from(vec![i as u32])),
        Arc::new(Float32Array::from(vec![source_x as f32])),
        Arc::new(Float32Array::from(vec![source_y as f32])),
        Arc::new(Float32Array::from(vec![source_z as f32])),
        Arc::new(Float32Array::from(vec![cfg.time_bin_size as f32])),
        Arc::new(Float32Array::from(vec![
            energy_absorbed.last().cloned().unwrap_or(0.0),
        ])),
    ];

    if cfg.write_absorbed_energy {
        columns.push(make_fixed_list_array_f32(&energy_absorbed, n as i32));
    }

    if cfg.write_lost_energy {
        columns.push(make_fixed_list_array_f32(&energy_lost, n as i32));
    }

    if let Some(t_sens) = sensor_temperature {
        let t_sens: Vec<f32> = t_sens.into_iter().map(|val| val as f32).collect();
        columns.push(make_fixed_list_array_f32(&t_sens, n as i32));
    }
    if cfg.post_processing.calculate_rise_time {
        let rise_time = rise_time.unwrap_or(0.0) as f32;
        columns.push(Arc::new(Float32Array::from(vec![rise_time])));
    }

    let batch =
        RecordBatch::try_new(schema.clone(), columns).expect("Unable to create record batch");
    writer.write(&batch).expect("Unable to write record batch");
}

fn make_fixed_list_array_f32(data: &[f32], n: i32) -> Arc<FixedSizeListArray> {
    let item_field = Arc::new(Field::new("item", DataType::Float32, false));
    let mut builder = FixedSizeListBuilder::new(Float32Builder::new(), n);
    builder = builder.with_field(item_field);

    builder.values().append_slice(data);
    builder.append(true); // finish entry

    Arc::new(builder.finish())
}
