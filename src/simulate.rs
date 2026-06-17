//! Run multiple particle simulations, either single-threaded or multi-threaded.

use crate::config::Config;
use crate::data_structures::{
    DiffuseDistribution, ScatteringPoint, SpecularDistribution, SpecularityModel,
};
use crate::materials::Si;
use crate::simulate_particle::simulate_particle;
use crate::top_bottom_scattering::{AbsorberPolygon, AbsorberRegion};
use crate::wall_scattering::Wall;

use rayon::prelude::*;
use std::vec::Vec; // for multiprocessing

/// Size of sine/cosine lookup tables. Must be a power of 2 for bitwise AND optimization.
pub const SIN_TABLE_SIZE: usize = 2048;
pub const SQRT_TABLE_SIZE: usize = 8192;
pub type SinTable = [f32; SIN_TABLE_SIZE];
pub type SqrtTable = [f64; SQRT_TABLE_SIZE];

#[derive(Clone)]
pub struct EnergyResults {
    pub e_loss: f64,
    pub e_absorbed_total: f64,
}

pub struct SimulationSetup {
    pub wall: Wall,
    pub absorber: AbsorberRegion,
    pub clamps_top: Vec<AbsorberPolygon>,
    pub clamps_bottom: Vec<AbsorberPolygon>,
    pub material: Si,
    pub specularity_model: SpecularityModel,
    pub specular_distribution: SpecularDistribution,
    pub diffuse_distribution: DiffuseDistribution,
    pub sin_table: SinTable,
    pub sqrt_table: SqrtTable,
}

/// Create sine and square root lookup tables for fast random direction sampling
#[allow(clippy::needless_range_loop)]
fn precompute_tables() -> (SinTable, SqrtTable) {
    use std::f32::consts::PI;

    debug_assert!(
        SIN_TABLE_SIZE.is_power_of_two(),
        "SIN_TABLE_SIZE must be a power of 2 for bitwise AND optimization, but got {}",
        SIN_TABLE_SIZE
    );
    debug_assert!(
        SQRT_TABLE_SIZE.is_power_of_two(),
        "SQRT_TABLE_SIZE must be a power of 2 for bitwise AND optimization, but got {}",
        SQRT_TABLE_SIZE
    );

    let mut sin_table = [0.0f32; SIN_TABLE_SIZE];
    let mut sqrt_table = [0.0f64; SQRT_TABLE_SIZE];
    for i in 0..SIN_TABLE_SIZE {
        let value = (i as f32) / (SIN_TABLE_SIZE as f32);
        // offsets so that cosine can also be accessed from sine table
        // See example in function crate::reflection_models::sample_sin_cos_uniform
        let offset = 0.25 * PI + 2.0 * PI * 0.5 / (SIN_TABLE_SIZE as f32);
        let angle = value * 2.0 * PI + offset;
        sin_table[i] = angle.sin();
    }
    for i in 0..SQRT_TABLE_SIZE {
        let value = ((i as f64) + 0.5) / (SQRT_TABLE_SIZE as f64);
        sqrt_table[i] = value.sqrt();
    }
    (sin_table, sqrt_table)
}

/// Simulate many particles using a single thread, no multiprocessing.
/// Includes simulation setup from config and
pub fn simulate_single_thread(
    cfg: &Config,
) -> Result<(Vec<Vec<ScatteringPoint>>, Vec<EnergyResults>), String> {
    let number_of_particles = cfg.number_of_particles;
    let setup = set_up_simulation(cfg)?;

    // Initialize structure for simulation results
    let mut e_results_vec = vec![
        EnergyResults {
            e_loss: 0.0,
            e_absorbed_total: 0.0,
        };
        cfg.number_of_timesteps
    ];

    let mut scattering_points_all = Vec::with_capacity(number_of_particles);

    // main simulation loop
    for _ in 0..number_of_particles {
        let (scattering_points, e_results) = simulate_particle(&setup, cfg);

        scattering_points_all.push(scattering_points);
        for t in 0..cfg.number_of_timesteps {
            e_results_vec[t].e_loss += e_results[t].e_loss;
            e_results_vec[t].e_absorbed_total += e_results[t].e_absorbed_total;
        }
    }

    process_energies(&mut e_results_vec, cfg);

    Ok((scattering_points_all, e_results_vec))
}

/// Simulate many particles using multiple threads.
pub fn simulate_parallel(
    cfg: &Config,
) -> Result<(Vec<Vec<ScatteringPoint>>, Vec<EnergyResults>), String> {
    let setup = set_up_simulation(cfg)?;

    // Prepare a template for time-energy records
    let e_results_template = vec![
        EnergyResults {
            e_loss: 0.0,
            e_absorbed_total: 0.0
        };
        cfg.number_of_timesteps
    ];

    // Scattering points are not collected in parallel version for simplicity
    // If you need that, use the single-threaded version
    let scattering_points_all: Vec<Vec<ScatteringPoint>> = Vec::new();

    // Parallel simulation
    let mut e_results_vec: Vec<EnergyResults> = (0..cfg.number_of_particles)
        .into_par_iter()
        // Run the simulation for each particle,
        // accumulating results in thread-local e_results vector.
        // Only care about the time-dependent energy results here
        .fold(
            || e_results_template.clone(),
            |mut e_results_local, _i| {
                let (_, e_results) = simulate_particle(&setup, cfg);
                for t in 0..cfg.number_of_timesteps {
                    e_results_local[t].e_loss += e_results[t].e_loss;
                    e_results_local[t].e_absorbed_total += e_results[t].e_absorbed_total;
                }
                e_results_local
            },
        )
        // combine results, e.g. from different threads
        .reduce(
            || e_results_template.clone(),
            |mut e_results_total, e_results_local| {
                for t in 0..cfg.number_of_timesteps {
                    e_results_total[t].e_loss += e_results_local[t].e_loss;
                    e_results_total[t].e_absorbed_total += e_results_local[t].e_absorbed_total;
                }
                e_results_total
            },
        );

    process_energies(&mut e_results_vec, cfg);

    Ok((scattering_points_all, e_results_vec))
}

fn set_up_simulation(cfg: &Config) -> Result<SimulationSetup, String> {
    let material = Si::new(cfg.default_speed);
    let wall = Wall::new(&cfg.outside_wall, &cfg.inside_walls);
    if !wall.is_strictly_inside(cfg.particle_source.x, cfg.particle_source.y) {
        eprintln!("Particle source is not inside simulation domain. Please check your config.");
        return Err("Particle source is not inside simulation domain.".to_string());
    }

    let specularity_model: SpecularityModel = match cfg.specularity_model.to_lowercase().as_str() {
        "constant" => SpecularityModel::Constant,
        "soffer" => SpecularityModel::Soffer,
        _ => {
            eprintln!(
                "Invalid specularity model specified in config. Valid options are: Constant, Soffer."
            );
            return Err("Invalid specularity model specified in config.".to_string());
        }
    };
    let specular_distribution = match cfg.specular_distribution.to_lowercase().as_str() {
        "ideal" => SpecularDistribution::Ideal,
        "phong" => SpecularDistribution::Phong,
        "phongrescaled" => SpecularDistribution::PhongRescaled,
        _ => {
            eprintln!(
                "Invalid specular distribution specified in config. Valid options are: ideal, phong, phongrescaled."
            );
            std::process::exit(1);
        }
    };
    let diffuse_distribution = match cfg.diffuse_distribution.to_lowercase().as_str() {
        "lambertian" => DiffuseDistribution::Lambertian,
        "uniform" => DiffuseDistribution::Uniform,
        _ => panic!("Invalid diffuse distribution specified in config."),
    };

    let mut polygons = Vec::new();
    for polygon in &cfg.absorbers.polygons {
        polygons.push(AbsorberPolygon::new(polygon));
    }
    let absorber = AbsorberRegion::new(polygons);

    let mut clamps_top = Vec::new();
    let mut clamps_bottom = Vec::new();
    if cfg.include_clamps {
        for polygon in &cfg.clamps_top.polygons {
            clamps_top.push(AbsorberPolygon::new(polygon));
        }
        for polygon in &cfg.clamps_bottom.polygons {
            clamps_bottom.push(AbsorberPolygon::new(polygon));
        }
    }

    let (sin_table, sqrt_table) = precompute_tables();

    let setup = SimulationSetup {
        wall,
        absorber,
        clamps_top,
        clamps_bottom,
        material,
        specularity_model,
        specular_distribution,
        diffuse_distribution,
        sin_table,
        sqrt_table,
    };

    Ok(setup)
}

/// Average energy results over particles, and compute cumulative absorbed and lost energy.
fn process_energies(e_results_vec: &mut [EnergyResults], cfg: &Config) {
    // Average results
    for element in e_results_vec.iter_mut() {
        let n = cfg.number_of_particles as f64;
        element.e_loss /= n;
        element.e_absorbed_total /= n;
    }

    // Compute cumulative sum of absorbed energy
    let mut e_cumulative = EnergyResults {
        e_loss: 0.0,
        e_absorbed_total: 0.0,
    };

    for element in e_results_vec.iter_mut() {
        e_cumulative.e_loss += element.e_loss;
        e_cumulative.e_absorbed_total += element.e_absorbed_total;

        element.e_loss = e_cumulative.e_loss;
        element.e_absorbed_total = e_cumulative.e_absorbed_total;
    }
}

pub fn get_zero_results(cfg: &Config) -> (Vec<EnergyResults>, Option<Vec<f64>>) {
    let e_results_vec = vec![
        EnergyResults {
            e_loss: 0.0,
            e_absorbed_total: 0.0,
        };
        cfg.number_of_timesteps
    ];

    if cfg.post_processing.calculate_sensor_temperature {
        let t_sens_vec = vec![0.0; cfg.number_of_timesteps];
        (e_results_vec, Some(t_sens_vec))
    } else {
        (e_results_vec, None)
    }
}
