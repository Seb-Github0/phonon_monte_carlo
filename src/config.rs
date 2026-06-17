//! Configuration struct that is loaded from a TOML file. Look here for description of input parameters.
use crate::data_structures::{AbsorberRegionConfig, ParticleSource, PointXYZ, WallConfig};

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::{fs, io::Read};

use arrow_array::{Array, Float64Array, RecordBatch};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

/// Struct representing the simulation configuration.
///
/// This struct is deserialized from a TOML configuration file (config.toml)
/// and contains all parameters that can be adjusted by the user to control the simulation.
///
/// See below for detailed documentation of each field.
// These are the same fields that have to be specified in the TOML configuration file.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Config {
    // General parameters
    /// Path to output folder where results will be saved.
    pub output_folder: String,

    /// Path to output file for scattering points.
    /// Data will be written in Parquet format.
    #[doc(hidden)]
    #[serde(skip)]
    pub output_file_points: String,

    /// Number of particles to simulate.
    pub number_of_particles: usize,

    /// Whether to use multiprocessing for the simulation.
    /// Uses all available CPU cores if true.
    /// If false, runs single-threaded.
    pub use_multiprocessing: bool,

    /// Number of worker threads to use for multiprocessing.
    /// If `use_multiprocessing` is true, this specifies the number of threads.
    /// If 0, uses all available CPU cores.
    /// If `use_multiprocessing` is false, this parameter is ignored.
    #[serde(default = "default_num_workers")]
    pub num_workers: usize,

    /// Whether to write scattering points to output file.
    /// Only works in single-threaded mode (`use_multiprocessing = false`)
    /// Disable to save disk space.
    /// If enabled, will write to path specified by `output_file_points`.
    ///
    /// If enabled, it is recommended to reduce `number_of_particles`
    /// to avoid (very) large output files.
    pub write_scattering_points: bool,

    /// Whether to write the lost energy fraction at each time step to the output file.
    /// Not necessary to get absorbed energy fraction or rise time. Disable to save disk space.
    pub write_lost_energy: bool,
    /// Whether to write the absorbed energy fraction at each time step to the output file.
    /// Not necessary to get rise time. Disable to save disk space.
    pub write_absorbed_energy: bool,

    // Time parameters
    /// Total simulation time in seconds.
    pub time_total: f64,
    /// Size of each time bin in seconds. This determines the time resolution of the
    /// data in `output_file_time_resolved`. It does not affect the simulation itself.
    pub time_bin_size: f64,
    #[doc(hidden)]
    #[serde(skip)]
    pub number_of_timesteps: usize,

    // Material
    /// Material name. Currently only "Si" (silicon) is supported.
    pub material: String,
    /// Particle speed in m/s.
    /// Typical value for transverse acoustic phonons in silicon is ~6000 m/s.
    pub default_speed: f64,
    /// Whether to use branch-specific speed given by material.
    /// If true, at each scattering event, a phonon branch (LA, TA1, TA2)
    /// is randomly assigned and the corresponding speed in the material is used.
    /// If false, `default_speed` is used.
    pub use_branch_speed: bool,

    // Absorption
    /// Absorption probability when a particle hits an absorber region (0.0 to 1.0).
    /// This probability is assumed to be constant, independent of incidence angle.
    pub absorptivity: f64,

    // Internal scattering
    /// Whether to include internal scattering events in the simulation.
    /// Currently not implemented.
    pub include_internal_scattering: bool,

    // System dimensions
    /// Thickness of the material in meters (z-dimension).
    /// The material extends from -thickness/2 to +thickness/2 in z.
    pub thickness: f64,
    /// Whether the material is two-dimensional (true) or three-dimensional (false).
    /// If true, z-velocity is set to zero, no top/bottom scattering.
    pub is_two_dimensional_material: bool,

    // Particle sources
    /// Particle source in the simulation.
    /// Each source defines position, size, and angular distribution of emitted particles.
    /// See `ParticleSource` struct for details.
    pub particle_source: ParticleSource,

    // Outside wall
    /// Defines the outside wall boundary of the simulation domain.
    /// Must be a convex polygon.
    ///
    /// See `WallConfig` struct for details.
    pub outside_wall: WallConfig,

    /// List of inside wall boundaries (holes) in the simulation domain.
    /// Must be convex polygons.
    /// See `WallConfig` struct for details.
    ///
    /// Can be empty or omitted.
    #[serde(default)]
    pub inside_walls: Vec<WallConfig>,

    // Absorber regions
    /// Defines the absorber regions at the top surface of the material.
    /// See `AbsorberRegion` struct for details.
    pub absorbers: AbsorberRegionConfig,

    /// Whether to include energy loss through clamps at the top and bottom surfaces.
    pub include_clamps: bool,
    /// List of polygons defining the clamp region at the top surface.
    /// Must be convex. Supports only line segments.
    pub clamps_top: AbsorberRegionConfig,
    /// List of polygons defining the clamp region at the bottom surface.
    /// Must be convex. Supports only line segments.
    pub clamps_bottom: AbsorberRegionConfig,
    /// Energy fraction lost by particles hitting the clamps, must be between 0.0 and 1.0.
    pub clamps_absorptivity: f64,

    // Roughness
    /// Model to determine whether scattering at the top/bottom surfaces is specular or diffuse.
    /// Must be "Constant" or "Soffer".
    /// If "Constant", the probability of specular scattering is given by `p_specular`,
    /// independent of incidence angle.
    /// If "Soffer", the probability of specular scattering depends on the incidence angle
    /// like ``p_specular(theta) = exp(-4 * pi^2 * roughness^2 * k^2 * cos(theta)^2``.
    /// Assuming that k is approximately constant, ``p_specular(theta) = exp(-C * cos(theta)^2)`` is used here.
    /// The prefactor C can be set by `specularity_roughness_prefactor`.
    pub specularity_model: String,
    /// Probability of specular scattering at top/bottom surfaces
    /// (1.0 = fully specular, 0.0 = fully diffuse). Must be between 0.0 and 1.0.
    pub p_specular: f64,
    /// Prefactor for the roughness in the Soffer model of specularity.
    /// Higher values correspond to a rougher surface and lower the probability of specular scattering..
    /// Must be >= 0.0.
    pub specularity_roughness_prefactor: f64,
    /// Angular distribution after specular scattering at top/bottom surfaces.
    /// Must be "Ideal", "Phong", or "PhongRescaled".
    pub specular_distribution: String,
    /// Phong exponent for specular scattering angular distribution.
    /// Only used if `specular_distribution` is "Phong" or "PhongRescaled".
    /// Higher values lead to a narrower distribution around the mirror angle.
    pub phong_exponent: i32,

    // tell serde not to deserialize this field
    #[doc(hidden)]
    #[serde(skip)]
    pub(crate) phong_exponent_sampling: f64,
    /// Angular distribution after diffuse scattering at top/bottom surfaces.
    /// Must be "Lambertian" or "Uniform".  
    ///
    /// If "Lambertian", the distribution is weighted by cos(theta),
    /// favoring angles close to the surface normal. This is how full diffuse scattering
    /// is usually modelled in Ray Tracing software for computer graphics.  
    ///
    /// If "Uniform", all angles are equally probable.
    /// This weights angles close to the surface more strongly compared to "Lambertian".
    pub diffuse_distribution: String,

    // Post-processing
    pub post_processing: PostProcessingConfig,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PostProcessingConfig {
    /// Whether to calculate sensor temperature rise based on absorbed energy timeline.
    /// If true, requires c_abs, c_sens, and g_abs_sens to be set and
    /// a column will be added to the time-resolved output file.
    pub calculate_sensor_temperature: bool,
    /// Heat capacity of the absorber in J/K
    pub c_abs: f64,
    /// Heat capacity of the sensor in J/K
    pub c_sens: f64,
    /// Thermal conductivity between absorber and sensor, in W/K
    pub g_abs_sens: f64,
    /// Whether to calculate the threshold-to-threshold rise time
    pub calculate_rise_time: bool,
    /// Start threshold for rise time calculation, as a fraction of the signal maximum (0.0 to 1.0).
    pub rise_time_threshold_start: f64,
    /// End threshold for rise time calculation, as a fraction of the signal maximum (0.0 to 1.0).
    pub rise_time_threshold_end: f64,
    /// Whether to convolve the absorbed energy signal with a decay template before calculating the rise time.
    pub convolve_with_decay_template: bool,
    /// Absolute path to the decay template file (CSV format). The file is assumed to have two columns, column 0 is time, column 1 is value, separated by a comma. The file is assumed to have one header row.
    pub file_decay_template: String,
}

impl Config {
    /// Load config from a TOML file path
    pub fn from_toml_file<P: AsRef<Path>>(path: P) -> Result<Config> {
        let path_ref = path.as_ref();

        let mut file = fs::File::open(&path)
            .map_err(|e| anyhow!("Unable to open file {}: {}", path.as_ref().display(), e))?;

        let mut s = String::new();
        file.read_to_string(&mut s)
            .map_err(|e| anyhow!("Failed to read {}: {}", path_ref.display(), e))?;
        let mut cfg: Config = toml::from_str::<Config>(&s).map_err(|e| {
            // Preserve toml error detail (it already mentions missing fields / line/col).
            anyhow!("Failed to parse TOML {}: {}. \nPlease fix the TOML file (missing or invalid keys) and try again.", path_ref.display(), e)
        })?;

        // normalise distribution name
        cfg.phong_exponent_sampling = if cfg.phong_exponent < 1 {
            1.0
        } else {
            1.0 / (cfg.phong_exponent as f64 + 1.0)
        };

        cfg.number_of_timesteps = (cfg.time_total / cfg.time_bin_size).ceil() as usize;

        Ok(cfg)
    }

    /// Check validity of fields provided by the user.
    #[allow(clippy::nonminimal_bool)]
    pub fn check_parameter_validity(&self) -> Result<()> {
        if self.number_of_particles == 0 {
            return Err(anyhow!("number_of_particles must be > 0"));
        }

        // Per-source check if inside boundary:
        let src = &self.particle_source;

        if src.z.abs() > self.thickness / 2.0 {
            return Err(anyhow!(
                "particle_source.z ({}) exceeds thickness/2 ({})",
                src.z,
                self.thickness / 2.0
            ));
        }
        if src.z.abs() > self.thickness / 2.0 {
            return Err(anyhow!(
                "particle_source (z= {}) exceeds thickness/2 ({})",
                src.z,
                self.thickness / 2.0
            ));
        }

        // Check that OutsideWall.points, .is_bridge, and circle_radius have the same length
        let num_points = self.outside_wall.points.len();
        let num_bridge = self.outside_wall.is_bridge.len();
        let num_circle_radius = self.outside_wall.circle_radius.len();
        if num_points != num_bridge {
            return Err(anyhow!(
                "outside_wall.points has length {} but outside_wall.is_bridge has length {};
                they must be the same",
                num_points,
                num_bridge
            ));
        }
        if num_points != num_circle_radius {
            return Err(anyhow!(
                "outside_wall.points has length {} but outside_wall.circle_radius has length {};
                they must be the same",
                num_points,
                num_circle_radius
            ));
        }

        // Check that absorptivity is between 0.0 and 1.0
        if self.absorptivity < 0.0 || self.absorptivity > 1.0 {
            return Err(anyhow!(
                "absorptivity must be between 0.0 and 1.0, got {}",
                self.absorptivity
            ));
        }

        // Check that clamps_absorptivity is between 0.0 and 1.0
        if self.clamps_absorptivity < 0.0 || self.clamps_absorptivity > 1.0 {
            return Err(anyhow!(
                "clamps_absorptivity must be between 0.0 and 1.0, got {}",
                self.clamps_absorptivity
            ));
        }

        match self.specularity_model.to_lowercase().as_str() {
            "constant" | "soffer" => {}
            _ => {
                return Err(anyhow!(
                    "specularity_model must be 'Constant' or 'Soffer', got '{}'",
                    self.specularity_model
                ));
            }
        }

        // Check that p_specular is between 0.0 and 1.0
        if self.p_specular < 0.0 || self.p_specular > 1.0 {
            return Err(anyhow!(
                "p_specular must be between 0.0 and 1.0, got {}",
                self.p_specular
            ));
        }

        if self.specularity_roughness_prefactor < 0.0 {
            return Err(anyhow!(
                "specularity_roughness_prefactor must be >= 0.0, got {}",
                self.specularity_roughness_prefactor
            ));
        }

        // Check that specular_distribution is valid
        match self.specular_distribution.to_lowercase().as_str() {
            "ideal" | "phong" | "phongrescaled" => {}
            _ => {
                return Err(anyhow!(
                    "specular_distribution must be 'Ideal', 'Phong', or 'PhongRescaled', got '{}'",
                    self.specular_distribution
                ));
            }
        }

        // Check that diffuse_distribution is valid
        match self.diffuse_distribution.to_lowercase().as_str() {
            "lambertian" | "uniform" => {}
            _ => {
                return Err(anyhow!(
                    "diffuse_distribution must be 'Lambertian' or 'Uniform', got '{}'",
                    self.diffuse_distribution
                ));
            }
        }

        // Check that each polygon (outside wall, clamps, inside walls) has at least 3 points
        if self.outside_wall.points.len() < 3 {
            return Err(anyhow!(
                "outside_wall must have at least 3 points, got {}",
                self.outside_wall.points.len()
            ));
        }

        for (i, polygon) in self.clamps_top.polygons.iter().enumerate() {
            if polygon.len() < 3 {
                return Err(anyhow!(
                    "clamps_top polygon {} must have at least 3 points, got {}",
                    i,
                    polygon.len()
                ));
            }
        }

        for (i, polygon) in self.clamps_bottom.polygons.iter().enumerate() {
            if polygon.len() < 3 {
                return Err(anyhow!(
                    "clamps_bottom polygon {} must have at least 3 points, got {}",
                    i,
                    polygon.len()
                ));
            }
        }

        for (i, wall_config) in self.inside_walls.iter().enumerate() {
            if wall_config.points.len() < 3 {
                return Err(anyhow!(
                    "inside_walls[{}] must have at least 3 points, got {}",
                    i,
                    wall_config.points.len()
                ));
            }
        }

        Ok(())
    }
}

pub fn load_particle_sources<P: AsRef<Path>>(path: P) -> Result<Vec<PointXYZ>> {
    let file = fs::File::open(&path)
        .map_err(|e| anyhow!("Unable to open file {}: {}", path.as_ref().display(), e))?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;

    let mut sources = Vec::new();

    for batch in reader {
        let batch: RecordBatch = batch?;

        let x_array = batch
            .column_by_name("Source X")
            .ok_or_else(|| anyhow!("Missing column in particle source parquet file: Source X"))?
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| anyhow!("Source X is not Float64"))?;

        let y_array = batch
            .column_by_name("Source Y")
            .ok_or_else(|| anyhow!("Missing column in particle source parquet file: Source Y"))?
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| anyhow!("Source Y is not Float64"))?;

        let z_array = batch
            .column_by_name("Source Z")
            .ok_or_else(|| anyhow!("Missing column in particle source parquet file: Source Z"))?
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| anyhow!("Source Z is not Float64"))?;

        for i in 0..batch.num_rows() {
            if x_array.is_null(i) || y_array.is_null(i) || z_array.is_null(i) {
                continue;
            }

            sources.push(PointXYZ {
                x: x_array.value(i),
                y: y_array.value(i),
                z: z_array.value(i),
            });
        }
    }

    Ok(sources)
}

fn default_num_workers() -> usize {
    // Default to using all available CPU cores
    0
}
