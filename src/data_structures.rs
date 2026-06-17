//! Data structures used throughout the crate.

mod point;
mod vec3;

pub use point::{PointXY, PointXYZ};
pub use vec3::Vec3;

use crate::config::Config;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Rectangle defined by min and max x and y coordinates.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Rectangle {
    pub x_min: f64,
    pub x_max: f64,
    pub y_min: f64,
    pub y_max: f64,
}

/// Position where particles are created.
///
/// If use_exponential_distribution is true, the position of the source is sampled from
/// an exponential distribution with mean depth mean_source_depth below the surface.
///
/// If use_exponential_distribution is false, the position of the source is fixed at z.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ParticleSource {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub use_exponential_distribution: bool,
    pub mean_source_depth: f64,
}

/// Model for specularity (probability of specular reflection).
pub enum SpecularityModel {
    Constant,
    Soffer,
}

/// Type of diffuse distribution for top and bottom scattering.
pub enum DiffuseDistribution {
    Lambertian,
    Uniform,
}

/// Distribution for specular top and bottom scattering.
pub enum SpecularDistribution {
    Ideal,
    Phong,
    PhongRescaled,
}

/// Outside or inside wall polygon consisting of line segments and/or circular arcs.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct WallConfig {
    /// List of XY-Points defining the polygon.
    /// *Must* specify a convex polygon.
    /// The last point is connected to the first point to close the polygon.
    pub points: Vec<PointXY>,
    /// List of booleans indicating whether each edge is a bridge (true) or solid wall (false).
    /// Particles are lost when hitting a bridge edge,
    /// but scattered when hitting a solid wall edge.  
    ///
    /// is_bridge\[i\] corresponds to the edge from points\[i\] to points\[i+1\]
    pub is_bridge: Vec<bool>,

    /// Circle radius for each edge. Set to 0.0 for line segments.
    /// Set to a positive value for Arc segments.
    pub circle_radius: Vec<f64>,

    /// Whether to use diffuse (else: specular) boundary reflection model,
    pub diffuse_scattering: bool,

    /// Whether this wall is an outer wall (else: inner wall = hole).
    pub is_outside: bool,
}

/// Absorber region defined by a list of polygons.
/// Each polygon is specified as list of points [[x1, y1], [x2, y2], ...]
/// Each polygon *must* be convex. Supports only line segments.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AbsorberRegionConfig {
    pub polygons: Vec<Vec<PointXY>>,
}

/// Contains information about the kind of location where a particle scattered.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum ScatteringLocation {
    Wall,
    Top,
    Bottom,
}

impl ScatteringLocation {
    pub fn as_str(&self) -> &'static str {
        match self {
            ScatteringLocation::Wall => "Wall",
            ScatteringLocation::Top => "Top",
            ScatteringLocation::Bottom => "Bottom",
        }
    }
}

/// Represents a point where a particle scatters during the simulation.
/// Used for writing scattering data to output file.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ScatteringPoint {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub time: f64,
    pub location: ScatteringLocation,
    #[cfg(debug_assertions)]
    pub is_inside: bool,
}

impl ParticleSource {
    /// Generate (x, y, z) coordinates.
    ///
    /// If use_exponential_distribution is true, the depth (z coordinate) is
    /// generated according to an exponential distribution with mean mean_source_depth.
    ///
    /// If use_exponential_distribution is false, the depth is fixed at z.
    pub fn generate_coordinates(&self, cfg: &Config) -> (f64, f64, f64) {
        if self.use_exponential_distribution {
            let mut rng = fastrand::Rng::new();
            let x = self.x;
            let y = self.y;
            loop {
                // Sample from exponential distribution
                let depth = -rng.f64().ln() * self.mean_source_depth;
                let z = cfg.thickness / 2.0 - depth;
                if z > -cfg.thickness / 2.0 {
                    return (x, y, z);
                }
            }
        } else {
            (self.x, self.y, self.z)
        }
    }

    /// Generate angles matching uniform distribution on the unit sphere.
    /// Return (phi, theta) in radians.
    pub fn generate_angles(&self) -> (f64, f64) {
        let mut rng = fastrand::Rng::new();

        let phi = 2.0 * PI * rng.f64();
        let theta = (2.0 * rng.f64() - 1.0).asin();

        (phi, theta)
    }
}

pub trait MinMax<T> {
    fn max(&self) -> T;
    fn min(&self) -> T;
}

impl MinMax<f64> for Vec<f64> {
    /// Return the maximum value in the vector, or f64::NEG_INFINITY if the vector is empty.
    fn max(&self) -> f64 {
        self.iter().fold(f64::NEG_INFINITY, |a, b| f64::max(a, *b))
    }
    /// Return the minimum value in the vector, or f64::INFINITY if the vector is empty.
    fn min(&self) -> f64 {
        self.iter().fold(f64::INFINITY, |a, b| f64::min(a, *b))
    }
}
