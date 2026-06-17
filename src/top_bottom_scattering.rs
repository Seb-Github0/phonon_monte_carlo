//! Scattering at top and bottom surfaces, specular/diffuse. Absorption into absorber regions.

use crate::config::Config;
use crate::data_structures::{
    DiffuseDistribution, MinMax, PointXY, PointXYZ, Rectangle, SpecularDistribution,
    SpecularityModel,
};
use crate::phonon::Phonon;
use crate::reflection_models::{
    cosine_sample_hemisphere, sample_phong_model, sample_phong_model_rescaled,
    uniform_sample_hemisphere,
};
use crate::simulate::{EnergyResults, SinTable, SqrtTable};
use fastrand::Rng;

/// Common function for top and bottom surface scattering.
/// With probability `p_specular`, does specular reflection, otherwise diffuse reflection.
/// For diffuse reflection, uses the distribution specified in the config file.
fn in_plane_surface_scattering(
    pt: &mut Phonon,
    cfg: &Config,
    specularity_model: &SpecularityModel,
    specular_distribution: &SpecularDistribution,
    diffuse_distribution: &DiffuseDistribution,
    rng: &mut Rng,
    sin_table: &SinTable,
    sqrt_table: &SqrtTable,
) {
    let p_specular = match specularity_model {
        SpecularityModel::Constant => cfg.p_specular,
        SpecularityModel::Soffer => {
            let cos_theta = pt.vz * pt.speed_inv;
            let c = cfg.specularity_roughness_prefactor;
            (-c * cos_theta * cos_theta).exp()
        }
    };

    if rng.f64() < p_specular {
        match specular_distribution {
            SpecularDistribution::Ideal => {
                pt.vz = -pt.vz;
            }
            SpecularDistribution::Phong => {
                let (x, y, z) = sample_phong_model(rng, pt, cfg.phong_exponent_sampling, sin_table);
                pt.vx = x * pt.speed;
                pt.vy = y * pt.speed;
                pt.vz = z * pt.speed;
                pt.vz_abs_inv = if pt.vz.abs() > 1e-12 {
                    1.0 / pt.vz.abs()
                } else {
                    1e12
                };
            }
            SpecularDistribution::PhongRescaled => {
                let (x, y, z) =
                    sample_phong_model_rescaled(rng, pt, cfg.phong_exponent_sampling, sin_table);
                pt.vx = x * pt.speed;
                pt.vy = y * pt.speed;
                pt.vz = z * pt.speed;
                pt.vz_abs_inv = if pt.vz.abs() > 1e-12 {
                    1.0 / pt.vz.abs()
                } else {
                    1e12
                };
            }
        }
    } else {
        let (x, y, z, _) = match diffuse_distribution {
            DiffuseDistribution::Lambertian => cosine_sample_hemisphere(rng, sin_table, sqrt_table),
            DiffuseDistribution::Uniform => uniform_sample_hemisphere(rng, sin_table),
        };

        pt.vx = x * pt.speed;
        pt.vy = y * pt.speed;
        let vz_abs = z * pt.speed;

        pt.vz = -pt.vz.signum() * vz_abs;
        pt.vz_abs_inv = if vz_abs > 1e-12 { 1.0 / vz_abs } else { 1e12 };
    }
}

// I left this in for possible later implementation. Cannot be used as is, though.
// fn internal_scattering(pt: &mut Phonon, flight: &Flight, scattering_types: &mut ScatteringTypes) {
//     if flight.time_since_previous_scattering >= pt.time_of_internal_scattering {
//         scattering_types.internal = random_scattering(pt);
//     }
// }

/// Calculate time until intersection with top surface and the corresponding intersection point.
/// Returns None if no intersection (i.e., vz <= 0).
#[inline(always)]
pub fn time_to_top(pt: &Phonon, cfg: &Config) -> Option<(f64, PointXYZ)> {
    if pt.vz > 0.0 {
        let time_to_intersection = (cfg.thickness * 0.5 - pt.z) * pt.vz_abs_inv;
        let intersection_point = PointXYZ {
            x: pt.x + pt.vx * time_to_intersection,
            y: pt.y + pt.vy * time_to_intersection,
            z: cfg.thickness * 0.5,
        };
        Some((time_to_intersection, intersection_point))
    } else {
        None
    }
}

/// Scattering at the top surface, including absorption into absorber regions and clamps.
pub fn top_scattering(
    pt: &mut Phonon,
    absorber_region: &AbsorberRegion,
    clamps_top: &[AbsorberPolygon],
    cfg: &Config,
    results: &mut EnergyResults,
    specularity_model: &SpecularityModel,
    specular_distribution: &SpecularDistribution,
    diffuse_distribution: &DiffuseDistribution,
    rng: &mut Rng,
    sin_table: &SinTable,
    sqrt_table: &SqrtTable,
) {
    // get idx of absorber region, if any, that the particle is in
    let is_inside_absorber = absorber_region.is_inside(pt.x, pt.y);

    if is_inside_absorber {
        results.e_absorbed_total = pt.energy_fraction_remaining * cfg.absorptivity;
        pt.energy_fraction_remaining *= 1.0 - cfg.absorptivity;
    }

    if cfg.include_clamps {
        let is_inside_clamp = clamps_top
            .iter()
            .any(|clamp_polygon| clamp_polygon.is_inside(pt.x, pt.y));

        if is_inside_clamp {
            results.e_loss = pt.energy_fraction_remaining * cfg.clamps_absorptivity;
            pt.energy_fraction_remaining *= 1.0 - cfg.clamps_absorptivity;
        }
    }

    in_plane_surface_scattering(
        pt,
        cfg,
        specularity_model,
        specular_distribution,
        diffuse_distribution,
        rng,
        sin_table,
        sqrt_table,
    );
}

/// Calculate time until intersection with bottom surface
/// and the corresponding intersection point.
/// Returns None if no intersection (i.e., vz >= 0).
#[inline(always)]
pub fn time_to_bottom(pt: &Phonon, cfg: &Config) -> Option<(f64, PointXYZ)> {
    if pt.vz < 0.0 {
        let time_to_intersection = (cfg.thickness * 0.5 + pt.z) * pt.vz_abs_inv;
        let intersection_point = PointXYZ {
            x: pt.x + pt.vx * time_to_intersection,
            y: pt.y + pt.vy * time_to_intersection,
            z: -cfg.thickness * 0.5,
        };
        Some((time_to_intersection, intersection_point))
    } else {
        None
    }
}

/// Scattering at the bottom surface.
#[inline(always)]
pub fn bottom_scattering(
    pt: &mut Phonon,
    clamps_bottom: &[AbsorberPolygon],
    cfg: &Config,
    results: &mut EnergyResults,
    specularity_model: &SpecularityModel,
    specular_distribution: &SpecularDistribution,
    diffuse_distribution: &DiffuseDistribution,
    rng: &mut Rng,
    sin_table: &SinTable,
    sqrt_table: &SqrtTable,
) {
    if cfg.include_clamps {
        let is_inside_clamp = clamps_bottom
            .iter()
            .any(|clamp_polygon| clamp_polygon.is_inside(pt.x, pt.y));
        if is_inside_clamp {
            results.e_loss = pt.energy_fraction_remaining * cfg.clamps_absorptivity;
            pt.energy_fraction_remaining *= 1.0 - cfg.clamps_absorptivity;
        }
    }

    in_plane_surface_scattering(
        pt,
        cfg,
        specularity_model,
        specular_distribution,
        diffuse_distribution,
        rng,
        sin_table,
        sqrt_table,
    );
}

/// Represents one or multiple convex, polygonal absorber regions on the top surface.
pub struct AbsorberRegion {
    polygons: Vec<AbsorberPolygon>,
}

/// Represents a convex, polygonal absorber region on the top surface.
pub struct AbsorberPolygon {
    edge_normals: Vec<PointXY>,
    offsets: Vec<f64>,
    outside_rectangle: Rectangle,
    n: usize,
}

impl AbsorberRegion {
    pub fn new(polygons: Vec<AbsorberPolygon>) -> Self {
        Self { polygons }
    }

    #[inline(always)]
    pub fn is_inside(&self, x: f64, y: f64) -> bool {
        self.polygons.iter().any(|polygon| polygon.is_inside(x, y))
    }
}

impl AbsorberPolygon {
    /// Creates a new PolygonAbsorber from the given points.
    /// Assumes the points define a convex polygon. Does not support arc segments.
    pub fn new(points: &[PointXY]) -> Self {
        let mut points = points.to_vec();

        // If points are in clockwise order, reverse to make them counter-clockwise.
        // Use Euler's shoelace formula to compute signed area and thus orientation of polygon
        let signed_area: f64 = points
            .iter()
            .zip(points.iter().cycle().skip(1))
            .map(|(p1, p2)| p1.x * p2.y - p2.x * p1.y)
            .sum();
        if signed_area < 0.0 {
            points.reverse();
        }
        // Can assume from here that points are in counter-clockwise order

        let n = points.len();
        let mut edges = Vec::with_capacity(n);
        let mut edge_normals = Vec::with_capacity(n);
        let mut offsets = Vec::with_capacity(n);

        for i in 0..n {
            let next = (i + 1) % n;
            let edge = PointXY {
                x: points[next].x - points[i].x,
                y: points[next].y - points[i].y,
            };
            edges.push(edge.clone());

            let mut normal = PointXY {
                x: edge.y,
                y: -edge.x,
            };
            let len = (normal.x * normal.x + normal.y * normal.y).sqrt();
            if len > 0.0 {
                normal.x /= len;
                normal.y /= len;
            }
            offsets.push(normal.x * points[i].x + normal.y * points[i].y);
            edge_normals.push(normal);
        }

        let x_points: Vec<f64> = points.iter().map(|p| p.x).collect();
        let y_points: Vec<f64> = points.iter().map(|p| p.y).collect();
        let outside_rectangle = Rectangle {
            x_min: x_points.min(),
            x_max: x_points.max(),
            y_min: y_points.min(),
            y_max: y_points.max(),
        };

        Self {
            edge_normals,
            offsets,
            outside_rectangle,
            n,
        }
    }

    /// Return true if (x, y) is strictly outside the polygon.
    pub fn is_outside(&self, x: f64, y: f64) -> bool {
        // Simplified a lot by assuming convex polygons stated in counter-clockwise order.
        // check against bounding rectangle first
        if x < self.outside_rectangle.x_min
            || x > self.outside_rectangle.x_max
            || y < self.outside_rectangle.y_min
            || y > self.outside_rectangle.y_max
        {
            return true;
        }

        let mut outside = 0u8;
        for i in 0..self.n {
            let n = &self.edge_normals[i];
            let dot = n.x * x + n.y * y - self.offsets[i];
            // point lies outside at this edge
            // branchless version, equivalent to: if n.x * x + n.y * y > offset {return true; }
            outside |= (dot > 0.0) as u8;
        }
        outside != 0
    }

    /// Return true if (x, y) is inside or on the edge of the polygon.
    #[inline(always)]
    pub fn is_inside(&self, x: f64, y: f64) -> bool {
        !self.is_outside(x, y)
    }
}
