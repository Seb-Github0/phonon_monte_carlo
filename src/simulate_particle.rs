//! Run a single particle simulation.

use crate::config::Config;
use crate::data_structures::{PointXYZ, ScatteringLocation, ScatteringPoint};
use crate::phonon::Phonon;
use crate::simulate::{EnergyResults, SimulationSetup};
use crate::top_bottom_scattering::{
    bottom_scattering, time_to_bottom, time_to_top, top_scattering,
};
use crate::wall_scattering::{EdgeIndex, PolygonIndex};
use fastrand::Rng;
use std::vec::Vec;

/// Run a single particle simulation, given the outside wall, absorbers, material, config, and precomputed tables.
pub fn simulate_particle(
    simulation_setup: &SimulationSetup,
    cfg: &Config,
) -> (Vec<ScatteringPoint>, Vec<EnergyResults>) {
    let wall = &simulation_setup.wall;
    let absorber = &simulation_setup.absorber;
    let clamps_top = &simulation_setup.clamps_top;
    let clamps_bottom = &simulation_setup.clamps_bottom;
    let material = &simulation_setup.material;
    let specularity_model = &simulation_setup.specularity_model;
    let specular_distribution = &simulation_setup.specular_distribution;
    let diffuse_distribution = &simulation_setup.diffuse_distribution;
    let sin_table = &simulation_setup.sin_table;
    let sqrt_table = &simulation_setup.sqrt_table;

    let mut scattering_points: Vec<ScatteringPoint> = Vec::new();

    let mut pt = Phonon::new(material, cfg);
    if cfg.use_branch_speed {
        pt.assign_random_speed();
    }

    let mut rng = Rng::new();

    let mut e_results = vec![
        EnergyResults {
            e_loss: 0.0,
            e_absorbed_total: 0.0,
        };
        cfg.number_of_timesteps
    ];

    let mut results = EnergyResults {
        e_loss: 0.0,
        e_absorbed_total: 0.0,
    };

    let time_total = cfg.time_total;
    let time_bin_size_inv = 1.0 / cfg.time_bin_size;
    let mut t = 0.0;
    while (t < time_total) && (pt.energy_fraction_remaining >= 1e-6) {
        // Get next scattering location
        let top_intersection = time_to_top(&pt, cfg);
        let bottom_intersection = time_to_bottom(&pt, cfg);
        let t_min = get_top_bottom_min_time(&top_intersection, &bottom_intersection);
        let wall_intersection = wall.time_to_wall(&pt, t_min);
        // might add internal scattering here later

        let (time_to_scatter, scattering_location, intersection_point) =
            select_next_event(&top_intersection, &bottom_intersection, &wall_intersection);

        // Update particle position
        t += time_to_scatter;
        pt.x = intersection_point.x;
        pt.y = intersection_point.y;
        pt.z = intersection_point.z;

        if cfg.write_scattering_points {
            let point = ScatteringPoint {
                x: pt.x,
                y: pt.y,
                z: pt.z,
                time: t,
                location: scattering_location.clone(),
                // Check that position that is about to be written is valid
                // This is expensive though, so only do it in debug mode
                #[cfg(debug_assertions)]
                is_inside: {
                    wall.is_inside(intersection_point.x, intersection_point.y, 1e-10)
                        && (intersection_point.z.abs() <= cfg.thickness * 0.5)
                },
            };
            scattering_points.push(point);
        }

        // Update particle velocity and energy
        match scattering_location {
            ScatteringLocation::Wall => {
                // unwrap ok because checked in get_next_event
                let (_, polygon_index, edge_index, _) = wall_intersection.unwrap();
                wall.wall_scattering(
                    &mut pt,
                    polygon_index,
                    edge_index,
                    &mut results,
                    &mut rng,
                    sin_table,
                );
            }
            ScatteringLocation::Top => {
                top_scattering(
                    &mut pt,
                    absorber,
                    clamps_top,
                    cfg,
                    &mut results,
                    specularity_model,
                    specular_distribution,
                    diffuse_distribution,
                    &mut rng,
                    sin_table,
                    sqrt_table,
                );
            }
            ScatteringLocation::Bottom => {
                bottom_scattering(
                    &mut pt,
                    clamps_bottom,
                    cfg,
                    &mut results,
                    specularity_model,
                    specular_distribution,
                    diffuse_distribution,
                    &mut rng,
                    sin_table,
                    sqrt_table,
                );
            }
        }
        if cfg.use_branch_speed {
            pt.assign_random_speed();
        }

        if t < time_total {
            let t_bin = (t * time_bin_size_inv).floor() as usize;
            e_results[t_bin].e_loss += results.e_loss;
            e_results[t_bin].e_absorbed_total += results.e_absorbed_total;

            results.e_loss = 0.0;
            results.e_absorbed_total = 0.0;
        }
    }

    (scattering_points, e_results)
}

/// Check which of the provided intersections happens first.
/// Returns the time to the next event, the location (top, bottom, wall), and the intersection point.
#[inline(always)]
#[allow(clippy::collapsible_if)]
fn select_next_event(
    top_intersection: &Option<(f64, PointXYZ)>,
    bottom_intersection: &Option<(f64, PointXYZ)>,
    wall_intersection: &Option<(f64, PolygonIndex, EdgeIndex, PointXYZ)>,
) -> (f64, ScatteringLocation, PointXYZ) {
    let mut min_time = f64::INFINITY;
    let mut location = None;
    let mut intersection_point = PointXYZ::new(0.0, 0.0, 0.0);

    if let Some((t, point)) = top_intersection {
        if *t < min_time {
            min_time = *t;
            location = Some(ScatteringLocation::Top);
            intersection_point = point.clone();
        }
    }

    if let Some((t, point)) = bottom_intersection {
        if *t < min_time {
            min_time = *t;
            location = Some(ScatteringLocation::Bottom);
            intersection_point = point.clone();
        }
    }

    if let Some((t, _, _, point)) = wall_intersection {
        if *t < min_time {
            min_time = *t;
            location = Some(ScatteringLocation::Wall);
            intersection_point = point.clone();
        }
    }

    if location.is_none() {
        panic!("No next scattering location found.");
    }

    (min_time, location.unwrap(), intersection_point)
}

/// Get the minimum time to either top or bottom intersection, if they exist.
#[inline(always)]
#[allow(clippy::collapsible_if)]
fn get_top_bottom_min_time(
    top_intersection: &Option<(f64, PointXYZ)>,
    bottom_intersection: &Option<(f64, PointXYZ)>,
) -> f64 {
    let mut min_time = f64::INFINITY;

    if let Some((t, _)) = top_intersection {
        if *t < min_time {
            min_time = *t;
        }
    }

    if let Some((t, _)) = bottom_intersection {
        if *t < min_time {
            min_time = *t;
        }
    }

    min_time
}
