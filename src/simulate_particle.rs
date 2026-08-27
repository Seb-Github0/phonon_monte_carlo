//! Run a single particle simulation.

use crate::config::Config;
use crate::data_structures::{PointXYZ, ScatteringEvent, ScatteringPoint};
use crate::downconversion::{calculate_downconversion_rate, downconversion};
use crate::isotope_scattering::{
    calculate_isotope_scattering_rate, isotope_scattering, sample_flight_time,
};
use crate::materials::{Branch, Si};
use crate::phonon::{Phonon, PhononState};
use crate::simulate::{EnergyResults, SimulationSetup};
use crate::top_bottom_scattering::{
    bottom_scattering, time_to_bottom, time_to_top, top_scattering,
};
use crate::wall_scattering::{EdgeIndex, PolygonIndex};

use fastrand::Rng;
use std::vec::Vec;

/// Simulate a single event.
/// The event can contain multiple particles if downconversion occurs.
pub fn simulate_event(
    simulation_setup: &SimulationSetup,
    cfg: &Config,
    event_number: usize,
) -> (Vec<ScatteringPoint>, Vec<EnergyResults>) {
    let mut rng = Rng::new();
    let mut scattering_points: Vec<ScatteringPoint> = Vec::new();
    let mut e_results = vec![
        EnergyResults {
            e_loss: 0.0,
            e_absorbed_total: 0.0,
        };
        cfg.number_of_timesteps
    ];
    let mut secondary_phonons: Option<(Phonon, Phonon)>;

    let mut particle_queue: Vec<Phonon> = Vec::with_capacity(10);
    let mut pt = Phonon::new(&simulation_setup.material, cfg);
    if cfg.use_branch_speed {
        pt.assign_random_speed();
    }
    particle_queue.push(pt);

    let mut track_number: usize = 0;

    while let Some(pt) = particle_queue.pop() {
        (scattering_points, e_results, secondary_phonons) = simulate_particle(
            simulation_setup,
            cfg,
            pt,
            scattering_points,
            e_results,
            &mut rng,
            event_number,
            track_number,
        );
        if let Some((phonon1, phonon2)) = secondary_phonons {
            particle_queue.push(phonon1);
            particle_queue.push(phonon2);
        }
        track_number += 1;
    }
    (scattering_points, e_results)
}

/// Run a single particle simulation, given the outside wall, absorbers, material, config, and precomputed tables.
pub fn simulate_particle(
    simulation_setup: &SimulationSetup,
    cfg: &Config,
    mut pt: Phonon,
    mut scattering_points: Vec<ScatteringPoint>,
    mut e_results: Vec<EnergyResults>,
    rng: &mut Rng,
    event_number: usize,
    track_number: usize,
) -> (
    Vec<ScatteringPoint>,
    Vec<EnergyResults>,
    Option<(Phonon, Phonon)>,
) {
    let wall = &simulation_setup.wall;
    let absorber = &simulation_setup.absorber;
    let clamps_top = &simulation_setup.clamps_top;
    let clamps_bottom = &simulation_setup.clamps_bottom;
    let top_scattering_cfg = &simulation_setup.top_scattering_settings;
    let bottom_scattering_cfg = &simulation_setup.bottom_scattering_settings;
    let sin_table = &simulation_setup.sin_table;
    let sqrt_table = &simulation_setup.sqrt_table;

    // Precompute scattering rates, ok because these only depend on pt.energy and that is fixed
    let isotope_scattering_rate =
        calculate_isotope_scattering_rate(Si::ISOTOPE_SCATTERING_RATE_1THZ, &pt);
    let downconversion_rate = calculate_downconversion_rate(Si::DOWNCONVERSION_RATE_1THZ, &pt);
    // Calculate inverses, to remove division in sampling later
    let isotope_scattering_mean_flight_time = 1.0 / isotope_scattering_rate;
    let downconversion_mean_flight_time = 1.0 / downconversion_rate;
    let mut secondary_phonons = None;

    // write start point
    if cfg.write_scattering_points {
        let point = ScatteringPoint {
            event_number,
            track_number,
            x: pt.x,
            y: pt.y,
            z: pt.z,
            time: pt.t,
            energy: pt.energy,
            event: ScatteringEvent::Start,
        };
        scattering_points.push(point);
    }

    let time_total = cfg.time_total;
    let time_bin_size_inv = 1.0 / cfg.time_bin_size;

    while (pt.t < time_total) && (pt.state == PhononState::Alive) {
        // Get time to scattering events
        let time_to_isotope_scattering =
            sample_flight_time(isotope_scattering_mean_flight_time, rng);
        // Downconversion only allowed for L branch
        let time_to_downconversion = if pt.branch == Branch::L {
            sample_flight_time(downconversion_mean_flight_time, rng)
        } else {
            f64::INFINITY
        };
        let top_intersection = time_to_top(&pt, cfg.thickness);
        let bottom_intersection = time_to_bottom(&pt, cfg.thickness);
        let t_min = get_min_time(
            &top_intersection,
            &bottom_intersection,
            time_to_isotope_scattering,
            time_to_downconversion,
        );
        let wall_intersection = wall.time_to_wall(&pt, t_min);

        // Select first upcoming event as scattering event
        let (time_to_scatter, scattering_location, intersection_point) = select_next_event(
            &top_intersection,
            &bottom_intersection,
            &wall_intersection,
            time_to_isotope_scattering,
            time_to_downconversion,
            &pt,
        );

        // Update particle position
        pt.t += time_to_scatter;
        pt.x = intersection_point.x;
        pt.y = intersection_point.y;
        pt.z = intersection_point.z;

        if cfg.write_scattering_points {
            let point = ScatteringPoint {
                event_number,
                track_number,
                x: pt.x,
                y: pt.y,
                z: pt.z,
                time: pt.t,
                energy: pt.energy,
                event: scattering_location.clone(),
            };
            scattering_points.push(point);
        }

        // Update particle velocity and energy
        match scattering_location {
            ScatteringEvent::BoundaryWall => {
                // unwrap ok because checked in get_next_event
                let (_, polygon_index, edge_index, _) = wall_intersection.unwrap();
                wall.wall_scattering(&mut pt, polygon_index, edge_index, rng, sin_table);
            }
            ScatteringEvent::BoundaryTop => {
                top_scattering(
                    &mut pt,
                    absorber,
                    clamps_top,
                    cfg,
                    top_scattering_cfg,
                    rng,
                    sin_table,
                    sqrt_table,
                );
            }
            ScatteringEvent::BoundaryBottom => {
                bottom_scattering(
                    &mut pt,
                    clamps_bottom,
                    cfg,
                    bottom_scattering_cfg,
                    rng,
                    sin_table,
                    sqrt_table,
                );
            }
            ScatteringEvent::IsotopeScattering => isotope_scattering(&mut pt, rng, sin_table),
            ScatteringEvent::Downconversion => {
                secondary_phonons = Some(downconversion(&mut pt, rng));
            }
            ScatteringEvent::Start => {}
        }
        if cfg.use_branch_speed {
            pt.assign_random_speed();
        }
    }
    if pt.t < time_total {
        let t_bin = (pt.t * time_bin_size_inv).floor() as usize;
        match pt.state {
            PhononState::Alive => {} // particle still running at sim. end - don't care
            PhononState::Absorbed => e_results[t_bin].e_absorbed_total += pt.energy,
            PhononState::Lost => e_results[t_bin].e_loss += pt.energy,
            PhononState::Downconverted => {} // energy carried instead by secondary phonons
        }
    }

    (scattering_points, e_results, secondary_phonons)
}

/// Check which of the provided intersections happens first.
/// Returns the time to the next event, the location (top, bottom, wall), and the intersection point.
#[inline(always)]
#[allow(clippy::collapsible_if)]
fn select_next_event(
    top_intersection: &Option<(f64, PointXYZ)>,
    bottom_intersection: &Option<(f64, PointXYZ)>,
    wall_intersection: &Option<(f64, PolygonIndex, EdgeIndex, PointXYZ)>,
    time_to_isotope_scattering: f64,
    time_to_downconversion: f64,
    pt: &Phonon,
) -> (f64, ScatteringEvent, PointXYZ) {
    let mut min_time = f64::INFINITY;
    let mut location = None;
    let mut intersection_point = PointXYZ::new(0.0, 0.0, 0.0);

    if let Some((t, point)) = top_intersection {
        if *t < min_time {
            min_time = *t;
            location = Some(ScatteringEvent::BoundaryTop);
            intersection_point = point.clone();
        }
    }

    if let Some((t, point)) = bottom_intersection {
        if *t < min_time {
            min_time = *t;
            location = Some(ScatteringEvent::BoundaryBottom);
            intersection_point = point.clone();
        }
    }

    if time_to_isotope_scattering < min_time {
        min_time = time_to_isotope_scattering;
        location = Some(ScatteringEvent::IsotopeScattering);
        intersection_point = PointXYZ::new(
            pt.x + pt.vx * min_time,
            pt.y + pt.vy * min_time,
            pt.z + pt.vz * min_time,
        )
    }

    if time_to_downconversion < min_time {
        min_time = time_to_downconversion;
        location = Some(ScatteringEvent::Downconversion);
        intersection_point = PointXYZ::new(
            pt.x + pt.vx * min_time,
            pt.y + pt.vy * min_time,
            pt.z + pt.vz * min_time,
        )
    }

    if let Some((t, _, _, point)) = wall_intersection {
        if *t < min_time {
            min_time = *t;
            location = Some(ScatteringEvent::BoundaryWall);
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
fn get_min_time(
    top_intersection: &Option<(f64, PointXYZ)>,
    bottom_intersection: &Option<(f64, PointXYZ)>,
    time_to_isotope_scattering: f64,
    time_to_downconversion: f64,
) -> f64 {
    let mut min_time = f64::min(time_to_isotope_scattering, time_to_downconversion);

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
