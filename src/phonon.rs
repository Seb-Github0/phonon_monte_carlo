//! Phonon, which is the particle being simulated.

use crate::config::Config;
use crate::materials::{Branch, Si};

/// Phonon particle.  
/// Has position (x,y,z) in m, velocity components (vx,vy,vz) in m/s.
/// Speed is magnitude of velocity in m/s.
/// Each phonon is spawned with energy fraction remaining = 1.0, which is reduced when hitting absorbers or bridges.
#[derive(Clone, Debug)]
pub struct Phonon {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub vx: f64,
    pub vy: f64,
    pub vz: f64,
    pub vz_abs_inv: f64,
    pub speed: f64,
    pub speed_inv: f64,
    pub energy_fraction_remaining: f64,
    pub rng: fastrand::Rng,
}

impl Phonon {
    pub fn new(material: &Si, cfg: &Config) -> Self {
        // Assign initial coordinates
        let source = &cfg.particle_source;
        let (x, y, mut z) = source.generate_coordinates(cfg);

        // Assign frequency
        let speed = material.default_speed;
        let energy_fraction_remaining = 1.0;

        // Assign initial angles
        let (phi, theta) = source.generate_angles();
        let vx = theta.cos() * phi.cos() * speed;
        let vy = theta.cos() * phi.sin() * speed;
        let mut vz = theta.sin() * speed;
        if cfg.is_two_dimensional_material {
            vz = 0.0;
            z = 0.0;
        }
        //let vxy_sq = vx * vx + vy * vy;
        //let vxy_sq_inv = if vxy_sq > 1e-12 { 1.0 / vxy_sq } else { 1e12 };
        let vz_abs_inv = if vz.abs() > 1e-12 {
            1.0 / vz.abs()
        } else {
            1e12
        };
        let speed_inv = 1.0 / speed;

        // Assign phonon branch
        let rng = fastrand::Rng::new();

        Phonon {
            x,
            y,
            z,
            vx,
            vy,
            vz,
            vz_abs_inv,
            speed,
            speed_inv,
            energy_fraction_remaining,
            rng,
        }
    }

    /// Randomly reassign the phonon branch and update the speed accordingly,
    /// while keeping the direction the same.
    #[inline(always)]
    pub fn assign_random_speed(&mut self) {
        let branch = self
            .rng
            .choice([Branch::LA, Branch::TA1, Branch::TA2])
            .unwrap();
        let speed_new = Si::get_speed(&branch);
        let speed_inv_new = Si::get_speed_inv(&branch);
        let factor = self.speed_inv * speed_new;
        let factor_inv = self.speed * speed_inv_new;

        self.speed = speed_new;
        self.speed_inv = speed_inv_new;
        self.vx *= factor;
        self.vy *= factor;
        self.vz *= factor;
        self.vz_abs_inv *= factor_inv;
    }
}
