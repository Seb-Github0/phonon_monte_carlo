//! Phonon, which is the particle being simulated.

use fastrand::Rng;

use crate::config::Config;
use crate::materials::{Branch, Si};

/// Phonon particle.  
/// Has position (x,y,z) in m, velocity components (vx,vy,vz) in m/s.
/// Speed is magnitude of velocity in m/s.
/// Each phonon is spawned with energy fraction remaining = 1.0, which is reduced when hitting absorbers or bridges.
#[derive(Clone, Debug)]
pub struct Phonon {
    pub t: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub vx: f64,
    pub vy: f64,
    pub vz: f64,
    pub vz_abs_inv: f64,
    pub speed: f64,
    pub speed_inv: f64,
    pub energy: f64,
    pub branch: Branch,
    pub state: PhononState,
    pub rng: fastrand::Rng,
}

#[derive(PartialEq, Clone, Debug)]
pub enum PhononState {
    Alive,
    Absorbed,
    Lost,
    Downconverted,
}

impl Phonon {
    pub fn new(material: &Si, cfg: &Config) -> Self {
        // Assign initial coordinates
        let source = &cfg.particle_source;
        let (x, y, mut z) = source.generate_coordinates(cfg);

        // Assign phonon branch
        let mut rng = Rng::new();

        // Assign frequency
        let speed = material.default_speed;
        let branch = Si::get_random_branch(&mut rng);
        let energy = cfg.initial_phonon_energy;
        let state = PhononState::Alive;

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

        Phonon {
            t: 0.0,
            x,
            y,
            z,
            vx,
            vy,
            vz,
            vz_abs_inv,
            speed,
            speed_inv,
            energy,
            branch,
            state,
            rng,
        }
    }

    /// Randomly reassign the phonon branch and update the speed accordingly,
    /// while keeping the direction the same.
    #[inline(always)]
    pub fn assign_random_speed(&mut self) {
        let branch = Si::get_random_branch(&mut self.rng);
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
