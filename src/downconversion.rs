use std::f64::consts::PI;

use fastrand::Rng;

use crate::data_structures::Vec3;
use crate::materials::{Branch, Si};
use crate::phonon::{Phonon, PhononState};

const H_PLANCK: f64 = 6.62607015e-34;
const H_PLANCK_INV: f64 = 1.0 / H_PLANCK;

#[inline(always)]
#[allow(non_snake_case)]
pub fn calculate_downconversion_rate(scattering_rate_1THz: f64, pt: &Phonon) -> f64 {
    let energy_THz = pt.energy * (H_PLANCK_INV * 1e-12);
    scattering_rate_1THz * f64::powi(energy_THz, 5)
}

pub fn downconversion(pt: &mut Phonon, rng: &mut Rng) -> (Phonon, Phonon) {
    pt.state = PhononState::Downconverted;
    if rng.f64() < Si::ANHARMONIC_TT_FRACTION {
        make_TT_secondaries(pt, rng)
    } else {
        make_LT_secondaries(pt, rng)
    }
}

#[allow(non_snake_case)]
fn make_LT_secondaries(pt: &mut Phonon, rng: &mut Rng) -> (Phonon, Phonon) {
    let x = sample_lt_energy_fraction(rng);
    let e1 = x * pt.energy;
    let e2 = pt.energy - e1;

    let theta_l = make_l_deviation_angle(x);
    let theta_t = make_t_deviation_angle(x);

    let v0 = Vec3 {
        x: pt.vx,
        y: pt.vy,
        z: pt.vz,
    };

    let axis = v0.get_orthogonal_vector().normalize();
    let phi = 2.0 * PI * rng.f64();

    let v1 = v0.rotate(&axis, theta_l).rotate(&v0.normalize(), phi);
    let v2 = v0.rotate(&axis, -theta_t).rotate(&v0.normalize(), phi);
    let v1_vz_abs_inv = if v1.z.abs() > 1e-12 {
        1.0 / v1.z.abs()
    } else {
        1e12
    };
    let v2_vz_abs_inv = if v2.z.abs() > 1e-12 {
        1.0 / v2.z.abs()
    } else {
        1e12
    };

    let branch1 = Branch::L;
    let branch2 = Si::get_random_transversal_branch(rng);

    let phonon1 = Phonon {
        t: pt.t,
        x: pt.x,
        y: pt.y,
        z: pt.z,
        vx: v1.x,
        vy: v1.y,
        vz: v1.z,
        vz_abs_inv: v1_vz_abs_inv,
        speed: pt.speed,
        speed_inv: pt.speed_inv,
        energy: e1,
        branch: branch1,
        state: PhononState::Alive,
        rng: rng.fork(),
    };
    let phonon2 = Phonon {
        t: pt.t,
        x: pt.x,
        y: pt.y,
        z: pt.z,
        vx: v2.x,
        vy: v2.y,
        vz: v2.z,
        vz_abs_inv: v2_vz_abs_inv,
        speed: pt.speed,
        speed_inv: pt.speed_inv,
        energy: e2,
        branch: branch2,
        state: PhononState::Alive,
        rng: rng.fork(),
    };
    (phonon1, phonon2)
}

#[allow(non_snake_case)]
fn make_TT_secondaries(pt: &mut Phonon, rng: &mut Rng) -> (Phonon, Phonon) {
    let x = sample_tt_energy_fraction(rng);
    let e1 = x * pt.energy;
    let e2 = pt.energy - e1;

    let theta1 = make_tt_deviation_angle(x);
    let theta2 = make_tt_deviation_angle(1.0 - x);

    let v0 = Vec3 {
        x: pt.vx,
        y: pt.vy,
        z: pt.vz,
    };

    let axis = v0.get_orthogonal_vector().normalize();
    let phi = 2.0 * PI * rng.f64();

    let v1 = v0.rotate(&axis, theta1).rotate(&v0.normalize(), phi);
    let v2 = v0.rotate(&axis, -theta2).rotate(&v0.normalize(), phi);
    let v1_vz_abs_inv = if v1.z.abs() > 1e-12 {
        1.0 / v1.z.abs()
    } else {
        1e12
    };
    let v2_vz_abs_inv = if v2.z.abs() > 1e-12 {
        1.0 / v2.z.abs()
    } else {
        1e12
    };

    let branch1 = Si::get_random_transversal_branch(rng);
    let branch2 = Si::get_random_transversal_branch(rng);

    let phonon1 = Phonon {
        t: pt.t,
        x: pt.x,
        y: pt.y,
        z: pt.z,
        vx: v1.x,
        vy: v1.y,
        vz: v1.z,
        vz_abs_inv: v1_vz_abs_inv,
        speed: pt.speed,
        speed_inv: pt.speed_inv,
        energy: e1,
        branch: branch1,
        state: PhononState::Alive,
        rng: rng.fork(),
    };
    let phonon2 = Phonon {
        t: pt.t,
        x: pt.x,
        y: pt.y,
        z: pt.z,
        vx: v2.x,
        vy: v2.y,
        vz: v2.z,
        vz_abs_inv: v2_vz_abs_inv,
        speed: pt.speed,
        speed_inv: pt.speed_inv,
        energy: e2,
        branch: branch2,
        state: PhononState::Alive,
        rng: rng.fork(),
    };
    (phonon1, phonon2)
}

const VL_OVER_VT: f64 = Si::SPEED_L / Si::SPEED_T;
#[allow(non_upper_case_globals)]
fn get_lt_decay_prob(x: f64) -> f64 {
    // Taken from G4CMP code, checked to be identical and < 2.8
    // I never checked these against Tamura1985, I felt too dumb
    let d = VL_OVER_VT;
    // x: fraction of energy in longitudinal mode, x=E_L'/E_L
    (1.0 / (x * x))
        * (1.0 - x * x)
        * (1.0 - x * x)
        * ((1.0 + x) * (1.0 + x) - d * d * ((1.0 - x) * (1.0 - x)))
        * (1.0 + x * x - d * d * ((1.0 - x) * (1.0 - x)))
        * (1.0 + x * x - d * d * ((1.0 - x) * (1.0 - x)))
}

#[allow(non_upper_case_globals)]
fn get_tt_decay_prob(x: f64) -> f64 {
    // Taken from G4CMP code, checked to be identical and < 0.8
    // I never checked these against Tamura1985, I felt too dumb
    const d: f64 = VL_OVER_VT;
    // dynamic constants from Tamura, PRL31, 1985
    const beta: f64 = Si::LATTICE_BETA / 1e11;
    const gamma: f64 = Si::LATTICE_GAMMA / 1e11;
    const lambda: f64 = Si::LATTICE_LAMBDA / 1e11;
    const mu: f64 = Si::LATTICE_MU / 1e11;

    const A: f64 = 0.5 * (1.0 - d * d) * (beta + lambda + (1.0 + d * d) * (gamma + mu));
    const B: f64 = beta + lambda + 2.0 * (gamma + mu) * d * d;
    const C: f64 = beta + lambda + 2.0 * (gamma + mu);
    const D: f64 = (1.0 - d * d) * (2.0 * beta + 4.0 * gamma + lambda + 3.0 * mu);

    f64::powi(A + B * d * x - B * x * x, 2)
        + f64::powi(
            C * x * (d - x) - D / (d - x) * (x - d - (1.0 - d * d) / (4.0 * x)),
            2,
        )
}

fn sample_lt_energy_fraction(rng: &mut Rng) -> f64 {
    // use rejection sampling to sample according to get_lt_decay_prob
    let upper_bound = 1.0;
    let lower_bound = (VL_OVER_VT - 1.0) / (VL_OVER_VT + 1.0);

    let limit = 2.8; // checked that get_lt_decay_prob < 2.8 for all x 
    loop {
        let u = rng.f64();
        let x = rng.f64() * (upper_bound - lower_bound) + lower_bound;
        let p = get_lt_decay_prob(x);
        if u * limit < p {
            return x;
        };
    }
}

#[allow(non_upper_case_globals)]
fn sample_tt_energy_fraction(rng: &mut Rng) -> f64 {
    // use rejection sampling to sample according to get_tt_decay_prob
    const upper_bound: f64 = (1.0 + (1.0 / VL_OVER_VT)) / 2.0;
    const lower_bound: f64 = (1.0 - (1.0 / VL_OVER_VT)) / 2.0;

    let limit = 0.8; // checked that get_tt_decay_prob < 0.8 for all x*d
    loop {
        let u = rng.f64();
        let x = rng.f64() * (upper_bound - lower_bound) + lower_bound;
        let p = get_tt_decay_prob(x * VL_OVER_VT);
        if u * limit < p {
            return x;
        };
    }
}

#[inline(always)]
fn make_l_deviation_angle(x: f64) -> f64 {
    // Taken from G4CMP code, checked to be identical
    let d = VL_OVER_VT;
    f64::acos((1.0 + x * x - d * d * ((1.0 - x) * (1.0 - x))) / (2.0 * x))
}

#[inline(always)]
fn make_t_deviation_angle(x: f64) -> f64 {
    // Taken from G4CMP code, checked to be identical
    let d = VL_OVER_VT;
    f64::acos((1.0 - x * x + d * d * ((1.0 - x) * (1.0 - x))) / (2.0 * d * (1.0 - x)))
}

#[inline(always)]
fn make_tt_deviation_angle(x: f64) -> f64 {
    // Taken from G4CMP code, checked to be identical
    let d = VL_OVER_VT;
    f64::acos((1.0 - d * d * ((1.0 - x) * (1.0 - x)) + d * d * (x * x)) / (2.0 * d * x))
}
