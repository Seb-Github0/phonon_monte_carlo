use crate::materials::Si;
use crate::phonon::Phonon;
use crate::reflection_models::uniform_sample_sphere;
use crate::simulate::SinTable;
use fastrand::Rng;

const H_PLANCK: f64 = 6.62607015e-34;
const H_PLANCK_INV: f64 = 1.0 / H_PLANCK;

#[inline(always)]
pub fn sample_flight_time(mean_flight_time: f64, rng: &mut Rng) -> f64 {
    // Sample from exponential distribution
    // Inverse-CDF sampling: for mean flight time tau, P(t) = 1 - exp(-t/tau), so
    // t = -ln(1 - u) * tau. Since u ~ U[0,1) implies (1-u) ~ U(0,1],
    // use -ln(1-u) so we never take ln(0).
    let u: f64 = rng.f64();
    -(1.0 - u).ln() * mean_flight_time
}

#[inline(always)]
#[allow(non_snake_case)]
pub fn calculate_isotope_scattering_rate(scattering_rate_1THz: f64, pt: &Phonon) -> f64 {
    let energy_THz = pt.energy * (H_PLANCK_INV * 1e-12);
    scattering_rate_1THz * f64::powi(energy_THz, 4)
}

pub fn isotope_scattering(pt: &mut Phonon, rng: &mut Rng, sin_table: &SinTable) {
    pt.branch = Si::get_random_branch(rng);

    // Randomize direction
    let (x, y, z, _) = uniform_sample_sphere(rng, sin_table);
    pt.vx = x * pt.speed;
    pt.vy = y * pt.speed;
    pt.vz = z * pt.speed;
    pt.vz_abs_inv = if pt.vz.abs() > 1e-12 {
        1.0 / pt.vz.abs()
    } else {
        1e12
    };
}
