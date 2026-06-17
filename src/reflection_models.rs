//! Reflection models for phonon scattering at top and bottom surfaces.

use crate::data_structures::Vec3;
use crate::phonon::Phonon;
use crate::simulate::{SIN_TABLE_SIZE, SQRT_TABLE_SIZE, SinTable, SqrtTable};
use fastrand::Rng;
use std::f64::consts::FRAC_2_PI;

/// Part of sampling from the Phong model.
/// Sample new direction in the reference frame where the
/// ideal reflection direction is along the z-axis
#[inline(always)]
fn sample_phong_pdf(
    rng: &mut Rng,
    phong_exponent_sampling: f64,
    sin_table: &SinTable,
) -> (f64, f64, f64, f64) {
    let u1 = rng.f64_inclusive();
    let cos_theta = u1.powf(phong_exponent_sampling);
    let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

    let (sin_phi, cos_phi) = sample_sin_cos_uniform(rng, sin_table);
    (sin_theta, cos_theta, sin_phi, cos_phi)
}

/// Part of sampling from the Phong model.
/// Rotate sampled direction from reference frame where the
/// ideal reflection direction is along the z-axis to the
/// reference frame where the surface normal is along the z-axis
#[inline(always)]
fn sample_phong_rotate(
    sin_theta: f64,
    cos_theta: f64,
    sin_phi: f64,
    cos_phi: f64,
    pt: &Phonon,
) -> (f64, f64, f64) {
    // Direction vector in local coordinates (z along surface normal)
    let x_local = sin_theta * cos_phi;
    let y_local = sin_theta * sin_phi;
    let z_local = cos_theta;

    let x_ideal = pt.vx * pt.speed_inv;
    let y_ideal = pt.vy * pt.speed_inv;
    let z_ideal = pt.vz.abs() * pt.speed_inv; // ideal reflection

    // Rotate local direction vector to match ideal reflection direction
    // Rodrigues' rotation formula applied to axis = cross((0,0,1), ideal)
    // by angle cos_angle = dot((0,0,1), ideal) = z_ideal; further simplified
    // R is orthonormal rotation matrix
    let a = 1.0 / (1.0 + z_ideal);
    let r_11 = z_ideal + a * y_ideal * y_ideal;
    let r_12 = -a * x_ideal * y_ideal;
    let r_13 = x_ideal;
    let r_21 = r_12;
    let r_22 = z_ideal + a * x_ideal * x_ideal;
    let r_23 = y_ideal;
    let r_31 = -x_ideal;
    let r_32 = -y_ideal;
    let r_33 = z_ideal;
    let x = r_11 * x_local + r_12 * y_local + r_13 * z_local;
    let y = r_21 * x_local + r_22 * y_local + r_23 * z_local;
    let z = r_31 * x_local + r_32 * y_local + r_33 * z_local;
    (x, y, z)
}

/// Sample new direction from Phong distribution.
pub fn sample_phong_model(
    rng: &mut Rng,
    pt: &Phonon,
    phong_exponent_sampling: f64,
    sin_table: &SinTable,
) -> (f64, f64, f64) {
    // Until not rejected:
    // 1. Sample theta, phi
    // 2. Compute direction vector in local coordinates
    // 3. Rotate to match ideal reflection direction
    loop {
        // should finish because chance of sign(z)==sign(vz)
        // is very small for large phong exponents
        let (sin_theta, cos_theta, sin_phi, cos_phi) =
            sample_phong_pdf(rng, phong_exponent_sampling, sin_table);

        let (x, y, z) = sample_phong_rotate(sin_theta, cos_theta, sin_phi, cos_phi, pt);

        let z = if pt.vz > 0.0 { -z } else { z };
        if z.signum() != pt.vz.signum() {
            return (x, y, z);
        }
    }
}

#[inline(always)]
fn sample_phong_rescale(cos_theta: f64, pt: &Phonon) -> (f64, f64) {
    // compute theta_max between ideal reflection direction and surface plane
    let theta_max = (pt.vz.abs() * pt.speed_inv).asin().abs();
    // divide by theta_max to rescale, then no rejection sampling needed in principle
    // rejection may still occur due to numerical inaccuracies, but very unlikely
    let theta = cos_theta.acos();
    let theta_rescaled = theta * theta_max * FRAC_2_PI;
    let sin_theta_rescaled = theta_rescaled.sin();
    let cos_theta_rescaled = theta_rescaled.cos();
    (sin_theta_rescaled, cos_theta_rescaled)
}

/// Sample new direction from Phong distribution.
/// In the Phong distribution, the probability of scattering
/// to the wrong side of the surface is non-zero.
/// This functions rescales the distribution to make this probability zero.
pub fn sample_phong_model_rescaled(
    rng: &mut Rng,
    pt: &Phonon,
    phong_exponent_sampling: f64,
    sin_table: &SinTable,
) -> (f64, f64, f64) {
    loop {
        let (_, cos_theta, sin_phi, cos_phi) =
            sample_phong_pdf(rng, phong_exponent_sampling, sin_table);
        let (sin_theta_rescaled, cos_theta_rescaled) = sample_phong_rescale(cos_theta, pt);
        let (x, y, z) =
            sample_phong_rotate(sin_theta_rescaled, cos_theta_rescaled, sin_phi, cos_phi, pt);

        let z = if pt.vz > 0.0 { -z } else { z };
        if z.signum() != pt.vz.signum() {
            return (x, y, z);
        }
    }
}

/// Sample new direction from Lambertian distribution over hemisphere.
/// Return (x, y, z, r) with r = sqrt(x^2 + y^2) for convenience.
/// z is in range 0..1. x and y are in range -1..1.
/// (x, y, z) is unit vector.
pub fn cosine_sample_hemisphere(
    rng: &mut Rng,
    sin_table: &SinTable,
    sqrt_table: &SqrtTable,
) -> (f64, f64, f64, f64) {
    // https://www.pbr-book.org/3ed-2018/Monte_Carlo_Integration/2D_Sampling_with_Multidimensional_Transformations#Cosine-WeightedHemisphereSampling
    // See also: Malley's method

    // 1. uniformly sample disk
    // let r = rng.f64().sqrt();
    // bitwise AND for fast modulo
    const MASK: usize = SQRT_TABLE_SIZE - 1;
    let r2_index = (rng.u64(..) as usize) & MASK;
    let r = sqrt_table[r2_index];

    let (theta_sin, theta_cos) = sample_sin_cos_uniform(rng, sin_table);

    // 2. project up to hemisphere
    let x = r * theta_cos;
    let y = r * theta_sin;

    // let z = (1.0 - r * r).sqrt();
    let z_index = MASK ^ r2_index; // bitwise XOR for fast (1 - r^2)
    let z = sqrt_table[z_index];

    (x, y, z, r)
}

/// Sample new direction from uniform distribution over hemisphere.
/// Return (x, y, z, r) with r = sqrt(x^2 + y^2) for convenience.
/// z is in range 0..1. x and y are in range -1..1.
/// (x, y, z) is unit vector.
#[inline(always)]
pub fn uniform_sample_hemisphere(rng: &mut Rng, sin_table: &SinTable) -> (f64, f64, f64, f64) {
    // Based on https://www.pbr-book.org/3ed-2018/Monte_Carlo_Integration/2D_Sampling_with_Multidimensional_Transformations#UniformlySamplingaHemisphere

    let z = rng.f64_inclusive();
    let r = (1.0 - z * z).sqrt();

    let (phi_sin, phi_cos) = sample_sin_cos_uniform(rng, sin_table);

    let x = r * phi_cos;
    let y = r * phi_sin;
    (x, y, z, r)
}

/// Draw (sin(theta), cos(theta)) with theta distributed uniformly over 0..2pi.
#[inline(always)]
fn sample_sin_cos_uniform(rng: &mut Rng, sin_table: &SinTable) -> (f64, f64) {
    // Note that the other part of this is precomputed in the SinTable
    // If you change anything here, you have adapt the precomputation accordingly
    let u = rng.u64(..) as usize;

    // bitwise AND for fast modulo (ok because SIN_TABLE_SIZE is power of 2)
    const MASK: usize = SIN_TABLE_SIZE - 1;
    let idx = u & MASK;
    let idx_cos = idx ^ MASK; // = MASK - idx

    (sin_table[idx] as f64, sin_table[idx_cos] as f64)

    // Example to show that indexing like this works:
    //
    // u = rng.usize(..)
    //     -> random u64, e.g. ...11010011101101
    //
    // MASK = 0x1FFF
    // idx = 0x14ED = 5357
    // idx_cos = idx ^ MASK = 0xB12 = 2834
    //
    // s = sin_table[idx]
    //   = sin(2*pi* ( 5357 + 0.5) / 8192 + 0.25*pi  )
    //   = sin(2*pi* 5357.5 / 8192 + 0.25*pi  )
    //   = sin(0.25*pi + 2*pi * 5357.5/8192)
    //
    // c = sin_table[idx_cos]
    //   = sin(2*pi* (2834 + 0.5)/8192 + 0.25*pi)
    //   = cos(pi/2 - 2*pi* (2834 + 0.5)/8192 - 0.25*pi)
    //   = cos(0.25*pi - 2*pi * (2834 + 0.5)/8192)
    //   = cos(0.25*pi - 2*pi * (2834 + 0.5)/8192 + 2*pi)
    //   = cos(0.25*pi + 2*pi * (8192 - 2834 - 0.5)/8192)
    //   = cos(0.25*pi + 2*pi * 5357.5/8192)
    //
    // -> indexing sine table like retrieves sine,cosine of the same value, as intended
}

/// Rotate vector (x,y,z) from hemisphere defined by normal (0, 0, 1)
/// to hemisphere defined by normal (nx, ny, 0).
/// Assume normal is normalized to length 1.
#[inline(always)]
pub fn rotate_vector_to_normal_hemisphere(
    x: f64,
    y: f64,
    z: f64,
    nx: f64,
    ny: f64,
) -> (f64, f64, f64) {
    // Use the Rodrigues' rotation formula v_rot = k x v + (k . v)k
    // Rotate around axis k = cross((0,0,1), normal) by angle 90° (special case nz=0)

    // let n = Vec3 { x: nx, y: ny, z: 0.0 }; // not needed, k computed directly below
    let v = Vec3 { x, y, z };
    let k = Vec3 {
        x: -ny,
        y: nx,
        z: 0.0,
    };

    let v_rot = k.cross(&v) + k.dot(&v) * k;

    (v_rot.x, v_rot.y, v_rot.z)
}
