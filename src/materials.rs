//! Properties of host material, currently only silicon implemented.

use fastrand::Rng;

/// Physical properties of silicon.
/// Contains the sound speeds for the LA and TA phonon branches in isotropic approximation.
#[derive(Clone, Debug)]
pub struct Si {
    pub default_speed: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Branch {
    L,
    FT,
    ST,
}

impl Si {
    pub const SPEED_L: f64 = 9131.0;
    pub const SPEED_FT: f64 = 5789.0;
    pub const SPEED_ST: f64 = 5302.0;
    const SPEED_L_INV: f64 = 1.0 / Self::SPEED_L;
    const SPEED_FT_INV: f64 = 1.0 / Self::SPEED_FT;
    const SPEED_ST_INV: f64 = 1.0 / Self::SPEED_ST;
    // LDOS 0.093 STDOS 0.531 FTDOS 0.376 (same as G4CMP paper)
    const DOS_FRACTION_L: f64 = 0.093;
    const DOS_FRACTION_ST: f64 = 0.531;
    const DOS_FRACTION_FT: f64 = 0.376;
    pub const ANHARMONIC_TT_FRACTION: f64 = 0.796;

    pub const ISOTOPE_SCATTERING_RATE_1THZ: f64 = 2.43e6; // s^-1
    pub const DOWNCONVERSION_RATE_1THZ: f64 = 1.23e5; // s^-1

    pub const LATTICE_BETA: f64 = -65.6e9; // Pa
    pub const LATTICE_GAMMA: f64 = -106.3e9; // Pa
    pub const LATTICE_LAMBDA: f64 = 52.4e9; // Pa
    pub const LATTICE_MU: f64 = 68e9; // Pa

    // pub const LATTICE_DELTA: f64 = ((Self::LATTICE_LAMBDA + 2.0 * Self::LATTICE_MU) / Self::LATTICE_MU).sqrt();
    // unfortunately .sqrt() is not a const function, so need to precompute manually:
    pub const LATTICE_DELTA: f64 = 1.6645;

    pub const fn new(default_speed: f64) -> Self {
        Self { default_speed }
    }

    #[inline(always)]
    pub const fn get_speed(branch: &Branch) -> f64 {
        match branch {
            Branch::L => Self::SPEED_L,
            Branch::FT => Self::SPEED_FT,
            Branch::ST => Self::SPEED_ST,
        }
    }

    #[inline(always)]
    pub const fn get_speed_inv(branch: &Branch) -> f64 {
        match branch {
            Branch::L => Self::SPEED_L_INV,
            Branch::FT => Self::SPEED_FT_INV,
            Branch::ST => Self::SPEED_ST_INV,
        }
    }

    #[inline(always)]
    pub fn get_random_branch(rng: &mut Rng) -> Branch {
        let x = rng.f64();
        if x < Si::DOS_FRACTION_L {
            Branch::L
        } else if x < Si::DOS_FRACTION_L + Si::DOS_FRACTION_ST {
            Branch::ST
        } else {
            Branch::FT
        }
    }

    #[inline(always)]
    pub fn get_random_transversal_branch(rng: &mut Rng) -> Branch {
        const FRACTION_ST: f64 = Si::DOS_FRACTION_ST / (Si::DOS_FRACTION_FT + Si::DOS_FRACTION_ST);
        if rng.f64() < FRACTION_ST {
            Branch::ST
        } else {
            Branch::FT
        }
    }
}
