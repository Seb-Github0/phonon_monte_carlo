//! Properties of host material, currently only silicon implemented.

/// Physical properties of silicon.
/// Contains the sound speeds for the LA and TA phonon branches in isotropic approximation.
#[derive(Clone, Debug)]
pub struct Si {
    pub default_speed: f64,
}

#[derive(Clone, Debug)]
pub enum Branch {
    LA,
    TA1,
    TA2,
}

impl Si {
    const SPEED_LA: f64 = 8433.0;
    const SPEED_TA: f64 = 5843.0;
    const SPEED_LA_INV: f64 = 1.0 / Self::SPEED_LA;
    const SPEED_TA_INV: f64 = 1.0 / Self::SPEED_TA;

    pub const fn new(default_speed: f64) -> Self {
        Self { default_speed }
    }

    #[inline(always)]
    pub const fn get_speed(branch: &Branch) -> f64 {
        match branch {
            Branch::LA => Self::SPEED_LA,
            Branch::TA1 => Self::SPEED_TA,
            Branch::TA2 => Self::SPEED_TA,
        }
    }

    #[inline(always)]
    pub const fn get_speed_inv(branch: &Branch) -> f64 {
        match branch {
            Branch::LA => Self::SPEED_LA_INV,
            Branch::TA1 => Self::SPEED_TA_INV,
            Branch::TA2 => Self::SPEED_TA_INV,
        }
    }
}
