//! 3D Vector implementation for notation convenience.
use std::ops::{Add, Mul};

#[derive(Clone)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

pub struct NormalizedVec3 {
    x: f64,
    y: f64,
    z: f64,
}

impl NormalizedVec3 {
    #[inline(always)]
    fn from_vec3(v: Vec3) -> Self {
        NormalizedVec3 {
            x: v.x,
            y: v.y,
            z: v.z,
        }
    }

    #[inline(always)]
    pub fn as_vec3(&self) -> Vec3 {
        Vec3 {
            x: self.x,
            y: self.y,
            z: self.z,
        }
    }
}

impl Vec3 {
    #[inline(always)]
    pub fn dot(&self, other: &Vec3) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    #[inline(always)]
    pub fn cross(&self, other: &Vec3) -> Vec3 {
        Vec3 {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    /// Rotate the vector about the axis by an angle theta.
    /// Assumes that axis is a normalized vector
    #[inline(always)]
    pub fn rotate(&self, axis: &NormalizedVec3, theta: f64) -> Vec3 {
        let (s, c) = theta.sin_cos();
        let v = self;
        let axis = axis.as_vec3();
        v * c + axis.cross(v) * s + &axis * (axis.dot(v) * (1.0 - c))
    }

    /// Deterministic vector orthogonal to `v`,
    /// picks the construction based on which component is smallest, to avoid
    /// near-zero cross products.
    #[inline(always)]
    pub fn get_orthogonal_vector(&self) -> Vec3 {
        let v = self;
        let (x, y, z) = (v.x.abs(), v.y.abs(), v.z.abs());
        if x < y {
            if x < z {
                Vec3 {
                    x: 0.0,
                    y: v.z,
                    z: -v.y,
                } // x smallest
            } else {
                Vec3 {
                    x: v.y,
                    y: -v.x,
                    z: 0.0,
                } // z smallest
            }
        } else {
            if y < z {
                Vec3 {
                    x: -v.z,
                    y: 0.0,
                    z: v.x,
                } // y smallest
            } else {
                Vec3 {
                    x: v.y,
                    y: -v.x,
                    z: 0.0,
                } // z smallest
            }
        }
    }

    #[inline(always)]
    pub fn normalize(&self) -> NormalizedVec3 {
        let v = self;
        let inv_len = 1.0 / v.dot(v).sqrt();
        NormalizedVec3::from_vec3(v * inv_len)
    }
}

// implement vector addition
impl Add for Vec3 {
    type Output = Vec3;

    #[inline(always)]
    fn add(self, rhs: Vec3) -> Vec3 {
        Vec3 {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

// implement multiplication by scalar using * operator, also from LHS
impl Mul<f64> for Vec3 {
    type Output = Vec3;

    #[inline(always)]
    fn mul(self, rhs: f64) -> Vec3 {
        Vec3 {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl Mul<f64> for &Vec3 {
    type Output = Vec3;

    #[inline(always)]
    fn mul(self, rhs: f64) -> Vec3 {
        Vec3 {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl Mul<Vec3> for f64 {
    type Output = Vec3;

    #[inline(always)]
    fn mul(self, rhs: Vec3) -> Vec3 {
        Vec3 {
            x: self * rhs.x,
            y: self * rhs.y,
            z: self * rhs.z,
        }
    }
}
