//! Data structures for 2D and 3D points.

use serde::{Deserialize, Serialize};

/// Point with x and y coordinates.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PointXY {
    pub x: f64,
    pub y: f64,
}

// implement element-wise addition and subtraction for PointXY
impl std::ops::Add for &PointXY {
    type Output = PointXY;
    #[inline(always)]
    fn add(self, other: &PointXY) -> PointXY {
        PointXY {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl std::ops::Sub for &PointXY {
    type Output = PointXY;
    #[inline(always)]
    fn sub(self, other: &PointXY) -> PointXY {
        PointXY {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

// unary negation
impl std::ops::Neg for &PointXY {
    type Output = PointXY;
    #[inline(always)]
    fn neg(self) -> PointXY {
        PointXY {
            x: -self.x,
            y: -self.y,
        }
    }
}

impl std::ops::Neg for PointXY {
    type Output = PointXY;
    #[inline(always)]
    fn neg(self) -> PointXY {
        PointXY {
            x: -self.x,
            y: -self.y,
        }
    }
}

impl PointXY {
    /// Rotate the point 90 degrees counter-clockwise around the origin.
    #[inline(always)]
    pub fn rotate_left_90(&self) -> PointXY {
        PointXY {
            x: -self.y,
            y: self.x,
        }
    }

    /// Compute the dot product with another PointXY.
    #[inline(always)]
    pub fn dot(&self, other: &PointXY) -> f64 {
        self.x * other.x + self.y * other.y
    }
}

/// Point with x, y, and z coordinates.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PointXYZ {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl PointXYZ {
    #[inline(always)]
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
}
