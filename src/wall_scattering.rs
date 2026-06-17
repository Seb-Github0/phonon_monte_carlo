//! Scattering at outer wall polygon.

use core::f64;

use crate::data_structures::{PointXY, PointXYZ, WallConfig};
use crate::phonon::Phonon;
use crate::reflection_models::{rotate_vector_to_normal_hemisphere, uniform_sample_hemisphere};
use crate::simulate::{EnergyResults, SinTable};
use fastrand::Rng;

const TOL: f64 = 1e-20;

/// Calculate center of circle of radius r passing through points (x1,y1) and (x2,y2).
/// Assumes that (x1,y1) and (x2,y2) are specified in counter-clockwise order
/// around the inside of the polygon.
/// Panics if radius is too small to create circle between the points.
fn circle_center(x1: f64, y1: f64, x2: f64, y2: f64, r: f64) -> PointXY {
    let dx = x2 - x1;
    let dy = y2 - y1;
    let length_sqr = dx * dx + dy * dy;
    let r_sqr = r * r;
    if length_sqr >= 4.0 * r_sqr {
        panic!(
            "Circle radius {r} too small to create circle
         between points ({},{}),({},{})",
            x1, y1, x2, y2
        );
    } else {
        let t = (r_sqr / length_sqr - 0.25).sqrt();
        let mid_x = 0.5 * (x1 + x2);
        let mid_y = 0.5 * (y1 + y2);
        let ox = mid_x - t * dy;
        let oy = mid_y + t * dx;
        PointXY { x: ox, y: oy }
    }
}

/// Check if circle segment between x1 and x2 with center c keeps convexity of polygon.
fn does_circle_segment_keep_convexity(
    x1: &PointXY,
    x2: &PointXY,
    c: &PointXY,
    edge_before: &PointXY,
    edge_after: &PointXY,
) -> bool {
    // Calculate radius vector from center to points,
    // rotate by 90° to get tangent direction,
    // take cross product with next/previous edge
    // (last two steps can be simplified to dot product)
    let tolerance = 1e-6;
    let bool1 = edge_after.dot(&(c - x2)) >= -tolerance;
    let bool2 = edge_before.dot(&(c - x1)) <= tolerance;
    bool1 && bool2
}

/// Check if a point is on an arc segment, assuming the point
/// is already known to be on the circle (with given center).
/// a and b are the endpoints of the circle segment,
/// specified in counter-clockwise order.
#[inline(always)]
fn is_on_circle_segment(point: &PointXY, center: &PointXY, a: &PointXY, b: &PointXY) -> bool {
    let tangent_a = (a - center).rotate_left_90();
    let tangent_b = (b - center).rotate_left_90();

    let tol = 1e-14;
    let cond1 = tangent_a.dot(&(point - a)) >= -tol;
    let cond2 = tangent_b.dot(&(point - b)) <= tol;
    cond1 && cond2
}

/// Computes time until intersection of particle trajectory with line segment.
/// The line segment is specified by its normal vector and an offset.
/// Point is on line if: (line equation) normal.x * x + normal.y * y = offset
/// Returns None if no intersection (i.e., parallel or moving away from line).
/// Does not check if intersection point is within a certain segment of the line.
#[inline(always)]
fn intersect_particle_with_line(
    pt: &Phonon,
    normal: &PointXY,
    offset: f64,
    t_min: f64,
) -> Option<f64> {
    // Particle trajectory: (x, y, z) + t*(vx, vy, vz)
    // Insert and solve for t: normal.x * (x + t*vx) + normal.y * (y + t*vy) = offset
    // => t = (offset - normal.x * x - normal.y * y) / (normal.x * vx + normal.y * vy)
    let denom = normal.x * pt.vx + normal.y * pt.vy;
    let numerator = offset - normal.x * pt.x - normal.y * pt.y;

    // Try avoiding (expensive) division as much as possible using beforehand checks
    // let check1 = denom <= 0.0;
    // denom == 0.0: Parallel,  denom < 0.0: Moving away from line
    // This check is already ensured by the following checks if t_min > TOL,
    // which we assert here;

    // numerator <= 0.0: intersection is behind particle
    // or numerator small: t very close to 0, likely numerical issues
    let check2 = numerator <= TOL * denom;

    // compare with t_min before doing expensive division
    // t = numerator / denom > t_min  <=>  numerator > denom * t
    let check3 = numerator >= denom * t_min;

    if check2 || check3 {
        return None;
    }

    let t = numerator / denom;

    Some(t)
}

#[inline(always)]
/// Check if a <= x <= b or b <= x <= a, within some tolerance
fn is_between(x: f64, a: f64, b: f64) -> bool {
    const EPS: f64 = 1e-11;
    if a <= b {
        if x < a - EPS || x > b + EPS {
            return false;
        }
    } else if x < b - EPS || x > a + EPS {
        return false;
    }
    true
}

#[inline(always)]
/// Computes intersection of particle trajectory with line segment defined by endpoints a and b.
/// The line containing the segment is specified by its normal vector and offset.
/// Returns None if no intersection (i.e., parallel, moving away from line, or intersection outside segment).
/// First computes intersection with infinite line, then checks if intersection point is within segment.
fn intersect_particle_with_line_segment(
    pt: &Phonon,
    normal: &PointXY,
    offset: f64,
    a: &PointXY,
    b: &PointXY,
    t_min: f64,
) -> Option<f64> {
    if let Some(t) = intersect_particle_with_line(pt, normal, offset, t_min) {
        let x = pt.x + pt.vx * t;
        let y = pt.y + pt.vy * t;
        if is_between(x, a.x, b.x) && is_between(y, a.y, b.y) {
            return Some(t);
        }
    }
    None
}

#[inline(always)]
/// Computes intersection of particle trajectory with
/// circle segment defined by center and radius.
/// a and b are the endpoints of the circle segment.
/// Returns None if no intersection (i.e., both intersections
/// are behind the particle or outside the arc segment).
fn intersect_particle_with_circle_segment(
    pt: &Phonon,
    a: &PointXY,
    b: &PointXY,
    center: &PointXY,
    r_sqr: f64,
    t_min: f64,
) -> Option<f64> {
    // Particle trajectory: (x, y, z) + t*(vx, vy, vz)
    // Insert into circle equation: (x - cx)^2 + (y - cy)^2 = r^2
    // Solve for t. This leads to a quadratic equation: a*t^2 + b*t + c = 0
    // with:
    //  a = vx^2 + vy^2
    //  b = 2 * (vx*(x - cx) + vy*(y - cy))
    //  c = (x - cx)^2 + (y - cy)^2 - r^2
    // Finally, check if t>0 and if intersection point is within the arc segment.
    let dx = pt.x - center.x;
    let dy = pt.y - center.y;
    let a_ = pt.vx * pt.vx + pt.vy * pt.vy;
    //let a_inv = pt.vxy_sq_inv; // precomputed 1/(vx^2 + vy^2)
    let b_ = 2.0 * (pt.vx * dx + pt.vy * dy);
    let c_ = dx * dx + dy * dy - r_sqr;

    let discriminant = b_ * b_ - 4.0 * a_ * c_;
    if discriminant < 0.0 {
        return None; // No intersection
    }
    let sqrt_disc = discriminant.sqrt();
    let inv_denom = 0.5 / a_;
    let t1 = (-b_ - sqrt_disc) * inv_denom;
    let t2 = (-b_ + sqrt_disc) * inv_denom; // t1 <= t2

    // Check both intersections
    let tol = TOL;
    if t2 <= tol {
        return None; // Both intersections are behind the particle
    }
    let p_int_1 = PointXY {
        x: pt.x + pt.vx * t1,
        y: pt.y + pt.vy * t1,
    };
    if t1 > tol && t1 < t_min && is_on_circle_segment(&p_int_1, center, a, b) {
        return Some(t1);
    }

    let p_int_2 = PointXY {
        x: pt.x + pt.vx * t2,
        y: pt.y + pt.vy * t2,
    };
    if t2 > tol && t2 < t_min && is_on_circle_segment(&p_int_2, center, a, b) {
        return Some(t2);
    }
    None
}

#[derive(Debug)]
enum Segment {
    Line { normal: PointXY, offset: f64 },
    Arc { center: PointXY, r_sqr: f64 },
}

// These structs exist to avoid accidently mixing up
// polygon and edge indices in function calls
#[derive(Debug)]
pub struct PolygonIndex(usize);

#[derive(Debug)]
pub struct EdgeIndex(usize);

#[derive(Debug)]
/// Represents the wall boundaries in the x-y plane.
/// Consists of an outer wall polygon and optionally multiple inner wall polygons (holes).
/// Polygons here can also contain arc segments.
pub struct Wall {
    polygons: Vec<WallPolygon>,
}

impl Wall {
    /// Creates outer wall polygon from outside_wall config and inner wall polygons from inside_walls configs.
    pub fn new(outside_wall: &WallConfig, inside_walls: &[WallConfig]) -> Self {
        let mut polygons = Vec::with_capacity(1 + inside_walls.len());
        polygons.push(WallPolygon::new(outside_wall));
        for hole in inside_walls {
            polygons.push(WallPolygon::new(hole));
        }
        Self { polygons }
    }

    /// Returns the time until intersection with any wall polygon,
    /// along with the index of the polygon and the index of the edge of
    /// the respective polygon, as well as the intersection point.
    ///
    /// Returns None if no intersection within time_to_beat.
    pub fn time_to_wall(
        &self,
        pt: &Phonon,
        time_to_beat: f64,
    ) -> Option<(f64, PolygonIndex, EdgeIndex, PointXYZ)> {
        // This is the hottest part of the code, taking the majority of the runtime.
        // I optimized this (and the functions called from here) as much as I could.

        let mut min_polygon_index = PolygonIndex(0);
        let mut min_edge_index = EdgeIndex(0);
        let mut t_min = time_to_beat;

        for (polygon_index, polygon) in self.polygons.iter().enumerate() {
            let (t, edge_index) = polygon.time_to_wall(pt, t_min);

            if t < t_min {
                t_min = t;
                min_polygon_index = PolygonIndex(polygon_index);
                min_edge_index = edge_index;
            }
        }

        let mut closest_point = PointXYZ {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
        if t_min < time_to_beat {
            closest_point.x = pt.x + pt.vx * t_min;
            closest_point.y = pt.y + pt.vy * t_min;
            closest_point.z = pt.z + pt.vz * t_min;
            Some((t_min, min_polygon_index, min_edge_index, closest_point))
        } else {
            None
        }
    }

    /// Scatter phonon at wall specified by polygon index.
    /// Pass edge index down to polygon for scattering at correct segment.
    pub fn wall_scattering(
        &self,
        pt: &mut Phonon,
        polygon_index: PolygonIndex,
        edge_index: EdgeIndex,
        results: &mut EnergyResults,
        rng: &mut Rng,
        sin_table: &SinTable,
    ) {
        self.polygons[polygon_index.0].wall_scattering(pt, edge_index, results, rng, sin_table);
    }

    /// Check if point (x, y) is strictly inside the simulation domain
    /// (i.e., inside outer wall and outside of all inner walls).
    pub fn is_strictly_inside(&self, x: f64, y: f64) -> bool {
        for polygon in &self.polygons {
            let is_inside = polygon.is_strictly_inside(x, y);
            // should not be inside a hole (inner wall)
            // should be inside of outer wall
            let should_be_inside = polygon.is_outer_wall;
            if is_inside != should_be_inside {
                return false;
            }
        }
        true
    }

    /// Same as is_strictly_inside but with tolerance.
    #[cfg(debug_assertions)]
    pub fn is_inside(&self, x: f64, y: f64, tolerance: f64) -> bool {
        for polygon in &self.polygons {
            let is_inside = polygon.is_inside(x, y, tolerance);
            // should not be inside a hole (inner wall)
            // should be inside of outer wall
            let should_be_inside = polygon.is_outer_wall;
            if is_inside != should_be_inside {
                return false;
            }
        }
        true
    }
}

#[derive(Debug)]
/// Represents a wall shape made of line and circle segments for boundary scattering and containment checks.
pub struct WallPolygon {
    points: Vec<PointXY>,
    points_rolled: Vec<PointXY>,
    is_bridge: Vec<bool>,
    segment_types: Vec<Segment>, // lines: (normal, offset) with lines pointing outside of domain
    n: usize,
    use_diffuse_scattering: bool,
    is_outer_wall: bool,
}

impl WallPolygon {
    /// Creates a new WallPolygon from the points in WallConfig.
    /// Precomputes quantities for fast scattering calculations during simulation.
    pub fn new(wall_config: &WallConfig) -> Self {
        let mut points = wall_config.points.clone();
        let mut is_bridge = wall_config.is_bridge.clone();
        let mut circle_radius = wall_config.circle_radius.clone();

        // If points are in clockwise order, reverse to make them counter-clockwise.
        // Use Euler's shoelace formula to compute signed area and thus orientation of polygon
        let signed_area = points
            .iter()
            .zip(points.iter().cycle().skip(1))
            .map(|(p1, p2)| p1.x * p2.y - p2.x * p1.y)
            .sum::<f64>()
            * 0.5;
        if signed_area < 0.0 {
            points.reverse();
            is_bridge.reverse();
            circle_radius.reverse();
        }

        let n = points.len();
        let mut points_rolled = Vec::with_capacity(n);
        let mut edges = Vec::with_capacity(n);

        let mut edge_normals = Vec::with_capacity(n);
        let mut offsets = Vec::with_capacity(n);
        let mut segment_types = Vec::with_capacity(n);

        // precompute edges, edge normals and edge offsets
        for i in 0..n {
            let next = (i + 1) % n;
            points_rolled.push(points[next].clone());
            let edge = &points[next] - &points[i];
            edges.push(edge.clone());

            let mut normal = PointXY {
                x: edge.y,
                y: -edge.x,
            };
            let len = (normal.x * normal.x + normal.y * normal.y).sqrt();
            if len > 0.0 {
                normal.x /= len;
                normal.y /= len;
            }

            // if inside wall, normal should point inside of polygon to point to outside of domain
            if !wall_config.is_outside {
                normal = -normal;
            }
            let offset = normal.x * points[i].x + normal.y * points[i].y;

            offsets.push(offset);
            edge_normals.push(normal);
        }

        // precompute vector with segment information (line or circle segment)
        for i in 0..n {
            match circle_radius[i] {
                0.0 => {
                    let normal = edge_normals[i].clone();
                    let offset = offsets[i];
                    segment_types.push(Segment::Line { normal, offset });
                }
                r => {
                    let edge_before = edges[(i + n - 1) % n].clone();
                    let edge_after = edges[(i + 1) % n].clone();
                    let c = circle_center(
                        points[i].x,
                        points[i].y,
                        points[(i + 1) % n].x,
                        points[(i + 1) % n].y,
                        r.abs(),
                    );
                    if !does_circle_segment_keep_convexity(
                        &points[i],
                        &points[(i + 1) % n],
                        &c,
                        &edge_before,
                        &edge_after,
                    ) {
                        panic!("Circle segment at edge {i} with radius {r} breaks outer wall convexity. 
                            Points: ({},{}) to ({},{}). Edges: before=({},{}) after=({},{}) Center=({},{})",
                            points[i].x, points[i].y,
                            points[(i+1)%n].x, points[(i+1)%n].y,
                            edge_before.x, edge_before.y,
                            edge_after.x, edge_after.y,
                            c.x, c.y
                        );
                    }
                    segment_types.push(Segment::Arc {
                        center: c,
                        r_sqr: r * r,
                    });
                }
            }
        }

        Self {
            points: points.clone(),
            points_rolled,
            is_bridge,
            segment_types,
            n,
            use_diffuse_scattering: wall_config.diffuse_scattering,
            is_outer_wall: wall_config.is_outside,
        }
    }

    /// Calculate time until intersection with outer wall.
    /// Returns time, edge index, and intersection point.
    /// Returns None if no intersection.
    #[allow(clippy::collapsible_if)]
    #[inline(always)]
    pub fn time_to_wall(&self, pt: &Phonon, t_min: f64) -> (f64, EdgeIndex) {
        let mut min_time = t_min;
        let mut min_edge_index = EdgeIndex(0);

        if self.is_outer_wall {
            for (i, segment) in self.segment_types.iter().enumerate() {
                let a = &self.points[i];
                let b = &self.points_rolled[i];
                let result = match segment {
                    Segment::Line { normal, offset } => {
                        intersect_particle_with_line(pt, normal, *offset, min_time)
                    }
                    Segment::Arc { center, r_sqr } => {
                        intersect_particle_with_circle_segment(pt, a, b, center, *r_sqr, min_time)
                    }
                };
                if let Some(t) = result {
                    min_time = t;
                    min_edge_index = EdgeIndex(i);
                }
            }
        } else {
            for (i, segment) in self.segment_types.iter().enumerate() {
                let a = &self.points[i];
                let b = &self.points_rolled[i];
                let result = match segment {
                    Segment::Line { normal, offset } => {
                        intersect_particle_with_line_segment(pt, normal, *offset, a, b, min_time)
                    }
                    Segment::Arc { center, r_sqr } => {
                        intersect_particle_with_circle_segment(pt, a, b, center, *r_sqr, min_time)
                    }
                };
                if let Some(t) = result {
                    min_time = t;
                    min_edge_index = EdgeIndex(i);
                }
            }
        }
        (min_time, min_edge_index)
    }

    /// Scatter phonon at wall segment specified by edge_index.
    /// When calling this function, the particle position must be
    /// identical to intersection point with wall segment.
    pub fn wall_scattering(
        &self,
        pt: &mut Phonon,
        edge_index: EdgeIndex,
        results: &mut EnergyResults,
        rng: &mut Rng,
        sin_table: &SinTable,
    ) {
        // unit normal pointing into the domain
        let normal = match &self.segment_types[edge_index.0] {
            Segment::Line { normal, .. } => -normal,
            Segment::Arc { center, .. } => {
                let mut n = PointXY {
                    // normal vector to circle
                    x: pt.x - center.x,
                    y: pt.y - center.y,
                };

                // make unit length
                let len_sq = n.x * n.x + n.y * n.y;
                if len_sq == 0.0 {
                    panic!(
                        "Phonon located exactly at circle segment center during scattering. 
                        The circle segment likely doesn't keep convexity."
                    );
                }
                let len_inv = 1.0 / len_sq.sqrt();
                n.x *= len_inv;
                n.y *= len_inv;

                if self.is_outer_wall { -n } else { n }
            }
        };

        match self.use_diffuse_scattering {
            false => {
                // specular reflection: reflect velocity vector across plane tangent to wall
                // v' = v - 2*(v . n)*n, where n is the normal vector
                let dot = pt.vx * normal.x + pt.vy * normal.y;
                pt.vx -= 2.0 * dot * normal.x;
                pt.vy -= 2.0 * dot * normal.y;
            }
            true => {
                // diffuse scattering: sample new direction from uniform distribution over hemisphere
                // then rotate to match the hemisphere defined by the wall normal vector
                let mut x_rot: f64;
                let mut y_rot: f64;
                let mut z_rot: f64;

                loop {
                    let (x, y, z, _) = uniform_sample_hemisphere(rng, sin_table);
                    (x_rot, y_rot, z_rot) =
                        rotate_vector_to_normal_hemisphere(x, y, z, normal.x, normal.y);

                    if normal.x * x_rot + normal.y * y_rot > 1e-12 {
                        // check if rotated vector is in correct hemisphere, otherwise resample
                        // can be not fulfilled e.g. due to numerical inaccuracies
                        break;
                    }
                }
                pt.vx = x_rot * pt.speed;
                pt.vy = y_rot * pt.speed;
                pt.vz = z_rot * pt.speed;
                pt.vz_abs_inv = if pt.vz.abs() > 1e-12 {
                    1.0 / pt.vz.abs()
                } else {
                    1e12
                };
                //}
            }
        }

        if self.is_bridge[edge_index.0] {
            results.e_loss = pt.energy_fraction_remaining;
            pt.energy_fraction_remaining = 0.0;
        }
    }

    /// Check if point (x, y) is strictly inside this wall polygon.
    pub fn is_strictly_inside(&self, x: f64, y: f64) -> bool {
        // This assumes a convex polygon with segments stated in counter-clockwise order.
        for i in 0..self.n {
            let a = &self.points[i];
            let b = &self.points_rolled[i];
            let segment = &self.segment_types[i];
            let is_outside = match segment {
                Segment::Line { normal, offset } => {
                    if self.is_outer_wall {
                        normal.x * x + normal.y * y > *offset
                    } else {
                        (-normal.x) * x + (-normal.y) * y > -*offset
                    }
                }
                Segment::Arc { center, r_sqr } => {
                    !is_strictly_inside_at_circle_segment(x, y, a, b, center, *r_sqr)
                }
            };
            if is_outside {
                return false;
            }
        }

        true
    }

    #[inline(always)]
    #[cfg(debug_assertions)]
    pub fn is_inside(&self, x: f64, y: f64, tolerance: f64) -> bool {
        for i in 0..self.n {
            let a = &self.points[i];
            let b = &self.points_rolled[i];
            let segment = &self.segment_types[i];
            let is_outside = match segment {
                Segment::Line { normal, offset } => {
                    if self.is_outer_wall {
                        normal.x * x + normal.y * y - tolerance > *offset
                    } else {
                        (-normal.x) * x + (-normal.y) * y - tolerance > -*offset
                    }
                }
                Segment::Arc { center, r_sqr } => {
                    !is_inside_at_circle_segment(x, y, a, b, center, *r_sqr, tolerance)
                }
            };
            if is_outside {
                return false;
            }
        }
        true
    }
}

/// For convex polygon, check if point is inside at arc segment
/// as part of the is_strictly_inside checks.
#[inline(always)]
fn is_strictly_inside_at_circle_segment(
    x: f64,
    y: f64,
    a: &PointXY,
    b: &PointXY,
    center: &PointXY,
    r_sqr: f64,
) -> bool {
    // Strategy:
    // 1. Consider the Circle segment as a line, check if would be inside. Then:
    //    If so: ok already
    // 2. If not, it is still inside (on the area segment between the circle and the line connecting a and b)
    //            if and only if it is inside the full circle

    // Compute normal pointing outside the polygon.
    // This assumes that a and b are specified in counter-clockwise order.
    // The normal is not normalized but this is fine here.
    let edge = b - a;
    let normal = PointXY {
        x: edge.y,
        y: -edge.x,
    };
    let offset = normal.x * a.x + normal.y * a.y;

    let is_inside_line = normal.x * x + normal.y * y < offset;
    let is_inside_circle = (x - center.x).powi(2) + (y - center.y).powi(2) < r_sqr;

    is_inside_line || is_inside_circle
}

#[inline(always)]
#[cfg(debug_assertions)]
fn is_inside_at_circle_segment(
    x: f64,
    y: f64,
    a: &PointXY,
    b: &PointXY,
    center: &PointXY,
    r_sqr: f64,
    tolerance: f64,
) -> bool {
    // Same as is_strictly_inside_at_circle_segment but with tolerance for points close to the boundary
    let edge = b - a;
    let normal = PointXY {
        x: edge.y,
        y: -edge.x,
    };
    let offset = normal.x * a.x + normal.y * a.y;

    let is_inside_line = normal.x * x + normal.y * y < offset + tolerance;

    let r = r_sqr.sqrt();
    let is_inside_circle =
        (x - center.x).powi(2) + (y - center.y).powi(2) < r_sqr + 2.0 * r * tolerance;

    is_inside_line || is_inside_circle
}
