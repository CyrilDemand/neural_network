use std::f64::consts::E;
#[derive(Clone, Copy)]
pub struct Activation<'a> {
    pub function: &'a dyn Fn(f32) -> f32,
    pub derivative: &'a dyn Fn(f32) -> f32,
}

pub const SIGMOID: Activation = Activation {
    function: &|x| (1.0 / (1.0 + E.powf(-x as f64))) as f32,
    derivative: &|x| x * (1.0 - x),
};