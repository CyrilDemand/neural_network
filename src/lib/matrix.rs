use rand::{Rng, thread_rng};
use std::fmt::{Debug, Formatter, Result};

#[derive(Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Vec<f32>>,
}

impl Matrix {
    pub fn zeros(rows: usize, cols: usize) -> Matrix {
        Matrix {
            rows,
            cols,
            data: vec![vec![0.0; cols]; rows],
        }
    }

    pub fn random(rows: usize, cols: usize) -> Matrix {
        let mut rng = thread_rng();

        let mut res = Matrix::zeros(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                res.data[i][j] = rng.gen::<f32>() * 2.0 - 1.0;
            }
        }
        res
    }


    pub fn from_array(arr: Vec<Vec<f32>>) -> Matrix {
       Matrix{
           rows: arr.len(),
           cols: arr[0].len(),
           data: arr,
       }
    }

    pub fn multiply(&mut self, other: &Matrix) -> Matrix {
        if self.cols != other.rows {
            panic!("Columns of A must match rows of B.");
        }

        let mut result = Matrix::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                     sum += self.data[i][k] * other.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
        result
    }

    pub fn add(&mut self, other: &Matrix) -> Matrix{
        // do a function that add two matrix
        if self.cols != other.cols || self.rows != other.rows {
            panic!("Columns and rows of A must match columns and rows of B.");
        }

        let mut result = Matrix::zeros(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[i][j] = self.data[i][j] + other.data[i][j];
            }
        }
        result
    }

    pub fn dot_multiplication(&self, other: &Matrix) -> Matrix {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Attempted to dot multiply by matrix of incorrect dimensions");
        }

        let mut res = Matrix::zeros(self.rows, self.cols);

        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[i][j] = self.data[i][j] * other.data[i][j];
            }
        }
        res

    }

    pub fn substract(&mut self, other: &Matrix) -> Matrix{
        // do a function that add two matrix
        if self.cols != other.cols || self.rows != other.rows {
            panic!("Columns and rows of A must match columns and rows of B.");
        }

        let mut result = Matrix::zeros(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[i][j] = self.data[i][j] - other.data[i][j];
            }
        }
        result
    }

    pub fn map(&mut self, function: &dyn Fn(f32) -> f32) -> Matrix {
        Matrix::from_array(
            (self.data)
                .clone()
                .into_iter()
                .map(|row| row.into_iter()
                    .map(|x| function(x))
                    .collect())
                .collect())
    }

    pub fn transpose(&mut self) -> Matrix {
        let mut result = Matrix::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[j][i] = self.data[i][j];
            }
        }
        result
    }
}

impl Debug for Matrix {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(
            f,
            "Matrix {{\n{}\n}}",
            (&self.data)
                .into_iter()
                .map(|row| "  ".to_string()
                    + &row
                    .into_iter()
                    .map(|value| value.to_string())
                    .collect::<Vec<String>>()
                    .join(" "))
                .collect::<Vec<String>>()
                .join("\n")
        )
    }
}