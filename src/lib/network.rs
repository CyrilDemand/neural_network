use std::fs::File;
use std::io::{Read, Write};
use crate::lib::activation::Activation;
use crate::lib::Matrix::Matrix;

use serde::{Deserialize, Serialize};
use serde_json::{from_str, json};

pub struct Network<'a> {
    layers: Vec<usize>,
    weights: Vec<Matrix>,
    biases: Vec<Matrix>,
    data: Vec<Matrix>,
    learning_rate: f32,
    activation: Activation<'a>,
}

#[derive(Serialize, Deserialize)]
struct SaveData {
    weights: Vec<Vec<Vec<f32>>>,
    biases: Vec<Vec<Vec<f32>>>,
}

impl Network<'_> {
    pub fn new(layers: Vec<usize>, learning_rate: f32, activation: Activation) -> Network{
        let mut weights = Vec::new();
        let mut biases = Vec::new();

        for i in 0..layers.len()-1 {
            weights.push(Matrix::random(layers[i+1], layers[i]));
            biases.push(Matrix::random(layers[i+1], 1));
        }
        Network {
            layers,
            weights,
            biases,
            data : Vec::new(),
            learning_rate,
            activation,
        }
    }

    pub fn feed_forward(&mut self, input: Vec<f32>) -> Vec<f32> {
        if input.len() != self.layers[0] {
            println!("{:?}", input.len());
            println!("{:?}", self.layers[0]);
            panic!("Input size must match the number of neurons in the input layer.");
        }

        let mut current = Matrix::from_array(vec![input]).transpose();
        self.data = vec![current.clone()];

        for i in 0..self.layers.len()-1 {
            current = self.weights[i].multiply(&current).add(&self.biases[i]).map(self.activation.function);
            self.data.push(current.clone());
        }
        current.data[0].to_owned()
    }

    pub fn back_propogate(&mut self, outputs: Vec<f32>, targets: Vec<f32>) {
        if targets.len() != self.layers[self.layers.len()-1] {
            panic!("Output size must match the number of neurons in the output layer.");
        }

        let mut parsed = Matrix::from_array(vec![outputs]).transpose();
        let mut errors = Matrix::from_array(vec![targets]).transpose().substract(&parsed);
        let mut gradients = parsed.map(self.activation.derivative);

        for i in (0..self.layers.len()-1).rev() {
            gradients = gradients
                .dot_multiplication(&errors)
                .map(&|x| x * self.learning_rate);

            self.weights[i] = self.weights[i].add(&gradients.multiply(&self.data[i].transpose()));
            self.biases[i] = self.biases[i].add(&gradients);

            errors = self.weights[i].transpose().multiply(&errors);
            gradients = self.data[i].map(self.activation.derivative);
        }
    }

    pub fn train(&mut self, inputs: Vec<Vec<f32>>, targets: Vec<Vec<f32>>, epochs: u32){
        if inputs.len() != targets.len() {
            panic!("Input and target size must match.");
        }

        for i in 0..epochs {
           // show progress
            if epochs > 100 && i % (epochs / 100) == 0 {
                println!("{}% done", i / (epochs / 100));
            }

            for j in 0..inputs.len() {
                let output = self.feed_forward(inputs[j].clone());
                self.back_propogate(output, targets[j].clone());
            }
        }
    }

    pub fn save(&self, file: String) {
        let mut file = File::create(file).expect("Unable to touch save file");

        file.write_all(
            json!({
				"weights": self.weights.clone().into_iter().map(|matrix| matrix.data).collect::<Vec<Vec<Vec<f32>>>>(),
				"biases": self.biases.clone().into_iter().map(|matrix| matrix.data).collect::<Vec<Vec<Vec<f32>>>>()
			}).to_string().as_bytes(),
        ).expect("Unable to write to save file");
    }

    pub fn load(&mut self, file: String) {
        let mut file = File::open(file).expect("Unable to open save file");
        let mut buffer = String::new();

        file.read_to_string(&mut buffer)
            .expect("Unable to read save file");

        let save_data: SaveData = from_str(&buffer).expect("Unable to serialize save data");

        let mut weights = vec![];
        let mut biases = vec![];

        for i in 0..self.layers.len() - 1 {
            weights.push(Matrix::from_array(save_data.weights[i].clone()));
            biases.push(Matrix::from_array(save_data.biases[i].clone()));
        }

        self.weights = weights;
        self.biases = biases;
    }
}