use crate::lib::Network::Network;
use crate::lib::activation::SIGMOID;
use std::path::Path;

pub mod lib;

// 0,0 -> 0
// 0,1 -> 1
// 1,0 -> 1
// 1,1 -> 0



fn main() {
    println!("Hello, world!");
    addition();
}

fn addition(){
    let mut path:String = String::from("ai_addition_bien.json");

    let rs:bool = Path::new(&path).exists();

    let mut network =  Network::new(vec![2, 3, 1], 0.5, SIGMOID);

    if rs == true{
        network.load(path);
    }
    else{
        println!("File does not exist");
        let mut inputs: Vec<Vec<f32>> = vec![
        ];
        let mut targets: Vec<Vec<f32>> = vec![
        ];

        let mut a: f32 = 1.0;
        loop {
            let mut b: f32 = 1.0;
            loop {
                inputs.push(vec![a/100.0 , b/100.0]);
                targets.push(vec![a/100.0 + b/100.0]);
                b += 1.0;
                if b == 51.0 { break;}
            }
            a += 1.0;
            if a == 51.0 { break;}
        }

        println!("{:?}", inputs.len());
        println!("{:?}", inputs[0]);
        println!("{:?}", targets.len());
        println!("{:?}", targets[0]);
        println!("Training");


        network.train(inputs, targets, 1000);
        network.save(path);
    }

    println!("8 + 3: {:?}", network.feed_forward(vec![0.08, 0.03])[0]*100.0);
    println!("15 + 24: {:?}", network.feed_forward(vec![0.15, 0.24])[0]*100.0);
    println!("50 + 30: {:?}", network.feed_forward(vec![0.5, 0.3])[0]*100.0);
    println!("29 + 31: {:?}", network.feed_forward(vec![0.29, 0.31])[0]*100.0);
}

fn xor(){
    let mut path:String = String::from("ai_xor.json");

    let rs:bool = Path::new(&path).exists();

    let mut network =  Network::new(vec![2, 3, 1], 0.5, SIGMOID);

    if rs == true{
        network.load(path);
    }
    else{
        println!("File does not exist");
        let inputs = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0]
        ];

        let targets = vec![
            vec![0.0],
            vec![1.0],
            vec![1.0],
            vec![0.0]
        ];

        network.train(inputs, targets, 100000);
        network.save(path);
    }

    println!("0 and 0: {:?}", network.feed_forward(vec![0.0, 0.0]));
    println!("1 and 0: {:?}", network.feed_forward(vec![1.0, 0.0]));
    println!("0 and 1: {:?}", network.feed_forward(vec![0.0, 1.0]));
    println!("1 and 1: {:?}", network.feed_forward(vec![1.0, 1.0]));
}