use crate::nnr::Layer;
use std::f64::consts::PI;
use rand::Rng;

mod nnr;
mod png;

fn main() {
    let mut img = vec![vec![vec![0.0;3];3];3];
    let mut count = 0.0;
    for k in 0..img.len() {
        for i in 0..img[k].len() {
            for j in 0..img[k][i].len() {
                img[i][j][k] = count;
                count += 1.0;
            }
        }
    }
    println!("{:?}", img);
    let img = nnr::cnn::img2col(img, 2, 2, 2, 1);
    println!("{:?}",img);
    /*
    let mut r = rand::thread_rng();
    let input: usize = 1;
    let middle: usize = 10;
    let output: usize = 1;
    let epoch: usize = 2000;
    let eta: f64 = 0.1;
    let batch: usize = 100;

    let interval = 200;

    let mut middle_layer = Layer::init(input, middle, 1, nnr::activations::sigmoid, nnr::activations::sigmoid_grad);
    let mut output_layer = Layer::init(middle, output, 1, nnr::activations::identity, nnr::activations::identity_grad);

    for i in 0..epoch {
        let mut err = 0.0;
        for j in 0..batch {
            let input: f64 = r.gen();
            let input = (input - 0.5) * 2.0;
            let t = (input * PI).sin();
            let input = vec![vec![input; 1]; 1];
            let train = vec![vec![t; 1]; 1];
            let y = middle_layer.forward(&input);
            let y = output_layer.forward(y).clone();
            let x = output_layer.backward(&train);
            let x = middle_layer.backward(&x);

            middle_layer.update(eta);
            output_layer.update(eta);

            err += (y[0][0] - t).powf(2.0) * 0.5;
        }
        if i % interval == 0 {
            println!("{:?}", err / batch as f64);
        }
    }*/
}
