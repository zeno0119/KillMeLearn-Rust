use crate::nnr::Layer;
use std::f64::consts::PI;
use rand::Rng;
use crate::png::{parse_png, img2vec};
use crate::nnr::matrix::{t, shrink_vec, shrink_img};

mod nnr;
mod png;

fn main() {
    let mut r = rand::thread_rng();
    let input: usize = 128 * 128 * 3 / 4;
    let middle: usize = 100;
    let output: usize = 4;
    let epoch: usize = 100;
    let eta: f64 = 0.1;
    let batch: usize = 50;

    let yasuna = 351;
    let sonya = 177;
    let agiri = 43;
    let botsu = 10;

    let interval = 1;

    let mut middle_layer = Layer::init(input, middle, 1, nnr::activations::sigmoid, nnr::activations::sigmoid_grad);
    let mut middle_layer2 = Layer::init(middle, middle, 1, nnr::activations::sigmoid, nnr::activations::sigmoid_grad);
    let mut output_layer = Layer::init(middle, output, 1, nnr::activations::soft_max, nnr::activations::identity_grad);

    let mut path: String = "".to_string();
    for i in 0..epoch {
        //yasuna 531 sonya 177 agiri 43 botsu 10
        let mut err = 0.0;
        for j in 0..batch {
            let input = r.gen::<usize>() % (yasuna + sonya + agiri + botsu);
            let mut train = vec![vec![0.0; 4]; 1];
            if input < yasuna {
                path = format!("./image/yasuna/{}.png", input);
                train[0][0] = 1.0;
            } else if input < yasuna + sonya {
                path = format!("./image/sonya/{}.png", input - yasuna);
                train[0][1] = 1.0;
            } else if input < yasuna + sonya + agiri {
                path = format!("./image/agiri/{}.png", input - yasuna - sonya);
                train[0][2] = 1.0;
            } else if input < yasuna + sonya + agiri + botsu {
                path = format!("./image/botsu/{}.png", input - yasuna - sonya - agiri);
                train[0][3] = 1.0;
            }
            let path = path.as_str();
            // println!("{:?}",path);
            let inp = vec![img2vec(shrink_img(&parse_png(path), 2)); 1];
            let y = middle_layer.forward(&inp);
            let y = middle_layer2.forward(y);
            let y = output_layer.forward(y).clone();
            let x = output_layer.backward(&train);
            let x = middle_layer2.backward(&x);
            let x = middle_layer.backward(&x);

            middle_layer.update(eta);
            middle_layer2.update(eta);
            output_layer.update(eta);

            //println!("{:?}", y);
            for i in 0..train.len() {
                for j in 0..train[i].len() {
                    err += (-train[i][j]) * (y[i][j] + 0.0000001).log(std::f64::consts::E);
                    //println!("{}", err);
                }
            }
            // println!("step {} end", j);
        }
        if i % interval == 0 {
            println!("{:?}", err / batch as f64);
        }
    }
}
