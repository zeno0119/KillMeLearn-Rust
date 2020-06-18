use crate::nnr::Layer;
use std::f64::consts::PI;
use rand::Rng;
use crate::png::{parse_png, img2vec};
use crate::nnr::matrix::{t, shrink_vec, shrink_img};

mod nnr;
mod png;

fn main() {
    let mut res = vec![vec![0.0;2]; 0];
    let mut r = rand::thread_rng();
    let input: usize = 128 * 128 * 3;
    let middle: usize = 200;
    let output: usize = 4;
    let epoch: usize = 50;
    let eta: f64 = 0.1;
    let batch: usize = 50;

    let yasuna = 349;
    let sonya = 173;
    let agiri = 39;
    let botsu = 8;

    let interval = 5;

    let mut middle_layer = Layer::init(input, middle, 1, nnr::activations::sigmoid, nnr::activations::sigmoid_grad);
    let mut middle_layer2 = Layer::init(middle, middle, 1, nnr::activations::sigmoid, nnr::activations::sigmoid_grad);
    let mut middle_layer3 = Layer::init(middle, middle, 1, nnr::activations::sigmoid, nnr::activations::sigmoid_grad);
    let mut middle_layer4 = Layer::init(middle, middle, 1, nnr::activations::sigmoid, nnr::activations::sigmoid_grad);
    let mut middle_layer5 = Layer::init(middle, middle, 1, nnr::activations::sigmoid, nnr::activations::sigmoid_grad);
    let mut output_layer = Layer::init(middle, output, 1, nnr::activations::soft_max, nnr::activations::identity_grad);

    let mut path: String = "".to_string();

    for i in 0..epoch {
        //yasuna 531 sonya 177 agiri 43 botsu 10
        let mut err = 0.0;
        let mut err2 = 0.0;
        let mut input:usize  = 0;
        for j in 0..batch {
            let mut train = vec![vec![0.0; 4]; 1];
            if input % 4 == 0 {
                let input = r.gen::<usize>() % yasuna;
                path = format!("./image/yasuna/{}.png", input);
                train[0][0] = 1.0;
            } else if input % 4 == 1{
                let input = r.gen::<usize>() % sonya;
                path = format!("./image/sonya/{}.png", input);
                train[0][1] = 1.0;
            } else if input % 4 == 2 {
                let input = r.gen::<usize>() % agiri;
                path = format!("./image/agiri/{}.png", input);
                train[0][2] = 1.0;
            } else if input % 4 == 3 {
                let input = r.gen::<usize>() % botsu;
                path = format!("./image/botsu/{}.png", input);
                train[0][3] = 1.0;
            }
            let path = path.as_str();
            // println!("{:?}",path);
            let inp = vec![img2vec(parse_png(path) ); 1];
            let y = middle_layer.forward(&inp);
            let y = middle_layer2.forward(y);
            let y = middle_layer3.forward(y);
            let y = middle_layer4.forward(y);
            let y = middle_layer5.forward(y);
            let y = output_layer.forward(y).clone();
            let x = output_layer.backward(&train);
            let x = middle_layer5.backward(&x);
            let x = middle_layer4.backward(&x);
            let x = middle_layer3.backward(&x);
            let x = middle_layer2.backward(&x);
            let x = middle_layer.backward(&x);

            middle_layer.update(eta);
            middle_layer2.update(eta);
            middle_layer3.update(eta);
            middle_layer4.update(eta);
            middle_layer5.update(eta);
            output_layer.update(eta);

            //println!("{:?}", y);
            for i in 0..train.len() {
                for j in 0..train[i].len() {
                    err += (-train[i][j]) * (y[i][j] + 0.0000001).log(std::f64::consts::E);
                    //println!("{}", err);
                }
            }
            input += 1;
            // println!("step {} end", j);
        }

        for j in 0..18 as usize {
            let mut train = vec![vec![0.0; 4]; 1];
            if j >= 0 && j < 5{train[0][0] = 1.0;}
            else if j >= 5 && j < 10{train[0][1] = 1.0;}
            else if j >= 10 && j < 15 {train[0][2] = 1.0;}
            else {train[0][3] = 1.0;}
            let path = format!("./image/test/{}.png", j);
            let path = path.as_str();
            if i % 10 == 0 {
                println!("{}", path);
            }
            let inp = vec![img2vec(parse_png(path)); 1];
            let y = middle_layer.forward(&inp);
            let y = middle_layer2.forward(y);
            let y = middle_layer3.forward(y);
            let y = middle_layer4.forward(y);
            let y = middle_layer5.forward(y);
            let y = output_layer.forward(y).clone();
            if i % 10 == 0 {
                println!("{:?}", y);
            }
            for i in 0..train.len() {
                for j in 0..train[i].len() {
                    err2 += (-train[i][j]) * (y[i][j] + 0.0000001).log(std::f64::consts::E);
                    //println!("{}", err);
                }
            }
        }
        if i % interval == 0 {
            println!("{:?}, {:?}", err / batch as f64, err2 / 18 as f64);
        }
        res.push(vec![err / batch as f64, err2 / 18 as f64]);
    }
    for re in res.iter() {
        println!("{:?}, {:?}", re[0], re[1]);
    }
}
