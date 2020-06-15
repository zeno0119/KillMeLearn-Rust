use std::cmp::{min, max};

pub fn identity (a: &Vec<Vec<f64>>) -> Vec<Vec<f64>>{
    let mut res = vec![vec![0.0;a[0].len()];a.len()];
    for i in 0..a.len() {
        for j in 0..a[0].len() {
            res[i][j] = a[i][j].clone();
        }
    }
    return res;
}

pub fn identity_grad (y: &Vec<Vec<f64>>, t: &Vec<Vec<f64>>) -> Vec<Vec<f64>>{
    let mut res = vec![vec![0.0;y[0].len()];y.len()];
    for i in 0..t.len() {
        for j in 0..t[0].len() {
            res[i][j] = y[i][j].clone() - t[i][j].clone();
        }
    }
    return res;
}

pub fn relu (a: &Vec<Vec<f64>>) -> Vec<Vec<f64>>{
    let mut res = vec![vec![0.0;a[0].len()];a.len()];
    for i in 0..a.len() {
        for j in 0..a[0].len() {
            res[i][j] = {
                if a[i][j] >= 0.0{
                    a[i][j]
                }else {
                    0.0
                }
            };
        }
    }
    return res;
}

pub fn relu_grad (y: &Vec<Vec<f64>>, t: &Vec<Vec<f64>>) -> Vec<Vec<f64>>{
    let mut res = vec![vec![0.0;y[0].len()];y.len()];
    for i in 0..t.len() {
        for j in 0..t[0].len() {
            res[i][j] = {
                if y[i][j] >= 0.0 {
                    t[i][j]
                }else {
                    0.0
                }
            };
        }
    }
    return res;
}

pub fn sigmoid (a: &Vec<Vec<f64>>) -> Vec<Vec<f64>>{
    let mut res = vec![vec![0.0;a[0].len()];a.len()];
    for i in 0..a.len() {
        for j in 0..a[0].len() {
            res[i][j] = 1.0 / (1.0 + (-a[i][j]).exp());
        }
    }
    return res;
}

pub fn sigmoid_grad(y: &Vec<Vec<f64>>, grad: &Vec<Vec<f64>>) -> Vec<Vec<f64>>{
    let mut res = vec![vec![0.0;y[0].len()];y.len()];
    for i in 0..y.len() {
        for j in 0..y[0].len() {
            res[i][j] = grad[i][j] * (1.0 - y[i][j])  * y[i][j];
        }
    }
    return res;
}

pub fn soft_max(a: &Vec<Vec<f64>>)->Vec<Vec<f64>>{
    //println!("a {:?}", a);
    let mut res = a.clone();
    for i in 0..res.len() {
        let mut sum = 0.0;
        for j in 0..res[i].len() {
            sum += res[i][j].exp();
            res[i][j] = res[i][j].exp();
        }
        for j in 0..res[i].len() {
            res[i][j] /= sum;
        }
    }
    return res;
}