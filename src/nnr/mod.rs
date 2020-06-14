pub mod matrix;
use rand::Rng;
pub mod activations;
pub mod cnn;

pub struct Layer {
    input: Vec<Vec<f64>>,
    weight: Vec<Vec<f64>>,
    bias: Vec<f64>,
    y: Vec<Vec<f64>>,
    activate: fn(&f64)->f64,
    gradient: fn(&f64, &f64)->f64,
    grad_w: Vec<Vec<f64>>,
    grad_b: Vec<f64>,
}

impl Layer {
    pub fn init (input: usize, output: usize, batch: usize,activate: fn(&f64)->f64, gradient: fn(&f64, &f64)->f64)->Layer{
        let mut rand = rand::thread_rng();
        let mut weight = vec![vec![0.0;output];input];
        let mut grad_w = vec![vec![0.0;output];input];
        let mut input = vec![vec![0.0;input];batch];
        for i in 0..weight.len() {
            for j in 0..weight[i].len() {
                weight[i][j] = (rand.gen::<f64>() - 0.5) * 2.0;
            }
        }
        let mut bias = vec![0.0; output];
        let mut grad_b = vec![0.0; output];
        for bia in &mut bias {
            *bia = (rand.gen::<f64>() - 0.5) * 2.0;
        }
        println!("{:?}, {:?}", weight, bias);
        let mut res = Layer{weight, bias, y:vec![vec![0.0;output];batch], activate, gradient, grad_w,grad_b, input};
        return res;
    }

    /**
    @param x: img2colを利用して変換した画像の2次行列
    **/
    /*pub fn convolution_forward(&mut self, x: &Vec<Vec<f64>>) ->&Vec<Vec<f64>>{
        //col to filtering vector
        let wh = self.weight.len() * self.weight[0].len();

    }*/

    pub fn forward(&mut self,x: &Vec<Vec<f64>>)-> &Vec<Vec<f64>>{
        let mut res = matrix::dot(x, & self.weight);
        self.input = x.clone();
        // bias
        for i in 0..res.len() {
            for j in 0..res[i].len() {
                let act = &self.activate;
                res[i][j] += self.bias[i];
                res[i][j] = act(&res[i][j]);
            }
        }
        self.y = res;
        return &self.y;
    }
    pub fn backward(&mut self, t: &Vec<Vec<f64>>)->Vec<Vec<f64>> {
        let mut res = vec![vec![0.0]];
        let mut delta = vec![vec![0.0;t[0].len()];t.len()];
        for i in 0..delta.len() {
            for j in 0..delta[i].len() {
                let g = self.gradient;
                delta[i][j] = g(&self.y[i][j], &t[i][j]);
            }
        }
        self.grad_w = matrix::dot(&matrix::t(&self.input), &delta);
        // println!("{:?}, {:?}", self.grad_b, delta);
        for i in 0..self.grad_b.len() {
            self.grad_b[i] = matrix::sum(&matrix::t(&delta)[i]);
        }
        res = matrix::dot(&delta, &matrix::t(&self.weight));
        return res;
    }

    pub fn update(&mut self, eta: f64){
        self.weight = matrix::sub(&self.weight, &matrix::scala_prod(eta, &self.grad_w));
        let mut b_col = vec![vec![0.0;self.bias.len()];1];
        b_col[0] = self.bias.clone();
        let mut grad_b_col = vec![vec![0.0;self.grad_b.len()];1];
        grad_b_col[0] = self.bias.clone();
        let b_col = matrix::sub(&b_col, &matrix::scala_prod(eta, &grad_b_col));
        self.bias = b_col[0].clone();
    }
}