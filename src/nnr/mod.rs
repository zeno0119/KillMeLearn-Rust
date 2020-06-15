pub mod matrix;
use rand::Rng;
pub mod activations;
pub mod cnn;

pub struct Layer {
    input: Vec<Vec<f64>>,
    weight: Vec<Vec<f64>>,
    bias: Vec<f64>,
    y: Vec<Vec<f64>>,
    activate: fn(&Vec<Vec<f64>>)->Vec<Vec<f64>>,
    gradient: fn(&Vec<Vec<f64>>, &Vec<Vec<f64>>)->Vec<Vec<f64>>,
    grad_w: Vec<Vec<f64>>,
    grad_b: Vec<f64>,
}

impl Layer {
    pub fn init (input: usize, output: usize, batch: usize,activate: fn(&Vec<Vec<f64>>)->Vec<Vec<f64>>, gradient: fn(&Vec<Vec<f64>>, &Vec<Vec<f64>>)->Vec<Vec<f64>>)->Layer{
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
        // println!("{:?}, {:?}", weight, bias);
        let mut res = Layer{weight, bias, y:vec![vec![0.0;output];batch], activate, gradient, grad_w,grad_b, input};
        return res;
    }

    pub fn forward(&mut self,x: &Vec<Vec<f64>>)-> &Vec<Vec<f64>>{
        let mut res = matrix::dot(x, & self.weight);
        self.input = x.clone();
        let act = &self.activate;
        // bias
        for i in 0..res.len() {
            for j in 0..res[i].len() {
                res[i][j] += self.bias[i];
            }
        }
        res = act(&res);
        self.y = res;
        return &self.y;
    }
    pub fn backward(&mut self, t: &Vec<Vec<f64>>)->Vec<Vec<f64>> {
        let mut res = vec![vec![0.0]];
        let mut delta = vec![vec![0.0;t[0].len()];t.len()];
        for i in 0..delta.len() {
            for j in 0..delta[i].len() {
                let g = self.gradient;
                delta = g(&self.y, &t);
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

pub struct ConvLayer {
    input: Vec<Vec<f64>>,
    weight: Vec<Vec<f64>>,
    bias: Vec<f64>,
    y: Vec<Vec<f64>>,
    activate: fn(&f64) -> f64,
    gradient: fn(&f64, &f64) -> f64,
    grad_w: Vec<Vec<f64>>,
    grad_b: Vec<f64>,
    x_ch: usize,
    x_h: usize,
    x_w:usize,
    n_flt:usize,
    flt_h: usize,
    flt_w: usize,
    y_ch: usize,
    y_h:usize,
    y_w:usize,
    stride: usize,
    pad: usize
}
/*
impl ConvLayer {
    pub fn init (x_ch: usize, x_h: usize, x_w:usize, n_flt:usize, flt_h: usize, flt_w: usize, stride: usize, pad: usize, activation: fn (&f64)->f64) -> ConvLayer{
        let mut r = rand::thread_rng();

        let mut y_h = (x_h - flt_h + 2 * pad) / stride + 1;
        let mut y_w = (x_w - flt_w + 2 * pad) / stride + 1;
        let mut y_ch = n_flt;
        let mut weight = vec![vec![0.0;x_ch * flt_h * flt_w];n_flt];
        for i in 0..weight.len() {
            for j in 0..weight[i].len() {
                weight[i][j] = r.gen::<f64>();
            }
        }
        let mut bias = vec![0.0; n_flt];
        for bia in &mut bias {
            *bia = r.gen();
        }

        /*let mut res = ConvLayer{weight, bias, n_flt, flt_w, flt_h, stride, pad, x_ch, x_h, x_w,
            grad_w: vec![vec![0.0;x_ch * flt_h * flt_w];n_flt],
            grad_b: vec![0.0;n_flt],
            activate: activation,
            y:
        };*/
    }

    pub fn forward(&mut self, x: Vec<Vec<Vec<f64>>>)-> &Vec<Vec<f64>>{
        self.input = cnn::img2col(x, self.flt_h, self.flt_w, self.stride, self.pad);
        self.y = matrix::dot(self.w, &self.input);
        let act = self.activate;
        for i in 0..self.y.len() {
            for j in 0..self.y[i].len() {
                self.y[i][j] += self.bias[i];
                self.y[i][j] = act(&self.y[i][j]);
            }
        }
        return &self.y;
    }

    pub fn backward(&mut self, grad_y: Vec<Vec<f64>>){
        let mut delta = vec![vec![0.0;grad_y[0].len()];grad_y.len()];
        for i in 0..delta.len() {
            for j in 0..delta[i].len() {
                delta[i][j] = {
                    if self.y[i][j] > 0.0 {
                        grad_y[i][j]
                    }else {
                        0
                    }
                };
            }
        }
    }
}*/