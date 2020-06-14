pub fn identity (a: &f64) -> f64{
    return a.clone();
}

pub fn identity_grad (y: &f64, t: &f64) -> f64{
    return y - t;
}

pub fn sigmoid (a: &f64) -> f64{
    return 1.0 / (1.0 + (-a).exp());
}

pub fn sigmoid_grad(y: &f64, grad: &f64) -> f64{
    return *grad * (1.0 - *y)  * *y;
}