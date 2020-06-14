pub fn dot(a: & Vec<Vec<f64>>, b: & Vec<Vec<f64>>)-> Vec<Vec<f64>>{
    let mut res :Vec<Vec<f64>> = vec![vec![0.0;b[0].len()];a.len()];
    // println!("{:?}, {:?}", a,b);
    // println!("{:?}, {:?}, {:?}", a.len(), b.len(), b[0].len());
    for i in 0..a.len() {
        for j in 0..b[0].len() {
            for k in 0..b.len() {
                res[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return res;
}

pub fn t(a: & Vec<Vec<f64>>)-> Vec<Vec<f64>>{
    let mut res = vec![vec![0.0;a.len()];a[0].len()];
    for i in 0..a.len() {
        for j in 0..a[i].len() {
            res[j][i] = a[i][j];
        }
    }
    return res;
}

pub fn sub(a:& Vec<Vec<f64>>,b:& Vec<Vec<f64>>)-> Vec<Vec<f64>>{
    let mut res = vec![vec![0.0;a[0].len()];a.len()];
    for i in 0..a.len() {
        for j in 0..a[i].len() {
            res[i][j] = a[i][j] - b[i][j];
        }
    }
    return res;
}

pub fn sum(a: &Vec<f64>)->f64{
    let mut res = 0.0;
    for el in a {
        res += *el;
    }
    return res;
}

pub fn scala_prod(scala: f64, mat: &Vec<Vec<f64>>)->Vec<Vec<f64>>{
    let mut res = vec![vec![0.0;mat[0].len()];mat.len()];
    for i in 0..mat.len() {
        for j in 0..mat[i].len() {
            res[i][j] = scala * mat[i][j];
        }
    }
    return res;
}