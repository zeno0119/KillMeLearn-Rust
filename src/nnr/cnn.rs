pub fn img2col(img: Vec<Vec<Vec<f64>>>, flt_h: usize, flt_w: usize, stride: usize, padding: usize) -> Vec<Vec<f64>> {
    //padding
    let y_h = (img.len() + padding * 2 - flt_h) / stride + 1;
    let y_w = (img[0].len() + padding * 2 - flt_w) / stride + 1;
    let batch: usize = 1;
    let channel: usize = 3;

    let mut padded = vec![vec![vec![0.0; 3]; img[0].len() + padding * 2]; img.len() + padding * 2];
    for i in 0..img.len() {
        for j in 0..img[i].len() {
            padded[i + padding][j + padding] = img[i][j].clone();
        }
    }

    println!("{:?}", padded);
    let mut res = vec![vec![0.0; y_h * y_w]; flt_h * flt_w * channel];

    for channel in 0..channel {
        let mut count: usize = 0;
        let mut h: usize = 0;
        while h < padded.len() {
            let mut w: usize = 0;
            while w < padded[h].len() {
                if h + flt_h > padded.len() { break; }
                if w + flt_w > padded[h].len() { break; }
                for f_h in 0..flt_h {
                    for f_w in 0..flt_w {
                        println!("{:?}", padded[h + f_h][w + f_w][channel]);
                        res[channel * flt_h * flt_w + f_h * flt_w + f_w][count] = padded[h + f_h][w + f_w][channel].clone();
                    }
                }
                w += stride;
                count += 1;
            }
            h += stride;
        }
    }

    return res;
}

pub fn col2img(col: Vec<Vec<f64>>, y_h: usize, y_w: usize, flt_h: usize, flt_w: usize, padding: usize, stride: usize) -> Vec<Vec<Vec<f64>>> {
    let channel = 3;

    let mut h = 0;
    let mut w = 0; //畳み込みの始点座標

    let mut res = vec![vec![vec![0.0; channel]; y_w + padding * 2]; y_h + padding * 2];

    for j in 0..col[0].len() {
        if w + flt_w - 1 >= y_w + padding * 2{
            h += stride;
            w = 0;
        }
        for i in 0..col.len() {
            let ch = i % (flt_w * flt_h);
            let chan = i / (flt_w * flt_h);
            // println!("{:?}, {:?}, {:?}",ch / flt_w + h,ch % flt_w + w,i / (flt_h * flt_w));
            // println!("{:?}", col[i][j]);
            res[ch / flt_w + h][ch % flt_w + w][i / (flt_h * flt_w)] += col[i][j];
        }
        w += stride;
    }
    let mut padded = vec![vec![vec![0.0; channel]; y_w]; y_h];
    for h in 0..y_h {
        for w in 0..y_w {
            padded[h][w] = res[h + padding][w + padding].clone();
        }
    }
    return padded;
}