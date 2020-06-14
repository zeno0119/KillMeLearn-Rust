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

pub fn col2img(col: Vec<Vec<f64>>)->Vec<Vec<Vec<f64>>>{
    //TODO 入力された画像の縦横のサイズを入力に加えて実装しましょう。
    let channel = 3;
    let y_h = (col.len()/channel + 2 * padding - flt_h)/ stride + 1;
    let y_w = (col[0].len() + 2 * padding - flt_w)/ stride + 1;

    let mut res = vec![vec![vec![0.0;channel];];];

    return res;
}