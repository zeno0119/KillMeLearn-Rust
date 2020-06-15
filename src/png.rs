use png;
use std::fs::File;

pub fn parse_png(str: &str) -> Vec<Vec<Vec<f64>>> {
    let mut d = png::Decoder::new(File::open(str).unwrap());
    let mut r = d.read_info().ok().unwrap().1;
    let mut res: Vec<Vec<Vec<f64>>> = Vec::new();
    {
        loop {
            let mut row = r.next_row();
            let data = row.ok();
            if data.is_none() { break; }
            let data = data.unwrap();
            if data.is_none() { break; }
            let data = data.unwrap();

            let mut el: Vec<Vec<f64>> = Vec::new();
            {
                let mut datum: usize = 0;

                while datum + 2 < data.len() {
                    let mut ele: Vec<f64> = Vec::new();
                    ele.push((data[datum] as f64 - 128.0) / 128.0);
                    ele.push((data[datum + 1] as f64 - 128.0) / 128.0);
                    ele.push((data[datum + 2] as f64 - 128.0) / 128.0);
                    el.push(ele);
                    datum += 3;
                }
            }
            res.push(el);
        }
    };
    return res;
}

/**
@param img: Vec<Vec<Vec<u8>: RGB>: column>: row
**/

pub fn img2col(img: Vec<Vec<Vec<f64>>>) -> Vec<Vec<f64>> {
    let mut res: Vec<Vec<f64>> = Vec::new();
    for el in img {
        let mut r: Vec<f64> = Vec::new();
        let mut g: Vec<f64> = Vec::new();
        let mut b: Vec<f64> = Vec::new();
        for column in el {
            r.push(column[0]);
            g.push(column[1]);
            b.push(column[2]);
        }
        let mut col: Vec<f64> = Vec::new();
        col.append(&mut r);
        col.append(&mut g);
        col.append(&mut b);
        res.push(col);
    }
    return res;
}

pub fn img2vec(img: Vec<Vec<Vec<f64>>>)->Vec<f64>{
    let mut res:Vec<f64> = Vec::new();
    for row in img {
        for col in row {
            for el in col {
                res.push(el);
            }
        }
    }
    return res;
}