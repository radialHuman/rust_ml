// https://machinelearningmastery.com/implement-simple-linear-regression-scratch-python/

pub fn mean<T>(list: &Vec<T>) -> f64
where
    T: std::iter::Sum<T>
        + std::ops::Div<Output = T>
        + Copy
        + std::str::FromStr
        + std::string::ToString
        + std::ops::Add<T, Output = T>
        + std::fmt::Debug
        + std::fmt::Display
        + std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    let zero: T = "0".parse().unwrap();
    let len_str = list.len().to_string();
    let length: T = len_str.parse().unwrap();
    (list.iter().fold(zero, |acc, x| acc + *x) / length)
        .to_string()
        .parse()
        .unwrap()
}

// pub fn median<T>(list: &Vec<T>) -> f64
// where
//     T: Copy
//         + std::cmp::PartialOrd
//         + std::ops::Rem<T, Output = T>
//         + std::ops::Div<T, Output = T>
//         + std::ops::Add<T, Output = T>
//         + std::ops::Sub<T, Output = T>
//         + std::string::ToString
//         + std::str::FromStr,
//     <T as std::str::FromStr>::Err: std::fmt::Debug,
// {
//     let mut output = list.clone();
//     // println!("Issue found in 1");
//     output.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
//     // println!("Issue found in 2");
//     let zero: T = "0".parse().unwrap();
//     // println!("Issue found in 3");
//     let half = 0.5;
//     // println!("Issue found in 4");
//     let two: T = "2".parse().unwrap();
//     // println!("Issue found in 5");
//     let len_str: T = output.len().to_string().parse().unwrap();
//     if len_str % two == zero {
//         let middle_of_list: usize = (len_str / two).to_string().parse().unwrap();
//         if output[middle_of_list] == output[middle_of_list + 1] {
//             println!("Issue found in 6");
//             output[middle_of_list].to_string().parse().unwrap()
//         } else {
//             println!("Issue found in 7");
//             (output[middle_of_list - 1]).to_string().parse().unwrap()
//         }
//     } else {
//         let middle_of_list: usize = ((len_str / two) - half.to_string().parse().unwrap())
//             .to_string()
//             .parse()
//             .unwrap();
//         println!("Issue found in 8");
//         output[middle_of_list].to_string().parse().unwrap()
//     }
// }

pub fn variance<T>(list: &Vec<T>) -> f64
where
    T: std::iter::Sum<T>
        + std::ops::Div<Output = T>
        + std::marker::Copy
        + std::fmt::Display
        + std::ops::Sub<T, Output = T>
        + std::ops::Add<T, Output = T>
        + std::ops::Mul<T, Output = T>
        + std::fmt::Debug
        + std::string::ToString
        + std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    let zero: T = "0".parse().unwrap();
    let mu = mean(list);
    let _len_str: T = list.len().to_string().parse().unwrap(); // is division is required
    let output: Vec<_> = list
        .iter()
        .map(|x| (*x - mu.to_string().parse().unwrap()) * (*x - mu.to_string().parse().unwrap()))
        .collect();
    // output
    let variance = output.iter().fold(zero, |a, b| a + *b); // / len_str;
    variance.to_string().parse().unwrap()
}

pub fn covariance<T>(list1: &Vec<T>, list2: &Vec<T>) -> f64
where
    T: std::iter::Sum<T>
        + std::ops::Div<Output = T>
        + std::fmt::Debug
        + std::fmt::Display
        + std::ops::Add
        + std::marker::Copy
        + std::ops::Add<T, Output = T>
        + std::ops::Sub<T, Output = T>
        + std::ops::Mul<T, Output = T>
        + std::string::ToString
        + std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    let mu1 = mean(list1);
    let mu2 = mean(list2);
    let zero: T = "0".parse().unwrap();
    let _len_str: T = list1.len().to_string().parse().unwrap(); // is division is required
    let tupled: Vec<_> = list1.iter().zip(list2).collect();
    let output = tupled.iter().fold(zero, |a, b| {
        a + ((*b.0 - mu1.to_string().parse().unwrap()) * (*b.1 - mu2.to_string().parse().unwrap()))
    });
    output.to_string().parse().unwrap() // / len_str
}

pub fn coefficient<T>(list1: &Vec<T>, list2: &Vec<T>) -> (f64, f64)
where
    T: std::iter::Sum<T>
        + std::ops::Div<Output = T>
        + std::fmt::Debug
        + std::fmt::Display
        + std::ops::Add
        + std::marker::Copy
        + std::ops::Add<T, Output = T>
        + std::ops::Sub<T, Output = T>
        + std::ops::Mul<T, Output = T>
        + std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    let b1 = covariance(list1, list2) / variance(list1);
    let b0 = mean(list2) - (b1 * mean(list1));
    (b0.to_string().parse().unwrap(), b1)
}

pub fn simple_linear_regression_prediction<T>(train: &Vec<(T, T)>, test: &Vec<(T, T)>) -> Vec<T>
where
    T: std::iter::Sum<T>
        + std::ops::Div<Output = T>
        + std::fmt::Debug
        + std::fmt::Display
        + std::ops::Add
        + std::marker::Copy
        + std::ops::Add<T, Output = T>
        + std::ops::Sub<T, Output = T>
        + std::ops::Mul<T, Output = T>
        + std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    let train_features = &train.iter().map(|a| a.0).collect();
    let test_features = &test.iter().map(|a| a.1).collect();
    let (offset, slope) = coefficient(train_features, test_features);
    let b0: T = offset.to_string().parse().unwrap();
    let b1: T = slope.to_string().parse().unwrap();
    let predicted_output = test.iter().map(|a| b0 + b1 * a.0).collect();
    let original_output: Vec<_> = test.iter().map(|a| a.0).collect();
    println!(
        "RMSE: {:?}",
        root_mean_square(&predicted_output, &original_output)
    );
    predicted_output
}

pub fn root_mean_square<T>(list1: &Vec<T>, list2: &Vec<T>) -> f64
where
    T: std::ops::Sub<T, Output = T>
        + Copy
        + std::ops::Mul<T, Output = T>
        + std::ops::Add<T, Output = T>
        + std::ops::Div<Output = T>
        + std::string::ToString
        + std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    println!("========================================================================================================================================================");
    let zero: T = "0".parse().unwrap();
    let tupled: Vec<_> = list1.iter().zip(list2).collect();
    let length: T = list1.len().to_string().parse().unwrap();
    let mean_square_error = tupled
        .iter()
        .fold(zero, |b, a| b + ((*a.1 - *a.0) * (*a.1 - *a.0)))
        / length;
    let mse: f64 = mean_square_error.to_string().parse().unwrap();
    mse.powf(0.5)
}

// reading in files for multi column operations
use std::collections::HashMap;
use std::fs;
pub fn read_csv(path: String, columns: i32) -> HashMap<String, Vec<String>> {
    println!("========================================================================================================================================================");
    println!("Reading the file ...");
    let file = fs::read_to_string(&path).unwrap();
    // making vec (rows)
    let x_vector: Vec<_> = file.split("\r\n").collect();
    let rows: i32 = (x_vector.len() - 1) as i32 / columns;
    println!("Input row count is {:?}", rows);
    // making vec of vec (table)
    let table: Vec<Vec<&str>> = x_vector.iter().map(|a| a.split(",").collect()).collect();
    println!("The header is {:?}", &table[0]);
    // making a dictionary
    let mut table_hashmap: HashMap<String, Vec<String>> = HashMap::new();
    for (n, i) in table[0].iter().enumerate() {
        let mut vector = vec![];
        for j in table[1..].iter() {
            vector.push(j[n]);
        }
        table_hashmap.insert(
            i.to_string(),
            vector.iter().map(|a| a.to_string()).collect(),
        );
    }
    table_hashmap
}
