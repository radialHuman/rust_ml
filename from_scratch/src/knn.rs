/*
Source:
Video:
Book: Trevor Hastie,  Robert Tibshirani, Jerome Friedman - The Elements of  Statistical Learning_  Data Mining, Inference, and Pred
Article: https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/
Library:

Procedure:
1. Prepare data : read, convert, split into 4 parts
2. Calculate distance between each test value and all train row using Euclidean or Chebyshev or cosine or Manhattan or Minkowski or Weighted
3. Distance to be sorted
4. Top K rows will be selected
5. Most frequent class will be selected and assigned
6. Calculate various metrics using predicted and actual values

TODO:
* Skipped minoskwi as it involved extra variable from user

*/
use simple_ml::*;

pub fn function(file_path: String, test_size: f64, target_column: usize, k: usize, method: &str) {
    /*
    method : Euclidean or Chebyshev or cosine or Manhattan or Minkowski or Weighted
    */
    // read a csv file
    let (columns, values) = read_csv(file_path); // output is row wise

    // converting vector of string to vector of f64s
    let random_data = BLR::float_randomize(&values); // blr needs to be removed

    // splitting it into train and test as per test percentage passed as parameter to get scores
    let (x_train, y_train, x_test, y_test) =
        BLR::preprocess_train_test_split(&random_data, test_size, target_column, ""); // blr needs to be removed

    // now to the main part
    // since it is row wise, conversion
    let train_rows = columns_to_rows_conversion(&x_train);
    let test_rows = columns_to_rows_conversion(&x_test);
    shape("train Rows:", &train_rows);
    shape("test Rows:", &test_rows);
    // println!("{:?}", y_train.len());

    // predicting values
    let predcited = predict(&train_rows, &y_train, &test_rows, method, k);
    println!("Metrics");
    BLR::confuse_me(
        &predcited.iter().map(|a| *a as f64).collect::<Vec<f64>>(),
        &y_test,
    ); // blr needs to be removed
}

pub fn predict(
    train_rows: &Vec<Vec<f64>>,
    train_values: &Vec<f64>,
    test_rows: &Vec<Vec<f64>>,
    method: &str,
    k: usize,
) -> Vec<i32> {
    match method {
        "e" => println!("\n\nCalculating KNN using euclidean distance ..."),
        "ma" => println!("\n\nCalculating KNN using manhattan distance ..."),
        "co" => println!("\n\nCalculating KNN using cosine distance ..."),
        "ch" => println!("\n\nCalculating KNN using chebyshev distance ..."),
        _ => panic!("The method has to be either 'e' or 'ma' or 'co' or 'ch'"),
    };
    let mut predcited = vec![];
    for j in test_rows.iter() {
        let mut class_found = vec![];
        for (n, i) in train_rows.iter().enumerate() {
            // println!("{:?},{:?},{:?}", j, n, i);
            match method {
                "e" => class_found.push((distance_euclidean(i, &j), train_values[n])),
                "ma" => class_found.push((distance_manhattan(i, &j), train_values[n])),
                "co" => class_found.push((distance_cosine(i, &j), train_values[n])),
                "ch" => class_found.push((distance_chebyshev(i, &j), train_values[n])),
                _ => (), // cant happen as it would panic in the previous match
            };
        }
        // sorting acsending the vector by first value of tuple
        class_found.sort_by(|(a, _), (c, _)| (*a).partial_cmp(c).unwrap());
        let k_nearest = class_found[..k].to_vec();
        let knn: Vec<f64> = k_nearest.iter().map(|a| a.1).collect();
        // converting classes to int and classifying
        let nearness = value_counts(&knn.iter().map(|a| *a as i32).collect());
        // finding the closest
        predcited.push(*nearness.iter().next_back().unwrap().0)
    }
    predcited
}

pub fn distance_euclidean(row1: &Vec<f64>, row2: &Vec<f64>) -> f64 {
    // sqrt(sum((row1-row2)**2))

    let distance = row1
        .iter()
        .zip(row2.iter())
        .map(|(a, b)| (*a - *b) * (*a - *b))
        .collect::<Vec<f64>>();
    distance.iter().fold(0., |a, b| a + b).sqrt()
}

pub fn distance_manhattan(row1: &Vec<f64>, row2: &Vec<f64>) -> f64 {
    // sum(|row1-row2|)

    let distance = row1
        .iter()
        .zip(row2.iter())
        .map(|(a, b)| (*a - *b).abs())
        .collect::<Vec<f64>>();
    distance.iter().fold(0., |a, b| a + b)
}

pub fn distance_cosine(row1: &Vec<f64>, row2: &Vec<f64>) -> f64 {
    // 1- (a.b)/(|a||b|)

    let numerator = row1
        .iter()
        .zip(row2.iter())
        .map(|(a, b)| (*a * *b))
        .collect::<Vec<f64>>()
        .iter()
        .fold(0., |a, b| a + b);
    let denominator = (row1
        .iter()
        .map(|a| a * a)
        .collect::<Vec<f64>>()
        .iter()
        .fold(0., |a, b| a + b)
        .sqrt())
        * (row2
            .iter()
            .map(|a| a * a)
            .collect::<Vec<f64>>()
            .iter()
            .fold(0., |a, b| a + b)
            .sqrt());
    1. - numerator / denominator
}

pub fn distance_chebyshev(row1: &Vec<f64>, row2: &Vec<f64>) -> f64 {
    // max(|row1-row2|)
    let distance = row1
        .iter()
        .zip(row2.iter())
        .map(|(a, b)| (*a - *b).abs())
        .collect::<Vec<f64>>();
    distance.iter().cloned().fold(0. / 0., f64::max)
}

/*
RUST OUTPUT

Reading the file ...
Number of rows = 1371
1098x5 becomes
5x1098
274x5 becomes
5x274
Using the actual values without preprocessing unless 's' or 'm' is passed
4x1098 becomes
1098x4
4x274 becomes
274x4
"train Rows:" : 1098x4
"test Rows:" : 274x4


Calculating KNN using euclidean distance ...
Metrics
|------------------------|
|  143.0    |   5.0
|------------------------|
|  0.0    |   126.0
|------------------------|
Accuracy : 0.982
Precision : 0.966
Recall (sensitivity) : 1.000
Specificity: 0.962
F1 : 2.000


Reading the file ...
Number of rows = 1371
1098x5 becomes
5x1098
274x5 becomes
5x274
Using the actual values without preprocessing unless 's' or 'm' is passed
4x1098 becomes
1098x4
4x274 becomes
274x4
"train Rows:" : 1098x4
"test Rows:" : 274x4


Calculating KNN using manhattan distance ...
Metrics
|------------------------|
|  159.0    |   2.0
|------------------------|
|  0.0    |   113.0
|------------------------|
Accuracy : 0.993
Precision : 0.988
Recall (sensitivity) : 1.000
Specificity: 0.983
F1 : 2.000


Reading the file ...
Number of rows = 1371
1098x5 becomes
5x1098
274x5 becomes
5x274
Using the actual values without preprocessing unless 's' or 'm' is passed
4x1098 becomes
1098x4
4x274 becomes
274x4
"train Rows:" : 1098x4
"test Rows:" : 274x4


Calculating KNN using cosine distance ...
Metrics
|------------------------|
|  149.0    |   3.0
|------------------------|
|  0.0    |   122.0
|------------------------|
Accuracy : 0.989
Precision : 0.980
Recall (sensitivity) : 1.000
Specificity: 0.976
F1 : 2.000


Reading the file ...
Number of rows = 1371
1098x5 becomes
5x1098
274x5 becomes
5x274
Using the actual values without preprocessing unless 's' or 'm' is passed
4x1098 becomes
1098x4
4x274 becomes
274x4
"train Rows:" : 1098x4
"test Rows:" : 274x4


Calculating KNN using chebyshev distance ...
Metrics
|------------------------|
|  153.0    |   6.0
|------------------------|
|  0.0    |   115.0
|------------------------|
Accuracy : 0.978
Precision : 0.962
Recall (sensitivity) : 1.000
Specificity: 0.950
F1 : 2.000
*/
