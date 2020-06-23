/*
Aim: To replicate codes from established libraries, books, videos and other sources to get similar results and leanr new things along the way

Procedure:
* Understand the concept from various sources
* Understand the codes and its functions
* Rewrite to get the same result
* Detailed documentation

*/

use simple_ml::*;

pub fn function(
    file_path: String,
    test_size: f64,
    target_column: usize,
    learning_rate: f64,
    iter_count: u32,
    binary_threshold: f64,
) {
    /*
        Source:
        Video:
        Book: Trevor Hastie,  Robert Tibshirani, Jerome Friedman - The Elements of  Statistical Learning_  Data Mining, Inference, and Pred
        Article: https://towardsdatascience.com/scale-standardize-or-normalize-with-scikit-learn-6ccc7d176a02
        Library:
    */

    // read a csv file
    let (columns, values) = read_csv(file_path); // output is row wise

    // converting vector of string to vector of f64s
    let random_data = float_randomize(&values);

    // splitting it into train and test as per test percentage passed as parameter to get scores
    let (x_train, y_train, x_test, y_test) =
        preprocess_train_test_split(&random_data, test_size, target_column, "");

    shape("Training features", &x_train);
    shape("Test features", &x_test);
    println!("Training target: {:?}", &y_train.len());
    println!("Test target: {:?}", &y_test.len());

    // now to the main part
    let length = x_train[0].len();
    let feature_count = x_train.len();
    // let class_count = (unique_values(&y_test).len() + unique_values(&y_test).len()) / 2;
    let intercept = vec![vec![1.; length]];
    let new_x_train = [&intercept[..], &x_train[..]].concat();
    let mut coefficients = vec![0.; feature_count + 1];

    let mut cost = vec![];
    print!("Reducing loss ...");
    for _ in 0..iter_count {
        let s = sigmoid(&new_x_train, &coefficients);
        cost.push(log_loss(&s, &y_train));
        let gd = gradient_descent(&new_x_train, &s, &y_train);
        coefficients = change_in_loss(&coefficients, learning_rate, &gd);
    }
    // println!("The intercept is : {:?}", coefficients[0]);
    // println!(
    //     "The coefficients are : {:?}",
    //     columns
    //         .iter()
    //         .zip(coefficients[1..].to_vec())
    //         .collect::<Vec<(&String, f64)>>()
    // );
    let predicted = predict(&x_test, &coefficients, binary_threshold);
    confuse_me(&predicted, &y_test);
}

pub fn confuse_me(predicted: &Vec<f64>, actual: &Vec<f64>) {
    // https://medium.com/@MohammedS/performance-metrics-for-classification-problems-in-machine-learning-part-i-b085d432082b
    let mut tp = 0.; // class_one_is_class_one
    let mut fp = 0.; // class_one_is_class_two(Type 1)
    let mut fng = 0.; // class_two_is_class_one (Type 1)
    let mut tng = 0.; // class_two_is_class_two

    for (i, j) in actual
        .iter()
        .zip(predicted.iter())
        .collect::<Vec<(&f64, &f64)>>()
        .iter()
    {
        if **i == 0.0 && **j == 0.0 {
            tp += 1.;
        }
        if **i == 1.0 && **j == 1.0 {
            tng += 1.;
        }
        if **i == 0.0 && **j == 1.0 {
            fp += 1.;
        }
        if **i == 1.0 && **j == 0.0 {
            fng += 1.;
        }
    }
    println!("|------------------------|");
    println!("|  {:?}    |   {:?}", tp, fp);
    println!("|------------------------|");
    println!("|  {:?}    |   {:?}", fng, tng);
    println!("|------------------------|");
    println!("Accuracy : {:.3}", (tp + tng) / (tp + fp + fng + tng));
    println!("Precision : {:.3}", (tp) / (tp + fp));
    let precision: f64 = (tp) / (tp + fp);
    println!("Recall (sensitivity) : {:.3}", (tp) / (tp + fng));
    let recall: f64 = (tp) / (tp + fng);
    println!("Specificity: {:.3}", (tng) / (fp + tng));
    println!(
        "F1 : {:.3}\n\n",
        (2. * precision * recall) / (precision * recall)
    );
}

pub fn predict(test_features: &Vec<Vec<f64>>, weights: &Vec<f64>, threshold: f64) -> Vec<f64> {
    let length = test_features[0].len();
    let intercept = vec![vec![1.; length]];
    let new_x_test = [&intercept[..], &test_features[..]].concat();
    let mut pred = sigmoid(&new_x_test, weights);
    pred.iter()
        .map(|a| if *a > threshold { 1. } else { 0. })
        .collect()
}

pub fn change_in_loss(coeff: &Vec<f64>, lr: f64, gd: &Vec<f64>) -> Vec<f64> {
    print!(".");
    if coeff.len() == gd.len() {
        element_wise_operation(coeff, &gd.iter().map(|a| a * lr).collect(), "add")
    } else {
        panic!("The dimensions do not match")
    }
}

pub fn gradient_descent(train: &Vec<Vec<f64>>, sigmoid: &Vec<f64>, y_train: &Vec<f64>) -> Vec<f64> {
    let part2 = element_wise_operation(sigmoid, y_train, "sub");
    let numerator = matrix_vector_product_f(train, &part2);
    numerator
        .iter()
        .map(|a| *a / (y_train.len() as f64))
        .collect()
}

pub fn log_loss(sigmoid: &Vec<f64>, y_train: &Vec<f64>) -> f64 {
    let part11 = sigmoid.iter().map(|a| a.log(1.0_f64.exp())).collect();
    let part12 = y_train.iter().map(|a| a * -1.).collect();
    let part21 = sigmoid
        .iter()
        .map(|a| (1. - a).log(1.0_f64.exp()))
        .collect();
    let part22 = y_train.iter().map(|a| 1. - a).collect();
    let part1 = element_wise_operation(&part11, &part12, "mul");
    let part2 = element_wise_operation(&part21, &part22, "mul");
    mean(&element_wise_operation(&part1, &part2, "sub"))
}

pub fn sigmoid(train: &Vec<Vec<f64>>, coeff: &Vec<f64>) -> Vec<f64> {
    let z = matrix_vector_product_f(&transpose(train), coeff);
    z.iter().map(|a| 1. / (1. + a.exp())).collect()
}

pub fn float_randomize(matrix: &Vec<Vec<String>>) -> Vec<Vec<f64>> {
    matrix
        .iter()
        .map(|a| {
            a.iter()
                .map(|b| (*b).replace("\r", "").parse::<f64>().unwrap())
                .collect::<Vec<f64>>()
        })
        .collect::<Vec<Vec<f64>>>()
}

pub fn preprocess_train_test_split(
    matrix: &Vec<Vec<f64>>,
    test_percentage: f64,
    target_column: usize,
    preprocess: &str,
) -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>, Vec<f64>) {
    /*
    preprocess : "s" : standardize, "m" : minmaxscaler, "_" : no change
    */

    let (train_data, test_data) = train_test_split_f(matrix, test_percentage);
    // println!("Training size: {:?}", train_data.len());
    // println!("Test size: {:?}", test_data.len());

    // converting rows to vector of columns of f64s
    let mut actual_train = row_to_columns_conversion(&train_data);
    let mut actual_test = row_to_columns_conversion(&test_data);

    match preprocess {
        "s" => {
            actual_train = actual_train
                .iter()
                .map(|a| standardize_vector_f(a))
                .collect::<Vec<Vec<f64>>>();
            actual_test = actual_test
                .iter()
                .map(|a| standardize_vector_f(a))
                .collect::<Vec<Vec<f64>>>();
        }
        "m" => {
            actual_train = actual_train
                .iter()
                .map(|a| min_max_scaler(a))
                .collect::<Vec<Vec<f64>>>();
            actual_test = actual_test
                .iter()
                .map(|a| min_max_scaler(a))
                .collect::<Vec<Vec<f64>>>();
        }

        _ => println!("Using the actual values without preprocessing unless 's' or 'm' is passed"),
    };

    (
        drop_column(&actual_train, target_column),
        actual_train[target_column - 1].clone(),
        drop_column(&actual_test, target_column),
        actual_test[target_column - 1].clone(),
    )
}

pub fn standardize_vector_f(list: &Vec<f64>) -> Vec<f64> {
    /*
    Preserves the shape of the original distribution. Doesn't
    reduce the importance of outliers. Least disruptive to the
    information in the original data. Default range for
    MinMaxScaler is O to 1.
        */
    list.iter()
        .map(|a| (*a - mean(list)) / std_dev(list))
        .collect()
}

pub fn min_max_scaler(list: &Vec<f64>) -> Vec<f64> {
    let (minimum, maximum) = min_max_f(&list);
    let range: f64 = maximum - minimum;
    list.iter().map(|a| 1. - ((maximum - a) / range)).collect()
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
"Training features" : 4x1098
"Test features" : 4x274
Training target: 1098
Test target: 274
Reducing loss ...........................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................The intercept is : -2.12921768845986
|------------------------|
|  148.0    |   1.0
|------------------------|
|  3.0    |   122.0
|------------------------|
Accuracy : 0.985
Precision : 0.993
Recall (sensitivity) : 0.980
Specificity: 0.992
F1 : 2.000
*/
