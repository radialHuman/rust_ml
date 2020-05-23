mod lib;
// use crate::lib::*;

fn main() {
    let v1 = vec![1., 2., 4., 3., 5.];
    let v2 = vec![1., 3., 3., 2., 5.];
    println!("Mean of {:?} is {}", &v1, lib::mean(&v1));
    println!("variance of {:?} is {}", &v1, lib::variance(&v1));
    println!("Mean of {:?} is {}", &v1, lib::mean(&v2));
    println!("variance of {:?} is {}", &v1, lib::variance(&v2));
    println!(
        "The covariance of {:?} and {:?} is {}",
        &v1,
        &v2,
        lib::covariance(&v1, &v2)
    );
    println!(
        "Coefficient of {:?} and {:?} are b0 = {} and b1 = {}",
        &v1,
        &v2,
        lib::coefficient(&v1, &v2).0,
        lib::coefficient(&v1, &v2).1
    );

    // Simple linear regression
    let to_train_on = vec![
        (1., 2.),
        (2., 3.),
        (4., 5.),
        (3., 5.),
        (6., 8.),
        (7., 8.),
        (9., 10.),
        (1., 2.5),
        (11., 12.),
        (5., 4.),
        (7., 7.),
        (6., 6.),
        (8., 9.),
    ];
    let to_test_on = vec![(10., 11.), (9., 12.), (11., 12.5)];
    let predicted_output = lib::simple_linear_regression_prediction(&to_train_on, &to_test_on);
    let original_output: Vec<_> = to_test_on.iter().map(|a| a.0).collect();
    println!(
        "Predicted is {:?}\nOriginal is {:?}",
        &predicted_output, &original_output
    );

    // reading in a file to have table
    let df = lib::read_csv("./src/dataset_iris.txt".to_string(), 5);
    println!("{:?}", df.values());
    println!(
        "Unique classes are {:?}",
        lib::unique_values(&df["species"].iter().map(|a| &*a).collect()) // converting String to &str as copy is not implemented for String
    );

    // type conversion
    let conversion = lib::convert_and_impute(&df["petal_length"], 0., 999.);
    let floating_petal_length = conversion.0.unwrap();
    let missing_value = conversion.1;
    println!(
        "{:?}\nis now\n{:?}\nwith missing values at\n{:?}",
        df["petal_length"], floating_petal_length, missing_value
    );

    // missing string imputation
    let mut species = df["species"].clone();
    println!(
        "{:?}\nis now\n{:?}",
        &df["species"],
        lib::impute_string(&mut species, "UNKNOWN")
    );
}

/*
OUTPUT
Mean of [1.0, 2.0, 4.0, 3.0, 5.0] is 3
variance of [1.0, 2.0, 4.0, 3.0, 5.0] is 10
Mean of [1.0, 2.0, 4.0, 3.0, 5.0] is 2.8
variance of [1.0, 2.0, 4.0, 3.0, 5.0] is 8.8
The covariance of [1.0, 2.0, 4.0, 3.0, 5.0] and [1.0, 3.0, 3.0, 2.0, 5.0] is 8
Coefficient of [1.0, 2.0, 4.0, 3.0, 5.0] and [1.0, 3.0, 3.0, 2.0, 5.0] are b0 = 0.39999999999999947 and b1 = 0.8
========================================================================================================================================================
RMSE: 2.080646271630258
Predicted is [11.92023172905526, 11.901403743315509, 11.93905971479501]
Original is [10.0, 9.0, 11.0]
========================================================================================================================================================
Reading the file ...
Input row count is 30
The header is ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
[["1.4", "1.4", "1.3", "1.5", "1.4", "", "1.4", "1.5", "1.4", "1.5", "1.5", "1.6", "1.4", "1.1", "1.2", "1.5", "1.3", "1.4", "1.7", "1.5", "1.7", "1.5", "1.0", "1.7", "1.9", "1.6", "1.6", "1.5", "1.4", "1.6", "1.6", "1.5", "1.5", "1.4", "1.5", "1.2", "1.3", "1.5", "1.3", "1.5", "1.3", "1.3", "1.3", "1.6", "1.9", "1.4", "1.6", "1.4", "1.5", "1.4", "4.7", "4.5", "4.9", "4.0", "4.6", "4.5", "4.7", "3.3", "4.6", "3.9", "3.5", "4.2", "4.0", "4.7", "3.6", "4.4", "4.5", "4.1", "4.5", "3.9", "4.8", "4.0", "4.9", "4.7", "4.3", "4.4", "4.8", "5.0", "4.5", "3.5", "3.8", "3.7", "3.9", "5.1", "4.5", "4.5", "4.7", "4.4", "4.1", "4.0", "4.4", "4.6", "4.0", "3.3", "4.2", "4.2", "4.2", "4.3", "3.0", "4.1", "6.0", "5.1", "5.9", "5.6", "5.8", "6.6", "4.5", "6.3", "5.8", "6.1", "5.1", "5.3", "5.5", "5.0", "5.1", "5.3", "5.5", "6.7", "6.9", "5.0", "5.7", "4.9", "6.7", "4.9", "5.7", "6.0", "4.8", "4.9", "5.6", "5.8", "6.1", "6.4", "5.6", "5.1", "5.6", "6.1", "5.6", "5.5", "4.8", "5.4", "5.6", "5.1", "5.1", "5.9", "5.7", "5.2", "5.0", "5.2", "5.4", "5.1"], ["0.2", "0.2", "0.2", "0.2", "0.2", "0.4", "0.3", "0.2", "0.2", "0.1", "0.2", "0.2", "0.1", "0.1", "0.2", "0.4", "0.4", "0.3", "0.3", "0.3", "0.2", "0.4", "0.2", "0.5", "0.2", "0.2", "0.4", "0.2", "0.2", "0.2", "0.2", "0.4", "0.1", "0.2", "0.1", "0.2", "0.2", "0.1", "0.2", "0.2", "0.3", "0.3", "0.2", "0.6", "0.4", "0.3", "0.2", "0.2", "0.2", "0.2", "1.4", "1.5", "1.5", "1.3", "1.5", "1.3", "1.6", "1.0", "1.3", "1.4", "1.0", "1.5", "1.0", "1.4", "1.3", "1.4", "1.5", "1.0", "1.5", "1.1", "1.8", "1.3", "1.5", "1.2", "1.3", "1.4", "1.4", "1.7", "1.5", "1.0", "1.1", "1.0", "1.2", "1.6", "1.5", "1.6", "1.5", "1.3", "1.3", "1.3", "1.2", "1.4", "1.2", "1.0", "1.3", "1.2", "1.3", "1.3", "1.1", "1.3", "2.5", "1.9", "2.1", "1.8", "2.2", "2.1", "1.7", "1.8", "1.8", "2.5", "2.0", "1.9", "2.1", "2.0", "2.4", "2.3", "1.8", "2.2", "2.3", "1.5", "2.3", "2.0", "2.0", "1.8", "2.1", "1.8", "1.8", "1.8", "2.1", "1.6", "1.9", "2.0", "2.2", "1.5", "1.4", "2.3", "2.4", "1.8", "1.8", "2.1", "2.4", "2.3", "1.9", "2.3", "2.5", "2.3", "1.9", "2.0", "2.3", "1.8"], ["5.1", "4.9", "4.7", "4.6", "5.0", "5.4", "4.6", "5.0", "4.4", "4.9", "5.4", "4.8", "4.8", "4.3", "5.8", "5.7", "5.4", "5.1", "5.7", "5.1", "5.4", "5.1", "4.6", "5.1", "4.8", "5.0", "5.0", "5.2", "5.2", "4.7", "4.8", "5.4", "5.2", "5.5", "4.9", "5.0", "5.5", "4.9", "4.4", "5.1", "5.0", "4.5", "4.4", "5.0", "5.1", "4.8", "5.1", "4.6", "5.3", "5.0", "7.0", "6.4", "6.9", "5.5", "6.5", "5.7", "6.3", "4.9", "6.6", "5.2", "5.0", "5.9", "6.0", "6.1", "5.6", "6.7", "5.6", "5.8", "6.2", "5.6", "5.9", "6.1", "6.3", "6.1", "6.4", "6.6", "6.8", "6.7", "6.0", "5.7", "5.5", "5.5", "5.8", "6.0", "5.4", "6.0", "6.7", "6.3", "5.6", "5.5", "5.5", "6.1", "5.8", "5.0", "5.6", "5.7", "5.7", "6.2", "5.1", "5.7", "6.3", "5.8", "7.1", "6.3", "6.5", "7.6", "4.9", "7.3", "6.7", "7.2", "6.5", "6.4", "6.8", "5.7", "5.8", "6.4", "6.5", "7.7", "7.7", "6.0", "6.9", "5.6", "7.7", "6.3", "6.7", "7.2", "6.2", "6.1", "6.4", "7.2", "7.4", "7.9", "6.4", "6.3", "6.1", "7.7", "6.3", "6.4", "6.0", "6.9", "6.7", "6.9", "5.8", "6.8", "6.7", "6.7", "6.3", "6.5", "6.2", "5.9"], ["setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica"], ["3.5", "3.0", "3.2", "3.1", "3.6", "3.9", "3.4", "3.4", "2.9", "3.1", "3.7", "3.4", "3.0", "3.0", "4.0", "4.4", "3.9", "3.5", "3.8", "3.8", "3.4", "3.7", "3.6", "3.3", "3.4", "3.0", "3.4", "3.5", "3.4", "3.2", "3.1", "3.4", "4.1", "4.2", "3.1", "3.2", "3.5", "3.1", "3.0", "3.4", "3.5", "2.3", "3.2", "3.5", "3.8", "3.0", "3.8", "3.2", "3.7", "3.3", "3.2", "3.2", "3.1", "2.3", "2.8", "2.8", "3.3", "2.4", "2.9", "2.7", "2.0", "3.0", "2.2", "2.9", "2.9", "3.1", "3.0", "2.7", "2.2", "2.5", "3.2", "2.8", "2.5", "2.8", "2.9", "3.0", "2.8", "3.0", "2.9", "2.6", "2.4", "2.4", "2.7", "2.7", "3.0", "3.4", "3.1", "2.3", "3.0", "2.5", "2.6", "3.0", "2.6", "2.3", "2.7", "3.0", "2.9", "2.9", "2.5", "2.8", "3.3", "2.7", "3.0", "2.9", "3.0", "3.0", "2.5", "2.9", "2.5", "3.6", "3.2", "2.7", "3.0", "2.5", "2.8", "3.2", "3.0", "3.8", "2.6", "2.2", "3.2", "2.8", "2.8", "2.7", "3.3", "3.2", "2.8", "3.0", "2.8", "3.0", "2.8", "3.8", "2.8", "2.8", "2.6", "3.0", "3.4", "3.1", "3.0", "3.1", "3.1", "3.1", "2.7", "3.2", "3.3", "3.0", "2.5", "3.0", "3.4", "3.0"]]
========================================================================================================================================================
Unique classes are ["setosa", "", "versicolor", "virginica"]
========================================================================================================================================================
Error found in 5th position of the vector
["1.4", "1.4", "1.3", "1.5", "1.4", "", "1.4", "1.5", "1.4", "1.5", "1.5", "1.6", "1.4", "1.1", "1.2", "1.5", "1.3", "1.4", "1.7", "1.5", "1.7", "1.5", "1.0", "1.7", "1.9", "1.6", "1.6", "1.5", "1.4", "1.6", "1.6", "1.5", "1.5", "1.4", "1.5", "1.2", "1.3", "1.5", "1.3", "1.5", "1.3", "1.3", "1.3", "1.6", "1.9", "1.4", "1.6", "1.4", "1.5", "1.4", "4.7", "4.5", "4.9", "4.0", "4.6", "4.5", "4.7", "3.3", "4.6", "3.9", "3.5", "4.2", "4.0", "4.7", "3.6", "4.4", "4.5", "4.1", "4.5", "3.9", "4.8", "4.0", "4.9", "4.7", "4.3", "4.4", "4.8", "5.0", "4.5", "3.5", "3.8", "3.7", "3.9", "5.1", "4.5", "4.5", "4.7", "4.4", "4.1", "4.0", "4.4", "4.6", "4.0", "3.3", "4.2", "4.2", "4.2", "4.3", "3.0", "4.1", "6.0", "5.1", "5.9", "5.6", "5.8", "6.6", "4.5", "6.3", "5.8", "6.1", "5.1", "5.3", "5.5", "5.0", "5.1", "5.3", "5.5", "6.7", "6.9", "5.0", "5.7", "4.9", "6.7", "4.9", "5.7", "6.0", "4.8", "4.9", "5.6", "5.8", "6.1", "6.4", "5.6", "5.1", "5.6", "6.1", "5.6", "5.5", "4.8", "5.4", "5.6", "5.1", "5.1", "5.9", "5.7", "5.2", "5.0", "5.2", "5.4", "5.1"]
is now
[1.4, 1.4, 1.3, 1.5, 1.4, 999.0, 1.4, 1.5, 1.4, 1.5, 1.5, 1.6, 1.4, 1.1, 1.2, 1.5, 1.3, 1.4, 1.7, 1.5, 1.7, 1.5, 1.0, 1.7, 1.9, 1.6, 1.6, 1.5, 1.4, 1.6, 1.6, 1.5, 1.5,
1.4, 1.5, 1.2, 1.3, 1.5, 1.3, 1.5, 1.3, 1.3, 1.3, 1.6, 1.9, 1.4, 1.6, 1.4, 1.5, 1.4, 4.7, 4.5, 4.9, 4.0, 4.6, 4.5, 4.7, 3.3, 4.6, 3.9, 3.5, 4.2, 4.0, 4.7, 3.6, 4.4, 4.5, 4.1, 4.5, 3.9, 4.8, 4.0, 4.9, 4.7, 4.3, 4.4, 4.8, 5.0, 4.5, 3.5, 3.8, 3.7, 3.9, 5.1, 4.5, 4.5, 4.7, 4.4, 4.1, 4.0, 4.4, 4.6, 4.0, 3.3, 4.2, 4.2, 4.2, 4.3, 3.0, 4.1, 6.0, 5.1, 5.9, 5.6, 5.8, 6.6, 4.5, 6.3, 5.8, 6.1, 5.1, 5.3, 5.5, 5.0, 5.1, 5.3, 5.5, 6.7, 6.9, 5.0, 5.7, 4.9, 6.7, 4.9, 5.7, 6.0, 4.8, 4.9, 5.6, 5.8, 6.1, 6.4, 5.6, 5.1, 5.6, 6.1, 5.6, 5.5, 4.8, 5.4, 5.6, 5.1, 5.1, 5.9, 5.7, 5.2, 5.0, 5.2, 5.4, 5.1]
with missing values at
[5]
========================================================================================================================================================
Missing value found in 11th position of the vector
["setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica"]
is now
["setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "UNKNOWN", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica"]
*/
