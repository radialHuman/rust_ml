mod lib;
use crate::lib::*;

fn main() {
    let v1 = vec![1., 2., 4., 3., 5.];
    let v2 = vec![1., 3., 3., 2., 5.];
    println!("Mean of {:?} is {}", &v1, mean(&v1));
    println!("variance of {:?} is {}", &v1, variance(&v1));
    println!("Mean of {:?} is {}", &v1, mean(&v2));
    println!("variance of {:?} is {}", &v1, variance(&v2));
    println!(
        "The covariance of {:?} and {:?} is {}",
        &v1,
        &v2,
        covariance(&v1, &v2)
    );
    println!(
        "Coefficient of {:?} and {:?} are b0 = {} and b1 = {}",
        &v1,
        &v2,
        coefficient(&v1, &v2).0,
        coefficient(&v1, &v2).1
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
    let predicted_output = simple_linear_regression_prediction(&to_train_on, &to_test_on);
    let original_output: Vec<_> = to_test_on.iter().map(|a| a.0).collect();
    println!(
        "Predicted is {:?}\nOriginal is {:?}",
        &predicted_output, &original_output
    );
}
