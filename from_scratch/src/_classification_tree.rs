/*
Source:
Video:
Book: Trevor Hastie,  Robert Tibshirani, Jerome Friedman - The Elements of  Statistical Learning_  Data Mining, Inference, and Pred
Article:
Library:

TODO:
*
*/
use simple_ml::*;

pub fn function(file_path: String, test_size: f64, target_column:usize) {
    // read a csv file
    let (columns, values) = read_csv(file_path); // output is row wise

    // converting vector of string to vector of f64s
    let random_data = BLR::float_randomize(&values);  // blr needs to be removed

    // splitting it into train and test as per test percentage passed as parameter to get scores
    let (x_train, y_train, x_test, y_test) =
        BLR::preprocess_train_test_split(&random_data, test_size, target_column, ""); // blr needs to be removed

    shape("Training features", &x_train);
    shape("Test features", &x_test);
    println!("Training target: {:?}", &y_train.len());
    println!("Test target: {:?}", &y_test.len());

    // now to the main part
    
}
