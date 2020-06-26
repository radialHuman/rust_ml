use simple_ml::*;

pub fn function(file_path: String, test_size: f64, target_column: usize) {
    /*
        Source:
        * Video:
        * Book:
        * Article: 
        * Library:

        Procedure:
        1. Prepare data : read, convert, split into 4 parts
        2. Add new constant 1 column (intercept) in fron to x_train
        3. Initialize coefficient column
        4. Calculate hinge loss cost using sigmoid function's output and actual output
        5. Reduce hinge loss cost function using gradient descent
        6. Predict x_test using new coefficients
        7. Calculate various metrics using predicted and actual values

        TODO:
        *
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
}
