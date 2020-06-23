use simple_ml::*;

pub fn function(file_path: String, test_size: f64) {
    /*
    Source:
    Video: https://www.youtube.com/watch?v=K_EH2abOp00
    Book: Trevor Hastie,  Robert Tibshirani, Jerome Friedman - The Elements of  Statistical Learning_  Data Mining, Inference, and Pred
    Article: https://towardsdatascience.com/regression-an-explanation-of-regression-metrics-and-what-can-go-wrong-a39a9793d914#:~:text=Root%20Mean%20Squared%20Error%3A%20RMSE,value%20predicted%20by%20the%20model.&text=Mean%20Absolute%20Error%3A%20MAE%20is,value%20predicted%20by%20the%20model.
    Library:

    TODO:
    * Whats the role of gradient descent in this?
    * rules of regression
    * p-value
    * Colinearity
    */

    // read a csv file
    let (columns, values) = read_csv(file_path); // output is row wise

    // converting vector of string to vector of f64s
    let random_data = randomize(&values)
        .iter()
        .map(|a| {
            a.iter()
                .map(|b| b.parse::<f64>().unwrap())
                .collect::<Vec<f64>>()
        })
        .collect::<Vec<Vec<f64>>>();

    // splitting it into train and test as per test percentage passed as parameter to get scores
    let (train_data, test_data) = train_test_split_f(&random_data, test_size);
    // println!("Training size: {:?}", train_data.len());
    // println!("Test size: {:?}", test_data.len());

    // converting rows to vector of columns of f64s
    let actual_train = row_to_columns_conversion(&train_data);
    // let actual_test = row_to_columns_conversion(&test_data);

    // // the read columns are in transposed form already, so creating vector of features X and adding 1 in front of it for b0
    let b0_vec: Vec<Vec<f64>> = vec![vec![1.; actual_train[0].len()]]; //[1,1,1...1,1,1]
    let X = [&b0_vec[..], &actual_train[..]].concat(); // [1,1,1...,1,1,1]+X
                                                       // shape(&X);
    let xt = MatrixF {
        matrix: X[..X.len() - 1].to_vec(),
    };

    // and vector of targets y
    let y = actual_train[actual_train.len() - 1..].to_vec();
    // println!(">>>>>\n{:?}", y);

    /*
    beta = np.linalg.inv(X.T@X)@(X.T@y)
     */

    // (X.T@X)
    let xtx = MatrixF {
        matrix: matrix_multiplication(&xt.matrix, &transpose(&xt.matrix)),
    };
    // println!("{:?}", MatrixF::inverse_f(&xtx));
    let slopes = &matrix_multiplication(
        &MatrixF::inverse_f(&xtx), // np.linalg.inv(X.T@X)
        &transpose(&vec![matrix_vector_product_f(&xt.matrix, &y[0])]), //(X.T@y)
    )[0];

    // combining column names with coefficients
    let output: Vec<_> = columns[..columns.len() - 1]
        .iter()
        .zip(slopes[1..].iter())
        .collect();
    // println!("****************** Without Gradient Descent ******************");
    println!(
        "\n\nThe coeficients of a columns as per simple linear regression on {:?}% of data is : \n{:?} and b0 is : {:?}",
        test_size * 100.,
        output,
        slopes[0]
    );

    // predicting the values for test features
    // multiplying each test feture row with corresponding slopes to predict the dependent variable
    let mut predicted_values = vec![];
    for i in test_data.iter() {
        predicted_values.push({
            let value = i
                .iter()
                .zip(slopes[1..].iter())
                .map(|(a, b)| (a * b))
                .collect::<Vec<f64>>();
            value.iter().fold(slopes[0], |a, b| a + b) // b0+b1x1+b2x2..+bnxn
        });
    }

    println!("RMSE : {:?}", rmse(&test_data, &predicted_values));
    println!("MSE : {:?}", mse(&test_data, &predicted_values)); // cost function
    println!("MAE : {:?}", mae(&test_data, &predicted_values));
    println!("MAPE : {:?}", mape(&test_data, &predicted_values));
    println!(
        "R2 and adjusted R2 : {:?}",
        r_square(
            &test_data
                .iter()
                .map(|a| a[test_data[0].len() - 1])
                .collect(), // passing only the target values
            &predicted_values,
            columns.len(),
        )
    );

    // println!();
    // println!();

    // ADDING COST FUNCTION REDUCTION USING GRADIENT DESCENT
}

fn shape(m: &Vec<Vec<f64>>) {
    // # of rows and columns of a matrix
    println!("{:?}x{:?}", m.len(), m[0].len());
}

fn rmse(test_data: &Vec<Vec<f64>>, predicted: &Vec<f64>) -> f64 {
    /*
    square root of (square of difference of predicted and actual divided by number of predications)
    */
    (mse(test_data, predicted)).sqrt()
}

fn mse(test_data: &Vec<Vec<f64>>, predicted: &Vec<f64>) -> f64 {
    /*
    square of difference of predicted and actual divided by number of predications
    */

    let mut square_error: Vec<f64> = vec![];
    for (n, i) in test_data.iter().enumerate() {
        let j = match i.last() {
            Some(x) => (predicted[n] - x) * (predicted[n] - x), // square difference
            _ => panic!("Something wrong in passed test data"),
        };
        square_error.push(j)
    }
    // println!("{:?}", square_error);
    square_error.iter().fold(0., |a, b| a + b) / (predicted.len() as f64)
}

fn mae(test_data: &Vec<Vec<f64>>, predicted: &Vec<f64>) -> f64 {
    /*
    average of absolute difference of predicted and actual
    */

    let mut absolute_error: Vec<f64> = vec![];
    for (n, i) in test_data.iter().enumerate() {
        let j = match i.last() {
            Some(x) => (predicted[n] - x).abs(), // absolute difference
            _ => panic!("Something wrong in passed test data"),
        };
        absolute_error.push(j)
    }
    // println!("{:?}", absolute_error);
    absolute_error.iter().fold(0., |a, b| a + b) / (predicted.len() as f64)
}

fn r_square(predicted: &Vec<f64>, actual: &Vec<f64>, features: usize) -> (f64, f64) {
    // https://github.com/radialHuman/rust/blob/master/util/util_ml/src/lib_ml.rs
    /*

    */
    let sst: Vec<_> = actual
        .iter()
        .map(|a| {
            (a - (actual.iter().fold(0., |a, b| a + b) / (actual.len() as f64))
                * (a - (actual.iter().fold(0., |a, b| a + b) / (actual.len() as f64))))
        })
        .collect();
    let ssr = predicted
        .iter()
        .zip(actual.iter())
        .fold(0., |a, b| a + (b.0 - b.1));
    let r2 = 1. - (ssr / (sst.iter().fold(0., |a, b| a + b)));
    // println!("{:?}\n{:?}", predicted, actual);
    let degree_of_freedom = predicted.len() as f64 - 1. - features as f64;
    let ar2 = 1. - ((1. - r2) * ((predicted.len() as f64 - 1.) / degree_of_freedom));
    (r2, ar2)
}

fn mape(test_data: &Vec<Vec<f64>>, predicted: &Vec<f64>) -> f64 {
    /*
    average of absolute difference of predicted and actual
    */

    let mut absolute_error: Vec<f64> = vec![];
    for (n, i) in test_data.iter().enumerate() {
        let j = match i.last() {
            Some(x) => (((predicted[n] - x) / predicted[n]).abs()) * 100., // absolute difference
            _ => panic!("Something wrong in passed test data"),
        };
        absolute_error.push(j)
    }
    // println!("{:?}", absolute_error);
    absolute_error.iter().fold(0., |a, b| a + b) / (predicted.len() as f64)
}

/*
RUST OUTPUT
Reading the file ...
Number of rows = 9567
7655x5 becomes
5x7655
Multiplication of 5x7655 and 7655x5
Output will be 5x5
Multiplication of 5x5 and 5x1
Output will be 5x1


The coeficients of a columns as per simple linear regression on 20.0% of data is :
[("AT", -1.9794217630416142), ("V", -0.2329044904145121), ("AP", 0.05541811582105538), ("RH", -0.15620936831788867)] and b0 is : 461.23237680789316
RMSE : 4.658343881285272
MSE : 21.700167716307934
MAE : 3.6770279609836938
MAPE : 0.8102719542108235
R2 and adjusted R2 : (1.000248677900341, 1.0002493299137136)


Reading the file ...
Number of rows = 9567
7176x5 becomes
5x7176
Multiplication of 5x7176 and 7176x5
Output will be 5x5
Multiplication of 5x5 and 5x1
Output will be 5x1


The coeficients of a columns as per simple linear regression on 25.0% of data is :
[("AT", -1.9756800607780178), ("V", -0.2312872310846501), ("AP", 0.0704811095251614), ("RH", -0.16195624771495432)] and b0 is : 446.2349077202962
RMSE : 4.529880621320492
MSE : 20.519818443414927
MAE : 3.6452449106734877
MAPE : 0.8028414652752397
R2 and adjusted R2 : (1.000245426403943, 1.0002459407090645)


Reading the file ...
Number of rows = 9567
6698x5 becomes
5x6698
Multiplication of 5x6698 and 6698x5
Output will be 5x5
Multiplication of 5x5 and 5x1
Output will be 5x1


The coeficients of a columns as per simple linear regression on 30.0% of data is :
[("AT", -1.9698001076058063), ("V", -0.24343968338206423), ("AP", 0.06571628393692208), ("RH", -0.15918273379395487)] and b0 is : 451.3678481276729
RMSE : 4.656991681635678
MSE : 21.6875715228239
MAE : 3.672797272376025
MAPE : 0.8090727437829338
R2 and adjusted R2 : (0.9999394863151237, 0.9999393806697242)


Reading the file ...
Number of rows = 9567
6220x5 becomes
5x6220
Multiplication of 5x6220 and 6220x5
Output will be 5x5
Multiplication of 5x5 and 5x1
Output will be 5x1


The coeficients of a columns as per simple linear regression on 35.0% of data is :
[("AT", -1.990523685520202), ("V", -0.2333029930290138), ("AP", 0.05052158741426638), ("RH", -0.1548556324194692)] and b0 is : 466.3405575223733
RMSE : 4.520461569402572
MSE : 20.434572800445565
MAE : 3.5563517805183538
MAPE : 0.7851810113234632
R2 and adjusted R2 : (1.0001809290166148, 1.0001811997063463)

*/
