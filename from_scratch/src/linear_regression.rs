use simple_ml::*;
pub fn function(test_size: f64) {
    /*
    Source:
    Video: https://www.youtube.com/watch?v=K_EH2abOp00
    Book: Trevor Hastie,  Robert Tibshirani, Jerome Friedman - The Elements of  Statistical Learning_  Data Mining, Inference, and Pred
    Article: https://towardsdatascience.com/regression-an-explanation-of-regression-metrics-and-what-can-go-wrong-a39a9793d914#:~:text=Root%20Mean%20Squared%20Error%3A%20RMSE,value%20predicted%20by%20the%20model.&text=Mean%20Absolute%20Error%3A%20MAE%20is,value%20predicted%20by%20the%20model.
    Library:

    TODO:
    * FIgure out why the difference in values between py and rs
    * Whats the role of gradient descent in this?
    */

    // read a csv file
    let (columns, values) = read_csv("../../rust/_garage/ccpp.csv".to_string()); // output is row wise

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
    println!("Training size: {:?}", train_data.len());
    println!("Test size: {:?}", test_data.len());

    // converting rows to vector of columns of f64s
    let actual_train = row_to_columns_conversion(&train_data);
    // let actual_test = row_to_columns_conversion(&test_data);

    // println!("{:?}", data);

    // the read columns are in transposed form already, so creating vector of features X
    let xt = MatrixF {
        matrix: actual_train[..actual_train.len() - 1].to_vec(),
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

    let betas = &matrix_multiplication(
        &MatrixF::inverse_f(&xtx), // np.linalg.inv(X.T@X)
        &transpose(&vec![matrix_vector_product_f(&xt.matrix, &y[0])]), //(X.T@y)
    )[0];

    // combining column names with coefficients
    let output: Vec<_> = columns.iter().zip(betas.iter()).collect();
    println!(
        "\n\nThe coeficients of a columns as per simple linear regression on {:?}% of data is : \n{:?}",
        test_size * 100.,
        output
    );

    // predicting the values for test features
    // multiplying each test feture row with corresponding betas to predict the dependent variable
    let mut predicted_values = vec![];
    for i in test_data.iter() {
        predicted_values.push({
            let value = i
                .iter()
                .zip(betas.iter())
                .map(|(a, b)| a * b)
                .collect::<Vec<f64>>();
            value.iter().fold(0., |a, b| a + b)
        });
    }
    println!("RMSE : {:?}", rmse(&test_data, &predicted_values));
    println!("MSE : {:?}", mse(&test_data, &predicted_values));
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

    println!();
    println!();
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
Training size: 7655
Test size: 1913
7655x5 becomes
5x7655
Multiplication of 4x7655 and 7655x4
Output will be 4x4
Multiplication of 4x4 and 4x1
Output will be 4x1


The coeficients of a columns as per simple linear regression on 20.0% of data is :
[("AT", -1.6735588456803612), ("V", -0.27616524703088885), ("AP", 0.5027517381845463), ("RH", -0.09813798522574757)]
RMSE : 5.062438652162896
MSE : 25.62828510691288
MAE : 4.113968080510857
MAPE : 0.9034972803569141
R2 and adjusted R2 : (0.9997990498651516, 0.9997985229901257)


Reading the file ...
Number of rows = 9567
Training size: 7176
Test size: 2392
7176x5 becomes
5x7176
Multiplication of 4x7176 and 7176x4
Output will be 4x4
Multiplication of 4x4 and 4x1
Output will be 4x1


The coeficients of a columns as per simple linear regression on 25.0% of data is :
[("AT", -1.6893489689938406), ("V", -0.26774609393780224), ("AP", 0.5028913443793108), ("RH", -0.10202433228249674)]
RMSE : 4.976978592429923
MSE : 24.770315909505737
MAE : 3.9915987394585732
MAPE : 0.8772890764188112
R2 and adjusted R2 : (0.9998663990527356, 0.9998661190842795)


Reading the file ...
Number of rows = 9567
Training size: 6698
Test size: 2870
6698x5 becomes
5x6698
Multiplication of 4x6698 and 6698x4
Output will be 4x4
Multiplication of 4x4 and 4x1
Output will be 4x1


The coeficients of a columns as per simple linear regression on 30.0% of data is :
[("AT", -1.6690784106236833), ("V", -0.2740858871805756), ("AP", 0.5022444997000939), ("RH", -0.09412680985215616)]
RMSE : 5.04740427300993
MSE : 25.476289895198907
MAE : 3.9852865306553538
MAPE : 0.8762432181491226
R2 and adjusted R2 : (0.9997207590083841, 0.9997202715066529)


Reading the file ...
Number of rows = 9567
Training size: 6220
Test size: 3348
6220x5 becomes
5x6220
Multiplication of 4x6220 and 6220x4
Output will be 4x4
Multiplication of 4x4 and 4x1
Output will be 4x1


The coeficients of a columns as per simple linear regression on 35.0% of data is :
[("AT", -1.6670863422592106), ("V", -0.2812921602464087), ("AP", 0.5032606286680128), ("RH", -0.10315217396873777)]
RMSE : 5.037347138254092
MSE : 25.374866191276688
MAE : 4.022720293731943
MAPE : 0.8839513341675529
R2 and adjusted R2 : (0.9998554323224882, 0.9998552160333237)

*/
