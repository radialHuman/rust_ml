use simple_ml::*;

pub fn function(
    file_path: String,
    drop_column_number: Vec<usize>,
    test_size: f64,
    learning_rate: f64,
    iter_count: i32,
    reg_strength: f64,
) {
    /*
        Source:
        * Video:
        * Book:
        * Article: https://medium.com/@MohammedS/performance-metrics-for-classification-problems-in-machine-learning-part-i-b085d432082b
        * Library:

        Procedure:
        1. Prepare data : read, convert, split into 4 parts
        2. Add new constant 1 column (intercept) in fron to x_train
        3. Initialize coefficient column
        4. Calculate hinge loss cost
        5. Reduce hinge loss cost function using gradient descent
        6. Predict x_test using new coefficients
        7. Calculate various metrics using predicted and actual values

        TODO:
        *
    */

    // read a csv file
    let (columns, values) = read_csv(file_path.clone()); // output is row wise

    // converting vector of string to vector of f64s
    // println!("___");
    let mut random_data = float_randomize(&values);

    println!(
        "The columns are\n{:?}\n",
        columns
            .iter()
            .filter(|a| **a != "\r".to_string())
            .map(|a| a.replace("\"", ""))
            .collect::<Vec<String>>()
    );
    shape("Before dropping columns the dimensions are", &random_data);

    // drop column after converting it to column wise data
    random_data = row_to_columns_conversion(&random_data);
    if drop_column_number.len() > 0 {
        for (n, i) in drop_column_number.iter().enumerate() {
            if n == 0 {
                println!("Dropping column #{}", i);
                random_data = drop_column(&random_data, *i);
            } else {
                println!("Dropping column #{}", i);
                random_data = drop_column(&random_data, *i - n);
            }
        }
    }
    // converting it back to row wise
    random_data = columns_to_rows_conversion(&random_data);

    shape("After dropping columns the dimensions are", &random_data);
    println!();

    head(&random_data, 5);

    // normalizing features in thier columns wise format
    let mut normalized = row_to_columns_conversion(&random_data);

    random_data = row_to_columns_conversion(&random_data);
    for (n, i) in random_data.iter().enumerate() {
        print!(".");
        if n != normalized.len() - 1 - drop_column_number.len() {
            normalized[n] = min_max_scaler(i);
        } else {
            normalized[n] = i.clone();
        }
    }
    println!("\nAfter normalizing:");

    // converting it back to row wise
    normalized = columns_to_rows_conversion(&normalized);
    head(&normalized, 5);
    println!();

    // splitting it into train and test as per test percentage passed as parameter to get scores
    let (mut x_train, y_train, mut x_test, y_test) =
        preprocess_train_test_split(&normalized, test_size, normalized[0].len(), "");

    // adding intercept column to feature
    let mut length = x_train[0].len();
    let intercept = vec![vec![1.; length]];
    x_train = [&intercept[..], &x_train[..]].concat();
    length = x_test[0].len();
    let intercept = vec![vec![1.; length]];
    x_test = [&intercept[..], &x_test[..]].concat();

    // converting into proper shape
    x_train = columns_to_rows_conversion(&x_train);
    x_test = columns_to_rows_conversion(&x_test);

    // checking the shapes
    shape("Training features", &x_train);
    shape("Test features", &x_test);
    println!("Training target: {:?}", &y_train.len());
    println!("Test target: {:?}", &y_test.len());

    let weights = sgd(&x_train, &y_train, iter_count, learning_rate, reg_strength);
    let predictions = predict(&x_test, &weights);
    confuse_me(&predictions, &y_test, -1., 1.);
    println!("Weights of intercept followed by features : {:?}", weights);
    // weights
}

fn sgd(
    features: &Vec<Vec<f64>>,
    output: &Vec<f64>,
    iter_count: i32,
    learning_rate: f64,
    reg_strength: f64,
) -> Vec<f64> {
    let max_epoch: i32 = iter_count;
    let mut weights = vec![0.; features[0].len()];
    let mut nth = 0.;
    let mut prev_cost = std::f64::INFINITY;
    let per_cost_threshold = 0.01;
    for epoch in 1..max_epoch {
        // shuffling inputs
        if epoch % 100 == 0 {
            print!("..");
        }
        let order = randomize_vector(&(0..output.len()).map(|a| a).collect());
        let mut x = vec![];
        let mut y = vec![];
        for i in order.iter() {
            x.push(features[*i].clone());
            y.push(output[*i]);
        }

        // calculating cost
        for (n, i) in x.iter().enumerate() {
            let ascent = calculate_cost_gradient(&weights, i, y[n], reg_strength);
            weights = element_wise_operation(
                &weights,
                &ascent.iter().map(|a| a * learning_rate).collect(),
                "sub",
            );
        }
        // println!("Ascent {:?}", weights);

        if epoch == 2f64.powf(nth) as i32 || epoch == max_epoch - 1 {
            let cost = compute_cost(&weights, features, output, reg_strength);
            println!("{} Epoch, has cost {}", epoch, cost);
            if (prev_cost - cost).abs() < (per_cost_threshold * prev_cost) {
                println!("{:?}", weights);
                return weights;
            }
            prev_cost = cost;
            nth += 1.;
        }
    }
    // println!();
    weights
}

fn compute_cost(weight: &Vec<f64>, x: &Vec<Vec<f64>>, y: &Vec<f64>, reg_strength: f64) -> f64 {
    // hinge loss
    let mut distance = element_wise_operation(&matrix_vector_product_f(x, weight), &y, "mul");
    // println!("{:?}", &matrix_vector_product_f(x, weight).len());
    // println!("Loss {:?}", distance);
    distance = distance.iter().map(|a| 1. - *a).collect();
    distance = distance
        .iter()
        .map(|a| if *a > 0. { *a } else { 0. })
        .collect();
    let hinge_loss = reg_strength * (distance.iter().fold(0., |a, b| a + b) / (x.len() as f64));
    (dot_product(&weight, &weight) / 2.) + hinge_loss
}

fn calculate_cost_gradient(
    weight: &Vec<f64>,
    x_batch: &Vec<f64>,
    y_batch: f64,
    reg_strength: f64,
) -> Vec<f64> {
    let distance = 1. - (dot_product(&x_batch, &weight) * y_batch);
    // println!("Distance {:?}", distance);
    let mut dw = vec![0.; weight.len()];
    let di;
    if distance < 0. {
        di = dw.clone();
    } else {
        let second_half = x_batch.iter().map(|a| a * reg_strength * y_batch).collect();
        di = element_wise_operation(weight, &second_half, "sub");
    }
    dw = element_wise_operation(&di, &dw, "add");
    // println!("di : {:?}", dw);
    dw
}

fn predict(test_features: &Vec<Vec<f64>>, weights: &Vec<f64>) -> Vec<f64> {
    let mut output = vec![];
    for i in test_features.iter() {
        if dot_product(i, weights) > 0. {
            output.push(1.);
        } else {
            output.push(-1.);
        }
    }
    println!("Predications : {:?}", output);
    output
}

fn float_randomize(matrix: &Vec<Vec<String>>) -> Vec<Vec<f64>> {
    randomize(
        &matrix
            .iter()
            .map(|a| {
                a.iter()
                    .map(|b| {
                        (b).replace("\r", "")
                            .replace("\n", "")
                            .parse::<f64>()
                            .unwrap()
                    })
                    .collect::<Vec<f64>>()
            })
            .collect::<Vec<Vec<f64>>>(),
    )
}

pub fn confuse_me(predicted: &Vec<f64>, actual: &Vec<f64>, class0: f64, class1: f64) {
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
        if **i == class0 && **j == class0 {
            tp += 1.;
        }
        if **i == class1 && **j == class1 {
            tng += 1.;
        }
        if **i == class0 && **j == class1 {
            fp += 1.;
        }
        if **i == class1 && **j == class0 {
            fng += 1.;
        }
    }
    println!("\n|------------------------|");
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

/*
RUST OUTPUT

Reading the file ...
Number of rows = 1371
The columns are
["cariance", "skwness", "cutosis", "entropy", "class\r"]

"Before dropping columns the dimensions are" : Rows: 1372, Columns: 5
"After dropping columns the dimensions are" : Rows: 1372, Columns: 5

.....
After normalizing:

Using the actual values without preprocessing unless 's' or 'm' is passed
"Training features" : Rows: 1098, Columns: 5
"Test features" : Rows: 274, Columns: 5
Training target: 1098
Test target: 274
1 Epoch, has cost 22882.993014691856
2 Epoch, has cost 14045.10710876613
4 Epoch, has cost 7834.9207730417
8 Epoch, has cost 4743.915514212458
16 Epoch, has cost 3725.977999098423
32 Epoch, has cost 2793.4691061214075
64 Epoch, has cost 2300.8207892162063
..128 Epoch, has cost 2147.9683458100635
..256 Epoch, has cost 2067.5961920016357
......512 Epoch, has cost 2190.5647076178866
..........1024 Epoch, has cost 2412.590782518554
....................2048 Epoch, has cost 2806.663172601735
........................................4096 Epoch, has cost 3420.698222093876
..................4999 Epoch, has cost 3693.416022921929
Predications : [-1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0]

|------------------------|
|  157.0    |   2.0
|------------------------|
|  1.0    |   114.0
|------------------------|
Accuracy : 0.989
Precision : 0.987
Recall (sensitivity) : 0.994
Specificity: 0.983
F1 : 2.000
*/
