// logistic regression example

use rustlearn::prelude::*;
use rustlearn::datasets::iris;
use rustlearn::cross_validation::CrossValidation;
use rustlearn::metrics::accuracy_score;

use rustlearn::trees::decision_tree;

pub fn logistic_code(){
    use rustlearn::linear_models::sgdclassifier::Hyperparameters;
    let (x,y) = iris::load_data();
    let num_splits = 5;
    let num_epochs = 10;

    let mut accuracy = 0.0;

    for (train_idx, test_idx) in CrossValidation::new(x.rows(), num_splits) {

        let x_train = x.get_rows(&train_idx);
        let y_train = y.get_rows(&train_idx);
        let x_test = x.get_rows(&test_idx);
        let y_test = y.get_rows(&test_idx);

        let mut model = Hyperparameters::new(x.cols())
            .learning_rate(0.5)
            .l2_penalty(0.0)
            .l1_penalty(0.0)
            .one_vs_rest();

        for _ in 0..num_epochs {
            model.fit(&x_train, &y_train).unwrap();
        }

        let prediction = model.predict(&x_test).unwrap();
        accuracy += accuracy_score(&y_test, &prediction);
    }

    accuracy /= num_splits as f32;

    println!("Accuracy with {} splits is {}", num_splits ,accuracy);
}

pub fn random_code(){
    use rustlearn::ensemble::random_forest::Hyperparameters;
    let (data, target) = iris::load_data();

    let mut tree_params = decision_tree::Hyperparameters::new(data.cols());
    tree_params.min_samples_split(10)
        .max_features(4);

    let mut model = Hyperparameters::new(tree_params, 10)
        .one_vs_rest();

    model.fit(&data, &target).unwrap();

    // Optionally serialize and deserialize the model

    // let encoded = bincode::serialize(&model).unwrap();
    // let decoded: OneVsRestWrapper<RandomForest> = bincode::deserialize(&encoded).unwrap();

    let prediction = model.predict(&data).unwrap();
    // incomplete
}