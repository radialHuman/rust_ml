/*
Aim: To replicate codes from established libraries, books, videos and other sources to get similar results and leanr new things along the way

Procedure:
* Understand the concept from various sources
* Understand the codes and its functions
* Rewrite to get the same result
* Detailed documentation

*/

// use simple_ml::*;
// mod classification_tree;
mod knn;
mod linear_regression;
mod logistic_regression;

fn main() {
    let mut file = "../../rust/_garage/ccpp.csv".to_string();
    // linear_regression::function(file.clone(), 0.20);
    // linear_regression::function(file.clone(), 0.25);
    // linear_regression::function(file.clone(), 0.30);
    // linear_regression::function(file.clone(), 0.35);

    // file = "../../rust/_garage/data_banknote_authentication.txt".to_string();
    // logistic_regression::function(file.clone(), 0.20, 5, 0.1, 1000, 0.5);

    // file = "../../rust/_garage/data_banknote_authentication.txt".to_string();
    // knn::function(file.clone(), 0.20, 5, 10, "e");
    // knn::function(file.clone(), 0.20, 5, 10, "ma");
    // knn::function(file.clone(), 0.20, 5, 10, "co");
    // knn::function(file.clone(), 0.20, 5, 10, "ch");

    file = "../../rust/_garage/data_banknote_authentication.txt".to_string();
    // classification_tree::function(file.clone(), 0.20,5);
}
