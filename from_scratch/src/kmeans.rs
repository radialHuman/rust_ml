use simple_ml::*;

pub fn function(file_path: String, k: usize, iterations: u32) {
    /*
        Source:
        Video:
        Book: Trevor Hastie,  Robert Tibshirani, Jerome Friedman - The Elements of  Statistical Learning_  Data Mining, Inference, and Pred
        Article: https://www.analyticsvidhya.com/blog/2019/08/comprehensive-guide-k-means-clustering/
        Library:

        ABOUT:
        * Assuming no duplicate rows exist
        * Only features and no targets in the input data

        Procedure:
        1. Prepare data : remove target if any
        2. Select K centroids
        3. Find closest points (Eucledian distance)
        4. Calcualte new mean
        5. Repeat 3,4 till the same points ened up in the cluster

        TODO:
        * Add cost function to minimize
    */

    // read a csv file
    let (columns, values) = read_csv(file_path); // output is row wise

    // converting vector of string to vector of f64s
    let random_data: Vec<_> = float_randomize(&values);

    // selecting first k points as centroid (already in random order)
    let mut centroids = randomize(&random_data)[..k].to_vec();
    print_a_matrix("Original means", &centroids);

    let mut new_mean: Vec<Vec<f64>> = vec![];
    for x in 0..iterations - 1 {
        let mut updated_cluster = vec![];
        let mut nearest_centroid_number = vec![];
        for i in random_data.iter() {
            let mut distance = vec![];
            for (centroid_number, j) in centroids.iter().enumerate() {
                let dis = Distance {
                    row1: i.clone(),
                    row2: j.clone(),
                };
                distance.push((centroid_number, dis.distance_euclidean()))
            }
            distance.sort_by(|m, n| m.1.partial_cmp(&n.1).unwrap());
            nearest_centroid_number.push(distance[0].0);
        }

        // combining cluster number and data
        let clusters: Vec<(&usize, &Vec<f64>)> = nearest_centroid_number
            .iter()
            .zip(random_data.iter())
            .collect();
        // println!("{:?}", clusters);

        // finding new centorid
        new_mean = vec![];
        for (m, _) in centroids.iter().enumerate() {
            let mut group = vec![];
            for i in clusters.iter() {
                if *i.0 == m {
                    group.push(i.1.clone());
                }
            }
            new_mean.push(
                group
                    .iter()
                    .fold(vec![0.; k], |a, b| element_wise_operation(&a, b, "add"))
                    .iter()
                    .map(|a| a / (group.len() as f64)) // the mean part in K-means
                    .collect(),
            );
            updated_cluster = clusters.clone()
        }
        println!("Iteration {:?}", x);
        if centroids == new_mean {
            // show in a list of cluster number as per the order of row in original data
            let mut rearranged_output = vec![];
            for i in values
                .iter()
                .map(|a| a.iter().map(|b| b.parse().unwrap()).collect())
                .collect::<Vec<Vec<f64>>>()
                .iter()
            {
                for (c, v) in updated_cluster.iter() {
                    if i == *v {
                        rearranged_output.push((c, v));
                        break;
                    }
                }
            }
            // displaying only the clusters assigned to  each row
            println!(
                "CLUSTERS\n{:?}",
                rearranged_output
                    .iter()
                    .map(|a| **(a.0))
                    .collect::<Vec<usize>>()
            );
            break;
        } else {
            centroids = new_mean.clone();
        }
    }
    print_a_matrix("Final means", &centroids);
}

/*
RUST OUTPUT

Reading the file ...
Number of rows = 9567
Original means
[26.79, 62.44, 1011.51, 72.46, 440.55]
[20.57, 60.1, 1011.16, 83.45, 451.88]
[25.63, 56.85, 1012.68, 49.7, 439.2]
[14.81, 43.69, 1017.19, 71.9, 470.71]
[24.81, 66.44, 1011.19, 69.96, 435.79]


Iteration 0
Iteration 1
Iteration 2
Iteration 3
Iteration 4
Iteration 5
Iteration 6
Iteration 7
Iteration 8
Iteration 9
Iteration 10
Iteration 11
Iteration 12
Iteration 13
Iteration 14
Iteration 15
Iteration 16
Iteration 17
Iteration 18
Iteration 19
Iteration 20
Iteration 21
Iteration 22
Iteration 23
Iteration 24
Iteration 25
Iteration 26
Iteration 27
CLUSTERS
[3, 1, 2, 1, 3, 3, 4, 3, 2, 3, 4, 0, 2, 3, 0, 2, 0, 3, 0, 2, 0, 2, 4, 3, 3, 3, 4, 3, 1, 2, 3, 1, 3, 2, 2, 0, 4, 2, 3, 2, 2, 1, 0, 2, 1, 2, 3, 1, 3, 4, 4, 2, 0, 4, 2, 3, 3, 4, 3, 4, 1, 2, 0, 2, 2, 4, 3, 0, 1, 4, 2, 2, 1, 3, 3, 2, 2, 4, 2, 4, 1, 4, 2, 0, 0, 3, 4, 2, 3, 2, 1, 2, 0, 0, 1, 2, 1, 3, 2, 3, 1, 2, 1, 3, 3, 3, 4, 4, 2, 1, 0, 4, 1, 1, 3, 2, 4, 4, 1, 0, 3, 0, 4, 0, 1, 0, 0, 3, 3, 4, 0, 4, 1, 3, 3, 4, 1, 3, 1, 0, 3, 3, 2, 3, 2, 3, 0, 1, 3, 1, 0, 0, 0, 3, 3, 1, 0, 0, 1, 1, 3, 2, 0, 0, 0, 2, 1, 3, 1, 0, 2, 2, 3, 0, 1, 4, 3, 0, 2, 2, 4, 1, 0, 3, 3, 3, 4, 4, 3, 1, 1, 1, 0, 1, 3, 1, 1, 1, 2, 3, 2, 1, 0, 2, 0, 3, 4, 3, 3, 3, 0, 2, 3, 2, 2, 3, 2, 4, 3, 2, 1, 2, 4, 2, 3, 0, 0, 1, 2, 3, 4, 2, 0, 2, 3, 0, 0, 4, 1, 3, 4, 1, 2, 4, 0, 3, 0, 1, 4, 1, 1, 2, 0, 0, 2, 3, 4, 3, 0, 1, 4, 3, 2, 2, 1, 0, 3, 2, 0, 2, 4, 4, 1, 2, 2, 4, 0, 2, 2, 0, 0, 3, 3, 0, 4, 3, 1, 2, 3, 2, 4, 1, 2, 2, 4, 3, 2, 0, 3, 3, 3, 3, 0, 0, 3, 0, 2, 3, 4, 0, 3, 0, 4, 3, 0, 1, 1, 4, 3, 2, 4, 2, 2, 2, 2, 4, 3, 1, 0, 3, 0, 0, 0, 1, 3, 4, 3, 3, 0, 1, 2, 2, 0, 2, 3, 3, 1, 3, 2, 3, 4, 1, 1, 3, 3, 1, 4, 0, 1, 1, 2, 1, 2, 3, 4, 1, 3, 2, 2, 1, 4, 0, 0, 1, 2, 2, 1, 0, 3, 0, 3, 3, 2, 1, 2, 2, 2, 0, 3, 3, 1, 2, 1, 4, 0, 4, 3, 3, 1, 3, 0, 3, 3, 3, 0, 3, 1, 3, 3, 2, 0, 1, 4, 0, 0, 1, 1, 2, 1, 2, 1, 2, 0, 2, 4, 3, 1, 4, 4, 3, 0, 4, 0, 2, 2, 4, 0, 3, 4, 1, 4, 0, 1, 3, 4, 3, 2, 1, 1, 3, 3, 1, 2, 1, 4, 1, 3, 3, 3, 3, 4, 0, 2, 2, 1, 3, 1, 4, 4, 0, 3, 1, 1, 2, 1, 4, 1, 4, 3, 2, 0, 4, 3, 2, 4, 3, 3, 2, 2, 3, 2, 1, 3, 4, 2, 0, 0, 1, 2, 1, 4, 4, 0, 0, 0, 1, 2, 3, 1, 4, 3, 4, 3, 4, 3, 2, 1, 3, 3, 2, 2, 3, 3, 1, 4, 4, 3, 2, 0, 4, 3, 1, 3, 3, 1, 3, 4, 0, 4, 3, 3, 4, 0, 2, 3, 4, 1, 1, 0, 4, 4, 3, 4, 3, 0, 1, 1, 3, 0, 3, 3, 3, 3, 3, 3, 1, 2, 2, 4, 3, 2, 4, 2, 1, 3, 0, 0, 3, 0, 2, 0, 2, 0, 1, 2, 4, 4, 2, 0, 3, 1, 1, 2, 1, 4, 2, 2, 1, 1, 4, 2, 1, 2, 1, 3, 2, 0, 1, 0, 1, 2, 4, 3, 1, 3, 2, 1, 3, 2, 4, 2, 0, 3, 2, 0, 3, 1, 0, 1, 3, 2, 1, 1, 2, 1, 0, 0, 1, 3, 1, 2, 2, 3, 3, 1, 0, 1, 2, 0, 3, 0, 2, 3, 0, 4, 4, 1, 3, 4, 1, 4, 2, 2, 2, 4, 1, 3, 3, 3, 1, 0, 1, 1, 2, 3, 3, 1, 4, 1, 0, 4, 1, 1, 2, 1, 4, 3, 0, 3, 3, 3, 3, 1, 3, 2, 2, 4, 4, 2, 3, 3, 2, 3, 4, 4, 1, 4, 1, 3, 3, 3, 1, 4, 3, 4, 3, 3, 4, 2, 2, 4, 2, 2, 3, 0, 2, 3, 2, 0, 3, 2, 2, 1, 3, 2, 3, 0, 3, 4, 3, 3, 1, 1, 3, 1, 4, 4, 3, 2, 3, 4, 3, 4, 3, 3, 0, 1, 4, 3, 2, 0, 3, 2, 1, 1, 3, 3, 3, 4, 3, 4, 2, 0, 0, 1, 1, 1, 0, 4, 4, 2, 1, 2, 2, 1, 0, 1, 3, 2, 2, 2, 2, 3, 3, 4, 4, 1, 1, 0, 2, 2, 0, 1, 3, 4, 0, 0, 1, 0, 1, 1, 1, 2, 3, 0, 1, 2, 2, 1, 0, 1, 0, 3, 3, 3, 2, 2, 4, 2, 4, 2, 1, 4, 3, 0, 3, 4, 4, 4, 1, 2, 1, 1, 3, 3, 3, 0, 2, 3, 1, 4, 2, 2, 4, 1, 1, 1, 3, 4, 1, 4, 2, 3, 1, 3, 3, 4, 1, 0, 0, 4, 1, 2, 2, 4, 2, 2, 0, 2, 0, 2, 3, 2, 1, 2, 0, 2, 0, 2, 3, 3, 3, 2, 3, 2, 4, 2, 3, 0, 3, 2, 1, 0, 0, 2, 0, 2, 0, 1, 4, 3, 2, 3, 1, 2, 2, 0, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 0, 2, 3, 3, 3, 2, 2, 4, 4, 3, 4, 2, 3, 3, 3, 3, 3, 0, 2, 2, 4, 1, 4, 1, 0, 4, 2, 1, 1, 3, 0, 2, 2, 2, 2, 4, 3, 4, 2, 1, 2, 4, 2, 0, 4, 4, 3, 2, 2, 0, 3, 2, 4, 2, 3, 0, 2, 1, 2, 0, 4, 3, 1, 4, 3, 0, 1, 0, 3, 4, 2, 3, 4, 2, 0, 1, 0, 4, 2, 4, 3, 4, 2, 1, 3, 4, 3, 4, 3, 1, 3, 1, 2, 3, 4, 3, 0, 3, 0, 1, 1, 1, 3, 3, 3, 3, 3, 0, 2, 4, 4, 2, 3, 1, 1, 0, 1, 2, 2, 3, 4, 3, 3, 4, 1, 3, 2, 1, 1, 4, 2, 3, 2, 3, 1, 0, 3, 0, 3, 3, 0, 4, 0, 2, 1, 4, 1, 4, 2, 2, 0, 3, 4, 1, 2, 2, 2, 4, 4, 3, 2, 3, 1, 4, 4, 2, 4, 2, 3, 3, 3, 0, 0, 0, 3, 3, 0, 2, 1, 3, 4, 2, 1, 2, 3, 3, 4, 4, 2, 2, 3, 3, 2, 1, 2, 1, 2, 3, 3, 2, 2, 1, 3, 2, 2, 4, 3, 3, 0, 4, 3, 3, 1, 1, 3, 3, 3, 2, 3, 1, 2, 0, 1, 2, 2, 1, 0, 3, 1, 0, 3, 1, 3, 0, 3, 0, 1, 3, 1, 3, 4, 3, 1, 3, 4, 3, 1, 3, 0, 3, 2, 2, 2, 1, 1, 3, 3, 3, 0, 2, 1, 1, 1, 2, 3, 2, 0, 2, 1, 3, 1, 4, 2, 3, 0, 3, 2, 1, 4, 1, 2, 3, 3, 1, 0, 4, 1, 1, 3, 0, 0, 1, 3, 3, 3, 4, 1, 2, 0, 1, 0, 4, 3, 3, 1, 2, 0, 4, 1, 3, 2, 3, 2, 2, 2, 2, 0, 2, 3, 4, 2, 0, 3, 1, 0, 3, 0, 4, 3, 2, 3, 4, 3, 4, 0, 3, 3, 1, 4, 3, 1, 2, 3, 2, 1, 3, 4, 0, 4, 1, 0, 3, 3, 3, 4, 1, 4, 3, 4, 1, 4, 1, 0, 4, 0, 0, 4, 3, 2, 1, 2, 3, 3, 2, 0, 4, 4, 2, 1, 2, 2, 2, 2, 0, 4, 3, 1, 3, 3, 2, 1, 1, 2, 2, 2, 3, 3, 2, 0, 4, 4, 0, 4, 2, 1, 2, 0, 2, 1, 1, 3, 3, 0, 2, 4, 3, 3, 0, 3, 1, 0, 1, 4, 3, 2, 1, 4, 3, 1, 2, 4, 3, 3, 0, 3, 3, 1, 2, 3, 2, 3, 3, 0, 4, 3, 3, 4, 3, 2, 3, 3, 0, 0, 1, 3, 1, 4, 3, 0, 1, 2, 1, 3, 2, 4, 0, 3, 3, 3, 1, 1, 0, 3, 3, 0, 3, 0, 1, 4, 1, 1, 0, 2, 2, 1, 4, 0, 1, 2, 0, 4, 2, 0, 2, 2, 3, 0, 2, 3, 1, 0, 0, 4, 2, 0, 3, 0, 3, 3, 3, 0, 3, 0, 3, 4, 4, 3, 0, 1, 4, 1, 1, 1, 3, 1, 3, 4, 0, 1, 2, 4, 3, 1, 2, 3, 4, 3, 3, 4, 1, 2, 2, 2, 2, 0, 3, 4, 4, 2, 0, 2, 2, 1, 2, 1, 2, 2, 4, 2, 2, 3, 1, 4, 3, 2, 1, 3, 1, 0, 2, 1, 4, 3, 4, 2, 3, 2, 1, 0, 3, 3, 2, 2, 0, 2, 2, 0, 4, 1, 1, 0, 1, 0, 0, 1, 2, 1, 3, 3, 2, 1, 4, 3, 0, 1, 0, 3, 1, 0, 2, 3, 4, 3, 4, 3, 3, 3, 4, 1, 1, 2, 1, 1, 0, 0, 2, 3, 4, 3, 1, 4, 2, 3, 2, 1, 0, 1, 0, 3, 4, 3, 0, 3, 0, 3, 1, 1, 2, 3, 3, 3, 0, 1, 0, 1, 4, 0, 3, 3, 3, 3, 3, 1, 1, 0, 2, 1, 4, 4, 4, 0, 3, 2, 1, 4, 4, 3, 1, 3, 4, 3, 1, 3, 2, 0, 3, 3, 0, 1, 0, 4, 2, 2, 1, 2, 4, 4, 4, 3, 1, 3, 1, 2, 0, 4, 3, 2, 4, 1, 0, 1, 3, 2, 3, 4, 1, 4, 4, 4, 3, 4, 0, 0, 2, 1, 1, 0, 4, 2, 4, 1, 0, 2, 0, 2, 1, 3, 2, 1, 1, 3, 3, 0, 3, 0, 1, 3, 3, 2, 4, 2, 3, 2, 3, 4, 3, 3, 3, 4, 3, 3, 2, 4, 4, 3, 2, 3, 4, 3, 3, 3, 3, 3, 1, 1, 1, 3, 0, 2, 2, 1, 3, 4, 3, 3, 1, 3, 3, 0, 1, 1, 2, 3, 0, 1, 1, 1, 4, 4, 2, 0, 3, 3, 1, 3, 3, 2, 2, 1, 4, 3, 3, 1, 3, 1, 2, 1, 3, 3, 0, 2, 0, 2, 2, 0, 3, 2, 0, 4, 4, 4, 4, 3, 0, 0, 3, 4, 0, 4, 4, 1, 1, 4, 3, 0, 2, 2, 2, 0, 4, 0, 2, 0, 2, 2, 2, 4, 4, 2, 4, 4, 4, 1, 2, 2, 1, 3, 3, 0, 3, 2, 2, 2, 1, 3, 0, 0, 3, 2, 1, 3, 4, 3, 1, 4, 0, 4, 2, 1, 1, 4, 4, 4, 3, 4, 1, 1, 2, 1, 1, 4, 3, 1, 4, 2, 0, 0, 0, 3, 2, 3, 4, 4, 4, 1, 3, 1, 3, 3, 3, 4, 0, 3, 1, 2, 3, 4, 2, 2, 2, 4, 3, 3, 3, 1, 1, 3, 3, 2, 4, 1, 1, 4, 1, 3, 2, 0, 3, 2, 1, 1, 4, 3, 4, 3, 4, 1, 4, 4, 2, 2, 2, 0, 4, 3, 3, 2, 1, 2, 3, 3, 0, 2, 4, 3, 0, 2, 1, 0, 3, 3, 1, 3, 3, 1, 1, 3, 0, 2, 4, 2, 2, 3, 0, 3, 0, 3, 3, 4, 3, 3, 1, 3, 0, 2, 0, 1, 0, 0, 4, 1, 3, 0, 2, 3, 3, 3, 4, 3, 2, 3, 0, 3, 2, 1, 0, 3, 2, 2, 2, 1, 4, 2, 1, 4, 4, 2, 3, 0, 2, 3, 0, 1, 0, 3, 3, 0, 1, 3, 0, 2, 3, 2, 2, 1, 1, 1, 2, 4, 2, 1, 4, 2, 4, 3, 3, 2, 2, 3, 3, 2, 3, 3, 0, 4, 4, 3, 1, 1, 3, 4, 3, 3, 3, 1, 0, 2, 2, 4, 3, 1, 0, 2, 1, 0, 3, 2, 4, 2, 3, 1, 0, 4, 3, 4, 3, 4, 3, 3, 2, 3, 3, 0, 3, 3, 0, 3, 4, 4, 1, 0, 1, 2, 3, 0, 1, 3, 3, 1, 2, 3, 0, 3, 3, 2, 1, 2, 0, 1, 3, 0, 3, 4, 0, 3, 3, 2, 3, 4, 4, 3, 1, 3, 1, 1, 3, 1, 0, 4, 0, 2, 1, 4, 3, 1, 1, 4, 2, 4, 4, 4, 4, 4, 3, 0, 2, 3, 4, 3, 1, 1, 1, 3, 3, 2, 3, 2, 3, 3, 4, 4, 3, 1, 4, 1, 1, 0, 2, 2, 1, 2, 1, 3, 3, 2, 3, 3, 0, 3, 2, 1, 2, 3, 2, 2, 3, 4, 2, 1, 4, 2, 1, 3, 3, 1, 2, 3, 4, 4, 4, 0, 1, 2, 3, 3, 0, 3, 4, 0, 3, 3, 4, 4, 4, 0, 3, 2, 0, 3, 1, 0, 2, 3, 3, 4, 3, 0, 3, 3, 2, 1, 0, 0, 0, 3, 3, 3, 3, 3, 2, 3, 4, 3, 3, 0, 0, 1, 0, 1, 3, 1, 4, 0, 0, 2, 1, 2, 1, 4, 2, 0, 3, 3, 3, 2, 2, 2, 4, 0, 1, 1, 0, 1, 3, 2, 4, 4, 3, 3, 1, 4, 0, 3, 0, 3, 3, 0, 0, 4, 4, 0, 4, 3, 2, 4, 1, 4, 3, 4, 3, 2, 0, 1, 2, 3, 1, 3, 1, 4, 2, 4, 2, 4, 2, 3, 1, 0, 1, 1, 1, 0, 2, 1, 0, 2, 2, 2, 4, 2, 2, 4, 2, 4, 0, 2, 2, 3, 3, 0, 3, 4, 4, 0, 1, 4, 3, 1, 2, 3, 3, 3, 0, 2, 1, 3, 3, 3, 3, 2, 2, 3, 2, 4, 3, 3, 2, 3, 3, 3, 4, 2, 4, 3, 0, 0, 0, 0, 1, 2, 3, 0, 2, 1, 3, 2, 3, 1, 2, 2, 0, 3, 3, 3, 3, 3, 2, 1, 2, 4, 3, 1, 0, 0, 0, 3, 3, 3, 3, 1, 0, 4, 3, 0, 3, 3, 3, 0, 3, 2, 3, 3, 0, 1, 0, 0, 2, 3, 2, 2, 3, 4, 3, 0, 0, 3, 3, 2, 3, 0, 2, 0, 4, 2, 2, 3, 2, 1, 4, 4, 4, 1, 2, 3, 2, 3, 0, 3, 0, 1, 3, 3, 3, 0, 0, 3, 4, 2, 2, 1, 3, 4, 2, 3, 2, 2, 4, 4, 2, 2, 2, 3, 4, 2, 2, 0, 3, 0, 0, 0, 2, 4, 2, 1, 4, 4, 3, 2, 3, 2, 1, 2, 3, 1, 1, 0, 3, 1, 4, 3, 2, 1, 4, 1, 3, 0, 0, 0, 4, 1, 3, 1, 0, 4, 0, 3, 3, 3, 3, 2, 1, 1, 3, 1, 2, 1, 4, 4, 2, 0, 2, 2, 3, 1, 4, 3, 2, 3, 2, 3, 3, 3, 2, 0, 1, 4, 0, 2, 3, 0, 3, 4, 0, 0, 2, 0, 3, 4, 4, 3, 1, 0, 1, 0, 3, 3, 1, 3, 3, 3, 1, 2, 4, 2, 1, 2, 1, 3, 4, 3, 3, 3, 1, 2, 3, 2, 0, 1, 2, 2, 4, 0, 3, 4, 0, 4, 0, 2, 4, 2, 3, 2, 1, 2, 4, 1, 1, 3, 0, 2, 4, 0, 1, 1, 3, 1, 2, 4, 4, 2, 1, 4, 3, 4, 0, 1, 2, 2, 2, 0, 0, 3, 0, 3, 3, 1, 1, 3, 3, 0, 2, 3, 3, 1, 2, 1, 0, 0, 4, 0, 2, 2, 2, 1, 0, 1, 4, 2, 2, 4, 1, 0, 2, 4, 3, 3, 0, 2, 0, 4, 3, 3, 4, 0, 3, 3, 0, 4, 0, 3, 3, 1, 0, 0, 1, 1, 4, 4, 2, 4, 4, 4, 0, 4, 4, 3, 0, 1, 2, 3, 2, 3, 2, 4, 3, 4, 3, 4, 2, 4, 2, 1, 2, 4, 3, 0, 0, 1, 2, 1, 3, 2, 3, 1, 4, 0, 1, 3, 4, 3, 3, 3, 1, 4, 3, 3, 0, 1, 3, 2, 0, 1, 1, 1, 2, 2, 1, 4, 3, 3, 4, 2, 1, 4, 2, 0, 4, 3, 2, 0, 3, 3, 0, 0, 0, 0, 3, 3, 4, 3, 4, 2, 2, 0, 1, 2, 3, 3, 0, 3, 3, 2, 3, 3, 4, 3, 0, 3, 4, 0, 3, 2, 3, 4, 1, 2, 3, 1, 4, 4, 4, 4, 0, 2, 4, 1, 2, 4, 1, 3, 1, 3, 3, 3, 3, 3, 3, 2, 4, 1, 2, 2, 1, 2, 1, 2, 0, 2, 3, 3, 4, 4, 0, 4, 3, 4, 3, 4, 0, 0, 1, 4, 3, 4, 1, 3, 1, 2, 4, 0, 2, 0, 2, 2, 1, 0, 3, 3, 4, 1, 4, 1, 2, 3, 1, 3, 3, 3, 2, 4, 3, 2, 3, 0, 1, 4, 4, 4, 3, 3, 0, 1, 3, 0, 2, 1, 2, 2, 1, 0, 3, 0, 1, 2, 3, 1, 0, 3, 3, 1, 3, 3, 1, 2, 0, 1, 3, 1, 2, 1, 3, 3, 1, 2, 0, 1, 2, 1, 3, 0, 0, 2, 3, 2, 1, 4, 2, 0, 0, 2, 4, 2, 1, 3, 1, 3, 0, 1, 2, 4, 2, 3, 3, 4, 3, 3, 2, 3, 1, 2, 1, 1, 0, 2, 2, 3, 1, 2, 0, 1, 2, 1, 2, 3, 0, 1, 0, 1, 3, 2, 2, 4, 0, 2, 3, 3, 4, 1, 1, 1, 3, 1, 2, 4, 3, 3, 3, 0, 2, 0, 1, 1, 2, 0, 1, 2, 2, 3, 4, 4, 2, 2, 3, 4, 4, 0, 1, 3, 2, 0, 2, 4, 2, 3, 2, 1, 3, 4, 1, 1, 4, 0, 2, 3, 4, 2, 3, 3, 2, 3, 1, 3, 1, 2, 4, 3, 1, 2, 0, 1, 2, 4, 2, 1, 1, 3, 2, 4, 1, 2, 3, 3, 0, 2, 2, 4, 0, 3, 2, 4, 1, 2, 1, 1, 2, 1, 2, 1, 3, 4, 0, 2, 1, 4, 1, 3, 0, 1, 2, 3, 3, 3, 3, 4, 2, 3, 3, 2, 2, 3, 4, 1, 2, 0, 2, 1, 3, 4, 3, 1, 3, 0, 2, 1, 2, 1, 4, 2, 3, 0, 1, 0, 2, 2, 2, 2, 3, 1, 3, 0, 1, 0, 3, 0, 0, 2, 4, 3, 3, 1, 1, 2, 3, 3, 0, 0, 1, 1, 1, 2, 3, 3, 4, 3, 0, 2, 0, 0, 2, 2, 2, 3, 1, 1, 4, 4, 3, 1, 1, 1, 4, 2, 2, 3, 1, 0, 1, 1, 3, 4, 4, 2, 4, 0, 4, 4, 3, 2, 2, 4, 2, 3, 4, 1, 2, 3, 1, 4, 3, 0, 0, 4, 4, 2, 3, 3, 2, 4, 3, 0, 3, 2, 0, 0, 1, 3, 4, 0, 3, 2, 1, 3, 3, 0, 1, 0, 4, 3, 3, 1, 4, 1, 3, 1, 3, 4, 1, 3, 1, 0, 2, 4, 4, 3, 1, 3, 0, 4, 3, 1, 3, 2, 0, 0, 3, 0, 3, 4, 2, 0, 4, 1, 1, 4, 3, 2, 1, 1, 0, 3, 4, 2, 2, 3, 1, 4, 3, 2, 1, 2, 0, 1, 1, 3, 4, 2, 3, 1, 3, 4, 2, 4, 4, 1, 2, 2, 0, 3, 3, 4, 0, 2, 3, 1, 3, 2, 3, 2, 1, 3, 2, 4, 0, 3, 4, 3, 1, 3, 3, 3, 0, 4, 1, 4, 1, 1, 1, 1, 4, 3, 3, 2, 2, 4, 2, 1, 4, 1, 2, 0, 4, 2, 0, 0, 3, 1, 2, 0, 4, 3, 1, 3, 3, 4, 3, 2, 2, 3, 3, 3, 4, 0, 1, 2, 2, 1, 3, 3, 2, 4, 3, 3, 0, 2, 4, 2, 4, 1, 2, 2, 1, 2, 4, 4, 2, 4, 3, 4, 3, 2, 4, 2, 2, 4, 3, 1, 2, 3, 3, 0, 4, 1, 0, 0, 3, 1, 3, 2, 0, 1, 0, 3, 2, 4, 3, 0, 3, 0, 0, 2, 1, 4, 4, 1, 3, 3, 0, 3, 1, 1, 2, 1, 1, 0, 0, 0, 0, 2, 4, 0, 1, 2, 4, 2, 4, 4, 1, 3, 2, 1, 0, 2, 3, 1, 4, 3, 2, 3, 2, 2, 1, 3, 2, 0, 4, 2, 0, 3, 0, 4, 0, 3, 3, 1, 1, 3, 4, 3, 0, 4, 2, 2, 1, 1, 3, 3, 4, 0, 0, 0, 1, 3, 1, 1, 0, 2, 1, 1, 4, 1, 3, 3, 2, 3, 0, 2, 0, 2, 0, 2, 2, 2, 0, 3, 2, 1, 4, 1, 0, 3, 3, 3, 3, 0, 0, 2, 1, 1, 3, 3, 0, 2, 3, 2, 4, 1, 4, 3, 3, 0, 0, 0, 1, 0, 0, 4, 4, 0, 3, 0, 1, 1, 1, 4, 0, 3, 1, 4, 3, 1, 2, 3, 4, 1, 3, 1, 4, 3, 3, 0, 0, 4, 0, 2, 3, 0, 4, 3, 2, 0, 1, 3, 0, 1, 0, 0, 3, 4, 2, 3, 1, 1, 4, 2, 0, 4, 4, 3, 3, 1, 4, 4, 3, 1, 1, 1, 4, 3, 1, 2, 3, 2, 4, 4, 2, 1, 1, 1, 2, 3, 2, 0, 2, 3, 3, 2, 2, 0, 1, 4, 4, 4, 3, 0, 2, 3, 4, 3, 2, 0, 1, 2, 0, 0, 0, 3, 4, 3, 3, 2, 2, 4, 3, 1, 3, 0, 1, 2, 4, 2, 2, 4, 3, 3, 2, 3, 2, 4, 4, 0, 0, 3, 2, 1, 3, 4, 1, 3, 2, 0, 4, 3, 3, 3, 2, 1, 4, 2, 3, 4, 2, 2, 3, 2, 4, 0, 0, 3, 1, 2, 0, 3, 1, 3, 4, 1, 3, 2, 2, 3, 2, 4, 4, 1, 4, 2, 0, 1, 1, 3, 1, 2, 0, 3, 2, 2, 2, 3, 3, 2, 0, 0, 2, 0, 3, 0, 1, 3, 3, 1, 4, 4, 3, 1, 3, 4, 2, 3, 0, 4, 2, 2, 3, 4, 2, 3, 3, 0, 1, 3, 1, 3, 2, 2, 0, 4, 3, 3, 3, 2, 2, 3, 3, 3, 3, 1, 4, 2, 1, 1, 4, 4, 1, 4, 2, 2, 1, 0, 4, 1, 3, 0, 2, 0, 2, 3, 0, 0, 3, 3, 0, 4, 0, 4, 2, 0, 1, 0, 4, 4, 2, 3, 1, 1, 2, 0, 4, 1, 4, 0, 1, 0, 1, 1, 0, 2, 2, 2, 2, 0, 3, 3, 1, 0, 4, 3, 0, 2, 3, 2, 1, 0, 3, 1, 1, 1, 3, 3, 4, 4, 2, 1, 1, 3, 1, 0, 4, 3, 0, 4, 0, 3, 3, 2, 0, 2, 2, 2, 3, 4, 3, 2, 0, 1, 1, 3, 1, 1, 1, 3, 0, 0, 0, 3, 3, 3, 1, 3, 0, 0, 4, 2, 1, 2, 3, 3, 3, 1, 2, 2, 3, 3, 1, 1, 2, 0, 2, 2, 0, 3, 0, 3, 0, 2, 1, 3, 4, 0, 4, 4, 1, 1, 0, 4, 1, 1, 2, 3, 4, 4, 3, 2, 3, 1, 2, 0, 0, 0, 3, 1, 2, 1, 0, 3, 2, 3, 4, 3, 4, 1, 2, 4, 3, 0, 4, 1, 0, 2, 3, 3, 4, 2, 1, 4, 1, 0, 2, 1, 1, 4, 1, 3, 0, 0, 3, 0, 3, 0, 4, 2, 1, 0, 2, 3, 1, 3, 0, 3, 3, 1, 3, 3, 4, 4, 0, 0, 4, 1, 3, 1, 3, 1, 3, 0, 1, 0, 3, 1, 3, 1, 1, 4, 2, 4, 0, 1, 3, 2, 3, 1, 4, 3, 2, 2, 2, 2, 4, 4, 0, 1, 0, 2, 0, 3, 2, 2, 4, 2, 0, 1, 0, 1, 4, 0, 0, 1, 2, 3, 4, 2, 4, 0, 1, 1, 3, 4, 1, 1, 2, 3, 4, 2, 1, 1, 3, 3, 3, 2, 0, 0, 3, 1, 2, 0, 3, 2, 3, 1, 4, 2, 0, 2, 4, 0, 4, 3, 3, 0, 4, 2, 4, 2, 1, 2, 3, 1, 3, 3, 2, 3, 4, 3, 2, 2, 2, 1, 2, 2, 2, 2, 3, 1, 4, 3, 4, 2, 0, 1, 3, 1, 1, 2, 2, 1, 3, 3, 3, 4, 3, 3, 2, 2, 0, 1, 4, 3, 1, 1, 4, 1, 2, 4, 4, 1, 4, 3, 2, 3, 2, 1, 0, 0, 3, 3, 0, 2, 2, 0, 1, 1, 1, 1, 0, 2, 2, 2, 2, 4, 3, 1, 2, 1, 0, 3, 2, 3, 3, 2, 3, 2, 0, 2, 3, 3, 0, 1, 1, 3, 1, 4, 1, 2, 0, 0, 3, 4, 2, 1, 2, 3, 4, 2, 0, 2, 2, 3, 4, 1, 0, 1, 2, 2, 3, 0, 4, 3, 3, 3, 3, 1, 3, 0, 0, 3, 3, 2, 3, 2, 2, 2, 2, 0, 3, 4, 4, 4, 0, 4, 4, 1, 4, 2, 0, 0, 3, 1, 1, 2, 2, 3, 3, 3, 2, 1, 2, 3, 2, 1, 4, 4, 1, 1, 3, 4, 3, 2, 2, 1, 0, 4, 0, 0, 2, 2, 3, 0, 0, 1, 3, 3, 1, 4, 1, 3, 4, 3, 1, 4, 1, 3, 4, 1, 1, 4, 1, 3, 3, 1, 1, 0, 0, 1, 2, 0, 2, 1, 0, 4, 3, 2, 3, 2, 3, 2, 3, 2, 4, 4, 3, 0, 1, 4, 3, 3, 4, 3, 1, 4, 2, 2, 2, 4, 4, 4, 2, 3, 1, 2, 4, 4, 2, 3, 2, 4, 1, 2, 1, 4, 2, 3, 1, 4, 1, 3, 4, 0, 1, 1, 0, 2, 4, 0, 3, 1, 3, 2, 4, 3, 3, 1, 1, 0, 2, 0, 1, 2, 3, 2, 2, 2, 0, 0, 0, 0, 3, 2, 0, 4, 1, 3, 3, 1, 1, 2, 4, 3, 0, 3, 0, 4, 1, 4, 3, 0, 3, 3, 1, 3, 4, 3, 3, 3, 0, 3, 1, 4, 3, 2, 3, 4, 3, 1, 1, 1, 2, 3, 4, 1, 2, 3, 1, 3, 2, 0, 3, 2, 3, 2, 3, 2, 1, 4, 3, 1, 2, 4, 1, 2, 1, 0, 2, 0, 2, 4, 2, 3, 1, 0, 0, 4, 3, 1, 1, 1, 4, 1, 4, 2, 1, 1, 1, 1, 3, 0, 1, 4, 4, 1, 0, 1, 0, 2, 4, 1, 0, 0, 3, 0, 1, 0, 2, 3, 3, 0, 2, 3, 1, 1, 3, 1, 4, 3, 4, 0, 0, 4, 4, 1, 4, 2, 1, 3, 4, 2, 3, 4, 4, 1, 0, 1, 4, 4, 2, 1, 3, 2, 4, 3, 4, 1, 0, 3, 1, 4, 3, 2, 2, 3, 3, 3, 3, 4, 2, 1, 3, 3, 3, 3, 1, 1, 2, 3, 0, 1, 0, 2, 3, 3, 4, 2, 1, 2, 1, 3, 1, 1, 4, 3, 1, 0, 1, 3, 3, 2, 3, 3, 3, 3, 2, 3, 1, 4, 2, 0, 1, 1, 2, 1, 3, 3, 1, 1, 3, 1, 1, 1, 3, 2, 1, 4, 3, 3, 3, 4, 1, 0, 1, 0, 4, 2, 4, 0, 2, 3, 3, 2, 4, 4, 0, 1, 4, 1, 0, 3, 2, 2, 3, 3, 3, 0, 3, 3, 4, 0, 3, 3, 2, 3, 4, 3, 4, 1, 4, 2, 2, 3, 1, 1, 0, 4, 3, 1, 1, 3, 0, 2, 1, 4, 2, 4, 2, 1, 2, 1, 3, 1, 0, 1, 3, 4, 4, 1, 0, 4, 2, 1, 2, 4, 2, 2, 4, 4, 3, 1, 3, 2, 1, 1, 3, 2, 0, 3, 2, 3, 2, 4, 4, 1, 2, 3, 1, 1, 0, 1, 3, 3, 3, 3, 4, 2, 2, 1, 2, 1, 4, 3, 3, 3, 4, 1, 4, 2, 2, 1, 4, 3, 0, 4, 3, 4, 1, 3, 2, 1, 4, 2, 4, 0, 2, 2, 1, 1, 2, 3, 3, 1, 0, 3, 1, 3, 3, 1, 2, 3, 4, 2, 1, 0, 2, 1, 2, 4, 0, 3, 1, 4, 0, 3, 1, 2, 1, 1, 4, 4, 2, 2, 2, 2, 4, 2, 4, 0, 0, 0, 1, 0, 3, 0, 4, 1, 2, 2, 3, 4, 3, 3, 1, 4, 3, 0, 1, 1, 1, 4, 3, 4, 4, 3, 2, 4, 3, 3, 4, 3, 1, 0, 1, 0, 2, 3, 3, 3, 2, 0, 0, 3, 3, 0, 2, 2, 0, 2, 1, 3, 2, 3, 2, 3, 2, 3, 2, 1, 3, 2, 3, 3, 3, 4, 1, 3, 2, 2, 1, 0, 2, 0, 4, 0, 1, 3, 2, 2, 4, 4, 3, 3, 3, 2, 4, 3, 3, 2, 0, 0, 2, 2, 4, 2, 4, 3, 2, 0, 3, 1, 3, 2, 2, 4, 4, 1, 4, 0, 1, 2, 0, 3, 0, 1, 4, 2, 3, 2, 0, 4, 1, 2, 1, 0, 3, 2, 3, 3, 2, 3, 1, 2, 4, 4, 3, 2, 4, 4, 4, 3, 4, 1, 3, 2, 3, 3, 1, 3, 1, 4, 1, 2, 3, 4, 2, 1, 1, 0, 1, 2, 3, 2, 0, 1, 2, 1, 1, 1, 1, 2, 2, 0, 3, 3, 3, 3, 0, 1, 4, 2, 4, 4, 0, 1, 3, 4, 2, 3, 2, 0, 2, 2, 4, 3, 3, 2, 4, 1, 0, 0, 4, 1, 2, 0, 2, 3, 4, 2, 3, 2, 2, 3, 3, 3, 4, 3, 2, 3, 1, 2, 1, 3, 4, 3, 4, 2, 1, 1, 4, 0, 2, 4, 1, 4, 3, 4, 4, 3, 2, 2, 0, 3, 2, 3, 3, 1, 0, 2, 3, 0, 4, 2, 3, 0, 0, 4, 0, 0, 4, 1, 3, 1, 3, 4, 2, 1, 3, 3, 0, 2, 1, 4, 2, 2, 1, 3, 1, 2, 2, 4, 0, 4, 1, 1, 1, 3, 0, 2, 4, 0, 0, 1, 3, 2, 3, 2, 3, 1, 1, 0, 1, 1, 0, 4, 1, 2, 2, 3, 1, 4, 4, 1, 4, 0, 1, 3, 2, 3, 1, 3, 2, 2, 1, 2, 1, 4, 1, 3, 4, 3, 1, 3, 4, 1, 3, 2, 3, 4, 0, 0, 1, 3, 2, 3, 0, 3, 0, 2, 3, 1, 4, 0, 4, 1, 1, 0, 3, 1, 4, 3, 4, 4, 3, 3, 2, 1, 1, 2, 3, 3, 1, 3, 1, 1, 2, 3, 1, 0, 3, 0, 3, 3, 4, 2, 0, 1, 2, 1, 4, 4, 4, 0, 1, 0, 1, 2, 1, 1, 3, 4, 3, 1, 4, 1, 2, 2, 0, 2, 0, 3, 0, 2, 0, 1, 4, 2, 1, 1, 4, 2, 2, 4, 3, 3, 3, 2, 3, 2, 2, 2, 0, 1, 2, 0, 0, 2, 2, 1, 3, 3, 0, 2, 3, 2, 3, 3, 1, 0, 2, 1, 3, 2, 0, 2, 2, 2, 1, 4, 2, 1, 2, 3, 2, 4, 0, 3, 0, 4, 1, 3, 3, 2, 3, 2, 4, 1, 1, 2, 1, 1, 0, 3, 2, 0, 2, 3, 2, 4, 3, 2, 4, 2, 3, 0, 1, 2, 1, 3, 1, 3, 1, 4, 4, 3, 2, 1, 0, 1, 1, 2, 4, 4, 2, 2, 3, 4, 2, 4, 2, 0, 3, 2, 3, 4, 0, 2, 2, 4, 0, 4, 1, 0, 4, 2, 0, 1, 4, 1, 2, 2, 1, 2, 2, 3, 1, 4, 3, 4, 3, 3, 3, 4, 1, 3, 4, 3, 0, 0, 2, 3, 2, 2, 2, 3, 1, 4, 0, 3, 4, 2, 3, 4, 0, 3, 4, 1, 4, 1, 2, 0, 4, 0, 4, 1, 2, 2, 2, 4, 1, 3, 1, 1, 2, 3, 4, 1, 3, 3, 2, 3, 4, 2, 1, 0, 2, 3, 3, 2, 2, 3, 3, 2, 0, 1, 2, 4, 4, 3, 1, 3, 2, 0, 4, 0, 1, 2, 2, 4, 0, 1, 0, 1, 2, 1, 3, 3, 3, 0, 3, 1, 4, 2, 2, 2, 3, 3, 1, 2, 2, 2, 0, 0, 2, 0, 1, 2, 1, 4, 1, 0, 2, 0, 4, 0, 3, 0, 4, 0, 4, 3, 0, 2, 4, 2, 1, 4, 2, 3, 2, 1, 0, 3, 3, 3, 3, 2, 3, 3, 2, 2, 3, 2, 1, 0, 2, 2, 2, 1, 1, 2, 1, 3, 3, 3, 1, 2, 1, 4, 1, 4, 3, 1, 3, 3, 3, 1, 1, 1, 2, 3, 1, 0, 2, 0, 3, 3, 2, 2, 4, 3, 3, 0, 1, 1, 1, 1, 3, 2, 2, 2, 3, 2, 2, 0, 1, 1, 3, 2, 1, 4, 3, 2, 2, 3, 0, 3, 3, 1, 4, 3, 3, 2, 3, 2, 1, 4, 3, 3, 2, 2, 3, 3, 3, 0, 4, 3, 3, 4, 3, 0, 3, 1, 4, 3, 3, 2, 4, 2, 4, 4, 2, 2, 4, 3, 3, 1, 3, 2, 1, 4, 3, 2, 3, 0, 1, 4, 0, 3, 3, 0, 0, 3, 2, 2, 3, 3, 3, 0, 2, 4, 3, 0, 1, 4, 2, 1, 4, 4, 2, 2, 3, 3, 4, 1, 0, 4, 2, 2, 3, 4, 4, 1, 0, 2, 4, 3, 3, 3, 2, 2, 3, 2, 4, 2, 4, 0, 2, 1, 1, 4, 1, 3, 1, 4, 1, 1, 4, 2, 1, 0, 0, 3, 2, 3, 3, 3, 2, 0, 2, 2, 1, 2, 2, 2, 0, 0, 0, 1, 3, 1, 2, 2, 0, 0, 3, 1, 4, 0, 3, 0, 3, 0, 4, 2, 4, 3, 0, 3, 4, 2, 3, 3, 3, 2, 4, 3, 2, 1, 4, 0, 4, 3, 2, 2, 3, 2, 3, 4, 2, 0, 2, 1, 2, 3, 4, 3, 1, 1, 4, 2, 3, 2, 1, 4, 0, 0, 2, 4, 4, 1, 3, 4, 4, 0, 2, 2, 0, 0, 3, 3, 4, 1, 3, 1, 4, 3, 2, 1, 4, 2, 3, 2, 1, 2, 4, 3, 3, 3, 1, 3, 2, 1, 4, 3, 0, 3, 1, 4, 0, 4, 2, 1, 1, 3, 4, 3, 4, 2, 4, 3, 2, 3, 3, 0, 0, 3, 0, 3, 3, 0, 1, 1, 0, 0, 3, 0, 3, 0, 0, 0, 0, 2, 1, 4, 0, 1, 3, 2, 0, 2, 3, 4, 1, 3, 1, 0, 3, 2, 3, 1, 0, 3, 3, 0, 0, 2, 1, 2, 0, 1, 2, 1, 4, 0, 4, 3, 1, 1, 4, 0, 2, 1, 3, 4, 4, 3, 2, 3, 3, 1, 4, 3, 3, 0, 4, 2, 2, 2, 3, 2, 1, 2, 2, 4, 3, 2, 3, 0, 4, 3, 2, 1, 1, 3, 3, 4, 1, 4, 2, 2, 2, 0, 0, 3, 4, 1, 3, 3, 2, 3, 3, 3, 3, 2, 2, 1, 1, 2, 1, 3, 4, 4, 1, 2, 0, 3, 1, 0, 4, 1, 2, 1, 4, 2, 2, 0, 3, 3, 2, 1, 2, 2, 3, 3, 3, 0, 1, 2, 1, 1, 4, 4, 2, 3, 2, 1, 3, 4, 0, 1, 2, 1, 3, 0, 3, 3, 3, 2, 1, 1, 1, 2, 3, 0, 3, 2, 1, 4, 2, 1, 1, 2, 1, 2, 3, 0, 2, 3, 0, 4, 3, 4, 4, 3, 1, 3, 2, 3, 1, 4, 3, 2, 3, 4, 2, 3, 3, 2, 1, 3, 3, 0, 3, 4, 1, 3, 4, 0, 4, 3, 2, 3, 3, 3, 1, 1, 3, 2, 2, 2, 3, 3, 4, 2, 4, 1, 3, 1, 3, 2, 4, 4, 3, 3, 2, 0, 2, 1, 4, 1, 4, 1, 3, 0, 2, 0, 0, 0, 4, 2, 0, 4, 4, 3, 1, 0, 1, 3, 3, 4, 2, 3, 4, 2, 1, 2, 3, 4, 2, 3, 4, 2, 3, 3, 4, 2, 3, 0, 3, 2, 1, 3, 3, 2, 0, 4, 1, 0, 2, 1, 2, 4, 4, 1, 2, 1, 4, 4, 3, 3, 2, 4, 0, 3, 2, 2, 1, 3, 4, 2, 0, 4, 1, 2, 2, 3, 0, 2, 4, 3, 3, 0, 3, 4, 0, 4, 2, 2, 3, 1, 3, 0, 4, 3, 4, 3, 3, 3, 3, 1, 0, 4, 1, 3, 4, 4, 3, 4, 1, 3, 4, 3, 4, 4, 1, 2, 0, 3, 3, 3, 4, 4, 3, 2, 3, 4, 1, 3, 2, 3, 2, 4, 2, 3, 3, 4, 3, 2, 1, 2, 0, 2, 3, 3, 1, 1, 3, 3, 4, 0, 0, 3, 1, 3, 3, 4, 1, 1, 1, 1, 4, 2, 2, 2, 2, 1, 1, 3, 4, 4, 1, 2, 4, 2, 4, 2, 4, 3, 0, 3, 4, 4, 0, 3, 2, 4, 3, 4, 4, 1, 2, 2, 2, 0, 1, 2, 3, 2, 3, 4, 2, 3, 0, 2, 0, 2, 1, 3, 0, 2, 3, 4, 3, 2, 2, 1, 1, 3, 3, 1, 1, 1, 2, 3, 3, 0, 0, 4, 3, 2, 4, 2, 4, 2, 1, 1, 1, 2, 1, 1, 3, 2, 4, 3, 1, 1, 0, 4, 2, 3, 2, 4, 4, 4, 1, 1, 0, 3, 2, 0, 3, 3, 1, 0, 3, 3, 3, 0, 1, 2, 1, 3, 2, 2, 4, 2, 3, 2, 2, 4, 2, 2, 1, 3, 3, 2, 0, 3, 0, 3, 2, 4, 2, 1, 2, 3, 3, 3, 1, 3, 1, 0, 0, 2, 1, 2, 1, 0, 1, 1, 3, 3, 0, 2, 4, 4, 2, 0, 2, 1, 3, 0, 3, 4, 1, 1, 4, 2, 3, 3, 4, 1, 1, 3, 3, 3, 3, 4, 0, 2, 2, 3, 0, 3, 1, 3, 2, 0, 3, 4, 3, 3, 3, 2, 2, 2, 2, 2, 0, 3, 4, 4, 4, 0, 3, 2, 3, 3, 3, 1, 3, 1, 3, 3, 3, 2, 3, 2, 3, 1, 4, 4, 4, 3, 0, 1, 2, 1, 0, 0, 2, 1, 1, 0, 4, 4, 4, 4, 3, 3, 4, 4, 2, 1, 2, 3, 3, 3, 3, 1, 3, 4, 3, 2, 0, 0, 3, 2, 2, 3, 4, 1, 0, 0, 3, 2, 0, 3, 1, 4, 3, 3, 2, 4, 3, 1, 3, 2, 2, 3, 4, 1, 3, 4, 0, 4, 2, 2, 2, 0, 2, 0, 3, 1, 1, 3, 3, 3, 3, 3, 2, 2, 3, 0, 1, 3, 1, 3, 3, 3, 3, 2, 3, 0, 0, 1, 3, 1, 2, 2, 0, 0, 2, 3, 2, 3, 1, 1, 2, 2, 3, 4, 0, 2, 3, 3, 1, 0, 2, 0, 0, 2, 3, 0, 2, 4, 3, 1, 0, 1, 2, 0, 3, 4, 4, 1, 4, 3, 3, 1, 1, 1, 1, 1, 4, 2, 3, 3, 0, 4, 3, 1, 2, 1, 0, 3, 1, 3, 1, 2, 4, 4, 1, 3, 4, 3, 0, 1, 3, 2, 0, 2, 1, 2, 2, 3, 4, 3, 2, 1, 2, 0, 3, 1, 2, 0, 4, 2, 0, 1, 2, 0, 3, 0, 3, 2, 0, 1, 2, 4, 4, 3, 3, 3, 0, 2, 3, 0, 0, 0, 0, 1, 4, 3, 3, 3, 0, 4, 1, 3, 0, 0, 3, 0, 3, 2, 1, 2, 1, 4, 2, 3, 1, 4, 1, 0, 4, 1, 3, 4, 1, 2, 4, 1, 3, 2, 2, 3, 3, 3, 4, 1, 3, 4, 1, 2, 0, 3, 1, 0, 1, 2, 4, 1, 3, 4, 2, 4, 2, 4, 1, 1, 3, 2, 2, 3, 1, 3, 2, 1, 1, 3, 1, 2, 4, 4, 4, 3, 1, 1, 2, 0, 1, 1, 4, 3, 2, 1, 3, 2, 2, 2, 1, 1, 4, 4, 0, 3, 1, 1, 2, 2, 3, 1, 0, 4, 2, 2, 1, 4, 4, 2, 1, 3, 1, 1, 3, 2, 2, 1, 1, 1, 2, 4, 1, 3, 4, 1, 1, 1, 0, 1, 1, 2, 4, 2, 3, 3, 0, 1, 0, 3, 4, 3, 0, 1, 3, 0, 4, 3, 1, 4, 1, 1, 3, 3, 1, 0, 4, 3, 2, 3, 3, 0, 2, 3, 2, 1, 3, 0, 4, 3, 4, 3, 4, 0, 0, 0, 1, 2, 4, 2, 3, 1, 0, 3, 4, 2, 3, 0, 3, 3, 0, 2, 1, 0, 1, 2, 0, 1, 0, 3, 1, 4, 2, 2, 0, 3, 4, 2, 3, 3, 3, 1, 2, 0, 1, 2, 2, 1, 1, 1, 3, 2, 1, 4, 0, 1, 1, 4, 2, 0, 4, 4, 1, 4, 4, 4, 4, 1, 4, 3, 1, 3, 2, 3, 2, 3, 4, 0, 4, 3, 3, 4, 2, 2, 4, 2, 3, 0, 1, 2, 1, 1, 3, 3, 1, 3, 3, 2, 4, 4, 3, 3, 2, 3, 1, 4, 3, 3, 3, 1, 2, 4, 3, 0, 0, 2, 1, 2, 3, 2, 1, 3, 4, 3, 2, 2, 2, 1, 0, 3, 3, 0, 2, 2, 4, 2, 1, 0, 1, 0, 3, 2, 1, 2, 1, 2, 2, 2, 4, 1, 0, 2, 1, 3, 3, 3, 3, 3, 0, 2, 1, 3, 2, 3, 4, 3, 4, 2, 2, 3, 0, 3, 1, 1, 4, 4, 2, 4, 2, 3, 1, 3, 3, 2, 2, 3, 1, 3, 4, 1, 1, 4, 1, 1, 2, 2, 1, 3, 3, 2, 0, 2, 3, 1, 3, 2, 1, 3, 1, 2, 3, 2, 1, 1, 1, 4, 4, 1, 2, 0, 4, 0, 3, 1, 2, 0, 2, 2, 3, 1, 1, 3, 4, 3, 2, 3, 2, 2, 2, 3, 3, 1, 1, 1, 4, 0, 3, 3, 1, 1, 1, 1, 2, 4, 1, 3, 2, 3, 0, 2, 2, 3, 0, 4, 2, 2, 3, 0, 1, 0, 2, 3, 3, 3, 3, 3, 1, 1, 3, 2, 3, 2, 2, 2, 4, 1, 4, 1, 2, 3, 3, 3, 4, 0, 4, 4, 3, 1, 2, 3, 2, 3, 4, 4, 1, 4, 3, 4, 2, 2, 4, 3, 3, 0, 4, 3, 1, 1, 4, 2, 2, 2, 3, 0, 2, 2, 1, 1, 2, 3, 2, 1, 1, 2, 1, 1, 2, 3, 1, 2, 2, 3, 4, 0, 3, 2, 2, 1, 4, 3, 0, 3, 2, 0, 3, 3, 1, 3, 3, 3, 2, 2, 1, 0, 0, 0, 0, 0, 4, 1, 3, 3, 4, 3, 3, 0, 3, 3, 1, 3, 3, 1, 2, 4, 3, 3, 1, 1, 4, 2, 2, 3, 4, 1, 4, 0, 4, 1, 2, 3, 1, 3, 2, 3, 1, 1, 0, 0, 3, 3, 4, 3, 3, 4, 3, 4, 1, 3, 3, 0, 1, 1, 4, 1, 2, 3, 3, 3, 3, 3, 4, 0, 1, 3, 2, 1, 0, 3, 0, 4, 1, 3, 4, 3, 3, 3, 4, 1, 1, 1, 2, 4, 1, 1, 1, 3, 2, 4, 2, 3, 3, 1, 1, 1, 0, 0, 2, 2, 2, 1, 1, 1, 2, 1, 1, 2, 1, 2, 4, 3, 2, 2, 2, 2, 2, 3, 1, 2, 4, 3, 3, 3, 4, 0, 4, 0, 3, 2, 3, 3, 4, 3, 4, 0, 3, 0, 0, 1, 3, 2, 3, 0, 2, 4, 2, 3, 3, 1, 3, 2, 1, 4, 2, 2, 2, 1, 1, 1, 1, 4, 0, 3, 1, 3, 4, 3, 4, 3, 0, 4, 1, 1, 1, 3, 2, 3, 3, 3, 4, 2, 0, 4, 4, 2, 0, 2, 1, 2, 0, 2, 3, 1, 3, 2, 3, 3, 4, 1, 1, 0, 3, 3, 2, 3, 4, 3, 2, 0, 4, 4, 2, 3, 3, 0, 0, 3, 1, 1, 2, 2, 3, 1, 4, 3, 2, 3, 0, 3, 4, 2, 4, 1, 2, 0, 1, 1, 3, 1, 2, 4, 1, 0, 4, 4, 2, 0, 2, 0, 0, 1, 1, 4, 1, 1, 2, 3, 1, 4, 1, 4, 1, 2, 3, 1, 3, 2, 3, 2, 0, 1, 0, 1, 3, 0, 2, 0, 2, 1, 3, 4, 3, 0, 2, 1, 0, 0, 1, 1, 2, 3, 4, 2, 2, 1, 0, 1, 2, 3, 4, 1, 1, 1, 3, 2, 0, 1, 2, 2, 1, 4, 3, 4, 2, 1, 3, 3, 4, 0, 1, 4, 2, 0, 3, 2, 1, 3, 3, 3, 0, 1, 1, 0, 3, 3, 3, 3, 3, 0, 4, 2, 0, 3, 2, 2, 4, 1, 1, 0, 4, 2, 4, 3, 2, 2, 3, 3, 2, 1, 1, 1, 1, 3, 3, 4, 3, 3, 3, 2, 2, 2, 2, 3, 2, 1, 1, 2, 4, 0, 4, 0, 3, 1, 1, 3, 0, 0, 4, 3, 3, 2, 0, 3, 1, 3, 4, 2, 1, 1, 1, 3, 3, 2, 4, 4, 2, 4, 3, 1, 3, 3, 0, 1, 4, 2, 4, 4, 0, 3, 3, 1, 3, 4, 2, 4, 0, 1, 2, 3, 3, 1, 3, 0, 4, 3, 4, 4, 3, 2, 0, 1, 0, 2, 2, 4, 2, 3, 1, 3, 2, 4, 3, 3, 1, 1, 3, 3, 4, 1, 2, 0, 2, 2, 1, 2, 0, 0, 2, 2, 3, 3, 2, 3, 3, 0, 0, 2, 3, 3, 4, 4, 1, 1, 2, 3, 4, 0, 2, 3, 1, 1, 0, 1, 3, 1, 3, 4, 1, 3, 2, 1, 3, 1, 1, 3, 2, 3, 3, 3, 3, 3, 3, 4, 3, 1, 0, 3, 2, 2, 4, 2, 1, 1, 4, 1, 1, 1, 4, 1, 3, 3, 3, 0, 3, 1, 2, 3, 4, 4, 2, 1, 1, 2, 2, 3, 2, 1, 0, 4, 1, 4, 2, 1, 1, 3, 0, 1, 2, 2, 2, 3, 3, 3, 4, 0, 1, 1, 0, 0, 0, 4, 3, 2, 0, 2, 3, 2, 0, 1, 2, 4, 3, 2, 3, 2, 2, 2, 3, 3, 1, 2, 3, 1, 3, 1, 4, 0, 3, 3, 2, 0, 4, 4, 1, 1, 2, 2, 1, 0, 1, 0, 1, 3, 3, 2, 1, 3, 0, 0, 3, 3, 4, 1, 2, 2, 1, 2, 0, 2, 4, 1, 2, 4, 4, 3, 2, 1, 2, 2, 0, 1, 1, 3, 4, 3, 3, 3, 3, 0, 2, 2, 4, 2, 0, 1, 1, 3, 4, 1, 2, 3, 2, 2, 4, 3, 1, 2, 1, 0, 2, 2, 4, 4, 2, 3, 4, 2, 0, 3, 3, 2, 3, 2, 3, 4, 2, 4, 3, 3, 1, 4, 2, 4, 0, 0, 1, 4, 3, 3, 1, 3, 1, 2, 2, 2, 4, 1, 2, 3, 2, 3, 4, 0, 0, 4, 0, 3, 3, 0, 0, 4, 2, 2, 4, 2, 3, 0, 1, 1, 4, 3, 3, 4, 3, 0, 3, 1, 2, 3, 3, 4, 2, 4, 2, 0, 2, 4, 2, 3, 4, 1, 3, 3, 3, 4, 4, 1, 3, 3, 2, 1, 1, 2, 4, 2, 1, 2, 0, 2, 4, 4, 4, 3, 0, 4, 4, 0, 3, 2, 2, 1, 1, 3, 1, 2, 3, 0, 3, 1, 3, 1, 2, 3, 3, 0, 3, 1, 1, 0, 3, 2, 2, 0, 2, 3, 0, 2, 2, 1, 1, 1, 2, 1, 4, 1, 4, 1, 3, 4, 2, 2, 2, 0, 0, 3, 4, 1, 3, 0, 3, 1, 0, 3, 4, 3, 3, 1, 1, 4, 4, 2, 2, 0, 2, 0, 3, 4, 3, 3, 0, 2, 3, 2, 2, 1, 0, 2, 0, 2, 2, 2, 4, 1, 1, 4, 1, 1, 1, 0, 1, 0, 4, 2, 4, 2, 0, 3, 3, 3, 1, 3, 1, 3, 1, 3, 4, 4, 4, 3, 2, 0, 2, 0, 4, 3, 4, 2, 3, 3, 3, 1, 3, 3, 1, 1, 0, 3, 3, 3, 3, 0, 2, 3, 3, 0, 3, 3, 1, 4, 3, 3, 2, 3, 1, 4, 1, 1, 1, 4, 0, 2, 0, 3, 2, 0, 1, 0, 3, 3, 4, 4, 2, 1, 2, 2, 2, 1, 2, 2, 2, 0, 3, 3, 2, 3, 3, 1, 0, 0, 0, 2, 4, 1, 3, 2, 2, 0, 3, 3, 3, 0, 0, 1, 2, 4, 1, 3, 0, 3, 4, 2, 3, 0, 3, 1, 2, 1, 3, 3, 3, 3, 0, 4, 2, 4, 3, 1, 0, 4, 2, 4, 3, 2, 3, 2, 1, 1, 4, 3, 3, 4, 2, 0, 2, 2, 2, 2, 3, 1, 3, 3, 0, 3, 3, 1, 3, 2, 0, 2, 1, 3, 3, 3, 4, 2, 3, 3, 2, 4, 0, 4, 3, 0, 3, 0, 4, 3, 3, 1, 2, 3, 3, 3, 3, 4, 3, 3, 0, 1, 3, 3, 1, 3, 1, 1, 3, 3, 2, 3, 1, 3, 0, 0, 2, 4, 0, 1, 4, 3, 2, 0, 3, 3, 4, 4, 2, 2, 0, 4, 3, 3, 2, 4, 0, 0, 3, 1, 2, 3, 4, 1, 2, 3, 4, 3, 4, 4, 3, 3, 1, 1, 3, 3, 0, 0, 0, 1, 3, 3, 2, 4, 3, 3, 0, 0, 1, 2, 4, 0, 2, 1, 3, 2, 3, 3, 3, 1, 3, 4, 3, 2, 3, 1, 4, 3, 3, 3, 1, 4, 0, 2, 1, 3, 1, 3, 1, 4, 3, 1, 3, 4, 3, 4, 2, 4, 3, 2, 2, 1, 0, 1, 2, 2, 1, 1, 4, 1, 3, 3, 3, 0, 1, 1, 0, 1, 3, 2, 4, 3, 4, 3, 4, 3, 1, 2, 4, 3, 3, 4, 2, 2, 3, 3, 3, 3, 2, 4, 2, 4, 2, 1, 2, 1, 3, 0, 1, 4, 4, 1, 1, 3, 1, 3, 3, 1, 1, 0, 1, 3, 2, 3, 2, 3, 4, 1, 2, 3, 1, 3, 2, 1, 2, 1, 1, 1, 4, 3, 3, 3, 2, 3, 3, 4, 2, 0, 4, 1, 1, 2, 3, 1, 3, 3, 1, 4, 2, 2, 0, 2, 4, 1, 0, 1, 1, 4, 1, 4, 0, 1, 4, 3, 0, 2, 1, 1, 3, 2, 2, 1, 2, 0, 1, 3, 3, 0, 1, 3, 2, 3, 3, 3, 2, 0, 1, 0, 2, 0, 2, 2, 1, 2, 1, 2, 2, 0, 2, 4, 4, 0, 4, 1, 4, 0, 3, 3, 3, 2, 1, 1, 2, 0, 3, 2, 0, 2, 1, 4, 1, 3, 0, 4, 4, 1, 1, 2, 3, 2, 3, 0, 2, 4, 3, 2, 4, 1, 3, 1, 3, 2, 4, 0, 4, 0, 1, 2, 1, 4, 3, 3, 3, 0, 3, 0, 0, 3, 0, 3, 1, 4, 4, 1, 3, 2, 3, 2, 2, 1, 3, 4, 3, 3, 2, 2, 3, 3, 3, 3, 3, 2, 2, 0, 0, 3, 1, 3, 1, 0, 3, 3, 1, 3, 3, 4, 3, 0, 3, 2, 2, 1, 1, 3, 3, 0, 1, 3, 3, 2, 0, 4, 3, 4, 2, 1, 3, 1, 3, 3, 0, 3, 2, 1, 2, 3, 0, 3, 1, 0, 0, 3, 0, 4, 4, 3, 2, 0, 2, 4, 2, 0, 1, 4, 3, 0, 1, 3, 1, 2, 1, 3, 1, 1, 3, 4, 1, 0, 2, 4, 1, 1, 2, 0, 3, 1, 3, 4, 3, 3, 3, 2, 2, 0, 4, 4, 1, 1, 1, 2, 3, 4, 4, 0, 2, 0, 4, 4, 4, 1, 4, 4, 3, 4, 3, 2, 0, 3, 2, 2, 3, 4, 1, 3, 3, 3, 3, 2, 2, 2, 1, 3, 4, 3, 3, 4, 2, 4, 1, 2, 3, 3, 1, 4, 3, 1, 1, 1, 4, 2, 1, 1, 3, 4, 1, 2, 3, 0, 3, 2, 1, 1, 3, 1, 0, 3, 4, 1, 1, 2, 3, 2, 1, 4, 3, 4, 4, 3, 3, 4, 3, 3, 1, 2, 0, 3, 4, 0, 0, 2, 4, 4, 4, 2, 3, 3, 0, 3, 0, 3, 3, 2, 3, 3, 3, 3, 3, 2, 1, 4, 4, 0, 4, 0, 2, 4, 2, 4, 2, 3, 3, 3, 3, 0, 2, 3, 0, 0, 4, 0, 3, 4, 1, 0, 3, 2, 1, 2, 0, 3, 1, 3, 3, 3, 3, 2, 3, 0, 2, 1, 3, 4, 0, 3, 3, 2, 1, 2, 0, 2, 3, 0, 4, 3, 4, 4, 3, 4, 2, 2, 2, 3, 4, 2, 3, 3, 2, 2, 2, 4, 2, 3, 2, 2, 3, 0, 4, 2, 3, 3, 1, 1, 1, 1, 2, 3, 0, 0, 3, 2, 3, 2, 3, 2, 2, 1, 3, 2, 1, 3, 2, 3, 3, 4, 1, 4, 2, 3, 0, 3, 1, 2, 2, 1, 2, 1, 0, 2, 0, 2, 2, 1, 1, 2, 1, 3, 2, 1, 1, 1, 1, 1, 0, 4, 3, 1, 3, 1, 3, 3, 3, 4, 3, 1, 1, 4, 2, 4, 3, 2, 2, 1, 3, 4, 2, 3, 4, 3, 4, 4, 4, 0, 2, 3, 3, 3, 0, 0, 2, 3, 1, 3, 2, 2, 2, 3, 2, 4, 4, 4, 1, 0, 1, 3, 2, 2, 1, 4, 3, 4, 2, 4, 3, 4, 2, 0, 3, 3, 1, 2, 0, 2, 2, 0, 2, 0, 1, 3, 1, 3, 2, 3, 3, 3, 3, 2, 2, 4, 1, 1, 1, 2, 0, 2, 3, 3, 3, 2, 1, 2, 4, 2, 1, 2, 2, 2, 2, 2, 2, 3, 3, 2, 1, 1, 3, 2, 3, 2, 0, 0, 1, 1, 2, 3, 3, 2, 0, 1, 4, 2, 0, 3, 1, 3, 0, 2, 3, 2, 4, 3, 3, 1, 2, 1, 3, 2, 0, 3, 4, 2, 4, 1, 0, 3, 2, 3, 2, 4, 2, 2, 3, 1, 2, 1, 3, 1, 2, 4, 4, 1, 2, 1, 2, 2, 1, 4, 4, 2, 3, 3, 3, 0, 4, 3, 3, 2, 4, 4, 3, 1, 3, 3, 3, 2, 1, 0, 0, 2, 3, 3, 2, 4, 4, 2, 1, 3, 3, 4, 2, 3, 3, 2, 3, 4, 2, 2, 3, 1, 3, 0, 3, 3, 0, 1, 1, 1, 4, 3, 3, 4, 2, 0, 3, 2, 4, 3, 1, 1, 4, 3, 4, 0, 1, 2, 4, 2, 1, 3, 1, 4, 2, 4, 2, 0, 1, 4, 1, 3, 2, 4, 3, 2, 3, 3, 2, 2, 4, 2, 3, 3, 3, 4, 1, 3, 4, 1, 4, 3, 1, 4, 1, 0, 4, 2, 3, 0, 3, 0, 4, 3, 0, 2, 4, 0, 4, 2, 1, 3, 2, 3, 1, 4, 1, 3, 4, 2, 3, 2, 2, 3, 4, 2, 0, 2, 2, 0, 2, 4, 1, 1, 3, 3, 4, 0, 3, 2, 2, 3, 2, 3, 4, 0, 3, 4, 3, 1, 3, 3, 0, 0, 2, 0, 3, 3, 3, 3, 1, 1, 3, 2, 0, 1, 0, 0, 2, 4, 1, 3, 4, 3, 0, 3, 0, 3, 0, 2, 0, 3, 2, 1, 1, 3, 3, 2, 3, 2, 0, 1, 3, 3, 4, 4, 2, 2, 0, 3, 0, 1, 2, 4, 3, 3, 0, 1, 4, 2, 2, 3, 3, 1, 4, 4, 3, 1, 1, 3, 2, 4, 4, 4, 4, 2, 3, 0, 0, 2, 3, 4, 1, 2, 3, 4, 2, 3, 0, 4, 2, 2, 2, 3, 2, 1, 3, 2, 4, 2, 4, 3, 1, 0, 2, 4, 4, 0, 0, 0, 3, 1, 4, 3, 4, 2, 0, 3, 4, 1, 3, 3, 2, 4, 0, 2, 1, 2, 3, 0, 0, 0, 4, 0, 3, 3, 4, 3, 2, 3, 1, 3, 4, 0, 1, 2, 3, 3, 0, 3, 3, 0, 2, 3, 0, 3, 3, 0, 0, 0, 2, 1, 1, 3, 3, 4, 3, 2, 0, 0, 0, 2, 2, 3, 1, 2, 3, 1, 4, 3, 1, 3, 0, 0, 0, 4, 0, 1, 0, 3, 3, 3, 1, 3, 4, 2, 2, 1, 2, 3, 4, 3, 3, 4, 0, 1, 3, 2, 1, 3, 2, 2, 2, 3, 3, 2, 3, 4, 4, 3, 0, 2, 2, 4, 3, 2, 0, 4, 3, 1, 0, 4, 2, 3, 1, 2, 3, 1, 3, 0, 2, 0, 1, 2, 1, 2, 3, 1, 3, 4, 2, 3, 3, 3, 4, 1, 3, 1, 3, 3, 3, 1, 1, 0, 2, 3, 1, 2, 1, 3, 3, 4, 1, 3, 3, 4, 1, 2, 3, 3, 3, 3, 0, 2, 4, 3, 3, 4, 4, 1, 2, 3, 4, 2, 1, 1, 2, 3, 3, 1, 4, 4, 1, 2, 2, 4, 1, 3, 3, 2, 2, 3, 1, 3, 4, 3, 1, 3, 0, 3, 2, 2, 3, 0, 3, 2, 0, 1, 3, 3, 1, 2, 1, 4, 1, 4, 1, 3, 0, 3, 2, 3, 4, 3, 3, 1, 2, 2, 2, 0, 0, 4, 3, 0, 0, 1, 0, 1, 3, 3, 3, 4, 3, 2, 0, 1, 3, 2, 3, 0, 2, 3, 1, 1]
Final means
[15.924294964028777, 43.17971223021576, 1015.9507338129501, 63.29263309352524, 464.8839352517975]
[20.547363104731485, 55.6364540138225, 1012.8802498670921, 80.3443540669856, 449.9810260499738]
[28.083189448441217, 65.37588489208636, 1010.921534772184, 54.478364508393184, 438.5201918465224]
[10.766866064710298, 41.26446576373229, 1016.5765613243078, 85.25373589164798, 474.9493190368703]
[25.783037323037284, 70.1005727155726, 1008.7720012870026, 78.58686615186629, 436.3137323037332]
*/
