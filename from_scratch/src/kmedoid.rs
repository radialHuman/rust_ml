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
        4. Calcualte new point that is closest to the mean
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

    let mut new_medioid: Vec<Vec<f64>> = vec![];
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
        new_medioid = vec![];
        for (m, _) in centroids.iter().enumerate() {
            let mut group = vec![];
            for i in clusters.iter() {
                if *i.0 == m {
                    group.push(i.1.clone());
                }
            }

            let mean = group
                .iter()
                .fold(vec![0.; k], |a, b| element_wise_operation(&a, b, "add"));

            for i in group
                .iter()
                .map(|a| nearest_point(a, &group))
                .collect::<Vec<Vec<f64>>>()
                .iter()
            {
                new_medioid.push(i.clone());
            }

            updated_cluster = clusters.clone()
        }
        println!("Iteration {:?}", x);
        if centroids == new_medioid {
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
            centroids = new_medioid.clone();
        }
    }
    print_a_matrix("Final medoids", &centroids);
}

pub fn nearest_point(mean: &Vec<f64>, points: &Vec<Vec<f64>>) -> Vec<f64> {
    // finding nearest point
    let mut nearest = points[0].clone();
    for i in points.iter() {
        let d = Distance {
            row1: mean.clone(),
            row2: i.clone(),
        };
        let n = Distance {
            row1: mean.clone(),
            row2: nearest.clone(),
        };
        if d.distance_euclidean() < n.distance_euclidean() {
            nearest = i.clone();
        }
        // println!("{:?},{:?} = {:?}", nearest, mean, n.distance_euclidean());
    }
    nearest
}

/*
RUST OUTPUT


*/
