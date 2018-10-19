parameters {
    row_vector[2] theta;
}

model {
    row_vector[2] mu_1 = [2, 12];
    matrix[2, 2] Sigma_1 = [[2, 0.5], [0.5, 1]];
    
    row_vector[2] mu_2 = [10, -8];
    matrix[2, 2] Sigma_2 = [[1, 0.9], [0.9, 2]];
    
    row_vector[2] mu_3 = [-12, -9];
    matrix[2, 2] Sigma_3 = [[0.5, 0], [0, 0.5]];
    
    // target += multi_normal_lpdf(theta | mu_1, Sigma_1) / 2;
    // target += multi_normal_lpdf(theta | mu_2, Sigma_2) / 4;
    //target += multi_normal_lpdf(theta | mu_3, Sigma_3) / 4;
    
    theta ~ multi_normal(mu_1, Sigma_1);
    theta ~ multi_normal(mu_2, Sigma_2);
    theta ~ multi_normal(mu_3, Sigma_3);
}