#include <iostream>
#include <string>
#include <armadillo>
#include <chrono>

using namespace std;
using namespace arma;

int main()
{
    // Load data from CSV file
    mat data;
    try {
        data.load("dFBA_Textbook_Model_Reaction_Fluxes.csv", csv_opts::delimiter('\t'));
    } catch (std::exception& e) {
        cerr << "Error: File not found or could not be opened." << endl;
        exit(1);
    }

    // Extract column names from header
    rowvec col_names = conv_to<rowvec>::from(data.row(0)).subvec(1, data.n_cols-1);

    // Drop first column ('Time')
    data = data.cols(1, data.n_cols-1);

    // Remove missing values (if any)
    //data = data.each_col([](vec& col){ col.replace(datum::nan, 0.0); });

    // Scale the data
    mat scaled_data = (data.each_row() - mean(data, 0)) / stddev(data, 0);

    // Normalize the data by columns
    mat normalized_data = (scaled_data.each_row() - min(scaled_data, 0)) / (max(scaled_data, 0) - min(scaled_data, 0));

    // Replace missing values with zeros
    normalized_data.elem(find_nonfinite(normalized_data)).fill(0.0);

    // Apply PCA
    auto start = chrono::high_resolution_clock::now();
    mat coeff, score, latent;
    princomp(coeff, score, latent, normalized_data);
    auto end = chrono::high_resolution_clock::now();

    // Compute explained variance ratio
    vec var_ratio = latent / sum(latent);

    // Compute cumulative explained variance
    vec cum_var = cumsum(var_ratio);

    // Plot scree plot
    int n_components = data.n_cols - 1;
    for (int i = 0; i < cum_var.n_elem; i++) {
        if (cum_var(i) >= 0.9) {
            n_components = i + 1;
            break;
        }
    }
    cout << "Number of components that explain 90% of variance: " << n_components << endl;

    // Perform PCA with n_components
    mat pca_data = score.cols(0, n_components-1);

    // Store PCA results in a new Armadillo matrix
    Mat<double> df_pca(pca_data.n_rows, pca_data.n_cols);
    for (int i = 0; i < pca_data.n_rows; i++) {
        for (int j = 0; j < pca_data.n_cols; j++) {
            df_pca(i, j) = pca_data(i, j);
        }
    }

    // Print the PCA results
    df_pca.print("PCA data:");

    // Compute and print the time taken to perform PCA
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "Time taken to perform PCA: " << duration.count() << " microseconds" << endl;

    return 0;
}
