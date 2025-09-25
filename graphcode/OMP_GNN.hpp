#ifndef OMP_GNN_HPP
#define OMP_GNN_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include<chrono>
#include <cmath>
#include <mkl.h>
#include<set>
#include <algorithm>
#include <random>
#include<time.h>
#include<omp.h>
#include<limits.h>
#include<unordered_map>
#include <unordered_set>
#include <numeric> 
#include <string>
#include <stdexcept>
#include <map>
#include <vector>
#include <algorithm>
#include <omp.h>



struct GCNContext_OMP
{

    std::vector<double> A_val;
    std::vector<int> A_row;
    std::vector<int> A_col;
    std::vector<int> labels;
    sparse_matrix_t A;

    std::vector<double> features;
    std::vector<std::vector<double>> W; 
    std::vector<std::vector<double>> Z; 
    std::vector<std::vector<double>> H; 
    std::vector<int> layer_dims; 
    std::vector<std::vector<double>> m; 
    std::vector<std::vector<double>> v;



    int num_nodes;
    int num_edges;
    int num_features;
    int num_classes;
    int num_layers;
    
};

GCNContext_OMP gcn_ctx_omp;
//Reading functions are defined herer

void read_graph(const std::string &filename, std::vector<int> &row_ptr, std::vector<int> &col_idx, std::vector<double> &weights, int &num_nodes)
{
    std::ifstream file(filename);
    if (!file)
    {
        std::cout << "Graph file " << filename << " doesn't exist or the filename is different" << std::endl;
        exit(1);
    }

    std::string index_zero;
    std::getline(file, index_zero);
    std::istringstream iss(index_zero);

    int a, b, c;
    bool is_weighted = false, is_edgelist = false;

    if (iss >> a >> b)
    {
        if (iss >> c)
        {
            is_edgelist = true;
            is_weighted = true;
        }
        else
        {
            is_edgelist = true;
        }
    }
    file.seekg(0);

    if (is_edgelist)
    {
        std::vector<std::pair<int, int>> edge_list;
        std::vector<int> edge_weights;
        int max_node = 0;

        std::string line;
        int line_number = 0;
        while (std::getline(file, line))
        {
            std::istringstream line_ss(line);
            int src, dst, weight = 1;

            if (!(line_ss >> src >> dst))
            {
                std::cerr << "Invalid edge format" << std::endl;
                exit(1);
            }

            if (is_weighted && !(line_ss >> weight))
            {
                std::cerr << "Edge weight is absent at line " << line_number << std::endl;
                exit(1);
            }

            std::string temp;
            if (line_ss >> temp)
            {
                std::cerr << "Too many values in edge" << std::endl;
                exit(1);
            }

            max_node = std::max(max_node, std::max(src, dst));

            // Undirected: add both directions
            edge_list.emplace_back(src, dst);
            edge_weights.push_back(weight);
            edge_list.emplace_back(dst, src);
            edge_weights.push_back(weight);

            ++line_number;
        }

        int num_nodes = max_node + 1;
        std::vector<std::vector<std::pair<int, int>>> adj(num_nodes);
        std::vector<int> degree(num_nodes, 0);

        for (size_t i = 0; i < edge_list.size(); ++i)
        {
            int src = edge_list[i].first;
            int dst = edge_list[i].second;
            int w = edge_weights[i];

            adj[src].emplace_back(dst, w);
        }

        // Add self-loop to each node
        for (int i = 0; i < num_nodes; ++i)
        {
            adj[i].emplace_back(i, 1);  // self-loop with weight 1
        }

        //Calculating degree
        for (int i = 0; i < num_nodes; ++i)
        {
            degree[i] = adj[i].size();
        }

        row_ptr.clear();
        col_idx.clear();
        weights.clear();
        row_ptr.push_back(0);

        for (int i = 0; i < num_nodes; ++i)
        {
            for (const auto &[dst, w] : adj[i])
            {
                double norm_w = static_cast<double>(w) / std::sqrt(degree[i] * degree[dst]);
                col_idx.push_back(dst);
                weights.push_back(norm_w);
            }
            row_ptr.push_back(col_idx.size());
        }
    }
    else
    {
        std::vector<std::vector<int>> matrix;
        std::string line;
        while (std::getline(file, line))
        {
            std::istringstream row_ss(line);
            std::vector<int> row;
            int val;
            while (row_ss >> val)
                row.push_back(val);
            matrix.push_back(row);
        }

        const size_t v = matrix.size();
        for (const auto &row : matrix)
        {
            if (row.size() != v)
            {
                std::cerr << "Non-square adjacency matrix" << std::endl;
                exit(1);
            }
        }

        // Add self-loop with weight 1
        for (size_t i = 0; i < v; ++i)
        {
            matrix[i][i] = 1;
        }

        // Compute degrees
        std::vector<int> degree(v, 0);
        for (size_t i = 0; i < v; ++i)
        {
            for (size_t j = 0; j < v; ++j)
            {
                if (matrix[i][j] != 0)
                    degree[i]++;
            }
        }

        row_ptr.clear();
        col_idx.clear();
        weights.clear();
        row_ptr.push_back(0);

        for (size_t i = 0; i < v; ++i)
        {
            for (size_t j = 0; j < v; ++j)
            {
                if (matrix[i][j] != 0)
                {
                    double norm_w = static_cast<double>(matrix[i][j]) / std::sqrt(degree[i] * degree[j]);
                    col_idx.push_back(j);
                    weights.push_back(norm_w);
                }
            }
            row_ptr.push_back(col_idx.size());
        }
    }

    gcn_ctx_omp.num_nodes = row_ptr.size() - 1; 
}


void read_features(const std::string &filename, std::vector<double> &features, int num_nodes, int &num_features)
{
    std::ifstream file(filename);
    if (!file)
    {
        std::cerr << "Error opening " << filename << std::endl;
        exit(1);
    }

    std::string line;
    std::vector<std::string> lines;
    std::vector<double> all_values;

    while (getline(file, line))
    {
        if (!line.empty())
        {
            lines.push_back(line);
            std::stringstream ss(line);
            double value;
            while (ss >> value)
            {
                all_values.push_back(value);
            }
        }
    }
    file.close();

    if (lines.size() == num_nodes)
    {
        // Row-wise format
        std::vector<std::vector<double>> feature_list;
        for (const auto &l : lines)
        {
            std::stringstream ss(l);
            double val;
            std::vector<double> row;
            while (ss >> val)
            {
                row.push_back(val);
            }
            feature_list.push_back(row);
        }

        num_features = feature_list[0].size();
        features.resize(num_nodes * num_features);

        for (int i = 0; i < num_nodes; i++)
        {
            for (int j = 0; j < num_features; j++)
            {
                features[i * num_features + j] = feature_list[i][j];
            }
        }
    }
    else if (all_values.size() % num_nodes == 0)
    {
        num_features = all_values.size() / num_nodes;
        features.resize(all_values.size());

        for (int i = 0; i < num_nodes; i++)
        {
            for (int j = 0; j < num_features; j++)
            {
                features[i * num_features + j] = all_values[i * num_features + j];
            }
        }
    }
    else
    {
        std::cerr << "Unable to determine format or mismatch in total values vs num_nodes." << std::endl;
        exit(1);
    }
}

void read_labels(const std::string &filename, std::vector<int> &labels, int &num_classes)
{
    std::ifstream file(filename);
    if (!file)
    {
        std::cerr << "Error opening " << filename << std::endl;
        exit(1);
    }

    int label;
    std::set<int> unique_labels;  
    while (file >> label)
    {
        labels.push_back(label);
        unique_labels.insert(label);  
    }

    file.close();
    num_classes = static_cast<int>(unique_labels.size());
    std::cout << "Number of classes are = " << num_classes << std::endl;
}


void adam(std::vector<double> &W, std::vector<double> &dW,
          std::vector<double> &m, std::vector<double> &v,
          int num_features, int hidden_dim,
          double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8,
          double learning_rate = 0.001, int t = 1)
{
    int size = num_features * hidden_dim;
    double beta1_t = pow(beta1, t);
    double beta2_t = pow(beta2, t);
    double bias_correction1 = 1.0 - beta1_t;
    double bias_correction2 = 1.0 - beta2_t;
    double one_minus_beta1 = 1.0 - beta1;
    double one_minus_beta2 = 1.0 - beta2;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < size; i++)
    {
        m[i] = beta1 * m[i] + one_minus_beta1 * dW[i];
 
        v[i] = beta2 * v[i] + one_minus_beta2 * dW[i] * dW[i];

        double m_hat = m[i] / bias_correction1;

        double v_hat = v[i] / bias_correction2;

        W[i] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
    }
}

void xavier_init(std::vector<double> &w, int in_dim, int out_dim)
{
    std::default_random_engine eng;
    std::uniform_real_distribution<double> dist(-std::sqrt(6.0) / std::sqrt(in_dim + out_dim),
                                                std::sqrt(6.0) / std::sqrt(in_dim + out_dim));
    for (auto &x : w)
        x = dist(eng);
}

void initialize_weights(std::vector<std::vector<double>> &W,
                        const std::vector<int> &layer_dims)
{
    int n_layers = layer_dims.size() - 1;
    W.resize(n_layers);
    for (int l = 0; l < n_layers; ++l)
    {
        int in_dim = layer_dims[l], out_dim = layer_dims[l + 1];
        W[l].resize(in_dim * out_dim);
        xavier_init(W[l], in_dim, out_dim);
    }
}

void gcn_forward_omp()
{

    std::copy(gcn_ctx_omp.features.begin(), gcn_ctx_omp.features.end(), gcn_ctx_omp.H[0].begin());
    int num_layers = gcn_ctx_omp.layer_dims.size() - 1;

    if (gcn_ctx_omp.H.size() != num_layers + 1 || gcn_ctx_omp.Z.size() != num_layers)
    {
        std::cerr << "Error: Vectors H and Z not properly initialized" << std::endl;
        return;
    }
    struct matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    for (int l = 0; l < num_layers; ++l)
    {
        int in_dim = gcn_ctx_omp.layer_dims[l];
        int out_dim = gcn_ctx_omp.layer_dims[l + 1];

        // temp = A * H[l]
        std::vector<double> temp(gcn_ctx_omp.num_nodes * in_dim, 0.0);
        mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE,
                        1.0, gcn_ctx_omp.A, descr,
                        SPARSE_LAYOUT_ROW_MAJOR,
                        gcn_ctx_omp.H[l].data(), in_dim, in_dim,
                        0.0, temp.data(), in_dim);

        // Z[l] = temp * W[l]
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    gcn_ctx_omp.num_nodes, out_dim, in_dim,
                    1.0, temp.data(), in_dim, gcn_ctx_omp.W[l].data(), out_dim,
                    0.0, gcn_ctx_omp.Z[l].data(), out_dim);

        // ReLU
        for (int i = 0; i < gcn_ctx_omp.num_nodes * out_dim; ++i)
        {
            gcn_ctx_omp.H[l + 1][i] = std::max(0.0, gcn_ctx_omp.Z[l][i]);
        }
    }
}

void gcn_backpropagation_omp(int epoch)
{
    epoch = epoch + 1; // Adam optimizer requires t to start from 1
    int L = gcn_ctx_omp.layer_dims.size() - 1;
    std::vector<std::vector<double>> dZ(L), dH(L);

    // calculating the gradient of the loss fpr output layer
    int out_dim = gcn_ctx_omp.layer_dims.back();
    dZ[L - 1].resize(gcn_ctx_omp.num_nodes * out_dim);

    // Softmax Cross-Entropy Loss Gradient
    std::vector<double> probs = gcn_ctx_omp.Z[L - 1];

    for (int i = 0; i < gcn_ctx_omp.num_nodes; i++)
    {
        double max_val = -1e9;
        for (int j = 0; j < out_dim; j++)
            max_val = std::max(max_val, gcn_ctx_omp.Z[L - 1][i * out_dim + j]);

        double sum_exp = 0.0;
        for (int j = 0; j < out_dim; j++)
        {
            probs[i * out_dim + j] = std::exp(gcn_ctx_omp.Z[L - 1][i * out_dim + j] - max_val);
            sum_exp += probs[i * out_dim + j];
        }

        for (int j = 0; j < out_dim; j++)
        {
            probs[i * out_dim + j] /= sum_exp;
            dZ[L - 1][i * out_dim + j] = probs[i * out_dim + j] - (j == gcn_ctx_omp.labels[i] ? 1.0 : 0.0);
        }
    }

    // propagating error backward
    for (int l = L - 1; l >= 0; --l)
    {
        int in_dim = gcn_ctx_omp.layer_dims[l];
        int out_dim = gcn_ctx_omp.layer_dims[l + 1];

        // Compute gradient of the loss w.r.t. the weights at this layer
        std::vector<double> dW(in_dim * out_dim, 0.0);
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    in_dim, out_dim, gcn_ctx_omp.num_nodes,
                    1.0, gcn_ctx_omp.H[l].data(), in_dim, dZ[l].data(), out_dim,
                    0.0, dW.data(), out_dim);

        // adam optimizer
        adam(gcn_ctx_omp.W[l], dW, gcn_ctx_omp.m[l], gcn_ctx_omp.v[l], in_dim, out_dim, 0.9, 0.999, 1e-8, 0.001, epoch);

        // Backpropagate the error to the previous layer
        if (l > 0)
        {
            dH[l - 1].resize(gcn_ctx_omp.num_nodes * in_dim);
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        gcn_ctx_omp.num_nodes, in_dim, out_dim,
                        1.0, dZ[l].data(), out_dim, gcn_ctx_omp.W[l].data(), out_dim,
                        0.0, dH[l - 1].data(), in_dim);

            dZ[l - 1].resize(gcn_ctx_omp.num_nodes * in_dim);
            for (int i = 0; i < gcn_ctx_omp.num_nodes * in_dim; ++i)
                dZ[l - 1][i] = gcn_ctx_omp.Z[l - 1][i] > 0 ? dH[l - 1][i] : 0.0; // activation derivative
        }
    }
}



double compute_loss()
{
    double loss = 0.0;
    for (int i = 0; i < gcn_ctx_omp.num_nodes; i++)
    {
        double max_val = -1e9;
        for (int j = 0; j < gcn_ctx_omp.layer_dims.back(); j++)
        {
            max_val = std::max(max_val, gcn_ctx_omp.Z.back()[i * gcn_ctx_omp.layer_dims.back() + j]);
        }

        double sum_exp = 0.0;
        for (int j = 0; j < gcn_ctx_omp.layer_dims.back(); j++)
        {
            sum_exp += std::exp(gcn_ctx_omp.Z.back()[i * gcn_ctx_omp.layer_dims.back() + j] - max_val);
        }

        double prob = std::exp(gcn_ctx_omp.Z.back()[i * gcn_ctx_omp.layer_dims.back() + gcn_ctx_omp.labels[i]] - max_val) / sum_exp;
        loss += -std::log(std::max(prob, 1e-10)); // to avoid log 0
    }

    return loss / gcn_ctx_omp.num_nodes;
}

double compute_accuracy()
{
    int correct = 0;
    for (int i = 0; i < gcn_ctx_omp.num_nodes; i++)
    {
        int pred = -1;
        double max_val = -1e9;
        for (int j = 0; j < gcn_ctx_omp.layer_dims.back(); j++)
        {
            double val = gcn_ctx_omp.Z.back()[i * gcn_ctx_omp.layer_dims.back() + j];
            if (val > max_val)
            {
                max_val = val;
                pred = j;
            }
        }
        if (pred == gcn_ctx_omp.labels[i])
            correct++;
    }
    return static_cast<double>(correct) / gcn_ctx_omp.num_nodes;
}








void init_omp(std::vector<int> &neuronsPerLayer, std::string initWeights , std::string folderPath)
{
    
    read_graph(folderPath + "_edgelist.txt",    gcn_ctx_omp.A_row , gcn_ctx_omp.A_col, gcn_ctx_omp.A_val, gcn_ctx_omp.num_nodes);

    read_features(folderPath + "_features.txt", gcn_ctx_omp.features, gcn_ctx_omp.num_nodes, gcn_ctx_omp.num_features);

    read_labels(folderPath + "_labels.txt", gcn_ctx_omp.labels, gcn_ctx_omp.num_classes);

    mkl_sparse_d_create_csr(&gcn_ctx_omp.A, SPARSE_INDEX_BASE_ZERO, gcn_ctx_omp.num_nodes-1, gcn_ctx_omp.num_nodes-1,
                            gcn_ctx_omp.A_row.data(), gcn_ctx_omp.A_row.data() + 1, gcn_ctx_omp.A_col.data(), gcn_ctx_omp.A_val.data());

    gcn_ctx_omp.layer_dims = neuronsPerLayer;
    gcn_ctx_omp.layer_dims[0] = gcn_ctx_omp.num_features;
    gcn_ctx_omp.layer_dims[gcn_ctx_omp.layer_dims.size() - 1] = gcn_ctx_omp.num_classes;  

    
    
    initialize_weights(gcn_ctx_omp.W, gcn_ctx_omp.layer_dims);

    int num_layers = gcn_ctx_omp.layer_dims.size() - 1;
    gcn_ctx_omp.Z.resize(num_layers);
    gcn_ctx_omp.H.resize(num_layers + 1);

    for (int l = 0; l < num_layers; ++l)
    {
        gcn_ctx_omp.Z[l].resize(gcn_ctx_omp.num_nodes * gcn_ctx_omp.layer_dims[l + 1], 0.0);
        gcn_ctx_omp.H[l].resize(gcn_ctx_omp.num_nodes * gcn_ctx_omp.layer_dims[l], 0.0);
    }
    gcn_ctx_omp.H[num_layers].resize(gcn_ctx_omp.num_nodes * gcn_ctx_omp.layer_dims[num_layers], 0.0);

    gcn_ctx_omp.m.resize(gcn_ctx_omp.W.size(), {});
    gcn_ctx_omp.v.resize(gcn_ctx_omp.W.size(), {});
    for (int i = 0; i < gcn_ctx_omp.W.size(); ++i)
    {
        gcn_ctx_omp.m[i].resize(gcn_ctx_omp.W[i].size(), 0.0);
        gcn_ctx_omp.v[i].resize(gcn_ctx_omp.W[i].size(), 0.0);
    }
}


void forward_omp(std::string modelType, std::string aggregationType)
{
    if(modelType=="GCN")
    {
        gcn_forward_omp();
    }
    // else if(modelType=="GAT")            //To be implemented by Saurabh
    // {
    //     GAT();
    // }
    else
    {
        std::cerr << "Unknown model type: " << modelType << std::endl;
        exit(1);
    }
}


void backprop_omp(std::string modelType, std::string aggregationType,int epoch)
{
   if(modelType=="GCN"){
    gcn_backpropagation_omp(epoch);
   }
    // else if(modelType=="GAT")                //To be implemented by Saurabh
    // {
    //     GAT();
    // }
    else
    {
        std::cerr << "Unknown model type: " << modelType << std::endl;
        exit(1);
    }
}


double compute_loss_omp()
{
    double loss = compute_loss();
    std::cout << "Loss: " << loss << std::endl;
    return loss;
}

double compute_accuracy_omp()
{
    double accuracy = compute_accuracy();
    std::cout << "Accuracy: " << accuracy << std::endl;
    return accuracy;
}


#endif 

// void test(graph& g , std::vector<int> neuronsPerLayer , int totalEpochs)
// {
//   GNN gnn;
//   gnn.init(neuronsPerLayer, "Xaviers", "");

//   for (int epochs = 0;
//     epochs < totalEpochs; epochs++) {

//     {
//       gnn.forward("GCN", "SUM");

//       gnn.backward("GCN", "SUM", epochs);

//     }

//   }


// }    // The test function is the reference on how the generated code would look like. The optimized is getting called from inside the gnn.backward function.




// int main(){              //This is how the main function would look like
//     std::vector<int> neuronsPerLayer = {16, 32, 64};
//     int totalEpochs = 100;
//     graph g("flickr/flickr_edgelist.txt");
//     g.parseGraph();
//
//     test(g, neuronsPerLayer, totalEpochs);
//     return 0;
// }


//g++ test_file.cc -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm  -O3  

// And to run on cluster use: "icc" instead of "g++" e.g. icc test_file.cc -O3

//Run the above command to compile and run the code. Make sure you have the necessary libraries installed and linked properly.