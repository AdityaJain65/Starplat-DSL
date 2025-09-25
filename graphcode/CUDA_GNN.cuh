#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>
#include <fstream>
#include <sstream>
#include <cassert>
#include <set>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <curand.h>


#define CHECK_CUBLAS(call)                                                       \
    {                                                                            \
        cublasStatus_t err = call;                                               \
        if (err != CUBLAS_STATUS_SUCCESS)                                        \
        {                                                                        \
            fprintf(stderr, "cuBLAS Error: %d (%s) at %s:%d\n", err,             \
                    cublasGetErrorString(err), __FILE__, __LINE__);             \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    }

const char* cublasGetErrorString(cublasStatus_t status)
{
    switch (status)
    {
        case CUBLAS_STATUS_SUCCESS: return "Success";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "Not initialized";
        case CUBLAS_STATUS_ALLOC_FAILED: return "Allocation failed";
        case CUBLAS_STATUS_INVALID_VALUE: return "Invalid value";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "Architecture mismatch";
        case CUBLAS_STATUS_MAPPING_ERROR: return "Mapping error";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "Execution failed";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "Internal error";
        default: return "Unknown";
    }
}

#define CHECK_CUDA(call)                                                                               \
    do                                                                                                 \
    {                                                                                                  \
        cudaError_t err = call;                                                                        \
        if (err != cudaSuccess)                                                                        \
        {                                                                                              \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                                        \
        }                                                                                              \
    } while (0)

#define CHECK_CUSPARSE(call)                                                              \
    do                                                                                    \
    {                                                                                     \
        cusparseStatus_t status = call;                                                   \
        if (status != CUSPARSE_STATUS_SUCCESS)                                            \
        {                                                                                 \
            fprintf(stderr, "cuSPARSE Error: %d at %s:%d\n", status, __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    } while (0)

#define CHECK_CURAND(call)                                                              \
    do                                                                                  \
    {                                                                                   \
        curandStatus_t status = call;                                                   \
        if (status != CURAND_STATUS_SUCCESS)                                            \
        {                                                                               \
            fprintf(stderr, "cuRAND Error: %d at %s:%d\n", status, __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                         \
        }                                                                               \
    } while (0)



__global__ void adam_kernel(double *w, const double *dw, double *m, double *v, int size, 
                           double beta1, double beta2, double epsilon, double learning_rate, int t)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        double g = dw[idx];
        
        // Update biased first moment estimate
        m[idx] = beta1 * m[idx] + (1.0 - beta1) * g;
        
        // Update biased second raw moment estimate
        v[idx] = beta2 * v[idx] + (1.0 - beta2) * g * g;
        
        // Compute bias-corrected first moment estimate
        double m_hat = m[idx] / (1.0 - pow(beta1, t));
        
        // Compute bias-corrected second raw moment estimate
        double v_hat = v[idx] / (1.0 - pow(beta2, t));
        
        // Update parameters
        w[idx] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
    }
}

__global__ void relu_kernel(double *out, const double *in, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        out[idx] = max(0.0, in[idx]);
    }
}

__global__ void softmax_cross_entropy_grad_kernel(const double *z, const int *labels, double *dz,
                                                  int num_nodes, int out_dim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_nodes)
    {
        int start_idx = i * out_dim;

        // Find maximum for numerical stability
        double max_val = z[start_idx];
        for (int j = 1; j < out_dim; j++)
        {
            max_val = max(max_val, z[start_idx + j]);
        }

        // Compute softmax probabilities
        double sum_exp = 0.0;
        for (int j = 0; j < out_dim; j++)
        {
            double exp_val = exp(z[start_idx + j] - max_val);
            dz[start_idx + j] = exp_val;
            sum_exp += exp_val;
        }

        // Normalize and compute gradient
        for (int j = 0; j < out_dim; j++)
        {
            dz[start_idx + j] /= sum_exp; // This is now p_j
            if (j == labels[i])
            {
                dz[start_idx + j] -= 1.0; // p_j - y_j
            }
        }
        
        // DO NOT DIVIDE BY num_nodes HERE. REMOVE THE OLD LOOP.
    }
}



__global__ void relu_derivative_kernel(const double *z, const double *dh, double *dz, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        dz[idx] = (z[idx] > 0.0) ? dh[idx] : 0.0;
    }
}

// Kernel implementaiton ends here

struct GCNContext_CUDA
{
    cusparseHandle_t sp_handle;
    cublasHandle_t bl_handle;

    cusparseSpMatDescr_t A_descr;
    std::vector<cusparseDnMatDescr_t> H_descr;
    std::vector<cusparseDnMatDescr_t> Z_descr;
    std::vector<cusparseDnMatDescr_t> Temp_descr;

    double *A_val_d;
    int *A_row_d;
    int *A_col_d;
    double *features_d;

    std::vector<int> layer_dims;
    std::vector<double *> H_d;
    std::vector<double *> Z_d;
    std::vector<double *> W_d;
    std::vector<double *> Temp_d;


    std::vector<cusparseDnMatDescr_t> dH_descr;
    std::vector<cusparseDnMatDescr_t> dTemp_descr;


    int num_nodes;
    int num_edges;
    int num_features;
    int num_classes;
    void *dBuffer;
    double* temp_H_buffer;
    cusparseDnMatDescr_t temp_H_descr;
     double* temp_T_buffer;
    cusparseDnMatDescr_t temp_T_descr;
    std::vector<double *> m_d;
    std::vector<double *> v_d;
    int *labels_d = nullptr;
    double accuracy = 0.0;
    double loss = 0.0;
};

extern GCNContext_CUDA gcn_ctx_cuda_cuda;
GCNContext_CUDA gcn_ctx_cuda; // Global context for GCN


// Reading fucntions implementation starts here
void read_graph_omp(const std::string &filename, std::vector<int> &row_ptr, std::vector<int> &col_idx, std::vector<double> &edgeweights, int &num_nodes)
{
    std::ifstream file(filename);
    std::string line;
    int u, v;
    std::vector<std::vector<int>> adj;

    int max_node = -1;
    std::vector<std::pair<int, int>> edges;
    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        if (ss >> u >> v)
        {
            edges.push_back({u, v});
            max_node = std::max({max_node, u, v});
        }
    }
    file.close();

    num_nodes = max_node + 1;
    adj.resize(num_nodes);
    for (const auto &edge : edges)
    {
        adj[edge.first].push_back(edge.second);
        adj[edge.second].push_back(edge.first);
    }

    row_ptr.resize(num_nodes + 1, 0);
    for (int i = 0; i < num_nodes; ++i)
    {
        std::sort(adj[i].begin(), adj[i].end());
        adj[i].erase(std::unique(adj[i].begin(), adj[i].end()), adj[i].end());
        row_ptr[i + 1] = row_ptr[i] + adj[i].size();
        for (int neighbor : adj[i])
        {
            col_idx.push_back(neighbor);
        }
    }
    edgeweights.resize(col_idx.size(), 1.0); //
    std::cout << "Read Graph: Nodes=" << num_nodes << ", Edges=" << col_idx.size() << std::endl;
    gcn_ctx_cuda.num_nodes = num_nodes;
    gcn_ctx_cuda.num_edges = col_idx.size();
}

void read_features_omp(const std::string &filename, std::vector<double> &features, int num_nodes, int &num_features)
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

void read_labels_omp(const std::string &filename, std::vector<int> &labels, int &num_classes)
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


void xavier_init_cuda(double *w, int in_dim, int out_dim) {
    curandGenerator_t gen;
    CHECK_CURAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen, time(NULL)));

    int size = in_dim * out_dim;
    std::vector<double> w_host(size);

    std::default_random_engine eng(time(0));
    double limit = std::sqrt(6.0) / std::sqrt(in_dim + out_dim);
    std::uniform_real_distribution<double> dist(-limit, limit);

    for (auto &x : w_host) {
        x = dist(eng);
    }
    CHECK_CUDA(cudaMemcpy(w, w_host.data(), size * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CURAND(curandDestroyGenerator(gen));
}


void initialize_weights_cuda(std::vector<double*> &W_d, const std::vector<int> &layer_dims) {
    int n_layers = layer_dims.size() - 1;
    W_d.resize(n_layers);
    for (int l = 0; l < n_layers; ++l) {
        int in_dim = layer_dims[l];
        int out_dim = layer_dims[l + 1];
        CHECK_CUDA(cudaMalloc(&W_d[l], in_dim * out_dim * sizeof(double)));
        xavier_init_cuda(W_d[l], in_dim, out_dim);
    }
}

void GCN(){
    int num_layers = gcn_ctx_cuda.layer_dims.size() - 1;
    double alpha = 1.0;
    double beta = 0.0;

    for (int l = 0; l < num_layers; ++l) {
        int in_dim = gcn_ctx_cuda.layer_dims[l];
        int out_dim = gcn_ctx_cuda.layer_dims[l + 1];

        CHECK_CUSPARSE(cusparseSpMM(gcn_ctx_cuda.sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, gcn_ctx_cuda.A_descr, gcn_ctx_cuda.H_descr[l],&beta, gcn_ctx_cuda.Temp_descr[l], CUDA_R_64F,CUSPARSE_SPMM_ALG_DEFAULT, gcn_ctx_cuda.dBuffer));

        
        CHECK_CUBLAS(cublasDgemm(gcn_ctx_cuda.bl_handle, CUBLAS_OP_N, CUBLAS_OP_N,out_dim, gcn_ctx_cuda.num_nodes, in_dim,&alpha, gcn_ctx_cuda.W_d[l], out_dim, gcn_ctx_cuda.H_d[l], in_dim, &beta, gcn_ctx_cuda.Z_d[l], out_dim));

        int size = gcn_ctx_cuda.num_nodes * out_dim;
        int threadsPerBlock = 256;
        int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
        relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(gcn_ctx_cuda.H_d[l + 1], gcn_ctx_cuda.Z_d[l], size);
        CHECK_CUDA(cudaGetLastError());
        // CHECK_CUDA(cudaDeviceSynchronize()); 
    }
}


void GCN_backprop(int epoch){
     int L = gcn_ctx_cuda.layer_dims.size() - 1;

    std::vector<double*> dZ_d(L), dH_d(L);
    double alpha = 1.0;
    double beta = 0.0;
    for (int l = 0; l < L; ++l) {
        CHECK_CUDA(cudaMalloc(&dZ_d[l], gcn_ctx_cuda.num_nodes * gcn_ctx_cuda.layer_dims[l + 1] * sizeof(double)));

        if(l > 0) CHECK_CUDA(cudaMalloc(&dH_d[l-1], gcn_ctx_cuda.num_nodes * gcn_ctx_cuda.layer_dims[l] * sizeof(double)));
    

    }
    int out_dim = gcn_ctx_cuda.layer_dims.back();
    int size = gcn_ctx_cuda.num_nodes * out_dim;
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    softmax_cross_entropy_grad_kernel<<<blocksPerGrid, threadsPerBlock>>>(gcn_ctx_cuda.Z_d[L - 1], gcn_ctx_cuda.labels_d, dZ_d[L - 1], gcn_ctx_cuda.num_nodes, out_dim);
    CHECK_CUDA(cudaGetLastError());

    for (int l = L - 1; l >= 0; --l) {
        int in_dim = gcn_ctx_cuda.layer_dims[l];
        int out_dim = gcn_ctx_cuda.layer_dims[l + 1];
        double *dW_d;
        CHECK_CUDA(cudaMalloc(&dW_d, in_dim * out_dim * sizeof(double)));

       
        CHECK_CUBLAS(cublasDgemm(gcn_ctx_cuda.bl_handle, CUBLAS_OP_N, CUBLAS_OP_T,out_dim, in_dim, gcn_ctx_cuda.num_nodes,&alpha, dZ_d[l], out_dim, gcn_ctx_cuda.H_d[l], in_dim,&beta, dW_d, out_dim));

        int w_size = in_dim * out_dim;
        int adam_blocks = (w_size + threadsPerBlock - 1) / threadsPerBlock;
        adam_kernel<<<adam_blocks, threadsPerBlock>>>(
            gcn_ctx_cuda.W_d[l], dW_d, gcn_ctx_cuda.m_d[l], gcn_ctx_cuda.v_d[l], w_size, 0.9, 0.999, 1e-8, 0.001, epoch);
        CHECK_CUDA(cudaGetLastError());
        // CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaFree(dW_d)); 

        if (l > 0) {

            CHECK_CUBLAS(cublasDgemm(gcn_ctx_cuda.bl_handle, CUBLAS_OP_T, CUBLAS_OP_N,in_dim, gcn_ctx_cuda.num_nodes, out_dim,&alpha, gcn_ctx_cuda.W_d[l], out_dim, dZ_d[l], out_dim,&beta, dH_d[l - 1], in_dim));

            int prev_size = gcn_ctx_cuda.num_nodes * in_dim;
            int relu_blocks = (prev_size + threadsPerBlock - 1) / threadsPerBlock;
            relu_derivative_kernel<<<relu_blocks, threadsPerBlock>>>(
                gcn_ctx_cuda.Z_d[l - 1], dH_d[l - 1], dZ_d[l - 1], prev_size);
            CHECK_CUDA(cudaGetLastError());
            // CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUDA(cudaFree(dH_d[l-1])); // Free dH
        }
    }

    for(int l=0; l<L; ++l) CHECK_CUDA(cudaFree(dZ_d[l]));
}






void forward_cuda(std::string modelType, std::string aggregationType="SUM")
{
    if(modelType=="GCN")
    {
        GCN();
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


void backprop_cuda(std::string modelType, std::string aggregationType,int epoch)
{
   if(modelType=="GCN"){
    GCN_backprop(epoch);
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

double compute_loss_and_accuracy_cuda(const double *Z_d, const int *labels_d, int num_nodes, int out_dim, double &accuracy)
{
    std::vector<double> Z_h(num_nodes * out_dim);
    std::vector<int> labels_h(num_nodes);
    CHECK_CUDA(cudaMemcpy(Z_h.data(), Z_d, num_nodes * out_dim * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(labels_h.data(), labels_d, num_nodes * sizeof(int), cudaMemcpyDeviceToHost));

    double loss = 0.0;
    int correct = 0;

    for (int i = 0; i < num_nodes; i++) {
        double max_val = -1e9;
        int pred = -1;

        for (int j = 0; j < out_dim; j++) {
            max_val = std::max(max_val, Z_h[i * out_dim + j]);
        }

        double sum_exp = 0.0;
        for (int j = 0; j < out_dim; j++) {
            sum_exp += std::exp(Z_h[i * out_dim + j] - max_val);
        }

        double prob = std::exp(Z_h[i * out_dim + labels_h[i]] - max_val) / sum_exp;
        loss += -std::log(std::max(prob, 1e-10));

        max_val = -1e9;
        for (int j = 0; j < out_dim; j++) {
            double val = Z_h[i * out_dim + j];
            if (val > max_val) {
                max_val = val;
                pred = j;
            }
        }
        if (pred == labels_h[i]) {
            correct++;
        }
    }

    accuracy = static_cast<double>(correct) / num_nodes;
    return loss / num_nodes;
}


void init_cuda(std::vector<int> neuronsPerLayer, std::string initWeights, std::string folderpath)
{

     std::vector<int> A_row_ptr_h, A_col_idx_h;
    std::vector<double> A_edgeweights_h;
    read_graph_omp(folderpath + "_edgelist.txt", A_row_ptr_h, A_col_idx_h, A_edgeweights_h, gcn_ctx_cuda.num_nodes);


    std::vector<double> features;
    read_features_omp(folderpath + "_features.txt", features, gcn_ctx_cuda.num_nodes, gcn_ctx_cuda.num_features);

    std::vector<int> labels;
    read_labels_omp(folderpath + "_labels.txt", labels, gcn_ctx_cuda.num_classes);


    gcn_ctx_cuda.layer_dims = neuronsPerLayer;
    gcn_ctx_cuda.layer_dims[0] = gcn_ctx_cuda.num_features; // Input layer size
    gcn_ctx_cuda.layer_dims[gcn_ctx_cuda.layer_dims.size() - 1] = gcn_ctx_cuda.num_classes; // Output layer size



    CHECK_CUBLAS(cublasCreate(&gcn_ctx_cuda.bl_handle));
    CHECK_CUSPARSE(cusparseCreate(&gcn_ctx_cuda.sp_handle));
    CHECK_CUDA(cudaMalloc(&gcn_ctx_cuda.A_row_d, (gcn_ctx_cuda.num_nodes + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&gcn_ctx_cuda.A_col_d, gcn_ctx_cuda.num_edges * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&gcn_ctx_cuda.A_val_d, gcn_ctx_cuda.num_edges * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&gcn_ctx_cuda.labels_d, gcn_ctx_cuda.num_nodes * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&gcn_ctx_cuda.features_d, gcn_ctx_cuda.num_nodes * gcn_ctx_cuda.num_features * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(gcn_ctx_cuda.A_row_d, A_row_ptr_h.data(), (gcn_ctx_cuda.num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gcn_ctx_cuda.A_col_d, A_col_idx_h.data(), gcn_ctx_cuda.num_edges * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gcn_ctx_cuda.A_val_d, A_edgeweights_h.data(), gcn_ctx_cuda.num_edges * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gcn_ctx_cuda.labels_d, labels.data(), gcn_ctx_cuda.num_nodes * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gcn_ctx_cuda.features_d, features.data(), gcn_ctx_cuda.num_nodes * gcn_ctx_cuda.num_features * sizeof(double), cudaMemcpyHostToDevice));
    initialize_weights_cuda(gcn_ctx_cuda.W_d, gcn_ctx_cuda.layer_dims);

    gcn_ctx_cuda.m_d.resize(gcn_ctx_cuda.W_d.size());
    gcn_ctx_cuda.v_d.resize(gcn_ctx_cuda.W_d.size());

    for (int i = 0; i < gcn_ctx_cuda.W_d.size(); ++i) {
        int size = gcn_ctx_cuda.layer_dims[i] * gcn_ctx_cuda.layer_dims[i+1];
        CHECK_CUDA(cudaMalloc(&gcn_ctx_cuda.m_d[i], size * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&gcn_ctx_cuda.v_d[i], size * sizeof(double)));
        CHECK_CUDA(cudaMemset(gcn_ctx_cuda.m_d[i], 0, size * sizeof(double)));
        CHECK_CUDA(cudaMemset(gcn_ctx_cuda.v_d[i], 0, size * sizeof(double)));
    }
    int num_layers = gcn_ctx_cuda.layer_dims.size() - 1;
    gcn_ctx_cuda.Z_d.resize(num_layers);
    gcn_ctx_cuda.H_d.resize(num_layers + 1);
    
    gcn_ctx_cuda.H_descr.resize(num_layers + 1);
    gcn_ctx_cuda.Z_descr.resize(num_layers);
    gcn_ctx_cuda.Temp_descr.resize(num_layers);

    gcn_ctx_cuda.H_d[0] = gcn_ctx_cuda.features_d;

    CHECK_CUSPARSE(cusparseCreateDnMat(&gcn_ctx_cuda.H_descr[0], gcn_ctx_cuda.num_nodes, gcn_ctx_cuda.layer_dims[0], gcn_ctx_cuda.layer_dims[0], gcn_ctx_cuda.H_d[0], CUDA_R_64F, CUSPARSE_ORDER_ROW));

    
     for (int l = 0; l < num_layers; ++l) {
        CHECK_CUDA(cudaMalloc(&gcn_ctx_cuda.Z_d[l], gcn_ctx_cuda.num_nodes * gcn_ctx_cuda.layer_dims[l + 1] * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&gcn_ctx_cuda.H_d[l+1], gcn_ctx_cuda.num_nodes * gcn_ctx_cuda.layer_dims[l + 1] * sizeof(double)));
        CHECK_CUSPARSE(cusparseCreateDnMat(&gcn_ctx_cuda.Z_descr[l], gcn_ctx_cuda.num_nodes, gcn_ctx_cuda.layer_dims[l+1], gcn_ctx_cuda.layer_dims[l+1], gcn_ctx_cuda.Z_d[l], CUDA_R_64F, CUSPARSE_ORDER_ROW));
        CHECK_CUSPARSE(cusparseCreateDnMat(&gcn_ctx_cuda.H_descr[l+1], gcn_ctx_cuda.num_nodes, gcn_ctx_cuda.layer_dims[l+1], gcn_ctx_cuda.layer_dims[l+1], gcn_ctx_cuda.H_d[l+1], CUDA_R_64F, CUSPARSE_ORDER_ROW));

        double* temp_d;
        CHECK_CUDA(cudaMalloc(&temp_d, gcn_ctx_cuda.num_nodes * gcn_ctx_cuda.layer_dims[l] * sizeof(double)));
        CHECK_CUSPARSE(cusparseCreateDnMat(&gcn_ctx_cuda.Temp_descr[l], gcn_ctx_cuda.num_nodes, gcn_ctx_cuda.layer_dims[l], gcn_ctx_cuda.layer_dims[l], temp_d, CUDA_R_64F, CUSPARSE_ORDER_ROW));
    }

    CHECK_CUSPARSE(cusparseCreateCsr(&gcn_ctx_cuda.A_descr, gcn_ctx_cuda.num_nodes, gcn_ctx_cuda.num_nodes, gcn_ctx_cuda.num_edges, gcn_ctx_cuda.A_row_d, gcn_ctx_cuda.A_col_d, gcn_ctx_cuda.A_val_d, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));



    size_t bufferSize = 0;
    double alpha = 1.0, beta = 0.0;
    int max_dim = 0;
    max_dim = gcn_ctx_cuda.layer_dims[0] ;
    double* temp_H_buffer; CHECK_CUDA(cudaMalloc(&temp_H_buffer, gcn_ctx_cuda.num_nodes * max_dim * sizeof(double)));
    cusparseDnMatDescr_t temp_H_descr; CHECK_CUSPARSE(cusparseCreateDnMat(&temp_H_descr, gcn_ctx_cuda.num_nodes, max_dim, max_dim, temp_H_buffer, CUDA_R_64F, CUSPARSE_ORDER_ROW));
    double* temp_T_buffer; CHECK_CUDA(cudaMalloc(&temp_T_buffer, gcn_ctx_cuda.num_nodes * max_dim * sizeof(double)));
    cusparseDnMatDescr_t temp_T_descr; CHECK_CUSPARSE(cusparseCreateDnMat(&temp_T_descr, gcn_ctx_cuda.num_nodes, max_dim, max_dim, temp_T_buffer, CUDA_R_64F, CUSPARSE_ORDER_ROW));

    CHECK_CUSPARSE(cusparseSpMM_bufferSize(gcn_ctx_cuda.sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, gcn_ctx_cuda.A_descr, temp_H_descr,
                                           &beta, temp_T_descr, CUDA_R_64F,
                                           CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize));
    CHECK_CUDA(cudaMalloc(&gcn_ctx_cuda.dBuffer, bufferSize));
    CHECK_CUDA(cudaFree(temp_H_buffer)); CHECK_CUDA(cudaFree(temp_T_buffer));
    CHECK_CUSPARSE(cusparseDestroyDnMat(temp_H_descr)); CHECK_CUSPARSE(cusparseDestroyDnMat(temp_T_descr));
}



void cleanup_cuda()                             //This function cleans up the allocated memory and resources used and should be called at the end of the program.
{
    for (auto &ptr : gcn_ctx_cuda.W_d)
        cudaFree(ptr);
    for (auto &ptr : gcn_ctx_cuda.H_d)
        cudaFree(ptr);
    for (auto &ptr : gcn_ctx_cuda.Z_d)
        cudaFree(ptr);
    for (auto &ptr : gcn_ctx_cuda.Temp_d)
        cudaFree(ptr);
    for (auto &ptr : gcn_ctx_cuda.m_d)
        cudaFree(ptr);
    for (auto &ptr : gcn_ctx_cuda.v_d)
        cudaFree(ptr);
    cudaFree(gcn_ctx_cuda.A_row_d);
    cudaFree(gcn_ctx_cuda.A_col_d);
    cudaFree(gcn_ctx_cuda.A_val_d);
    cudaFree(gcn_ctx_cuda.labels_d);
    cudaFree(gcn_ctx_cuda.features_d);

    for (auto &desc : gcn_ctx_cuda.H_descr)
        cusparseDestroyDnMat(desc);
    for (auto &desc : gcn_ctx_cuda.Z_descr)
        cusparseDestroyDnMat(desc);
    for (auto &desc : gcn_ctx_cuda.Temp_descr)
        cusparseDestroyDnMat(desc);
    cusparseDestroySpMat(gcn_ctx_cuda.A_descr);

    cublasDestroy(gcn_ctx_cuda.bl_handle);
    cusparseDestroy(gcn_ctx_cuda.sp_handle);
}



void compute_loss_cuda(){
    gcn_ctx_cuda.loss = compute_loss_and_accuracy_cuda(gcn_ctx_cuda.Z_d[gcn_ctx_cuda.layer_dims.size() - 2],gcn_ctx_cuda.labels_d,gcn_ctx_cuda.num_nodes,gcn_ctx_cuda.layer_dims.back(),gcn_ctx_cuda.accuracy);
    std::cout << "Loss: " << gcn_ctx_cuda.loss << std::endl;
}

void compute_accuracy_cuda(){
    gcn_ctx_cuda.accuracy = 0.0;
    compute_loss_and_accuracy_cuda(gcn_ctx_cuda.Z_d[gcn_ctx_cuda.layer_dims.size() - 2], gcn_ctx_cuda.labels_d, gcn_ctx_cuda.num_nodes, gcn_ctx_cuda.layer_dims.back(), gcn_ctx_cuda.accuracy);
    std::cout << "Accuracy: " << gcn_ctx_cuda.accuracy * 100.0 << "%" << std::endl;
}


// void print_cuda_info()               //This function prints information about the CUDA devices may or may not be useful but very helpful if you are using large graphs. 
// {
//     int device_count;
//     CHECK_CUDA(cudaGetDeviceCount(&device_count));
//     if (device_count == 0)
//     {
//         std::cout << "No CUDA devices found." << std::endl;
//         return;
//     }

//     for (int i = 0; i < device_count; ++i)
//     {
//         cudaDeviceProp prop;
//         CHECK_CUDA(cudaGetDeviceProperties(&prop, i));
//         std::cout << "Device " << i << ": " << prop.name << std::endl;
//         std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
//         std::cout << "  Total Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
//         std::cout << "  Multiprocessor Count: " << prop.multiProcessorCount << std::endl;
//         std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
//     }
// }


// int main()               //This can be used if you want to test the CUDA GNN implementation directly
// {

//     std::string folderpath = "flickr/flickr";

//     std::vector<int> neuronsPerLayer = {0, 16, 0};
//     std::string init_type = "xavier"; // or HE

//     // Initialize
//     init_cuda(neuronsPerLayer, init_type, folderpath);

//     int epochs = 200;
//     auto start = std::chrono::high_resolution_clock::now();
//     for (int epoch = 1; epoch <= epochs; ++epoch)
//     {

//         gcn_forward_cuda();
//         gcn_backpropagation_cuda(epoch);

//     }
//     double acc = 0.0;
//     double loss = compute_loss_and_accuracy_cuda(
//         gcn_ctx_cuda.Z_d[gcn_ctx_cuda.layer_dims.size() - 2],
//         gcn_ctx_cuda.labels_d,
//         gcn_ctx_cuda.num_nodes,
//         gcn_ctx_cuda.layer_dims.back(),
//         acc);

//     auto end = std::chrono::high_resolution_clock::now();
//     double duration = std::chrono::duration<double>(end - start).count();

//     std::cout << "Loss: " << loss
//               << ", Accuracy: " << acc * 100.0 << "%"
//               << ", Time: " << duration << "s" << std::endl;
//     cleanup_cuda();
//     return 0;
// }



// int main()               //StarPlat inspired main function to test the CUDA GNN implementation
// {

//     std::string folderpath = "flickr/flickr";

//     std::string init_type = "xavier"; // or HE

//   graph g("flickr/flickr_edgelist.txt");
//   g.parseGraph();

//   test(g,{16,16,16},100,folderpath);


//     return 0;
// }

// nvcc test.cu -lcublas -lcusparse -lcurand 

//Run the above command to compile and run the code. Make sure you have the necessary libraries installed and linked properly.