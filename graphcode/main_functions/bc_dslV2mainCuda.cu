//make sure to first generate the cuda backend code:
//by cd into 'src'
//command to run on your terminal to generate "betweeness centrality (for static graphs) cuda backend code" using bc_dslV2 as input dsl:
// ./StarPlat -s -f ../graphcode/staticDSLCodes/bc_dslV2 -b cuda
//for executing on Google collab: remember to add ! before the above command.

//generating the bc_dslV2 cuda backend code by using the above command will generate two files 'bc_dslV2.cu' and 'bc_dslV2.h' 
//in the directory ../graphcode/generated_cuda which is required for this code to run
//as we have the main caller function for that code written here in this file.
//nvcc bc_dslV2mainCuda.cu -o bc_dslV2mainCuda -arch=sm_70 -std=c++14 -rdc=true
//./bc_dslV2mainCuda ../graphcode/generated_cuda/sample_graph.txt ../graphcode/generated_cuda/src_nodes.txt

#include "../generated_cuda/bc_dslV2.cu"

// main fn: reads Input and Calls `Compute_BC`
int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <graph_file> <source_nodes_file>" << std::endl;
        return 1;
    }

    char* graphFilePath = argv[1];
    char* sourceNodesFilePath = argv[2];

    // Load graph
    graph g(graphFilePath);
    g.parseGraph();

    std::cout << "Number of nodes: " << g.num_nodes() << std::endl;
    std::cout << "Number of edges: " << g.num_edges() << std::endl;

    // Read source nodes from file
    std::set<int> sourceSet;
    std::ifstream sourceFile(sourceNodesFilePath);
    if (!sourceFile) {
        std::cerr << "Error: Unable to open source nodes file: " << sourceNodesFilePath << std::endl;
        return 1;
    }

    int node;
    while (sourceFile >> node) {
        if (node >= 0 && node < g.num_nodes()) {
            sourceSet.insert(node);
        }
    }
    sourceFile.close();

    std::cout << "Source nodes loaded: ";
    for (int src : sourceSet) {
      std::cout << src << " ";
    }
    std::cout << std::endl;



    float* BC = new float[g.num_nodes()];
    std::fill_n(BC, g.num_nodes(), 0.0f);//to set the array BC values to 0

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

//to list out all the edges of each node in the graph.
    for (int i = 0; i < g.num_nodes(); ++i) {
      std::cout << "Node " << i << " has edges: ";
        for (int j = g.indexofNodes[i]; j < g.indexofNodes[i + 1]; j++) {
          std::cout << g.edgeList[j] << " ";
        }
      std::cout << std::endl;//new line after listing out all the edges of the node i.
    }

    cudaEventRecord(start, 0);
    Compute_BC(g, BC, sourceSet);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "\nGPU Time: " << milliseconds / 1000.0 << " seconds" << std::endl;

    for (int i = 0; i < g.num_nodes(); ++i) {
        std::cout << "Node " << i << " BC: " << BC[i] << std::endl;
    }

    delete[] BC;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
