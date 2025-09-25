#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <cstring>
#include <climits>
// #include "./generated_omp/v_cover.cc"
int main(int argc, char*argv[]) {
    char* filePath;

    if (argc == 1) {
        std::string inputPath;
        std::cout << "Enter the path to the graph file: ";
        std::getline(std::cin, inputPath);

        filePath = new char[inputPath.length() + 1]; 
        std::strcpy(filePath, inputPath.c_str());
    } else if (argc == 2) {
        filePath = argv[1];
    } else {
        return 1;
    }

    graph g(filePath);
    g.parseGraph();
    int num_nodes = g.num_nodes();
    std::vector<int> dist(num_nodes, INT_MAX);
    
    int num_edges = g.num_edges();
    std::vector<int> weight(num_edges);
    int edgeNum = 0;
    for (int v = 0; v < num_nodes; v ++) {
        for (int edge = g.indexofNodes[v]; edge < g.indexofNodes[v+1]; edge ++) {
            int nbr = g.edgeList[edge];
            weight[edgeNum] = g.getEdge(v, nbr).weight;
            edgeNum++;
        }
    }

    int src = 0;

    double starttime = omp_get_wtime();
    Compute_SSSP(g, dist, weight, src);
    double endtime = omp_get_wtime();

    // std::cout << "Shortest paths from node " << src << ":" << std::endl;
    // for (int i = 0; i < g.num_nodes(); ++i) {
    //     std::cout << "Node " << i << ": Distance = " << g.get_node_property(i, "dist") << std::endl;
    // }
    std::cout << "\nTime taken: " << endtime - starttime << " seconds" << std::endl;
    return 0;
    }
