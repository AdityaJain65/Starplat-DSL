#include <iostream>
#include "mpi_header/graph_mpi.cc"
#include "mpi_header/profileAndDebug/mpi_debug.c"
#include "./generated_mpi/bc_dslV2.cc"
using namespace std;

void Compute_BC(Graph &, NodeProperty<float> &, std::set<int> &, boost::mpi::communicator);

int main(int argc, char *argv[])
{
    char *filePath;
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;
    if (argc == 1)
    {
        std::string inputPath;
        std::cout << "Enter the path to the graph file: ";
        std::getline(std::cin, inputPath);

        filePath = new char[inputPath.length() + 1];
        std::strcpy(filePath, inputPath.c_str());
    }
    else if (argc >= 2)
    {
        filePath = argv[1];
    }
    else
    {
        return 1;
    }

    Graph graph(argv[1], world, 1);
    NodeProperty<float> BC;
    set<int> source_set;
    for (int i = 0; i < graph.num_nodes(); i++)
        source_set.insert(i);
    // Graph& g, NodeProperty<float>& BC, std::set<int>& sourceSet, boost::mpi::communicator world
    double t1 = MPI_Wtime();
    Compute_BC(graph, BC, source_set, world);
    cout << "No of nodes  " << graph.num_nodes() << endl;
    double t2 = MPI_Wtime();
    print_mpi_statistics();
    // if (world.rank() == 0)
    // {
    //     for(auto x:graph.nodes())
    //     {

    //     }
    // }
    world.barrier();
    return 0;
}