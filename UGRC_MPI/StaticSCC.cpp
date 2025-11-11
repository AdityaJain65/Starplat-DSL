// fb_scc_mpi_subgraphs.cpp
// Forward-Backward SCC with parallel processing of subgraphs using MPI_Comm_split.
// Compile: mpicxx -O2 fb_scc_mpi_subgraphs.cpp -o fb_scc_mpi_subgraphs
// Run: mpirun -np <P> ./fb_scc_mpi_subgraphs graph.txt

#include <mpi.h>
#include <bits/stdc++.h>
using namespace std;

struct Graph
{
    int n;
    int m;
    // CSR-style using vectors for simplicity. Each process stores full graph.
    vector<int> head_fwd; // head_fwd[u] = index of first outgoing edge (or -1)
    vector<int> to_fwd;   // to_fwd[e] = destination vertex
    vector<int> next_fwd; // next_fwd[e] = next edge index from same source
    vector<int> head_rev; // reverse graph heads
    vector<int> to_rev;
    vector<int> next_rev;
    Graph() : n(0), m(0) {}
};

void read_graph(Graph &g)
{
    int n, m;
    cin >> n >> m;
    g.n = n;
    g.m = m;
    vector<int> U(m), V(m);
    for (int i = 0; i < m; ++i)
        cin >> U[i] >> V[i];
    // forward CSR
    g.head_fwd.assign(n, -1);
    g.to_fwd.resize(m);
    g.next_fwd.resize(m);
    for (int i = 0; i < m; ++i)
    {
        int u = U[i], v = V[i];
        g.to_fwd[i] = v;
        g.next_fwd[i] = g.head_fwd[u];
        g.head_fwd[u] = i;
    }
    // reverse CSR
    g.head_rev.assign(n, -1);
    g.to_rev.resize(m);
    g.next_rev.resize(m);
    for (int i = 0; i < m; ++i)
    {
        int u = U[i], v = V[i];
        // reverse edge v -> u
        g.to_rev[i] = u;
        g.next_rev[i] = g.head_rev[v];
        g.head_rev[v] = i;
    }
}

// Helper: find local owner of a vertex v inside a communicator of size 'comm_size'.
// Ownership is contiguous ranges: chunk = ceil(n / comm_size)
inline int owner_of_vertex(int v, int n, int comm_size)
{
    int chunk = (n + comm_size - 1) / comm_size;
    return v / chunk;
}

// Parallel BFS/reachability inside communicator 'comm' from pivot, restricted to vertices
// marked as in_subgraph (byte array: 1 = present in this subgraph).
// If rev == false -> use forward graph; else use reverse graph.
// visited_out is written with 0/1 bytes (size n), indicating vertices reachable from pivot
// but restricted to in_subgraph (we never traverse outside in_subgraph).
void parallel_reachability_bfs(const Graph &g, MPI_Comm comm,
                               const vector<unsigned char> &in_subgraph,
                               vector<unsigned char> &visited_out,
                               int pivot, bool rev)
{
    int comm_rank, comm_size;
    MPI_Comm_rank(comm, &comm_rank);
    MPI_Comm_size(comm, &comm_size);
    int n = g.n;

    // local arrays
    vector<unsigned char> local_visited(n, 0);
    vector<unsigned char> local_frontier(n, 0);
    vector<unsigned char> combined(n, 0);

    // initialize
    if (in_subgraph[pivot])
    {
        local_frontier[pivot] = 1;
        local_visited[pivot] = 1;
    }
    else
    {
        // pivot not in this subgraph (shouldn't happen) -> no reachability
        // still proceed, local arrays are zero
    }

    while (true)
    {
        // local_next discovered by expanding local_frontier using adjacency (only on vertices present in in_subgraph)
        vector<unsigned char> local_next(n, 0);

        // Each rank expands only vertices it "owns" within this communicator to parallelize work
        int chunk = (n + comm_size - 1) / comm_size;
        int local_lo = comm_rank * chunk;
        int local_hi = min(n, (comm_rank + 1) * chunk);

        for (int u = local_lo; u < local_hi; ++u)
        {
            if (!local_frontier[u])
                continue;
            if (!in_subgraph[u])
                continue;
            int eid = rev ? g.head_rev[u] : g.head_fwd[u];
            while (eid != -1)
            {
                int v = rev ? g.to_rev[eid] : g.to_fwd[eid];
                if (in_subgraph[v] && !local_visited[v])
                {
                    local_next[v] = 1;
                }
                eid = rev ? g.next_rev[eid] : g.next_fwd[eid];
            }
        }

        // merge local_next into local_visited
        for (int i = 0; i < n; ++i)
            if (local_next[i])
                local_visited[i] = 1;

        // combine local_visited across communicator
        MPI_Allreduce(local_visited.data(), combined.data(), n, MPI_UNSIGNED_CHAR, MPI_MAX, comm);

        // check if any new global discoveries compared to visited_out
        bool changed = false;
        for (int i = 0; i < n; ++i)
        {
            if (combined[i] && !visited_out[i])
            {
                changed = true;
                break;
            }
        }
        // update visited_out
        visited_out = combined;

        if (!changed)
            break;

        // prepare next frontier = local_next (simple strategy)
        fill(local_frontier.begin(), local_frontier.end(), 0);
        for (int i = 0; i < n; ++i)
            if (local_next[i])
                local_frontier[i] = 1;
    }
}

// Find pivot inside in_subgraph: smallest vertex id with in_subgraph[v]==1.
// Return -1 if none. Uses Allreduce to compute global min within comm.
int find_pivot_in_subgraph(MPI_Comm comm, const vector<unsigned char> &in_subgraph)
{
    int comm_rank, comm_size;
    MPI_Comm_rank(comm, &comm_rank);
    MPI_Comm_size(comm, &comm_size);
    int n = (int)in_subgraph.size();
    int local_min = INT_MAX;
    for (int i = 0; i < n; ++i)
    {
        if (in_subgraph[i])
        {
            local_min = i;
            break;
        }
    }
    int global_min;
    MPI_Allreduce(&local_min, &global_min, 1, MPI_INT, MPI_MIN, comm);
    if (global_min == INT_MAX)
        return -1;
    return global_min;
}

// Main recursive function: processes subgraph defined by in_subgraph inside communicator 'comm'.
// Each rank stores full graph 'g'.
void forward_backward_recursive(const Graph &g, MPI_Comm comm, vector<unsigned char> &in_subgraph)
{
    int comm_rank, comm_size;
    MPI_Comm_rank(comm, &comm_rank);
    MPI_Comm_size(comm, &comm_size);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int n = g.n;

    // Count vertices in this in_subgraph (local and global)
    int local_present = 0;
    for (int i = 0; i < n; ++i)
        if (in_subgraph[i])
            local_present++;
    int global_present;
    MPI_Allreduce(&local_present, &global_present, 1, MPI_INT, MPI_SUM, comm);
    if (global_present == 0)
    {
        // nothing to do in this communicator
        return;
    }

    // If global_present == 1, we have a single vertex -> it's an SCC by itself
    if (global_present == 1)
    {
        // find the vertex and print SCC
        int local_v = -1;
        for (int i = 0; i < n; ++i)
            if (in_subgraph[i])
            {
                local_v = i;
                break;
            }
        int vertices_present = 0;
        // gather globally who has the vertex (choose to print from world rank 0)
        // We'll let communicator root print it
        if (comm_rank == 0)
        {
            // collect which vertex (all ranks agree there's exactly one)
            int v_global = local_v;
            // broadcast from each rank? simpler: reduce with max
            MPI_Allreduce(&v_global, &v_global, 1, MPI_INT, MPI_MAX, comm);
            // print
            int world_root;
            MPI_Comm_rank(MPI_COMM_WORLD, &world_root); // not used — print with comm root info
            // cout << "[comm root world_rank=" << world_rank << "] SCC: {" << v_global << "}\n";
        }
        return;
    }

    // 1) choose pivot inside this in_subgraph
    int pivot = find_pivot_in_subgraph(comm, in_subgraph);
    if (pivot == -1)
        return; // nothing left

    // if (comm_rank == 0)
    // {
    //     cout << "[comm root world_rank=" << world_rank << "] pivot = " << pivot << " (subgraph size " << global_present << ")\n";
    // }

    // 2) compute forward reachable set F (restricted to in_subgraph)
    vector<unsigned char> visitedF(n, 0);
    parallel_reachability_bfs(g, comm, in_subgraph, visitedF, pivot, false);

    // 3) compute backward reachable set B (on reverse graph), restricted to in_subgraph
    vector<unsigned char> visitedB(n, 0);
    parallel_reachability_bfs(g, comm, in_subgraph, visitedB, pivot, true);

    // 4) form S = visitedF & visitedB & in_subgraph ; and Fonly = visitedF \ S ; Bonly = visitedB \ S ; R = in_subgraph \ (visitedF ∪ visitedB)
    vector<unsigned char> S_local(n, 0), Fonly_local(n, 0), Bonly_local(n, 0), R_local(n, 0);

    int local_S_count = 0, local_Fonly_count = 0, local_Bonly_count = 0, local_R_count = 0;
    for (int i = 0; i < n; ++i)
    {
        if (!in_subgraph[i])
            continue;
        if (visitedF[i] && visitedB[i])
        {
            S_local[i] = 1;
            local_S_count++;
        }
        else if (visitedF[i] && !visitedB[i])
        {
            Fonly_local[i] = 1;
            local_Fonly_count++;
        }
        else if (!visitedF[i] && visitedB[i])
        {
            Bonly_local[i] = 1;
            local_Bonly_count++;
        }
        else
        {
            R_local[i] = 1;
            local_R_count++;
        }
    }

    int global_S_count = 0, global_Fonly_count = 0, global_Bonly_count = 0, global_R_count = 0;
    MPI_Allreduce(&local_S_count, &global_S_count, 1, MPI_INT, MPI_SUM, comm);
    MPI_Allreduce(&local_Fonly_count, &global_Fonly_count, 1, MPI_INT, MPI_SUM, comm);
    MPI_Allreduce(&local_Bonly_count, &global_Bonly_count, 1, MPI_INT, MPI_SUM, comm);
    MPI_Allreduce(&local_R_count, &global_R_count, 1, MPI_INT, MPI_SUM, comm);

    // Combine S across ranks in communicator to obtain S_global mask (bytewise max)
    vector<unsigned char> S_global(n, 0);
    MPI_Allreduce(S_local.data(), S_global.data(), n, MPI_UNSIGNED_CHAR, MPI_MAX, comm);

    // 5) Prepare subgraphs (Fonly, Bonly, R) and assign ranks to them
    // We'll create a vector of (subgraph_id, size) for non-empty subgraphs
    // subgraph ids: 1 -> Fonly, 2 -> Bonly, 3 -> R
    vector<pair<int, int>> subs; // (id, global_size)
    if (global_Fonly_count > 0)
        subs.emplace_back(1, global_Fonly_count);
    if (global_Bonly_count > 0)
        subs.emplace_back(2, global_Bonly_count);
    if (global_R_count > 0)
        subs.emplace_back(3, global_R_count);

    if (subs.empty())
    {
        // nothing left to recurse
        return;
    }

    // assign contiguous ranges of communicator ranks to subs proportional to sizes
    int K = (int)subs.size();
    vector<int> part_start(K), part_end(K); // inclusive start, exclusive end in rank indices [0..comm_size)
    int ranks_assigned = 0;
    int remaining_ranks = comm_size;
    long long total_nodes = 0;
    for (auto &p : subs)
        total_nodes += p.second;
    int rank_ptr = 0;
    for (int i = 0; i < K; ++i)
    {
        // compute ranks for this subgraph: round((size/total_nodes)*comm_size)
        double frac = (double)subs[i].second / (double)total_nodes;
        int r = (int)floor(frac * comm_size + 0.5);
        // ensure at least 1 rank assigned if subgraph non-empty and we still have ranks
        if (r < 1)
            r = 1;
        // last partition takes remaining ranks
        if (i == K - 1)
            r = comm_size - rank_ptr;
        part_start[i] = rank_ptr;
        part_end[i] = rank_ptr + r;
        rank_ptr += r;
    }

    // Determine this rank's color: color = subs[idx].first, or MPI_UNDEFINED if not assigned (shouldn't happen)
    int color = MPI_UNDEFINED;
    int myidx = -1;
    for (int i = 0; i < K; ++i)
    {
        if (comm_rank >= part_start[i] && comm_rank < part_end[i])
        {
            color = subs[i].first; // 1,2 or 3
            myidx = i;
            break;
        }
    }

    // Build mask of vertices this communicator split should process:
    // For ranks with color==1 -> Fonly mask, color==2 -> Bonly mask, color==3 -> R mask.
    vector<unsigned char> my_submask(n, 0);
    if (color == 1)
    {
        for (int i = 0; i < n; ++i)
            if (Fonly_local[i])
                my_submask[i] = 1;
    }
    else if (color == 2)
    {
        for (int i = 0; i < n; ++i)
            if (Bonly_local[i])
                my_submask[i] = 1;
    }
    else if (color == 3)
    {
        for (int i = 0; i < n; ++i)
            if (R_local[i])
                my_submask[i] = 1;
    }
    else
    {
        // This communicator rank was not assigned to any subgraph — mark empty mask
        // (should not happen because we assigned all ranks)
    }

    // Now split the communicator
    MPI_Comm newcomm;
    MPI_Comm_split(comm, color, comm_rank, &newcomm);

    // Free local temporary vectors that are large (help memory)
    // (they will be destructed at scope exit)

    // Recurse in new communicator if color != MPI_UNDEFINED
    if (color != MPI_UNDEFINED)
    {
        // recurse
        forward_backward_recursive(g, newcomm, my_submask);
        MPI_Comm_free(&newcomm);
    }
    else
    {
        // not assigned — do nothing
    }

    // After recursing into subgraphs we are done for this communicator
    return;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    // read graph on all processes (simple approach)
    Graph g;

    if (world_rank == 0)
    {
        read_graph(g);
    }

    // Broadcast n and m to all ranks
    MPI_Bcast(&g.n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&g.m, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Resize arrays on all processes
    if (world_rank != 0)
    {
        g.head_fwd.assign(g.n, -1);
        g.to_fwd.resize(g.m);
        g.next_fwd.resize(g.m);
        g.head_rev.assign(g.n, -1);
        g.to_rev.resize(g.m);
        g.next_rev.resize(g.m);
    }

    // Now broadcast all arrays (all are simple int vectors)
    MPI_Bcast(g.head_fwd.data(), g.n, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(g.to_fwd.data(), g.m, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(g.next_fwd.data(), g.m, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(g.head_rev.data(), g.n, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(g.to_rev.data(), g.m, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(g.next_rev.data(), g.m, MPI_INT, 0, MPI_COMM_WORLD);
    vector<unsigned char> in_subgraph(g.n, 1);
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start_total = MPI_Wtime(); // start total timing
    double t_start_read = MPI_Wtime();
    forward_backward_recursive(g, MPI_COMM_WORLD, in_subgraph);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end_total = MPI_Wtime();

    if (world_rank == 0)
    {
        cerr << "------------------------------\n";
        cerr << "[Rank 0] Total time = " << (t_end_total - t_start_total) << " sec\n";
        cerr << "Processes: " << world_size << "\n";
        cerr << "Vertices: " << g.n << "  Edges: " << g.m << "\n";
        cerr << "------------------------------\n";
    }

    MPI_Finalize();
}
