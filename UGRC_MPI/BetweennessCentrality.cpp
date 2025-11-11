// mpi_dynamic_bc_incremental.cpp
#include <bits/stdc++.h>
#include <mpi.h>
using namespace std;

struct Graph
{
    int n;
    vector<vector<int>> adj;
    Graph(int n = 0) : n(n), adj(n) {}
    void add_edge(int u, int v)
    {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    void remove_edge(int u, int v)
    {
        adj[u].erase(remove(adj[u].begin(), adj[u].end(), v), adj[u].end());
        adj[v].erase(remove(adj[v].begin(), adj[v].end(), u), adj[v].end());
    }
};

// ---------- Brandes single-source BFS (builds dist,sigma,P and delta) ----------
void bfs_from_source(int s, const Graph &G,
                     vector<int> &dist, vector<long long> &sigma,
                     vector<vector<int>> &P, vector<double> &delta)
{
    int n = G.n;
    dist.assign(n, -1);
    sigma.assign(n, 0);
    P.assign(n, {});
    delta.assign(n, 0.0);

    queue<int> q;
    vector<int> S;
    dist[s] = 0;
    sigma[s] = 1;
    q.push(s);

    while (!q.empty())
    {
        int v = q.front();
        q.pop();
        S.push_back(v);
        for (int w : G.adj[v])
        {
            if (dist[w] < 0)
            {
                dist[w] = dist[v] + 1;
                q.push(w);
            }
            if (dist[w] == dist[v] + 1)
            {
                sigma[w] += sigma[v];
                P[w].push_back(v);
            }
        }
    }

    for (int i = (int)S.size() - 1; i >= 0; --i)
    {
        int w = S[i];
        for (int v : P[w])
        {
            if (sigma[w] > 0)
                delta[v] += ((double)sigma[v] / (double)sigma[w]) * (1.0 + delta[w]);
        }
    }
}

void recompute_dependencies(int s, const vector<int> &dist,
                            const vector<long long> &sigma,
                            const vector<vector<int>> &P,
                            vector<double> &delta)
{
    int n = dist.size();
    vector<double> new_delta(n, 0.0);
    vector<int> nodes;
    nodes.reserve(n);
    for (int i = 0; i < n; ++i)
        if (dist[i] >= 0)
            nodes.push_back(i);
    sort(nodes.begin(), nodes.end(), [&](int a, int b)
         { return dist[a] > dist[b]; });

    for (int w : nodes)
    {
        if (sigma[w] == 0)
            continue;
        for (int v : P[w])
        {
            if (sigma[w] > 0)
                new_delta[v] += ((double)sigma[v] / (double)sigma[w]) * (1.0 + new_delta[w]);
        }
    }
    delta.swap(new_delta);
}

void propagate_sigma(int start, const Graph &G,
                     const vector<int> &dist, vector<long long> &sigma,
                     const vector<vector<int>> &P)
{
    int n = G.n;
    vector<char> inq(n, 0);
    queue<int> q;
    q.push(start);
    inq[start] = 1;
    while (!q.empty())
    {
        int x = q.front();
        q.pop();
        inq[x] = 0;
        for (int w : G.adj[x])
        {
            if (dist[w] == dist[x] + 1)
            {
                long long new_sigma = 0;
                for (int p : P[w])
                    new_sigma += sigma[p];
                if (new_sigma != sigma[w])
                {
                    sigma[w] = new_sigma;
                    if (!inq[w])
                    {
                        q.push(w);
                        inq[w] = 1;
                    }
                }
            }
        }
    }
}

void add_edge_update_for_source(const Graph &G, int u, int v, int s,
                                vector<int> &dist, vector<long long> &sigma,
                                vector<vector<int>> &P, vector<double> &delta)
{
    if (dist[u] < 0 && dist[v] < 0)
        return;

    if (dist[u] >= 0 && (dist[v] < 0 || dist[u] + 1 < dist[v]))
    {
        bfs_from_source(s, G, dist, sigma, P, delta);
        return;
    }
    if (dist[v] >= 0 && (dist[u] < 0 || dist[v] + 1 < dist[u]))
    {
        bfs_from_source(s, G, dist, sigma, P, delta);
        return;
    }

    if (dist[u] >= 0 && dist[u] + 1 == dist[v])
    {
        if (find(P[v].begin(), P[v].end(), u) == P[v].end())
        {
            P[v].push_back(u);
            sigma[v] += sigma[u];
            propagate_sigma(v, G, dist, sigma, P);
            recompute_dependencies(s, dist, sigma, P, delta);
        }
    }
    if (dist[v] >= 0 && dist[v] + 1 == dist[u])
    {
        if (find(P[u].begin(), P[u].end(), v) == P[u].end())
        {
            P[u].push_back(v);
            sigma[u] += sigma[v];
            propagate_sigma(u, G, dist, sigma, P);
            recompute_dependencies(s, dist, sigma, P, delta);
        }
    }
}

void remove_edge_update_for_source(const Graph &G, int u, int v, int s,
                                   vector<int> &dist, vector<long long> &sigma,
                                   vector<vector<int>> &P, vector<double> &delta)
{
    if (dist[u] < 0 && dist[v] < 0)
        return;

    auto ituv = find(P[v].begin(), P[v].end(), u);
    if (ituv != P[v].end())
    {
        P[v].erase(ituv);
        long long new_sigma_v = 0;
        for (int p : P[v])
            new_sigma_v += sigma[p];
        if (new_sigma_v <= 0)
        {
            bfs_from_source(s, G, dist, sigma, P, delta);
            return;
        }
        else
        {
            sigma[v] = new_sigma_v;
            propagate_sigma(v, G, dist, sigma, P);
            recompute_dependencies(s, dist, sigma, P, delta);
        }
    }

    auto itvu = find(P[u].begin(), P[u].end(), v);
    if (itvu != P[u].end())
    {
        P[u].erase(itvu);
        long long new_sigma_u = 0;
        for (int p : P[u])
            new_sigma_u += sigma[p];
        if (new_sigma_u <= 0)
        {
            bfs_from_source(s, G, dist, sigma, P, delta);
            return;
        }
        else
        {
            sigma[u] = new_sigma_u;
            propagate_sigma(u, G, dist, sigma, P);
            recompute_dependencies(s, dist, sigma, P, delta);
        }
    }
}

void dynamic_bc_mpi_incremental(Graph &G, const vector<pair<int, int>> &updates,
                                const vector<int> &update_type)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = G.n;

    vector<int> my_sources;
    for (int s = rank; s < n; s += size)
        my_sources.push_back(s);
    int m = (int)my_sources.size();

    vector<vector<int>> dist_local(m);
    vector<vector<long long>> sigma_local(m);
    vector<vector<vector<int>>> P_local(m);
    vector<vector<double>> delta_local(m);

    vector<double> localBC_total(n, 0.0);
    for (int idx = 0; idx < m; ++idx)
    {
        int s = my_sources[idx];
        bfs_from_source(s, G, dist_local[idx], sigma_local[idx], P_local[idx], delta_local[idx]);
        for (int v = 0; v < n; ++v)
            if (v != s)
                localBC_total[v] += delta_local[idx][v];
    }
    double total_time = 0.0;
    int num_updates = (int)updates.size();

    for (int i = 0; i < (int)updates.size(); ++i)
    {
        MPI_Barrier(MPI_COMM_WORLD); // synchronize before each update
        double start_time = MPI_Wtime();
        int u = updates[i].first;
        int v = updates[i].second;

        // apply graph-level structural change first (so all ranks proceed on same G)
        if (update_type[i] == 1)
            G.add_edge(u, v);
        else
            G.remove_edge(u, v);

        // for each local source, compute change in delta and update localBC_total
        for (int idx = 0; idx < m; ++idx)
        {
            int s = my_sources[idx];

            // save old delta
            vector<double> old_delta = delta_local[idx];

            // apply update for this source (mutates dist_local/sigma_local/P_local/delta_local)
            if (update_type[i] == 1)
            {
                add_edge_update_for_source(G, u, v, s,
                                           dist_local[idx], sigma_local[idx],
                                           P_local[idx], delta_local[idx]);
            }
            else
            {
                remove_edge_update_for_source(G, u, v, s,
                                              dist_local[idx], sigma_local[idx],
                                              P_local[idx], delta_local[idx]);
            }

            // update localBC_total by difference (new_delta - old_delta)
            for (int x = 0; x < n; ++x)
            {
                if (x == s)
                    continue; // delta for source node not added to BC
                localBC_total[x] += (delta_local[idx][x] - old_delta[x]);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        double end_time = MPI_Wtime();
        double elapsed = end_time - start_time;
        total_time += elapsed;

        // now reduce across ranks once and print
        vector<double> globalBC_after(n, 0.0);
        MPI_Allreduce(localBC_total.data(), globalBC_after.data(), n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }
    if (rank == 0)
    {
        cout << "\nTotal time for " << num_updates << " updates: " << total_time << " sec\n";
        cout << "Average time per update: " << (total_time / num_updates) << " sec\n";
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Read graph and updates on rank 0 only.
    int n = 0, m = 0;
    vector<int> edges_flat; // 2*m ints
    int num_updates = 0;
    vector<int> updates_flat; // 2*num_updates ints
    vector<int> update_type;  // num_updates ints (1=add,0=remove)

    if (rank == 0)
    {
        // If user provided graph filename as argv[1], read from it; otherwise read from stdin.
        std::istream *inptr = &cin;
        std::ifstream fin;
        if (argc >= 2)
        {
            fin.open(argv[1]);
            if (!fin)
            {
                cerr << "Failed to open graph file: " << argv[1] << "\n";
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            inptr = &fin;
        }
        std::istream &in = *inptr;

        if (!(in >> n >> m))
        {
            cerr << "Failed to read n m\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        edges_flat.resize(2 * m);
        for (int i = 0; i < m; ++i)
        {
            int u, v;
            in >> u >> v;
            edges_flat[2 * i] = u;
            edges_flat[2 * i + 1] = v;
        }

        // Now read updates from stdin (after graph). If you want a separate updates file,
        // run: mpirun -np P ./BC graph.txt < updates.txt  and this will read updates from stdin.
        if (!(cin >> num_updates))
        {
            // if no updates provided, set to 0
            num_updates = 0;
        }
        else
        {
            updates_flat.resize(2 * num_updates);
            update_type.resize(num_updates);
            for (int i = 0; i < num_updates; ++i)
            {
                string op;
                int u, v;
                cin >> op >> u >> v;
                updates_flat[2 * i] = u;
                updates_flat[2 * i + 1] = v;
                update_type[i] = (op == "add") ? 1 : 0;
            }
        }
    }

    // Broadcast n and m
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Broadcast edges_flat length and data
    if (rank != 0)
        edges_flat.resize(2 * m);
    if (m > 0)
        MPI_Bcast(edges_flat.data(), 2 * m, MPI_INT, 0, MPI_COMM_WORLD);

    // Broadcast num_updates
    MPI_Bcast(&num_updates, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0)
    {
        updates_flat.resize(2 * num_updates);
        update_type.resize(num_updates);
    }
    if (num_updates > 0)
    {
        MPI_Bcast(updates_flat.data(), 2 * num_updates, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(update_type.data(), num_updates, MPI_INT, 0, MPI_COMM_WORLD);
    }

    // Reconstruct Graph on every rank
    Graph G(n);
    for (int i = 0; i < m; ++i)
    {
        int u = edges_flat[2 * i], v = edges_flat[2 * i + 1];
        // optionally adjust for 1-based input: if (u>0 && v>0 && max index==n) subtract 1
        if (u >= 0 && u < n && v >= 0 && v < n)
            G.add_edge(u, v);
    }

    // Convert updates_flat to vector<pair<int,int>>
    vector<pair<int, int>> updates;
    updates.reserve(num_updates);
    for (int i = 0; i < num_updates; ++i)
        updates.emplace_back(updates_flat[2 * i], updates_flat[2 * i + 1]);

    dynamic_bc_mpi_incremental(G, updates, update_type);
    MPI_Finalize();
    return 0;
}