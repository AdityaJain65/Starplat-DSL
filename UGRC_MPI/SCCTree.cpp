// dynamic_scc_tree_decr.cpp
// Reference CPU implementation of SCC-Tree with deletion propagation (UNREACHABLE / Rn/Un)
// Build: g++ -std=c++17 -O2 dynamic_scc_tree_decr.cpp -o dynamic_scc_tree_decr
// Run: ./dynamic_scc_tree_decr

#include <bits/stdc++.h>
#include <chrono>
using namespace std;
using Vid = int;
class Edge
{
public:
    Vid u, v;
    Vid u_orig, v_orig;
    Edge(int u, int v)
    {
        this->u = u;
        this->v = v;
        if (u >= 0)
            u_orig = u;
        if (v >= 0)
            v_orig = v;
    }
    Edge(int u, int v, int u_orig, int v_orig)
    {
        this->u = u;
        this->v = v;
        this->u_orig = u_orig;
        this->v_orig = v_orig;
    }
};

// ---------------- Tarjan SCC ----------------
struct Tarjan
{
    int n;
    vector<vector<int>> g;
    vector<int> disc, low, st;
    vector<char> inStack;
    int timeCounter = 0;
    vector<vector<int>> sccs;
    Tarjan(int n = 0) : n(n), g(n), disc(n, -1), low(n, 0), inStack(n, 0) {}
    void reset(int N)
    {
        n = N;
        g.assign(N, {});
        disc.assign(N, -1);
        low.assign(N, 0);
        inStack.assign(N, 0);
        st.clear();
        timeCounter = 0;
        sccs.clear();
    }
    void add_edge(int a, int b) { g[a].push_back(b); }
    void dfs(int u)
    {
        disc[u] = low[u] = timeCounter++;
        st.push_back(u);
        inStack[u] = 1;
        for (int v : g[u])
        {
            if (disc[v] == -1)
            {
                dfs(v);
                low[u] = min(low[u], low[v]);
            }
            else if (inStack[v])
                low[u] = min(low[u], disc[v]);
        }
        if (low[u] == disc[u])
        {
            vector<int> comp;
            while (true)
            {
                int w = st.back();
                st.pop_back();
                inStack[w] = 0;
                comp.push_back(w);
                if (w == u)
                    break;
            }
            sccs.push_back(move(comp));
        }
    }
    vector<vector<int>> run()
    {
        for (int i = 0; i < n; i++)
            if (disc[i] == -1)
                dfs(i);
        return sccs;
    }
};

// ---------------- SCC-tree structures ----------------
struct SCTNode
{
    int label;                // negative internal labels, master = -1
    vector<Vid> verts;        // original vertices inside D(N)
    vector<Edge> local_edges; // edges (original endpoints) considered inside D(N)
    int induced_vertex = -1;
    int parent = 0;       // parent label
    vector<int> children; // if child >=0 -> leaf vertex id, else internal node label
    bool is_leaf() const { return children.empty(); }
    SCTNode() = default;
};

struct LabelGen
{
    int cur = -2;
    int next() { return cur--; }
} labgen;

struct SCCForest
{
    unordered_map<int, SCTNode> nodes;
    int master_label = -1;
    unordered_map<Vid, int> vertex_to_leaf; // mapping vertex -> node label that directly contains it (parent)
};

// ---------------- helper utilities ----------------
static void initialize_master(SCCForest &F, const vector<Vid> &verts, const vector<Edge> &edges)
{
    F.nodes.clear();
    labgen.cur = -2;
    SCTNode root;
    root.label = -1;
    root.verts = verts;
    root.children = verts;
    root.local_edges = edges;
    root.induced_vertex = -1;
    root.parent = 0;
    F.nodes[-1] = move(root);
    for (Vid v : verts)
        F.vertex_to_leaf[v] = -1;
}

static int create_internal_node(SCCForest &F)
{
    int label = labgen.next();
    SCTNode n;
    n.label = label;
    F.nodes[label] = move(n);
    return label;
}

// build adjacency (index mapping) for a node's subgraph where necessary
static vector<vector<int>> build_adj(const vector<Vid> &verts, const vector<Edge> &edges, unordered_map<Vid, int> &vid_to_idx_out)
{
    int n = (int)verts.size();
    unordered_map<Vid, int> vid_to_idx;
    for (int i = 0; i < n; i++)
        vid_to_idx[verts[i]] = i;
    vid_to_idx_out = vid_to_idx;
    vector<vector<int>> adj(n);
    for (const Edge &e : edges)
    {
        auto it1 = vid_to_idx.find(e.u), it2 = vid_to_idx.find(e.v);
        if (it1 != vid_to_idx.end() && it2 != vid_to_idx.end())
        {
            adj[it1->second].push_back(it2->second);
        }
    }
    return adj;
}

// SPLITANDCONDENSE but returning condensed graph (component -> member original indices) and mapping of which comp contains d_in / d_out
struct Condensed
{
    // components : each is vector of *split-graph indices* (we will map them to original vertices)
    vector<vector<int>> comps;
    // mapping from split index -> comp id
    vector<int> node_to_comp;
    // adjacency of condensed graph (component graph)
    vector<vector<int>> comp_adj;
    vector<vector<int>> comp_members_orig; // each comp -> set of original vertex ids (deduped)
    int comp_of_din = -1;
    int comp_of_dout = -1;
};

// Build split graph for node: indices 0..n-1 original vertices, optionally extra indices for d_in,d_out
// Then run Tarjan on split graph and build condensed graph
static Condensed split_and_condense_graph(const SCTNode &node, Vid induced_vertex)
{
    Condensed out;
    int n = (int)node.verts.size();
    unordered_map<Vid, int> vid_to_idx;
    for (int i = 0; i < n; i++)
        vid_to_idx[node.verts[i]] = i;
    bool use_split = (induced_vertex != -1 && vid_to_idx.count(induced_vertex));
    int extra = use_split ? 2 : 0;
    int N = n + extra;
    int idx_din = use_split ? n : -1;
    int idx_dout = use_split ? n + 1 : -1;
    vector<vector<int>> adj(N);
    // build split edges
    for (const Edge &e : node.local_edges)
    {
        auto itu = vid_to_idx.find(e.u), itv = vid_to_idx.find(e.v);
        if (itu == vid_to_idx.end() || itv == vid_to_idx.end())
            continue;
        int iu = itu->second, iv = itv->second;
        if (use_split)
        {
            if (e.v == induced_vertex)
                iv = idx_din;
            if (e.u == induced_vertex)
                iu = idx_dout;
        }
        adj[iu].push_back(iv);
    }
    // run tarjan on split graph
    Tarjan T(N);
    T.reset(N);
    for (int u = 0; u < N; u++)
        for (int v : adj[u])
            T.add_edge(u, v);
    auto sccs = T.run();
    // map node index -> comp id
    vector<int> node_to_comp(N, -1);
    // cout << "******\n\n";
    for (int i = 0; i < (int)sccs.size(); i++)
    {
        for (int idx : sccs[i])
        {
            node_to_comp[idx] = i;
            // cout << idx << " " << i << endl;
        }
    }
    int C = (int)sccs.size();
    vector<unordered_set<int>> comp_adj_set(C);
    // build condensed adjacency between components
    for (int u = 0; u < N; u++)
    {
        int cu = node_to_comp[u];
        for (int v : adj[u])
        {
            int cv = node_to_comp[v];
            if (cu != cv)
                comp_adj_set[cu].insert(cv);
        }
    }
    out.comps = sccs;                // index
    out.node_to_comp = node_to_comp; // index to comp
    out.comp_adj.assign(C, {});
    for (int i = 0; i < C; i++)
    {
        for (int t : comp_adj_set[i])
            out.comp_adj[i].push_back(t);
    }
    // build comp_members_orig: map each comp -> set of original vertex ids (dedup)
    out.comp_members_orig.assign(C, {}); // vertices to components
    for (int comp = 0; comp < C; ++comp)
    {
        unordered_set<int> s;
        for (int idx : sccs[comp])
        {
            if (idx < n)
            {
                // original vertex index => real original vid
                s.insert(node.verts[idx]);
            }
            else
            {
                // idx corresponds to d_in or d_out -> map it to induced_vertex
                if (use_split && (idx == idx_din || idx == idx_dout))
                    s.insert(induced_vertex);
            }
        }
        // copy to vector
        for (int v : s)
            out.comp_members_orig[comp].push_back(v);
    }
    if (use_split)
    {
        out.comp_of_din = node_to_comp[idx_din];
        out.comp_of_dout = node_to_comp[idx_dout];
    }
    else
    {
        out.comp_of_din = out.comp_of_dout = -1;
    }
    return out;
}

// BFS on condensed graph from a start comp
static vector<char> bfs_comp(const vector<vector<int>> &comp_adj, int start)
{
    int C = (int)comp_adj.size();
    vector<char> vis(C, 0);
    if (start < 0 || start >= C)
        return vis;
    deque<int> dq;
    dq.push_back(start);
    vis[start] = 1;
    while (!dq.empty())
    {
        int x = dq.front();
        dq.pop_front();
        for (int y : comp_adj[x])
        {
            if (!vis[y])
            {
                vis[y] = 1;
                dq.push_back(y);
            }
        }
    }
    return vis;
}

// reverse adjacency
static vector<vector<int>> reverse_adj(const vector<vector<int>> &adj)
{
    int n = (int)adj.size();
    vector<vector<int>> r(n);
    for (int u = 0; u < n; u++)
        for (int v : adj[u])
            r[v].push_back(u);
    return r;
}

// find node label that contains vertex v (search internal nodes and leaves)
static int find_leaf_parent(SCCForest &F, Vid v)
{
    auto it = F.vertex_to_leaf.find(v);
    if (it != F.vertex_to_leaf.end())
        return it->second;
    // fallback: search nodes
    for (auto &kv : F.nodes)
    {
        int lab = kv.first;
        auto &n = kv.second;
        if (find(n.verts.begin(), n.verts.end(), v) != n.verts.end())
            return lab;
    }
    return F.master_label;
}

// --------- create_scc_tree simple reference used for building initial tree -------------
static void create_scc_tree_rec(SCCForest &F, int node_label);

static void create_scc_tree(SCCForest &F)
{
    create_scc_tree_rec(F, F.master_label);
}

// This is the same creation code as earlier, used to initialize nodes recursively
static void create_scc_tree_rec(SCCForest &F, int node_label)
{
    // if (F.nodes.find(0) != F.nodes.end())
    // {
    //     cout << "**" << 1 << endl;
    // }
    auto &node = F.nodes[node_label];
    if (node_label != -1 && !node.verts.empty())
        node.induced_vertex = node.verts[0];
    if (node.verts.size() <= 1)
    {
        node.children.clear();
        // node.induced_vertex = -1;
        for (Vid v : node.verts)
        {
            node.children.push_back((int)v);
            if (v >= 0)
                F.vertex_to_leaf[v] = node_label;
            else
                F.nodes[v].parent = node_label;
        }
        // if (F.nodes.find(0) != F.nodes.end())
        // {
        //     cout << "**" << 2 << node_label << endl;
        // }
        return;
    }
    Vid d = node.induced_vertex;
    auto cond = split_and_condense_graph(node, d);
    // If only one component and equal size -> no split
    // compute total unique original vertices in cond
    unordered_set<Vid> unionv;
    unordered_map<Vid, int> vertex_to_child;
    for (auto &vec : cond.comp_members_orig)
        for (Vid x : vec)
            unionv.insert(x);
    if (cond.comp_members_orig.size() == 1 && unionv.size() == node.verts.size())
    {
        node.children.clear();
        // node.induced_vertex=-1;
        if (cond.comp_members_orig[0].size() == 1)
            for (Vid v : node.verts)
            {
                node.children.push_back((int)v);
                vertex_to_child[v] = v;
                if (v >= 0)
                    F.vertex_to_leaf[v] = node_label;
                else
                    F.nodes[v].parent = node_label;
            }
        else
        {
            int child_label = create_internal_node(F);
            auto &child = F.nodes[child_label];
            child.label = child_label;
            child.verts = node.verts;
            child.parent = node_label;
            // copy edges internal to comp
            for (auto vertex : node.verts)
                vertex_to_child[vertex] = child_label;
            for (const Edge &e : node.local_edges)
            {
                if (find(node.verts.begin(), node.verts.end(), e.u) != node.verts.end() &&
                    find(node.verts.begin(), node.verts.end(), e.v) != node.verts.end())
                {
                    child.local_edges.push_back(e);
                }
                else
                {
                    cout << "error1\n";
                }
            }
            for (auto vertex : child.verts)
            {
                if (vertex >= 0)
                {
                    F.vertex_to_leaf[vertex] = child_label;
                }
                else
                {
                    F.nodes[vertex].parent = child_label;
                }
            }
            node.local_edges.clear();
            node.children.push_back(child_label);
            node.verts.clear();
            node.verts.push_back(child_label);
            create_scc_tree_rec(F, child_label);
        }
        return;
    }
    node.children.clear();
    // node.induced_vertex = -1;
    for (auto &comp_members : cond.comp_members_orig)
    {
        if (comp_members.size() == 1)
        {
            Vid v = comp_members[0];
            if (find(node.children.begin(), node.children.end(), (int)v) == node.children.end())
            {
                node.children.push_back((int)v);
            }
            if (v >= 0)
                F.vertex_to_leaf[v] = node_label;
            else
                F.nodes[v].parent = node_label;
            vertex_to_child[v] = v;
        }
        else
        {
            int child_label = create_internal_node(F);
            auto &child = F.nodes[child_label];
            child.label = child_label;
            child.verts = comp_members;
            child.parent = node_label;
            // copy edges internal to comp
            for (auto vertex : child.verts)
            {
                vertex_to_child[vertex] = child_label;
            }
            for (auto vertex : child.verts)
            {
                if (vertex >= 0)
                {
                    F.vertex_to_leaf[vertex] = child_label;
                }
                else
                {
                    F.nodes[vertex].parent = child_label;
                }
            }
            node.children.push_back(child_label);
        }
    }
    vector<Edge> new_edges;
    for (const Edge &e : node.local_edges)
    {
        int u = e.u;
        int v = e.v;
        int mapped_u = vertex_to_child[u];
        int mapped_v = vertex_to_child[v];
        if (mapped_u == mapped_v)
        {
            F.nodes[mapped_u].local_edges.push_back({u, v, e.u_orig, e.v_orig});
        }
        else
        {
            new_edges.push_back({mapped_u, mapped_v, e.u_orig, e.v_orig});
        }
    }
    node.verts = node.children;
    node.local_edges = new_edges;
    for (auto child_label : node.children)
    {
        if (child_label < 0)
            create_scc_tree_rec(F, child_label);
    }
}

// ----------------- deletion propagation (UNREACHABLE / Rn/Un) -----------------
static void promote_vertices_to_parent(SCCForest &F, int node_label, const unordered_set<Vid> &vertsToPromote)
{
    // node_label is the node being removed/partially removed; we attach vertsToPromote as children to parent
    if (F.nodes.find(node_label) == F.nodes.end())
        return;
    int parent = F.nodes[node_label].parent;
    if (parent == 0)
        parent = F.master_label;
    // ensure parent's node exists
    auto itp = F.nodes.find(parent);
    if (itp == F.nodes.end())
    {
        // create parent placeholder (shouldn't happen)
        SCTNode p;
        p.label = parent;
        F.nodes[parent] = p;
        cout << "error6\n";
        return;
    }
    auto &P = F.nodes[parent];
    // Add promoted verts as direct children (leaf form). If parent already has internal child containing some, skip duplicates.
    for (Vid v : vertsToPromote)
    {
        // if parent already has a child equal to vertex, skip
        bool exists = false;
        // for (int ch : P.children)
        // {
        //     if (ch >= 0 && ch == v)
        //     {
        //         exists = true;
        //         break;
        //     }
        //     if (ch < 0)
        //     {
        //         // check if this child contains v
        //         auto itc = F.nodes.find(ch);
        //         if (itc != F.nodes.end())
        //         {
        //             auto &cn = itc->second;
        //             if (find(cn.verts.begin(), cn.verts.end(), v) != cn.verts.end())
        //             {
        //                 exists = true;
        //                 break;
        //             }
        //         }
        //     }
        // }
        if (!exists)
        {
            P.children.push_back((int)v);
            P.verts.push_back(v);
        }
        // map vertex to parent in vertex_to_leaf
        if (v >= 0)
            F.vertex_to_leaf[v] = parent;
        else
            F.nodes[v].parent = parent;
    }
}

void get_map(SCCForest &F, unordered_map<int, int> &vertices_to_child, int curr_vertex, int child)
{

    vertices_to_child[curr_vertex] = child;
    if (curr_vertex < 0)
        for (auto vertex : F.nodes[curr_vertex].verts)
        {
            get_map(F, vertices_to_child, vertex, child);
        }
}
void update_parent_edges(SCCForest &F, int node_label, const unordered_set<Vid> &vertsToPromote)
{
    unordered_map<int, int> vertices_to_child;
    for (auto vertex : vertsToPromote)
        get_map(F, vertices_to_child, vertex, vertex);
    auto &node = F.nodes[node_label];
    for (auto &edge : node.local_edges)
    {
        int u_orig = edge.u_orig;
        int v_orig = edge.v_orig;
        if (vertices_to_child.find(u_orig) != vertices_to_child.end())
        {
            edge.u = vertices_to_child[u_orig];
        }
        if (vertices_to_child.find(v_orig) != vertices_to_child.end())
        {
            edge.v = vertices_to_child[v_orig];
        }
    }
}

// helper to move edges from node N to parent P when they involve vertices not in Rn
static void move_edges_to_parent(SCCForest &F, int node_label, const unordered_set<Vid> &Rn_set)
{
    auto itnode = F.nodes.find(node_label);
    if (itnode == F.nodes.end())
        return;
    auto &node = itnode->second;
    int parent = node.parent;
    if (parent == 0)
        parent = F.master_label;
    auto &P = F.nodes[parent];
    vector<Edge> remaining;
    for (Edge e : node.local_edges)
    {
        bool u_in = Rn_set.count(e.u) > 0;
        bool v_in = Rn_set.count(e.v) > 0;
        if (u_in && v_in)
        {
            remaining.push_back(e); // stays with node
        }
        else
        {
            if (u_in)
                e.u = node_label;
            else if (v_in)
                e.v = node_label;
            P.local_edges.push_back(e);
        }
    }
    node.local_edges = move(remaining);
}

// Convert set<int> comps to set<Vid> of original vertices given Condensed data
static unordered_set<Vid> comps_to_vertices(const Condensed &cond, const vector<int> &compsIndexList)
{
    unordered_set<Vid> s;
    for (int cid : compsIndexList)
    {
        for (Vid v : cond.comp_members_orig[cid])
            s.insert(v);
    }
    return s;
}

static void update_scc_node(SCCForest &F, int node_label);

// The main decremental propagation starting at node_label (after edge removed from node.local_edges)
static void deletion_propagate(SCCForest &F, int node_label)
{

    if (F.nodes.find(node_label) == F.nodes.end())
        return;
    // 1111111111
    if (node_label == F.master_label)
    {
        // Rebuild/refresh the master node after changes bubbled up.
        // This will recompute SPLIT/CONDENSE on the root and update children.
        // 111111111111111111
        // update_scc_node(F, F.master_label);

        // Optional: delete any now-unreferenced internal nodes across forest
        // (use recursive deletion to avoid global O(n) scan if you implemented delete_subtree)
        // If you prefer a conservative cleanup: run the existing cleanup for master-level old internals here.
        update_scc_node(F, node_label);
        return;
    }
    auto &node = F.nodes[node_label];
    if (node.verts.empty())
    {
        // nothing
        return;
    }
    // Build split-and-condense for (current) D(N)
    // cout << node.verts.size() << endl;
    if (find(node.verts.begin(), node.verts.end(), node.induced_vertex) == node.verts.end())
    {
        node.induced_vertex = -1;
    }
    if (node.induced_vertex == -1)
    {
        node.induced_vertex = node.verts[0];
    }
    Condensed cond = split_and_condense_graph(node, node.induced_vertex);
    // if split not used (no induced vertex present or single comp) then Rn = all comps -> nothing to do
    if (cond.comp_adj.empty() && cond.comps.size() <= 1)
    {
        // nothing to do
        return;
    }
    int C = (int)cond.comps.size();
    // cout << C << "**" << node.induced_vertex << endl;
    // identify comp_of_dout and comp_of_din
    int comp_sout = cond.comp_of_dout;
    int comp_sin = cond.comp_of_din;
    if (comp_sout < 0 || comp_sin < 0)
    {
        // No split (should not happen here) -> fallback to recompute node only
        // keep as is
        // cout << "***" << node.induced_vertex << endl;
        cout << "error5\n";
        return;
    }
    // BFS from sout on condensed graph
    vector<char> reach_sout = bfs_comp(cond.comp_adj, comp_sout);
    // BFS from sin on reversed condensed graph
    auto rev = reverse_adj(cond.comp_adj);
    vector<char> reach_sin = bfs_comp(rev, comp_sin);
    // Rn = intersection
    vector<char> in_Rn(C, 0);
    for (int i = 0; i < C; i++)
        if (reach_sout[i] && reach_sin[i])
            in_Rn[i] = 1;
    // collect Un comps (those not in Rn)
    vector<int> Un_comps, Rn_comps;
    for (int i = 0; i < C; i++)
    {
        if (in_Rn[i])
        {
            Rn_comps.push_back(i);
            // cout << "**" << i << endl;
        }
        else
            Un_comps.push_back(i);
    }
    // map comps -> original vertices
    unordered_set<Vid> Rn_orig = comps_to_vertices(cond, Rn_comps);
    // Rn_orig.insert(node.induced_vertex);
    unordered_set<Vid> Un_orig = comps_to_vertices(cond, Un_comps);
    if (Rn_orig.count(node.induced_vertex))
    {
        Un_orig.erase(node.induced_vertex);
    }
    // Un_orig.erase(node.induced_vertex);
    // for (auto x : Un_orig)
    //     cout << x << endl;
    if (Rn_comps.empty())
    {
        // Entire node becomes unreachable => remove node
        // Move its vertices to parent as siblings (promote)
        promote_vertices_to_parent(F, node_label, Un_orig);
        update_parent_edges(F, F.nodes[node_label].parent, Un_orig);
        // Move edges from node to parent
        move_edges_to_parent(F, node_label, /*Rn_set=*/unordered_set<Vid>{}); // all moved
        // Remove node from parent's children list
        int parent = node.parent;
        if (parent == 0)
            parent = F.master_label;
        auto &P = F.nodes[parent];
        P.children.erase(remove(P.children.begin(), P.children.end(), node_label), P.children.end());
        P.verts.erase(remove(P.verts.begin(), P.verts.end(), node_label), P.verts.end());
        // delete node
        F.nodes.erase(node_label);
        // Now propagate to parent
        if (parent != 0)
            deletion_propagate(F, parent);
        return;
    }
    else
    {
        // Some reachable remain: shrink node to Rn_orig, promote Un_orig to parent
        // Update node.verts to Rn_orig
        if (Rn_orig.count(node.induced_vertex) == 0)
            cout << "error_infinity\n";
        vector<Vid> new_verts;
        for (Vid v : node.verts)
            if (Rn_orig.count(v))
                new_verts.push_back(v);
        node.verts = move(new_verts);
        node.children = node.verts;
        // If induced vertex moved out of node (i.e., not in Rn_orig), pick new induced vertex from remaining verts
        if (node.induced_vertex != -1 && !Rn_orig.count(node.induced_vertex))
        {
            if (!node.verts.empty())
                node.induced_vertex = node.verts[0];
            else
                node.induced_vertex = -1;
        }
        // move Un vertices to parent
        promote_vertices_to_parent(F, node_label, Un_orig);
        update_parent_edges(F, F.nodes[node_label].parent, Un_orig);
        // move crossing edges to parent (edges not fully inside Rn_orig)
        move_edges_to_parent(F, node_label, Rn_orig);
        // rebuild node's children structure for the new node. We'll re-run a small local rebuild:
        // remove children that correspond to promoted vertices
        // vector<int> new_children;
        // for (int ch : node.children)
        // {
        //     if (ch >= 0)
        //     {
        //         // leaf vertex - if it was promoted, skip, else keep
        //         if (Rn_orig.count(ch))
        //             new_children.push_back(ch);
        //         else
        //         {
        //             // already was promoted
        //         }
        //     }
        //     else
        //     {
        //         // internal child: keep it if it contains any vertex in Rn_orig, else it was promoted
        //         auto itc = F.nodes.find(ch);
        //         if (itc != F.nodes.end())
        //         {
        //             bool keep = false;
        //             for (Vid v : itc->second.verts)
        //                 if (Rn_orig.count(v))
        //                 {
        //                     keep = true;
        //                     break;
        //                 }
        //             if (keep)
        //             {
        //                 // shrink the child as well? For simplicity, we leave internal child to be cleaned up when its own deletion_propagate runs.
        //                 new_children.push_back(ch);
        //             }
        //             else
        //             {
        //                 // orphan internal child: change its parent (promote its verts to parent)
        //                 unordered_set<Vid> tmp;
        //                 for (Vid v : itc->second.verts)
        //                     tmp.insert(v);
        //                 promote_vertices_to_parent(F, ch, tmp);
        //                 // remove old node if unreferenced
        //                 // we will not aggressively erase here; parent propagation will clean up
        //             }
        //         }
        //     }
        // }
        // node.children = move(new_children);
        // Now propagate upwards to parent: parent may need to be adjusted because new children (promoted vertices) were added to it
        int parent = node.parent;
        if (parent == 0)
            parent = F.master_label;
        deletion_propagate(F, parent);
        return;
    }
}

// ----------------- insert & remove edge (updated) -----------------
static vector<int> path_to_root(SCCForest &F, int node_label)
{
    vector<int> path;
    int cur = node_label;
    if (cur >= 0)
    {
        auto it = F.vertex_to_leaf.find(cur);
        if (it != F.vertex_to_leaf.end())
            cur = it->second;
    }
    while (true)
    {
        path.push_back(cur);
        if (cur == F.master_label)
            break;
        auto it = F.nodes.find(cur);
        if (it == F.nodes.end())
            break;
        int p = it->second.parent;
        if (p == 0)
            break;
        cur = p;
    }
    return path;
}

static pair<int, pair<int, int>> find_lca(SCCForest &F, Vid u, Vid v)
{
    auto it1 = F.vertex_to_leaf.find(u), it2 = F.vertex_to_leaf.find(v);
    if (it1 == F.vertex_to_leaf.end() || it2 == F.vertex_to_leaf.end())
    {
        cout << "error3\n";
        return {F.master_label, {-1, -1}};
    }
    int lu = it1->second, lv = it2->second;
    auto pu = path_to_root(F, lu), pv = path_to_root(F, lv);
    reverse(pu.begin(), pu.end());
    reverse(pv.begin(), pv.end());
    int lca = F.master_label;
    pu.push_back(u);
    pv.push_back(v);
    int m = min(pu.size(), pv.size());
    int A = u;
    int B = v;
    for (int i = 0; i < m; i++)
    {
        if (pu[i] == pv[i])
        {
            lca = pu[i];
            if (i + 1 < pu.size())
            {
                A = pu[i + 1];
            }
            if (i + 1 < pv.size())
            {
                B = pv[i + 1];
            }
        }
        else
            break;
    }
    return {lca, {A, B}};
}
// Helper function — place outside the main function
void delete_subtree(SCCForest &F, int node_label)
{
    if (F.nodes.find(node_label) == F.nodes.end())
        return;
    auto children = F.nodes[node_label].children; // copy because we’ll erase
    for (int ch : children)
    {
        if (ch < 0)
            delete_subtree(F, ch);
    }
    F.nodes.erase(node_label);
}

static void update_scc_node(SCCForest &F, int node_label)
{
    // For creation and merges we can reuse previous create_scc_tree_rec on that node
    if (F.nodes.find(node_label) == F.nodes.end())
        return;
    auto &node = F.nodes[node_label];
    // 11111111111111111111
    if (node.verts.size() <= 1)
    // if (node_label != F.master_label && node.verts.size() <= 1)
    {
        node.children.clear();
        // node.induced_vertex = -1;
        for (Vid v : node.verts)
        {
            node.children.push_back((int)v);
            if (v >= 0)
                F.vertex_to_leaf[v] = node_label;
            else
                F.nodes[v].parent = node_label;
        }
        return;
    }
    if (find(node.verts.begin(), node.verts.end(), node.induced_vertex) == node.verts.end())
    {
        node.induced_vertex = -1;
    }
    if (node_label != F.master_label && node.induced_vertex == -1 && !node.verts.empty())
        node.induced_vertex = node.verts[0];
    if (node.verts.size() <= 1)
    {
        return;
    }
    // if (node_label == -1)
    //     cout << node.induced_vertex << endl;
    Condensed cond = split_and_condense_graph(node, node.induced_vertex);
    // if cond has only one comp with same vertices -> just make leaves
    unordered_set<Vid> unionv;
    for (auto &vec : cond.comp_members_orig)
        for (Vid x : vec)
            unionv.insert(x);
    if (cond.comp_members_orig.size() == 1 && unionv.size() == node.verts.size())
    {
        node.children.clear();
        // node.induced_vertex = -1;
        if (cond.comp_members_orig[0].size() == 1)
        {
            for (Vid v : node.verts)
            {
                node.children.push_back((int)v);
                if (v >= 0)
                    F.vertex_to_leaf[v] = node_label;
                else
                    F.nodes[v].parent = node_label;
            }
            node.verts = node.children;
        }
        else
        {
            int child_label = create_internal_node(F);
            auto &child = F.nodes[child_label];
            child.label = child_label;
            child.verts = node.verts;
            child.parent = node_label;
            // copy edges internal to comp
            for (const Edge &e : node.local_edges)
            {
                if (find(node.verts.begin(), node.verts.end(), e.u) != node.verts.end() &&
                    find(node.verts.begin(), node.verts.end(), e.v) != node.verts.end())
                {
                    child.local_edges.push_back(e);
                }
                else
                    cout << "error2\n";
            }
            node.children.push_back(child_label);
            node.verts.clear();
            node.verts = node.children;
            node.local_edges.clear();
            for (auto vertex : child.verts)
            {
                if (vertex < 0)
                {
                    F.nodes[vertex].parent = child_label;
                }
                else
                    F.vertex_to_leaf[vertex] = child_label;
            }
            create_scc_tree_rec(F, child_label);
        }
        return;
    }
    // otherwise rebuild children using condensed components
    vector<int> old_children = node.children;
    unordered_set<int> old_internals;
    for (int ch : old_children)
        if (ch < 0)
            old_internals.insert(ch);
    node.children.clear();
    // node.induced_vertex = -1;
    map<Vid, int> vertex_to_child;
    for (auto &comp_members : cond.comp_members_orig)
    {
        if (comp_members.size() == 1)
        {
            if (comp_members.size() == 1)
            {
                Vid v = comp_members[0];
                // avoid duplicate children
                if (find(node.children.begin(), node.children.end(), (int)v) == node.children.end())
                {
                    node.children.push_back((int)v);
                }
                if (v >= 0)
                    F.vertex_to_leaf[v] = node_label;
                else
                    F.nodes[v].parent = node_label;
                vertex_to_child[v] = v;
            }
        }
        else
        {
            int child_label = create_internal_node(F);
            auto &child = F.nodes[child_label];
            child.label = child_label;
            child.verts = comp_members;
            child.parent = node_label;
            for (auto vertex : child.verts)
                vertex_to_child[vertex] = child.label;
            for (auto vertex : child.verts)
            {
                if (vertex < 0)
                {
                    F.nodes[vertex].parent = child_label;
                }
                else
                    F.vertex_to_leaf[vertex] = child_label;
            }
            node.children.push_back(child_label);
        }
    }
    vector<Edge> new_edges;
    for (const Edge &e : node.local_edges)
    {
        int u = e.u;
        int v = e.v;
        int mapped_u = vertex_to_child[u];
        int mapped_v = vertex_to_child[v];
        if (mapped_u == mapped_v)
        {
            F.nodes[mapped_u].local_edges.push_back({u, v, e.u_orig, e.v_orig});
        }
        else
        {
            new_edges.push_back({mapped_u, mapped_v, e.u_orig, e.v_orig});
        }
    }
    node.verts = node.children;
    node.local_edges = new_edges;
    for (auto child_label : node.children)
    {
        if (child_label < 0)
            create_scc_tree_rec(F, child_label);
    }

    // cleanup old internals not referenced
    // --- recursive cleanup ---
    for (int oldlab : old_internals)
    {
        if (oldlab == node_label)
            continue;
        if (F.nodes.find(oldlab) != F.nodes.end())
        {
            bool referenced = false;
            for (auto &kv : F.nodes)
                for (int ch : kv.second.children)
                    if (ch == oldlab)
                        referenced = true;

            if (!referenced)
                delete_subtree(F, oldlab); // deletes the whole subtree efficiently
        }
    }
}

// insertion (unchanged logic)
static void insert_edge(SCCForest &F, Vid u, Vid v)
{
    pair<int, pair<int, int>> p = find_lca(F, u, v);
    int lca = p.first;
    int A = p.second.first;
    int B = p.second.second;
    auto &ln = F.nodes[lca];
    ln.local_edges.push_back({A, B, u, v});
    update_scc_node(F, lca);
}

// remove_edge now uses deletion_propagate
static void remove_edge(SCCForest &F, Vid u, Vid v)
{
    pair<int, pair<int, int>> p = find_lca(F, u, v);
    int lca = p.first;
    int A = p.second.first;
    int B = p.second.second;
    // cout << A << B << endl;
    auto &root = F.nodes[lca];
    root.local_edges.erase(remove_if(root.local_edges.begin(), root.local_edges.end(),
                                     [&](const Edge &e)
                                     { return e.u == A && e.v == B && e.u_orig == u && e.v_orig == v; }),
                           root.local_edges.end());
    deletion_propagate(F, lca);
}

// some utilities for debugging / printing
static void print_forest(SCCForest &F)
{
    cout << "--- SCC Forest State ---\n";
    for (auto &kv : F.nodes)
    {
        int lab = kv.first;
        auto &n = kv.second;
        cout << "Node label: " << lab << " parent: " << n.parent << " verts: {";
        for (auto v : n.verts)
            cout << v << ",";
        cout << "} children: [";
        for (auto c : n.children)
            cout << c << ",";
        cout << "] edges: {";
        for (auto &e : n.local_edges)
            cout << "(" << e.u << "->" << e.v << "),";
        cout << "}\n";
    }
    cout << "vertex_to_leaf mapping:\n";
    for (auto &kv : F.vertex_to_leaf)
        cout << kv.first << " -> " << kv.second << "\n";
    cout << "------------------------\n";
}

static bool query_same_scc(SCCForest &F, Vid u, Vid v)
{
    auto it1 = F.vertex_to_leaf.find(u), it2 = F.vertex_to_leaf.find(v);
    if (it1 == F.vertex_to_leaf.end() || it2 == F.vertex_to_leaf.end())
        return false;
    return it1->second == it2->second;
}
static int sccCount(SCCForest &F)
{
    auto it = F.nodes.find(F.master_label);
    if (it == F.nodes.end())
        return 0;
    return (int)it->second.children.size();
}

// ------------------- example main -------------------
int main()
{
    vector<Vid> verts;
    int n;
    int m;
    cin >> n >> m;
    for (int i = 0; i < n; i++)
        verts.push_back(i);
    vector<Edge> edges;
    // vector<Edge> edges = {
    //     {0, 1}, {1, 2}, {2, 0}, {2, 4}, {4, 6}, {6, 2}, {0, 3}, {3, 5}, {5, 7}, {7, 3}};
    for (int i = 0; i < m; i++)
    {
        int u, v;
        cin >> u >> v;
        edges.push_back({u, v});
    }
    SCCForest F;
    initialize_master(F, verts, edges);
    create_scc_tree(F);
    // cout << "Initial sccCount(): " << sccCount(F) << "\n";
    // print_forest(F);
    int num_updates;
    cin >> num_updates;
    // cout << "\nInserting edge 4->3\n";
    // insert_edge(F, 4, 3);
    // print_forest(F);
    // cout << "sccCount(): " << sccCount(F) << "\n";

    // cout << "\nRemoving edge 2->4 (propagating)\n";
    // remove_edge(F, 2, 4);
    // print_forest(F);
    // cout << "sccCount(): " << sccCount(F) << "\n";

    // cout << "\nInserting edge 3->0\n";
    // insert_edge(F, 3, 0);
    // print_forest(F);
    // cout << "sccCount(): " << sccCount(F) << "\n";
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_updates; i++)
    {
        string op;
        int u, v;
        cin >> op >> u >> v;
        if (op == "add")
        {
            insert_edge(F, u, v);
            // cout << sccCount(F) << "\n";
            // print_forest(F);
        }
        else
        {
            remove_edge(F, u, v);
            // cout << sccCount(F) << "\n";
            // print_forest(F);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate duration
    std::chrono::duration<double> duration = end - start;
    std::cout << "Time taken for " << num_updates << " updates: " << duration.count() << " seconds" << endl;
    cout << "Average time per update: " << duration.count() / num_updates << " seconds" << endl;
    return 0;
}
