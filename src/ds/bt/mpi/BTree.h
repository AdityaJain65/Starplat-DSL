#ifndef MPI_BTREE_H
#define MPI_BTREE_H

// required libraries
#include <mpi.h>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <climits>
#include <ctime>
#include <algorithm>
#include <iostream>


// a structure that defines each element node of the btree
struct Node {
    // integers representing the key and the value
    int key;
    int value;

    // constructors
    Node(int k, int v) : key(k), value(v) {};
    // this default constructor is needed to create the MPI_NODE object which is needed to spread the btree array among all the processes
    Node() : key(0), value(0) {};
};


// Btree class
class btree {
    // method declarations
    public:
        btree();
        ~btree();

        // user APIs
        void insertNode(int, int);
        int search(int);
        void remove(int);
        void update(int, int);

        void batchInsert(Node*, int);
        int* batchSearch(int*, int);
        void batchRemove(int*, int);
        void batchUpdate(Node*, int);
        
        int pid;
        int total_process;
        MPI_Datatype MPI_Node;

        Node *addBatch;
        int *searchBatch;
        int *deleteBatch;
        Node *updateBatch;
        
        int addCount;
        int searchCount;
        int deleteCount;
        int updateCount;

        void printBTree();
    
    // array declarations
    private:
        void initMemory();
        void freeMemory();

        Node *btreeArray;
        int **holes;
};

#endif