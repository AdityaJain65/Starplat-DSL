#ifndef OMP_BTREE_H
#define OMP_BTREE_H

// required libraries
#include <omp.h>
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

    // constructor
    Node(int k, int v) : key(k), value(v) {};
    Node() : key(INT_MIN), value(INT_MIN) {};
};


// Btree class
class btree {
    // user method declarations
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

        
        Node *addBatch;
        int *searchBatch;
        int *deleteBatch;
        Node *updateBatch;
        
        int addCount;
        int searchCount;
        int deleteCount;
        int updateCount;

        void printBTree();
        
    private:
        void initMemory();
        void freeMemory();
        
        int* parallelUpdateSearch(Node*, Node*, int, int*);
        void parallelInsert(Node*, int**, int*, int*, Node*, int);
        int* parallelSearch(Node*, int*, int , int*);
        void parallelRemove(Node*, int**, int*, int*, int);

        Node *btreeArray;
        int** holes;
        int *holesCount;
        int *chunkElementCounter;
};

#endif
