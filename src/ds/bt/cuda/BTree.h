#ifndef CUDA_BTREE_H
#define CUDA_BTREE_H

// required libraries
#include <cuda.h>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <climits>
#include <iostream>


// a structure that defines each element node of the btree
struct Node {
    // integers representing the key and the value
    int key;
    int value;

    // constructor
    Node(int k, int v) : key(k), value(v) {};
};


// Btree class
class btree {
    // method declarations
    public:
        btree();
        ~btree();

        // user APIs
        void insertNode(int, int);
        int searchDevice(int);
        int searchHost(int);
        void remove(int);
        void update(int, int);

        void batchInsert(Node*, int);
        int* batchSearchHost(int*, int);
        int* batchSearchDevice(int*, int);
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

        Node *btreeArray;
        int **holes;
        int *elementCounter;
        unsigned int *lastHoleCounter;

        // these two arrays are alloted to store the batch elemenets and the search result for the searchDevice function
        // this is done because doing cudaMalloc() inside __device__ function is not allowed
        // hence we do the cudaMalloc() in the constructor
        int *batchElementsSearchDevice;
        int *searchResultDevice;
        
        int insertionChunkPos;
        int *insertionChunksHost;
};

#endif