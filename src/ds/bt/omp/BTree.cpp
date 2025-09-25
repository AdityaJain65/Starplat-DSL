// including the header file
#include "BTree.h"

// The different parameters for the code
#define BATCH_SIZE 1024
#define MAX_BTREE_SIZE 20000000
#define MAX_VALUE_RANGE 10000000
#define MAX_HOLES_SIZE 10000000
#define CHUNK_SIZE 10000
#define CLUSTER_HOLES_SIZE 20000


// a structure that defines the custom comparator to find the lower bound of the batch based on the keys of the element
struct LowerBoundComparator {
    bool operator()(const Node& node, const int& value) const {
        return node.key < value;
    }
};


// a structure that defines the custom comparator to find the upper bound of the batch based on the keys of the element
struct UpperBoundComparator {
    bool operator()(const int& value, const Node& node) const {
        return value < node.key;
    }
};


// a structure that defines the custom comparator to sort the batch based on the keys of the element
struct CompareByKey {
    bool operator()(const Node &a, const Node &b) const {
        return a.key < b.key;
    }
};


// function to perform parallel insertion
void btree::parallelInsert(Node *btreeArray, int **holes, int *holesCount, int *chunkElementCounter, Node *addBatch, int addCount) {
    /*
        Parameters:
            btreeArray : pointer to btree array
            holes : pointer to 2D holes array
            holesCount : pointer to an array that stores the number of elements present in the holes array for the current cluster
            chunkElementCounter : pointer to an array that stores the current total number of elements in each chunk
            addBatch : pointer to the batch to insert
            addCount : an integer denoting the number of elements in the current batch to be inserted
    */

    // variables to store the different ids of threads
    int index, tid, numt;

    // parallel execution starts
    // tid and index are private to each thread
    // numt is shared across all threads
    #pragma omp parallel private(tid, index) shared(numt)
    {
        // num represents total number of threads
        numt = omp_get_num_threads();
        // tid represents id of each thread
        tid  = omp_get_thread_num();

        // finding the range of cluster values for each process
        int startClusterValue = (MAX_BTREE_SIZE / numt) * tid;
        int endClusterValue = (MAX_BTREE_SIZE / numt) * (tid + 1) - 1;
        if (tid == numt - 1)
            endClusterValue = MAX_BTREE_SIZE - 1;

        // finding the range of elements from the insert batch on which each thread will work on (done by binary search)
        Node* start_itr = std::lower_bound(addBatch, addBatch + addCount, startClusterValue - MAX_VALUE_RANGE, LowerBoundComparator());
        Node* end_itr = std::upper_bound(addBatch, addBatch + addCount, endClusterValue - MAX_VALUE_RANGE, UpperBoundComparator());

        // if the thread has some elements to insert
        if (start_itr != end_itr)
        {
            // the start and end indices in the array
            int startIdx = start_itr - addBatch;
            int endIdx = end_itr - addBatch;

            // the thread is responsible for inserting these elements
            for (index = startIdx; index < endIdx; index++) {
                int elem = addBatch[index].key;     // we are selecting the chunk based on the key (so now the btree is sorted based on the keys)

                // random chunk calculation
                int cluster = abs((elem + MAX_VALUE_RANGE) / (4 * CHUNK_SIZE));             //cluster size = 4 * chunk size
                srand(time(0));
                int chunk = rand() % 4;
                int chunkStart = (cluster * 4 * CHUNK_SIZE) + (chunk * CHUNK_SIZE);
                int chunkEnd = chunkStart + CHUNK_SIZE - 1;

                int insertIndex = INT_MIN;

                // checking if a hole is present in the cluster
                for (int holeIndex = 0; holeIndex < holesCount[cluster]; holeIndex++) {
                    if(holes[cluster][holeIndex] != INT_MIN && holes[cluster][holeIndex] >= chunkStart && holes[cluster][holeIndex] <= chunkEnd) {
                        // acquiring the hole
                        //no need of synchronization as no other thread will work on this cluster simultaneously
                        insertIndex = holes[cluster][holeIndex];
                        holes[cluster][holeIndex] = INT_MIN;
                    }
                }

                // if no hole is found
                if (insertIndex == INT_MIN) {
                    // iterate over the four chunks in the cluster
                    while (chunkElementCounter[cluster * 4 + chunk] == CHUNK_SIZE) {
                        if (chunk == 3) chunk = 0;
                        else chunk++;
                    }

                    // calculate the index where the element should be inserted
                    insertIndex = cluster * 4 * CHUNK_SIZE + chunk * CHUNK_SIZE + chunkElementCounter[cluster * 4 + chunk];
                    chunkElementCounter[cluster * 4 + chunk]++;
                }

                // insert the element
                btreeArray[insertIndex].key = elem;
                btreeArray[insertIndex].value = addBatch[index].value;

                // std::cout << "inserted key : " << btreeArray[insertIndex].key << "  Inserted value : " << btreeArray[insertIndex].value << std::endl;
            }
        }
    }
}


// function to perform parallel searching
int* btree::parallelSearch (Node *btreeArray, int *searchBatch, int searchCount, int *searchResult) {
    /*
        Parameters:
            btreeArray : pointer to btree array
            searchBatch : pointer to the batch to search
            searchCount : an integer denoting the number of elements in the current batch to be searched
            searchResult : pointer to an array that stores the result of the searching
    */

    // variables to store the different ids of threads
    int index, tid, numt;
    
    // parallel execution starts
    // tid and index are private to each thread
    // numt is shared across all threads
    #pragma omp parallel private(tid, index) shared(numt)
    {
        // num represents total number of threads
        numt = omp_get_num_threads();
        // tid represents id of each thread
        tid  = omp_get_thread_num();

        // finding the range of cluster values for each process
        int startClusterValue = (MAX_BTREE_SIZE / numt) * tid;
        int endClusterValue = (MAX_BTREE_SIZE / numt) * (tid + 1) - 1;
        if (tid == numt - 1)
            endClusterValue = MAX_BTREE_SIZE - 1;

        // finding the range of elements from the search batch on which each thread will work on (done by binary search)
        int* start_itr = std::lower_bound(searchBatch, searchBatch + searchCount, startClusterValue - MAX_VALUE_RANGE);
        int* end_itr = std::upper_bound(searchBatch, searchBatch + searchCount, endClusterValue - MAX_VALUE_RANGE);

        // if the thread has some elements to search
        if (start_itr != end_itr)
        {
            // the start and end indices in the array
            int startIdx = start_itr - searchBatch;
            int endIdx = end_itr - searchBatch;
            
            // the thread is responsible for searching these elements
            for (index = startIdx; index < endIdx; index++) {
                int elem = searchBatch[index];
                
                // calculating the cluster where the element should be present
                int cluster = abs((elem + MAX_VALUE_RANGE) / (4 * CHUNK_SIZE));
                int clusterStart = cluster * 4 * CHUNK_SIZE;
                int clusterEnd = clusterStart + 4 * CHUNK_SIZE;

                bool searchFlag = false;

                //searching over the cluster for the element
                for (int searchIndex = clusterStart; searchIndex < clusterEnd; searchIndex++) {
                    if (btreeArray[searchIndex].key == elem) {
                        // store the result of successful searching into the searchResult array
                        searchResult[index] = btreeArray[searchIndex].value;
                        searchFlag = true;
                        break;
                    }
                }

                // store the result of unsuccessful searching into the searchResult array
                if (!searchFlag) {
                    searchResult[index] = INT_MAX;
                }
            }
        }
    }

    return searchResult;
}


// function to perform parallel deletion
void btree::parallelRemove (Node *btreeArray, int **holes, int *holesCount, int *deleteBatch, int deleteCount) {
    /*
        Parameters:
            btreeArray : pointer to btree array
            holes : pointer to 2D holes array
            holesCount : pointer to an array that stores the number of elements present in the holes array for the current cluster
            deleteBatch : pointer to the batch to delete
            deleteCount : an integer denoting the number of elements in the current batch to be deleted
    */

    // variables to store the different ids of threads
    int index, tid, numt;

    // parallel execution starts
    // tid and index are private to each thread
    // numt is shared across all threads
    #pragma omp parallel private(tid, index) shared(numt)
    {
        // num represents total number of threads
        numt = omp_get_num_threads();
        // tid represents id of each thread
        tid  = omp_get_thread_num();
        
        // finding the range of cluster values for each process
        int startClusterValue = (MAX_BTREE_SIZE / numt) * tid;
        int endClusterValue = (MAX_BTREE_SIZE / numt) * (tid + 1) - 1;
        if (tid == numt - 1)
            endClusterValue = MAX_BTREE_SIZE - 1;

        // finding the range of elements from the delete batch on which each thread will work on (done by binary search)
        int* start_itr = std::lower_bound(deleteBatch, deleteBatch + deleteCount, startClusterValue - MAX_VALUE_RANGE);
        int* end_itr = std::upper_bound(deleteBatch, deleteBatch + deleteCount, endClusterValue - MAX_VALUE_RANGE);

        // if the thread has some elements to delete
        if (start_itr != end_itr)
        {
            // the start and end indices in the array
            int startIdx = start_itr - deleteBatch;
            int endIdx = end_itr - deleteBatch;

            // the thread is responsible for deleting these elements
            for (index = startIdx; index < endIdx; index++) {
                int elem = deleteBatch[index];

                // calculating the cluster where the element should be present
                int cluster = abs((elem + MAX_VALUE_RANGE) / (4 * CHUNK_SIZE));
                int clusterStart = cluster * 4 * CHUNK_SIZE;
                int clusterEnd = clusterStart + 4 * CHUNK_SIZE;

                //searching over the cluster for the element
                for (int searchIndex = clusterStart; searchIndex < clusterEnd; searchIndex++) {
                    if (btreeArray[searchIndex].key == elem) {
                        //no need of synchronization as no other thread will work on this cluster simultaneously
                        // mark the index in the holes array
                        holes[cluster][holesCount[cluster]] = searchIndex;
                        // mark the index as deleted in the btree array
                        btreeArray[searchIndex].key = INT_MIN;
                        btreeArray[searchIndex].value = INT_MIN;
                        // increment the holes count
                        if (holesCount[cluster] < CLUSTER_HOLES_SIZE)
                            holesCount[cluster]++;
                        break;
                    }
                }
            }
        }
    }
}


// function to perform parallel search for update
int* btree::parallelUpdateSearch(Node *btreeArray, Node *updateBatch, int updateCount, int *updateResult) {
    /*
        Parameters:
            btreeArray : pointer to btree array
            updateBatch : pointer to the batch to update
            updateCount : an integer denoting the number of elements in the current batch to be updated
            updateResult : pointer to an array that stores the result of the updating
    */

    // variables to store the different ids of threads
    int index, tid, numt;
    
    // parallel execution starts
    // tid and index are private to each thread
    // numt is shared across all threads
    #pragma omp parallel private(tid, index) shared(numt)
    {
        // num represents total number of threads
        numt = omp_get_num_threads();
        // tid represents id of each thread
        tid  = omp_get_thread_num();

        // finding the range of cluster values for each process
        int startClusterValue = (MAX_BTREE_SIZE / numt) * tid;
        int endClusterValue = (MAX_BTREE_SIZE / numt) * (tid + 1) - 1;
        if (tid == numt - 1)
            endClusterValue = MAX_BTREE_SIZE - 1;

        // finding the range of elements from the search batch on which each thread will work on (done by binary search)
        Node* start_itr = std::lower_bound(updateBatch, updateBatch + updateCount, startClusterValue - MAX_VALUE_RANGE, LowerBoundComparator());
        Node* end_itr = std::upper_bound(updateBatch, updateBatch + updateCount, endClusterValue - MAX_VALUE_RANGE, UpperBoundComparator());

        // if the thread has some elements to search
        if (start_itr != end_itr)
        {
            // the start and end indices in the array
            int startIdx = start_itr - updateBatch;
            int endIdx = end_itr - updateBatch;
            
            // the thread is responsible for searching these elements
            for (index = startIdx; index < endIdx; index++) {
                int elem = updateBatch[index].key;
                int val = updateBatch[index].value;
                
                // calculating the cluster where the element should be present
                int cluster = abs((elem + MAX_VALUE_RANGE) / (4 * CHUNK_SIZE));
                int clusterStart = cluster * 4 * CHUNK_SIZE;
                int clusterEnd = clusterStart + 4 * CHUNK_SIZE;

                bool searchFlag = false;

                //searching over the cluster for the element
                for (int searchIndex = clusterStart; searchIndex < clusterEnd; searchIndex++) {
                    if (btreeArray[searchIndex].key == elem) {
                        // store the result of successful searching into the searchResult array
                        btreeArray[searchIndex].value = val;
                        updateResult[index] = val;
                        searchFlag = true;
                        break;
                    }
                }

                // store the result of unsuccessful searching into the searchResult array
                if (!searchFlag) {
                    updateResult[index] = INT_MAX;
                }
            }
        }
    }

    return updateResult;
}


// constructor
btree::btree() {
    // calling the function to allocate the memory for the different arrays
    initMemory();
}


// destructor
btree::~btree() {
    // calling the function to deallocate the memory for the different arrays
    freeMemory();
}


// function to allocate memory for the different arrays
void btree::initMemory() {
    // allocate memory for btree
    btreeArray = (Node*) malloc (MAX_BTREE_SIZE * sizeof(Node));
    std::fill(btreeArray, btreeArray + MAX_BTREE_SIZE, Node{});
    
    // allocate memory for holes
    int numClusters = ceil((double)MAX_BTREE_SIZE / (4 * CHUNK_SIZE));
    holes = (int**)malloc(numClusters * sizeof(int*));

    // allocate memory for holesCount and initilize them to 0
    holesCount = (int*)malloc(numClusters * sizeof(int));
    for (int i = 0; i < numClusters; i++) {
        holes[i] = (int*)malloc(CLUSTER_HOLES_SIZE * sizeof(int));
        holesCount[i] = 0;
    }
    
    // allocate memory for chunkElementCounter
    chunkElementCounter = (int*) malloc ((ceil(MAX_BTREE_SIZE/(double)CHUNK_SIZE)*sizeof(int)) * sizeof(int));

    // allocating memory for the batch arrays
    addBatch = (Node*) malloc(BATCH_SIZE * sizeof(Node));
    searchBatch = (int*) malloc(BATCH_SIZE * sizeof(int));
    deleteBatch = (int*) malloc(BATCH_SIZE * sizeof(int));
    updateBatch = (Node*) malloc(BATCH_SIZE * sizeof(Node));

    // initializing the count of elements in the batches
    addCount = 0;
    searchCount = 0;
    deleteCount = 0;
    updateCount = 0;
}


// function to deallocate memory for the different arrays
void btree::freeMemory() {
    free (btreeArray);
    free (holes);
    free (holesCount);
    free (chunkElementCounter);
    free (addBatch);
    free (searchBatch);
    free (deleteBatch);
    free (updateBatch);
}


// insert function for user to insert a batch
void btree::batchInsert(Node* addBatch, int addCount) { 
    /*
        Parameters:
            addBatch : pointer to the batch to add
            addCount : an integer denoting the number of elements in the current batch to be added
    */

    // sort the batch (to allow binary searching)
    if(addCount > 1)
        std::sort(addBatch, addBatch + addCount, CompareByKey());

    // parallel insert function call
    parallelInsert(btreeArray, holes, holesCount, chunkElementCounter, addBatch, addCount);
}


// search function for user to insert a batch
int* btree::batchSearch(int* searchBatch, int searchCount) {
    /*
        Parameters:
            searchBatch : pointer to the batch to search
            searchCount : an integer denoting the number of elements in the current batch to be searched
    */

    // sort the batch (to allow binary searching)
    if(searchCount > 1)
        std::sort(searchBatch, searchBatch + searchCount);

    // allocate memory for the result array
    int *searchResult = (int*)malloc((searchCount) * sizeof(int));

    // parallel search function call
    searchResult = parallelSearch (btreeArray, searchBatch, searchCount, searchResult);

    return searchResult;
}


// delete function for user to insert a batch
void btree::batchRemove(int* deleteBatch, int deleteCount) {
    /*
        Parameters:
            deleteBatch : pointer to the batch to delete
            deleteCount : an integer denoting the number of elements in the current batch to be deleted
    */

    // sort the batch (to allow binary searching)
    if(deleteCount > 1)
        std::sort(deleteBatch, deleteBatch + deleteCount);

    // parallel delete function call
    parallelRemove (btreeArray, holes, holesCount, deleteBatch, deleteCount);
}


void btree::batchUpdate(Node* updateBatch, int updateCount) {

    // sort the batch (to allow binary searching)
    if(updateCount > 1)
        std::sort(updateBatch, updateBatch + updateCount, CompareByKey());

    // allocate memory for the result array
    int *searchResult = (int*)malloc(updateCount * sizeof(int));

    // parallel search function call
    searchResult = parallelUpdateSearch (btreeArray, updateBatch, updateCount, searchResult);

    Node* updateInsertBatch = (Node*)malloc(updateCount * sizeof(Node));
    int updateInsertCount = 0;

    for (int i = 0; i < updateCount; i++) {
        if (searchResult[i] == INT_MAX)
            updateInsertBatch[updateInsertCount++] = updateBatch[i];
    }
    
    if (updateInsertCount > 0)
        parallelInsert(btreeArray, holes, holesCount, chunkElementCounter, updateInsertBatch, updateInsertCount);

    free (searchResult);
    free (updateInsertBatch);
}


// declaring a function to insert one key-value pair
void btree::insertNode(int key, int value) {
    /*
        Parameters:
            key : key of the element which is to be inserted
            value : value corresponding to the key of the element which is to be inserted
    */

    // creating the node object
    Node newNode(key, value);

    // adding the element to the batch
    addBatch[addCount++] = newNode;

    // calling the batch insert function to insert the element (the size of the batch passed is of one element)
    batchInsert(addBatch, addCount);

    // resetting the batch size count
    addCount = 0;
}


// declaring a function to search one key-value pair
int btree::search(int key) {
    /*
        Parameters:
            key : key of the element which is to be searched
    */

    // adding the key to the batch
    searchBatch[searchCount++] = key;

    // calling the batch search function to search for the key (the size of the batch passed is of one element)
    int* searchResult = batchSearch(searchBatch, searchCount);

    // resetting the batch size count
    searchCount = 0;

    // returing back the pointer to the array that stores the result of the search
    return searchResult[0];
}


// declaring a function to delete one key-value pair
void btree::remove(int key) {
    /*
        Parameters:
            key : key of the element which is to be deleted
    */

    // adding the key to the batch
    deleteBatch[deleteCount++] = key;

    // calling the batch delete function to delete for the key (the size of the batch passed is of one element)
    batchRemove(deleteBatch, deleteCount);

    // resetting the batch size count
    deleteCount = 0;
}


// declaring a function to update one key-value pair
void btree::update(int key, int newValue) {
    /*
        Parameters:
            key : key of the element which is to be updated
            newValue : new value corresponding to the key of the element which is to be updated
    */

    // creating the node object
    Node newNode(key, newValue);

    // adding the element to the batch
    updateBatch[updateCount++] = newNode;

    batchUpdate(updateBatch, updateCount);

    updateCount = 0;
}


// function to print the tree
void btree::printBTree() {
    for (int i = 0; i < MAX_BTREE_SIZE; i++) {
        if (btreeArray[i].key != INT_MIN)
            std::cout << "Key : " << btreeArray[i].key << " Value : " << btreeArray[i].value << std::endl;
    }
}