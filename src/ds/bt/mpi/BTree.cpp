// including the header file
#include "BTree.h"

// The different parameters for the code
#define BATCH_SIZE 30000
#define MAX_BTREE_SIZE 20000000
#define CHUNK_SIZE 10000
#define MAX_VALUE_RANGE 10000000
#define CLUSTER_HOLES_SIZE 20000


// constructor
btree::btree() {
    // allocating the ranks to each processes and finding the total number of processes
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &total_process);


    // creating a mpi node to braodcast the btree array
    int block_lengths[2] = {1, 1};
    MPI_Aint displacements[2];
    MPI_Datatype types[2] = {MPI_INT, MPI_INT};
    Node dummy;
    MPI_Aint base_address;
    MPI_Get_address(&dummy, &base_address);
    MPI_Get_address(&dummy.key, &displacements[0]);
    MPI_Get_address(&dummy.value, &displacements[1]);
    displacements[0] -= base_address;
    displacements[1] -= base_address;
    MPI_Type_create_struct(2, block_lengths, displacements, types, &MPI_Node);
    MPI_Type_commit(&MPI_Node);

    // calling the function to allocate the memory for the different arrays
    initMemory();
}


// destructor
btree::~btree() {
    // freeing the data type node created
    MPI_Type_free(&MPI_Node);

    // calling the function to deallocate the memory for the different arrays
    freeMemory();
}


// function to allocate memory for the different arrays
void btree::initMemory() {
    // allocate memory for the btree
    btreeArray = (Node*) malloc (MAX_BTREE_SIZE * sizeof(Node));
    // this is initialized to 0 so that we can keep the chunkElementCounter at the index 0 of each chunk
    std::fill(btreeArray, btreeArray + MAX_BTREE_SIZE, Node{});
    
    // allocate memory for the holes
    int numClusters = ceil((double)MAX_BTREE_SIZE / (4 * CHUNK_SIZE));
    holes = (int**)malloc(numClusters * sizeof(int*));
    for (int cluster = 0; cluster < numClusters; cluster++) {
        holes[cluster] = (int*)malloc(CLUSTER_HOLES_SIZE * sizeof(int));
        // INT_MAX denotes that the particular index is not occupied
        std::fill(&holes[cluster][0], &holes[cluster][0] + CLUSTER_HOLES_SIZE, INT_MAX);
        // the first index of each cluster will store the lastHoleCounter for each cluster
        holes[cluster][0] = 0;
    }
    
    // allocating memory for the batch arrays
    addBatch = (Node*)malloc(BATCH_SIZE * sizeof(Node));
    searchBatch = (int*)malloc(BATCH_SIZE * sizeof(int));
    deleteBatch = (int*)malloc(BATCH_SIZE * sizeof(int));
    updateBatch = (Node*)malloc(BATCH_SIZE * sizeof(Node));
    
    // initializing the count of elements in the batches
    addCount = 0;
    deleteCount = 0;
    searchCount = 0;
    updateCount = 0;
}


// function to deallocate memory for the different arrays
void btree::freeMemory() {
    free(btreeArray);
    free(holes);
    free (addBatch);
    free (searchBatch);
    free (deleteBatch);
    free (updateBatch);
}


// parallel insert function for user to insert a batch
void btree::batchInsert(Node *addBatch, int addCount) {
    /*
        Parameters:
            addBatch : pointer to the array containing the batch to be inserted
            addCount : an integer denoting the number of elements in the current batch to be inserted
    */

    // calculating the total number of clusters in the btree
    int numClusters = ceil((double)MAX_BTREE_SIZE / (4 * CHUNK_SIZE));
    
    // braodcast the btree among all the processes
    MPI_Bcast(btreeArray, MAX_BTREE_SIZE, MPI_Node, 0, MPI_COMM_WORLD);

    // broadcast the holes array among all the processes
    MPI_Bcast(holes, numClusters * CLUSTER_HOLES_SIZE, MPI_INT, 0, MPI_COMM_WORLD);

    // broadcast the insertion batch among all the processes
    MPI_Bcast(addBatch, addCount, MPI_Node, 0, MPI_COMM_WORLD);

    // parallel execution starts

    // finding the range of cluster values for each process
    int startClusterValue = (MAX_BTREE_SIZE / total_process) * pid;
    int endClusterValue = (MAX_BTREE_SIZE / total_process) * (pid + 1) - 1;
    if (pid == total_process - 1)
        endClusterValue = MAX_BTREE_SIZE - 1;

    // iterating over the whole batch for each process and finding which element belongs to its range to insert
    for (int index = 0; index < addCount; index++) {
        int elem = addBatch[index].key;
        
        // check if the element is within the range of the current process
        if (elem >= (startClusterValue - MAX_VALUE_RANGE) && elem <= (endClusterValue - MAX_VALUE_RANGE)) {

            // random chunk calculation
            int cluster = abs((elem + MAX_VALUE_RANGE) / (4 * CHUNK_SIZE));
            srand(time(0));
            int chunk = rand() % 4;
            int chunkStart = (cluster * 4 * CHUNK_SIZE) + (chunk * CHUNK_SIZE) + 1;
            int chunkEnd = chunkStart + CHUNK_SIZE - 1;

            int insertIndex = INT_MIN;
            
            // checking if a hole is present in the cluster which can be utilised for insertion
            for (int holeIndex = 1; holeIndex < CLUSTER_HOLES_SIZE; holeIndex++) {
                if(holes[cluster][holeIndex] != INT_MIN && holes[cluster][holeIndex] >= chunkStart && holes[cluster][holeIndex] <= chunkEnd) {
                    // acquiring the hole
                    // no need of synchronization as no other process will work on this cluster simultaneously
                    insertIndex = holes[cluster][holeIndex];
                    holes[cluster][holeIndex] = INT_MIN;
                }
                // as soon as we reach a hole index which is unoccupied we break out of the loop (in this way we do not traverse the whole array on each iteration)
                else if (holes[cluster][holeIndex] == INT_MAX)
                    break;
            }

            // if no hole is found
            int iter = 0;
            if (insertIndex == INT_MIN) {
                // we will try the four chunks in the cluster in worst case
                while (iter < 4) {
                    int chunkElementCount = btreeArray[cluster * 4 * CHUNK_SIZE + chunk * CHUNK_SIZE].value;

                    // this check satisfies means the chunk is not yet full and hence can be used for insertion
                    if (chunkElementCount < (CHUNK_SIZE - 1))
                        break;
                    if (chunk == 3) chunk = 0;
                    else chunk++;

                    iter++;
                }
                
                // calculate the index to insert from the first index in the btree cluster
                btreeArray[cluster * 4 * CHUNK_SIZE + chunk * CHUNK_SIZE].value++;
                insertIndex = cluster * 4 * CHUNK_SIZE + chunk * CHUNK_SIZE + btreeArray[cluster * 4 * CHUNK_SIZE + chunk * CHUNK_SIZE].value;
            }

            // insert the element
            btreeArray[insertIndex].key = addBatch[index].key;
            btreeArray[insertIndex].value = addBatch[index].value;
        }
    }

    // collect back all the arrays into rank 0 process (master process) one by one
    int btreePart = endClusterValue - startClusterValue + 1;
    MPI_Gather(&btreeArray[startClusterValue], btreePart, MPI_Node, btreeArray, btreePart, MPI_Node, 0, MPI_COMM_WORLD);
    MPI_Reduce(pid == 0 ? MPI_IN_PLACE : holes, holes, numClusters * CLUSTER_HOLES_SIZE, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
}


// parallel delete function for user to delete a batch
void btree::batchRemove(int *deleteBatch, int deleteCount) {
    /*
        Parameters:
            deleteBatch : pointer to the array containing the batch to be deleted
            deleteCount : an integer denoting the number of elements in the current batch to be deleted
    */

    // calculating the total number of clusters in the btree
    int numClusters = ceil((double)MAX_BTREE_SIZE / (4 * CHUNK_SIZE));

    // braodcast the btree among all the processes
    MPI_Bcast(btreeArray, MAX_BTREE_SIZE, MPI_Node, 0, MPI_COMM_WORLD);

    // broadcast the holes array among all the processes
    MPI_Bcast(holes, numClusters * CLUSTER_HOLES_SIZE, MPI_INT, 0, MPI_COMM_WORLD);
    
    // broadcast the batch among all the processes
    MPI_Bcast(deleteBatch, deleteCount, MPI_INT, 0, MPI_COMM_WORLD);
    
    // parallel execution starts

    // finding the range of cluster values for each process
    int startClusterValue = (MAX_BTREE_SIZE / total_process) * pid;
    int endClusterValue = (MAX_BTREE_SIZE / total_process) * (pid + 1) - 1;
    if (pid == total_process - 1)
        endClusterValue = MAX_BTREE_SIZE - 1;

    // iterating over the whole batch for each process and finding which element belongs to its range to delete
    for (int index = 0; index < deleteCount; index++) {
        int elem = deleteBatch[index];
        
        // check if the element is within the range of the current process
        if (elem >= (startClusterValue - MAX_VALUE_RANGE) && elem <= (endClusterValue - MAX_VALUE_RANGE)) {
            int cluster = abs((elem + MAX_VALUE_RANGE) / (4 * CHUNK_SIZE));
            int clusterStart = cluster * 4 * CHUNK_SIZE;
            int clusterEnd = clusterStart + 4 * CHUNK_SIZE;

            // search for the element in the whole cluster
            for (int searchIndex = clusterStart; searchIndex < clusterEnd; searchIndex++) {
                if (btreeArray[searchIndex].key == elem) {
                    //no need of synchronization as no other process will work on this cluster simultaneously
                    // calculate the next hole index to be utilised from the stored value in the 0th index
                    int nextHoleToInsert = holes[cluster][0];
                    if (nextHoleToInsert < CLUSTER_HOLES_SIZE)
                        holes[cluster][nextHoleToInsert] = searchIndex;
                    // increment the 0th stored value
                    if (holes[cluster][0] < CLUSTER_HOLES_SIZE - 1)
                        holes[cluster][0]++;
                    //mark the btree index as deleted
                    btreeArray[searchIndex].key = INT_MIN;
                    btreeArray[searchIndex].value = INT_MIN;
                    break;
                }
            }
        }
    }
    
    // collect back all the arrays into rank 0 process (master process) one by one
    int btreePart = endClusterValue - startClusterValue + 1;
    MPI_Gather(&btreeArray[startClusterValue], btreePart, MPI_Node, btreeArray, btreePart, MPI_Node, 0, MPI_COMM_WORLD);
    MPI_Reduce(pid == 0 ? MPI_IN_PLACE : holes, holes, numClusters * CLUSTER_HOLES_SIZE, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
}


// parallel search function for user to search a batch
int* btree::batchSearch(int *searchBatch, int searchCount) {
    /*
        Parameters:
            searchBatch : pointer to the array containing the batch to be searched
            searchCount : an integer denoting the number of elements in the current batch to be searched
    */

    // braodcast the btree among all the processes first
    MPI_Bcast(btreeArray, MAX_BTREE_SIZE, MPI_Node, 0, MPI_COMM_WORLD);
                    
    // broadcast the batch now among all the processes
    MPI_Bcast(searchBatch, searchCount, MPI_INT, 0, MPI_COMM_WORLD);

    // parallel execution starts

    // finding the range of cluster values for each process
    int startClusterValue = (MAX_BTREE_SIZE / total_process) * pid;
    int endClusterValue = (MAX_BTREE_SIZE / total_process) * (pid + 1) - 1;
    if (pid == total_process - 1)
        endClusterValue = MAX_BTREE_SIZE - 1;

    // iterating over the whole batch for each process and finding which element belongs to its range to search
    for (int index = 0; index < searchCount; index++) {
        int elem = searchBatch[index];

        // check if the element is within the range of the current process
        if (elem >= (startClusterValue - MAX_VALUE_RANGE) && elem <= (endClusterValue - MAX_VALUE_RANGE)) {
            int cluster = abs((elem + MAX_VALUE_RANGE) / (4 * CHUNK_SIZE));
            int clusterStart = cluster * 4 * CHUNK_SIZE;
            int clusterEnd = clusterStart + 4 * CHUNK_SIZE;

            bool searchFlag = false;
            
            // iterate over the cluster to find the element
            for (int searchIndex = clusterStart; searchIndex < clusterEnd; searchIndex++) {
                if (btreeArray[searchIndex].key == elem) {
                    // store the result of successful searching into the search batch itself
                    searchBatch[index] = btreeArray[searchIndex].value;
                    searchFlag = true;
                    break;
                }
            }

            // store the result of unsuccessful searching into the search batch itself
            if (!searchFlag)
            searchBatch[index] = INT_MIN;
        }
    }

    // collect back the result array into rank 0 process (master process)
    MPI_Reduce(pid == 0 ? MPI_IN_PLACE : searchBatch, searchBatch, searchCount, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    if (pid == 0)
        return searchBatch;
}


// parallel update function for user to search a batch
void btree::batchUpdate(Node *updateBatch, int updateCount) {
    /*
        Parameters:
            updateBatch : pointer to the array containing the batch to be updated
            updateCount : an integer denoting the number of elements in the current batch to be updated
    */

    // calculating the total number of clusters in the btree
    int numClusters = ceil((double)MAX_BTREE_SIZE / (4 * CHUNK_SIZE));

    // braodcast the btree among all the processes first
    MPI_Bcast(btreeArray, MAX_BTREE_SIZE, MPI_Node, 0, MPI_COMM_WORLD);

    // broadcast the holes array among all the processes
    MPI_Bcast(holes, numClusters * CLUSTER_HOLES_SIZE, MPI_INT, 0, MPI_COMM_WORLD);
                    
    // broadcast the batch now among all the processes
    MPI_Bcast(updateBatch, updateCount, MPI_Node, 0, MPI_COMM_WORLD);

    // parallel execution starts

    // finding the range of cluster values for each process
    int startClusterValue = (MAX_BTREE_SIZE / total_process) * pid;
    int endClusterValue = (MAX_BTREE_SIZE / total_process) * (pid + 1) - 1;
    if (pid == total_process - 1)
        endClusterValue = MAX_BTREE_SIZE - 1;

    // iterating over the whole batch for each process and finding which element belongs to its range to search
    for (int index = 0; index < updateCount; index++) {
        int elem = updateBatch[index].key;
        int val = updateBatch[index].value;

        // check if the element is within the range of the current process
        if (elem >= (startClusterValue - MAX_VALUE_RANGE) && elem <= (endClusterValue - MAX_VALUE_RANGE)) {
            int cluster = abs((elem + MAX_VALUE_RANGE) / (4 * CHUNK_SIZE));
            int clusterStart = cluster * 4 * CHUNK_SIZE;
            int clusterEnd = clusterStart + 4 * CHUNK_SIZE;

            bool searchFlag = false;
            
            // iterate over the cluster to find the element
            for (int searchIndex = clusterStart; searchIndex < clusterEnd; searchIndex++) {
                if (btreeArray[searchIndex].key == elem) {
                    // store the result of successful searching into the search batch itself
                    btreeArray[searchIndex].value = val;
                    searchFlag = true;
                    break;
                }
            }

            // if search was unsuccessful then we need to insert the element directly into the tree
            if (!searchFlag) {
                srand(time(0));
                int chunk = rand() % 4;
                int chunkStart = (cluster * 4 * CHUNK_SIZE) + (chunk * CHUNK_SIZE) + 1;
                int chunkEnd = chunkStart + CHUNK_SIZE - 1;

                int insertIndex = INT_MIN;
                
                // checking if a hole is present in the cluster which can be utilised for insertion
                for (int holeIndex = 1; holeIndex < CLUSTER_HOLES_SIZE; holeIndex++) {
                    if(holes[cluster][holeIndex] != INT_MIN && holes[cluster][holeIndex] >= chunkStart && holes[cluster][holeIndex] <= chunkEnd) {
                        // acquiring the hole
                        // no need of synchronization as no other process will work on this cluster simultaneously
                        insertIndex = holes[cluster][holeIndex];
                        holes[cluster][holeIndex] = INT_MIN;
                    }
                    // as soon as we reach a hole index which is unoccupied we break out of the loop (in this way we do not traverse the whole array on each iteration)
                    else if (holes[cluster][holeIndex] == INT_MAX)
                        break;
                }

                // if no hole is found
                int iter = 0;
                if (insertIndex == INT_MIN) {
                    // we will try the four chunks in the cluster in worst case
                    while (iter < 4) {
                        int chunkElementCount = btreeArray[cluster * 4 * CHUNK_SIZE + chunk * CHUNK_SIZE].value;

                        // this check satisfies means the chunk is not yet full and hence can be used for insertion
                        if (chunkElementCount < (CHUNK_SIZE - 1))
                            break;
                        if (chunk == 3) chunk = 0;
                        else chunk++;

                        iter++;
                    }
                    
                    // calculate the index to insert from the first index in the btree cluster
                    btreeArray[cluster * 4 * CHUNK_SIZE + chunk * CHUNK_SIZE].value++;
                    insertIndex = cluster * 4 * CHUNK_SIZE + chunk * CHUNK_SIZE + btreeArray[cluster * 4 * CHUNK_SIZE + chunk * CHUNK_SIZE].value;
                }

                // insert the element
                btreeArray[insertIndex].key = elem;
                btreeArray[insertIndex].value = val;
            }
        }
    }

    // collect back all the arrays into rank 0 process (master process) one by one
    int btreePart = endClusterValue - startClusterValue + 1;
    MPI_Gather(&btreeArray[startClusterValue], btreePart, MPI_Node, btreeArray, btreePart, MPI_Node, 0, MPI_COMM_WORLD);
    MPI_Reduce(pid == 0 ? MPI_IN_PLACE : holes, holes, numClusters * CLUSTER_HOLES_SIZE, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
}


// parallel insert function for user to insert one element
void btree::insertNode(int key, int value) {
    /*
        Parameters:
            key : key of the element which is to be inserted
            value : value corresponding to the key of the element which is to be inserted
    */

    // calculating the total number of clusters in the btree
    int numClusters = ceil((double)MAX_BTREE_SIZE / (4 * CHUNK_SIZE));

    if (pid == 0)
        addBatch[addCount++] = Node(key, value);
    
    // braodcast the btree among all the processes
    MPI_Bcast(btreeArray, MAX_BTREE_SIZE, MPI_Node, 0, MPI_COMM_WORLD);

    // broadcast the holes array among all the processes
    MPI_Bcast(holes, numClusters * CLUSTER_HOLES_SIZE, MPI_INT, 0, MPI_COMM_WORLD);

    // broadcast the insertion batch among all the processes
    MPI_Bcast(addBatch, addCount, MPI_Node, 0, MPI_COMM_WORLD);

    // parallel execution starts

    // finding the range of cluster values for each process
    int startClusterValue = (MAX_BTREE_SIZE / total_process) * pid;
    int endClusterValue = (MAX_BTREE_SIZE / total_process) * (pid + 1) - 1;
    if (pid == total_process - 1)
        endClusterValue = MAX_BTREE_SIZE - 1;

    // iterating over the whole batch for each process and finding which element belongs to its range to insert
    for (int index = 0; index < addCount; index++) {
        int elem = addBatch[index].key;
        
        // check if the element is within the range of the current process
        if (elem >= (startClusterValue - MAX_VALUE_RANGE) && elem <= (endClusterValue - MAX_VALUE_RANGE)) {

            // random chunk calculation
            int cluster = abs((elem + MAX_VALUE_RANGE) / (4 * CHUNK_SIZE));
            srand(time(0));
            int chunk = rand() % 4;
            int chunkStart = (cluster * 4 * CHUNK_SIZE) + (chunk * CHUNK_SIZE) + 1;
            int chunkEnd = chunkStart + CHUNK_SIZE - 1;

            int insertIndex = INT_MIN;
            
            // checking if a hole is present in the cluster which can be utilised for insertion
            for (int holeIndex = 1; holeIndex < CLUSTER_HOLES_SIZE; holeIndex++) {
                if(holes[cluster][holeIndex] != INT_MIN && holes[cluster][holeIndex] >= chunkStart && holes[cluster][holeIndex] <= chunkEnd) {
                    // acquiring the hole
                    // no need of synchronization as no other process will work on this cluster simultaneously
                    insertIndex = holes[cluster][holeIndex];
                    holes[cluster][holeIndex] = INT_MIN;
                }
                // as soon as we reach a hole index which is unoccupied we break out of the loop (in this way we do not traverse the whole array on each iteration)
                else if (holes[cluster][holeIndex] == INT_MAX)
                    break;
            }

            // if no hole is found
            int iter = 0;
            if (insertIndex == INT_MIN) {
                // we will try the four chunks in the cluster in worst case
                while (iter < 4) {
                    int chunkElementCount = btreeArray[cluster * 4 * CHUNK_SIZE + chunk * CHUNK_SIZE].value;

                    // this check satisfies means the chunk is not yet full and hence can be used for insertion
                    if (chunkElementCount < (CHUNK_SIZE - 1))
                        break;
                    if (chunk == 3) chunk = 0;
                    else chunk++;

                    iter++;
                }
                
                // calculate the index to insert from the first index in the btree cluster
                btreeArray[cluster * 4 * CHUNK_SIZE + chunk * CHUNK_SIZE].value++;
                insertIndex = cluster * 4 * CHUNK_SIZE + chunk * CHUNK_SIZE + btreeArray[cluster * 4 * CHUNK_SIZE + chunk * CHUNK_SIZE].value;
            }

            // insert the element
            btreeArray[insertIndex].key = addBatch[index].key;
            btreeArray[insertIndex].value = addBatch[index].value;
        }
    }

    // collect back all the arrays into rank 0 process (master process) one by one
    int btreePart = endClusterValue - startClusterValue + 1;
    MPI_Gather(&btreeArray[startClusterValue], btreePart, MPI_Node, btreeArray, btreePart, MPI_Node, 0, MPI_COMM_WORLD);
    MPI_Reduce(pid == 0 ? MPI_IN_PLACE : holes, holes, numClusters * CLUSTER_HOLES_SIZE, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    //resetting the batch element count
    if (pid == 0)
        addCount = 0;
}


// parallel delete function for user to delete one element
void btree::remove(int key) {
    /*
        Parameters:
            key : key of the element which is to be deleted
    */

    // calculating the total number of clusters in the btree
    int numClusters = ceil((double)MAX_BTREE_SIZE / (4 * CHUNK_SIZE));

    if (pid == 0)
        deleteBatch[deleteCount++] = key;

    // braodcast the btree among all the processes
    MPI_Bcast(btreeArray, MAX_BTREE_SIZE, MPI_Node, 0, MPI_COMM_WORLD);

    // broadcast the holes array among all the processes
    MPI_Bcast(holes, numClusters * CLUSTER_HOLES_SIZE, MPI_INT, 0, MPI_COMM_WORLD);
    
    // broadcast the batch among all the processes
    MPI_Bcast(deleteBatch, deleteCount, MPI_INT, 0, MPI_COMM_WORLD);
    
    // parallel execution starts

    // finding the range of cluster values for each process
    int startClusterValue = (MAX_BTREE_SIZE / total_process) * pid;
    int endClusterValue = (MAX_BTREE_SIZE / total_process) * (pid + 1) - 1;
    if (pid == total_process - 1)
        endClusterValue = MAX_BTREE_SIZE - 1;

    // iterating over the whole batch for each process and finding which element belongs to its range to delete
    for (int index = 0; index < deleteCount; index++) {
        int elem = deleteBatch[index];
        
        // check if the element is within the range of the current process
        if (elem >= (startClusterValue - MAX_VALUE_RANGE) && elem <= (endClusterValue - MAX_VALUE_RANGE)) {
            int cluster = abs((elem + MAX_VALUE_RANGE) / (4 * CHUNK_SIZE));
            int clusterStart = cluster * 4 * CHUNK_SIZE;
            int clusterEnd = clusterStart + 4 * CHUNK_SIZE;

            // search for the element in the whole cluster
            for (int searchIndex = clusterStart; searchIndex < clusterEnd; searchIndex++) {
                if (btreeArray[searchIndex].key == elem) {
                    //no need of synchronization as no other process will work on this cluster simultaneously
                    // calculate the next hole index to be utilised from the stored value in the 0th index
                    int nextHoleToInsert = holes[cluster][0];
                    if (nextHoleToInsert < CLUSTER_HOLES_SIZE)
                        holes[cluster][nextHoleToInsert] = searchIndex;
                    // increment the 0th stored value
                    if (holes[cluster][0] < CLUSTER_HOLES_SIZE - 1)
                        holes[cluster][0]++;
                    //mark the btree index as deleted
                    btreeArray[searchIndex].key = INT_MIN;
                    btreeArray[searchIndex].value = INT_MIN;
                    break;
                }
            }
        }
    }
    
    // collect back all the arrays into rank 0 process (master process) one by one
    int btreePart = endClusterValue - startClusterValue + 1;
    MPI_Gather(&btreeArray[startClusterValue], btreePart, MPI_Node, btreeArray, btreePart, MPI_Node, 0, MPI_COMM_WORLD);
    MPI_Reduce(pid == 0 ? MPI_IN_PLACE : holes, holes, numClusters * CLUSTER_HOLES_SIZE, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    //resetting the batch element count
    if (pid == 0)
        deleteCount = 0;
}


// parallel search function for user to search one element
int btree::search(int key) {
    /*
        Parameters:
            key : key of the element which is to be searched
    */

    if (pid == 0)
        searchBatch[searchCount++] = key;

    // braodcast the btree among all the processes first
    MPI_Bcast(btreeArray, MAX_BTREE_SIZE, MPI_Node, 0, MPI_COMM_WORLD);
                    
    // broadcast the batch now among all the processes
    MPI_Bcast(searchBatch, searchCount, MPI_INT, 0, MPI_COMM_WORLD);

    // parallel execution starts

    // finding the range of cluster values for each process
    int startClusterValue = (MAX_BTREE_SIZE / total_process) * pid;
    int endClusterValue = (MAX_BTREE_SIZE / total_process) * (pid + 1) - 1;
    if (pid == total_process - 1)
        endClusterValue = MAX_BTREE_SIZE - 1;

    // iterating over the whole batch for each process and finding which element belongs to its range to search
    for (int index = 0; index < searchCount; index++) {
        int elem = searchBatch[index];

        // check if the element is within the range of the current process
        if (elem >= (startClusterValue - MAX_VALUE_RANGE) && elem <= (endClusterValue - MAX_VALUE_RANGE)) {
            int cluster = abs((elem + MAX_VALUE_RANGE) / (4 * CHUNK_SIZE));
            int clusterStart = cluster * 4 * CHUNK_SIZE;
            int clusterEnd = clusterStart + 4 * CHUNK_SIZE;

            bool searchFlag = false;
            
            // iterate over the cluster to find the element
            for (int searchIndex = clusterStart; searchIndex < clusterEnd; searchIndex++) {
                if (btreeArray[searchIndex].key == elem) {
                    // store the result of successful searching into the search batch itself
                    searchBatch[index] = btreeArray[searchIndex].value;
                    searchFlag = true;
                    break;
                }
            }

            // store the result of unsuccessful searching into the search batch itself
            if (!searchFlag)
            searchBatch[index] = INT_MIN;
        }
    }

    // collect back the result array into rank 0 process (master process)
    MPI_Reduce(pid == 0 ? MPI_IN_PLACE : searchBatch, searchBatch, searchCount, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    //resetting the batch element count
    if (pid == 0) {
        searchCount = 0;
        return searchBatch[searchCount];
    }
}


// parallel update function for user to update one element
void btree::update(int key, int newValue) {
    /*
        Parameters:
            key : key of the element which is to be updated
            newValue : new value corresponding to the key of the element which is to be updated
    */
    
    // calculating the total number of clusters in the btree
    int numClusters = ceil((double)MAX_BTREE_SIZE / (4 * CHUNK_SIZE));

    if (pid == 0)
        updateBatch[updateCount++] = Node(key, newValue);

    // braodcast the btree among all the processes first
    MPI_Bcast(btreeArray, MAX_BTREE_SIZE, MPI_Node, 0, MPI_COMM_WORLD);

    // broadcast the holes array among all the processes
    MPI_Bcast(holes, numClusters * CLUSTER_HOLES_SIZE, MPI_INT, 0, MPI_COMM_WORLD);
                    
    // broadcast the batch now among all the processes
    MPI_Bcast(updateBatch, updateCount, MPI_Node, 0, MPI_COMM_WORLD);

    // parallel execution starts

    // finding the range of cluster values for each process
    int startClusterValue = (MAX_BTREE_SIZE / total_process) * pid;
    int endClusterValue = (MAX_BTREE_SIZE / total_process) * (pid + 1) - 1;
    if (pid == total_process - 1)
        endClusterValue = MAX_BTREE_SIZE - 1;

    // iterating over the whole batch for each process and finding which element belongs to its range to search
    for (int index = 0; index < updateCount; index++) {
        int elem = updateBatch[index].key;
        int val = updateBatch[index].value;

        // check if the element is within the range of the current process
        if (elem >= (startClusterValue - MAX_VALUE_RANGE) && elem <= (endClusterValue - MAX_VALUE_RANGE)) {
            int cluster = abs((elem + MAX_VALUE_RANGE) / (4 * CHUNK_SIZE));
            int clusterStart = cluster * 4 * CHUNK_SIZE;
            int clusterEnd = clusterStart + 4 * CHUNK_SIZE;

            bool searchFlag = false;
            
            // iterate over the cluster to find the element
            for (int searchIndex = clusterStart; searchIndex < clusterEnd; searchIndex++) {
                if (btreeArray[searchIndex].key == elem) {
                    // store the result of successful searching into the search batch itself
                    btreeArray[searchIndex].value = val;
                    searchFlag = true;
                    break;
                }
            }

            // if search was unsuccessful then we need to insert the element directly into the tree
            if (!searchFlag) {
                srand(time(0));
                int chunk = rand() % 4;
                int chunkStart = (cluster * 4 * CHUNK_SIZE) + (chunk * CHUNK_SIZE) + 1;
                int chunkEnd = chunkStart + CHUNK_SIZE - 1;

                int insertIndex = INT_MIN;
                
                // checking if a hole is present in the cluster which can be utilised for insertion
                for (int holeIndex = 1; holeIndex < CLUSTER_HOLES_SIZE; holeIndex++) {
                    if(holes[cluster][holeIndex] != INT_MIN && holes[cluster][holeIndex] >= chunkStart && holes[cluster][holeIndex] <= chunkEnd) {
                        // acquiring the hole
                        // no need of synchronization as no other process will work on this cluster simultaneously
                        insertIndex = holes[cluster][holeIndex];
                        holes[cluster][holeIndex] = INT_MIN;
                    }
                    // as soon as we reach a hole index which is unoccupied we break out of the loop (in this way we do not traverse the whole array on each iteration)
                    else if (holes[cluster][holeIndex] == INT_MAX)
                        break;
                }

                // if no hole is found
                int iter = 0;
                if (insertIndex == INT_MIN) {
                    // we will try the four chunks in the cluster in worst case
                    while (iter < 4) {
                        int chunkElementCount = btreeArray[cluster * 4 * CHUNK_SIZE + chunk * CHUNK_SIZE].value;

                        // this check satisfies means the chunk is not yet full and hence can be used for insertion
                        if (chunkElementCount < (CHUNK_SIZE - 1))
                            break;
                        if (chunk == 3) chunk = 0;
                        else chunk++;

                        iter++;
                    }
                    
                    // calculate the index to insert from the first index in the btree cluster
                    btreeArray[cluster * 4 * CHUNK_SIZE + chunk * CHUNK_SIZE].value++;
                    insertIndex = cluster * 4 * CHUNK_SIZE + chunk * CHUNK_SIZE + btreeArray[cluster * 4 * CHUNK_SIZE + chunk * CHUNK_SIZE].value;
                }

                // insert the element
                btreeArray[insertIndex].key = elem;
                btreeArray[insertIndex].value = val;
            }
        }
    }

    // collect back all the arrays into rank 0 process (master process) one by one
    int btreePart = endClusterValue - startClusterValue + 1;
    MPI_Gather(&btreeArray[startClusterValue], btreePart, MPI_Node, btreeArray, btreePart, MPI_Node, 0, MPI_COMM_WORLD);
    MPI_Reduce(pid == 0 ? MPI_IN_PLACE : holes, holes, numClusters * CLUSTER_HOLES_SIZE, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    //resetting the batch element count
    if (pid == 0)
        updateCount = 0;
}


// function to print the tree
void btree::printBTree() {
    for (int i = 0; i < MAX_BTREE_SIZE; i++) {
        std::cout << "Key : " << btreeArray[i].key << " Value : " << btreeArray[i].value << std::endl;
    }
}