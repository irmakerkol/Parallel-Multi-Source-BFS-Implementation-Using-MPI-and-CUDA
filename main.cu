
#include <mpi.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <iomanip> 

using namespace std;

__global__ void BFSKernal(const int *d_row_offsets, const int *d_col_indices, const int num_vertices, volatile int *d_distances, int current_frontier_level, volatile bool *d_updated)
{
   int tid = blockDim.x * blockIdx.x + threadIdx.x;
   if (tid < num_vertices) 
   {
       if (d_distances[tid] == current_frontier_level) 
       {
           // Explore neighbors
           int row_start = d_row_offsets[tid];
           int row_end   = d_row_offsets[tid+1];
           for (int offset = row_start; offset < row_end; offset++) 
           {
              int neighbor = d_col_indices[offset];
              // If neighbor not visited
              if (d_distances[neighbor] == -1) 
              {
                  d_distances[neighbor] = current_frontier_level + 1;
                  *d_updated = true;  
              }
           }
       }
   }
}

void GPUMultiSourceBFS( int n, const vector<int> &sources, int *d_row_offsets, int *d_col_indices, int *d_distances)
{
   cudaMemset(d_distances, -1, n * sizeof(int));
   // Host buffer for sources
   vector<int> h_tmp(n, -1);
   // Mark distances for each source as 0
   for (size_t i = 0; i < sources.size(); i++) 
   {
       int s = sources[i];
       if (s >= 0 && s < n) 
           h_tmp[s] = 0;
   }
   // Copy these initial distances to device
   cudaMemcpy(d_distances, h_tmp.data(), n*sizeof(int), cudaMemcpyHostToDevice);
   //keep iterating until no change
   bool h_updated = true;
   bool *d_updated;
   cudaMalloc((void**)&d_updated, sizeof(bool));
   int level = 0;
   const int threads = 256;
   const int blocks  = (n + threads - 1) / threads;
   while(h_updated) 
   {
       h_updated = false;
       cudaMemcpy(d_updated, &h_updated, sizeof(bool), cudaMemcpyHostToDevice);

       BFSKernal<<<blocks, threads>>>(d_row_offsets, d_col_indices, n, d_distances, level, d_updated);
       cudaDeviceSynchronize();

       cudaMemcpy(&h_updated, d_updated, sizeof(bool), cudaMemcpyDeviceToHost);
       level++;
   }
   cudaFree(d_updated);
}

long long ComputeFofU(int n, int *d_distances)
{
   // Copy distances back to host
   vector<int> h_distances(n);
   cudaMemcpy(h_distances.data(), d_distances, n*sizeof(int), cudaMemcpyDeviceToHost);

   long long sum_dist = 0;
   for (int i = 0; i < n; i++) 
   {
       if (h_distances[i] < 0) 
           continue;
       sum_dist += h_distances[i];
    }
   return sum_dist;
}


void LoadGraphBin(const char* filename,int &n, long long &m,vector<int> &row_offsets,vector<int> &col_indices)
{
   FILE* f = fopen(filename, "rb");
   if(!f) 
   {
       fprintf(stderr, "Could not open graph file %s\n", filename);
       exit(EXIT_FAILURE);
    }

   // Read number of vertices (4 bytes)
   fread(&n, sizeof(int), 1, f);
   // Read number of edges (8 bytes)
   fread(&m, sizeof(long long), 1, f);

   vector<vector<int>> adj(n);

   for(long long i = 0; i < m; i++)
   {
       int u, v;
       fread(&u, sizeof(int), 1, f);
       fread(&v, sizeof(int), 1, f);
       // Undirected => store both ways
       adj[u].push_back(v);
       adj[v].push_back(u);
   }
   fclose(f);

   row_offsets.resize(n+1, 0);
   for(int i = 0; i < n; i++)
       row_offsets[i+1] = row_offsets[i] + (int)adj[i].size();
   

   col_indices.resize(row_offsets[n]);
   for(int i = 0; i < n; i++)
   {
       int start = row_offsets[i];
       copy(adj[i].begin(), adj[i].end(), col_indices.begin() + start);
   }
}



void LoadQueryBin(const char* filename,int &K,vector<vector<int>> &queries)
{
   FILE* f = fopen(filename, "rb");
   if(!f) 
   {
       fprintf(stderr, "Could not open query file %s\n", filename);
       exit(EXIT_FAILURE);
   }

   unsigned char K_char;
   fread(&K_char, sizeof(unsigned char), 1, f);
   K = K_char;  // up to 64

   queries.resize(K);
   for(int i = 0; i < K; i++)
   {
       unsigned char size_char;
       fread(&size_char, sizeof(unsigned char), 1, f);
       int set_size = size_char; // up to 128

       queries[i].resize(set_size);
       for(int j = 0; j < set_size; j++)
       {
           int v;
           fread(&v, sizeof(int), 1, f);
           queries[i][j] = v;
       }
   }

   fclose(f);
}

struct Pair 
{ 
   int q; 
   long long fv; 
};

// Function to create MPI datatype for Pair
MPI_Datatype CreateMPIPairType() 
{
   MPI_Datatype MPI_PAIR;
   int block_lengths[2] = {1, 1};
   MPI_Aint displacements[2];
   Pair temp_pair;

   MPI_Aint base_address;
   MPI_Get_address(&temp_pair, &base_address);
   MPI_Get_address(&temp_pair.q, &displacements[0]);
   MPI_Get_address(&temp_pair.fv, &displacements[1]);

   displacements[0] = displacements[0] - base_address;
   displacements[1] = displacements[1] - base_address;

   MPI_Datatype types[2] = {MPI_INT, MPI_LONG_LONG};

   MPI_Type_create_struct(2, block_lengths, displacements, types, &MPI_PAIR);
   MPI_Type_commit(&MPI_PAIR);
   return MPI_PAIR;
}

int main(int argc, char* argv[])
{
   MPI_Init(&argc, &argv);

   int world_size, world_rank;
   MPI_Comm_size(MPI_COMM_WORLD, &world_size);
   MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

   // Parse arguments
   if (argc < 5) {
       if(world_rank == 0) {
           cerr << "Usage: mpirun -np <ranks> "
                   << argv[0] << " -g <graph.bin> -q <query.bin> -gn <numGPU>"
                   << endl;
       }
       MPI_Finalize();
       return -1;
   }

   string graphFile, queryFile;
   int numGPU = 1; // default
   for(int i = 1; i < argc; i++){
       if(strcmp(argv[i], "-g") == 0){
           graphFile = argv[++i];
       } else if(strcmp(argv[i], "-q") == 0){
           queryFile = argv[++i];
       } else if(strcmp(argv[i], "-gn") == 0){
           numGPU = atoi(argv[++i]);
       }
   }

   // If each rank is using one GPU, pick device = world_rank % numGPU
   int device_id = world_rank % numGPU;
   cudaSetDevice(device_id);

   //Rank 0 loads the graph; broadcast to others
   int n = 0; 
   long long m = 0; 
   vector<int> row_offsets, col_indices;

   auto start_preprocessing = chrono::high_resolution_clock::now();

   if (world_rank == 0) {
       LoadGraphBin(graphFile.c_str(), n, m, row_offsets, col_indices);
   }

   // Broadcast n, m
   MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&m, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

   // Resize local vectors on other ranks, then broadcast
   if (world_rank != 0) {
       row_offsets.resize(n+1);
       // col_indices will be resized after receiving row_offsets
   }
   MPI_Bcast(row_offsets.data(), n+1, MPI_INT, 0, MPI_COMM_WORLD);

   if (world_rank != 0) {
       col_indices.resize(row_offsets[n]);
   }
   MPI_Bcast(col_indices.data(), row_offsets[n], MPI_INT, 0, MPI_COMM_WORLD);

   //Rank 0 loads the queries; broadcast
   int K = 0;
   vector<vector<int>> queries;
   if (world_rank == 0) {
       LoadQueryBin(queryFile.c_str(), K, queries);
   }
   // broadcast K
   MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);
   // broadcast queries
   if (world_rank != 0) {
       queries.resize(K);
   }
   for(int k = 0; k < K; k++){
       if(world_rank == 0){
           int size_k = (int)queries[k].size();
           MPI_Bcast(&size_k, 1, MPI_INT, 0, MPI_COMM_WORLD);
           MPI_Bcast(queries[k].data(), size_k, MPI_INT, 0, MPI_COMM_WORLD);
       } else {
           int size_k;
           MPI_Bcast(&size_k, 1, MPI_INT, 0, MPI_COMM_WORLD);
           queries[k].resize(size_k);
           MPI_Bcast(queries[k].data(), size_k, MPI_INT, 0, MPI_COMM_WORLD);
       }
   }

   // Allocate CSR data on GPU
   int *d_row_offsets = nullptr;
   int *d_col_indices = nullptr;

   cudaMalloc(&d_row_offsets, (n+1)*sizeof(int));
   cudaMalloc(&d_col_indices, row_offsets[n]*sizeof(int));

   // Copy CSR to device
   cudaMemcpy(d_row_offsets, row_offsets.data(), (n+1)*sizeof(int), cudaMemcpyHostToDevice);
   cudaMemcpy(d_col_indices, col_indices.data(), row_offsets[n]*sizeof(int), cudaMemcpyHostToDevice);

   // Allocate a distances array on device
   int *d_distances = nullptr;
   cudaMalloc(&d_distances, n*sizeof(int));

   auto end_preprocessing = chrono::high_resolution_clock::now();
   double preprocessing_time = chrono::duration<double>(end_preprocessing - start_preprocessing).count();

   // Computation
   auto start_computation = chrono::high_resolution_clock::now();

   // local_k_to_global maps from local index to the real query index
   vector<int> local_k_to_global;
   for(int kidx = world_rank; kidx < K; kidx += world_size) {
       local_k_to_global.push_back(kidx);
   }

   // For each assigned query, run BFS, compute F(U_k)
   vector<long long> local_F_values(local_k_to_global.size(), 0LL);

   for(size_t i = 0; i < local_k_to_global.size(); i++){
       int qindex = local_k_to_global[i];
       vector<int> &sources = queries[qindex];

       // Multi-source BFS
       GPUMultiSourceBFS(n, sources, d_row_offsets, d_col_indices, d_distances);

       // Compute F(U_k)
       long long F_val = ComputeFofU(n, d_distances);
       local_F_values[i] = F_val;
   }

   // Gather all F-values on rank 0
   vector<long long> all_F_values(K, -1);

   // Initialize the custom MPI datatype for Pair
   MPI_Datatype MPI_PAIR = CreateMPIPairType();

   // Prepare local_pairs as (query_index, F_value)
   vector<Pair> local_pairs(local_k_to_global.size());
   for(size_t i = 0; i < local_k_to_global.size(); i++){
       local_pairs[i].q  = local_k_to_global[i];
       local_pairs[i].fv = local_F_values[i];
   }

   // Gather counts as number of Pairs
   int local_pairs_count = (int)local_pairs.size();
   vector<int> all_counts(world_size);
   MPI_Gather(&local_pairs_count, 1, MPI_INT, 
           all_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

   // Calculate displacements
   int total_pairs = 0;
   vector<int> displs(world_size, 0);
   if(world_rank == 0) {
       for(int i = 0; i < world_size; i++){
           total_pairs += all_counts[i];
       }
       for(int i = 1; i < world_size; i++){
           displs[i] = displs[i-1] + all_counts[i-1];
       }
   }

   // Allocate global_pairs only on rank 0
   vector<Pair> global_pairs;
   if(world_rank == 0){
       global_pairs.resize(total_pairs);
   }

   // Perform MPI_Gatherv using the custom MPI_PAIR datatype
   MPI_Gatherv(local_pairs.data(), local_pairs_count, MPI_PAIR,
               global_pairs.data(), (world_rank == 0 ? all_counts.data() : nullptr),
               (world_rank == 0 ? displs.data() : nullptr),
               MPI_PAIR, 0, MPI_COMM_WORLD);

   // Clean up the custom datatype after use
   MPI_Type_free(&MPI_PAIR);

   if (world_rank == 0) {
       // Place the F-values in the correct order
       for(auto &p : global_pairs) {
           all_F_values[p.q] = p.fv;
       }
   }

   // Now we have all F(U_k) in rank 0
   // Find min
   long long minF = -1;
   int minK = -1; // the query with min F
   if(world_rank == 0){
       // Initialize minF and minK with the first valid F value
       for(int i = 0; i < K; i++){
           if(all_F_values[i] >= 0){
               minF = all_F_values[i];
               minK = i;
               break;
           }
       }
       // Find the minimum F value
       for(int i = 0; i < K; i++){
           if(all_F_values[i] < minF && all_F_values[i] >= 0){
               minF = all_F_values[i];
               minK = i;
           }
       }
   }

   auto end_computation = chrono::high_resolution_clock::now();
   double computation_time = chrono::duration<double>(end_computation - start_computation).count();

   // Print results from rank 0 with high precision
   if(world_rank == 0){
       // Set the output to fixed format with 9 decimal places
       cout << fixed << setprecision(9);

       cout << "Graph: " << graphFile << "\n";
       cout << "Query: " << queryFile << "\n";
       cout << "Query number (k) with minimum F value: " << (minK + 1) << "\n";
       cout << "Minimum F value: " << minF << "\n";
       cout << "GPU # : " << numGPU << " GPU\n";
       cout << "Preprocessing time: " << preprocessing_time << " s\n";
       cout << "Computation time: " << computation_time << " s\n";
   }

   cudaFree(d_row_offsets);
   cudaFree(d_col_indices);
   cudaFree(d_distances);

   MPI_Finalize();
   return 0;
}
