# Parallel Multi-Source BFS Using MPI and CUDA

This project implements a **highly parallel multi-source Breadth-First Search (BFS)** system on massive graphs using **MPI (Message Passing Interface)** and **CUDA (GPU programming)**. It solves the **Distance-to-Set** problem efficiently by distributing queries across MPI ranks and executing BFS on the GPU in parallel for each query group.

## 🚀 Problem Overview

For each group of source vertices (query), compute the minimum distances to all other vertices in the graph. The goal is to find the query group that yields the minimum total distance across the graph.

## ⚙️ Technologies

- **C++**
- **CUDA** (GPU computation)
- **MPI** (distributed computation)
- CSR (Compressed Sparse Row) graph representation
- Binary I/O for graph and query loading

## 🔍 Key Features

- **Multi-source BFS on GPU** with one thread per node
- **Parallel query handling** via MPI
- **Custom MPI DataType** for communicating `(query_id, distance_sum)` pairs
- **CSR Graph Format** to efficiently store sparse graphs
- **Round-robin query distribution** for workload balancing
- **Time measurement** for both preprocessing and computation


## 🧪 Usage

```bash
mpirun -np <num_ranks> ./main -g <graph.bin> -q <query.bin> -gn <num_GPUs>


