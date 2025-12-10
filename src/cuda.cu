#include <cuda_runtime.h>
#include "sequential.hpp"
#include <iostream>
#include <vector>
#include <cstdint> 
#include <iostream>
#include <algorithm>
#include <limits>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <numeric>
#include <cmath>

// #define SIZE (1 << 20)
// #define BLOCK_SIZE 1024

// ------------------------- KERNEL SUM -------------------------
__global__ void sum_reduce_int64(int64_t* g_idata, int64_t* g_odata) {
    extern __shared__ int64_t sdata_sum_int64[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata_sum_int64[tid] = g_idata[i];
    __syncthreads();

	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (threadIdx.x < s) {
			sdata_sum_int64[threadIdx.x] += sdata_sum_int64[threadIdx.x + s];
		}
		__syncthreads();
	}

    if (tid == 0) g_odata[blockIdx.x] = sdata_sum_int64[0];
}

__global__ void sum_reduce_double(double* g_idata, double* g_odata) {
    extern __shared__ double sdata_sum_double[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata_sum_double[tid] = g_idata[i];
    __syncthreads();

	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (threadIdx.x < s) {
			sdata_sum_double[threadIdx.x] += sdata_sum_double[threadIdx.x + s];
		}
		__syncthreads();
	}

    if (tid == 0) g_odata[blockIdx.x] = sdata_sum_double[0];
}

// ------------------------- KERNEL MIN -------------------------
__global__ void min_reduce_int64(int64_t* g_idata, int64_t* g_odata) {
    extern __shared__ int64_t sdata_min_int64[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata_min_int64[tid] = g_idata[i];
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (sdata_min_int64[tid + s] < sdata_min_int64[tid]) sdata_min_int64[tid] = sdata_min_int64[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata_min_int64[0];
}

__global__ void min_reduce_double(double* g_idata, double* g_odata) {
    extern __shared__ double sdata_min_double[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata_min_double[tid] = g_idata[i];
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (sdata_min_double[tid + s] < sdata_min_double[tid]) sdata_min_double[tid] = sdata_min_double[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata_min_double[0];
}

// ------------------------- KERNEL MAX -------------------------
__global__ void max_reduce_int64(int64_t* g_idata, int64_t* g_odata) {
    extern __shared__ int64_t sdata_max_int64[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata_max_int64[tid] = g_idata[i];
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (sdata_max_int64[tid + s] < sdata_max_int64[tid]) sdata_max_int64[tid] = sdata_max_int64[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata_max_int64[0];
}

__global__ void max_reduce_double(double* g_idata, double* g_odata) {
    extern __shared__ double sdata_max_double[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata_max_double[tid] = g_idata[i];
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (sdata_max_double[tid + s] < sdata_max_double[tid]) sdata_max_double[tid] = sdata_max_double[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata_max_double[0];
}

// ------------------------- KERNEL PREFIX-SUM -------------------------
__global__ void prefix_sum_block_int64(int64_t* g_idata, int64_t* g_odata, size_t n) {
    extern __shared__ int64_t sdata_scan_int64[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;

    sdata_scan_int64[tid] = (i < n) ? g_idata[i] : 0;
    __syncthreads();

    for (unsigned int offset = 1; offset < blockDim.x; offset <<= 1) {
        int idx = (tid + 1) * offset * 2 - 1;
        if (idx < blockDim.x)
            sdata_scan_int64[idx] += sdata_scan_int64[idx - offset];
        __syncthreads();
    }

    if (tid == 0) sdata_scan_int64[blockDim.x - 1] = 0;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        int idx = (tid + 1) * offset * 2 - 1;
        if (idx < blockDim.x) {
            int64_t t = sdata_scan_int64[idx - offset];
            sdata_scan_int64[idx - offset] = sdata_scan_int64[idx];
            sdata_scan_int64[idx] += t;
        }
        __syncthreads();
    }

    if (i < n)
        g_odata[i] = sdata_scan_int64[tid];
}

__global__ void prefix_sum_block_double(double* g_idata, double* g_odata, size_t n) {
    extern __shared__ double sdata_scan_double[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;

    sdata_scan_double[tid] = (i < n) ? g_idata[i] : 0;
    __syncthreads();

    for (unsigned int offset = 1; offset < blockDim.x; offset <<= 1) {
        int idx = (tid + 1) * offset * 2 - 1;
        if (idx < blockDim.x)
            sdata_scan_double[idx] += sdata_scan_double[idx - offset];
        __syncthreads();
    }

    if (tid == 0) sdata_scan_double[blockDim.x - 1] = 0;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        int idx = (tid + 1) * offset * 2 - 1;
        if (idx < blockDim.x) {
            double t = sdata_scan_double[idx - offset];
            sdata_scan_double[idx - offset] = sdata_scan_double[idx];
            sdata_scan_double[idx] += t;
        }
        __syncthreads();
    }

    if (i < n)
        g_odata[i] = sdata_scan_double[tid];
}

// --------------------- HOST SUM --------------------
int64_t gpu_sum_int64(const std::vector<int64_t>& h_data, int block_size, float& time) {
    size_t N = h_data.size();
    int64_t* d_data;
    cudaMalloc(&d_data, N * sizeof(int64_t));
    cudaMemcpy(d_data, h_data.data(), N * sizeof(int64_t), cudaMemcpyHostToDevice);

    int64_t* d_current = d_data;
    int64_t* d_out = nullptr;
    size_t current_size = N;
    int64_t result = 0;
    
    float elapsed_ms = 0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    while (true) {
        size_t num_blocks = (current_size + block_size - 1) / block_size;
        cudaMalloc(&d_out, num_blocks * sizeof(int64_t));
        size_t shared_mem_size = block_size * sizeof(int64_t);

        sum_reduce_int64<<<num_blocks, block_size, shared_mem_size>>>(d_current, d_out);
        cudaDeviceSynchronize();
        
        if (d_current != d_data) cudaFree(d_current);

        if (num_blocks == 1) {
            cudaMemcpy(&result, d_out, sizeof(int64_t), cudaMemcpyDeviceToHost);
            cudaFree(d_out);
            break;
        }

        d_current = d_out;
        current_size = num_blocks;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    time = elapsed_ms;

    // std::cout << "GPU reduction time: " << elapsed_ms << " ms\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_data);
    return result;
}

double gpu_sum_double(const std::vector<double>& h_data, int block_size) {
    size_t N = h_data.size();
    double* d_data;
    cudaMalloc(&d_data, N * sizeof(double));
    cudaMemcpy(d_data, h_data.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    double* d_current = d_data;
    double* d_out = nullptr;
    size_t current_size = N;
    double result = 0.0;

    while (true) {
        size_t num_blocks = (current_size + block_size - 1) / block_size;
        cudaMalloc(&d_out, num_blocks * sizeof(double));
        size_t shared_mem_size = block_size * sizeof(double);

        sum_reduce_double<<<num_blocks, block_size, shared_mem_size>>>(d_current, d_out);
        cudaDeviceSynchronize();

        if (d_current != d_data) cudaFree(d_current);

        if (num_blocks == 1) {
            cudaMemcpy(&result, d_out, sizeof(double), cudaMemcpyDeviceToHost);
            cudaFree(d_out);
            break;
        }

        d_current = d_out;
        current_size = num_blocks;
    }

    cudaFree(d_data);
    return result;
}

// --------------------- HOST MIN --------------------
int64_t gpu_min_int64(const std::vector<int64_t>& h_data, int block_size) {
    size_t N = h_data.size();
    int64_t* d_data;
    cudaMalloc(&d_data, N * sizeof(int64_t));
    cudaMemcpy(d_data, h_data.data(), N * sizeof(int64_t), cudaMemcpyHostToDevice);

    int64_t* d_current = d_data;
    int64_t* d_out = nullptr;
    size_t current_size = N;
    int64_t result = INT64_MAX;

    while (true) {
        size_t num_blocks = (current_size + block_size - 1) / block_size;
        cudaMalloc(&d_out, num_blocks * sizeof(int64_t));
        size_t shared_mem_size = block_size * sizeof(int64_t);

        min_reduce_int64<<<num_blocks, block_size, shared_mem_size>>>(d_current, d_out);
        cudaDeviceSynchronize();

        if (d_current != d_data) cudaFree(d_current);

        if (num_blocks == 1) {
            cudaMemcpy(&result, d_out, sizeof(int64_t), cudaMemcpyDeviceToHost);
            cudaFree(d_out);
            break;
        }

        d_current = d_out;
        current_size = num_blocks;
    }

    cudaFree(d_data);
    return result;
}

double gpu_min_double(const std::vector<double>& h_data, int block_size) {
    size_t N = h_data.size();
    double* d_data;
    cudaMalloc(&d_data, N * sizeof(double));
    cudaMemcpy(d_data, h_data.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    double* d_current = d_data;
    double* d_out = nullptr;
    size_t current_size = N;
    double result = std::numeric_limits<double>::max();

    while (true) {
        size_t num_blocks = (current_size + block_size - 1) / block_size;
        cudaMalloc(&d_out, num_blocks * sizeof(double));
        size_t shared_mem_size = block_size * sizeof(double);

        min_reduce_double<<<num_blocks, block_size, shared_mem_size>>>(d_current, d_out);
        cudaDeviceSynchronize();

        if (d_current != d_data) cudaFree(d_current);

        if (num_blocks == 1) {
            cudaMemcpy(&result, d_out, sizeof(double), cudaMemcpyDeviceToHost);
            cudaFree(d_out);
            break;
        }

        d_current = d_out;
        current_size = num_blocks;
    }

    cudaFree(d_data);
    return result;
}

// --------------------- HOST MAX --------------------
int64_t gpu_max_int64(const std::vector<int64_t>& h_data) {
    size_t N = h_data.size();
    int64_t* d_data;
    cudaMalloc(&d_data, N * sizeof(int64_t));
    cudaMemcpy(d_data, h_data.data(), N * sizeof(int64_t), cudaMemcpyHostToDevice);

    int64_t* d_current = d_data;
    int64_t* d_out = nullptr;
    size_t current_size = N;
    int64_t result = INT64_MIN;

    while (true) {
        size_t num_blocks = (current_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        cudaMalloc(&d_out, num_blocks * sizeof(int64_t));
        size_t shared_mem_size = BLOCK_SIZE * sizeof(int64_t);

        max_reduce_int64<<<num_blocks, BLOCK_SIZE, shared_mem_size>>>(d_current, d_out);
        cudaDeviceSynchronize();

        if (d_current != d_data) cudaFree(d_current);

        if (num_blocks == 1) {
            cudaMemcpy(&result, d_out, sizeof(int64_t), cudaMemcpyDeviceToHost);
            cudaFree(d_out);
            break;
        }

        d_current = d_out;
        current_size = num_blocks;
    }

    cudaFree(d_data);
    return result;
}

double gpu_max_double(const std::vector<double>& h_data) {
    size_t N = h_data.size();
    double* d_data;
    cudaMalloc(&d_data, N * sizeof(double));
    cudaMemcpy(d_data, h_data.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    double* d_current = d_data;
    double* d_out = nullptr;
    size_t current_size = N;
    double result = std::numeric_limits<double>::lowest();

    while (true) {
        size_t num_blocks = (current_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        cudaMalloc(&d_out, num_blocks * sizeof(double));
        size_t shared_mem_size = BLOCK_SIZE * sizeof(double);

        max_reduce_double<<<num_blocks, BLOCK_SIZE, shared_mem_size>>>(d_current, d_out);
        cudaDeviceSynchronize();

        if (d_current != d_data) cudaFree(d_current);

        if (num_blocks == 1) {
            cudaMemcpy(&result, d_out, sizeof(double), cudaMemcpyDeviceToHost);
            cudaFree(d_out);
            break;
        }

        d_current = d_out;
        current_size = num_blocks;
    }

    cudaFree(d_data);
    return result;
}

// ------------------------- HOST PREFIX-SUM -------------------------
void gpu_prefix_sum_int64(std::vector<int64_t>& h_data) {
    size_t N = h_data.size();
    int64_t *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(int64_t));
    cudaMalloc(&d_out, N * sizeof(int64_t));
    cudaMemcpy(d_in, h_data.data(), N * sizeof(int64_t), cudaMemcpyHostToDevice);

    size_t num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t shared_mem = BLOCK_SIZE * sizeof(int64_t);

    prefix_sum_block_int64<<<num_blocks, BLOCK_SIZE, shared_mem>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_data.data(), d_out, N * sizeof(int64_t), cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
}

void gpu_prefix_sum_double(std::vector<double>& h_data) {
    size_t N = h_data.size();
    double *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(double));
    cudaMalloc(&d_out, N * sizeof(double));
    cudaMemcpy(d_in, h_data.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    size_t num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t shared_mem = BLOCK_SIZE * sizeof(double);

    prefix_sum_block_double<<<num_blocks, BLOCK_SIZE, shared_mem>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_data.data(), d_out, N * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <type: int|double> <operation: sum|min|max|prefix> [power] [threads]" << std::endl;
        return 1;
    }
    std::string type_str = argv[1] ? argv[1] : "double";
    std::string op_str = argv[2] ? argv[2] : "sum";
    std::string power = argv[3] ? argv[3] : "20";
    std::string threads_arg = argv[4] ? argv[4] : "512";

    const size_t SIZE = 1 << stoi(power);

    int threads = stoi(threads_arg);

    // INT64
    std::cout << "INT64:" << std::endl;
    {
        std::vector<int64_t> h_data(SIZE);
        // std::mt19937_64 gen(1234);
        // std::uniform_int_distribution<int64_t> dist(0, 1000000);
        for (auto &x : h_data) x = 1;

        float time;
        auto result = gpu_sum_int64(h_data, time);
        std::cout << "SUM = " << result << ", TIME = " << time << "ms" << std::endl;
    }

    {
        std::vector<int64_t> h_data(SIZE);
        for (size_t i = 0; i < SIZE; i++) h_data[i] = i - (SIZE / 2);
        int64_t min = gpu_min_int64(h_data);
        std::cout << "MIN = " << min << std::endl;
    }

    {
        std::vector<int64_t> h_data(SIZE);
        for (size_t i = 0; i < SIZE; i++) h_data[i] = i - (SIZE / 2);
        int64_t max = gpu_max_int64(h_data);
        std::cout << "MAX = " << max << std::endl;
    }

    {
        std::vector<int64_t> h_data = {3, 1, 7, 0, 4, 1, 6, 3};
        // for (size_t i = 0; i < SIZE; i++) h_data[i] = i - (SIZE / 2);
        gpu_prefix_sum_int64(h_data);
        for (int i = 0; i < h_data.size(); i++) {
            std::cout << h_data[i] << " ";
        }
        std::cout << std::endl;
    }

    // DOUBLE
    std::cout << "DOUBLE:" << std::endl;
    {
        std::vector<double> h_data(SIZE);
        for (size_t i = 0; i < SIZE; i++) h_data[i] = (double)1;
        double sum = gpu_sum_double(h_data);
        std::cout << "SUM = " << sum << std::endl;
    }

    {
        std::vector<double> h_data(SIZE);
        for (size_t i = 0; i < SIZE; i++) h_data[i] = (double)i - (double)(SIZE / 2);
        double min = gpu_min_double(h_data);
        std::cout << "MIN = " << min << std::endl;
    }

    {
        std::vector<double> h_data(SIZE);
        for (size_t i = 0; i < SIZE; i++) h_data[i] = (double)i - (double)(SIZE / 2);
        double max = gpu_max_double(h_data);
        std::cout << "MAX = " << max << std::endl;
    }

    {
        std::vector<double> h_data = {3.0, 1.0, 7.0, 0.0, 4.0, 1.0, 6.0, 3.0};
        // for (size_t i = 0; i < SIZE; i++) h_data[i] = i - (SIZE / 2);
        gpu_prefix_sum_double(h_data);
        for (int i = 0; i < h_data.size(); i++) {
            std::cout << h_data[i] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
