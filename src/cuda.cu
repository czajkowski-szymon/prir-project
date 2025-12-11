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

// ------------------------- KERNEL SUM -------------------------
__global__ void sum_reduce_int64(int64_t* g_idata, int64_t* g_odata, size_t N) {
    extern __shared__ int64_t sdata_sum_int64[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata_sum_int64[tid] = (i < N) ? g_idata[i] : 0;
    __syncthreads();

	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (threadIdx.x < s) {
			sdata_sum_int64[threadIdx.x] += sdata_sum_int64[threadIdx.x + s];
		}
		__syncthreads();
	}

    if (tid == 0) g_odata[blockIdx.x] = sdata_sum_int64[0];
}

__global__ void sum_reduce_double(double* g_idata, double* g_odata, size_t N) {
    extern __shared__ double sdata_sum_double[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata_sum_double[tid] = (i < N) ? g_idata[i] : 0.0f;
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
__global__ void min_reduce_int64(int64_t* g_idata, int64_t* g_odata, size_t N, int64_t int_max) {
    extern __shared__ int64_t sdata_min_int64[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata_min_int64[tid] = (i < N) ? g_idata[i] : int_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (sdata_min_int64[tid + s] < sdata_min_int64[tid]) sdata_min_int64[tid] = sdata_min_int64[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata_min_int64[0];
}

__global__ void min_reduce_double(double* g_idata, double* g_odata, size_t N, double dbl_max) {
    extern __shared__ double sdata_min_double[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata_min_double[tid] = (i < N) ? g_idata[i] : dbl_max;
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
__global__ void max_reduce_int64(int64_t* g_idata, int64_t* g_odata, size_t N, int64_t int_min) {
    extern __shared__ int64_t sdata_max_int64[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata_max_int64[tid] = (i < N) ? g_idata[i] : int_min;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (sdata_max_int64[tid + s] > sdata_max_int64[tid]) sdata_max_int64[tid] = sdata_max_int64[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata_max_int64[0];
}

__global__ void max_reduce_double(double* g_idata, double* g_odata, size_t N, double dbl_min) {
    extern __shared__ double sdata_max_double[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata_max_double[tid] = (i < N) ? g_idata[i] : dbl_min;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (sdata_max_double[tid + s] > sdata_max_double[tid]) sdata_max_double[tid] = sdata_max_double[tid + s];
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

        sum_reduce_int64<<<num_blocks, block_size, shared_mem_size>>>(d_current, d_out, current_size);
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

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_data);
    return result;
}

double gpu_sum_double(const std::vector<double>& h_data, int block_size, float& time) {
    size_t N = h_data.size();
    double* d_data;
    cudaMalloc(&d_data, N * sizeof(double));
    cudaMemcpy(d_data, h_data.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    double* d_current = d_data;
    double* d_out = nullptr;
    size_t current_size = N;
    double result = 0.0;

    float elapsed_ms = 0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    while (true) {
        size_t num_blocks = (current_size + block_size - 1) / block_size;
        cudaMalloc(&d_out, num_blocks * sizeof(double));
        size_t shared_mem_size = block_size * sizeof(double);

        sum_reduce_double<<<num_blocks, block_size, shared_mem_size>>>(d_current, d_out, current_size);
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

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    time = elapsed_ms;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_data);
    return result;
}

// --------------------- HOST MIN --------------------
int64_t gpu_min_int64(const std::vector<int64_t>& h_data, int block_size, float& time) {
    size_t N = h_data.size();
    int64_t* d_data;
    cudaMalloc(&d_data, N * sizeof(int64_t));
    cudaMemcpy(d_data, h_data.data(), N * sizeof(int64_t), cudaMemcpyHostToDevice);

    int64_t* d_current = d_data;
    int64_t* d_out = nullptr;
    size_t current_size = N;
    int64_t result = INT64_MAX;

    float elapsed_ms = 0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    while (true) {
        size_t num_blocks = (current_size + block_size - 1) / block_size;
        cudaMalloc(&d_out, num_blocks * sizeof(int64_t));
        size_t shared_mem_size = block_size * sizeof(int64_t);

        min_reduce_int64<<<num_blocks, block_size, shared_mem_size>>>(d_current, d_out, current_size, INT64_MAX);
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

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_data);
    return result;
}

double gpu_min_double(const std::vector<double>& h_data, int block_size, float& time) {
    size_t N = h_data.size();
    double* d_data;
    cudaMalloc(&d_data, N * sizeof(double));
    cudaMemcpy(d_data, h_data.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    double* d_current = d_data;
    double* d_out = nullptr;
    size_t current_size = N;
    double result = std::numeric_limits<double>::max();

    float elapsed_ms = 0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    while (true) {
        size_t num_blocks = (current_size + block_size - 1) / block_size;
        cudaMalloc(&d_out, num_blocks * sizeof(double));
        size_t shared_mem_size = block_size * sizeof(double);

        min_reduce_double<<<num_blocks, block_size, shared_mem_size>>>(d_current, d_out, current_size, (double)std::numeric_limits<double>::max());
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

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    time = elapsed_ms;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_data);
    return result;
}

// --------------------- HOST MAX --------------------
int64_t gpu_max_int64(const std::vector<int64_t>& h_data, int block_size, float& time) {
    size_t N = h_data.size();
    int64_t* d_data;
    cudaMalloc(&d_data, N * sizeof(int64_t));
    cudaMemcpy(d_data, h_data.data(), N * sizeof(int64_t), cudaMemcpyHostToDevice);

    int64_t* d_current = d_data;
    int64_t* d_out = nullptr;
    size_t current_size = N;
    int64_t result = INT64_MIN;

    float elapsed_ms = 0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    while (true) {
        size_t num_blocks = (current_size + block_size - 1) / block_size;
        cudaMalloc(&d_out, num_blocks * sizeof(int64_t));
        size_t shared_mem_size = block_size * sizeof(int64_t);

        max_reduce_int64<<<num_blocks, block_size, shared_mem_size>>>(d_current, d_out, current_size, INT64_MIN);
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

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_data);
    return result;
}

double gpu_max_double(const std::vector<double>& h_data, int block_size, float& time) {
    size_t N = h_data.size();
    double* d_data;
    cudaMalloc(&d_data, N * sizeof(double));
    cudaMemcpy(d_data, h_data.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    double* d_current = d_data;
    double* d_out = nullptr;
    size_t current_size = N;
    double result = std::numeric_limits<double>::lowest();

    float elapsed_ms = 0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    while (true) {
        size_t num_blocks = (current_size + block_size - 1) / block_size;
        cudaMalloc(&d_out, num_blocks * sizeof(double));
        size_t shared_mem_size = block_size * sizeof(double);

        max_reduce_double<<<num_blocks, block_size, shared_mem_size>>>(d_current, d_out, current_size, (double)std::numeric_limits<double>::min());
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

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    time = elapsed_ms;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);

    return result;
}

// ------------------------- HOST PREFIX-SUM -------------------------
void gpu_prefix_sum_int64(std::vector<int64_t>& h_data, int block_size, float& time) {
    size_t N = h_data.size();
    int64_t *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(int64_t));
    cudaMalloc(&d_out, N * sizeof(int64_t));
    cudaMemcpy(d_in, h_data.data(), N * sizeof(int64_t), cudaMemcpyHostToDevice);

    size_t num_blocks = (N + block_size - 1) / block_size;
    size_t shared_mem = block_size * sizeof(int64_t);

    float elapsed_ms = 0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    prefix_sum_block_int64<<<num_blocks, block_size, shared_mem>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    time = elapsed_ms;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(h_data.data(), d_out, N * sizeof(int64_t), cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
}

void gpu_prefix_sum_double(std::vector<double>& h_data, int block_size, float& time) {
    size_t N = h_data.size();
    double *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(double));
    cudaMalloc(&d_out, N * sizeof(double));
    cudaMemcpy(d_in, h_data.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    size_t num_blocks = (N + block_size - 1) / block_size;
    size_t shared_mem = block_size * sizeof(double);

    float elapsed_ms = 0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    prefix_sum_block_double<<<num_blocks, block_size, shared_mem>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    time = elapsed_ms;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

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
    std::string block_arg = argv[4] ? argv[4] : "512";

    const size_t SIZE = 1 << stoi(power);

    int block_size = stoi(block_arg);

    if (type_str == "int") {
        std::vector<int64_t> h_data(SIZE);
        std::mt19937_64 gen(1234);
        std::uniform_int_distribution<int64_t> dist(0, 1000000);
        for (auto &x : h_data) x = dist(gen);

        if (op_str == "sum") {
            auto seq = sequential_sum(h_data);
            std::cout << "[SEQ] Sum: " << seq << std::endl;

            float time;
            auto result = gpu_sum_int64(h_data, block_size, time);

            double abs_err = fabs((double)seq - (double)result);
            double rel_err = fabs(seq) > 0 ? abs_err / fabs((double)seq) : 0.0;
        
            std::cout << "[PAR] Sum: " << result << " time(ms): " << time << std::endl;

            std::string filename = "../results/cuda/cuda_sum_int.csv";
            std::ofstream fout(filename, std::ios::app);
            fout << power << "," << block_size << "," << time << "," << abs_err << "," << rel_err << std::endl;
            fout.close();
        } else if (op_str == "min") {
            auto seq = sequential_min(h_data);
            std::cout << "[SEQ] Min: " << seq << std::endl;

            float time;
            auto result = gpu_min_int64(h_data, block_size, time);

            double abs_err = fabs((double)seq - (double)result);
            double rel_err = fabs(seq) > 0 ? abs_err / fabs((double)seq) : 0.0;
        
            std::cout << "[PAR] Min: " << result << " time(ms): " << time << std::endl;

            std::string filename = "../results/cuda/cuda_min_int.csv";
            std::ofstream fout(filename, std::ios::app);
            fout << power << "," << block_size << "," << time << "," << abs_err << "," << rel_err << std::endl;
            fout.close();
        } else if (op_str == "max") {
            auto seq = sequential_max(h_data);
            std::cout << "[SEQ] Max: " << seq << std::endl;

            float time;
            auto result = gpu_max_int64(h_data, block_size, time);

            double abs_err = fabs((double)seq - (double)result);
            double rel_err = fabs(seq) > 0 ? abs_err / fabs((double)seq) : 0.0;
        
            std::cout << "[PAR] Max: " << result << " time(ms): " << time << std::endl;

            std::string filename = "../results/cuda/cuda_max_int.csv";
            std::ofstream fout(filename, std::ios::app);
            fout << power << "," << block_size << "," << time << "," << abs_err << "," << rel_err << std::endl;
            fout.close();
        } else if (op_str == "prefix") {            
            auto seq = sequential_exclusive_scan(h_data);

            float time;
            gpu_prefix_sum_int64(h_data, block_size, time);

            double err_sq = 0.0, denom_sq = 0.0;
            for (size_t i = 0; i < seq.size(); ++i) {
                double d = seq[i] - h_data[i];
                err_sq += d*d;
                denom_sq += seq[i]*seq[i];
            }
            double abs_err = sqrt(err_sq);
            double denom = sqrt(denom_sq);
            double rel_err = denom > 0 ? abs_err / denom : 0.0;
        
            std::cout << "[PAR] Prefix time(ms): " << time << std::endl;

            std::string filename = "../results/cuda/cuda_prefix_double.csv";
            std::ofstream fout(filename, std::ios::app);
            fout << power << "," << block_size << "," << time << "," << abs_err << "," << rel_err << std::endl;
            fout.close();
        } else {
            std::cerr << "Unknown operation: " << op_str << std::endl;
            return 1;
        }
    } else if (type_str == "double") {
        std::vector<double> h_data(SIZE);
        std::mt19937_64 gen(1234);
        std::uniform_real_distribution<double> dist(0.0, 1000000.0);
        for (auto &x : h_data) x = dist(gen);

        if (op_str == "sum") {
            auto seq = sequential_sum(h_data);
            std::cout << "[SEQ] Sum: " << seq << std::endl;

            float time;
            auto result = gpu_sum_double(h_data, block_size, time);

            double abs_err = fabs((double)seq - (double)result);
            double rel_err = fabs(seq) > 0 ? abs_err / fabs((double)seq) : 0.0;
        
            std::cout << "[PAR] Sum: " << result << " time(ms): " << time << std::endl;

            std::string filename = "../results/cuda/cuda_sum_double.csv";
            std::ofstream fout(filename, std::ios::app);
            fout << power << "," << block_size << "," << time << "," << abs_err << "," << rel_err << std::endl;
            fout.close();
        } else if (op_str == "min") {
            auto seq = sequential_min(h_data);
            std::cout << "[SEQ] Min: " << seq << std::endl;

            float time;
            auto result = gpu_min_double(h_data, block_size, time);

            double abs_err = fabs((double)seq - (double)result);
            double rel_err = fabs(seq) > 0 ? abs_err / fabs((double)seq) : 0.0;
        
            std::cout << "[PAR] Min: " << result << " time(ms): " << time << std::endl;

            std::string filename = "../results/cuda/cuda_min_double.csv";
            std::ofstream fout(filename, std::ios::app);
            fout << power << "," << block_size << "," << time << "," << abs_err << "," << rel_err << std::endl;
            fout.close();
        } else if (op_str == "max") {
            auto seq = sequential_max(h_data);
            std::cout << "[SEQ] Max: " << seq << std::endl;

            float time;
            auto result = gpu_max_double(h_data, block_size, time);

            double abs_err = fabs((double)seq - (double)result);
            double rel_err = fabs(seq) > 0 ? abs_err / fabs((double)seq) : 0.0;
        
            std::cout << "[PAR] Max: " << result << " time(ms): " << time << std::endl;

            std::string filename = "../results/cuda/cuda_max_double.csv";
            std::ofstream fout(filename, std::ios::app);
            fout << power << "," << block_size << "," << time << "," << abs_err << "," << rel_err << std::endl;
            fout.close();
        } else if (op_str == "prefix") {
            for (int i = 0; i < 50; i++) std::cout << h_data[i] << " ";
            std::cout << std::endl;

            auto seq = sequential_exclusive_scan(h_data);

            float time;
            gpu_prefix_sum_double(h_data, block_size, time);

            double err_sq = 0.0, denom_sq = 0.0;
            for (size_t i = 0; i < seq.size(); ++i) {
                double d = seq[i] - h_data[i];
                err_sq += d*d;
                denom_sq += seq[i]*seq[i];
            }
            double abs_err = sqrt(err_sq);
            double denom = sqrt(denom_sq);
            double rel_err = denom > 0 ? abs_err / denom : 0.0;
        
            std::cout << "[PAR] Prefix time(ms): " << time << std::endl;

            std::string filename = "../results/cuda/cuda_prefix_double.csv";
            std::ofstream fout(filename, std::ios::app);
            fout << power << "," << block_size << "," << time << "," << abs_err << "," << rel_err << std::endl;
            fout.close();
        } else {
            std::cerr << "Unknown operation: " << op_str << std::endl;
            return 1;
        }
    } else {
        std::cerr << "Unknown type: " << type_str << std::endl;
        return 1;
    }

    return 0;
}
