#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define TILE_SIZE 16

// Function to get current time in seconds
double get_clock() {
    struct timeval tv;
    int ok;
    ok = gettimeofday(&tv, (void*)0);
    if (ok < 0) {
        printf("gettimeofday error");
    }
    return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}

// GPU kernel for tiled matrix multiplication
__global__ void tiledMatrixMultiply(double* A, double* B, double* C, int N) {
    __shared__ double tileA[TILE_SIZE][TILE_SIZE];
    __shared__ double tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    double sum = 0.0;
    for (int k = 0; k < (N + TILE_SIZE - 1) / TILE_SIZE; ++k) {
        if (row < N && (k * TILE_SIZE + threadIdx.x) < N)
            tileA[threadIdx.y][threadIdx.x] = A[row * N + k * TILE_SIZE + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0;

        if (col < N && (k * TILE_SIZE + threadIdx.y) < N)
            tileB[threadIdx.y][threadIdx.x] = B[(k * TILE_SIZE + threadIdx.y) * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();

        for (int t = 0; t < TILE_SIZE; ++t)
            sum += tileA[threadIdx.y][t] * tileB[t][threadIdx.x];

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

int main() {
    int sizes[] = {100, 200, 500, 1000, 2000}; // Test sizes
    for (int idx = 0; idx < 5; idx++) {
        int N = sizes[idx];
        double *A, *B, *C, *d_A, *d_B, *d_C;

        // Allocate memory
        A = (double*)malloc(N * N * sizeof(double));
        B = (double*)malloc(N * N * sizeof(double));
        C = (double*)malloc(N * N * sizeof(double));
        cudaMalloc(&d_A, N * N * sizeof(double));
        cudaMalloc(&d_B, N * N * sizeof(double));
        cudaMalloc(&d_C, N * N * sizeof(double));

        // Initialize matrices
        for (int i = 0; i < N * N; ++i) {
            A[i] = 1.0;
            B[i] = 1.0;
        }

        cudaMemcpy(d_A, A, N * N * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B, N * N * sizeof(double), cudaMemcpyHostToDevice);

        // Kernel configuration
        dim3 threads(TILE_SIZE, TILE_SIZE);
        dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

        // Perform matrix multiplication
        double t0 = get_clock();
        tiledMatrixMultiply<<<blocks, threads>>>(d_A, d_B, d_C, N);
        cudaDeviceSynchronize();
        double t1 = get_clock();

        printf("GPU Tiled Matrix Multiply - Size %d: %f s\n", N, t1 - t0);

        // Copy results back
        cudaMemcpy(C, d_C, N * N * sizeof(double), cudaMemcpyDeviceToHost);

        // Free memory
        free(A);
        free(B);
        free(C);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }

    return 0;
}
