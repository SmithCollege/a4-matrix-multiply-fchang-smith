#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

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

// GPU kernel for basic matrix multiplication
__global__ void gpuMatrixMultiply(double* A, double* B, double* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        double sum = 0.0;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    int sizes[] = {100, 200, 500, 1000, 2000};
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
        int threadsPerBlock = 16;
        dim3 threads(threadsPerBlock, threadsPerBlock);
        dim3 blocks((N + threadsPerBlock - 1) / threadsPerBlock, (N + threadsPerBlock - 1) / threadsPerBlock);

        // Perform matrix multiplication
        double t0 = get_clock();
        gpuMatrixMultiply<<<blocks, threads>>>(d_A, d_B, d_C, N);
        cudaDeviceSynchronize();
        double t1 = get_clock();

        printf("GPU Matrix Multiply - Size %d: %f s\n", N, t1 - t0);

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
