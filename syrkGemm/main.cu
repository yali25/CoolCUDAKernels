#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda.h>
#include <cublas_v2.h>


using std::cout;
using std::cerr;
using std::vector;

void printSquareMatrix(int cols, double *cu_output1){
    double *cpu_result_buffer = new double[cols*cols];
    cudaMemcpy(cpu_result_buffer, cu_output1, cols*cols*sizeof(double), cudaMemcpyDeviceToHost);
    // 6. Print the result square matrix 
    for(int i=0;i<cols;i++)
    {
      for(int j=0;j<cols;j++)
      {
        std::cout << cpu_result_buffer[i*cols+j] << " ";
      }
      std::cout << "\n";
    }
    delete[] cpu_result_buffer;
}
__global__
void add_matrices(int nblocks, 
                  int out_index,
	          int nm, 
                  int ms, 
                  const double *in_c_matrices, 
                  double *result_matrix)
{
       // start index in the input array and global tid
       const int start_idx = blockIdx.x * blockDim.x + threadIdx.x;
       
       const int start_matrix = (int)(start_idx / (float)ms);
       // cut off the last threads which lie on the last matrix which is not fully covered by threads
       if(nblocks != start_matrix)
       {
           // fetch the first element
           double my_sum = in_c_matrices[start_idx];
           // calculate the the matrix element the current thread processes
           const int out_element = (start_idx % ms);
           const int nn = nblocks*ms;
           for (int i = ((start_matrix + nblocks)*ms+out_element); i < nm; i += nn)
           {
           	my_sum += in_c_matrices[i];
           }
           atomicAdd(&result_matrix[out_element], my_sum);
      }
}


int main()
{
    cublasHandle_t cublas_handle;
    cublasStatus_t  blaserr = cublasCreate(&cublas_handle);

    if(blaserr != CUBLAS_STATUS_SUCCESS)
    {
       std::cerr << "Could not init the cublas library .\n";
       return 1;
    }

    // 1. Create two large matrices each with 1e6 rows and 16 columns
    const int rows = 1e6;
    const int cols = 22;
    const int msize = rows*cols;
    double *matrixA = new double[msize];
    double *matrixB = new double[msize];
    std::fill(matrixA, matrixA+msize, 1.0);
    std::fill(matrixB, matrixB+msize, 1.0);

    // 2. Allocate memory on the device for the two matrices and the result matrices
    const int batch_size = 32;
    double *cu_matrixA;
    double *cu_matrixB;
    double *cu_matrixC;
    double *cu_output1;
    double *cu_output2;
    cudaMalloc(&cu_matrixA, msize*sizeof(double));    
    cudaMalloc(&cu_matrixB, msize*sizeof(double));    
    cudaMalloc(&cu_matrixC, (rows/batch_size)*cols*cols*sizeof(double));    
    cudaMalloc(&cu_output1, cols*cols*sizeof(double));    
    cudaMalloc(&cu_output2, cols*cols*sizeof(double));    
    double *result;
    cudaMalloc(&result, cols*cols*sizeof(double));

    // 3. copy the data from the host to the device
    cudaMemcpy(cu_matrixA, matrixA, msize*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(cu_matrixB, matrixB, msize*sizeof(double), cudaMemcpyHostToDevice);
    const double alpha = 1.0;
    const double beta  = 0.0;

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float gemm  = 0.0;
    float bgemm = 0.0;
    float dsyrk = 0.0;
    vector<float>v_gemm;
    vector<float>v_syrk;
    vector<float>v_batch;
    const int MAX_IT = 101;

    for(int i=0;i<MAX_IT;i++){
         // 4.1 call normal gemm
         cudaEventRecord(start);
         cublasDgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, cols, cols, rows,
                     &alpha, cu_matrixA, rows,
                     cu_matrixA, rows,
                     &beta,  cu_output1, cols);
         cudaEventRecord(stop);
         cudaEventSynchronize(stop);
         cudaEventElapsedTime(&gemm, start, stop);
         v_gemm.push_back(gemm);

         // 4.2 call syrk
         cudaEventRecord(start);
         cublasDsyrk(cublas_handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, cols, rows,
                     &alpha, cu_matrixA, rows,
                     &beta,  cu_output2, cols);
         cudaEventRecord(stop);
         cudaEventSynchronize(stop);
         cudaEventElapsedTime(&dsyrk, start, stop);
         v_syrk.push_back(dsyrk);

         // 4. call gemmBatched 
         cudaEventRecord(start);
         cublasDgemmStridedBatched(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, cols, cols, batch_size,
                                   &alpha, cu_matrixA, rows, batch_size,
                                           cu_matrixA, rows, batch_size,
                                   &beta,  cu_matrixC, cols, cols*cols, rows/batch_size);

         // 5. add up all small matrices to a result matrix
         int blocks  = 1024;   
         int threads = 64;
         int step_size        = floor(((blocks*threads) / (double)(cols*cols)));
         int out_matrix_index = (step_size)*(cols*cols);
         cudaMemset(result,0,cols*cols*sizeof(double));
         add_matrices << <blocks, threads >> > (step_size, out_matrix_index,
                                               (cols*cols)*rows/batch_size, cols*cols, cu_matrixC, result);
         cudaEventRecord(stop);
         cudaEventSynchronize(stop);
         cudaEventElapsedTime(&bgemm, start, stop);
         v_batch.push_back(bgemm);
    }
    printSquareMatrix(cols,cu_output1);
    printSquareMatrix(cols,cu_output2);
    printSquareMatrix(cols,result);

    for(int i=1;i<MAX_IT;i++){
     cout << "Run: " << i << " " << v_gemm[i] << " " << v_syrk[i] << " " << v_batch[i] << "\n";
    }

    //cout << "Gemm: "<< gemm << "\n";
    //cout << "Syrk: "<< dsyrk << "\n";
    //cout << "Bgemm: "<< bgemm << "\n";
    // 9. free all memory ...
     
    return 0;
}
