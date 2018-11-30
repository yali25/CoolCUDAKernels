#include <iostream>
#include <algorithm>
#include <cuda.h>
#include <cublas_v2.h>


using std::cout;
using std::cerr;

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
    const int cols = 8;
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
    cudaMalloc(&cu_matrixA, msize*sizeof(double));    
    cudaMalloc(&cu_matrixB, msize*sizeof(double));    
    cudaMalloc(&cu_matrixC, (rows/batch_size)*cols*cols*sizeof(double));    

    // 3. copy the data from the host to the device
    cudaMemcpy(cu_matrixA, matrixA, msize*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(cu_matrixB, matrixB, msize*sizeof(double), cudaMemcpyHostToDevice);

    // 4. Setup the pointer arrays to the device memory
    double **pp_mA = new double*[rows/batch_size];
    double **pp_mB = new double*[rows/batch_size];
    double **pp_mC = new double*[rows/batch_size];

    // iterate over the matrices A and B and divide the matrices into (rows/batch_size) smaller matrices
    for(int i=0, counter=0, c_count=0; i<rows; i+=batch_size,counter++,c_count+=(cols*cols))
    {
       pp_mA[counter] = cu_matrixA+i;
       pp_mB[counter] = cu_matrixB+i;
       pp_mC[counter] = cu_matrixC+c_count; 
    }

    // 5. transfer the pointer arrays to the device memory
    const size_t pp_size = rows/batch_size*sizeof(double*);
    double **cu_pp_mA;
    double **cu_pp_mB;
    double **cu_pp_mC;
    cudaMalloc(&cu_pp_mA, pp_size);
    cudaMalloc(&cu_pp_mB, pp_size);
    cudaMalloc(&cu_pp_mC, pp_size);
    cudaMemcpy(cu_pp_mA, pp_mA, pp_size, cudaMemcpyHostToDevice);
    cudaMemcpy(cu_pp_mB, pp_mB, pp_size, cudaMemcpyHostToDevice);
    cudaMemcpy(cu_pp_mC, pp_mC, pp_size, cudaMemcpyHostToDevice);

    // 6. call gemmBatched 
    const double alpha = 1.0;
    const double beta  = 0.0;
    cublasDgemmBatched(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, cols, cols, batch_size,
                      &alpha, (const double**) cu_pp_mA, rows,
                              (const double**) cu_pp_mB, rows,
                      &beta, cu_pp_mC, cols, rows/batch_size);

    // 7. add up all small matrices to a result matrix
    int blocks  = 1024;   
    int threads = 64;
    int step_size        = floor(((blocks*threads) / (double)(cols*cols)));
    int out_matrix_index = (step_size)*(cols*cols);
    double *result;
    cudaMalloc(&result, cols*cols*sizeof(double));
    cudaMemset(result,0,cols*cols*sizeof(double));
    add_matrices << <blocks, threads >> > (step_size, out_matrix_index,
                                          (cols*cols)*rows/batch_size, cols*cols, cu_matrixC, result);

    double *cpu_result_buffer = new double[cols*cols];
    cudaMemcpy(cpu_result_buffer, result, cols*cols*sizeof(double), cudaMemcpyDeviceToHost);

    // 8. Print the result square matrix 
    for(int i=0;i<cols;i++)
    {
      for(int j=0;j<cols;j++)
      {
        std::cout << cpu_result_buffer[i*cols+j] << " ";
      }
      std::cout << "\n";
    }

    // 9. free all memory ...
     
    return 0;
}
