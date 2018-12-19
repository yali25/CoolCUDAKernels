# Description
This file describes each example. All examples are provided without a makefile, they can be compiled by simply calling the nvcc compiler and linking against CUBLAS.
## batchedGemm
This example shows how to multiply two Tall and Skinny (TS) matrices using CUBLAS batchedGemm. TS matrices are matrices which have much more rows than columns (rows >> columns). All standard gemm implementations (including CUBLAS) perform very poorly on this kind of matrices, however by using batchedGemm a significant performance improvment can be achieved. 
batchedGemm divides the input TS matrices into smaller matrices and then multiplies matrix pairs together. The small result matrices are then written back to the (device) memory, in the final step all small matrices are summed up to the final result matrix.
## stridedBatchedGemm
This example does the same like batchedGemm example but uses a different function. Instead of cublasDgemmBatched (previous example) cublasDgemmStridedBatched is used. cublasDgemmStidedBatched does the same like cublasDgemmBatched but it is easier to use because it does not use a pointer-to-pointer interface. 
