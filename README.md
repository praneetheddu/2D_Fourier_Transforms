                            2D Discrete Fourier Transform on CPU and GPU Report
# **Introduction:** 
The program solves for Two-Dimensional Discrete Fourier Transform using MPI, pthreads, and CUDA. For p-threads, there are 8 threads used 
to run the calculations. Since the equation runs in O(N^2) time, the alternate solution is to compute the timesteps by running multiple threads. The algorithm is tested on multiple matrices with increasing matrix dimensions with each iteration. Execution time will be recorded and the output matrix is exported in a text file format. pthreads, MPI, and Nvidia CUDA libraries are imported to the run the program.

# **Execution:** 
Use the following commands in a linux terminal to build the modules:

module load gcc/4.9.0 \
module load openmpi \
module load cmake/3.9.1 \
module load cuda/9.1 \
mkdir newBuildDir \
cd newBuildDir \
cmake .. \
make \
ls (should show p31, p32, and p33 executables) 


To run* (we are all undergrad so only forward): 

#p31 (cpu implementation): \
./p31 forward ../Tower256.txt ../out.txt 

#p32 (mpi implementation): \
Mpirun -np 8 ./p32 forward ../Tower256.txt ../out.txt 

#p33 (cuda implementation): \
./p33 forward ../Tower256.txt ../out.txt 

#Tested Using \
qsub -I -q coc-ice -l nodes=1:ppn=8:nvidiagpu,walltime=3:00:00,pmem=5gb 

## **Multi-Threaded CPU Implementation:** 
Matrix Dimensions   Execution Time (s) \
128 x 128           0.058542 \
256 x 256           0.46119 \
512 x 512           3.635721 \
1024 x 1024         28.655291 \
2048 x 2048         228.442804  


## **MPI Implementation:** 
Matrix Dimensions 	Execution Time (s) \
128 x 128 			0.225 \
256 x 256 			0.253 \
512 x 512 			0.271 \
1024 x 1024 		0.311 \
2048 x 2048 		0.492 

## **CUDA Implementation:** 
Matrix Dimensions 	Execution Time (s) \
128 x 128 			0.270 \
256 x 256 			0.383 \
512 x 512 			0.679 \
1024 x 1024 		2.008 \
2048 x 2048 		7.474 

# **Results:** 
CUDA and MPI outperformed p-threads which was expected since the GPU is capable of running thousands of threads where as p-threads only used 8 threads to run the algorithm. The difference is significantly higher since p-threads took 228.44 seconds run on a 2048 * 2048 matrix and CUDA and MPI had significantly lower execution times. 
