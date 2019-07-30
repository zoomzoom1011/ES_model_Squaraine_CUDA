# ES_model_Squaraine_CUDA

Employ Essential State model (ESM) for simulating the photophysical properties of squaraine dye aggregates, 
characterized respectively by substantial (permanent) dipole and quadrupole moments.

This is the CUDA version code for Essential State model. It is 1000 times faster than CPP code. 

To run the code, you need to: 
1. Install CUDA and set up variable environment
2. Install intel compiler (optional)  
3. Compile magma

The baisc idea for CUDA computation is: 
1. define all the objects in the CUDA and magma
2. allocate GPU memory for the matrix (cudaMalloc)
3. copy the matrix to the the GPU
4. use diagonalization function in magma
5. copy the result from GPU to Memory
6. delete CUDA & magma objects and GPU memory space



