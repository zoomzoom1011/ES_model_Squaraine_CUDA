src     = sub/esm_c_magma.cu
exes    = esm_c_magma.exe
NVCC    = nvcc
INC     = -I$(CUDADIR)/include -I$(MKLROOT)/include -I$(MAGMADIR)/include
FLAGS	= -Xcompiler "-fPIC -Wall -Wno-unused-function" -DMKL_ILP64 -Wno-deprecated-gpu-targets -DADD_
LIBS    = -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -lgomp -lpthread -lm -ldl -lcufft -lmagma \
		-lcublas -lcusparse -lcudart
LIBDIRS = -L/opt/intel/mkl/lib -L/Developer/NVIDIA/CUDA-10.1/lib -L/Users/user/magma-2.4.0/lib -L/${HOME}/opt/lib
INCDIRS = -I/opt/intel/mkl/include -I/Developer/NVIDIA/CUDA-10.1/include -I/Users/user/magma-2.4.0/include 


#all: ${exes}

${exes}: ${src}
	$(NVCC) $(src) -o $(exes) $(FLAGS) $(LIBDIRS) $(LIBS) $(INCDIRS)

clean:
	rm esm_c_magma.exe
