# Group:
# Joshua Taing, Max Mazal

FLAGS= -DDEBUG
LIBS= -lm
ALWAYS_REBUILD=makefile
#NVCC= nvcc  #used for CUDA code
#NVCCFLAGS = -DDEBUG --device-debug

nbody: nbody.o compute.o
	nvcc $(FLAGS) $^ -o $@ $(LIBS)
nbody.o: nbody.cu planets.h config.h vector.h $(ALWAYS_REBUILD)
	nvcc $(FLAGS) -c $< 
compute.o: compute.cu config.h vector.h $(ALWAYS_REBUILD)
	nvcc $(FLAGS) -c $< 
clean:
	rm -f *.o nbody 
