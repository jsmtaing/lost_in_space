# Group:
# Joshua Taing, Max Mazal

FLAGS= -DDEBUG
LIBS= -lm
ALWAYS_REBUILD=makefile
NVCC= nvcc  #used for CUDA code

nbody: nbody.o compute.o
	$(NVCC) $(FLAGS) $^ -o $@ $(LIBS)
nbody.o: nbody.cu planets.h config.h vector.h $(ALWAYS_REBUILD)
	gcc $(FLAGS) -c $< 
compute.o: compute.cu config.h vector.h $(ALWAYS_REBUILD)
	$(NVCC) $(FLAGS) -c $< 
clean:
	rm -f *.o nbody 
