RUN := a.out

all: $(RUN)

$(RUN): main.cu
	nvcc $< -o $@ -arch=sm_30 -Xcompiler "-fopenmp" -I ../common

run: $(RUN)
	./$(RUN)

clean:
	rm -f $(RUN)

.PHONY: clean
