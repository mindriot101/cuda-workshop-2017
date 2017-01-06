RUN := a.out

all: $(RUN)

$(RUN): main.cu
	nvcc $< -o $@ -arch=sm_30 -I ../common

run: $(RUN)
	./$(RUN)

clean:
	rm -f $(RUN)

.PHONY: clean
