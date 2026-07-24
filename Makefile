ARCH   ?= sm_86
NVCC   ?= nvcc
CFLAGS := -O3 -arch=$(ARCH) -lineinfo

CUDA_SRC := $(wildcard src/cuda/*/*.cu)
CUDA_BIN := $(patsubst src/cuda/%.cu,build/%,$(CUDA_SRC))

.PHONY: all clean ncu
all: $(CUDA_BIN)

build/%: src/cuda/%.cu | build
	mkdir -p $(dir $@)
	$(NVCC) $(CFLAGS) $< -o $@ -lcublas

build:
	mkdir -p build

# usage: make ncu BIN=build/gemm/02_tiled
ncu: $(BIN)
	ncu --set full --export ncu/$(notdir $(BIN)) --force-overwrite $(BIN)

clean:
	rm -rf build
