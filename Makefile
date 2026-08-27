ARCH   ?= sm_86
NVCC   ?= nvcc
DEBUG  ?= 0

# release (default): full host+device opt, fast-math on the GPU math library.
# fast-math trades strict IEEE compliance for speed (reassociation, no denormals,
# approximate div/sqrt) — GemmKernel::validateGemm uses an atol+rtol check to account for this.
# debug (DEBUG=1): -G disables device-side optimization for cuda-gdb; no fast-math.
ifeq ($(DEBUG),1)
CFLAGS := -O0 -g -G -arch=$(ARCH)
else
CFLAGS := -O3 -arch=$(ARCH) -lineinfo -DNDEBUG --use_fast_math \
          -Xcompiler -O3,-march=native,-ffast-math
endif

CUDA_SRC := $(wildcard src/cuda/*/*.cu)
CUDA_BIN := $(patsubst src/cuda/%.cu,build/%,$(CUDA_SRC))
CUDA_HDR := $(wildcard src/cuda/*.cuh src/cuda/*/*.cuh)
FLAGS_STAMP := build/.cflags

.PHONY: all clean ncu
all: $(CUDA_BIN)

# depends on every header, not just its own .cu, so a shared-header change (e.g. gemmKernel.cuh)
# forces every binary to rebuild instead of make silently keeping stale ones by .cu mtime alone.
build/%: src/cuda/%.cu $(CUDA_HDR) $(FLAGS_STAMP) | build
	mkdir -p $(dir $@)
	$(NVCC) $(CFLAGS) $< -o $@ -lcublas

build:
	mkdir -p build

# touches only when CFLAGS actually changed, so switching DEBUG=0/1 (or ARCH) forces
# a rebuild of every binary instead of make silently reusing stale ones by mtime.
$(FLAGS_STAMP): FORCE | build
	@if [ ! -f $@ ] || [ "$$(cat $@)" != "$(CFLAGS)" ]; then \
		echo "$(CFLAGS)" > $@; \
	fi

.PHONY: FORCE
FORCE:

# usage: make ncu BIN=build/gemm/02_tiled
ncu: $(BIN)
	ncu --set full --export ncu/$(notdir $(BIN)) --force-overwrite $(BIN)

clean:
	rm -rf build
