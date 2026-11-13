CC := gcc
# -MMD -MP: each object's headers recorded (build/*.d), so a changed header rebuilds what includes it
DEPFLAGS = -MMD -MP
CFLAGS = -Wall -Wextra -Werror -Iinclude -Isrc/model -Isrc/mem -O2 -g
LDLIBS := -lpthread -lm

AR := ar
ARFLAGS := rcs

SRC_DIR  := src
TEST_DIR := tests
OBJ_DIR  := build
BIN_DIR  := bin
LIB_DIR  := lib

# the host library is pure C; hip_*.c (the HIP runtime's C API) go into a library of their own, so no host test or
# sanitizer run links HIP
SRCS := $(shell find $(SRC_DIR) -name '*.c' ! -name 'hip_*.c')
# libevt (the operator's, C, MIT): the forward's reads and osmos-serve's loop - its sources built into libosmos.a with
# this library's flags, so every test and sanitizer run covers it too. the submodule third_party/libevt (git submodule
# update --init; LIBEVT=<dir> builds against another checkout)
LIBEVT ?= third_party/libevt
EVT_SRCS := $(shell find $(LIBEVT)/src -name '*.c')
EVT_OBJS := $(patsubst $(LIBEVT)/src/%.c,$(OBJ_DIR)/libevt/%.o,$(EVT_SRCS))
override CFLAGS += -I$(LIBEVT)/include
HIP_SRCS := $(shell find $(SRC_DIR) -name 'hip_*.c')
# the NPU (0.7.0, include/osmos/npu/npu.h): NPU=1 builds src/npu/npu_xrt.cpp (XRT, libxrt-dev) into the HIP
# library and links XRT into its binaries; else its stub (hip_npu_none.c), and no XRT anywhere
NPU ?= 0
ifeq ($(NPU),1)
HIP_SRCS := $(filter-out $(SRC_DIR)/npu/hip_npu_none.c,$(HIP_SRCS))
NPU_OBJS := $(OBJ_DIR)/npu/npu_xrt.o
NPU_LIBS := -lxrt_coreutil
endif
HIP_OBJS := $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(HIP_SRCS))
# the kernels (HIP C++, gfx1151 only) behind their C interfaces, into the HIP library; no contraction but the fmaf
# a kernel writes, so a sum's rounding is the source's and the same in every instantiation
KERNEL_DIR := kernels
KERNEL_SRCS := $(wildcard $(KERNEL_DIR)/*.hip)
KERNEL_OBJS := $(patsubst $(KERNEL_DIR)/%.hip,$(OBJ_DIR)/kernels/%.o,$(KERNEL_SRCS))
KERNEL_HDRS := $(wildcard $(KERNEL_DIR)/*.h)
GPU_TESTS := $(patsubst $(TEST_DIR)/gpu/%.hip, $(BIN_DIR)/gpu/%, $(wildcard $(TEST_DIR)/gpu/*.hip))
ROCM ?= /opt/rocm
HIPCC := $(ROCM)/bin/hipcc
# the HIP binaries find ROCm's libraries where they were built against (a RUNPATH: LD_LIBRARY_PATH still wins) - a
# ROCm outside the system's library path (a vendored SDK, ROCM=...) runs as built, not only /opt/rocm's in ld.so.conf
HIP_RUNPATH := -Wl,-rpath,$(ROCM)/lib -Wl,--enable-new-dtags
HIP_ARCH ?= gfx1151
OBJS := $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(SRCS))
TESTS := $(patsubst $(TEST_DIR)/%.c, $(BIN_DIR)/%, $(wildcard $(TEST_DIR)/*.c))

LIB := $(LIB_DIR)/libosmos.a
HIP_LIB := $(LIB_DIR)/libosmos_hip.a

all: $(LIB)

$(LIB): $(OBJS) $(EVT_OBJS) | $(LIB_DIR)
	$(AR) $(ARFLAGS) $@ $(filter %.o,$^)

$(OBJ_DIR)/libevt/%.o: $(LIBEVT)/src/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(DEPFLAGS) -c $< -o $@

# made afresh: ar adds and replaces members, never drops one - an NPU=1 build's npu_xrt.o beside the stub after
# NPU=0 (or the stub beside it) would be a duplicate symbol. NPU's value recorded: a change rebuilds the archive
$(HIP_LIB): $(HIP_OBJS) $(KERNEL_OBJS) $(NPU_OBJS) $(OBJ_DIR)/npu.$(NPU) | $(LIB_DIR)
	rm -f $@
	$(AR) $(ARFLAGS) $@ $(filter %.o,$^)

$(OBJ_DIR)/npu.$(NPU):
	@mkdir -p $(OBJ_DIR)
	rm -f $(OBJ_DIR)/npu.0 $(OBJ_DIR)/npu.1
	touch $@

$(OBJ_DIR)/npu/npu_xrt.o: $(SRC_DIR)/npu/npu_xrt.cpp
	@mkdir -p $(dir $@)
	$(HIPCC) -O2 -Wall -Werror $(DEPFLAGS) -Iinclude -I$(SRC_DIR)/npu/include -c $< -o $@

# the CPU lane (src/cpu): the GPU's sums repeated bit for bit - no contraction but the fmaf it writes (as the
# kernels'), the FMA and f16 conversion in hardware, and the kernels' shared headers (quant_grids.h, lane.h)
$(OBJ_DIR)/cpu/%.o: override CFLAGS += -ffp-contract=off -mavx2 -mfma -mf16c -I$(KERNEL_DIR)
# test_cpu_q8's and test_cpu_lane's replays of lane.h's contract: their products each rounded, as the kernels'
$(BIN_DIR)/test_cpu_q8: override CFLAGS += -ffp-contract=off -I$(KERNEL_DIR)
$(BIN_DIR)/test_cpu_lane: override CFLAGS += -ffp-contract=off -I$(KERNEL_DIR)
# test_qtype: the lane contract's ids (lane.h)
$(BIN_DIR)/test_qtype: override CFLAGS += -I$(KERNEL_DIR)

# the host code that calls the HIP runtime (hip_*.c): its headers
$(HIP_OBJS): CFLAGS += -I$(ROCM)/include -D__HIP_PLATFORM_AMD__ -Isrc/run -I$(KERNEL_DIR)

$(OBJ_DIR)/kernels/%.o: $(KERNEL_DIR)/%.hip $(KERNEL_HDRS)
	@mkdir -p $(dir $@)
	$(HIPCC) -O2 -ffp-contract=off -Wall -Werror $(DEPFLAGS) -Iinclude -I$(KERNEL_DIR) --offload-arch=$(HIP_ARCH) -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(DEPFLAGS) -c $< -o $@

$(BIN_DIR)/gpu/%: $(TEST_DIR)/gpu/%.hip $(LIB) $(HIP_LIB)
	@mkdir -p $(dir $@)
	@mkdir -p $(OBJ_DIR)/tests/gpu
	$(HIPCC) -O2 $(DEPFLAGS) -MF $(OBJ_DIR)/tests/gpu/$*.d -Iinclude -I$(KERNEL_DIR) --offload-arch=$(HIP_ARCH) $< -L$(LIB_DIR) -losmos_hip -losmos -lhsa-runtime64 $(NPU_LIBS) $(HIP_RUNPATH) -o $@

# the server (tools/serve): the engine over stdin / stdout, maya's serve protocol; --parallel's loop libevt's (in
# libosmos.a, as above)
SERVE := $(BIN_DIR)/osmos-serve
SERVE_OBJS := $(patsubst tools/serve/%.c,$(OBJ_DIR)/tools/serve/%.o,$(wildcard tools/serve/*.c))
$(OBJ_DIR)/tools/serve/%.o: tools/serve/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(DEPFLAGS) -c $< -o $@
$(SERVE): $(SERVE_OBJS) $(LIB) $(HIP_LIB)
	@mkdir -p $(dir $@)
	$(HIPCC) $(SERVE_OBJS) -L$(LIB_DIR) -losmos_hip -losmos -lhsa-runtime64 $(NPU_LIBS) -lpthread $(HIP_RUNPATH) -o $@
serve: $(SERVE)

# osmos-api (tools/api): the engine as an OpenAI-compatible server - osmos-serve its child; host C only
API := $(BIN_DIR)/osmos-api
API_OBJS := $(patsubst tools/api/%.c,$(OBJ_DIR)/tools/api/%.o,$(wildcard tools/api/*.c))
$(OBJ_DIR)/tools/api/%.o: tools/api/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(DEPFLAGS) -Isrc/text -c $< -o $@
$(API): $(API_OBJS) $(LIB)
	@mkdir -p $(dir $@)
	$(CC) $(API_OBJS) -L$(LIB_DIR) -losmos $(LDLIBS) -o $@
api: $(API)

# osmos-api against a fake engine (tests/api: the protocol's, scripted)
api-test: $(API)
	python3 tests/api/test_api.py

# requant_dspark (tools/requant): a drafter pack's experts requantized with ggml's quantizers (llama.cpp, MIT; its
# libggml-base built beside this checkout: cmake -S . -B build-ggml ... --target ggml-base)
GGML ?= ../llama.cpp
REQUANT := $(BIN_DIR)/requant_dspark
$(REQUANT): tools/requant/requant_dspark.c $(LIB) $(GGML)/build-ggml/ggml/src/libggml-base.a
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -I$(GGML)/ggml/include $< $(LIB) $(GGML)/build-ggml/ggml/src/libggml-base.a -lstdc++ -lm -lpthread -o $@
requant: $(REQUANT)

# the GPU's tests built, not run (CI: no gfx1151 there)
gpu-build: $(GPU_TESTS) $(SERVE)

# the GPU's tests (a gfx1151 and the model): exit 77 is a skip
gpu-tests: $(GPU_TESTS)
	@for t in $(GPU_TESTS); do \
		echo "→ $$t"; \
		$$t; rc=$$?; [ $$rc -eq 0 ] || [ $$rc -eq 77 ] || exit 1; \
	done

$(BIN_DIR)/%: $(TEST_DIR)/%.c $(LIB) | $(BIN_DIR)
	$(CC) $(CFLAGS) $(DEPFLAGS) -MF $(OBJ_DIR)/tests/$*.d $< -L$(LIB_DIR) -losmos $(LDLIBS) -o $@

$(OBJ_DIR) $(BIN_DIR) $(LIB_DIR):
	mkdir -p $@

$(TESTS): | $(OBJ_DIR)/tests

$(OBJ_DIR)/tests:
	mkdir -p $@

-include $(shell find $(OBJ_DIR) -name '*.d' 2>/dev/null)

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR) $(LIB_DIR)

tests: $(TESTS)
	@echo "running tests..."
	@for t in $(TESTS); do \
		echo "→ $$t"; \
		$$t || exit 1; \
	done

# no leaks, no faults: the same tests under AddressSanitizer + LeakSanitizer + UndefinedBehaviorSanitizer, then
# ThreadSanitizer
asan:
	$(MAKE) clean
	$(MAKE) CFLAGS="-Wall -Wextra -Werror -Iinclude -Isrc/model -Isrc/mem -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all" LDLIBS="-lpthread -lm -fsanitize=address,undefined" tests

tsan:
	$(MAKE) clean
	$(MAKE) CFLAGS="-Wall -Wextra -Werror -Iinclude -Isrc/model -Isrc/mem -O1 -g -fsanitize=thread" LDLIBS="-lpthread -lm -fsanitize=thread" tests

.PHONY: all clean tests asan tsan gpu-tests gpu-build serve requant api api-test
