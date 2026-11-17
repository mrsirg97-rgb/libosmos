# the CUDA build (DESIGN.md, "a second GPU: CUDA"): the same kernels (kernels/*.hip, through kernels/port.h) and the
# same host files (src/**/hip_*.c, the HIP runtime's names mapped by include/osmos/gpu/cuda/hip/) for sm_120, by
# clang's CUDA mode over the CUDA toolkit - hipcc's own frontend, so _Float16, the vector types and
# __builtin_bit_cast read as they do on HIP. its objects under build/cuda, its library lib/libosmos_cuda.a.
#   make cuda               the library
#   make cuda-tests         the shared GPU tests (CUDA_TESTS), built and run: exit 77 a skip
# not in it (each its own ask): test_sleep_gpu (AMD's s_sleep itself), the NPU, hsa's memory pools (src/mem/hip_slots.c: src/mem/cuda_slots.c in its place), the families' MoE paths' tests
CUDA ?= /usr/local/cuda
CLANG_CUDA ?= clang++-20
CUDA_ARCH ?= sm_120
CUDA_OBJ := $(OBJ_DIR)/cuda
CUDA_SHIM := include/osmos/gpu/cuda
# the toolkit clang reads: CUDA's own, but bin/fatbinary tools/cuda/fatbinary (clang 20 still writes the --image
# option CUDA 13 dropped); every other entry a link to $(CUDA)
CUDA_VIEW := $(CUDA_OBJ)/toolkit
export OSMOS_CUDA_REAL := $(CUDA)
CUDA_DEBUG ?=
CUDA_GPUFLAGS = -x cuda --cuda-path=$(CUDA_VIEW) --cuda-gpu-arch=$(CUDA_ARCH) -O2 $(CUDA_DEBUG) -ffp-contract=off -Wall -Werror \
	-Wno-unknown-cuda-version -Wno-pass-failed -isystem $(CUDA_SHIM)/compat -I$(CUDA_SHIM) -Iinclude -I$(KERNEL_DIR)
CUDA_LINK = -L$(CUDA)/lib64 -lcudart -Wl,-rpath,$(CUDA)/lib64 -lpthread -lm

CUDA_HOST_SRCS := $(filter-out $(SRC_DIR)/mem/hip_slots.c,$(filter-out $(SRC_DIR)/npu/%,$(shell find $(SRC_DIR) -name 'hip_*.c'))) \
	$(SRC_DIR)/npu/hip_npu_none.c $(SRC_DIR)/mem/cuda_slots.c
CUDA_HOST_OBJS := $(patsubst $(SRC_DIR)/%.c,$(CUDA_OBJ)/%.o,$(CUDA_HOST_SRCS))
# q4x_qsa.hip (Qwen-Next's indexer) out: its tiles' 57.6 KB of static shared memory past CUDA's 48 KB (RDNA's LDS
# is 64 KB) - dynamic shared memory when Qwen-Next comes to CUDA. -Wno-pass-failed: an unroll clang's CUDA pipeline
# declines (the MoE routers') is a speed, not a result
CUDA_KERNEL_SRCS := $(filter-out $(KERNEL_DIR)/q4x_qsa.hip,$(KERNEL_SRCS))
CUDA_KERNEL_OBJS := $(patsubst $(KERNEL_DIR)/%.hip,$(CUDA_OBJ)/kernels/%.o,$(CUDA_KERNEL_SRCS))
CUDA_LIB := $(LIB_DIR)/libosmos_cuda.a
CUDA_TESTS ?= test_kvfmt_gpu test_mv_gpu test_mv_f32_wide_gpu test_mm_gpu test_mmq_gpu test_q8_gpu test_quant_gpu \
	test_attn_gpu test_select_gpu test_argmax_gpu test_fuse_gpu test_copy_gpu test_gated_mean_gpu \
	test_wait_gpu test_q4x_attn_gpu test_q4x_deltanet_gpu test_q4x_conv_sum_gpu test_q35_gpu test_q35_lsa_gpu test_lsa_gpu test_q35_attn_gpu test_mm_fp4_gpu test_lfm3_rope_gpu
CUDA_TEST_BINS := $(patsubst %,$(BIN_DIR)/cuda/%,$(CUDA_TESTS))

$(CUDA_VIEW)/.made: tools/cuda/fatbinary
	rm -rf $(CUDA_VIEW) && mkdir -p $(CUDA_VIEW)/bin
	for e in $(CUDA)/*; do [ "$$(basename $$e)" = bin ] || ln -s $$e $(CUDA_VIEW)/; done
	for e in $(CUDA)/bin/*; do [ "$$(basename $$e)" = fatbinary ] || ln -s $$e $(CUDA_VIEW)/bin/; done
	ln -s $(abspath tools/cuda/fatbinary) $(CUDA_VIEW)/bin/fatbinary
	touch $@

$(CUDA_OBJ)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(DEPFLAGS) -I$(CUDA_SHIM) -I$(CUDA)/include -Isrc/run -I$(KERNEL_DIR) -c $< -o $@

$(CUDA_OBJ)/kernels/%.o: $(KERNEL_DIR)/%.hip $(KERNEL_HDRS) $(CUDA_VIEW)/.made
	@mkdir -p $(dir $@)
	$(CLANG_CUDA) $(CUDA_GPUFLAGS) $(DEPFLAGS) -c $< -o $@

$(CUDA_LIB): $(CUDA_HOST_OBJS) $(CUDA_KERNEL_OBJS) | $(LIB_DIR)
	rm -f $@
	$(AR) $(ARFLAGS) $@ $^

$(BIN_DIR)/cuda/%: $(TEST_DIR)/gpu/%.hip $(LIB) $(CUDA_LIB) $(CUDA_VIEW)/.made
	@mkdir -p $(dir $@) $(CUDA_OBJ)/tests/gpu
	$(CLANG_CUDA) $(CUDA_GPUFLAGS) $(DEPFLAGS) -MF $(CUDA_OBJ)/tests/gpu/$*.d $< -L$(LIB_DIR) -losmos_cuda -losmos \
		$(CUDA_LINK) -o $@

cuda: $(LIB) $(CUDA_LIB)

cuda-tests-build: $(CUDA_TEST_BINS)

cuda-tests: $(CUDA_TEST_BINS)
	@for t in $(CUDA_TEST_BINS); do \
		echo "→ $$t"; \
		$$t; rc=$$?; [ $$rc -eq 0 ] || [ $$rc -eq 77 ] || exit 1; \
	done

.PHONY: cuda cuda-tests cuda-tests-build

# the FP4 matrix cores' instructions are sm_120a's (the arch-specific target), not sm_120's: that file alone
$(CUDA_OBJ)/kernels/mm_fp4.o: CUDA_ARCH = sm_120a

# the FP4 matmul alone as a shared library, for Python (incarnate's train/fp4.py, ctypes): `make fp4-so`
FP4_SO := $(LIB_DIR)/libosmos_fp4.so
$(FP4_SO): $(KERNEL_DIR)/mm_fp4.hip $(KERNEL_HDRS) $(CUDA_VIEW)/.made | $(LIB_DIR)
	$(CLANG_CUDA) $(filter-out --cuda-gpu-arch=%,$(CUDA_GPUFLAGS)) --cuda-gpu-arch=sm_120a -fPIC -shared $< -o $@ $(CUDA_LINK)
fp4-so: $(FP4_SO)
.PHONY: fp4-so
