CC := gcc
CFLAGS = -Wall -Wextra -Werror -Iinclude -Isrc/model -Isrc/mem -O2 -g
LDLIBS := -lpthread

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
HIP_SRCS := $(shell find $(SRC_DIR) -name 'hip_*.c')
HIP_OBJS := $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(HIP_SRCS))
GPU_TESTS := $(patsubst $(TEST_DIR)/gpu/%.hip, $(BIN_DIR)/gpu/%, $(wildcard $(TEST_DIR)/gpu/*.hip))
ROCM ?= /opt/rocm
HIPCC := $(ROCM)/bin/hipcc
HIP_ARCH ?= gfx1151
OBJS := $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(SRCS))
TESTS := $(patsubst $(TEST_DIR)/%.c, $(BIN_DIR)/%, $(wildcard $(TEST_DIR)/*.c))

LIB := $(LIB_DIR)/libsubstrate.a
HIP_LIB := $(LIB_DIR)/libsubstrate_hip.a

all: $(LIB)

$(LIB): $(OBJS) | $(LIB_DIR)
	$(AR) $(ARFLAGS) $@ $(filter %.o,$^)

$(HIP_LIB): $(HIP_OBJS) | $(LIB_DIR)
	$(AR) $(ARFLAGS) $@ $(filter %.o,$^)

$(OBJ_DIR)/%/hip_pinner.o: $(SRC_DIR)/%/hip_pinner.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -I$(ROCM)/include -D__HIP_PLATFORM_AMD__ -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(BIN_DIR)/gpu/%: $(TEST_DIR)/gpu/%.hip $(LIB) $(HIP_LIB)
	@mkdir -p $(dir $@)
	$(HIPCC) -O2 -Iinclude --offload-arch=$(HIP_ARCH) $< -L$(LIB_DIR) -lsubstrate_hip -lsubstrate -o $@

# the GPU's tests (a gfx1151 and the model): exit 77 is a skip
gpu-tests: $(GPU_TESTS)
	@for t in $(GPU_TESTS); do \
		echo "→ $$t"; \
		$$t; rc=$$?; [ $$rc -eq 0 ] || [ $$rc -eq 77 ] || exit 1; \
	done

$(BIN_DIR)/%: $(TEST_DIR)/%.c $(LIB) | $(BIN_DIR)
	$(CC) $(CFLAGS) $< -L$(LIB_DIR) -lsubstrate $(LDLIBS) -o $@

$(OBJ_DIR) $(BIN_DIR) $(LIB_DIR):
	mkdir -p $@

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
	$(MAKE) CFLAGS="-Wall -Wextra -Werror -Iinclude -Isrc/model -Isrc/mem -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all" LDLIBS="-lpthread -fsanitize=address,undefined" tests

tsan:
	$(MAKE) clean
	$(MAKE) CFLAGS="-Wall -Wextra -Werror -Iinclude -Isrc/model -Isrc/mem -O1 -g -fsanitize=thread" LDLIBS="-lpthread -fsanitize=thread" tests

.PHONY: all clean tests asan tsan gpu-tests
