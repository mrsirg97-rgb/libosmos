CC := gcc
CFLAGS = -Wall -Wextra -Werror -Iinclude -Isrc/model -O2 -g
LDLIBS := -lpthread

AR := ar
ARFLAGS := rcs

SRC_DIR  := src
TEST_DIR := tests
OBJ_DIR  := build
BIN_DIR  := bin
LIB_DIR  := lib

SRCS := $(shell find $(SRC_DIR) -name '*.c')
OBJS := $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(SRCS))
TESTS := $(patsubst $(TEST_DIR)/%.c, $(BIN_DIR)/%, $(wildcard $(TEST_DIR)/*.c))

LIB := $(LIB_DIR)/libsubstrate.a

all: $(LIB)

$(LIB): $(OBJS) | $(LIB_DIR)
	$(AR) $(ARFLAGS) $@ $(filter %.o,$^)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

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
	$(MAKE) CFLAGS="-Wall -Wextra -Werror -Iinclude -Isrc/model -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all" LDLIBS="-lpthread -fsanitize=address,undefined" tests

tsan:
	$(MAKE) clean
	$(MAKE) CFLAGS="-Wall -Wextra -Werror -Iinclude -Isrc/model -O1 -g -fsanitize=thread" LDLIBS="-lpthread -fsanitize=thread" tests

.PHONY: all clean tests asan tsan
