CC := clang
CFLAGS := -Wall -lz 
OPTFLAGS := -O3
SRC := ops.c shapetracker.c tensor.c buffer.c utils.c dataloader.c
TEST_SRC := test/dot.c test/binary_ops.c test/unary_ops.c test/movement.c test/tensor_forward.c test/tensor_backward.c test/init.c test/test.c
MAIN_SRC := main.c
BIN := bin/test
MAIN_BIN := bin/main

.PHONY: test main clean

test: $(BIN)
	./$(BIN)

main: $(MAIN_BIN)
	./$(MAIN_BIN)

$(BIN): $(SRC) $(TEST_SRC)
	@mkdir -p bin
	$(CC) $(CFLAGS) $^ -o $@

$(MAIN_BIN): $(SRC) $(MAIN_SRC)
	@mkdir -p bin
	$(CC) $(CFLAGS) $(OPTFLAGS) $^ -o $@

clean:
	rm -f $(BIN) $(MAIN_BIN)
