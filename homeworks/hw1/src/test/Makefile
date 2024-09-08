
CC ?= clang
CFLAGS ?= -w -fsanitize=address -fsanitize=undefined -fsanitize=fuzzer
LDFLAGS ?= -I $(LIB_FUZZING_ENGINE)
LIBS ?=

.PHONY:	file_harness
file_harness:
	$(CC) $(CFLAGS) -D ___TEST___ -o filein_harness filein_harness.c $(LDFLAGS) $(LIBS)
.PHONY: clean
clean:
	rm filein_harness
