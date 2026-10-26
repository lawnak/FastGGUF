# fastgguf — build with GNU make (Linux, macOS, or Windows with MSYS2 / w64devkit)
#
#   make              optimized for THIS machine (-march=native)
#   make portable     runs on any x86-64 CPU: carries AVX-512 VNNI, AVX-VNNI, AVX2 and plain
#                     x86-64 builds of the engine and starts the fastest one the CPU supports
#   make selftest     build and run the kernel self-test
#   make clean

ifeq ($(origin CC),default)
  CC = gcc
endif
ARCH    ?= -march=native
# OpenMP (multi-threading) when the compiler has it; otherwise a single-threaded build
ifeq ($(origin OMP),undefined)
  ifneq ($(wildcard $(shell $(CC) -print-file-name=libgomp.a) $(shell $(CC) -print-file-name=libgomp.dll.a)),)
    OMP = -fopenmp
  else
    $(warning OpenMP library not found for $(CC): building single-threaded)
    OMP =
  endif
endif
CFLAGS  ?= -O3 $(ARCH) $(OMP) -Wall -Wno-unused-function
SRC      = fastgguf.c

ifeq ($(OS),Windows_NT)
  EXE     = .exe
  LDLIBS  = -lm -lws2_32
  # fully static .exe when the toolchain has a static OpenMP library (w64devkit, mingw-w64);
  # MSYS2 ships libgomp only as a DLL, so then keep libgomp-1.dll next to the .exe
  ifneq ($(wildcard $(shell $(CC) -print-file-name=libgomp.a)),)
    LDFLAGS = -static
  else
    LDFLAGS = -static-libgcc
  endif
  # MinGW gcc does not always keep the stack 32/64-byte aligned for AVX spills (gcc bug 54412);
  # this assembler option turns the aligned vector moves into unaligned ones, which are just as fast
  ifneq ($(shell echo "int x;" | $(CC) -Wa,-muse-unaligned-vector-move -x c -c -o /dev/null - 2>/dev/null && echo ok),)
    CFLAGS += -Wa,-muse-unaligned-vector-move
    PORT_WIN = -Wa,-muse-unaligned-vector-move
  endif
else
  EXE     =
  LDLIBS  = -lm -ldl
  LDFLAGS =
endif

BINS = fastgguf$(EXE) fastgguf-server$(EXE) fastgguf-console$(EXE)

.PHONY: all portable selftest clean

all: $(BINS)

# one binary with every command: chat / serve / inspect / dump / gpu / selftest
fastgguf$(EXE): $(SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

# server only: fastgguf-server model.gguf [model2.gguf ...] [port] [ctx]
fastgguf-server$(EXE): $(SRC)
	$(CC) $(CFLAGS) -DFG_SERVER_MAIN -o $@ $< $(LDFLAGS) $(LDLIBS)

# terminal chat only: fastgguf-console model.gguf [ctx] [options]
fastgguf-console$(EXE): $(SRC)
	$(CC) $(CFLAGS) -DFG_CONSOLE_MAIN -o $@ $< $(LDFLAGS) $(LDLIBS)

# portable: fastgguf.c compiled once per instruction set (FG_VARIANT), dispatch.c picks one at
# start-up (FASTGGUF_CPU=avx512|avxvnni|avx2|sse2 forces one)
PORT_FLAGS = -O3 $(OMP) -Wall -Wno-unused-function $(PORT_WIN)
PORT_OBJS  = port-avx512.o port-avxvnni.o port-avx2.o port-sse2.o

portable: $(PORT_OBJS)
	$(CC) $(PORT_FLAGS) -o fastgguf$(EXE) dispatch.c $(PORT_OBJS) $(LDFLAGS) $(LDLIBS)
	$(CC) $(PORT_FLAGS) -DFG_SERVER_MAIN -o fastgguf-server$(EXE) dispatch.c $(PORT_OBJS) $(LDFLAGS) $(LDLIBS)
	$(CC) $(PORT_FLAGS) -DFG_CONSOLE_MAIN -o fastgguf-console$(EXE) dispatch.c $(PORT_OBJS) $(LDFLAGS) $(LDLIBS)
	rm -f $(PORT_OBJS)

port-avx512.o: $(SRC)
	$(CC) $(PORT_FLAGS) -march=x86-64-v4 -mavx512vnni -DFG_VARIANT=avx512 -c -o $@ $<
port-avxvnni.o: $(SRC)
	$(CC) $(PORT_FLAGS) -march=x86-64-v3 -mavxvnni -DFG_VARIANT=avxvnni -c -o $@ $<
port-avx2.o: $(SRC)
	$(CC) $(PORT_FLAGS) -march=x86-64-v3 -DFG_VARIANT=avx2 -c -o $@ $<
port-sse2.o: $(SRC)
	$(CC) $(PORT_FLAGS) -march=x86-64 -DFG_VARIANT=sse2 -c -o $@ $<

selftest: fastgguf$(EXE)
	./fastgguf$(EXE) selftest

clean:
	rm -f $(BINS) $(PORT_OBJS)
