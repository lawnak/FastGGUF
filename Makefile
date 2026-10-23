# fastgguf — build with GNU make (Linux, macOS, or Windows with MSYS2 / w64devkit)
#
#   make              optimized for THIS machine (-march=native)
#   make portable     runs on any x86-64 CPU with AVX2 (Haswell / Zen 1 and newer)
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

portable:
	$(MAKE) -B ARCH="-march=x86-64-v3"

selftest: fastgguf$(EXE)
	./fastgguf$(EXE) selftest

clean:
	rm -f $(BINS)
