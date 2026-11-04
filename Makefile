ARCH ?= $(shell uname -m | sed 's/x86_64/x86/; s/aarch64/arm64/')
MULTIARCH := $(shell gcc -print-multiarch 2>/dev/null)
BPF_CFLAGS := -O2 -g -target bpf -D__TARGET_ARCH_$(ARCH) -I/usr/include/$(MULTIARCH) -Ibpf -Wall
BUILD := bpf/build

.PHONY: all bpf test clean
all: bpf
bpf: $(BUILD)/probes.bpf.o $(BUILD)/loader

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/probes.bpf.o: bpf/probes.bpf.c bpf/events.h | $(BUILD)
	clang $(BPF_CFLAGS) -c $< -o $@

$(BUILD)/loader: bpf/loader.c bpf/events.h | $(BUILD)
	gcc -O2 -Wall -Ibpf $< -o $@ -lbpf

test:
	pytest -q --tb=short

clean:
	rm -rf build bpf/build .pytest_cache
