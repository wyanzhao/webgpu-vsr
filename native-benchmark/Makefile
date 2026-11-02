# webgpu-vsr native Metal benchmark
#
#   make            export model specs/weights, compile kernels and the CLI into build/
#   make metallib   compile shaders/*.metal -> build/vsr_kernels.metallib
#   make bench      build, then run the full benchmark table (--all)
#   make validate   build, then run every correctness check (exit status != 0 on failure)
#   make clean      remove build/

BUILD    := build
SDK      := macosx
METAL    := xcrun -sdk $(SDK) metal
METALLIB := xcrun -sdk $(SDK) metallib
SWIFTC   := xcrun -sdk $(SDK) swiftc
PYTHON   := python3

METAL_FLAGS := -std=metal3.2 -O3 -fno-fast-math -ffp-contract=off -Wall -Werror
SWIFT_FLAGS := -O -wmo -swift-version 5 -warnings-as-errors

SHADERS  := shaders/vsr_kernels.metal
SOURCES  := $(wildcard Sources/*.swift)
BIN      := $(BUILD)/vsr-bench
LIB      := $(BUILD)/vsr_kernels.metallib
SPECS    := $(BUILD)/models/low.json $(BUILD)/models/medium.json $(BUILD)/models/high.json $(BUILD)/models/ultra.json
PACKED   := $(BUILD)/weights/high.bin $(BUILD)/weights/high_manifest.json $(BUILD)/weights/ultra.bin $(BUILD)/weights/ultra_manifest.json

.PHONY: all metallib bench validate clean

all: $(SPECS) $(PACKED) $(LIB) $(BIN)

# One exporter run writes every spec and packed weight file. A pattern rule with several
# targets is the form make 3.81 (the macOS /usr/bin/make) runs ONCE for all of them (they share
# the stem $* = $(BUILD)); grouped targets (&:) need make >= 4.3 and are parsed by 3.81 as
# independent targets, which ran four concurrent exporters under -j.
%/models/low.json %/models/medium.json %/models/high.json %/models/ultra.json %/weights/high.bin %/weights/high_manifest.json %/weights/ultra.bin %/weights/ultra_manifest.json: tools/export_models.py ../extension/public/weights/low.bin ../extension/public/weights/medium.bin
	$(PYTHON) tools/export_models.py $*

metallib: $(LIB)

$(BUILD)/vsr_kernels.air: $(SHADERS)
	@mkdir -p $(BUILD)
	$(METAL) $(METAL_FLAGS) -c $< -o $@

$(LIB): $(BUILD)/vsr_kernels.air
	$(METALLIB) $< -o $@

$(BIN): $(SOURCES)
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SWIFT_FLAGS) $(SOURCES) -o $@ -framework Metal -framework ImageIO -framework CoreGraphics

bench: all
	./$(BIN) --all

validate: all
	./$(BIN) --validate

clean:
	rm -rf $(BUILD)
