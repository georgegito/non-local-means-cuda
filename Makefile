CC=g++
NVCC = nvcc
CFLAGS= -O3

BUILD_DIR=build
SRC_DIR=src
INCLUDE_DIR=./include
DATA_DIR=data/out
SOURCES := $(shell find $(SRC_DIR) -name '*.cu')

imageNum=0
patchSize=5
filterSigma=0.06
patchSigma=0.8
useGpu=0
useSharedMem=0

$(info $(shell mkdir -p $(BUILD_DIR)))
$(info $(shell mkdir -p $(DATA_DIR)))

default: test

test: compile run

compile:
	$(NVCC) -o $(BUILD_DIR)/main -I$(INCLUDE_DIR) $(SOURCES) $(CFLAGS) 

.PHONY: clean

run:
	@printf "\n** Testing\n\n"
	./build/main $(imageNum) $(patchSize) $(filterSigma) $(patchSigma) $(useGpu) $(useSharedMem)

clean:
	rm -rf $(BUILD_DIR)