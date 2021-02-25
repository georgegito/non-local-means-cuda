CC=g++
NVCC = nvcc
CFLAGS= -O3

BUILD_DIR=build
SRC_DIR=src
INCLUDE_DIR=./include
DATA_DIR=data
SOURCES := $(shell find $(SRC_DIR) -name '*.cu')

$(info $(shell mkdir -p $(BUILD_DIR)))
$(info $(shell mkdir -p $(DATA_DIR)))

default: test

test: compile run

all: compile run_all

compile:
	$(NVCC) -o $(BUILD_DIR)/main -I$(INCLUDE_DIR) $(SOURCES) $(CFLAGS) 

.PHONY: clean

run:
	@printf "\n** Testing\n\n"
	./build/main 0 5 0.06 0.8 0 0


run_all:
	@printf "\n** Testing\n\n"
	#FIX THIS
	$(shell ./scripts/run_all.sh)

clean:
	rm -rf $(BUILD_DIR)