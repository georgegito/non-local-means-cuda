CC=g++
CFLAGS= -O3

BUILD_DIR=build
SRC_DIR=src
INCLUDE_DIR=./include
DATA_DIR=data
SOURCES := $(shell find $(SRC_DIR) -name '*.cpp')

$(info $(shell mkdir -p $(BUILD_DIR)))
$(info $(shell mkdir -p $(DATA_DIR)))

default: test

test: compile run

all: compile run_all

compile:
	$(CC) -o $(BUILD_DIR)/main -I$(INCLUDE_DIR) $(SOURCES) $(CFLAGS) 

.PHONY: clean

run:
	@printf "\n** Testing\n\n"
	./build/main


run_all:
	@printf "\n** Testing\n\n"
	#FIX THIS
	$(shell ./scripts/run_all.sh)

clean:
	rm -rf $(BUILD_DIR)