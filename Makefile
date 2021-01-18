CC=g++
CFLAGS= -O3

BUILD_DIR=build
SRC_DIR=src
INCLUDE_DIR=./include
SOURCES := $(shell find $(SRC_DIR) -name '*.cpp')

$(info $(shell mkdir -p $(BUILD_DIR)))

default: all

all: compile run

compile:
	$(CC) -o $(BUILD_DIR)/main -I$(INCLUDE_DIR) $(SOURCES) $(CFLAGS) 

.PHONY: clean

run:
	@printf "\n** Testing\n\n"
	./build/main

clean:
	rm -rf $(BUILD_DIR)