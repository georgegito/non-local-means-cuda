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
	for number in 1 2 3 4 ; do \
        echo $$number ; \
    done
	for patchSize in 3 ; do \
		for filterSigma in 0.02 ; do \
			for patchSigma in 0.7 ; do \
				./build/main $$patchSize $$filterSigma $$patchSigma ;\
			done \
		done \
	done

clean:
	rm -rf $(BUILD_DIR)