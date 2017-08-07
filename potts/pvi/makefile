# Compiler
CC=gcc

# Options
SOURCES=src/lib/twister.c src/lib/lbfgs.c src/pvi.c src/bayes.c src/inference.c
GCCFLAGS=-std=c99 -lm -O3 -msse4.2
CLANGFLAGS=-lm -Wall -Ofast -msse4.2

all:
	gcc $(SOURCES) -o bin/pvi $(GCCFLAGS)

all-dev:
	gcc $(SOURCES) -o bin/pvi $(GCCFLAGS) -Wall

all-openmp:
	gcc $(SOURCES) -o bin/pvi -fopenmp $(GCCFLAGS)

all-openmp32:
	gcc $(SOURCES) -o bin/pvi -fopenmp $(GCCFLAGS) -D USE_FLOAT

all-mac:
	clang $(SOURCES) -o bin/pvi $(CLANGFLAGS)

all-mac32:
	clang $(SOURCES) -o bin/pvi $(CLANGFLAGS) -D USE_FLOAT

clean:
	rm -rf bin/*