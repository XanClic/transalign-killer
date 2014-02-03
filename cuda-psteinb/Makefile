NVCC ?= nvcc
CXX ?= g++

ifeq ($(dbg),1)
CXXFLAGS += -g -G
else
CXXFLAGS += -O3 -use_fast_math --compiler-options '-O3 -march=native' 
endif

NVFLAGS ?= -gencode arch=compute_35,code=sm_35 -I/opt/cuda/5.5.22/include
SRCFILES =$(wildcard *.cu)
APPTARGETS=$(patsubst %.cu,%,$(SRCFILES))

all: $(APPTARGETS)

$(APPTARGETS): % : %.cu
	$(NVCC) $< $(CXXFLAGS) $(NVFLAGS) -o $@ 
