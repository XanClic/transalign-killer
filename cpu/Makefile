CXX ?= g++
CXXFLAGS ?= -O3 -march=native
RM = rm -f

.PHONY: all clean

all: transalign_killer

clean:
	$(RM) transalign_killer

transalign_killer: transalign_killer.cpp
	$(CXX) $(CXXFLAGS) -std=c++11 $^ -o $@
