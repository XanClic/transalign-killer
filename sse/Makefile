CC ?= gcc
CXX ?= g++
ASM = fasm
CFLAGS ?= -O3 -march=native
CFLAGS += -std=c11 -Wall -Wextra -pedantic
CXXFLAGS ?= -O3 -march=native
CXXFLAGS += -std=c++11 -Wall -Wextra -pedantic
LIBS =
RM = rm -f

OBJECTS = $(patsubst %.c,%.o,$(wildcard *.c)) $(patsubst %.cpp,%.o,$(wildcard *.cpp)) $(patsubst %.asm,%.o,$(wildcard *.asm))

.PHONY: all clean

all: transalign_killer

transalign_killer: $(OBJECTS)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.asm
	$(ASM) $< $@

clean:
	$(RM) transalign_killer $(OBJECTS)
