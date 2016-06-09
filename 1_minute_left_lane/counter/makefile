CPPC		= g++
CPPFLAGS	= `pkg-config --cflags opencv` \
				-pedantic \
				-Wall \
				-Werror \
				-std=c++11 \
				-O3 \
				-g \
				-o

LIBS		= `pkg-config --libs opencv` \
				-larmadillo \

COMPILECPP	= $(CPPC) $(CPPFLAGS)

OBJECTS		= count.o

all: $(OBJECTS) count

count: $(OBJECTS)
	$(COMPILECPP) $@ $^ $(LIBS)

%.o: %.cpp
	$(COMPILECPP) $@ -c $<

clean:
	rm -rfv *.o count