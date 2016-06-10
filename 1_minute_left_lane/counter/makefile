CC			= gcc
CFLAGS		= -pedantic \
				-I$(INC_DIR) `pkg-config --cflags opencv gtk+-2.0` \
				-Werror \
				-std=gnu99 \
				-O3 \
				-g \
				-o

CPPC		= g++
CPPFLAGS	= `pkg-config --cflags opencv` \
				-Wall \
				-Werror \
				-Wno-write-strings \
				-std=c++11 \
				-O3 \
				-g \
				-o

LIBS		= `pkg-config --libs opencv` -lm `pkg-config --libs opencv gtk+-2.0`

COMPILEC	= $(CC) $(CFLAGS)
COMPILECPP	= $(CPPC) $(CPPFLAGS)

OBJECTS		= count.o imgfeatures.o utils.o sift.o kdtree.o minpq.o xform.o

all: $(OBJECTS) count

count: $(OBJECTS)
	$(COMPILECPP) $@ $^ $(LIBS)

%.o: %.cpp
	$(COMPILECPP) $@ -c $<

%.o: %.c
	$(COMPILEC) $@ -c $<

clean:
	rm -rfv *.o count