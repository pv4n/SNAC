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

LIBS		= `pkg-config --libs opencv` \
				`pkg-config --libs opencv gtk+-2.0` \
				-lm

COMPILEC	= $(CC) $(CFLAGS)
COMPILECPP	= $(CPPC) $(CPPFLAGS)

OBJECTS		= depth.o \
				imgfeatures.o \
				kdtree.o \
				minpq.o \
				sift.o \
				utils.o \
				xform.o

all: $(OBJECTS) depth

depth: $(OBJECTS)
	$(COMPILECPP) $@ $^ $(LIBS)

%.o: %.c
	$(COMPILEC) $@ -c $<

%.o: %.cpp
	$(COMPILECPP) $@ -c $<

clean:
	rm -rfv *.o depth