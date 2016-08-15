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

OBJECTS		= imgfeatures.o \
				kdtree.o \
				minpq.o \
				sift.o \
				utils.o \
				xform.o

OBJDIR		= bin
SRCDIR		= src

all: dirs $(addprefix $(OBJDIR)/, $(OBJECTS)) traffic

dirs: $(OBJDIR)

$(OBJDIR):
	mkdir -p $@

traffic: $(addprefix $(OBJDIR)/, $(OBJECTS)) traffic.cpp
	$(COMPILECPP) traffic -c traffic.cpp
	$(COMPILECPP) $@ $^ $(LIBS)

$(OBJDIR)/%.o: $(addprefix $(SRCDIR)/, %.c)
	$(COMPILEC) $@ -c $<

$(OBJDIR)/%.o: $(addprefix $(SRCDIR)/, %.cpp)
	$(COMPILECPP) $@ -c $<

clean:
	rm -rfv $(OBJDIR) traffic speed octave-workspace