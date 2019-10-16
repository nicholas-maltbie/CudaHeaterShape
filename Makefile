# Makefile

NVCC		= nvcc

NVCCFLAGS	=
LIBS		= -lGL -lGLU -lglut

CU_SRCS  	= heat.cu 
CU_SRCS_2D	= heat_2d.cu

OBJS 		= $(CU_SRCS:.cu=.o)
OBJS_2D 	= $(CU_SRCS_2D:.cu=.o)

TARGET 		= heat
TARGET_2D	= heat_2d

all: $(TARGET) $(TARGET_2D)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $< $(LIBS) -o $@

$(TARGET_2D): $(OBJS_2D)
	$(NVCC) $(NVCCFLAGS) $< $(LIBS) -o $@

.SUFFIXES: 

.SUFFIXES: .cu .o

.cu.o:
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<

clean:
	rm -rf $(TARGET) $(TARGET_2D) *.o
