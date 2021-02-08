.PHONY = all time
CC := nvcc
CFLAGS := 
FLAGS := -lSDL2_image -lSDL2 -lGLU -lglut -lGL 
OBJFILES := gpu.o window.o 
TARGET := boid

all: ${TARGET}

${TARGET}: ${OBJFILES}
	${CC} -o ${TARGET} ${OBJFILES} ${FLAGS}

time :
	$(CC) $(CLFAGS) test.cu -o time

gpu.o : gpu.cu gpu.h const.h 
	${CC} $(CFLAGS) -c gpu.cu 