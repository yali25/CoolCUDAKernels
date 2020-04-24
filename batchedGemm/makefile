PROG=batchGemm
CC=pgc++
CCU=nvcc -ccbin=${CC}

OBJS=main.o
OPTS=-ta=tesla:cc70 -acc -Minfo=accel -Minfo
INCLUDES=-I../common/inc
LIBS=-Mcudalib=cublas,cusparse

# in this compiler setup rdc=false
# is needed otherwise the code fails
# at runtime
%.o: %.cu
	${CCU} -arch=compute_70 -code=sm_70 -rdc=false -Xcompiler "${OPTS}" ${INCLUDES} -c $<
all:${PROG}
${PROG}:${OBJS}
	${CC} -Mcuda ${OPTS} ${LIBS}  -o $@ ${OBJS}

clean:
	rm -f ${PROG} main.o
