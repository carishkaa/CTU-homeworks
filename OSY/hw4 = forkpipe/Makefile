CC = gcc
CFLAGS = -Wall -pedantic

all: forkpipe nsd

forkpipe: forkpipe.o
	$(CC) $(CFLAGS) -o forkpipe forkpipe.o

nsd: nd.o nsd.o nsd_main.c nsd.h nd.h
	$(CC) $(CFLAGS) -o nsd nd.o nsd.o nsd_main.c

test:
	make
	./forkpipe
	make clean

tar:
	make clean
	tar czf osy04.tgz Makefile *.[ch]

clean:
	rm -f *.o forkpipe nsd osy04.tgz
