CC = gcc
CFLAGS = -Wall -g -O2
EXTRA_CFLAGS ?= -fsanitize=address -fno-omit-frame-pointer

factory: factory.c
	$(CC) $(CFLAGS) $(EXTRA_CFLAGS) factory.c -lpthread -o factory

test: 
	make factory
	./factory < test.txt
	make clean

tar:
	make clean
	tar czf osy06.tgz Makefile *.[ch]
	tar tf osy06.tgz

clean:
	rm -f *.o factory osy06.tgz