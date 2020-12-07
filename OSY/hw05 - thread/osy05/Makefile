CC = gcc
CFLAGS = -Wall -g -O2
EXTRA_CFLAGS ?= -fsanitize=address -fno-omit-frame-pointer

prod-cons: prodcons.c
	$(CC) $(CFLAGS) $(EXTRA_CFLAGS) prodcons.c -lpthread -o prod-cons

test: 
	make
	./prod-cons
	make clean

tar:
	make clean
	tar czf osy05.tgz Makefile *.[ch]
	tar tf osy05.tgz

clean:
	rm -f *.o prod-cons osy05.tgz