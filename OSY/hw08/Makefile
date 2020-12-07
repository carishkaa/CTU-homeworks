CC = gcc
CFLAGS = -nostdlib -nostdinc -static -m32 -Wall -g -O2

hexconv: hexconv.c
	$(CC) $(CFLAGS) hexconv.c -o hexconv

test: 
	make hexconv
	./hexconv
	make clean


clean:
	rm -f *.o hexconv