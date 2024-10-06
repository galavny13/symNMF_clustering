CC = gcc
CFLAGS = -std=c99 -Wall -Wextra -Werror -pedantic-errors -Wno-unused-parameter
LDFLAGS = -lm
PYTHON = python3

all: symnmf symnmfmodule.so

symnmf: symnmf.o
	$(CC) $(CFLAGS) -o symnmf symnmf.o $(LDFLAGS)

symnmf.o: symnmf.c symnmf.h
	$(CC) $(CFLAGS) -c symnmf.c

symnmfmodule.so: setup.py symnmfmodule.c symnmf.c symnmf.h
	$(PYTHON) setup.py build_ext --inplace

clean:
	rm -f symnmf symnmf.o
	rm -f symnmfmodule.cpython-*.so
	rm -rf build
	rm -rf __pycache__
	rm -rf *.egg-info
