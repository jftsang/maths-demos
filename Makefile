SHELL := /bin/bash
.ONESHELL:

QMD := $(wildcard *.qmd)
IPYNB := $(QMD:%.qmd=build/%.ipynb)
VENV := .venv

all: notebooks

notebooks:
	mkdir -p build
	cp *.py build/
	quarto render
	cp requirements.txt build/
	cp README.md build/

build/%.ipynb: %.qmd
	mkdir -p build
	quarto render $<

clean:
	rm -rf build

.PHONY: all clean
