SHELL := /bin/bash
.ONESHELL:

QMD := $(wildcard *.qmd)
IPYNB := $(QMD:%.qmd=build/%.ipynb)
VENV := .venv

notebooks: $(IPYNB)
	cp *.py build/
	cp *.ipynb build/
	cp requirements.txt build/
	cp README.md build/

build/%.ipynb: %.qmd
	. $(VENV)/bin/activate
	mkdir -p build
	quarto convert $< -o $@

clean:
	rm -rf build

.PHONY: all clean
