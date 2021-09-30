SHELL := /bin/bash

default: neural.go
	go build -o neural neural.go

run:
	go run neural.go

test: neural.go neural_test.go
	go test neural.go neural_test.go
	
clean:
	rm -f neural
