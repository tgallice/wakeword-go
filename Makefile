.PHONY: build test vet lint bench oracle check

build:
	CGO_ENABLED=0 go build ./...

test:
	CGO_ENABLED=0 go test ./...

vet:
	go vet ./...

lint:
	golangci-lint run ./...

bench:
	CGO_ENABLED=0 go test -run '^$$' -bench . -benchmem ./...

oracle:
	@if [ ! -d tools/oracle/.venv ]; then python3 -m venv tools/oracle/.venv; fi
	tools/oracle/.venv/bin/pip install -q -r tools/oracle/requirements.txt
	cd tools/oracle && .venv/bin/python gen_model_parity.py
	cd tools/oracle && .venv/bin/python gen_frontend_golden.py

check: build vet test
