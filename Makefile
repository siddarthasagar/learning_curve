.PHONY: clean install

clean:
	poetry env remove --all
	rm -rf .venv

install: clean
	poetry install


