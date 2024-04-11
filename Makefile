run:
	python3 classifier.py

install:
	pip install -r requirements.txt

build:
	python3 setup.py build bdist_wheel

clear:
	rd /s /q build
	re /s /q dist
	rd /s /q Artificial-neural-network.egg-info