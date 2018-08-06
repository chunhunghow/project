.PHONY:init,prepare_dev

.DEFAULT: help
help:
	@echo "init"
	@echo "		install dependencies"
	@echo "prepare_dev"
	@echo "		setup project env using pipenv"
init:
	pipenv shell python=3.5.2
	pipenv install -r requirements.txt

prepare_dev:
	sudo apt-get -y install python3.5 python3-pip
	pip install pipenv
	
