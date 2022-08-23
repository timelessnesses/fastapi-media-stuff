all:
	black .
	isort .
	autoflake . --remove-unused-variables  --remove-all-unused-imports --expand-star-imports