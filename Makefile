.PHONY: dev-api dev-ui test docker-up docker-down

dev-api:
	cd backend && uvicorn app.main:app --reload

dev-ui:
	cd frontend && npm run dev

test:
	cd backend && pytest

docker-up:
	docker compose up --build

docker-down:
	docker compose down

