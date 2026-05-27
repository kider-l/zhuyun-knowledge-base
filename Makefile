.PHONY: dev-up dev-down dev-api dev-ui test docker-up docker-down

dev-up:
	powershell -ExecutionPolicy Bypass -File .\dev-up.ps1

dev-down:
	powershell -ExecutionPolicy Bypass -File .\dev-down.ps1

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
