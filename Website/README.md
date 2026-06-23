# ImageSR Web Studio

Professional Image Super Resolution web application with:

- React + TypeScript + Tailwind frontend
- FastAPI backend with model preloading on startup
- LR/HR slider comparison + side-by-side comparison
- Download generated SR output
- Optional quality evaluation metrics
- Processing timeline states and recent run history

## Project structure

- `src/` React frontend
- `main.py` FastAPI backend API
- `requirements-backend.txt` Python dependencies for backend

## Run frontend

```bash
npm install
npm run dev
```

Frontend runs on `http://127.0.0.1:5173`.

## Run backend

```bash
pip install -r requirements-backend.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Backend runs on `http://127.0.0.1:8000`.

## API endpoints

- `GET /api/health` - backend status and loaded model info
- `POST /api/enhance` - enhance image
  - form-data fields:
    - `file` (image)
    - `return_metrics` (`true` / `false`)

## Notes

- Backend runs in fixed `4x` mode and attempts to load `RealESRGAN_x4plus` on startup.
- If Real-ESRGAN dependencies or weights are unavailable, backend auto-falls back to high-quality bicubic mode so the app still works.
- Vite is configured to proxy `/api` requests to `http://127.0.0.1:8000` during development.
