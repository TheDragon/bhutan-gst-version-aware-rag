# GST Lens application architecture

## Frontend

React + Vite in `app/frontend`. The interface is intentionally broader than a
chatbot and includes Overview, Ask & Inspect, Compare Systems, Version Timeline,
and Corpus Library views.

## Backend

FastAPI in `app/backend/main.py`. The API reads the same validated artifacts as
the notebooks and exposes:

- `GET /api/health`
- `GET /api/stats`
- `GET /api/documents`
- `GET /api/timeline/{section}`
- `POST /api/search`
- `POST /api/answer`
- `POST /api/compare`

The retrieval engine and generator are lazy singletons so application startup
is fast and the large models load only when required.

## Research safeguards

Every answer contains evidence records with document, section, page, effective
period, retrieval scores, and official source URL. The interface displays the
prototype warning because amendment relationships and benchmark answers still
need group legal review.
