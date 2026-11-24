# GST Lens backend

From the project root:

```powershell
.venv\Scripts\python.exe -m uvicorn app.backend.main:app --reload --port 8000
```

Interactive API documentation is available at `http://localhost:8000/docs`.

The first answer request lazily loads `google/flan-t5-base`. Set
`GST_GENERATOR_MODEL=extractive-evidence-v1` for a faster offline smoke test.
