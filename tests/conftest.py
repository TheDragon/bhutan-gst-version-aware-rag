import os

# Keep the test suite offline and deterministic even when a Gemini key is in .env.
os.environ["GST_USE_GEMINI"] = "0"
