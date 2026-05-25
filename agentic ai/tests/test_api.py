"""
Unit Tests for FastAPI Backend API

Verifies:
  - GET /health endpoint (with mocked and real environment checks)
  - GET /images/{filepath} endpoint (successful serve, 404 not found, 400 path traversal)
  - POST /chat endpoint (mocking LangGraph response and validating contracts)
  - Ephemeral session management behavior
  - Error handling (500 internal errors)
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

# Add project root to python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.main import app
from api import session


class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def setUp(self):
        # Clear sessions before each test
        session._sessions.clear()

    def tearDown(self):
        # Clear sessions after each test
        session._sessions.clear()

    # =========================================================================
    #  GET /health tests
    # =========================================================================

    def test_health_endpoint(self):
        """Verify /health returns 200 and the correct JSON keys."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertIn("milvus", data)
        self.assertIn("gemini", data)
        self.assertEqual(data["status"], "ok")

    # =========================================================================
    #  GET /images/{filepath} tests
    # =========================================================================

    def test_images_endpoint_success(self):
        """Verify serving an existing image returns 200 and image/png type."""
        images_dir = PROJECT_ROOT / "data" / "images"
        png_files = list(images_dir.glob("**/*.png"))
        
        if png_files:
            rel_path = png_files[0].relative_to(images_dir).as_posix()
            response = self.client.get(f"/images/{rel_path}")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers.get("content-type"), "image/png")
        else:
            with patch("pathlib.Path.is_file", return_value=True), \
                 patch("fastapi.responses.FileResponse") as mock_file_response:
                mock_file_response.return_value = "fake-file-response"
                response = self.client.get("/images/arduino/fake_image.png")
                self.assertEqual(response.status_code, 200)

    def test_images_endpoint_404(self):
        """Verify GET /images/ with nonexistent file returns 404."""
        response = self.client.get("/images/arduino/nonexistent_image_12345.png")
        self.assertEqual(response.status_code, 404)
        self.assertIn("detail", response.json())
        self.assertIn("Image not found", response.json()["detail"])

    def test_images_endpoint_path_traversal(self):
        """Verify path traversal attempts are blocked/rejected."""
        response = self.client.get("/images/../api/main.py")
        self.assertIn(response.status_code, (400, 404))

    # =========================================================================
    #  POST /chat tests (Mocking LangGraph)
    # =========================================================================

    @patch("api.main.compiled_graph")
    def test_chat_text_only(self, mock_graph):
        """Verify ChatResponse format for text-only intent."""
        mock_state = {
            "query": "What is Arduino?",
            "session_id": "session-123",
            "session_history": [],
            "intent": "text",
            "final_answer": "Arduino is an open-source electronics platform.",
            "image_results": [],
            "sources": ["Arduino Intro.pdf"],
            "verification_status": "pass",
        }
        mock_graph.invoke = MagicMock(return_value=mock_state)

        payload = {
            "query": "What is Arduino?",
            "session_id": "session-123"
        }
        response = self.client.post("/chat", json=payload)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data["session_id"], "session-123")
        self.assertEqual(data["intent"], "text")
        self.assertEqual(data["answer"], "Arduino is an open-source electronics platform.")
        self.assertIsNone(data["image_url"])
        self.assertIsNone(data["image_caption"])
        self.assertEqual(data["sources"], ["Arduino Intro.pdf"])

        # Verify turn was appended to session
        history = session.get_history("session-123")
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0], {"role": "user", "content": "What is Arduino?"})
        self.assertEqual(history[1], {"role": "assistant", "content": "Arduino is an open-source electronics platform."})

    @patch("api.main.compiled_graph")
    def test_chat_both_with_images(self, mock_graph):
        """Verify ChatResponse has correct relative image_url for 'both' intent."""
        mock_state = {
            "query": "Show me Arduino Uno pins",
            "session_id": "session-456",
            "session_history": [],
            "intent": "both",
            "final_answer": "Here is the pinout diagram for Arduino Uno.",
            "image_results": [
                {
                    "image_path": "data/images/arduino/uno_pins.png",
                    "caption": "Arduino Uno Pinout Diagram",
                    "score": 0.85
                },
                {
                    "image_path": "data/images/arduino/uno_back.png",
                    "caption": "Arduino Uno Back View",
                    "score": 0.50
                }
            ],
            "sources": ["Arduino Uno Datasheet"],
            "verification_status": "pass",
        }
        mock_graph.invoke = MagicMock(return_value=mock_state)

        payload = {
            "query": "Show me Arduino Uno pins",
            "session_id": "session-456"
        }
        response = self.client.post("/chat", json=payload)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data["session_id"], "session-456")
        self.assertEqual(data["intent"], "both")
        self.assertEqual(data["answer"], "Here is the pinout diagram for Arduino Uno.")
        
        # Must pick the top scoring image and transform path to relative URL
        self.assertEqual(data["image_url"], "/images/arduino/uno_pins.png")
        self.assertEqual(data["image_caption"], "Arduino Uno Pinout Diagram")
        self.assertEqual(data["sources"], ["Arduino Uno Datasheet"])

    @patch("api.main.compiled_graph")
    def test_chat_clarification(self, mock_graph):
        """Verify clarification_question maps to 'answer' field when intent is 'clarify'."""
        mock_state = {
            "query": "it does not work",
            "session_id": "session-789",
            "session_history": [],
            "intent": "clarify",
            "clarification_question": "Which board or component are you trying to use?",
            "final_answer": None,
            "image_results": [],
            "sources": [],
            "verification_status": "pending",
        }
        mock_graph.invoke = MagicMock(return_value=mock_state)

        payload = {
            "query": "it does not work",
            "session_id": "session-789"
        }
        response = self.client.post("/chat", json=payload)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data["session_id"], "session-789")
        self.assertEqual(data["intent"], "clarify")
        self.assertEqual(data["answer"], "Which board or component are you trying to use?")
        self.assertIsNone(data["image_url"])
        self.assertIsNone(data["image_caption"])
        self.assertEqual(data["sources"], [])

    @patch("api.main.compiled_graph")
    def test_chat_uuid_generation(self, mock_graph):
        """Verify session_id is auto-generated as a UUID if omitted."""
        mock_state = {
            "query": "Hello",
            "session_id": "dummy",
            "session_history": [],
            "intent": "text",
            "final_answer": "Hi there!",
            "image_results": [],
            "sources": [],
        }
        mock_graph.invoke = MagicMock(return_value=mock_state)

        payload = {"query": "Hello"}
        response = self.client.post("/chat", json=payload)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("session_id", data)
        self.assertTrue(len(data["session_id"]) > 0)
        self.assertNotEqual(data["session_id"], "dummy")

    @patch("api.main.compiled_graph")
    def test_chat_error_handling(self, mock_graph):
        """Verify graph failures return a 500 error response with a clear message."""
        mock_graph.invoke = MagicMock(side_effect=Exception("Database connection timed out"))

        payload = {"query": "Error query"}
        response = self.client.post("/chat", json=payload)
        self.assertEqual(response.status_code, 500)
        
        data = response.json()
        self.assertIn("detail", data)
        self.assertIn("Agent processing failed", data["detail"])
        self.assertIn("Database connection timed out", data["detail"])


if __name__ == "__main__":
    unittest.main()
