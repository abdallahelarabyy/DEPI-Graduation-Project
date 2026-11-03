"""API tests with a fake Stable Diffusion pipeline (no torch / GPU / model download)."""
import base64
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import main


class FakePipeline:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        img = Image.new("RGB", (kwargs["width"], kwargs["height"]), "purple")
        return type("Result", (), {"images": [img]})()


@pytest.fixture
def client(monkeypatch, tmp_path):
    fake = FakePipeline()
    monkeypatch.setattr(main, "load_pipeline", lambda: (fake, "cpu"))
    monkeypatch.setattr(main, "make_generator", lambda device, seed: ("generator", seed))
    monkeypatch.setattr(main, "OUTPUT_DIR", str(tmp_path))
    with TestClient(main.app) as c:
        c.fake = fake
        yield c


def test_health_reports_loaded_model(client):
    body = client.get("/health").json()
    assert body["model_loaded"] is True
    assert body["device"] == "cpu"
    assert body["model_id"] == main.MODEL_ID


def test_generate_base64_with_seed(client):
    r = client.post("/generate", data={"prompt": "a lake", "width": 320, "height": 256,
                                       "seed": 123, "return_format": "base64"})
    assert r.status_code == 200
    body = r.json()
    assert body["seed"] == 123
    png = base64.b64decode(body["base64_image"].split(",", 1)[1])
    assert Image.open(io.BytesIO(png)).size == (320, 256)
    assert client.fake.calls[0]["generator"] == ("generator", 123)


def test_generate_file_returns_png_and_seed_header(client):
    r = client.post("/generate", data={"prompt": "a lake"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.headers["x-seed"].isdigit()


@pytest.mark.parametrize("field,value", [
    ("width", 500),                  # not a multiple of 8
    ("height", 2048),                # too large
    ("num_inference_steps", 0),
    ("guidance_scale", 99),
    ("return_format", "gif"),
    ("prompt", "   "),
])
def test_invalid_parameters_rejected(client, field, value):
    data = {"prompt": "a lake", field: value}
    r = client.post("/generate", data=data)
    assert r.status_code == 422
    assert client.fake.calls == []


def test_model_load_failure_returns_503(monkeypatch):
    def broken():
        raise RuntimeError("no weights")
    monkeypatch.setattr(main, "load_pipeline", broken)
    with TestClient(main.app) as c:
        assert c.get("/health").json()["model_loaded"] is False
        r = c.post("/generate", data={"prompt": "a lake"})
        assert r.status_code == 503
        assert "no weights" in r.json()["detail"]
