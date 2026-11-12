"""Tests for the dataset download, including the no-Kaggle fallback."""

from pathlib import Path

import pytest

from src.data import download_data

CSV_BYTES = b"customerID,tenure,Churn\nC001,12,No\nC002,3,Yes\n"


def fake_fetch(url: str) -> bytes:
    return CSV_BYTES


@pytest.mark.unit
def test_public_copy_is_written_to_target(tmp_path: Path) -> None:
    target = tmp_path / "raw" / "telco.csv"

    result = download_data.download_public_copy(
        "https://example.com/t.csv", target, fetch=fake_fetch
    )

    assert result == target
    assert target.read_bytes() == CSV_BYTES


@pytest.mark.unit
def test_public_copy_rejects_empty_download(tmp_path: Path) -> None:
    target = tmp_path / "telco.csv"

    with pytest.raises(RuntimeError, match="empty"):
        download_data.download_public_copy(
            "https://example.com/t.csv", target, fetch=lambda url: b""
        )

    assert not target.exists()


@pytest.mark.unit
def test_without_kaggle_credentials_uses_public_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.setattr(download_data, "get_data_paths", lambda: {"raw": tmp_path})
    calls = []

    def fake_public_copy(url: str, target: Path, fetch=None) -> Path:
        calls.append((url, target))
        return target

    monkeypatch.setattr(download_data, "download_public_copy", fake_public_copy)

    result = download_data.download_telco_data()

    assert len(calls) == 1
    url, target = calls[0]
    assert url.startswith("https://")
    assert target == tmp_path / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    assert result == target
