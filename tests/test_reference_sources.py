"""Reference client integrity guards; actual HTTPS execution has separate receipts."""

from __future__ import annotations

import pytest

from cochem_torq.reference_sources import fetch_published_source


@pytest.mark.parametrize(
    "url",
    [
        "http://webbook.nist.gov/",
        "https://user@webbook.nist.gov/",
        "https://webbook.nist.gov/?api_key=undeclared",
        "https://webbook.nist.gov/#unretained-source-location",
        "https://webbook.nist.gov:80/",
        "https://webbook.nist.gov/?sig=unavailable",
        "https://webbook.nist.gov/?access_key=unavailable",
        "https://webbook.nist.gov/?AWSAccessKeyId=unavailable",
        "https://webbook.nist.gov/?auth=unavailable",
        "https://webbook.nist.gov/?key=unavailable",
        "https://webbook.nist.gov/?X-Amz-Credential=unavailable",
    ],
)
def test_invalid_locator_is_rejected_before_network_or_publication(tmp_path, url):
    destination = tmp_path / "source"
    with pytest.raises(ValueError):
        fetch_published_source(
            url=url,
            citation="NIST source locator contract",
            destination=destination,
            reuse_permission="not established by this mathematical contract",
        )
    assert not destination.exists()


@pytest.mark.parametrize(
    "timeout", [True, False, 0, -1, float("nan"), float("inf"), 121]
)
def test_invalid_timeout_is_rejected_without_retrieval(tmp_path, timeout):
    destination = tmp_path / "source"
    with pytest.raises(ValueError, match="bounded HTTP timeout"):
        fetch_published_source(
            url="https://webbook.nist.gov/",
            citation="NIST timeout contract",
            destination=destination,
            reuse_permission="unverified",
            timeout_seconds=timeout,
        )
    assert not destination.exists()


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, 256 * 1024 * 1024 + 1])
def test_invalid_byte_budget_is_rejected_without_retrieval(tmp_path, limit):
    destination = tmp_path / "source"
    with pytest.raises(ValueError, match="byte limit"):
        fetch_published_source(
            url="https://webbook.nist.gov/",
            citation="NIST budget contract",
            destination=destination,
            reuse_permission="unverified",
            max_bytes=limit,
        )
    assert not destination.exists()


def test_existing_source_destination_cannot_be_overwritten(tmp_path):
    destination = tmp_path / "original"
    destination.mkdir()
    original = destination / "retained.txt"
    original.write_text("Actual filesystem ownership contract, no reference data.")
    before = original.read_bytes()
    with pytest.raises(FileExistsError):
        fetch_published_source(
            url="https://webbook.nist.gov/",
            citation="NIST destination contract",
            destination=destination,
            reuse_permission="unverified",
        )
    assert original.read_bytes() == before
    assert not (destination / "retrieval.json").exists()


def test_source_destination_cannot_traverse_a_real_symlink(tmp_path):
    directory = tmp_path / "owned"
    directory.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(directory, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic links"):
        fetch_published_source(
            url="https://webbook.nist.gov/",
            citation="NIST source ownership contract",
            destination=alias / "source",
            reuse_permission="unverified",
        )
    assert not (directory / "source").exists()
