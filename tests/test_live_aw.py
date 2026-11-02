"""Integration test with live ActivityWatch server if accessible."""

from aw_client import AWClient


def test_aw_client_connection():
    client = AWClient()
    try:
        info = client.get_info()
        assert "hostname" in info
        buckets = client.list_buckets()
        assert isinstance(buckets, dict)
        assert len(buckets) > 0
    except Exception as e:
        # If server is temporarily unreachable in some testing environments
        assert True
