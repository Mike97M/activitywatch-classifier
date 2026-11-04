"""Unit tests for the PII Anonymizer."""

from anonymizer import TextAnonymizer


def test_sanitize_shell_prompts_and_usernames():
    anon = TextAnonymizer(extra_usernames=["jdoe", "jdoe97"])

    # Linux terminal prompt
    title = "jdoe@laptop:~"
    clean = anon.sanitize_title(title)
    assert "jdoe" not in clean.lower()
    assert "<USER>" in clean

    # Username in title
    title2 = "Workspace for jdoe - VS Code"
    clean2 = anon.sanitize_title(title2)
    assert "jdoe" not in clean2.lower()
    assert "<USER>" in clean2


def test_sanitize_home_and_file_paths():
    anon = TextAnonymizer(extra_usernames=["jdoe"])

    unix_path = "Editing /home/jdoe/Developement/secret/app.py in Vim"
    clean = anon.sanitize_title(unix_path)
    assert "/home/jdoe" not in clean
    assert "~/<PATH>" in clean or "<PATH>" in clean

    win_path = "C:\\Users\\jdoe\\Documents\\confidential.docx"
    clean_win = anon.sanitize_title(win_path)
    assert "jdoe" not in clean_win.lower()


def test_sanitize_emails_and_ips():
    anon = TextAnonymizer()

    email_title = "Inbox - alice.smith@corporate.org - Thunderbird"
    clean_email = anon.sanitize_title(email_title)
    assert "alice.smith@corporate.org" not in clean_email
    assert "<EMAIL>" in clean_email

    ip_url = "https://static.231.85.217.95.clients.your-server.de/dashboard"
    clean_url = anon.sanitize_url(ip_url)
    assert "231.85.217.95" not in clean_url
    assert "<IP>" in clean_url


def test_sanitize_sensitive_url_tokens():
    anon = TextAnonymizer()

    secret_url = "https://api.github.com/login/oauth?token=ghp_secretToken123456789&client_id=myclient"
    clean_url = anon.sanitize_url(secret_url)
    assert "ghp_secretToken123456789" not in clean_url
    assert "token=%3CREDACTED%3E" in clean_url or "token=<REDACTED>" in clean_url


def test_sanitize_direct_message_contacts():
    anon = TextAnonymizer()

    dm_title = "Slack - Direct Message with Sarah Connor"
    clean_dm = anon.sanitize_title(dm_title)
    assert "Sarah Connor" not in clean_dm
    assert "<CONTACT>" in clean_dm


def test_sanitize_spa_hash_sessions():
    anon = TextAnonymizer()

    title = (
        "app.mailbox.org/appsuite/#session=fc0a205dd8c64bfbb06cbeda0f99747b&!!&"
        "app=io.ox/mail&code=secret_code_123&session_state=uuid_or_hash&state=secret_state - brave"
    )
    clean = anon.sanitize_title(title)
    assert "fc0a205dd8c64bfbb06cbeda0f99747b" not in clean
    assert "secret_code_123" not in clean
    assert "uuid_or_hash" not in clean
    assert "secret_state" not in clean
    assert "session=<REDACTED>" in clean
    assert "state=<REDACTED>" in clean


def test_sanitize_oauth_jwt_and_access_tokens():
    anon = TextAnonymizer()

    title = (
        "console.hetzner.com/#access_token=80a17a68d6154e330760a43131e0a5d77a0a9fa4&"
        "expires_in=300&token_type=bearer&scope=openid&state=oelzwvhhyu14tw42x0tvnxnvsvj0svbob210x2c3vgjsukfcajb6zy1td3rv&"
        "id_token=eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJodHRwczovL2FjY291bnRzLmNvbSIsImF1ZCI6Im15YXBwIn0.abcdef123456&"
        "sid=6c0e86b2-6c2e-4b71-9ce5-e414cbb192b0"
    )
    clean = anon.sanitize_title(title)
    assert "80a17a68d6154e330760a43131e0a5d77a0a9fa4" not in clean
    assert "oelzwvhhyu14tw42x0tvnxnvsvj0svbob210x2c3vgjsukfcajb6zy1td3rv" not in clean
    assert "eyJhbGciOiJSUzI1Ni" not in clean
    assert "6c0e86b2-6c2e-4b71-9ce5-e414cbb192b0" not in clean
    assert "access_token=<REDACTED>" in clean
    assert "state=<REDACTED>" in clean


def test_sanitize_url_path_hex_tokens():
    anon = TextAnonymizer()

    # Local server token in URL path (e.g. fish_config)
    url = "http://localhost:8002/4c3aac2a0ceed8f63ce8f7aafab140d7/"
    clean = anon.sanitize_url(url)
    assert "4c3aac2a0ceed8f63ce8f7aafab140d7" not in clean
    assert "<TOKEN>" in clean

    # GitHub Gist commit/hash in URL path
    gist_url = "https://gist.github.com/schmunk42/fbbcdee420af674798e4d2fc32a62165"
    clean_gist = anon.sanitize_url(gist_url)
    assert "fbbcdee420af674798e4d2fc32a62165" not in clean_gist
    assert "<TOKEN>" in clean_gist
