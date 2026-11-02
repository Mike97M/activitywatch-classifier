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
