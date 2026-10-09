from jobbot.browser_apply import captcha_reason


def test_captcha_reason_flags_captcha():
    assert captcha_reason("Please verify you are human") is not None
    assert captcha_reason("Complete the reCAPTCHA challenge") is not None


def test_captcha_reason_ignores_login_walls():
    assert captcha_reason("Please sign in to continue") is None
    assert captcha_reason("First Name Last Name") is None
    assert captcha_reason("") is None
