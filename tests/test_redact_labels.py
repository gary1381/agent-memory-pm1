"""Labelled short codes (setup / verification codes, Chinese or English labels) are redacted."""
from pm1.redact import MARK, redact


def test_labelled_codes_are_redacted():
    for text, gone in (('- **首次设置码**：`AbCdEfGhJk`（本机浏览器通常可直接完成注册）', 'AbCdEfGhJk'),
                       ('验证码: 482913，五分钟内有效', '482913'),
                       ('初始密码：Welcome2026', 'Welcome2026'),
                       ('Setup code: q7RtZpLmXw', 'q7RtZpLmXw'),
                       ('pin code = 90210771', '90210771')):
        out = redact(text)
        assert gone not in out and MARK in out, text


def test_label_words_alone_stay():
    for text in ('统一 API Key：控制台已打印（也可在网页 Keys 页查看）', '密码：见控制台', '设置码在网页上显示',
                 'the setup code is printed in the console', '令牌：无'):
        assert redact(text) == text, text
