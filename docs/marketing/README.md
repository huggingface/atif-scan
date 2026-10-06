# Marketing page

A static, single-page overview of atif-scan in fast-agent's Forward design system
(ivory/petrol/amber, one orange campaign burst, Ultra only inside the burst). No
presenter: the burst is the hero.

```bash
uv run python -m http.server 8787 --bind 127.0.0.1 --directory docs/marketing
```

The page has no scripts or images. Its only third-party requests are Google Fonts
(Fraunces, Ultra, Figtree, DM Mono), allowed by its Content-Security-Policy; without
them it falls back to system fonts. All numbers are synthetic or come from the README
and the check catalogue; `tests/test_marketing_page.py` keeps the check count in step.
