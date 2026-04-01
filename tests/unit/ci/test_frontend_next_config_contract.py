from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
NEXT_CONFIG_PATH = REPO_ROOT / "frontend" / "next.config.ts"



def test_frontend_next_config_includes_cloudflare_insights_in_script_src():
    content = NEXT_CONFIG_PATH.read_text(encoding="utf-8")

    assert "https://static.cloudflareinsights.com" in content
    assert "const scriptSrc = [" in content
    assert "`script-src ${scriptSrc}`" in content



def test_frontend_next_config_keeps_api_origin_in_connect_src_contract():
    content = NEXT_CONFIG_PATH.read_text(encoding="utf-8")

    assert "const connectSrc = [\"'self'\", apiBaseUrl].filter(Boolean).join(\" \")" in content
    assert "`connect-src ${connectSrc}`" in content
