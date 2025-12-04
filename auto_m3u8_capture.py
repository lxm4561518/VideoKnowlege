import argparse
import json
import sys
import subprocess
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright
import os


def parse_netscape_cookie_file(path):
    cookies = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) != 7:
                continue
            domain, tailmatch, cpath, secure, expiry, name, value = parts
            try:
                expiry = int(expiry)
            except Exception:
                expiry = None
            cookies.append({
                "name": name,
                "value": value,
                "domain": domain,
                "path": cpath or "/",
                "expires": expiry,
                "httpOnly": False,
                "secure": secure.upper() == "TRUE",
                "sameSite": "Lax",
            })
    return cookies


def is_audio_playlist(url: str) -> bool:
    u = url.lower()
    return ("m3u8" in u) or ("audio" in u and (u.endswith(".m4a") or ".m4a" in u or ".aac" in u or ".mp3" in u))


def extract_audio_from_playurl_json(data: dict) -> str:
    # Bilibili playurl may have data in data["dash"]["audio"][0]["baseUrl"] or similar
    candidates = []
    root = data
    if "data" in root:
        root = root["data"]
    if isinstance(root, dict):
        dash = root.get("dash")
        if isinstance(dash, dict):
            aud = dash.get("audio")
            if isinstance(aud, list) and aud:
                item = aud[0]
                for key in ("baseUrl", "base_url", "url"):
                    if key in item and isinstance(item[key], str):
                        candidates.append(item[key])
                bkp = item.get("backupUrl") or item.get("backup_url")
                if isinstance(bkp, list):
                    candidates.extend([x for x in bkp if isinstance(x, str)])
    return candidates[0] if candidates else None


def main():
    parser = argparse.ArgumentParser(description="自动抓取B站音频直链并转写")
    parser.add_argument("url", help="B站视频页面URL")
    parser.add_argument("--cookies-file", default=str(Path("cookies") / "bilibili.txt"), help="Netscape Cookie文件")
    parser.add_argument("--out", default="outputs", help="输出目录")
    parser.add_argument("--model", default="small")
    parser.add_argument("--lang", default="zh")
    args = parser.parse_args()

    cookies = parse_netscape_cookie_file(args.cookies_file)
    found_url = None

    with sync_playwright() as p:
        chrome_ud = Path(os.environ.get("LOCALAPPDATA", "")) / "Google" / "Chrome" / "User Data"
        browser = None
        context = None
        try:
            if chrome_ud.exists():
                context = p.chromium.launch_persistent_context(
                    user_data_dir=str(chrome_ud),
                    channel="chrome",
                    headless=False,
                )
                page = context.new_page()
            else:
                raise RuntimeError("Chrome用户数据目录不存在")
        except Exception:
            browser = p.chromium.launch(channel="chrome", headless=False)
            context = browser.new_context()
            if cookies:
                context.add_cookies(cookies)
            page = context.new_page()

        def handle_response(response):
            nonlocal found_url
            url = response.url
            if found_url:
                return
            if "playurl" in url or is_audio_playlist(url):
                ctype = (response.headers or {}).get("content-type", "")
                try:
                    if "json" in ctype:
                        data = response.json()
                        au = extract_audio_from_playurl_json(data)
                        if au:
                            found_url = au
                            return
                    # If it's a playlist or direct audio
                    if is_audio_playlist(url):
                        found_url = url
                        return
                    # Try reading text for m3u8 content
                    txt = ""
                    try:
                        txt = response.text()
                    except Exception:
                        txt = ""
                    if "#EXTM3U" in (txt or ""):
                        found_url = url
                except Exception:
                    pass

        page.on("response", handle_response)
        page.goto(args.url, wait_until="domcontentloaded")
        try:
            page.wait_for_selector("video", timeout=5000)
            page.evaluate("document.querySelector('video') && document.querySelector('video').play()")
            page.keyboard.press("Space")
            page.evaluate("document.querySelector('.bpx-player-ctrl-play') && document.querySelector('.bpx-player-ctrl-play').click()")
        except Exception:
            pass
        # Try directly reading player info globals
        if not found_url:
            try:
                au = page.evaluate(
                    "(() => {\n"
                    "  const info = window.__playinfo__;\n"
                    "  if (info && info.dash && info.dash.audio && info.dash.audio.length) {\n"
                    "    const a = info.dash.audio[0];\n"
                    "    return a.baseUrl || a.base_url || a.url || (a.backupUrl && a.backupUrl[0]) || null;\n"
                    "  }\n"
                    "  return null;\n"
                    "})()"
                )
                if au:
                    found_url = au
            except Exception:
                pass
        page.wait_for_timeout(40000)
        try:
            context.close()
        except Exception:
            pass
        try:
            if browser:
                browser.close()
        except Exception:
            pass

    if not found_url:
        print("未能自动抓取音频直链")
        sys.exit(1)

    m3u8_or_audio = found_url
    cmd = [
        sys.executable, "download_m3u8_and_transcribe.py",
        "--m3u8", m3u8_or_audio,
        "--out", args.out,
        "--cookies-file", args.cookies_file,
        "--model", args.model,
        "--lang", args.lang,
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
