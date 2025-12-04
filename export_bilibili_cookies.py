import os
import sys
import argparse
from pathlib import Path
import browser_cookie3
import http.cookiejar as cookielib


def export_chrome_cookies(output_path: str) -> str:
    jar = browser_cookie3.chrome()
    cj = cookielib.MozillaCookieJar(output_path)
    bilibili_domains = (".bilibili.com", "bilibili.com")
    for c in jar:
        if any(d in c.domain for d in bilibili_domains):
            cj.set_cookie(c)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cj.save(ignore_discard=True, ignore_expires=True)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="导出Chrome的B站Cookie为Netscape格式")
    parser.add_argument("--out", default=str(Path("cookies") / "bilibili.txt"), help="Cookie文件输出路径")
    args = parser.parse_args()
    try:
        out = export_chrome_cookies(args.out)
        print("Cookie已导出:", out)
        sys.exit(0)
    except Exception as e:
        print("导出失败:", e)
        print("请确保：1) Chrome已关闭；2) 当前用户与Chrome登录用户一致；3) 若仍失败，可使用浏览器扩展导出为Netscape格式。")
        sys.exit(1)


if __name__ == "__main__":
    main()
