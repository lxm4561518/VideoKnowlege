import os
import sys
import json
import base64
import sqlite3
import argparse
from pathlib import Path

from Cryptodome.Cipher import AES
import win32crypt


def get_chrome_paths(profile="Default"):
    base = Path(os.path.expanduser(r"~\AppData\Local\Google\Chrome\User Data"))
    local_state = base / "Local State"
    cookies_db = base / profile / "Network" / "Cookies"
    if not cookies_db.exists():
        cookies_db = base / profile / "Cookies"
    return local_state, cookies_db


def get_aes_key(local_state_path: Path) -> bytes:
    data = json.loads(local_state_path.read_text(encoding="utf-8"))
    enc_key_b64 = data["os_crypt"]["encrypted_key"]
    enc_key = base64.b64decode(enc_key_b64)
    if enc_key.startswith(b"DPAPI"):
        enc_key = enc_key[5:]
    key = win32crypt.CryptUnprotectData(enc_key, None, None, None, 0)[1]
    return key


def chrome_time_to_epoch(expires_utc: int) -> int:
    if not expires_utc:
        return 0
    return int((expires_utc - 11644473600000000) / 1000000)


def decrypt_cookie_value(raw: bytes, aes_key: bytes) -> str:
    if raw is None:
        return ""
    if raw[:3] in (b"v10", b"v11"):
        data = raw[3:]
        nonce = data[:12]
        ct = data[12:-16]
        tag = data[-16:]
        cipher = AES.new(aes_key, AES.MODE_GCM, nonce=nonce)
        return cipher.decrypt_and_verify(ct, tag).decode("utf-8", errors="ignore")
    else:
        try:
            return win32crypt.CryptUnprotectData(raw, None, None, None, 0)[1].decode("utf-8", errors="ignore")
        except Exception:
            return raw.decode("utf-8", errors="ignore")


def export_bilibili_cookies(profile: str, output_path: Path):
    local_state, cookies_db = get_chrome_paths(profile)
    if not local_state.exists():
        raise RuntimeError(f"Local State不存在: {local_state}")
    if not cookies_db.exists():
        raise RuntimeError(f"Cookies数据库不存在: {cookies_db}")
    aes_key = get_aes_key(local_state)
    tmp_db = output_path.parent / "cookies_tmp.db"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # 复制数据库以避免锁定
    with open(cookies_db, "rb") as src, open(tmp_db, "wb") as dst:
        dst.write(src.read())
    conn = sqlite3.connect(str(tmp_db))
    cur = conn.cursor()
    cur.execute(
        "SELECT host_key, name, path, is_secure, expires_utc, encrypted_value, value FROM cookies WHERE host_key LIKE '%bilibili.com%'"
    )
    lines = ["# Netscape HTTP Cookie File"]
    for host_key, name, path, is_secure, expires_utc, encrypted_value, value in cur.fetchall():
        try:
            val = value if value else decrypt_cookie_value(encrypted_value, aes_key)
        except Exception:
            val = value or ""
        val = (val or "").replace("\n", "").replace("\r", "").replace("\t", "")
        try:
            val = val.encode("latin-1", errors="ignore").decode("latin-1")
        except Exception:
            val = ""
        tailmatch = "TRUE" if host_key.startswith(".") else "FALSE"
        secure = "TRUE" if is_secure else "FALSE"
        expiry = chrome_time_to_epoch(expires_utc)
        lines.append(f"{host_key}\t{tailmatch}\t{path}\t{secure}\t{expiry}\t{name}\t{val}")
    conn.close()
    tmp_db.unlink(missing_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="直接读取Chrome并导出B站Cookie(Netscape格式)")
    parser.add_argument("--profile", default="Default", help="Chrome配置目录，如Default")
    parser.add_argument("--out", default=str(Path("cookies") / "bilibili.txt"), help="输出Cookie路径")
    args = parser.parse_args()
    try:
        out = export_bilibili_cookies(args.profile, Path(args.out))
        print("Cookie已导出:", out)
        sys.exit(0)
    except Exception as e:
        print("导出失败:", e)
        print("若失败，请确认Chrome已关闭，或者改用扩展导出Cookies为Netscape格式。")
        sys.exit(1)


if __name__ == "__main__":
    main()
