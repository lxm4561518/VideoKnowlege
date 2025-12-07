import requests
import json

url = "http://localhost:8000/transcript"
payload = {
    "bilibili_url": "https://www.bilibili.com/video/BV15yUHBkEkV"
}
headers = {
    "Content-Type": "application/json"
}

try:
    print(f"Sending request to {url}...")
    response = requests.post(url, json=payload, headers=headers, timeout=300)
    print(f"Status Code: {response.status_code}")
    try:
        print("Response JSON:")
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    except:
        print("Response Text:")
        print(response.text)
except Exception as e:
    print(f"Request failed: {e}")
