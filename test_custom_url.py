import requests
import json
import os

# The specific URL provided by the user
target_url = "https://www.bilibili.com/video/BV1SUSvBgEpz/?spm_id_from=333.1007.tianma.2-1-4.click&vd_source=65ed214bc3bc9271284d90d5a82986db"

url = "http://localhost:8000/transcript"
payload = {
    "bilibili_url": target_url,
    "need_summary": False  # No summary as LLM is disabled
}

print(f"Sending request to {url}...")
print(f"Target URL: {target_url}")

try:
    response = requests.post(url, json=payload, timeout=600) # 10 minutes timeout for download/transcribe
    print(f"Response Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Success: {data.get('success')}")
        
        if data.get('success'):
            print(f"Title: {data.get('title')}")
            content = data.get('content')
            if content:
                print(f"Content Preview: {content[:200]}...")
            else:
                print("Content is empty or None")
            
            print(f"Summary: {data.get('summary')}")
            
            if data.get('error'):
                 print(f"Warning/Error in data: {data.get('error')}")
        else:
            print(f"Error: {data.get('error')}")
    else:
        print(f"Request failed: {response.text}")

except Exception as e:
    print(f"Exception occurred: {e}")
