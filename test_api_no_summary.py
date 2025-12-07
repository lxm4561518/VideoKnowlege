import requests
import json
import sys

def test_no_summary():
    url = "http://127.0.0.1:8000/transcript"
    
    # Test case: Request without summary
    # Using the URL provided by the user: https://www.bilibili.com/video/BV15yUHBkEkV
    payload = {
        "bilibili_url": "https://www.bilibili.com/video/BV15yUHBkEkV",
        "need_summary": False
    }
    
    print(f"Sending POST request to {url} with payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(url, json=payload)
        
        print(f"Response Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            # Print keys to avoid flooding console with full content
            print("Response Keys:", list(data.keys()))
            print(f"Success: {data.get('success')}")
            
            if data.get('success'):
                print(f"Title: {data.get('title')}")
                
                # Check Summary (Should be None)
                summary = data.get('summary')
                if summary is None:
                    print("\nSUCCESS: 'summary' field is None as expected.")
                else:
                    print(f"\nFAILURE: 'summary' field should be None, but got: {summary[:100]}...")
                
                # Check Content (Should be present)
                content = data.get('content')
                if content and len(content) > 0:
                    print(f"\nSUCCESS: 'content' field is present ({len(content)} chars).")
                    print(f"Content Preview: {content[:100]}...")
                else:
                    print("\nFAILURE: 'content' field is missing or empty.")
            else:
                print(f"Error: {data.get('error')}")
        else:
            print(f"Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to server. Make sure http_video_knowledge.py is running.")

if __name__ == "__main__":
    test_no_summary()
