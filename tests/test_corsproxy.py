import urllib.request
import json
import ssl

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

req = urllib.request.Request(
    'https://corsproxy.io/?https://api.groq.com/openai/v1/chat/completions',
    data=json.dumps({"model": "test"}).encode('utf-8'),
    headers={
        'Content-Type': 'application/json',
        'Authorization': 'Bearer 123',
        'User-Agent': 'Mozilla/5.0'
    },
    method='POST'
)

try:
    with urllib.request.urlopen(req, context=ctx) as response:
        print("Status:", response.status)
        print("Body:", response.read().decode('utf-8'))
except urllib.error.HTTPError as e:
    print("HTTP Error:", e.code)
    print("Error Body:", e.read().decode('utf-8'))
except Exception as e:
    print("Error:", e)
