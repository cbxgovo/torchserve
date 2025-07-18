import requests
import json

url = "http://172.16.46.20:8080/predictions/fasttext"
headers = {"Content-Type": "application/json"}

batch = [
  {"body": "这是一条测试文本"},
  {"body": "另一1条文本"},
  {"body": "另一2条文本"},
  {"body": "另一3条文本"}
]


data = json.dumps(batch)
resp = requests.post(url, headers=headers, data=data.encode("utf-8"))
print(resp.json())
