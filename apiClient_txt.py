import requests
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

headers = {
    "Content-Type": "text/plain",
}

# 输入输出文件路径
# input_file = "/workspace/xumh3@xiaopeng.com/text_quality/output/test.txt"
input_file = "test0702.txt"
output_file = "result1.txt"

# 模型服务地址
# url = "http://172.16.45.131:8080/predictions/fasttext"
url = "http://172.16.49.21:8080/predictions/fasttext"

# 最大并发线程数（根据服务端性能调整，建议 10~50）
MAX_WORKERS = 25

# 读取所有非空行
with open(input_file, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

start_time = time.time()

# 用于保存每个结果
results = [""] * len(lines)

def predict(idx, text):
    try:
        response = requests.post(url, headers=headers, data=text.encode("utf-8"))
        response.raise_for_status()
        return idx, str(response.json())
    except Exception as e:
        print(f"\n请求失败: {e}")
        # print(f"\n请求失败: {e}，输入内容: {text}")
        return idx, "ERROR"

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(predict, i, line): i for i, line in enumerate(lines)}
    for future in tqdm(as_completed(futures), total=len(futures), desc="Predicting", ncols=80):
        idx, result = future.result()
        results[idx] = result

# 统一写出文件（按原顺序）
with open(output_file, "w", encoding="utf-8") as fout:
    for res in results:
        fout.write(f"{res}\n")

end_time = time.time()
print(f"\n全部完成，共处理 {len(lines)} 条，耗时 {end_time - start_time:.2f} 秒")
