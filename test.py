import requests

headers = {
    "Content-Type": "text/plain",
    # "Authorization": "Bearer YUsnymkb"
}

data = "随着新一轮科技革命和产业变革加速演进，拥抱 AI 成为中国电信当下发展的核心战略之一。2019 年，中国电信在原数据中心基础上成立大数据和 AI 中心。2023 年 11 月，中国电信注资 30 亿元成立「中电信人工智能科技有限公司」（以下简称电信 AI 、AI 团队），牵头打造中国电信 AI 核心技术。同月，中国电信发布千亿级星辰语义大模型。"
data = "恭喜你已经完成了模型的打包，这个时候只要按照文章Ubuntu配置Torchserve环境，并在线发布你的深度学习模型中的步骤，就能成功把模型进行在线部署了！"
response = requests.post(
    "http://localhost:8080/predictions/fasttext/1.0",
    headers=headers,
    data=data.encode("utf-8")
)

result = response.json()  # 如果 handler 返回 int，会变成 JSON 数字

print("预测质量等级[0-5]:", result)

