模型接口部署

自己封装handler.py

model.py为handler.py的依赖模型结构

FastText.ckpt、vocab.pkl当前为空文件占位用。

# 模型打包

```
torch-model-archiver \
  --model-name fasttext \
  --version 1.0 \
  --serialized-file FastText.ckpt \
  --handler handler.py \
  --extra-files "model-config.yaml,vocab.pkl,model.py" \
  --export-path model_store \
  --force
```

# 模型部署

```

# 安装 torch-model-archiver：
pip install torch-model-archiver torchserve 

# 拉取
cp -r /workspace/xumh3@*.com/PPU/torchserve ./

# java依赖
apt update
apt install openjdk-11-jdk -y

# 环境变量
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH


# 启动 TorchServe 服务
#切换目录
cd torchserve
#启动服务
nohup torchserve --start --ncs --model-store model_store --models fasttext=fasttext.mar --ts-config config.properties > torchserve.log 2>&1 &


# 查看ip
ip a | grep inet
```


# 接口调用

apiClient_one.py为单个文本调用接口示例

apiClient_txt.py为txt文件逐行接口调用示例
