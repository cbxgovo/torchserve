# handler.py
import torch
import torch.nn.functional as F
from ts.torch_handler.base_handler import BaseHandler
import pickle
import os
import logging
from model import FastText

# 设置日志
log_file = "/tmp/fasttext_handler.log"
logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

class FastTextHandler(BaseHandler):
    def __init__(self):
        super(FastTextHandler, self).__init__()
        self.initialized = False

    def initialize(self, context):
        # 加载模型和词表
        model_dir = context.system_properties.get("model_dir")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        try:
            vocab_path = os.path.join(model_dir, "vocab.pkl")
            with open(vocab_path, "rb") as f:
                self.vocab = pickle.load(f)
        except Exception as e:
            logging.error(f"加载词表失败: {e}")
            raise

        # 配置模型参数
        self.pad_size = 2048
        self.n_vocab = len(self.vocab)
        self.embed = 300
        self.dropout = 0.5
        self.hidden_size = 256
        self.num_classes = 6
        self.n_gram_vocab = 250499
        self.labels = ["0", "1", "2", "3", "4", "5"]

        # 初始化模型结构
        self.model = FastText(
            embedding_pretrained=None,
            n_vocab=self.n_vocab,
            embed=self.embed,
            dropout=self.dropout,
            hidden_size=self.hidden_size,
            num_classes=self.num_classes,
            n_gram_vocab=self.n_gram_vocab
        )

        # 加载权重
        model_path = os.path.join(model_dir, "FastText.ckpt")
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            logging.error(f"模型加载失败: {e}")
            raise

        self.initialized = True
        logging.info("模型初始化完成")

    def preprocess(self, data):
        # logging.info("接收到原始输入: %s", str(data))

        text = data[0].get("body")
        if isinstance(text, (bytes, bytearray)):
            text = text.decode("utf-8")

        tokenizer = lambda x: [y for y in x]
        vocab = self.vocab
        UNK, PAD = '<UNK>', '<PAD>'

        def biGramHash(sequence, t, buckets):
            t1 = sequence[t - 1] if t - 1 >= 0 else 0
            return (t1 * 14918087) % buckets

        def triGramHash(sequence, t, buckets):
            t1 = sequence[t - 1] if t - 1 >= 0 else 0
            t2 = sequence[t - 2] if t - 2 >= 0 else 0
            return (t2 * 14918087 * 18408749 + t1 * 14918087) % buckets

        def to_tensor(content):
            token = tokenizer(content)
            seq_len = min(len(token), self.pad_size)
            token = token[:self.pad_size] + [PAD] * max(0, self.pad_size - len(token))
            words_line = [vocab.get(word, vocab.get(UNK)) for word in token]
            bigram = [biGramHash(words_line, i, self.n_gram_vocab) for i in range(self.pad_size)]
            trigram = [triGramHash(words_line, i, self.n_gram_vocab) for i in range(self.pad_size)]

            x = torch.LongTensor([words_line]).to(self.device)
            seq_len_tensor = torch.LongTensor([[seq_len]]).to(self.device)
            bigram_tensor = torch.LongTensor([bigram]).to(self.device)
            trigram_tensor = torch.LongTensor([trigram]).to(self.device)

            # logging.info("预处理成功，样本长度: %d", seq_len)
            return (x, seq_len_tensor, bigram_tensor, trigram_tensor)

        return to_tensor(text)

    def inference(self, inputs):
        try:
            with torch.no_grad():
                outputs = self.model(inputs)
                pred_label = torch.max(outputs.data, 1)[1].cpu().numpy()[0]
                result = [self.labels[pred_label]]
                # logging.info("模型推理成功，预测结果: %s", result)
                return result
        except Exception as e:
            logging.error("推理失败: %s", str(e))
            return ["error"]

    def postprocess(self, inference_output):
        # logging.info("后处理结果: %s", str(inference_output))
        return inference_output
