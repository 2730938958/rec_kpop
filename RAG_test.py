import os
import torch
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig
)
import re
import pdfplumber
# -------------------------- 1. 基础配置（解决符号链接警告、路径设置） --------------------------
# 禁用 Hugging Face 符号链接警告（Windows 环境专用）
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# 设置 PDF 路径（替换为你的 PDF 文件路径，相对/绝对路径均可）
PDF_PATH = "example.pdf"  # 示例：若 PDF 在代码同目录，直接写文件名；否则写完整路径如 "C:/docs/my_doc.pdf"
# 设置向量库持久化目录（自动创建，无需手动新建）
VECTOR_DB_DIR = "./qwen2.5_rag_chroma_db"


class PDFPlumberLoader:
    """使用pdfplumber加载PDF并转换为LangChain Document对象"""

    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        """加载PDF并返回Document对象列表"""
        documents = []

        with pdfplumber.open(self.file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):  # 页码从1开始
                # 提取页面文本
                text = page.extract_text()

                if text:
                    # 清理文本：移除乱码和特殊字符
                    cleaned_text = self.clean_text(text)

                    # 创建Document对象，包含页面内容和元数据
                    doc = Document(
                        page_content=cleaned_text,
                        metadata={
                            "source": self.file_path,
                            "page": page_num,
                            "total_pages": len(pdf.pages)
                        }
                    )
                    documents.append(doc)

        return documents

    def clean_text(self, text):
        """清理文本中的乱码和无效字符"""
        # 保留中文、英文、数字和常见标点
        cleaned = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9.,;!?，。；！？\n\s:：()]', '', text)
        # 去除多余空行
        cleaned = re.sub(r'\n+', '\n', cleaned).strip()
        return cleaned

# -------------------------- 2. 文档加载与分割（中文优化） --------------------------
def load_and_split_documents(pdf_path):
    """加载 PDF 并按中文语义分割文本"""
    # 加载 PDF（支持多页 PDF，自动提取文本）
    loader = PDFPlumberLoader(pdf_path)
    documents = loader.load()
    print(f"成功加载 PDF，共 {len(documents)} 页")

    # 中文优化分割器：按中文标点优先级分割，避免破坏语义
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,          # 每个文本片段长度（适配 Qwen2.5 上下文窗口）
        chunk_overlap=100,       # 片段间重叠长度（保证上下文连贯）
        separators=["\n\n", "\n", "。", "，", "；", "！", "？", " ", ""]  # 中文分割符优先级
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"文档分割完成，共生成 {len(split_docs)} 个文本片段")
    return split_docs


# -------------------------- 3. 向量库构建（自动持久化，无需手动调用 persist()） --------------------------
def build_vector_store(split_docs):
    """基于分割后的文本构建 Chroma 向量库（中文嵌入模型）"""
    # 中文专用嵌入模型（效果优于默认模型，支持中文语义理解）
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-zh-v1.5",
        model_kwargs={
            "device": "cuda" if torch.cuda.is_available() else "cpu",  # 自动使用 GPU/CPU
            "trust_remote_code": True
        },
        encode_kwargs={"normalize_embeddings": True}  # 归一化向量，提升检索精度
    )

    # 构建/加载向量库（若目录存在则加载，不存在则新建并持久化）
    if os.path.exists(VECTOR_DB_DIR):
        # 加载已有向量库（跳过重复解析 PDF）
        vector_store = Chroma(
            persist_directory=VECTOR_DB_DIR,
            embedding_function=embeddings
        )
        print(f"成功加载已有向量库，路径：{VECTOR_DB_DIR}")
    else:
        # 新建向量库并自动持久化（Chroma 0.4+ 无需手动调用 persist()）
        vector_store = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            persist_directory=VECTOR_DB_DIR
        )
        print(f"成功新建向量库并持久化，路径：{VECTOR_DB_DIR}")
    return vector_store


# -------------------------- 4. 加载 Qwen2.5-7B-Instruct 量化模型（4-bit 显存优化） --------------------------
def load_qwen2_quant_model():
    """加载 4-bit 量化的 Qwen2.5-7B-Instruct 模型（显存需求降至 4-6GB）"""
    # 4-bit 量化配置（关键：节省显存，16GB 显存可流畅运行）
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,                # 启用 4-bit 量化
        bnb_4bit_use_double_quant=True,   # 双量化优化（进一步减少显存占用）
        bnb_4bit_quant_type="nf4",        # 量化类型（适配大模型的最优类型）
        bnb_4bit_compute_dtype=torch.float16  # 计算精度（平衡速度与效果）
    )

    # 模型与分词器加载（自动下载，首次运行需等待 5-10 分钟，后续直接加载缓存）
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right"  # 右填充（避免生成时警告）
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype="auto",
        device_map="auto",  # 自动分配设备（优先 GPU，无 GPU 则用 CPU）
        trust_remote_code=True,
        low_cpu_mem_usage=True  # 减少 CPU 内存占用
    )
    model.eval()  # 推理模式（禁用训练相关层，节省显存）
    print(f"Qwen2.5-7B-Instruct 模型加载完成，当前设备：{model.device}")

    # 自定义 Qwen 提示词格式（适配模型要求的 <system>/<user>/<assistant> 标签）
    def qwen_formatted_generate(prompt, **kwargs):
        # 构建包含参考文档的对话结构
        messages = [
            {"role": "system", "content": "你是专业的文档问答助手，仅基于提供的参考文档回答问题，不编造信息；若文档无相关内容，需明确说明。"},
            {"role": "user", "content": prompt}
        ]
        # 应用模型专属模板（自动添加 <assistant> 生成标记）
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        # Tokenize 输入（适配模型输入格式）
        model_inputs = tokenizer([formatted_prompt], return_tensors="pt").to(model.device)
        # 生成回答（禁用梯度计算，节省显存）
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512,    # 最大生成长度（避免回答过长）
                temperature=0.3,       # 随机性（0.1-0.5 适合精准问答，值越大越随机）
                top_p=0.9,             # 采样阈值（控制生成多样性）
                repetition_penalty=1.1, # 重复惩罚（减少“车轱辘话”）
                do_sample=True,        # 启用采样（提升回答自然度）
                eos_token_id=tokenizer.eos_token_id  # 结束符（避免无限生成）
            )
        # 提取并解码生成结果（排除输入的 prompt 部分）
        generated_text = tokenizer.batch_decode(
            generated_ids[:, model_inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0]
        return generated_text

    # 创建 LangChain 兼容的 Pipeline（修复 func 参数错误）
    text_gen_pipeline = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer
    )
    # 封装为 LangChain LLM（通过 pipeline_kwargs 传递自定义生成逻辑）
    llm = HuggingFacePipeline(
        pipeline=text_gen_pipeline,
        model_kwargs={"temperature": 0.7}
    )
    return llm


# -------------------------- 5. 构建 RAG 链并执行问答 --------------------------
def build_rag_chain(vector_store, llm):
    """构建 RAG 问答链（检索+生成）"""
    # 中文优化提示词模板（明确要求基于文档回答，减少幻觉）
    prompt_template = """
    请严格按照以下规则回答用户问题：
    1. 必须基于提供的参考文档片段回答，不使用文档外的知识；
    2. 回答需简洁明了，分点说明（若有多个要点），避免冗余。

    参考文档片段：
    {context}

    用户问题：
    {question}

    回答：
    """
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]  # 必须与模板变量对应
    )

    # 创建 RAG 链（stuff 模式：将所有相关片段拼接进 prompt，适合短文档）
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(
            search_kwargs={"k": 1}  # 检索 top3 最相关片段（平衡精度与长度）
        ),
        return_source_documents=False,  # 返回答案来源片段（便于验证）
        chain_type_kwargs={"prompt": prompt}  # 应用自定义提示词
    )
    return rag_chain


# -------------------------- 6. 主函数（串联所有流程） --------------------------
def main():
    try:
        # 步骤1：加载并分割 PDF
        split_docs = load_and_split_documents(PDF_PATH)
        # 步骤2：构建/加载向量库
        vector_store = build_vector_store(split_docs)
        # 步骤3：加载量化模型
        llm = load_qwen2_quant_model()
        # 步骤4：构建 RAG 链
        rag_chain = build_rag_chain(vector_store, llm)
        # 步骤5：交互式问答
        print("\n=== RAG 文档问答系统已启动（输入 'exit' 退出）===")
        while True:
            user_query = input("\n请输入你的问题：")
            if user_query.lower() == "exit":
                print("感谢使用，再见！")
                break
            # 执行问答
            result = rag_chain({"query": user_query})
            # 输出结果
            print("\n【回答】")
            print(result["result"])
            # 输出来源片段（可选，便于确认回答依据）
            # print("\n【答案来源】")
            # for i, doc in enumerate(result["source_documents"], 1):
            #     page_num = doc.metadata.get("page", "未知")
            #     doc_path = doc.metadata.get("source", "未知")
            #     snippet = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
            #     print(f"\n片段 {i}：")
            #     print(f"  内容：{snippet}")
            #     print(f"  页码：第 {page_num + 1} 页（PDF 路径：{doc_path}）")  # +1 是因为 PDF 页码从 0 开始计数

    except Exception as e:
        print(f"运行出错：{str(e)}")
        # 常见错误提示
        if "No such file or directory" in str(e):
            print(f"提示：请检查 PDF 路径是否正确，当前路径为：{PDF_PATH}")
        elif "CUDA out of memory" in str(e):
            print("提示：GPU 显存不足，可尝试：1. 关闭其他程序释放显存；2. 注释量化配置（用 CPU 运行）")


if __name__ == "__main__":
    main()