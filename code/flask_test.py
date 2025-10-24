from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from neo4j import GraphDatabase  # 导入Neo4j驱动
import torch
import asyncio
from langchain_community.utilities import SerpAPIWrapper
import os
os.environ['SERPAPI_API_KEY'] = '5f637d55472a8b1a905c0648dd0b79637288ca2e28c5a35bd248c38b7d921ceb'
from transformers import GenerationConfig
# 在 flask_test.py 中添加以下代码（可放在模型加载部分之后）

# 导入必要的 LangChain 组件
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.memory import ConversationBufferMemory
from langchain import PromptTemplate
import datetime

# 从 langchain_qwen_agent_test.py 导入 Qwen25LLM 类
from langchain_qwen_agent_test import Qwen25LLM

# 初始化 LangChain 工具
def get_current_time(*args, **kwargs) -> str:
    """获取当前系统时间"""
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"current time: {current_time}"

time_tool = Tool(
    name="GetCurrentTime",
    func=get_current_time,
    description="获取当前系统时间，格式为YYYY-MM-DD HH:MM:SS，调用时Action Input应为空"
)

# 初始化搜索工具和LLM
tavily_search = TavilySearchResults(max_results=2)


# 定义Agent提示模板
agent_prompt_template = """
the history conversation:

{chat_history}

Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Note: If sufficient information has not been obtained after three tool calls, please provide the best answer directly based on the available information.
Begin!

Question: {input}
Thought:{agent_scratchpad}
"""

# 创建提示模板
agent_prompt = PromptTemplate(
    input_variables=["chat_history", "tools", "tool_names", "input", "agent_scratchpad"],
    template=agent_prompt_template
)

# 初始化对话记忆（使用全局记忆存储对话历史）
agent_memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 定义工具列表
tools = [tavily_search, time_tool]

# ---------------------- 1. 初始化FastAPI和CORS配置 ----------------------
app = FastAPI(title="Qwen LLM + Neo4j Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境替换为具体前端域名
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------- 2. 定义请求模型 ----------------------
class LLMRequest(BaseModel):
    prompt: str  # 用户输入（含“知识库”则触发Neo4j）
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    temperature: float = 0.7
    max_new_tokens: int = 512


# ---------------------- 3. Neo4j配置与检索函数（新增） ----------------------
# Neo4j连接信息（替换为你的实际配置）
NEO4J_URI = "neo4j+s://30acc171.databases.neo4j.io"
NEO4J_AUTH = ("neo4j", "A_Arjzc6q8TkRAC0wtSULmanpNpTLSmCJqtXmNrtyMY")
NEO4J_DATABASE = "neo4j"

def load_singer_list(file_path):
    singer_set = set()  # 用set比list快，避免重复匹配
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            singer = line.strip()  # 去除每行的换行符、空格
            if singer:  # 跳过空行
                singer_set.add(singer.lower())  # 转小写，实现不区分大小写匹配
    return singer_set

def detect_singer(input_text, singer_set):
    input_text_lower = input_text.lower()  # 输入文本也转小写
    detected_singers = []
    for singer in singer_set:
        if singer in input_text_lower:  # 字符串包含匹配（如“我喜欢itzy”会匹配到“itzy”）
            detected_singers.append(singer.title())  # 转回首字母大写，输出标准名
    return detected_singers

def retrieve_from_neo4j(artist_name: str = "ITZY") -> list[dict]:
    """
    从Neo4j检索相关节点数据（仅当用户输入含“知识库”时调用）
    :param artist_name: 检索的目标艺人（可根据需求调整）
    :return: 相关节点的KV列表（空列表表示检索失败/无数据）
    """
    driver = None
    related_li = []
    try:
        # 建立Neo4j连接
        driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
        driver.verify_connectivity()  # 验证连接有效性
        print("✅ Neo4j连接成功，开始检索数据")

        # 执行Cypher查询（匹配目标艺人的相关节点，取前5条避免数据过载）
        records, summary, keys = driver.execute_query("""
            MATCH (ITZY:Artist {artistName: "ITZY"})-[r]-(related)
            RETURN ITZY, r, related
            """,
            database_="neo4j".replace('ITZY', artist_name),)

        related_li = []
        # Loop through results and do something with them
        for record in records[:5]:
            related_li.append({
                'artistName': record.data()['related']['artistName'],
                'topTrack': record.data()['related']['topTrack'],
                'topTrackLink': record.data()['related']['topTrackLink'],
                'topTrackAlbum': record.data()['related']['topTrackAlbum'],
                'topTrackAlbumLink': record.data()['related']['topTrackAlbumLink'],
            })  # obtain record as dict

        print(f"✅ Neo4j检索完成，获取到 {len(related_li)} 个相关节点")
        return related_li

    except Exception as e:
        print(f"❌ Neo4j检索失败: {str(e)}")
        return []  # 检索失败时返回空列表，不阻断LLM流程

    finally:
        # 确保关闭连接，避免资源泄漏
        if driver:
            driver.close()
            print("✅ Neo4j连接已关闭")


# ---------------------- 4. LLM模型加载（保持原有逻辑） ----------------------
model_name = "Qwen/Qwen2.5-7B-Instruct"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
singer_list = load_singer_list("singer.txt")
print("开始加载Qwen模型...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
generation_config = GenerationConfig(
            max_new_tokens=512,
            do_sample=False,
            temperature=0.7,
            top_p=0.95,
            eos_token_id=151645,
            pad_token_id=151643
        )
langchain_llm = Qwen25LLM(model=model, tokenizer=tokenizer, generation_config=generation_config)  # 实例化Qwen25LLM
# 创建Agent执行器
agent_executor = AgentExecutor(
    agent=create_react_agent(langchain_llm, tools, agent_prompt),
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    memory=agent_memory,
    max_iterations=3,
    early_stopping_method="force"
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"模型加载完成，使用设备: {model.device}")


# ---------------------- 5. 文本生成函数（新增Neo4j数据整合逻辑） ----------------------
def generate_text(text: str, temperature: float, max_new_tokens: int) -> str:
    """原有文本生成逻辑，保持不变"""
    model_inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id  # 避免警告
    )
    generated_ids_slice = generated_ids[0][len(model_inputs.input_ids[0]):]
    return tokenizer.decode(generated_ids_slice, skip_special_tokens=True)


def build_prompt_with_neo4j(user_prompt: str, neo4j_data: list[dict], singer) -> str:
    """
    整合Neo4j数据构建新Prompt（仅当有Neo4j数据时调用）
    :param user_prompt: 用户原始输入
    :param neo4j_data: Neo4j检索到的KV列表
    :return: 包含知识库信息的完整Prompt
    """
    # 格式化Neo4j数据为可读文本（每行一个KV）
    neo4j_info = ""
    for idx, node in enumerate(neo4j_data, 1):
        neo4j_info += f"\n【艺人{idx}】\n"
        for key, value in node.items():
            neo4j_info += f"- {key}：{value}\n"

    # 组装含知识库的Prompt
    print(neo4j_info)
    full_prompt = f"""
    你是专业的K-Pop音乐助手，用户想让你推荐几个kpop歌手，你需基于以下【知识库歌手信息】进行推荐，要求回答自然，用英语回答：
    
    【知识库歌手信息】
    {neo4j_info}

    
    """
    return full_prompt  # 去除多余空格


# ---------------------- 6. API端点（核心：判断是否触发Neo4j） ----------------------
@app.post("/api/llm")
async def generate_response(request: LLMRequest):
    try:
        user_prompt = request.prompt.strip()
        neo4j_data = []  # 存储Neo4j数据（默认空）
        final_prompt = user_prompt  # 最终传入LLM的Prompt（默认用户原始输入）

        # ---------------------- 关键判断：用户输入是否含“知识库” ----------------------
        if "knowledge base" in user_prompt:
            print("🔍 检测到用户输入含“知识库”，调用Neo4j检索")
            singer = detect_singer(user_prompt, singer_list)
            singer = singer[0] if len(singer) > 0 else "ITZY"
            neo4j_data = retrieve_from_neo4j(artist_name=singer)

            if neo4j_data:
                final_prompt = build_prompt_with_neo4j(user_prompt, neo4j_data, singer)
            else:
                final_prompt = f"{user_prompt}\n（注：知识库暂未获取到相关数据，将基于默认知识回答）"

        elif "search" in user_prompt or "internet" in user_prompt:
            search = SerpAPIWrapper()
            result = search.run(user_prompt)
            final_prompt = f"用户现在有以下联网搜索要求[{user_prompt}]\n以下是联网搜索到的内容：{result},请直接生成一段给用户的答案"

        else:
            print("使用LangChain Agent回答ing")
            # 调用LangChain Agent处理用户输入
            loop = asyncio.get_event_loop()
            # 由于Agent调用是同步的，需要用run_in_executor包装为异步
            agent_result = await loop.run_in_executor(
                None,
                lambda: agent_executor.invoke({"input": user_prompt})
            )
            response_text = agent_result["output"]
            # 直接返回Agent结果，不需要再调用generate_text
            return {
                "response": response_text,
                "recommendations": neo4j_data,
                "used_knowledge_base": False,
                "knowledge_base_count": 0
            }

        # ---------------------- 构建对话模板并调用LLM（原有逻辑，处理知识库和搜索分支） ----------------------
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": final_prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        loop = asyncio.get_event_loop()
        response_text = await loop.run_in_executor(
            None,
            generate_text,
            text,
            request.temperature,
            request.max_new_tokens
        )
        print(f'📝 模型生成的答复：{response_text}')

        return {
            "response": response_text,
            "recommendations": neo4j_data,
            "used_knowledge_base": "knowledge base" in user_prompt,
            "knowledge_base_count": len(neo4j_data)
        }

    except Exception as e:
        error_msg = f"服务处理失败: {str(e)}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)


# ---------------------- 7. 根路径测试 ----------------------
@app.get("/")
async def root():
    return {
        "message": "Qwen LLM + Neo4j Service is running",
        "tip": "用户输入含“知识库”关键词时，将调用Neo4j检索数据"
    }


# ---------------------- 8. 启动服务 ----------------------
if __name__ == "__main__":
    import uvicorn

    # 单进程模式（确保调试断点生效）
    uvicorn.run(app, host="0.0.0.0", port=7899, workers=1, reload=False)