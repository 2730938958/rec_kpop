import os
import torch
from typing import Any, List, Optional, Dict, Union
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig
)
from langchain.tools import Tool
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.llms import BaseLLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.schema import Generation, LLMResult
from langchain.memory import ConversationBufferMemory
from langchain import PromptTemplate

# 设置Tavily搜索工具API密钥
os.environ["TAVILY_API_KEY"] = "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"

# Qwen2.5-7B-Instruct模型名称
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

prompt = hub.pull("hwchase17/react")

# 配置4位量化
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)


# 封装Qwen为LangChain BaseLLM子类
class Qwen25LLM(BaseLLM):
    # 新增：接收外部模型和分词器
    model: Any
    tokenizer: Any
    generation_config : Any

    def __init__(self, model: Any, tokenizer: Any, generation_config: Dict[str, Any], **kwargs):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            generation_config=generation_config, **kwargs
        )
        self.model = model  # 使用外部传入的模型
        self.tokenizer = tokenizer  # 使用外部传入的分词器
        self.generation_config = generation_config

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any,
    ) -> str:
        """核心单条prompt调用方法"""
        # 构建Qwen格式的对话历史
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant. "
                                          "When using tools, follow the React agent format strictly. "
                                          "Remember the conversation history and use it to answer follow-up questions."},
            {"role": "user", "content": prompt}
        ]

        # 转换为模型输入格式
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize输入
        model_inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.model.device)

        # 调用模型生成结果
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                generation_config=self.generation_config
            )

        # 解码结果
        input_ids_len = model_inputs.input_ids.shape[1]
        generated_ids_slice = generated_ids[0][input_ids_len:]
        response = self.tokenizer.decode(
            generated_ids_slice,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        # 处理stop词
        if stop:
            for stop_word in stop:
                if stop_word in response:
                    response = response.split(stop_word)[0]
        return response

    def _generate(
            self,
            prompts: List[str],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any,
    ) -> LLMResult:
        """批量prompt调用方法"""
        generations = []
        for prompt in prompts:
            response = self._call(prompt, stop=stop, run_manager=run_manager, **kwargs)
            generations.append([Generation(text=response)])
        return LLMResult(generations=generations)

    @property
    def _identifying_params(self) -> dict:
        """模型标识参数"""
        return {
            "model_name": MODEL_NAME,
            "quantization": "4-bit NF4",
            "generation_config": self.generation_config.to_dict()
        }

    @property
    def _llm_type(self) -> str:
        """LLM类型标识"""
        return "qwen2.5-7b-instruct"




# 构建多轮对话Agent
def main():
    # 初始化Qwen LLM
    llm = Qwen25LLM()
    print("Qwen2.5-7B-Instruct模型初始化完成（4位量化）")

    import datetime
    def get_current_time(*args, **kwargs) -> str:
        """Get the current system time and date in the format YYYY-MM-DD HH:MM:SS (year-month-day-hour: minute: second)"""
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"current time: {current_time}"
    time_tool = Tool(
        name="GetCurrentTime",  # 工具名称（模型调用时需严格匹配）
        func=get_current_time,  # 绑定的函数
        description="Get current system time, should be used in current time, date, year and so on"
        "When calling, the Action Input should be empty and directly written as: \nAction Input:"
        )

    tools = [TavilySearchResults(max_results=2), time_tool]

    # 获取React Agent提示模板
    prompt_template = """
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

    # 用自定义模板创建 Prompt
    prompt = PromptTemplate(
        input_variables=["chat_history", "tools", "input", ...],  # 必须包含记忆变量
        template=prompt_template
    )

    # 初始化对话记忆
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # 构建Agent和执行器
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        memory=memory,
        max_iterations=3,
        early_stopping_method="force"
    )

    # 多轮对话循环
    print("\n欢迎使用多轮对话助手！输入'退出'结束对话。")
    while True:
        user_input = input("\n请输入您的问题：")

        if user_input.lower() in ["退出", "quit", "exit"]:
            print("对话结束，再见！")
            break

        try:
            # 执行查询并使用记忆
            result = agent_executor.invoke({"input": user_input})

            # 输出结果
            print("\n==================== 回答 ====================")
            print(result["output"])

            # 显示当前记忆内容（可选）
            # print("\n==================== 当前记忆 ====================")
            # print(memory.load_memory_variables({}))

        except Exception as e:
            print(f"处理过程中出现错误：{str(e)}")


if __name__ == "__main__":
    main()