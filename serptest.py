from langchain_community.utilities import SerpAPIWrapper
import os
os.environ['SERPAPI_API_KEY'] = '5f637d55472a8b1a905c0648dd0b79637288ca2e28c5a35bd248c38b7d921ceb'
# # 自定义搜索参数
# params = {
#     "engine": "bing",
#     "gl": "us",
#     "hl": "en",
# }
# 创建带有自定义参数的SerpAPIWrapper实例
search = SerpAPIWrapper()
# 运行搜索查询
result = search.run("illit有哪些作品?")
print(result)