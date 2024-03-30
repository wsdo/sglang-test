from openai import OpenAI
# from langfuse.decorators import observe
from langfuse.openai import openai # OpenAI integration
# from langfuse.decorators import langfuse_context


# 加载 .env 到环境变量

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
client = OpenAI()

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": """给定下面的单词列表，请判断指定的单词是否在列表中：

        列表：[苹果，香蕉，橙子，梨，草莓，蓝莓，樱桃，葡萄，西瓜，芒果，柠檬，桃子，猕猴桃，杏，黑莓，覆盆子，荔枝，龙眼，柚子，无花果]

        验证的单词：苹果

        请仅以“是”或“否”的形式回答。
        """
        },
        {
            "role": "assistant",
            "content": "是"
        },
        {
            "role": "user",
            "content": """给定下面的单词列表，请判断指定的单词是否在列表中：

        列表：[苹果，香蕉，橙子，梨，草莓，蓝莓，樱桃，葡萄，西瓜，芒果，柠檬，桃子，猕猴桃，杏，黑莓，覆盆子，荔枝，龙眼，柚子，无花果]
        
        验证的单词：白瓜
        
        请仅以“是”或“否”的形式回答。
        """
        },
        {
            "role": "assistant",
            "content": "否"
        },
        {
            "role": "user",
            "content": """给定下面的单词列表，请判断指定的单词是否在列表中：

列表：[苹果，香蕉，橙子，梨，草莓，蓝莓，樱桃，葡萄，西瓜，芒果，柠檬，桃子，猕猴桃，杏，黑莓，覆盆子，荔枝，龙眼，柚子，无花果]

验证的单词：黑瓜

请仅以“是”或“否”的形式回答。
"""
        },

    ],
    model="gpt-3.5-turbo",
)
print(chat_completion.choices[0].message.content)
