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
            "content": "在《红楼梦》中，贾宝玉和林黛玉的关系是怎样的？"
        },
        {
            "role": "assistant",
            "content": "《红楼梦》中，贾宝玉和林黛玉是表亲关系，两人的关系非常复杂。贾宝玉是贾府的少爷，林黛玉是林家的女儿，两人从小一起长大，情投意合，情意深厚。他们之间有着深厚的感情，但由于家族和社会的种种因素，两人的爱情并不顺利，最终导致了悲剧的结局。贾宝玉和林黛玉的爱情经历了许多挫折和考验，展现出了复杂的情感纠葛和社会现实的无情打击。整个故事通过贾宝玉和林黛玉之间的爱情线索，揭示了现实社会中爱情的无奈和"
        },
        {
            "role": "user",
            "content": "《红楼梦》中，薛宝钗和贾宝玉是什么关系？"
        }
    ],
    model="gpt-3.5-turbo",
    max_tokens=256
)
print(chat_completion.choices[0].message.content)
