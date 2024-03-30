"""
Usage:
export OPENAI_API_KEY=sk-******
python3 openai_example_chat.py
"""

import sglang as sgl
# from langfuse.decorators import observe
from langfuse.openai import openai # OpenAI integration
# from langfuse.decorators import langfuse_context


# 加载 .env 到环境变量

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
@sgl.function
def multi_turn_question(s, question_1, question_2):
    s += sgl.system("You are a helpful assistant.")
    s += sgl.user(question_1)
    s += sgl.assistant(sgl.gen("answer_1", max_tokens=256))
    s += sgl.user(question_2)
    s += sgl.assistant(sgl.gen("answer_2", max_tokens=256))


def single():
    state = multi_turn_question.run(
        question_1="""给定下面的单词列表，请判断指定的单词是否在列表中：

        列表：[苹果，香蕉，橙子，梨，草莓，蓝莓，樱桃，葡萄，西瓜，芒果，柠檬，桃子，猕猴桃，杏，黑莓，覆盆子，荔枝，龙眼，柚子，无花果]

        验证的单词：荔枝

        请仅以“是”或“否”的形式回答。
        """,
        question_2="""给定下面的单词列表，请判断指定的单词是否在列表中：

            列表：[苹果，香蕉，橙子，梨，草莓，蓝莓，樱桃，葡萄，西瓜，芒果，柠檬，桃子，猕猴桃，杏，黑莓，覆盆子，荔枝，龙眼，柚子，无花果]

            验证的单词：黄瓜

            请仅以“是”或“否”的形式回答。
            """,
    )

    for m in state.messages():
        print(m["role"], ":", m["content"])

    print("\n-- answer_1 --\n", state["answer_1"])


def stream():
    state = multi_turn_question.run(
        question_1="""给定下面的单词列表，请判断指定的单词是否在列表中：

列表：[苹果，香蕉，橙子，梨，草莓，蓝莓，樱桃，葡萄，西瓜，芒果，柠檬，桃子，猕猴桃，杏，黑莓，覆盆子，荔枝，龙眼，柚子，无花果]

验证的单词：荔枝

请仅以“是”或“否”的形式回答。
""",
        question_2="""给定下面的单词列表，请判断指定的单词是否在列表中：

    列表：[苹果，香蕉，橙子，梨，草莓，蓝莓，樱桃，葡萄，西瓜，芒果，柠檬，桃子，猕猴桃，杏，黑莓，覆盆子，荔枝，龙眼，柚子，无花果]

    验证的单词：黄瓜

    请仅以“是”或“否”的形式回答。
    """,
        stream=True
    )

    for out in state.text_iter():
        print(out, end="", flush=True)
    print()


def batch():
    states = multi_turn_question.run_batch([
        {"question_1": "What is the capital of the United States?",
         "question_2": "List two local attractions."},

        {"question_1": "What is the capital of France?",
         "question_2": "What is the population of this city?"},
    ])

    for s in states:
        print(s.messages())


if __name__ == "__main__":
    sgl.set_default_backend(sgl.OpenAI("gpt-3.5-turbo"))

    # Run a single request
    print("\n========== single ==========\n")
    single()

    # Stream output
    # print("\n========== stream ==========\n")
    # stream()

    # Run a batch of requests
    # print("\n========== batch ==========\n")
    # batch()
