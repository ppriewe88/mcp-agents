from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-4.1",
    temperature=0.1,
    max_completion_tokens=1000,
    timeout=30
)
if __name__ == "__main__":

    response = model.invoke("Why do parrots talk?")
    print(response)