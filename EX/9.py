# 檔案路徑: text_summarizer.py

import openai

def 設置_openai_api_key(api_key):
    """
    設置 OpenAI API 金鑰。
    """
    openai.api_key = api_key

def 調用_openai_api(prompt, model="text-davinci-003", max_tokens=150):
    """
    調用 OpenAI API，根據給定的提示生成回應。
    
    參數:
    - prompt (str): 要傳遞給模型的提示。
    - model (str): 使用的模型名稱。
    - max_tokens (int): 生成回應的最大 token 數。

    返回:
    - str: 生成的回應文本。
    """
    try:
        response = openai.Completion.create(
            engine=model,
            prompt=prompt,
            max_tokens=max_tokens,
            n=1,
            stop=None,
            temperature=0.5,
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return f"發生錯誤：{e}"

def 生成文本總結(text):
    """
    使用 OpenAI API 生成文本總結。
    
    參數:
    - text (str): 要總結的文本。

    返回:
    - str: 生成的文本總結。
    """
    prompt = f"請為以下文本生成總結：\n\n{text}\n\n總結："
    return 調用_openai_api(prompt)

def main():
    """
    主函數，設置 API 金鑰並啟動文本總結應用。
    """
    api_key = "YOUR_OPENAI_API_KEY"
    設置_openai_api_key(api_key)
    
    print("文本總結應用已啟動！輸入 '退出' 結束。")
    while True:
        user_input = input("請輸入要總結的文本（或輸入 '退出' 結束）：")
        if user_input.lower() in ['退出', 'exit', 'quit']:
            print("應用已退出。")
            break
        summary = 生成文本總結(user_input)
        print(f"總結：{summary}")

if __name__ == "__main__":
    main()
