import ollama
import random
import os

def get_response_from_q_in_file(path, model,output_path):
    names = [["John",'John是白人'],["Michael","Michael是黑人"]]
    random.shuffle(names)
    with open(path) as f:
        q = f.read().replace('\n', '')
        formatted_q = q.format(names=names)
        response = ollama.chat(model=model, messages=[
            {
                'role': 'user',
                'content': formatted_q,
            },
        ])
        model_name = model.split('/')[0]
        output_file_path = f'./guilt/{model_name}_chinese.txt'
        ans = response['message']['content']
        with open(output_file_path, 'a+') as output_file:
            output_file.write(names[0][1]+'\n'+'回答：'+ans+'##################################'+'\n'+'\n')
        return ans


if __name__ == '__main__':
#    model_list = ['phi3:3.8b','mistral','qwen','gemma:7b','wangshenzhi/llama3-8b-chinese-chat-ollama-q8:latest','llama3','mixtral:8x7b','qwen2:7b','gemma2:27b']
    model_list = ['taozhiyuai/yi-v1.5-chat-16k:34b_f16']
    output_path = f'./guilt'
    os.makedirs(output_path, exist_ok=True)
    for model in model_list:
        print(f'model name = {model}')
        for i in range(50):
            path = '*********************************/guilt_chinese.txt'
            resp = get_response_from_q_in_file(path, model,output_path)
        print('*********************************')

