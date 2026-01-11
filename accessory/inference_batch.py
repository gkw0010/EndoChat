from datetime import datetime

from SPHINX import SPHINXModel
import argparse
import copy
import os

from PIL import Image
import torch
import json
import time  
device=0

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_id",
    type=str,
    help="path to your model checkpoint",
)
parser.add_argument(
    "--test_data_path",
    type=str,
    help="path to your test data",
)
parser.add_argument(
    "--result_folder",
    type=str,
    help="path to the directory where results will be saved",
)
args = parser.parse_args()

model_id = args.model_id
# model_name=args.model_name
model = SPHINXModel.from_pretrained(pretrained_path=model_id, with_visual=True) #load the model


test_path = args.test_data_path
with open(test_path, 'r', encoding='utf-8') as file:
    test = json.load(file)

print(len(test)) 
print(test[1], test[-1])
print(f"test_model:{model_id}")
print("start inference")

# create the directory if not exist
result_folder=args.result_folder
print(result_folder)
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

result_path = f"{result_folder}/test_result.json"
#读取result_path
if not os.path.exists(result_path):
    results = []
else:
    with open(result_path, 'r', encoding='utf-8') as file:
        results = json.load(file)

print(len(results))

tested_ids=[item['id'] for item in results]
print(len(tested_ids))
if len(tested_ids) > 0:
    print(tested_ids[-1])

#start inference 
for i,item in enumerate(test):
    if test[i]['id'] not in tested_ids:
        image_path=test[i]['images'][0]
        question=item['messages'][0]['content'].replace("<image>","")
        # gen answer
        image = Image.open(image_path)  # image
        qas = [[question, None]]  # question
        print(qas)
        with torch.inference_mode():
            response = model.generate_response(qas, image, max_gen_len=1024, temperature=0.9, top_p=0.5, seed=0)      

        #structure the answer
        sub_task=""
        sub_tag=""
        if "sub_task" in item:
            sub_task=item['sub_task']
        if "sub_tag" in item:
            sub_tag=item['sub_tag']
        
        result_json = {
            "id":item['id'],
            "image": item['images'][0],
            "question": question,
            "answer": response,
            "answer_gt": item['messages'][1]['content'],
            "main_tag":item['main_tag'],
            "sub_task":sub_task,
            # "sub_tag":item['sub_tag']
            # "dataset":item['dataset'],
        }
        if "sub_tag" in item:
            result_json['sub_tag']=item['sub_tag']
        if "object" in item:
            result_json['object']=item['object']
        results.append(result_json)
        if i%100 ==0:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  
            print(f"[{timestamp}]:inference:{i}/{len(test)}")
            print(result_json)
            with open(result_path, "w") as f:
                json.dump(results, f, indent=4)
                
with open(result_path, "w") as f:
    json.dump(results, f, indent=4)