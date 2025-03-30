from datetime import datetime

from SPHINX import SPHINXModel
import argparse
import copy
import os

from PIL import Image
import torch
import json

def main():
    model_name="your_model_name"
    model = SPHINXModel.from_pretrained(pretrained_path=f"/mnt/data1_hdd/wgk/SurgBot/accessory/output/llama_ens5_light_13b_esd/{model_name}", with_visual=True)

    paths = {
        "/path/to/your/jsonfile/test.json" #put your test file here
    }

    for key, value in paths.items():
        print(f"test dataset:{key}")
        print(f"Key: {key}, Value: {value}")
        with open(value, 'r', encoding='utf-8') as file:
            test = json.load(file)
        print(len(test))
        # print(test[:2], test[-2:])
        print("start inference")
        results = []
        # 保存结果的路径
        result_path = f"result/endochat/result.json"
        for i in range(0, len(test)):
            # read images
            test[i]['image'] = test[i]['image'].replace("\\", "/")
            image_folder_path="/path/to/your/image/folder"

            image_path = image_folder_path + test[i]['image']
            # 生成答案
            image = Image.open(image_path)  # 图片
            qas = [[test[i]['conversations'][0]['value'], None]]  # 问题
            response = model.generate_response(qas, image, max_gen_len=1024, temperature=0.9, top_p=0.5, seed=0)
            result_json = {
                "image": test[i]['image'],
                "question": test[i]['conversations'][0]['value'],
                "answer": response,
                "answer_gt": test[i]['conversations'][1]['value']
            }
            results.append(result_json)
            if i % 100 == 0 or i == len(test) - 1:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 获取当前时间并格式化
                print(f"[{timestamp}]:inferencing:{i}/{len(test)}")
                print(result_json)
                with open(result_path, "w") as f:
                    json.dump(results, f, indent=4)

        with open(result_path, "w") as f:
            json.dump(results, f, indent=4)




if __name__ == "__main__":
    # Default args here will train EnDora-XL/2 with the hyperparameters we used in our paper (except training iters).
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--path", type=str, default="")
    #
    # args = parser.parse_args()
    # main(args.path)
    main()

