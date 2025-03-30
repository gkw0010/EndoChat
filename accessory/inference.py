from SPHINX import SPHINXModel
from PIL import Image
import torch


# Besides loading the `consolidated.*.pth` model weights, from_pretrained will also try to 
# use `tokenizer.model', 'meta.json', and 'config.json' under `pretrained_path` to configure
# the `tokenizer_path`, `llama_type`, and `llama_config` of the model. You may also override
# the configurations by explicitly specifying the arguments
model = SPHINXModel.from_pretrained(pretrained_path="/mnt/data1_hdd/wgk/SurgBot/accessory/output/llama_ens5_light_13b_esd/epoch0_esd_3C80_3endo_dd", with_visual=True)

image = Image.open("/mnt/data1_hdd/wgk/SurgBot/accessory/finetune_data/CoPESD/023521/0001.jpg") 
qas = [["Explain the various aspects of the image before you.", None]] #input your question here

response = model.generate_response(qas, image, max_gen_len=1024, temperature=0.9, top_p=0.5, seed=0)
# print("answer:")
print(response)

