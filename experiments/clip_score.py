import os
import sys

from PIL import Image
import torch
from tqdm import tqdm
from torchmetrics.functional.multimodal.clip_score import _get_clip_model_and_processor, _clip_score_update
from torchvision.transforms.functional import to_pil_image
import open_clip

path = sys.argv[1]
if os.path.isdir(path):
    files = os.listdir(path)
else:
    files = [path]

files = sorted(files)
for sub_file in files:
    
    if os.path.isdir(path):
        file_path = os.path.join(path, sub_file)
    else:
        file_path = sub_file

    if not file_path.endswith('.pt'):
        print(f"[ERROR] {file_path} Not a pt file. Please check your path")
    
    if os.path.isdir(file_path):
        continue
    
    print("Loading file:", file_path)
    ckpt = torch.load(file_path)
    all_images = [ to_pil_image(i, mode=None) for i in ckpt['images'] ] 
    prompt_list = ckpt['prompts']
    #print(len(prompt_list))

    print("Loading model: ViT-g-14")
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s34b_b88k') # https://github.com/Nota-NetsPresso/BK-SDM/blob/a27494fe46d6d4ca0ea45291b0b8b5b547b635fd/src/eval_clip_score.py#L25
    tokenizer = open_clip.get_tokenizer('ViT-g-14')

    model.cuda()
    batch_size = 16

    with torch.no_grad(), torch.cuda.amp.autocast():
        all_score = []
        num_batch = len(prompt_list) // batch_size
        if len(prompt_list) % batch_size != 0:
            num_batch += 1

        for i in tqdm(range(num_batch)):
            img_subset = torch.stack([preprocess(i) for i in all_images[i*batch_size:(i+1)*batch_size]], 0).cuda()
            prompt_subset = prompt_list[i*batch_size:(i+1)*batch_size]
            prompts = tokenizer(prompt_subset).cuda()
            
            image_features = model.encode_image(img_subset)
            text_features = model.encode_text(prompts)
        
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            score = 100.0 * (image_features * text_features).sum(axis=-1)
            score = torch.max(score, torch.zeros_like(score))

            all_score.append(score.detach().cpu())

    final_score = torch.cat(all_score).mean(0)
    print(file_path, ", Time= ", file_path.split('-')[-1].split('.pt')[0], "s, Score=", final_score.item())
    print()