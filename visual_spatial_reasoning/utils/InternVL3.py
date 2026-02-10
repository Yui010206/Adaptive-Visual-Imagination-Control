import math
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoConfig, AutoTokenizer

def build_transform(input_size):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def split_model(model_path):
    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0
    return device_map



if __name__ == '__main__':
    import json
    from tqdm import tqdm
    
    path = '/nas-ssd2/shoubin/pretrained_models/InternVL3-8B/'
    device_map = split_model(path)
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map=device_map).eval()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    generation_config = dict(max_new_tokens=1024, do_sample=False) # greedy decoding
    
    
    data = json.load(open('test.json', 'r'))
    system_prompt = 'You are an AI assistant designed to help with spatial reasoning in a 3D indoor scene. You must analyze any provided images or observations and answer the question.\nThese are the images that pair with the question.\n' 
    all_results = []
    total_token = 0
    acc = 0

    
    for item in tqdm(data):
        results = {}
        img_path = item['img_paths']
        pixel_values_org = [load_image('.'+img, max_num=12).to(torch.bfloat16).cuda() for img in img_path]
        pixel_values = torch.cat(pixel_values_org, dim=0)
        num_patches_list = []
        for i in range(0, len(pixel_values_org)):
            num_patches_list.append(pixel_values_org[i].size(0))
            
        image_prompt = ['Image-{}: <image>\n'.format(i) for i in range(len(img_path))]
        image_prompt = ''.join(image_prompt)

        question = 'Question: {}\n'.format(item['question'])
        question += 'Answer Choices:\n'
        for idx, choice in enumerate(item['answer_choices']):
            question += '{}\n'.format(choice)
        question += 'Output the exact answer from the choices.\nAnswer: '

        prompt = system_prompt + image_prompt + question
        # print(prompt)
        response, history = model.chat(tokenizer, pixel_values, prompt, generation_config, num_patches_list=num_patches_list, history=None, return_history=True)
        
        # import pdb; pdb.set_trace()
        response_tokens = tokenizer(
            prompt + response,
            return_tensors="pt",
            add_special_tokens=False
        )["input_ids"].shape[-1]
        
        if item['correct_answer'] in response.lower():
            acc += 1
            
        total_token += response_tokens + pixel_values.size(0) + 1
        
        results['response'] = response  
        results['response_tokens'] = response_tokens + pixel_values.size(0) + 1 
        results['correct_answer'] = item['correct_answer']
        results['question_type'] = item['question_type']
        all_results.append(results)
        # print(f'User: {prompt}\nAssistant: {response}')
        # break
    
    print(f'Accuracy: {acc/len(data)}, Average Tokens: {total_token/len(data)}')
    
    with open('internvl3_8b_results.json', 'w') as f:
        json.dump(all_results, f, indent=4)