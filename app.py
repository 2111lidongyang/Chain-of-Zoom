import os
import sys
import glob
import argparse
import torch
import numpy as np
from PIL import Image
import gradio as gr
import tempfile
import time
from pathlib import Path

# 设置环境变量
sys.path.append(os.getcwd())

# 导入必要的模块
from torchvision import transforms
from ram.models.ram_lora import ram
from ram import inference_ram as inference
from utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix

# 全局变量
global_model = None
global_dape = None
global_vlm_model = None
global_vlm_processor = None
global_process_vision_info = None
global_args = None
weight_dtype = torch.float32
device = "cuda" if torch.cuda.is_available() else "cpu"

# 定义图像处理转换
tensor_transforms = transforms.Compose([transforms.ToTensor()])
ram_transforms = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def resize_and_center_crop(img: Image.Image, size: int) -> Image.Image:
    """调整图像大小并进行中心裁剪"""
    w, h = img.size
    scale = size / min(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - size) // 2
    top = (new_h - size) // 2
    return img.crop((left, top, left + size, top + size))

def get_validation_prompt(image, prompt_image_path, dape_model=None, vlm_model=None):
    """获取验证提示 - 更接近原始实现"""
    global global_args, global_vlm_processor, global_process_vision_info, weight_dtype
    
    # 准备低分辨率张量
    lq = tensor_transforms(image).unsqueeze(0).to(device)
    
    # 根据提示类型生成提示
    if global_args.prompt_type == "null":
        prompt_text = global_args.prompt or ""
    elif global_args.prompt_type == "dape":
        lq_ram = ram_transforms(lq).to(dtype=weight_dtype)
        captions = inference(lq_ram, dape_model)
        prompt_text = f"{captions[0]}, {global_args.prompt}," if global_args.prompt else captions[0]
    elif global_args.prompt_type == "vlm":
        message_text = None
        
        if global_args.rec_type == "recursive":
            message_text = "What is in this image? Give me a set of words."
            messages = [
                {"role": "system", "content": f"{message_text}"},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": prompt_image_path}
                    ]
                }
            ]
            text = global_vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = global_process_vision_info(messages)
            inputs = global_vlm_processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            
        elif global_args.rec_type == "recursive_multiscale":
            start_image_path, input_image_path = prompt_image_path
            message_text = "The second image is a zoom-in of the first image. Based on this knowledge, what is in the second image? Give me a set of words."
            messages = [
                {"role": "system", "content": f"{message_text}"},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": start_image_path},
                        {"type": "image", "image": input_image_path}
                    ]
                }
            ]

            text = global_vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = global_process_vision_info(messages)
            inputs = global_vlm_processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

        else:
            raise ValueError(f"VLM prompt generation not implemented for rec_type: {global_args.rec_type}")

        inputs = inputs.to(device)

        # 移动SR模型组件以释放显存
        original_sr_devices = {}
        if global_args.efficient_memory and global_model is not None:
            if hasattr(global_model.model, 'text_enc_1'):
                original_sr_devices['text_enc_1'] = global_model.model.text_enc_1.device
                original_sr_devices['text_enc_2'] = global_model.model.text_enc_2.device
                original_sr_devices['text_enc_3'] = global_model.model.text_enc_3.device
                original_sr_devices['transformer'] = global_model.model.transformer.device
                original_sr_devices['vae'] = global_model.model.vae.device
                
                global_model.model.text_enc_1.to('cpu')
                global_model.model.text_enc_2.to('cpu')
                global_model.model.text_enc_3.to('cpu')
                global_model.model.transformer.to('cpu')
                global_model.model.vae.to('cpu')
                
                # 确保VLM模型在GPU上
                vlm_model.to(device)

        generated_ids = vlm_model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = global_vlm_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        prompt_text = f"{output_text[0]}, {global_args.prompt}," if global_args.prompt else output_text[0]

        # 恢复SR模型组件
        if global_args.efficient_memory and global_model is not None and hasattr(global_model.model, 'text_enc_1'):
            vlm_model.to('cpu')  # 移动VLM模型到CPU
            global_model.model.text_enc_1.to(original_sr_devices['text_enc_1'])
            global_model.model.text_enc_2.to(original_sr_devices['text_enc_2'])
            global_model.model.text_enc_3.to(original_sr_devices['text_enc_3'])
            global_model.model.transformer.to(original_sr_devices['transformer'])
            global_model.model.vae.to(original_sr_devices['vae'])
    else:
        raise ValueError(f"Unknown prompt_type: {global_args.prompt_type}")
    
    return prompt_text, lq

def process_single_image(input_image, rec_num, align_method):
    """处理单张图像并返回所有递归步骤的结果"""
    global global_model, global_dape, global_vlm_model, global_args
    
    # 确保模型已加载
    if global_model is None:
        raise gr.Error("模型未初始化，请先点击'处理图像'按钮")
    
    # 创建临时目录用于处理
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        rec_dir = tmp_path / "recursive"
        rec_dir.mkdir(parents=True, exist_ok=True)
        
        # 处理第一张图像
        first_image = resize_and_center_crop(input_image, global_args.process_size)
        first_image_path = rec_dir / "0.png"
        first_image.save(first_image_path)
        
        # 保存所有步骤的图像
        full_images = [first_image]
        current_image = first_image
        step_prompts = ["Initial image"]
        
        # 递归处理
        for rec in range(global_args.rec_num):
            print(f"Processing recursion {rec+1}/{global_args.rec_num}")
            w, h = current_image.size
            rscale = global_args.upscale
            new_w, new_h = w // rscale, h // rscale
            
            # 裁剪中心区域
            cropped_region = current_image.crop(
                ((w - new_w) // 2, 
                 (h - new_h) // 2, 
                 (w + new_w) // 2, 
                 (h + new_h) // 2))
            
            # 放大到原始尺寸
            lr_image = cropped_region.resize((w, h), Image.BICUBIC)
            
            # 准备提示生成
            prompt_image_path = None
            
            if global_args.rec_type == "recursive":
                input_image_path = rec_dir / f"{rec+1}_input.png"
                lr_image.save(input_image_path)
                prompt_image_path = str(input_image_path)
            elif global_args.rec_type == "recursive_multiscale":
                start_image_path = rec_dir / f"{rec}.png"
                input_image_path = rec_dir / f"{rec+1}_input.png"
                lr_image.save(input_image_path)
                prompt_image_path = (str(start_image_path), str(input_image_path))
            
            # 生成提示
            validation_prompt, lq = get_validation_prompt(
                lr_image, 
                prompt_image_path, 
                dape_model=global_dape, 
                vlm_model=global_vlm_model
            )
            
            # 超分辨率处理
            with torch.no_grad():
                lq = lq * 2 - 1
                
                # 确保模型组件在正确设备上
                if global_args.efficient_memory and global_model is not None:
                    if hasattr(global_model.model, 'text_enc_1'):
                        global_model.model.text_enc_1.to(device)
                        global_model.model.text_enc_2.to(device)
                        global_model.model.text_enc_3.to(device)
                    global_model.model.transformer.to(device)
                    global_model.model.vae.to(device)
                
                output_tensor = global_model(lq, prompt=validation_prompt)
                output_image = torch.clamp(output_tensor[0].cpu(), -1.0, 1.0)
                sr_image = transforms.ToPILImage()(output_image * 0.5 + 0.5)
                
                # 颜色校正
                if align_method == 'adain':
                    sr_image = adain_color_fix(target=sr_image, source=lr_image)
                elif align_method == 'wavelet':
                    sr_image = wavelet_color_fix(target=sr_image, source=lr_image)
            
            # 保存当前步骤结果
            current_image = sr_image
            output_path = rec_dir / f"{rec+1}.png"
            sr_image.save(output_path)
            full_images.append(sr_image)
            step_prompts.append(f"Step {rec+1}: {validation_prompt}")
        
        # 返回所有步骤的结果
        return full_images, step_prompts

def initialize_models():
    """初始化所有需要的模型 - 使用原始参数"""
    global global_model, global_dape, global_vlm_model, global_vlm_processor, global_process_vision_info, global_args, weight_dtype
    
    # 创建参数对象
    class Args:
        pass
    
    args = Args()
    args.pretrained_model_name_or_path = '/openbayes/input/input0/stable-diffusion-3-medium'
    args.seed = 42
    args.process_size = 512
    args.upscale = 4
    args.align_method = 'nofix'
    args.lora_path = 'ckpt/SR_LoRA/model_20001.pkl'
    args.vae_path = 'ckpt/SR_VAE/vae_encoder_20001.pt'
    args.prompt = ''
    args.prompt_type = 'vlm'
    args.ram_path = '/openbayes/input/input0/RAM/ram_swin_large_14m.pth'
    args.ram_ft_path = 'ckpt/DAPE/DAPE.pth'
    args.save_prompts = True
    args.mixed_precision = 'fp16'
    args.merge_and_unload_lora = False
    args.lora_rank = 4
    args.vae_decoder_tiled_size = 224
    args.vae_encoder_tiled_size = 1024
    args.latent_tiled_size = 96
    args.latent_tiled_overlap = 32
    args.rec_type = 'recursive_multiscale'
    args.rec_num = 4
    args.efficient_memory = False
    
    global_args = args
    
    # 设置数据类型
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32
    
    # 初始化SR模型
    try:
        from osediff_sd3 import OSEDiff_SD3_TEST, SD3Euler
        
        model = SD3Euler()
        model.text_enc_1.to(device)
        model.text_enc_2.to(device)
        model.text_enc_3.to(device)
        model.transformer.to(device, dtype=torch.float32)
        model.vae.to(device, dtype=torch.float32)
        
        for p in [model.text_enc_1, model.text_enc_2, model.text_enc_3, model.transformer, model.vae]:
            p.requires_grad_(False)
        
        global_model = OSEDiff_SD3_TEST(args, model)
        print("超分辨率模型加载成功")
    except Exception as e:
        print(f"超分辨率模型加载失败: {e}")
        return f"超分辨率模型加载失败: {str(e)}"
    
    # 加载DAPE模型
    if args.prompt_type == "dape":
        try:
            dape = ram(
                pretrained=args.ram_path,
                pretrained_condition=args.ram_ft_path,
                image_size=384,
                vit='swin_l'
            )
            dape.eval().to(device)
            global_dape = dape.to(dtype=weight_dtype)
            print("DAPE模型加载成功")
        except Exception as e:
            print(f"DAPE模型加载失败: {e}")
            return f"DAPE模型加载失败: {str(e)}"
    
    # 加载VLM模型
    if args.prompt_type == "vlm":
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            from qwen_vl_utils import process_vision_info
            
            vlm_model_name = "/openbayes/input/input0/Qwen2.5-VL-3B-Instruct"
            print(f"Loading base VLM model: {vlm_model_name}")
            global_vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                vlm_model_name,
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                device_map="auto"
            )
            global_vlm_processor = AutoProcessor.from_pretrained(vlm_model_name)
            global_process_vision_info = process_vision_info
            print('Base VLM LOADING COMPLETE')
        except Exception as e:
            print(f"VLM模型加载失败: {e}")
            return f"VLM模型加载失败: {str(e)}"
    
    return "所有模型初始化完成！"



# 创建Gradio界面
with gr.Blocks(title="高质量图像超分辨率", theme=gr.themes.Soft(primary_hue="teal")) as demo:
    gr.Markdown("# Chain-of-Zoom: Extreme Super-Resolution via Scale Autoregression and Preference Alignment")
    gr.Markdown("Upload an image and perform super-resolution processing using advanced recursive multi-scale technology.")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Input Image", height=300)
            process_btn = gr.Button("Start Super-resolution Processing", variant="primary", size="lg")
            with gr.Accordion("Advanced Settings", open=False):
                with gr.Row():
                    rec_num = gr.Slider(minimum=1, maximum=4, value=4, step=1, label="rec_num")
                    align_method = gr.Dropdown(
                        choices=["wavelet", "adain", "nofix"], 
                        value="nofix", 
                        label="align_method"
                    )
                prompt_input = gr.Textbox(label="Prompt", placeholder="Enter additional descriptive information to guide the processing process.")
            
            # 添加示例部分
            gr.Examples(
                examples=[
                    ["/openbayes/home/Chain-of-Zoom/samples/0064.png", 4, "nofix", ""],
                    ["/openbayes/home/Chain-of-Zoom/samples/0245.png", 3, "wavelet", ""],
                    ["/openbayes/home/Chain-of-Zoom/samples/0393.png", 2, "adain", ""],
                    ["/openbayes/home/Chain-of-Zoom/samples/0457.png", 4, "nofix", ""], 
                    ["/openbayes/home/Chain-of-Zoom/samples/0479.png", 4, "nofix", ""]
                ],
                inputs=[image_input, rec_num, align_method, prompt_input],
                label="Try these examples:",
                examples_per_page=3
            )
            

            
        with gr.Column():
            # 使用画廊组件展示所有处理步骤
            gallery = gr.Gallery(label="Processing steps", columns=3, height=500, preview=True)
            gr.Markdown("### Process information")
            info_output = gr.Textbox(label="Processing details", interactive=False)
            prompt_output = gr.Textbox(label="Step hint information", interactive=False, lines=5)
    
    def process_wrapper(input_image, rec_num, align_method, prompt):
        global global_args
        if not input_image:
            return None, "Please upload an image first", "", ""
        
        start_time = time.time()
        try:
            # 设置全局参数
            global_args.rec_num = int(rec_num)
            if prompt:
                global_args.prompt = prompt
            
            # 处理图像
            result_images, step_prompts = process_single_image(input_image, rec_num, align_method)
            
            # 为画廊创建带标签的图像
            gallery_images = []
            for i, img in enumerate(result_images):
                label = f"Step {i}" if i > 0 else "Initial image"
                gallery_images.append((img, label))
            
            # 生成提示信息文本
            prompt_text = "\n".join(step_prompts)
            
            # 计算耗时
            elapsed = time.time() - start_time
            info = f"Completed! Time taken: {elapsed:.2f}Second\nRecursion count: {rec_num}\nalign_method: {align_method}"
            if prompt:
                info += f"\nAdditional Tips: {prompt}"
            
            return gallery_images, info, prompt_text
        except Exception as e:
            elapsed = time.time() - start_time
            error_msg = f"Failed processing: {str(e)} | Time-consuming: {elapsed:.2f}Second"
            return None, error_msg, error_msg
    
    demo.load(inputs=None, queue=False)
    
    # 处理按钮事件
    process_btn.click(
        process_wrapper,
        inputs=[image_input, rec_num, align_method, prompt_input],
        outputs=[gallery, info_output, prompt_output]
    )

# 启动应用
if __name__ == "__main__":
    # 创建示例目录（如果不存在）
    # examples_dir = "./examples"
    # os.makedirs(examples_dir, exist_ok=True)
    
    # 初始化模型
    print("正在初始化模型...")
    init_status = initialize_models()
    print(init_status)
    
    # 启动Gradio应用
    print("启动Gradio界面...")
    demo.launch(
        server_name="0.0.0.0", 
        server_port=8080,
    )