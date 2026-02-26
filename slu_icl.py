import argparse
import json
import logging
import base64
import os
import random
import prompt

import utils.gpt_slu_util as gs

from pathlib import Path
from typing import Optional, List, Dict, Any

# --- 新增 ---: 导入 OpenAI 库用于本地接口调用
try:
    from openai import OpenAI
except ImportError:
    # 如果用户不使用本地模式，这个库不是必需的
    OpenAI = None 

# 使用 tqdm 显示进度条
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        print("tqdm is not installed, progress bar will not be shown. "
              "Install it with: pip install tqdm")
        return iterable

# --- 日志配置 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def setup_arg_parser() -> argparse.ArgumentParser:
    """设置命令行参数解析器"""
    parser = argparse.ArgumentParser(description="使用 LLM API 进行统一的语音语言理解 (SLU)")
    parser.add_argument(
        "--input-file", type=str, required=True, help="输入的JSONL元数据文件路径"
    )
    parser.add_argument(
        "--audio-dir", type=str,  help="存放音频文件 (.wav) 的目录路径"
    )
    parser.add_argument(
        "--output-file", type=str, required=True, help="输出的JSONL文件路径"
    )
    
    # --- 修改 ---: 增加'local'选项，并更新help说明
    parser.add_argument(
        "--provider",
        type=str,
        default="google",
        choices=["google", "azure", "local"],
        help="API提供商: 'google'/'azure' (通过Dashscope), 或 'local' (本地OpenAI兼容接口)"
    )
    # --- 新增 ---: 为本地部署模型增加 --api-base 参数
    parser.add_argument(
        "--api-base",
        type=str,
        default="http://0.0.0.0:12355/v1",
        help="本地LLM服务的API基地址 (仅当 --provider='local' 时使用)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="LLM服务的API key。本地服务通常不需要，云服务则必需。"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="gemini-2.5-flash",
        help="要使用的模型名称 (例如: 'gemini-2.5-flash', 或本地部署的模型名)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="生成文本的温度。"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=512, help="模型生成的最大token数量。"
    )

    # few-shot
    parser.add_argument(
        "--n-shot", 
        type=int,
        default=0,
        help="在提示中使用的少量示例数量。"
    )

    # text dataset train_split
    parser.add_argument(
        "--train-input-file",
        type=str,
        default=None,
        help="可选的训练集JSONL文件路径，用于few-shot示例选择。"
    )

    # audio dataset train_split
    parser.add_argument(
        "--train-audio-dir",
        type=str,
        default=None,
        help="可选的训练集音频目录路径，用于few-shot示例选择。"
    )

    parser.add_argument(
        "--prompt-mode",
        type=str,
        default="vanilla",
        choices=["vanilla", "croprompt(id2sf)", "croprompt(sf2id)", "gpt-slu"],
        help="选择使用的提示模板类型。"
    )
    return parser

def encode_audio_to_base64(audio_path: Path) -> Optional[str]:
    """读取音频文件，进行Base64编码，并返回字符串。"""
    try:
        with open(audio_path, "rb") as audio_file:
            binary_data = audio_file.read()
            return base64.b64encode(binary_data).decode('utf-8')
    except FileNotFoundError:
        logging.error(f"音频文件未找到: {audio_path}")
        return None
    except Exception as e:
        logging.error(f"编码音频文件时出错 {audio_path}: {e}", exc_info=True)
        return None


def extract_json_string(text: Any) -> str:
    import re
    if not isinstance(text, str):
        return "[]"

    # 1. 移除 <think> 標籤及其內容
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    # 2. 移除 Markdown 程式碼塊語法 (```json ... ```)
    text = re.sub(r'```(?:json)?\s*|```', '', text).strip()
    
    # 3. 尋找 JSON 列表的邊界 [ ... ]
    start = text.find('[')
    end = text.rfind(']')
    
    if start != -1 and end != -1:
        return text[start:end+1]
    
    # 4. 如果沒找到 [ ]，但內容看起來像空結果，回傳標準空列表字串
    return "[]"

# Raw semantics -> Standard semantics
def transform_semantics_to_standard(
    raw_semantics: Dict[str, Any], 
    with_intent: bool=True,
    with_slot: bool=True,
) -> List[Dict[str, Any]]:
    """
    將原始訓練集的 semantics 格式轉換為系統提示詞要求的標準格式。
    """
    standard_list = []
    
    # 遍歷 意图1, 意图2...
    for intent_key, domains in raw_semantics.items():
        # 遍歷 領域 (如 音乐, 地图)
        for domain_name, slots_list in domains.items():
            if with_intent and with_slot:
                new_frame = {
                    "domain": domain_name,
                    "intent": "",
                    "slots": {}
                }
            elif with_intent:
                new_frame = {
                    "domain": domain_name,
                    "intent": "",
                }
            
            # 提取 intent 欄位並重組 slots
            actual_slots = {}
            for item in slots_list:
                if item.get("name") == "intent":
                    if with_intent:
                        new_frame["intent"] = item.get("value", "")
                else:
                    # 將 {"name": "歌手名", "value": "周杰倫"} 轉為 "歌手名": "周杰倫"
                    if with_slot:
                        if with_intent:
                            actual_slots[item["name"]] = item["value"]
                        else:
                            # 如果不包含 intent，則 slots 的 key 需要加上 domain 前綴
                            prefixed_key = f"{domain_name}-{item['name']}"
                            actual_slots[prefixed_key] = item["value"]
            
            if with_slot:
                if with_intent:
                    new_frame["slots"] = actual_slots
                else:
                    new_frame = actual_slots

            standard_list.append(new_frame)
            
    return standard_list

# --- 新增 ---: 专门用于调用本地 OpenAI 兼容接口的函数
def call_local_api(
    api_base: str,
    model_name: str,
    audio_path: Path,
    temperature: float,
    max_tokens: int,
    text_query: str="",
    prompt_mode: str="vanilla",
    stage: int=1,
    stage1_res: Optional[str]=None,
    shot_list: Optional[List[Dict[str, Any]]]=None,
) -> Optional[str]:
    """
    Use OpenAI-compatible local API to process SLU requests.
    """
    # 1) Load OpenAI module
    if OpenAI is None:
        raise ImportError("OpenAI module is needed on calling local api: pip install openai")

    client = OpenAI(base_url=api_base, api_key="ollama")

    messages = []
    
    # 1) Prepare system prompt
    system_prompt = prompt.VANILLA_PROMPT
    if prompt_mode == "croprompt(id2sf)":
        if stage == 1:
            system_prompt = prompt.ID2SF_STAGE1
        else:
            system_prompt = f"""
                {prompt.ID2SF_STAGE2}
                ---
                這是可能的意圖：
                {stage1_res}
            """
    elif prompt_mode == "croprompt(sf2id)":
        if stage == 1:
            system_prompt = prompt.SF2ID_STAGE1
        else:
            system_prompt = f"""
                {prompt.SF2ID_STAGE2}
                ---
                這是可能的槽位：
                {stage1_res}
            """
    elif prompt_mode == "gpt-slu":
        # ==== GPT-SLU stage ====
        # stage 1: ID1
        # stage 2: SF1
        # stage 3: ID2
        # stage 4: SF2
        if stage == 1:
            system_prompt = prompt.GPT_SLU_STAGE1_ID
        elif stage == 2:
            system_prompt = prompt.GPT_SLU_STAGE1_SF
        elif stage == 3:
            system_prompt = f"""
                {prompt.GPT_SLU_STAGE2_ID}
                ---
                這是第一階段預測的槽位：
                {stage1_res}
            """
        elif stage == 4:
            system_prompt = f"""
                {prompt.GPT_SLU_STAGE2_SF}
                ---
                這是第一階段預測的意圖：
                {stage1_res}
            """

    
    messages.append({
        "role": "system",
        "content": system_prompt,
    })

    # 2) Prepare few-shot examples
    if shot_list is not None and len(shot_list) > 0:
        for shot in shot_list:
            """
            shot list 結構:
            {
                "audio_path": Path,
                "query": str,
                "semantics": List[Dict[str, Any]]
            }
            """
            if shot.get("audio_path") == "":
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": shot["query"]
                        },
                    ],
                })
            else:
                shot_audio_base64 = encode_audio_to_base64(shot["audio_path"])
                if not shot_audio_base64:
                    continue
                
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "audio_url",
                            "audio_url": {"url": f"data:audio/wav;base64,{shot_audio_base64}"}
                        },
                    ],
                })
            
            if prompt_mode == "croprompt(id2sf)":
                if stage == 1:
                    standard_output = transform_semantics_to_standard(shot["semantics"], with_intent=True, with_slot=False)
                else:
                    standard_output = transform_semantics_to_standard(shot["semantics"], with_intent=True, with_slot=True)
            elif prompt_mode == "croprompt(sf2id)":
                if stage == 1:
                    standard_output = transform_semantics_to_standard(shot["semantics"], with_intent=False, with_slot=True)
                else:
                    standard_output = transform_semantics_to_standard(shot["semantics"], with_intent=True, with_slot=True)
            elif prompt_mode == "gpt-slu":
                # ==== GPT-SLU stage ====
                # stage 1: ID1
                # stage 2: SF1
                # stage 3: ID2
                # stage 4: SF2
                if stage in [1, 3]:
                    standard_output = transform_semantics_to_standard(shot["semantics"], with_intent=True, with_slot=False)
                elif stage in [2, 4]:
                    standard_output = transform_semantics_to_standard(shot["semantics"], with_intent=False, with_slot=True)
            else:
                standard_output = transform_semantics_to_standard(shot["semantics"], with_intent=True, with_slot=True)

    
            messages.append({
                "role": "assistant",
                "content": json.dumps(standard_output, ensure_ascii=False)
            })

    # 3) Current query
    if audio_path == "":
        if text_query == "":
            logging.error("Error: 本地API调用时，必须提供音频路径或文本查询。")
            return None
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": text_query
                },
            ],
        })
    else:
        audio_base64 = encode_audio_to_base64(audio_path)
        if not audio_base64:
            return None

        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "audio_url",
                    "audio_url": {"url": f"data:audio/wav;base64,{audio_base64}"}
                },
            ],
        })

    # 4) Call local API
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False
        )
        text = response.choices[0].message.content.strip()
        return extract_json_string(text)
    except Exception as e:
        logging.error(f"调用本地 API 时发生错误 (audio: {audio_path.name if audio_path != '' else 'text query'}): {e}", exc_info=True)
        return None   

def process_file(args: argparse.Namespace):
    # 1) Preparing file paths
    input_file = Path(args.input_file)
    if args.audio_dir is None:
        audio_dir = ""
    else:
        audio_dir = Path(args.audio_dir)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        logging.error(f"Error: Cannot find test input file {input_file}")
        return

    # 2) Few-shot：Build shot-list
    shot_list = []
    if(args.n_shot > 0):
        """
        shot list 結構:
        {
            "audio_path": Path,
            "query": str,
            "semantics": List[Dict[str, Any]]
        }
        """
        logging.info(f"使用 {args.n_shot}-shot，正在建構 shot list...")

        ## Checking train text file
        if(args.train_input_file is None):
            logging.error("Error: 使用 few-shot 时，必须提供 --train-file 参数。")
            return
        train_input_file = Path(args.train_input_file)

        ## Load train text lines
        try:
            with open(train_input_file, 'r', encoding='utf-8') as f:
                train_lines = f.readlines()
        except FileNotFoundError:
            logging.error(f"Error: Cannot find train input file {train_input_file}")
            return

        ## randomly select n-shot examples
        if(len(train_lines) < args.n_shot):
            logging.error(f"Error: The number of train dataset ({len(train_lines)}) is less than number of shots ({args.n_shot})。")
            return

        shot_indices = random.sample(range(len(train_lines)), args.n_shot)

        ## Checking audio dir for train set
        if args.train_audio_dir is None:
            train_audio_dir = ""
        else:
            train_audio_dir = Path(args.train_audio_dir)

        ## Build shot list
        for idx in shot_indices:
            shot_data = json.loads(train_lines[idx])
            item_id = shot_data.get("id")
            query = shot_data.get("query")
            semantics = shot_data.get("semantics", [])

            if (train_audio_dir != ""):
                audio_path = train_audio_dir / f"id_{item_id}.wav"
                if not audio_path.exists():
                    logging.warning(f"跳过 few-shot 示例，找不到音频文件: {audio_path}")
                    continue
            else:
                audio_path = ""

            shot_list.append({
                "audio_path": audio_path,
                "query": query,
                "semantics": semantics
            })
        
    # 3) Processing each line
    logging.info(f"Starting to process {len(lines)} lines from {input_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for line in tqdm(lines, desc="Processing dataset"):
            try:
                data = json.loads(line)
                item_id = data.get("id")
                ground_truth_query = data.get("query")
                if not item_id:
                    logging.warning(f"Miss line id: {line.strip()}")
                    continue
                
                if (audio_dir != ""):
                    audio_path = audio_dir / f"id_{item_id}.wav"
                    if not audio_path.exists():
                        logging.warning(f"Cannot find the audio: {audio_path}")
                        continue
                else:
                    audio_path = ""

                model_output_str = None
                
                ## ====== Call local api ======
                prompt_mode = args.prompt_mode
                if prompt_mode == "vanilla":
                    model_output_str = call_local_api(
                        api_base=args.api_base,
                        model_name=args.model_name,
                        text_query=ground_truth_query,
                        audio_path=audio_path,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        shot_list=shot_list
                    )
                elif prompt_mode == "croprompt(id2sf)":
                    # Stage 1: ID
                    stage1_output = call_local_api(
                        api_base=args.api_base,
                        model_name=args.model_name,
                        text_query=ground_truth_query,
                        audio_path=audio_path,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        prompt_mode=prompt_mode,
                        stage=1,
                        shot_list=shot_list
                    )
                    if stage1_output is None:
                        logging.error(f"Stage 1 调用失败。ID: {item_id}")
                        continue

                    # Stage 2: SF
                    model_output_str = call_local_api(
                        api_base=args.api_base,
                        model_name=args.model_name,
                        text_query=ground_truth_query,
                        audio_path=audio_path,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        prompt_mode=prompt_mode,
                        stage=2,
                        stage1_res=stage1_output,
                        shot_list=shot_list
                    )
                elif prompt_mode == "croprompt(sf2id)":
                    # Stage 1: SF
                    stage1_output = call_local_api(
                        api_base=args.api_base,
                        model_name=args.model_name,
                        text_query=ground_truth_query,
                        audio_path=audio_path,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        prompt_mode=prompt_mode,
                        stage=1,
                        shot_list=shot_list
                    )
                    if stage1_output is None:
                        logging.error(f"Stage 1 调用失败。ID: {item_id}")
                        continue

                    # Stage 2: ID
                    model_output_str = call_local_api(
                        api_base=args.api_base,
                        model_name=args.model_name,
                        text_query=ground_truth_query,
                        audio_path=audio_path,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        prompt_mode=prompt_mode,
                        stage=2,
                        stage1_res=stage1_output,
                        shot_list=shot_list
                    )
                elif prompt_mode == "gpt-slu":
                    # ==== GPT-SLU stage ====
                    # stage 1: ID1
                    id1 = call_local_api(
                        api_base=args.api_base,
                        model_name=args.model_name,
                        text_query=ground_truth_query,
                        audio_path=audio_path,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        prompt_mode=prompt_mode,
                        stage=1,
                        shot_list=shot_list
                    )
                    if id1 is None:
                        logging.error(f"ID1 调用失败。ID: {item_id}")
                        continue

                    # stage 2: SF1
                    sf1 = call_local_api(
                        api_base=args.api_base,
                        model_name=args.model_name,
                        text_query=ground_truth_query,
                        audio_path=audio_path,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        prompt_mode=prompt_mode,
                        stage=2,
                        stage1_res=id1,
                        shot_list=shot_list
                    )
                    if sf1 is None:
                        logging.error(f"SF1 调用失败。ID: {item_id}")
                        continue

                    # stage 3: ID2
                    id2 = call_local_api(
                        api_base=args.api_base,
                        model_name=args.model_name,
                        text_query=ground_truth_query,
                        audio_path=audio_path,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        prompt_mode=prompt_mode,
                        stage=3,
                        stage1_res=sf1,
                        shot_list=shot_list
                    )
                    if id2 is None:
                        logging.error(f"ID2 调用失败。ID: {item_id}")
                        continue

                    # stage 4: SF2
                    sf2 = call_local_api(
                        api_base=args.api_base,
                        model_name=args.model_name,
                        text_query=ground_truth_query,
                        audio_path=audio_path,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        prompt_mode=prompt_mode,
                        stage=4,
                        stage1_res=id2,
                        shot_list=shot_list
                    )
                    if sf2 is None:
                        logging.error(f"SF2 调用失败。ID: {item_id}")
                        continue

                    logging.debug(f"ID2 Output: {id2}")
                    logging.debug(f"SF2 Output: {sf2}")
                    
                    # ID2 和 SF2 的输出都需要解析，最终合并成一个语义列表
                    parsed_id2: List[Dict[str, Any]] = []
                    parsed_sf2: List[Dict[str, Any]] = []

                    try:
                        if id2.startswith("```json"):
                            id2 = id2.strip("```json\n").strip("`")
                        parsed_id2 = json.loads(id2)
                    except json.JSONDecodeError as e:
                        logging.warning(f"\n无法解码 ID2 的 JSON。ID: {item_id}, Error: {e}, Output: {id2}")
                    
                    try:
                        if sf2.startswith("```json"):
                            sf2 = sf2.strip("```json\n").strip("`")
                        parsed_sf2 = json.loads(sf2)
                    except json.JSONDecodeError as e:
                        logging.warning(f"\n无法解码 SF2 的 JSON。ID: {item_id}, Error: {e}, Output: {sf2}")

                    # 合并 ID2 和 SF2
                    semantics: List[Dict[str, Any]] = gs.align_and_merge_gpt_slu(
                        id2_frames=parsed_id2,
                        sf2_groups=parsed_sf2,
                    )

                # 解析逻辑保持不变
                parsed_semantics_list: List[Dict[str, Any]] = []
                if prompt_mode not in ["gpt-slu"]:
                    if model_output_str:
                        try:
                            if model_output_str.startswith("```json"):
                                model_output_str = model_output_str.strip("```json\n").strip("`")
                            
                            parsed_output = json.loads(model_output_str)

                            if isinstance(parsed_output, list):
                                parsed_semantics_list = parsed_output
                            else:
                                raise json.JSONDecodeError("Model output is not a list", model_output_str, 0)

                        except json.JSONDecodeError as e:
                            logging.warning(f"\n无法解码 JSON。ID: {item_id}, Error: {e}, Output: {model_output_str}")
                    else:
                        logging.warning(f"\nAPI 调用失败或返回空。ID: {item_id}。")
                    result = {"id": item_id, "query": ground_truth_query, "semantics": parsed_semantics_list}
                else:
                    result = {"id": item_id, "query": ground_truth_query, "semantics": semantics}
                outfile.write(json.dumps(result, ensure_ascii=False) + '\n')

            except Exception as e:
                logging.error(f"处理行时发生意外错误: {line.strip()}. 错误: {e}", exc_info=True)
            
    logging.info(f"\n处理完成。结果已保存到 {output_file}")


def main():
    """主函数"""
    parser = setup_arg_parser()
    args = parser.parse_args()

    # --- 修改 ---: 仅在非本地模式下检查 API Key
    if args.provider != "local":
        if not args.api_key:
            args.api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not args.api_key:
            logging.error(f"错误: 使用 '{args.provider}' 提供商时，必须提供 API key。")
            return
    
    if args.provider == "local" and OpenAI is None:
        logging.error("错误: 要使用 'local' 提供商, 请先安装 openai 库: pip install openai")
        return

    process_file(args)

if __name__ == "__main__":
    main()

# /share/nas169/andyfang/mac_slu/audio/audio_test
# /share/nas169/andyfang/mac_slu/label/test_set.jsonl