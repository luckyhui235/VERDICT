import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import yaml
import argparse
import random
from tqdm import tqdm

from mas.agents import Agent
from mas.module_map import module_map
from mas.reasoning import ReasoningBase
from mas.memory import MASMemoryBase
from mas.llm import LLMCallable, GPTChat, get_price, QwenLocalChat
from mas.mas import MetaMAS
from mas.utils import EmbeddingFunc


try:
    from .mas_workflow import get_mas
    from .envs import get_env, get_recorder, get_task
    from .prompts import get_dataset_system_prompt, get_task_few_shots
    from .utils import get_model_type
except ImportError:
    sys.path.append(os.path.dirname(__file__))
    from mas_workflow import get_mas
    from envs import get_env, get_recorder, get_task
    from prompts import get_dataset_system_prompt, get_task_few_shots
    from utils import get_model_type


with open(os.path.join(os.path.dirname(__file__), 'configs.yaml')) as reader:
    CONFIG: dict = yaml.safe_load(reader)

WORKING_DIR: str = None

class TaskManager:
    def __init__(self, task_name, mas_type, memory_type, tasks, env, recorder, mas, mas_config=None, mem_config=None):
        self.task_name = task_name
        self.mas_type = mas_type
        self.memory_type = memory_type
        self.tasks = tasks
        self.env = env
        self.recorder = recorder
        self.mas = mas
        self.mas_config = mas_config or {}
        self.mem_config = mem_config or {}


def build_task(task: str, mas_type: str, memory_type: str, max_steps: int) -> TaskManager:

    with open(CONFIG.get(task).get('env_config_path')) as reader:
        config = yaml.safe_load(reader)
        # 设置CAIL的成功阈值为2/3，使“法条+罪名正确”也计作成功
        if task == 'cail2018':
            try:
                config['success_threshold'] = max(0.0, min(1.0, 2.0/3.0))
            except Exception:
                config['success_threshold'] = 2.0/3.0

    env = get_env(task, config, max_steps)
    recorder = get_recorder(task, working_dir=WORKING_DIR, namespace='total_task')
    tasks = get_task(task)
    mas_workflow: MetaMAS = get_mas(mas_type)
    mas_config: dict = CONFIG.get(mas_type, {})

    return TaskManager(
        task_name=task,
        mas_type=mas_type,
        memory_type=memory_type,
        tasks=tasks,
        env=env,
        recorder=recorder,
        mas=mas_workflow,
        mas_config=mas_config
    )


def build_mas(task_manager: TaskManager, reasoning: str = None, mas_memory: str = None, llm_type: str = None, llm_model: LLMCallable = None) -> None:
    # 环境默认：避免 CUDA 内存碎片化
    os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    # 可通过环境变量控制嵌入设备与批量（优先使用环境）
    embedding_device = os.environ.get('EMBEDDING_DEVICE')  # 建议设置为 'cuda:1'
    try:
        embedding_batch_size = int(os.environ.get('EMBEDDING_BATCH_SIZE', '16'))
    except Exception:
        embedding_batch_size = 16

    embed_func = EmbeddingFunc(
        CONFIG.get('embedding_model', "sentence-transformers/all-MiniLM-L6-v2"),
        device=embedding_device,
        batch_size=embedding_batch_size,
    )
    reasoning_module_type, mas_memory_module_type = module_map(reasoning, mas_memory)

    # 自动选择本地 Qwen 模型或默认 GPTChat
    use_local_qwen = False
    if llm_type:
        try:
            use_local_qwen = (os.path.exists(llm_type) or ('qwen' in llm_type.lower()) or ('qwen2' in llm_type.lower()))
        except Exception:
            use_local_qwen = False

    if llm_model is None:
        llm_model = QwenLocalChat(model_name=llm_type) if use_local_qwen else GPTChat(model_name=llm_type)
    reasoning_module: ReasoningBase = reasoning_module_type(llm_model=llm_model)
    mas_memory_module: MASMemoryBase = mas_memory_module_type(
        namespace=mas_memory,
        global_config=task_manager.mem_config,
        llm_model=llm_model,
        embedding_func=embed_func
    )

    task_manager.mas.add_observer(task_manager.recorder)
    task_manager.mas.build_system(reasoning_module, mas_memory_module, task_manager.env, task_manager.mas_config)


# 移除多进程切片执行，统一采用顺序模式（共享一个 MAS/LLM 实例）


def run_task(task_manager: TaskManager) -> None:

    total = len(task_manager.tasks)
    task_manager.recorder.dataset_begin()
    success_cnt = 0
    for global_task_id, task_config in enumerate(tqdm(task_manager.tasks, total=total, desc="Running Tasks")):
        task_manager.recorder.task_begin(global_task_id, task_config)
        task_main, task_description = task_manager.mas.env.set_env(task_config)
        few_shots: list[str] = get_task_few_shots(
            dataset=task_manager.task_name,
            task_config=task_config,
            few_shots_num=CONFIG.get(task_manager.task_name).get('few_shots_num', 0)
        )
        task_config.update(task_main=task_main, task_description=task_description, few_shots=few_shots)
        for agent in task_manager.mas.agents_team.values():
            if task_manager.task_name != 'cail2018':
                task_manager.recorder.log(f'------------ MAS Agent: {agent.name} ------------')
                task_instruction: str = get_dataset_system_prompt(task_manager.task_name, task_config=task_config)
                task_manager.recorder.log(agent.add_task_instruction(task_instruction))
        reward, done = task_manager.mas.schedule(task_config)
        try:
            if done:
                success_cnt += 1
        except Exception:
            pass
        try:
            task_manager.recorder.save_eval(task_manager.mas.env)
        except Exception:
            pass
        task_manager.recorder.task_end(reward, done)
    task_manager.recorder.dataset_end()
    print(f"[Sequential] processed={total}/{total}, success={success_cnt}")
    try:
        task_manager.recorder.log(f"[Sequential] processed={total}/{total}, success={success_cnt}")
    except Exception:
        pass


if __name__ == '__main__':
    random.seed(42)

    parser = argparse.ArgumentParser(description='Run legal-task with specified modules.')
    parser.add_argument('--task', type=str, choices=['cail2018'])
    parser.add_argument('--mas_type', type=str, choices=['macnet'])
    parser.add_argument('--mas_memory', type=str, default='case-memory', help='Specify mas memory module')
    parser.add_argument('--reasoning', type=str, default='io', help='Specify reasoning module')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo-0125', help='Specify the LLM model type')
    parser.add_argument('--max_trials', type=int, default=12, help='max number of steps')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='test', help='Run mode for MAS (train|test)')
    parser.add_argument('--successful_topk', type=int, default=1, help='Number of successful trajs to be retrieved from memory.')
    parser.add_argument('--failed_topk', type=int, default=0, help='Number of failed trajs to be retrieved from memory.')
    parser.add_argument('--insights_topk', type=int, default=3, help='Number of insights to be retrieved from memory.')
    parser.add_argument('--threshold', type=float, default=0.0, help='threshold for traj similarity.')
    parser.add_argument('--use_projector', action='store_true', help='whether to use role projector.')
    parser.add_argument('--hop', type=int, default=1, help='hop for traj similarity.')

    args = parser.parse_args()

    task: str = args.task
    mas_type: str = args.mas_type
    max_trials: int = args.max_trials
    model_type: str = args.model
    mas_memory_type: str = args.mas_memory
    reasoning_type: str = args.reasoning

    # 将嵌入模型名也纳入工作目录命名，避免不同维度的向量互相污染
    # ---- Derive embedding dimension to isolate working directory per model+dim ----
    embedding_model_cfg = CONFIG.get('embedding_model')
    embedding_device_env = os.environ.get('EMBEDDING_DEVICE')
    try:
        embedding_batch_size_env = int(os.environ.get('EMBEDDING_BATCH_SIZE', '16'))
    except Exception:
        embedding_batch_size_env = 16

    # Probe the current embedding dimension (small, single encode)
    try:
        _probe_embed = EmbeddingFunc(
            embedding_model_cfg,
            device=embedding_device_env,
            batch_size=embedding_batch_size_env,
        )
        _probe_vec = _probe_embed.embed_query("__dim_probe__")
        _embed_dim = len(_probe_vec) if hasattr(_probe_vec, '__len__') else None
    except Exception:
        _embed_dim = None

    embed_base = os.path.basename(embedding_model_cfg).replace('/', '_')
    embed_id = f"{embed_base}-d{_embed_dim if _embed_dim else 'unknown'}"
    # New working dir includes embedding dimension to avoid Chroma dimension mismatches across runs
    WORKING_DIR = os.path.join('./.db', get_model_type(model_type), embed_id, task, mas_type, f'{mas_memory_type}')
    # Log a concise startup line about embedding settings
    print(f"[Init] Embedding model={embedding_model_cfg}, device={embedding_device_env}, batch_size={embedding_batch_size_env}, dim={_embed_dim}")
    os.makedirs(WORKING_DIR, exist_ok=True)

    task_configs: TaskManager = build_task(task, mas_type, mas_memory_type, max_trials)
    task_configs.mas_config['successful_topk'] = args.successful_topk
    task_configs.mas_config['failed_topk'] = args.failed_topk
    task_configs.mas_config['insights_topk'] = args.insights_topk
    task_configs.mas_config['threshold'] = args.threshold
    task_configs.mas_config['use_projector'] = args.use_projector
    task_configs.mem_config.update(
        working_dir=WORKING_DIR,
        hop=args.hop,
        insights_sem_batch=16,
        embed_text_clip=4096
    )
    # 传递任务名给 MAS，用于数据集特定提示与行为
    task_configs.mas_config['task_name'] = task
    # Wire run mode into MAS config for workflow-level switching
    task_configs.mas_config['mode'] = args.mode

    # 为法条检索器传递独立的嵌入配置（与G-memory解耦，但可共用同一模型）
    task_configs.mas_config['law_embedding_model'] = embedding_model_cfg
    task_configs.mas_config['law_embedding_device'] = embedding_device_env
    task_configs.mas_config['law_embedding_batch_size'] = embedding_batch_size_env
    # 额外日志：法条检索嵌入配置
    print(f"[LawInit] law_embedding_model={embedding_model_cfg}, device={embedding_device_env}, batch_size={embedding_batch_size_env}")

    if task == 'cail2018':
        law_path = CONFIG.get('cail2018', {}).get('law_articles_path')
        # Read retrieval config from yaml
        laws_topk = CONFIG.get('cail2018', {}).get('laws_topk', 5)
        use_bm25 = CONFIG.get('cail2018', {}).get('retrieval_use_bm25', True)
        ranking_mode = CONFIG.get('cail2018', {}).get('retrieval_mode', 'rrf')
        bm25_top_k = CONFIG.get('cail2018', {}).get('retrieval_bm25_top_k', 50)
        faiss_top_k = CONFIG.get('cail2018', {}).get('retrieval_faiss_top_k', 100)
        fusion_alpha = CONFIG.get('cail2018', {}).get('retrieval_fusion_alpha', 0.7)
        task_configs.mas_config['law_articles_path'] = law_path
        task_configs.mas_config['laws_topk'] = laws_topk
        task_configs.mas_config['retrieval_use_bm25'] = use_bm25
        task_configs.mas_config['retrieval_mode'] = ranking_mode
        task_configs.mas_config['retrieval_bm25_top_k'] = bm25_top_k
        task_configs.mas_config['retrieval_faiss_top_k'] = faiss_top_k
        task_configs.mas_config['retrieval_fusion_alpha'] = fusion_alpha

    # Pass llm type down for per-process init
    task_configs.mas_config['llm_type'] = model_type
    build_mas(task_configs, reasoning_type, mas_memory_type, llm_type=model_type)
    run_task(task_configs)

    completion_tokens, prompt_tokens, _ = get_price()
    task_configs.recorder.log(f'completion_tokens:{completion_tokens}, prompt_tokens:{prompt_tokens}, price={completion_tokens*15/1000000+prompt_tokens*5/1000000}')
