import json
import os

from .base_env import BaseEnv, BaseRecorder
from .cail_env import CAILEnv, CAILRecorder


_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

TASKS_PATH = {
    'cail2018': os.path.join(_ROOT_DIR, 'data', 'cail2018', 'test', '2025-buk.jsonl')
}


def get_task(task: str) -> list[dict]:
    if task == 'cail2018':
        with open(TASKS_PATH['cail2018'], 'r') as f:
            tasks = [
                {
                    'task': row.get('fact'),
                    'expected': row.get('meta'),
                    'env_name': 'cail2018'
                }
                for row in (json.loads(line) for line in f)
            ]
        return tasks
    else:
        raise ValueError(f'Unsupported task type: {task}')


def get_env(task: str, env_config: dict, max_trials: int) -> BaseEnv:
    if task == 'cail2018':
        return CAILEnv(env_config, max_trials)
    else:
        raise ValueError(f'Unsupported task type: {task}')


def get_recorder(task: str, working_dir: str, namespace: str) -> BaseRecorder:
    if task == 'cail2018':
        return CAILRecorder(working_dir=working_dir, namespace=namespace)
    else:
        raise ValueError(f'Unsupported task type: {task}')
