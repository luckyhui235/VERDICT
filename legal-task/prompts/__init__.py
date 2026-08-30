def get_dataset_system_prompt(task: str, task_config: dict) -> str:
    if task == 'cail2018':
        return (
            "You are a legal judgment prediction agent for CAIL2018.\n"
            "Rules:\n"
            "- Use two steps only: Thought and Finish.\n"
            "- Finish MUST be a single JSON object wrapped by Finish[...].\n"
            "- JSON schema: {\"relevant_articles\": [int], \"accusation\": [str], \"term_of_imprisonment\": {\"death_penalty\": bool, \"life_imprisonment\": bool, \"imprisonment\": int}}\n"
            "- Single-label articles: output exactly ONE article id in the array, e.g., [351].\n"
            "- Keep JSON minimal and valid, no comments or trailing commas.\n"
            "- If uncertain, state assumptions in Thought."
        )
    else:
        raise ValueError(f'Unsupported task type: {task}')


def get_task_few_shots(dataset: str, task_config: dict, few_shots_num: int) -> list[str]:
    if dataset == 'cail2018':
        # 暂不提供 few-shot，避免模型输出固定模板错误；可后续补充
        return []
    else:
        raise ValueError(f'Unsupported dataset type: {dataset}')
