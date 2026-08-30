def get_model_type(model_name: str) -> str:
    name = (model_name or '').lower()
    valid_model_types: list[str] = [
        'gpt-4o-mini',
        'qwen2.5-7b',
        'qwen2.5-14b',
        'qwen2.5-32b',
        'qwen2.5-72b',
        'intern',
        'deepseek-v3',
        'deepseek-chat',
        'deepseek-reasoner',
        'r1'
    ]

    for model_type in valid_model_types:
        if model_type in name:
            # normalize deepseek variants for folder name stability
            if 'deepseek' in model_type or model_type == 'r1':
                return 'deepseek-v3'
            return model_type

    # also map generic 'deepseek' keyword
    if 'deepseek' in name:
        return 'deepseek-v3'
    return 'unknown'