task_solve_with_insights = """
## Facts
Use the following facts as the primary basis:
{facts}
---

## Retrieved Precedents (Examples)
Refer to similar successful cases and your past successful trajectories:
{few_shots}

{memory_few_shots}
---

## Insights
Key insights distilled from related tasks:
{insights}
---

## Relevant Law Articles
Consider the following law articles when reasoning:
{law_context}
---

## Task
Integrate facts, precedents, insights, and law articles to solve:
{task_description}
"""

task_format = """
### 成功先例：任务描述
## Facts
{facts}

## 输出要求
请根据上述法律事实，预测出相关法条、罪名和刑期，输出格式为一个 JSON 对象，包含三个字段：
- relevant_articles: 法条编号（单个int类数字），如 [266]
- accusation: 罪名，如 ["诈骗罪"]
- term_of_imprisonment: death_penalty: bool, life_imprisonment: bool, imprisonment: 月数int

格式要求：必须以 Finish[<JSON>] 输出，其中 <JSON> 为无注释、可解析的标准 JSON。

### 成功先例：协作推理摘要
{agent_steps}

### 成功先例：判决结果
{result_finish}
"""


def format_task_prompt_with_insights(
    few_shots: list[str],
    memory_few_shots: list[str],
    insights: list[str],
    task_description: str,
    extra_context: str = "",
    facts_text: str | None = None
) -> str:
    existing_rules_text: str = '\n'.join([f'{i}. {r}' for i, r in enumerate(insights, 1)])
    memory_few_shots_text: str = '\n\n'.join([f"Task {i+1}:\n{shot}" for i, shot in enumerate(memory_few_shots)])
    law_ctx: str = (extra_context or '').strip()
    facts: str = (facts_text or task_description or '').strip()

    user_prompt: str = task_solve_with_insights.format(
        facts=facts,
        few_shots='\n'.join(few_shots),
        memory_few_shots=memory_few_shots_text,
        insights=existing_rules_text,
        law_context=law_ctx,
        task_description=task_description
    )
    return user_prompt


def format_task_context(
    facts: str,
    task_description: str | None = None,
    agent_steps: dict[str, str] | None = None,
    result_finish: str | None = None
) -> str:
    if agent_steps:
        try:
            agent_block = "\n".join([f"- {name}: {summary}" for name, summary in agent_steps.items()])
        except Exception:
            agent_block = ""
    else:
        agent_block = ""

    return task_format.format(
        facts=facts or (task_description or ''),
        agent_steps=agent_block,
        result_finish=result_finish or ''
    )
