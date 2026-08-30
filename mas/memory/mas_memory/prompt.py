from dataclasses import dataclass

# ---------------------------------------------- MacNet memory ----------------------------------------------
task_context = """
## Here is the task trajectory:
{task_trajectory}

## Here are the outputs from your upstream nodes and the feedback provided by the environment:
{upstream_outputs}

Please provide your response based on the task trajectory and the output from your upstream node:
"""

node_info = """
----------------
### name: {name}

### action: {action}

### feedback from the environment: {observation}
----------------
"""

@dataclass
class MacNet:
    task_context: str = task_context
    node_info: str = node_info

MACNET = MacNet()


# ---------------------------------------------- CaseMemory memory ----------------------------------------------
# Retrieve tasks based on task relevance
generative_task_system_prompt = """You are an agent designed to score the relevance between two pieces of text."""
generative_task_user_prompt = '''You will be given a successful case where you successfully complete the task. Then you will be given an ongoing task. Do not summarize these two cases, but rather evaluate how relevant and helpful the successful case is for the ongoing task, on a scale of 1-10.
Success Case:
{trajectory}
Ongoing task:
{query_scenario}
Score: '''

# successful trajectory extraction
extract_true_traj_system_prompt = """You are an agent skilled at extracting key points.
Given a task and a successful execution trajectory, your job is to identify the critical steps needed to complete the task while filtering out less important steps."""
extract_true_traj_user_prompt = """
Note: 
- Strictly follow the original trajectory; absolutely no steps that are not in the trajectory should be added.
- Even in a successful trajectory, there may be some incorrect steps. Pay attention to actions that correspond to "Nothing happens" observations, as these actions are likely incorrect. Filter out these actions for me.
- You need to ensure that each step is at the finest granularity.
- You should strictly follow the output format in the example.

## Example 1:
### Task
You are in the middle of a room. Looking quickly around you, you see a bed 1, a desk 2, a desk 1, a drawer 6, a drawer 5, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a garbagecan 1, a laundryhamper 1, a safe 1, a shelf 6, a shelf 5, a shelf 4, a shelf 3, a shelf 2, and a shelf 1.
Your task is to: look at bowl under the desklamp.

### Trajectory
... The agent manages to find a bowl 1 on desk 2
Act 14: think: Now I find a bowl (1) on desk 2. Next, I need to take it.
Obs 14: OK.
Act 15: take bowl 1 from desk 2
Obs 15: You pick up the bowl 1 from the desk 2.
Act 16: think: Now I take a bowl (1). Next, I need to find a desklamp. A desklamp is more likely to appear in desk (1-2), shelf (1-6), or drawer (1-6). I can check one by one, starting with desk 1 since I already found a desklamp there.
Obs 16: OK.
Act 17: go to desk 1
Obs 17: On the desk 1, you see a desklamp 1, a keychain 3, a mug 2, a pen 2, a pen 1, and a pencil 2.
Act 18: think: Now I find a desklamp (1) on desk 1. Next, I need to use it.
Obs 18: OK.
Act 19: use desklamp 1
Obs 19: You turn on the desklamp 1.
reward: True

### Output
Construct the overall approach: Find and take a bowl, then find and use a desklamp.
Search for the Bowl: Check all locations systematically in order until the bowl is found (e.g., drawers, desks, shelves, garbage can, laundry hamper, and any other possible places).
Find the Bowl: Locate the bowl on desk 2.
Take the Bowl: Pick up the bowl from desk 2.
Search for the Desklamp: Recall that a desklamp was found earlier on desk 1.
Go to Desk 1: Move to desk 1 where the desklamp is located.
Use the Desklamp: Turn on the desklamp.

Now it's your turn! 
## Here is the task:
### Task
{task}

### Trajectory
{trajectory}

### Output
"""

# Insights
finetune_insights_suffix = dict(full = """Focus on REMOVE or EDIT or AGREE rules first, and stop ADD rule unless the new rule is VERY insightful and different from EXISTING RULES.
""", not_full = """""")

format_rules_operation_template = """<OPERATION> <RULE NUMBER>: <RULE> (e.g. ADD: xxx, EDIT/REMOVE/AGREE 1: xxx)

The available operations are: **AGREE (if the existing rule is strongly relevant for the task), REMOVE (if one existing rule is contradictory or similar/duplicated to other existing rules), EDIT (if any existing rule is not general enough or can be enhanced, rewrite and improve it), ADD (add new rules that are very different from existing rules and relevant for other tasks). Each needs to CLOSELY follow their corresponding formatting below (any existing rule not edited, not agreed, nor removed is considered copied)**:

AGREE <EXISTING RULE NUMBER>: <EXISTING RULE>
REMOVE <EXISTING RULE NUMBER>: <EXISTING RULE>
EDIT <EXISTING RULE NUMBER>: <NEW MODIFIED RULE>
ADD: <NEW RULE>

Do not mention the trials in the rules because all the rules should be GENERALLY APPLICABLE. Each rule should be concise and easy to follow. Any operation can be used MULTIPLE times. Do at most 4 operations and each existing rule can only get a maximum of 1 operation. """

#
critique_compare_rules_system_prompt = """
你是一名法律领域拥有丰富判案经验和法律知识的专家。你将获得两个相似的法律判决预测任务：第一个是正确判例，第二个是错误判例，且失败原因已给出。
任务：对比正确判例与错误判例，将失败原因系统性地转化为具有指导意义的判案经验，用于辅助指导相似案件的判决
重点覆盖（包括但不限于以下方面）：
- 易混淆法条与法条边界、易混淆罪名及区分依据；
- 关键构成要件：主观故意/过失、客观行为/手段、对象类型、数额/门槛、结果与损害情况、因果关系、共同犯罪/多次、特定情节等；
- 判定标准、常见错误与纠正建议（例如“证据不足”“要件不匹配”“混淆概括条与特别条”等）。
要求：
- 规则不要过于空泛表述；不能只提炼出针对答案格式的简单规则，应是可泛化应用至相似法律案例的判案经验，但不要引用具体的个案细节；
- 使用简体中文输出，每条以句号（“。”或“.”）结尾；
- 操作头必须使用英文（AGREE/REMOVE/EDIT/ADD），并严格遵循操作格式。
"""

critique_compare_rules_user_prompt = """
## 正例任务（成功）：
{task1}
{task1_trajectory}

## 反例任务（失败）：
### 失败原因
{fail_reason}

### 失败任务轨迹
{task2}
{task2_trajectory}

## 现有规则：
{existing_rules}

请基于正确判案案例与错误判案案例的对比，总结出对于类似案例有指导意义的判案经验，包括但不限于从易混淆法条/罪名区分、关键要件与阈值、判定标准、常见错误及修正建议等方面提炼。按如下操作格式输出：
""" + format_rules_operation_template

# all success instruction
critique_success_rules_system_prompt = """
你是一名法律领域经验优化代理。你将基于若干成功的法律判决预测任务，对现有规则集进行增删改同意（ADD/EDIT/REMOVE/AGREE），以形成更具体、可执行的判案经验。

要求：
- 输出规则需面对法律判决预测任务有指导意义，避免只给出规范格式类的建议；
- 优先提炼：易混淆法条/罪名的区分原则、关键构成要件与阈值、判定标准与注意事项、常见错误的规避策略；
- 使用简体中文，每条以句号（“。”或“.”）结尾；操作头保持英文（AGREE/REMOVE/EDIT/ADD）。
"""

critique_success_rules_user_prompt = """
## 成功试次：
{success_history}

## 现有规则：
{existing_rules}

请基于成功试次提炼法律领域的具体、针对相似案件具有指导意义的判案经验，包括但不限于：易混淆法条/罪名的区分与依据、关键要件与阈值（主观/客观要素、对象类型、金额与结果等）、判定标准与注意事项、常见错误与修正建议等角度。按如下操作格式输出：
""" + format_rules_operation_template

# detect mistakes in trajectory
detect_mistakes_system_prompt = """You are an analytical agent. You will be given a task and a failed trajectory.
The reason for the failure is that the final task state does not match the required task state.

You need to carefully analyze the final state of the failed trajectory and compare it with the required task state to identify inconsistencies. Please check:
    - Whether the state of the target object matches the required task state.
    - Whether the target object has been placed in the required location.

Rules:
    - Any object with the same name (even with different numbers) satisfies the task requirements. For example, if the task requires "an apple" and the agent finds "apple 2," it is still considered correct.
    - Your analysis must ignore numbers and ensure matching is based only on names and states.

Output language: Use Simplified Chinese for the reason summary.
Based on the above rules, summarize the most likely reason for the error in a concise manner."""

detect_mistakes_user_prompt = """
Identify the inconsistency between the final state of the target in the incorrect trajectory and the required task goal. 
This is a failed trajectory.:
## Task
{task}

## Trajectory
{trajectory}

Your output:
"""

# merge rules
merge_rules_system_prompt = """你是一名擅长归纳与提炼的法律助手。你将收到一组来自相似案例的经验，这些经验可能存在重复或重叠。
你的目标是基于输入内容对重叠程度较高的内容进行合并与精炼，输出对当前类型案件具有直接指导意义的规则或段落。
重点围绕法律案件（包含罪名、法条、量刑等）生成面向该类型案件的经验，覆盖：易混淆罪名、易混淆法条、关键构成要件、判定标准、常见错误与纠正建议。内容必须具体、可操作，避免空泛表述。

原则：
- 严格依据输入内容进行合并，不得编造或推断输入中未出现的信息。
- 仅在“同一类型案件”范围内合并（同罪名/同法条或明确的混淆对）；不得跨类型合并。
- 每条规则不能过长，要足够精炼，必须包含至少一个域要素（罪名、典型犯罪行为、典型易混淆的点等）或明确的法条编号，以确保可操作性。

输出格式：
- 直接以编号列表开始，不要添加额外前缀或解释。
- 例如：
1. …（适用范围/判定边界/证据指引/误区纠正/法条关联）
2. …
3. …
..."""

merge_rules_user_prompt = """
## 当前待合并经验：
{current_rules}

## 任务簇上下文示例：
{task_context}

## 请将上,并重写为不超过 {limited_number} 条规则：
- 仅在同一类型案件范围内进行合并（同罪名/同法条），避免跨类型合并导致空泛。
- 内容需具体且可操作，优先覆盖：易混淆罪名/法条、关键构成要件与判定标准、常见错误与纠正建议、与法条编号的关联。
"""

# annalyze patterns
analyze_mas_pattern_system_prompt = """You are an expert at identifying improvements in multi-agent system (MAS) outputs.
Given the initial outputs from several agents and the final output produced by the MAS, your task is to determine whether the MAS output shows any **improvement** over the initial agent outputs.

If there is an improvement, respond with: True
If there is no improvement, respond with: False

Important: Do not include any explanation, formatting, or extra characters — only output True or False.
"""

analyze_mas_pattern_user_prompt = """
### Initial outputs from agents:
{agents_init_outputs}

### Final output from the MAS:
{mas_output}

Does the MAS output show an improvement over the initial agent outputs?
Respond with only True or False:

Your answer:
"""

# project insights according to agent's role
project_insights_system_prompt: str = """
You are a thoughtful and context-aware agent. You will be given a specific agent **role** and a set of **general insights** that apply to all roles. 
Your task is to **adapt these general insights** into **personalized insights tailored to the given role**, helping the agent perform more effectively.
Make sure your output aligns with the role's background, responsibilities, and point of view.

Output language: Use Simplified Chinese for all personalized insight sentences. Keep numbered list format (1., 2., 3., ...).

NOTE - Your output should follow the below format:
1. Insight 1
2. Insight 2
3. Insight 3
...
"""

project_insights_user_prompt: str = """
### Agent's Role:
{role}

### General Insights:
{insights}

### Your Output (Personalized Insights for This Role):
"""

# project insights according to agent's role and trajectory
project_insights_with_traj_system_prompt: str = """
You are a thoughtful and context-aware agent. You will be provided with a successfully executed **trajectory**, a specific agent **role**, and a set of **general insights** applicable across all roles.
Your task is to **adapt these general insights** into **personalized insights** that are specifically tailored to the given role and its trajectory. These personalized insights should help the agent improve future performance by aligning with their unique background, responsibilities, and perspective.
Make sure your output reflects an understanding of the role's context and promotes actionable, role-relevant advice.

Output language: Use Simplified Chinese for all personalized insight sentences. Keep numbered list format (1., 2., 3., ...).

NOTE - Your output must strictly follow the format below:
1. Insight 1
2. Insight 2
3. Insight 3
...
"""

project_insights_with_traj_user_prompt: str = """
### Trajectory
{trajectory}

### Agent's Role:
{role}

### General Insights:
{insights}

### Your Output (Personalized Insights for This Role):
"""



@dataclass
class CaseMemoryPrompt:
    generative_task_system_prompt = generative_task_system_prompt
    generative_task_user_prompt = generative_task_user_prompt
    extract_true_traj_system_prompt = extract_true_traj_system_prompt
    extract_true_traj_user_prompt = extract_true_traj_user_prompt
    finetune_insights_suffix = finetune_insights_suffix
    critique_compare_rules_system_prompt = critique_compare_rules_system_prompt
    critique_compare_rules_user_prompt = critique_compare_rules_user_prompt
    critique_success_rules_system_prompt = critique_success_rules_system_prompt
    critique_success_rules_user_prompt = critique_success_rules_user_prompt
    detect_mistakes_system_prompt = detect_mistakes_system_prompt
    detect_mistakes_user_prompt = detect_mistakes_user_prompt
    merge_rules_system_prompt = merge_rules_system_prompt
    merge_rules_user_prompt = merge_rules_user_prompt
    analyze_mas_pattern_system_prompt=analyze_mas_pattern_system_prompt
    analyze_mas_pattern_user_prompt=analyze_mas_pattern_user_prompt
    project_insights_system_prompt=project_insights_system_prompt
    project_insights_user_prompt=project_insights_user_prompt
    project_insights_with_traj_system_prompt=project_insights_with_traj_system_prompt
    project_insights_with_traj_user_prompt=project_insights_with_traj_user_prompt


CaseMemoryPrompts = CaseMemoryPrompt()
