from dataclasses import dataclass
import os
import numpy as np
from collections import deque

from mas.agents import Agent
from mas.memory.common import MASMessage, AgentMessage
from mas.mas import MetaMAS
from mas.reasoning import ReasoningBase, ReasoningConfig
from mas.memory import MASMemoryBase, CaseMemory
from mas.agents import Env
from mas.llm import Message

from .graph import GraphMaskInfo, gen_graph_mask_info
from .node import Node
from .graph_prompt import (
    build_verification_train_user,
    decision_system_prompt,
    solver_event_extraction,
    solver_judgment,
    solver_retrieval,
    solver_verification,
    solver_verification_train_system,
)
from ..format import format_task_context
from retrieval.law_retriever import LawArticleRetriever


@dataclass
class MacNet(MetaMAS):
    def __post_init__(self):
        self.observers = []
        self.reasoning_config = ReasoningConfig(temperature=0, stop_strs=None)
        self._agent_nodes: dict[int, Node] = {}
        self._decision_node: Node = None
        self._dataset_name: str = None

    def build_system(self, reasoning: ReasoningBase, mas_memory: MASMemoryBase, env: Env, config: dict):
        graph_type: str = config.get('graph_type', 'Debate')
        node_num: int = config.get('node_num', 4)
        self._max_rounds: int | None = config.get('round_num')
        self._use_critic: bool = config.get('use_critic', False)
        self._successful_topk: int = config.get('successful_topk', 1)
        self._failed_topk: int = config.get('failed_topk', 0)
        self._insights_topk: int = config.get('insights_topk', 3)
        self._threshold: float = config.get('threshold', 0)
        self._verify_threshold: float = float(config.get('verify_threshold', 0.7))
        self._use_projector: bool = config.get('use_projector', False)
        self._decision_max_tokens: int | None = config.get('decision_max_tokens')

        self.notify_observers(f"Configuration Loaded:")
        self.notify_observers(f"Node Number       : {node_num}")
        self.notify_observers(f"Graph Type        : {graph_type}")
        self.notify_observers(f"Use Critic        : {self._use_critic}")
        self.notify_observers(f"Successful Topk   : {self._successful_topk}")
        self.notify_observers(f"Failed Topk       : {self._failed_topk}")
        self.notify_observers(f"Insights Topk     : {self._insights_topk}")
        self.notify_observers(f"Retrieve Threshold: {self._threshold}")
        self.notify_observers(f"Verify Threshold   : {self._verify_threshold}")
        self.notify_observers(f"Use Role Projector: {self._use_projector}")
        self._run_mode: str = str(config.get('mode', 'test')).lower()
        self.notify_observers(f"Run Mode          : {self._run_mode}")
        self._training_mode_verification: bool = (self._run_mode == 'train') or bool(config.get('training_mode_verification', False))
        self.notify_observers(f"Training Mode (verification sees labels): {self._training_mode_verification}")
        self._dataset_name = config.get('task_name')
        self._disable_event: bool = bool(config.get('disable_event_agent', False))

        self._law_path: str = config.get('law_articles_path')
        self._law_initial_topk: int = int(config.get('laws_initial_topk', 10))
        self._law_topk: int = int(config.get('laws_topk', 5))
        self._laws_screen_min_k: int = int(config.get('laws_screen_min_k', 3))
        self._law_enabled: bool = bool(self._law_path)
        self._law_retriever: LawArticleRetriever | None = None
        if self._law_enabled:
            try:
                persist_dir = getattr(mas_memory, 'persist_dir', './.db/law')
                law_embed_model = str(config.get('law_embedding_model', mas_memory.embedding_func.model_type))
                law_embed_device = (
                    os.environ.get('LAW_EMBEDDING_DEVICE')
                    or config.get('law_embedding_device')
                    or os.environ.get('EMBEDDING_DEVICE')
                )
                try:
                    law_embed_bs = int(
                        os.environ.get('LAW_EMBEDDING_BATCH_SIZE', str(config.get('law_embedding_batch_size', os.environ.get('EMBEDDING_BATCH_SIZE', '16'))))
                    )
                except Exception:
                    law_embed_bs = 16
                self._law_retriever = LawArticleRetriever(
                    law_jsonl_path=self._law_path,
                    persist_dir=os.path.join(persist_dir, 'law_articles'),
                    embedding_model=law_embed_model,
                    embed_func=None,
                    embedding_device=law_embed_device,
                    embedding_batch_size=law_embed_bs,
                    use_bm25=bool(config.get('retrieval_use_bm25', True)),
                    bm25_top_k=int(config.get('retrieval_bm25_top_k', 50)),
                    faiss_top_k=int(config.get('retrieval_faiss_top_k', 100)),
                    fusion_alpha=float(config.get('retrieval_fusion_alpha', 0.7))
                )
                self._ranking_mode = str(config.get('retrieval_mode', 'rrf'))
                self.notify_observers(
                    f"Law retriever initialized. InitialTopK={self._law_initial_topk}, FilterTopK={self._law_topk}, mode={self._ranking_mode}, "
                    f"law_embed_model={law_embed_model}, device={law_embed_device}, bs={law_embed_bs}"
                )
            except Exception as e:
                self.notify_observers(f"Law retriever init failed: {e}")
                self._law_enabled = False

        self.compute_graph: GraphMaskInfo = gen_graph_mask_info(mode=graph_type, N=node_num)
        self._size: int = len(self.compute_graph.fixed_spatial_masks)
        self._agent_nodes, self._decision_node = self._init_nodes(reasoning)

        self._spatial_matrix: np.ndarray[bool] = self._construct_spatial_connection()
        self._temporal_matrix: np.ndarray[bool] = self._construct_temporal_connection()

        for node in self._agent_nodes.values():
            self.hire([node._agent])
        self.set_env(env)
        self.meta_memory = mas_memory

    def schedule(self, task_config: dict) -> tuple[float, bool]:
        role_cn = {
            'event': '事件抽取',
            'retrieval': '检索',
            'judgment': '判别',
            'verification': '验证',
            'decision': '最终决策'
        }
        def get_state_graph_upstream_node_ids(node: Node, upstream_node_ids: dict[str, str]) -> list[str]:
            upstream_ids: list[str] = []
            for n in node.spatial_predecessors:
                if n.id not in upstream_node_ids.keys():
                    raise ValueError('Upstream node should be in the `upstream_node_ids` dict.')
                upstream_ids.append(upstream_node_ids.get(n.id))
            return upstream_ids

        def _get_stored_law_ids() -> list[int]:
            try:
                stored_ids = self.meta_memory.current_task_context.get_extra_field('retrieved_law_ids') or []
            except Exception:
                stored_ids = []
            out: list[int] = []
            for x in stored_ids:
                try:
                    out.append(int(x))
                except Exception:
                    pass
            return out

        def _get_law_ids_from_retrieval_output(role_outputs: dict[str, str]) -> list[int]:
            candidates_text = (role_outputs or {}).get("retrieval", "") or ""
            if not candidates_text:
                return []
            import re
            m = re.search(r"(?is)finish\s*\[\s*\[\s*(.*?)\s*\]\s*\]", candidates_text)
            id_list_str = m.group(1) if m else ""
            if not id_list_str:
                m = re.search(r"(?is)finish\s*\[\s*.*?法条\s*:\s*\[(.*?)\]", candidates_text)
                id_list_str = m.group(1) if m else ""
            out: list[int] = []
            for s in re.findall(r"\d+", id_list_str or ""):
                try:
                    out.append(int(s))
                except Exception:
                    pass
            return out

        def _get_law_ids_from_context(text: str) -> list[int]:
            if not text:
                return []
            import re
            out: list[int] = []
            for s in re.findall(r"\bArticle\s*(\d{2,3})\b", text):
                try:
                    out.append(int(s))
                except Exception:
                    pass
            for s in re.findall(r"第\s*(\d{2,3})\s*条", text):
                try:
                    out.append(int(s))
                except Exception:
                    pass
            return out

        if task_config.get('task_main') is None:
            raise ValueError("Missing required keys `task_main` in task_config")
        if task_config.get('task_description') is None:
            raise ValueError("Missing required keys `task_description` in task_config")

        task_main: str = task_config.get('task_main')
        task_description: str = task_config.get('task_description')

        env = self.env
        env.reset()
        for node in self._agent_nodes.values():
            node.clear_state()
        self._decision_node.clear_state()
        self.meta_memory.init_task_context(task_main, task_description)
        try:
            expected_meta = (getattr(self.env, 'config', {}) or {}).get('expected')
        except Exception:
            expected_meta = None
        try:
            if expected_meta is not None:
                self.meta_memory.current_task_context.add_extra_field('expected', expected_meta)
        except Exception:
            pass

        successful_trajectories: list[MASMessage] = []
        insights: list[dict] = []
        successful_shots: list[str] = []
        raw_rules: list[str] = []
        extra_context = ""
        filtered_law_context = ""
        query_text = task_main
        if query_text and len(query_text) > 8192:
            query_text = query_text[:8192]
        if hasattr(self.meta_memory, 'retrieve_law_context'):
            try:
                extra_context = self.meta_memory.retrieve_law_context(
                    query_text,
                    top_k=getattr(self, '_law_initial_topk', 10)
                )
            except Exception as e:
                self.notify_observers(f"Law retrieval via memory failed: {e}")
                extra_context = ""

        if not extra_context and getattr(self, '_law_enabled', False) and getattr(self, '_law_retriever', None) is not None:
            try:
                candidates = self._law_retriever.retrieve(
                    query_text,
                    top_k=getattr(self, '_law_initial_topk', 10),
                    ranking_mode=getattr(self, '_ranking_mode', 'dense')
                )
                extra_context = self._law_retriever.format_context(candidates)
            except Exception as e:
                self.notify_observers(f"Law retrieval failed: {e}")
                extra_context = ""

        initial_candidate_ids: list[int] = []
        try:
            import re
            for m in re.finditer(r"-\s*Article\s+(\d+)", extra_context or ""):
                try:
                    initial_candidate_ids.append(int(m.group(1)))
                except Exception:
                    continue
        except Exception:
            initial_candidate_ids = []

        facts_text = task_main
        agents_task_desc = (
            "按角色协作完成任务：先进行要点提取，再进行信息检索，随后进行法条推荐与验证；最终由决策智能体输出数据集要求的 JSON。"
            if self._dataset_name == 'cail2018' else self.meta_memory.summarize(upstream_agent_ids=None)
        )

        user_prompt: str = (
            f"## Facts\nUse the following facts as the primary basis:\n{facts_text}\n---\n"
        )
        self.notify_observers(f"【上下文合成】角色任务描述：{agents_task_desc}")
        if successful_shots:
            self.notify_observers(f"【上下文合成】成功案例 Few-Shots 条数：{len(successful_shots)}")
        if raw_rules:
            self.notify_observers(f"【上下文合成】洞见/规则条数：{len(raw_rules)}")
        if extra_context:
            self.notify_observers("【上下文合成】已生成法条检索候选（不直接注入）")
        self.notify_observers(f"【上下文合成】基础 FACT 提示：\n{user_prompt}")

        max_rounds = min(env.max_trials, self._max_rounds or env.max_trials)
        prev_need_rerun_retrieval: bool = False
        prev_need_rerun_judgment: bool = False
        prev_carry_verify_suggestions: str = ""
        prev_carry_verify_thoughts: str = ""
        prev_round_judgment_raw: str = ""
        prev_round_verification_raw: str = ""
        for i in range(max_rounds):
            try:
                self.notify_observers(f"—— 回合开始：{i+1} ——")
            except Exception:
                pass
            upstream_node_ids: dict[str, str] = {}
            in_degree = {node.id: len(node.spatial_predecessors) for node in self._agent_nodes.values()}
            zero_in_degree_queue = [node_id for node_id, deg in in_degree.items() if deg == 0]
            allowed_roles: set[str] = {"event", "retrieval", "judgment", "verification"} if i == 0 else {"judgment", "verification"}
            role_outputs: dict[str, str] = {}
            role_raw_outputs: dict[str, str] = {}

            while zero_in_degree_queue:
                current_node_id = zero_in_degree_queue.pop(0)
                curr_node: Node = self._find_agent_node_by_uuid(current_node_id)
                try:
                    agent_index = next((idx for idx, n in self._agent_nodes.items() if n.id == curr_node.id), -1)
                except Exception:
                    agent_index = -1
                display_role = role_cn.get(curr_node._agent.profile, curr_node._agent.profile)
                role_name = curr_node._agent.profile
                # 角色输入不注入洞见，避免噪声；洞见仅在决策阶段统一注入

                # 若该回合不执行该角色，则复用上一回合输出，并将其写入内存以维持上游依赖链
                if role_name not in allowed_roles:
                    try:
                        self.notify_observers(
                            f"===== Agent[{agent_index}] {display_role}（{curr_node._agent.name}）====="
                        )
                        self.notify_observers("本回合跳过执行，复用上一回合输出")
                    except Exception:
                        pass

                    node_inputs, node_outputs = curr_node.memory
                    effective_user_prompt = node_inputs[-1] if node_inputs else user_prompt
                    action = node_outputs[-1] if node_outputs else ""
                    role_outputs[role_name] = action
                    agent_message: AgentMessage = AgentMessage(
                        agent_name=curr_node._agent.name,
                        system_instruction=curr_node._agent.system_instruction,
                        user_instruction=effective_user_prompt,
                        message=action
                    )
                    current_id: str = self.meta_memory.add_agent_node(
                        agent_message, upstream_agent_ids=get_state_graph_upstream_node_ids(curr_node, upstream_node_ids)
                    )
                    upstream_node_ids[curr_node.id] = current_id
                    try:
                        self.notify_observers(
                            f"输出（处理）：{action if action else '（空）'}"
                        )
                        self.notify_observers(
                            f"===== Agent[{agent_index}] End ====="
                        )
                    except Exception:
                        pass

                    for successor in curr_node.spatial_successors:
                        in_degree[successor.id] -= 1
                        if in_degree[successor.id] == 0:
                            zero_in_degree_queue.append(successor.id)
                    continue
                # 基础 FACT-only；判别角色采用精简提示词结构（事件要点 + 5候选 + 输出格式约束）
                used_event_points: bool = False
                facts_block: str = (
                    f"## Facts\nUse the following facts as the primary basis:\n{facts_text}\n---\n"
                )
                if role_name == "judgment":
                    event_points_raw: str = ""
                    try:
                        event_points_raw = role_outputs.get("event", "") or role_raw_outputs.get("event", "")
                        import re
                        m = re.search(r"(?is)finish\s*\[\s*(.*?)\s*\]", event_points_raw)
                        event_points_text: str = m.group(1) if m else event_points_raw
                    except Exception:
                        event_points_text = event_points_raw
                    fact_text_for_judge = event_points_text.strip() if (event_points_text and event_points_text.strip()) else facts_text
                    if i > 0 and prev_need_rerun_judgment:
                        has_verify_suggestions = bool(prev_carry_verify_suggestions)
                        candidates_for_judge = filtered_law_context.strip()
                        if not candidates_for_judge:
                            try:
                                topk = getattr(self, '_law_topk', 5)
                                fallback_ids = initial_candidate_ids[:topk] if initial_candidate_ids else []
                                arts_fb = self._law_retriever.get_articles_by_ids(fallback_ids) if fallback_ids else []
                                candidates_for_judge = self._law_retriever.format_context(arts_fb, max_chars=2000)
                            except Exception:
                                candidates_for_judge = extra_context.strip()
                        concise_user = (
                            "验证智能体认为你上一次给出的推荐法条或者解释可能不够准确。据法律事实和候选法条，参考验证代理给出的意见，重新推荐与判定此案最相关的法条。\n"
                            f"FACT:\n{facts_text}\n\n"
                            f"FACT(事件要点):\n{fact_text_for_judge}\n\n"
                            + ("\n## 验证代理意见:\n" + prev_carry_verify_suggestions + "\n" if has_verify_suggestions else "\n")
                            + f"CANDIDATES:\n{candidates_for_judge}\n\n"
                            "note1:给出推荐法条id和解释，格式为：{'predicted_article': , 'explanation': ''}\n"
                            "note2:按格式要求输出结果，不需要输出其他内容\n"
                            "note3:候选仅供参考；若候选无法准确覆盖，可选择候选之外的法条，但需在解释中说明理由。\n"
                        )
                    else:
                        candidates_for_judge = filtered_law_context.strip()
                        if not candidates_for_judge:
                            try:
                                topk = getattr(self, '_law_topk', 5)
                                fallback_ids = initial_candidate_ids[:topk] if initial_candidate_ids else []
                                arts_fb = self._law_retriever.get_articles_by_ids(fallback_ids) if fallback_ids else []
                                candidates_for_judge = self._law_retriever.format_context(arts_fb, max_chars=2000)
                                try:
                                    self.notify_observers(f"【判别回退】使用初始候选补足至{len(arts_fb)}条")
                                except Exception:
                                    pass
                            except Exception:
                                candidates_for_judge = extra_context.strip()
                        concise_user = (
                            "以下是一段原始法律事实、事件要点和检索代理筛选出的与裁定此案相关的候选法条，请结合这些信息，仔细分析本案被告的犯罪行为（理清主次、先后，不要仅仅考虑浅层语义相似性），推荐与裁定此案最相关的法条，并解释推荐理由。\n"
                            f"FACT:\n{facts_text}\n\n"
                            f"FACT(事件要点):\n{fact_text_for_judge}\n"
                            f"CANDIDATES:\n{candidates_for_judge}\n\n"
                            "note1:给出推荐法条id和解释，格式为：{'predicted_article': , 'explanation': ''}\n"
                            "note2:按格式要求输出结果，不要输出其他内容\n"
                            "note3:候选仅供参考；若候选无法准确覆盖，可选择候选之外的法条，但需在解释中说明理由。\n"
                        )
                    facts_block = concise_user
                    used_event_points = bool(event_points_text and event_points_text.strip())
                role_user_prompt_parts: list[str] = [facts_block]
                aux_types: list[str] = []
                if used_event_points:
                    aux_types.append("事件要点")
                if role_name == "judgment" and i > 0 and prev_need_rerun_judgment and prev_carry_verify_suggestions:
                    aux_types.append("验证建议")
                try:
                    if role_name == "judgment":
                        # 判别角色不再注入Few-Shots与洞见，保持输入精简
                        pass
                    elif role_name == "retrieval":
                        # 检索角色也默认不注入Few-Shots与洞见，仅依赖FACT与候选法条
                        pass
                except Exception:
                    pass
                # 让检索与判别角色都能看到“候选法条”上下文，检索角色将优先从候选中选择编号
                # 检索看到初始候选；判别只看到检索筛选后的法条
                if role_name == "retrieval" and extra_context:
                    role_user_prompt_parts.append("## Relevant Law Articles (Candidates)\n" + extra_context + "\n")
                    if initial_candidate_ids:
                        role_user_prompt_parts.append(
                            "## Candidate Law IDs (编号=法条条文号)\n" + 
                            ", ".join(str(x) for x in initial_candidate_ids) + "\n"
                        )
                    retrieval_rules = (
                        "## Retrieval Rules\n"
                        "- 精排依据：根据被告主观意图、犯罪行为手段、对象类型、结果与损害(未遂、既遂等或严重程度)等，仔细比对法条内对犯罪行为的叙述，与裁定此案越相关的法条越靠前；能约束目标罪名的条款优先；\n"
                        "- 在已给候选中选择你认为可能与裁定此案相关的法条，不做定罪或量刑结论。\n"
                        "- 若检索候选无法精准覆盖，可在必要时选择候选之外更匹配的法条（不限制数量）。\n"
                        "- 你仅给出最终结果即可，不需要给出任何分析。\n"
                        "## Output\nFinish[[编号1,编号2,编号3]]\n"
                    )
                    role_user_prompt_parts.append(retrieval_rules)
                    role_user_prompt_parts.append(
                            "note: Finish中给出真实法条编号，例如Finish[[272,384,185]]，不要输出[1,2,3]这类序号。\n"
                        )
                    aux_types.append("法条上下文")
                # 验证角色需要看到原始事实 + 事件要点 + 判别输出 + 候选法条，以便提出可执行建议
                if role_name == "verification":
                    # 事件要点文本（用于为验证提供对齐参考）
                    event_points_raw_v: str = role_outputs.get("event", "") or role_raw_outputs.get("event", "")
                    event_points_text_v: str = ""
                    try:
                        import re
                        mv = re.search(r"(?is)finish\s*\[\s*(.*?)\s*\]", event_points_raw_v)
                        event_points_text_v = mv.group(1) if mv else event_points_raw_v
                    except Exception:
                        event_points_text_v = event_points_raw_v
                    # 判别输出（用于验证一致性与建议方向）
                    judgment_out: str = role_outputs.get("judgment", "") or role_raw_outputs.get("judgment", "")
                    law_ctx = (filtered_law_context or extra_context) if (filtered_law_context or extra_context) else None
                    precedents_text: str = ""
                    insights_text: str = ""
                    try:
                        import re
                        ordered_ids = []
                        try:
                            stored_ids = self.meta_memory.current_task_context.get_extra_field('retrieved_law_ids') or []
                            for x in stored_ids:
                                try:
                                    ordered_ids.append(int(x))
                                except Exception:
                                    pass
                        except Exception:
                            ordered_ids = []
                        if not ordered_ids:
                            candidates_text_proc = role_outputs.get("retrieval", "") or ""
                            m_ids = re.search(r"(?is)finish\s*\[\s*\[\s*(.*?)\s*\]\s*\]", candidates_text_proc)
                            id_list_str = m_ids.group(1) if m_ids else ""
                            if not id_list_str:
                                m_ids = re.search(r"(?is)finish\s*\[\s*.*?法条\s*:\s*\[(.*?)\]", candidates_text_proc)
                                id_list_str = m_ids.group(1) if m_ids else ""
                            for s in re.findall(r"\d+", id_list_str or ""):
                                try:
                                    ordered_ids.append(int(s))
                                except Exception:
                                    pass
                        seen_ids = set()
                        top3_ids = []
                        for rid in ordered_ids:
                            if rid not in seen_ids:
                                top3_ids.append(rid)
                                seen_ids.add(rid)
                            if len(top3_ids) >= 3:
                                break
                        prec_items = []
                        query_clean = (task_main or '').strip()
                        try:
                            if query_clean.lower().startswith('fact:'):
                                query_clean = query_clean[len('fact:'):].strip()
                        except Exception:
                            pass
                        try:
                            ids_list = self.meta_memory.main_memory.get().get('ids') or []
                            prefetch_k = min(max(32, 8), max(1, len(ids_list)))
                            prefetch = self.meta_memory.main_memory.similarity_search_with_score(query=query_clean, k=prefetch_k)
                        except Exception:
                            prefetch = []
                        for aid in top3_ids:
                            best = None
                            for doc, dist in prefetch:
                                try:
                                    md = doc.metadata
                                    if md.get('label') is not True:
                                        continue
                                    msg = MASMessage.from_dict(md)
                                    obj = msg.get_extra_field('result_json') or {}
                                    arts = obj.get('relevant_articles') or []
                                    arts_int = set()
                                    for x in arts:
                                        try:
                                            arts_int.add(int(x))
                                        except Exception:
                                            pass
                                    if aid in arts_int:
                                        sim = 1.0 - float(dist)
                                        if (best is None) or (sim > best[0]):
                                            try:
                                                shot = format_task_context(
                                                    facts=(msg.get_extra_field('facts_summary') or msg.task_main or (msg.task_description or "")),
                                                    task_description=msg.task_description or "",
                                                    agent_steps=msg.get_extra_field('agent_steps') or {},
                                                    result_finish=msg.get_extra_field('result_finish') or ""
                                                )
                                            except Exception:
                                                shot = format_task_context(
                                                    facts=(msg.get_extra_field('facts_summary') or msg.task_main or (msg.task_description or "")),
                                                    task_description=msg.task_description or "",
                                                    result_finish=msg.get_extra_field('result_finish') or ""
                                                )
                                            best = (sim, f"- Article {aid}\n{shot}")
                                except Exception:
                                    continue
                            if best is not None:
                                prec_items.append(best[1])
                        if prec_items:
                            precedents_text = "\n\n".join(prec_items)
                        else:
                            precedents_text = ""
                        try:
                            retrieved_ids_set: set[int] = set()
                            try:
                                stored_ids = self.meta_memory.current_task_context.get_extra_field('retrieved_law_ids') or []
                                for x in stored_ids:
                                    try:
                                        retrieved_ids_set.add(int(x))
                                    except Exception:
                                        pass
                            except Exception:
                                retrieved_ids_set = set()
                            if not retrieved_ids_set:
                                candidates_text_proc = role_outputs.get("retrieval", "") or ""
                                m_ids = re.search(r"(?is)finish\s*\[\s*\[\s*(.*?)\s*\]\s*\]", candidates_text_proc)
                                id_list_str = m_ids.group(1) if m_ids else ""
                                if not id_list_str:
                                    m_ids = re.search(r"(?is)finish\s*\[\s*.*?法条\s*:\s*\[(.*?)\]", candidates_text_proc)
                                    id_list_str = m_ids.group(1) if m_ids else ""
                                for s in re.findall(r"\d+", id_list_str or ""):
                                    try:
                                        retrieved_ids_set.add(int(s))
                                    except Exception:
                                        pass
                            ins_map = {}
                            try:
                                for ins in getattr(self.meta_memory.insights_layer, 'insights_memory', []) or []:
                                    rt = ins.get('rule')
                                    if rt:
                                        ins_map[rt] = ins
                            except Exception:
                                ins_map = {}
                            def _arts_from_tasks(task_names: list[str]) -> set[int]:
                                im = getattr(self.meta_memory, 'insights_layer', None)
                                if im is None:
                                    return set()
                                return im.articles_from_tasks(task_names or [])
                            base_rules = []
                            try:
                                base_rules = [ins.get('rule') for ins in getattr(self.meta_memory.insights_layer, 'insights_memory', []) or []]
                            except Exception:
                                base_rules = []
                            scored = []
                            for r in base_rules:
                                s = 0
                                has_id_intersection = False
                                if retrieved_ids_set:
                                    try:
                                        ins = ins_map.get(r)
                                        if ins:
                                            src_tasks = ins.get('positive_correlation_tasks') or []
                                            src_arts = _arts_from_tasks(src_tasks)
                                            inter = src_arts & retrieved_ids_set
                                            if inter:
                                                has_id_intersection = True
                                                s += len(inter) + 1
                                    except Exception:
                                        has_id_intersection = False
                                if retrieved_ids_set and not has_id_intersection:
                                    continue
                                scored.append((s, r))
                            scored.sort(key=lambda x: x[0], reverse=True)
                            insights_text = "\n".join([r for _, r in scored[:self._insights_topk]]) if scored else ""
                        except Exception:
                            insights_text = insights_text
                    except Exception:
                        precedents_text = precedents_text
                    if self._training_mode_verification:
                        try:
                            import json, re
                            expected_meta = None
                            try:
                                expected_meta = (getattr(self.env, 'config', {}) or {}).get('expected')
                            except Exception:
                                expected_meta = None
                            gold_text = json.dumps(expected_meta or {}, ensure_ascii=False)
                            gold_law_text = None
                            try:
                                gold_ids = []
                                for s in (expected_meta or {}).get('relevant_articles') or []:
                                    try:
                                        gold_ids.append(int(s))
                                    except Exception:
                                        pass
                                if gold_ids and getattr(self, '_law_retriever', None) is not None:
                                    arts_full = self._law_retriever.get_articles_by_ids(gold_ids)
                                    gold_law_text = self._law_retriever.format_context(arts_full, max_chars=2000)
                            except Exception:
                                gold_law_text = None
                            verify_pack = build_verification_train_user(
                                facts=facts_text,
                                event_points=event_points_text_v or None,
                                judgment_out=judgment_out or None,
                                law_context=law_ctx,
                                gold_text=gold_text,
                                gold_law_text=gold_law_text
                            )
                            role_user_prompt_parts.append(verify_pack)
                            aux_types.append("判别输出")
                            aux_types.append("法条上下文")
                            if precedents_text:
                                role_user_prompt_parts.append("\n## Precedents\n" + precedents_text + "\n")
                                aux_types.append("先例")
                            if insights_text:
                                role_user_prompt_parts.append("\n## Insights\n" + insights_text + "\n")
                                aux_types.append("洞见")
                            aux_types.append("训练标签")
                        except Exception:
                            verify_pack = (
                                ("\n## Event Points\n" + (event_points_text_v or "") + "\n" if event_points_text_v else "") +
                                ("\n## Judgment Output\n" + (judgment_out or "") + "\n" if judgment_out else "") +
                                ("\n## Law Candidates\n" + (law_ctx or "") + "\n" if law_ctx else "")
                            )
                            role_user_prompt_parts.append(verify_pack)
                            aux_types.append("判别输出")
                            aux_types.append("法条上下文")
                            if precedents_text:
                                role_user_prompt_parts.append("\n## Precedents\n" + precedents_text + "\n")
                                aux_types.append("先例")
                            if insights_text:
                                role_user_prompt_parts.append("\n## Insights\n" + insights_text + "\n")
                                aux_types.append("洞见")
                    else:
                        # 常规测试模式提示
                        verify_pack = (
                            ("\n## Event Points\n" + (event_points_text_v or "") + "\n" if event_points_text_v else "") +
                            ("\n## Judgment Output\n" + (judgment_out or "") + "\n" if judgment_out else "") +
                            ("\n## Law Candidates\n" + (law_ctx or "") + "\n" if law_ctx else "")
                        )
                        role_user_prompt_parts.append(verify_pack)
                        aux_types.append("判别输出")
                        aux_types.append("法条上下文")
                        if precedents_text:
                            role_user_prompt_parts.append("\n## Precedents\n" + precedents_text + "\n")
                            aux_types.append("先例")
                        if insights_text:
                            role_user_prompt_parts.append("\n## Insights\n" + insights_text + "\n")
                            aux_types.append("洞见")
                if i > 0 and prev_carry_verify_thoughts:
                    if role_name in ("retrieval", "judgment"):
                        if role_name == "retrieval":
                            role_user_prompt_parts.append("## Verification Thoughts\n" + prev_carry_verify_thoughts + "\n")
                            aux_types.append("验证思考")
                if i > 0 and prev_carry_verify_suggestions:
                    if role_name == "retrieval" and prev_need_rerun_retrieval:
                        role_user_prompt_parts.append("## Verification Suggestions\n" + prev_carry_verify_suggestions + "\n")
                        aux_types.append("验证建议")
                role_user_prompt: str = "".join(role_user_prompt_parts)
                user_message: Message = Message('user', role_user_prompt)
                tries = 0
                # Reduce retries to 1 to cut repeated slow calls
                # 注入事件要点到检索角色的系统提示，强化一致性（保留原system_instruction，添加前缀）
                if role_name == "retrieval":
                    event_points_raw_r = role_outputs.get("event", "") or role_raw_outputs.get("event", "")
                    try:
                        import re
                        mr = re.search(r"(?is)finish\s*\[\s*(.*?)\s*\]", event_points_raw_r)
                        event_points_text_r = mr.group(1) if mr else event_points_raw_r
                    except Exception:
                        event_points_text_r = event_points_raw_r
                    if event_points_text_r:
                        curr_node._agent.system_instruction = (
                            "[事件要点参考]" + event_points_text_r + "\n\n" + (curr_node._agent.system_instruction or "")
                        )
                
                while tries < 1:
                    try:
                        # Agent 整块日志：统一展示序号、角色与输入
                        try:
                            self.notify_observers(
                                f"===== Agent[{agent_index}] {display_role}（{curr_node._agent.name}）====="
                            )
                            if aux_types:
                                self.notify_observers(
                                    f"输入（FACT+辅助）：{role_user_prompt}"
                                )
                                self.notify_observers(
                                    f"附加辅助内容：{', '.join(aux_types)}"
                                )
                            else:
                                self.notify_observers(
                                    f"输入（FACT）：{role_user_prompt}"
                                )
                            # 简要提示附加的上游来源，避免重复粘贴长文本
                            upstream_info = curr_node.get_spatial_upstream_info()
                            if upstream_info:
                                upstream_roles = sorted({info['role'] for info in upstream_info.values()})
                                self.notify_observers(
                                    f"附加上游来源：{', '.join(upstream_roles)}"
                                )
                        except Exception:
                            pass

                        raw_action: str = curr_node.execute(user_message, use_critic=self._use_critic)
                        if raw_action == '':
                            # Avoid infinite loop when model returns empty output
                            tries += 1
                            continue
                        # 记录原始输出（包含 Thought 与 Finish），用于后续注入验证思考
                        role_raw_outputs[role_name] = raw_action
                        # 原始输出
                        try:
                            self.notify_observers(
                                f"输出（原始）：{raw_action}"
                            )
                        except Exception:
                            pass

                        action = self.env.process_action(raw_action)
                        role_outputs[role_name] = action

                        # 若当前是检索角色，解析其输出中的法条编号，构造“筛选后的法条上下文”
                        if role_name == "retrieval" and getattr(self, '_law_retriever', None) is not None:
                            try:
                                import re
                                # 尝试从处理后的输出中提取编号列表（优先，因为结构规范）
                                candidates_text_proc = role_outputs.get("retrieval", "") or action or ""
                                m = re.search(r"(?is)finish\s*\[\s*\[\s*(.*?)\s*\]\s*\]", candidates_text_proc)
                                id_list_str = m.group(1) if m else ""
                                if not id_list_str:
                                    m = re.search(r"(?is)finish\s*\[\s*.*?法条\s*:\s*\[(.*?)\]", candidates_text_proc)
                                    id_list_str = m.group(1) if m else ""
                                ids: list[int] = []
                                if id_list_str:
                                    raw_ids = re.findall(r"\d+", id_list_str)
                                    for s in raw_ids:
                                        try:
                                            ids.append(int(s))
                                        except Exception:
                                            pass
                                # 若处理后未解析到，则兼容原始裸列表格式，例如 "[345, 397, 342, 413, 168]"
                                if not ids:
                                    candidates_text_raw = role_raw_outputs.get("retrieval", "") or ""
                                    if candidates_text_raw.strip().startswith("["):
                                        raw_ids = re.findall(r"\d+", candidates_text_raw)
                                        for s in raw_ids:
                                            try:
                                                ids.append(int(s))
                                            except Exception:
                                                pass
                                # 去重保持顺序（不在此处截断，统一在后续按 _law_topk 控制数量）
                                seen = set()
                                ordered_ids = []
                                for x in ids:
                                    if x not in seen:
                                        ordered_ids.append(x)
                                        seen.add(x)
                                # 记录Agent原样选择的ID序列，便于对比补全或过滤后的变化
                                try:
                                    self.notify_observers(f"【检索原样】Agent选择={ordered_ids}")
                                except Exception:
                                    pass
                                # 若输出的是候选列表序号（如1..10），将其映射为候选法条条号
                                try:
                                    init_ids = initial_candidate_ids or []
                                    init_set = set(init_ids)
                                    if ordered_ids and init_ids:
                                        has_overlap = any(x in init_set for x in ordered_ids)
                                        all_are_indices = all(1 <= x <= len(init_ids) for x in ordered_ids)
                                        if not has_overlap and all_are_indices:
                                            mapped_ids: list[int] = []
                                            for idx in ordered_ids:
                                                # 将序号1映射到init_ids[0]等
                                                true_id = init_ids[idx - 1]
                                                if true_id not in mapped_ids:
                                                    mapped_ids.append(true_id)
                                            # 使用映射后的真实法条条号
                                            ordered_ids = mapped_ids
                                            try:
                                                self.notify_observers(f"【检索筛选】检测到序号输出，已映射为法条编号：{ordered_ids}")
                                            except Exception:
                                                pass
                                except Exception:
                                    pass
                                # 若提供了初始候选，则强制仅在候选集合中选择
                                try:
                                    if initial_candidate_ids:
                                        init_set = set(initial_candidate_ids)
                                        before_len = len(ordered_ids)
                                        ordered_ids = [x for x in ordered_ids if x in init_set]
                                        if len(ordered_ids) < before_len:
                                            try:
                                                self.notify_observers(f"【检索筛选】已剔除非候选ID，保留：{ordered_ids}")
                                            except Exception:
                                                pass
                                except Exception:
                                    pass
                                # 通过检索器按ID取全文，并格式化为上下文；限制为5条，不足补全
                                topk = getattr(self, '_law_topk', 5)
                                selected_ids: list[int] = []
                                if ordered_ids:
                                    selected_ids = ordered_ids[:topk]
                                # 不足补全：从初始候选补足至 topk
                                if len(selected_ids) < topk:
                                    try:
                                        for cid in initial_candidate_ids:
                                            if len(selected_ids) >= topk:
                                                break
                                            if cid not in selected_ids:
                                                selected_ids.append(cid)
                                        if len(selected_ids) < topk and not initial_candidate_ids:
                                            # 若没有初始候选可用，回退到当前查询再检索 topk
                                            # 回退检索补全：FACT-only
                                            query_text = task_main
                                            mode = getattr(self, '_ranking_mode', 'dense')
                                            resc = self._law_retriever.retrieve(query_text, top_k=topk, ranking_mode=mode)
                                            for rid, _ in resc:
                                                if len(selected_ids) >= topk:
                                                    break
                                                if rid not in selected_ids:
                                                    selected_ids.append(rid)
                                    except Exception:
                                        pass
                                if selected_ids:
                                    arts = self._law_retriever.get_articles_by_ids(selected_ids)
                                    filtered_law_context = self._law_retriever.format_context(arts, max_chars=2000)
                                    
                                    try:
                                        extra_note = "（已补全至5条）" if len(ordered_ids) < topk else ""
                                        self.notify_observers(f"【检索筛选】IDs={selected_ids}; 条数={len(arts)} {extra_note}")
                                    except Exception:
                                        pass
                                    # 标签筛选用的编号集合：若检索输出不足 min_k，则仅补足到 min_k；否则保持原样
                                    try:
                                        screen_min = getattr(self, '_laws_screen_min_k', 3)
                                        screen_ids: list[int] = list(ordered_ids)
                                        if len(screen_ids) < screen_min:
                                            for cid in initial_candidate_ids:
                                                if len(screen_ids) >= screen_min:
                                                    break
                                                if cid not in screen_ids:
                                                    screen_ids.append(cid)
                                            if len(screen_ids) < screen_min and not initial_candidate_ids:
                                                try:
                                                    mode = getattr(self, '_ranking_mode', 'dense')
                                                    resc = self._law_retriever.retrieve(task_main, top_k=screen_min, ranking_mode=mode)
                                                    for rid, _ in resc:
                                                        if len(screen_ids) >= screen_min:
                                                            break
                                                        if rid not in screen_ids:
                                                            screen_ids.append(rid)
                                                except Exception:
                                                    pass
                                        # 写入用于标签筛选的编号集合（不强制到 _law_topk 数量）
                                        self.meta_memory.current_task_context.add_extra_field('retrieved_law_ids', screen_ids)
                                        try:
                                            self.notify_observers(f"【标签筛选法条IDs】用于筛选先例/洞见：{screen_ids}")
                                        except Exception:
                                            pass
                                    except Exception:
                                        pass
                                else:
                                    filtered_law_context = ""
                            except Exception as e:
                                try:
                                    self.notify_observers(f"【检索筛选】解析失败：{e}")
                                except Exception:
                                    pass
                        break
                    except Exception as e:
                        print(f"Error during execution of node {current_node_id}: {e}")
                        tries += 1

                # 规范化输出与结束标记
                try:
                    self.notify_observers(
                        f"输出（处理）：{action}"
                    )
                    self.notify_observers(
                        f"===== Agent[{agent_index}] End ====="
                    )
                except Exception:
                    pass

                # 将节点真实输入（FACT + 上游输出）写入内存，避免重复且便于溯源
                node_inputs, _ = curr_node.memory
                effective_user_prompt = node_inputs[-1] if node_inputs else user_prompt
                agent_message: AgentMessage = AgentMessage(
                    agent_name=curr_node._agent.name,
                    system_instruction=curr_node._agent.system_instruction,
                    user_instruction=effective_user_prompt,
                    message=action
                )
                current_id: str = self.meta_memory.add_agent_node(
                    agent_message, upstream_agent_ids=get_state_graph_upstream_node_ids(curr_node, upstream_node_ids)
                )
                upstream_node_ids[curr_node.id] = current_id

                for successor in curr_node.spatial_successors:
                    in_degree[successor.id] -= 1
                    if in_degree[successor.id] == 0:
                        zero_in_degree_queue.append(successor.id)

            # 记录各角色本回合的输入/输出到节点内存，供决策阶段按时间维度读取
            self._update_memory()

            # ====== 验证门控：解析结构化验证输出（score/pass/action/suggestions）以控制是否进入最终决策和下一回合动作 ======
            verify_text: str = role_outputs.get("verification", "") or ""
            verify_raw_text: str = role_raw_outputs.get("verification", "") or ""
            next_need_rerun_retrieval = False
            next_need_rerun_judgment = False
            next_carry_verify_suggestions = verify_text
            # 同步抽取验证 Thought 内容，供下一回合注入
            next_carry_verify_thoughts: str = ""
            try:
                import re
                m = re.search(r"(?is)thought\s*\[\s*(.+?)\s*\]", verify_raw_text)
                if not m:
                    m = re.search(r"(?is)thought\s*\(\s*(.+?)\s*\)", verify_raw_text)
                if not m:
                    m = re.search(r"(?i)thought\s*:\s*(.+)", verify_raw_text)
                if m:
                    next_carry_verify_thoughts = m.group(1).strip()
            except Exception:
                pass

            # 默认通过规则（兼容旧提示）：包含“正确”则通过；包含“修正/错误/不一致/存疑”等则不通过
            passed_by_text: bool = ("正确" in verify_text) and ("需要修正" not in verify_text and "不一致" not in verify_text and "错误" not in verify_text and "存疑" not in verify_text)

            # 新：尝试解析 JSON 风格的 Finish 参数，提取是否需要重判与建议
            parsed_need_rejudge: bool | None = None
            try:
                import json, re
                # 提取 Finish[...] 中的主体
                m = re.search(r"(?i)finish\s*\[\s*(.+?)\s*\]", verify_text)
                body = m.group(1) if m else verify_text
                candidate = body.strip()
                # 抽取 JSON 子串（若存在）
                jstart = candidate.find('{')
                jend = candidate.rfind('}')
                if jstart != -1 and jend != -1 and jend > jstart:
                    json_str = candidate[jstart:jend+1]
                else:
                    json_str = candidate
                obj = json.loads(json_str)
                if isinstance(obj, dict):
                    if 'need_rejudge' in obj:
                        parsed_need_rejudge = bool(obj.get('need_rejudge'))
                    if 'suggestions' in obj:
                        next_carry_verify_suggestions = str(obj.get('suggestions'))
            except Exception:
                # 保持向后兼容，不阻断流程
                pass

            # 仅使用 need_rejudge 控制是否重判；若未提供，则退回文本关键词判断
            if parsed_need_rejudge is not None:
                next_need_rerun_judgment = parsed_need_rejudge
            else:
                next_need_rerun_judgment = any(k in verify_text for k in ["重新判定", "判定修正", "再判", "修正判决"]) or ("need_rejudge" in verify_text.lower())
            # 不再支持重新检索流程
            next_need_rerun_retrieval = False

            # 将下一回合注入所需的信息保存为 prev_* 变量
            prev_need_rerun_retrieval = next_need_rerun_retrieval
            prev_need_rerun_judgment = next_need_rerun_judgment
            prev_carry_verify_suggestions = next_carry_verify_suggestions if next_need_rerun_judgment else ""
            prev_carry_verify_thoughts = next_carry_verify_thoughts
            try:
                if self._run_mode == 'train':
                    existing_sugs = self.meta_memory.current_task_context.get_extra_field('verify_suggestions') or []
                    if isinstance(existing_sugs, list):
                        if prev_carry_verify_suggestions:
                            existing_sugs.append(prev_carry_verify_suggestions)
                        self.meta_memory.current_task_context.add_extra_field('verify_suggestions', existing_sugs)
            except Exception:
                pass

            # 若未通过且仍有剩余回合，则跳过最终决策，进入下一回合
            passed = not next_need_rerun_judgment
            should_do_decision: bool = passed or (i == max_rounds - 1)
            if not should_do_decision:
                try:
                    self.notify_observers("验证未通过，进行再判")
                    if next_need_rerun_judgment:
                        self.notify_observers("验证建议：下一回合进行判定修正（默认已执行）")
                except Exception:
                    pass
                # 常规“回合结束”提示后继续下一回合
                try:
                    self.notify_observers(f"—— 回合结束：{i+1} ——")
                except Exception:
                    pass
                # 进入下一回合前，缓存本回合的判别/验证原始输出，供下一回合决策注入
                try:
                    prev_round_judgment_raw = role_raw_outputs.get("judgment", "") or prev_round_judgment_raw
                except Exception:
                    pass
                try:
                    prev_round_verification_raw = role_raw_outputs.get("verification", "") or prev_round_verification_raw
                except Exception:
                    pass
                continue

            # ====== 最终决策 ======
            self._connect_decision_node()
            # Decision stage: block-style logs like agents
            try:
                self.notify_observers(
                    f"===== 最终决策（{self._decision_node._agent.name}）====="
                )
                self.notify_observers(
                    f"{self._decision_node._agent.system_instruction}"
                )
            except Exception:
                pass

            # 在进入最终决策前，先根据检索代理输出的编号集合，检索“先例+洞见”，并按编号交集做筛选；
            # 同时采用变量缓存上一回合的判别与验证原始输出，若为第二轮则注入到决策输入。
            try:
                successful_trajectories, _, insights = self.meta_memory.retrieve_memory(
                    query_task=task_main,
                    successful_topk=self._successful_topk,
                    failed_topk=self._failed_topk,
                    insight_topk=self._insights_topk,
                    threshold=(float(self._threshold) if self._threshold is not None else 0.3),
                    importance=False
                )
            except Exception:
                successful_trajectories, insights = [], []
            # 统一格式化先例 Few-Shots
            successful_shots = []
            for traj in successful_trajectories:
                try:
                    shot = format_task_context(
                        facts=(traj.get_extra_field('facts_summary') or traj.task_main or (traj.task_description or "")),
                        task_description=traj.task_description or "",
                        agent_steps=traj.get_extra_field('agent_steps') or {},
                        result_finish=traj.get_extra_field('result_finish') or ""
                    )
                    successful_shots.append(shot)
                except Exception:
                    successful_shots.append(
                        format_task_context(
                            facts=(traj.get_extra_field('facts_summary') or traj.task_main or (traj.task_description or "")),
                            task_description=traj.task_description or "",
                            result_finish=traj.get_extra_field('result_finish') or ""
                        )
                    )
            raw_rules = [insight for insight in insights]
            try:
                if successful_shots:
                    self.notify_observers(f"【上下文合成】成功案例 Few-Shots 条数：{len(successful_shots)}")
                if raw_rules:
                    self.notify_observers(f"【上下文合成】洞见/规则条数：{len(raw_rules)}")
            except Exception:
                pass
            # 将法条候选上下文一并注入最终决策输入，帮助约束 relevant_articles 与 accusation 一致性
            decision_input_parts: list[str] = ["## Facts\n" + task_main + "\n"]
            retrieved_ids_set: set[int] = set(_get_stored_law_ids())
            if not retrieved_ids_set:
                for x in _get_law_ids_from_retrieval_output(role_outputs):
                    retrieved_ids_set.add(x)
            if not retrieved_ids_set:
                for x in _get_law_ids_from_context(filtered_law_context):
                    retrieved_ids_set.add(x)
            if filtered_law_context:
                decision_input_parts.append("## Relevant Law Articles (Filtered by Retrieval)\n" + filtered_law_context + "\n")
            # 先例按“检索候选的前三个法条编号”分组：为每个编号选择与当前案件最相似的一个成功先例
            try:
                import re
                # 解析检索候选的有序ID列表（优先使用内存中保存的筛选ID序列）
                ordered_ids: list[int] = _get_stored_law_ids()
                if not ordered_ids:
                    ordered_ids = _get_law_ids_from_retrieval_output(role_outputs)
                # 取前三个唯一编号
                seen_ids = set()
                top3_ids: list[int] = []
                for rid in ordered_ids:
                    if rid not in seen_ids:
                        top3_ids.append(rid)
                        seen_ids.add(rid)
                    if len(top3_ids) >= 3:
                        break
                # 预检索一批相似任务，供分组内选择最相似的先例
                prec_items: list[str] = []
                query_clean: str = (task_main or '').strip()
                try:
                    if query_clean.lower().startswith('fact:'):
                        query_clean = query_clean[len('fact:'):].strip()
                except Exception:
                    pass
                # 预抓取最多32条全局相似任务，减少多次IO
                try:
                    ids_list = self.meta_memory.main_memory.get().get('ids') or []
                    prefetch_k = min(max(32, 8), max(1, len(ids_list)))
                    prefetch = self.meta_memory.main_memory.similarity_search_with_score(query=query_clean, k=prefetch_k)
                except Exception:
                    prefetch = []
                # 为每个法条编号选出分数最高的成功先例
                for aid in top3_ids:
                    best = None
                    for doc, dist in prefetch:
                        try:
                            md = doc.metadata
                            if md.get('label') is not True:
                                continue
                            msg = MASMessage.from_dict(md)
                            obj = msg.get_extra_field('result_json') or {}
                            arts = obj.get('relevant_articles') or []
                            arts_int = set()
                            for x in arts:
                                try:
                                    arts_int.add(int(x))
                                except Exception:
                                    pass
                            if aid in arts_int:
                                sim = 1.0 - float(dist)
                                if (best is None) or (sim > best[0]):
                                    # 生成对应的 Few-Shot 文本
                                    try:
                                        shot = format_task_context(
                                            facts=(msg.get_extra_field('facts_summary') or msg.task_main or (msg.task_description or "")),
                                            task_description=msg.task_description or "",
                                            agent_steps=msg.get_extra_field('agent_steps') or {},
                                            result_finish=msg.get_extra_field('result_finish') or ""
                                        )
                                    except Exception:
                                        shot = format_task_context(
                                            facts=(msg.get_extra_field('facts_summary') or msg.task_main or (msg.task_description or "")),
                                            task_description=msg.task_description or "",
                                            result_finish=msg.get_extra_field('result_finish') or ""
                                        )
                                    best = (sim, f"- Article {aid}\n{shot}")
                        except Exception:
                            continue
                    if best is not None:
                        prec_items.append(best[1])
                if not prec_items:
                    # 回退：使用原有过滤逻辑的前3条或全部成功先例
                    fallback = "\n\n".join(successful_shots[:3]) if successful_shots else ""
                    prec_content = fallback
                else:
                    prec_content = "\n\n".join(prec_items)
            except Exception:
                prec_content = "\n\n".join(successful_shots[:3]) if successful_shots else ""
            decision_input_parts.append("## Retrieved Precedents (Examples)\n" + (prec_content if prec_content.strip() else "暂无可用先例") + "\n")
            # 基于检索编号集合，进一步过滤洞见候选（metadata-only，避免文本数字阈误）
            try:
                ins_map = {}
                try:
                    for ins in getattr(self.meta_memory.insights_layer, 'insights_memory', []) or []:
                        rt = ins.get('rule')
                        if rt:
                            ins_map[rt] = ins
                except Exception:
                    ins_map = {}
                def _arts_from_tasks(task_names: list[str]) -> set[int]:
                    im = getattr(self.meta_memory, 'insights_layer', None)
                    if im is None:
                        return set()
                    return im.articles_from_tasks(task_names or [])
                scored = []
                base_rules = list(raw_rules or [])
                # 若预取为空，使用全部洞见作为候选进行元数据打分兜底
                if not base_rules:
                    try:
                        base_rules = [ins.get('rule') for ins in getattr(self.meta_memory.insights_layer, 'insights_memory', []) or []]
                    except Exception:
                        base_rules = []
                for r in base_rules:
                    s = 0
                    tok_overlap = 0
                    if toks:
                        for t in toks:
                            if t and (t in r):
                                tok_overlap += 1
                    s += tok_overlap
                    has_id_intersection = False
                    if retrieved_ids_set:
                        try:
                            ins = ins_map.get(r)
                            if ins:
                                src_tasks = ins.get('positive_correlation_tasks') or []
                                src_arts = _arts_from_tasks(src_tasks)
                                inter = src_arts & retrieved_ids_set
                                if inter:
                                    has_id_intersection = True
                                    s += len(inter) + 1
                        except Exception:
                            has_id_intersection = False
                    if retrieved_ids_set and not has_id_intersection:
                        continue
                    scored.append((s, r))
                scored.sort(key=lambda x: x[0], reverse=True)
                ins_content = "\n".join([r for _, r in scored[:self._insights_topk]]) if scored else ""
            except Exception:
                ins_content = "\n".join(raw_rules[:self._insights_topk]) if raw_rules else ""
            decision_input_parts.append("## Insights\n" + (ins_content if ins_content.strip() else "暂无可用洞见") + "\n")
            # 按角色定制洞见（若启用 Projector），仅在最终决策阶段使用，避免在检索/判别阶段造成输入噪声
            try:
                roles_rules = self._project_insights([r for _, r in scored[:self._insights_topk]]) if scored else {}
            except Exception:
                roles_rules = {}
            guidance_text = (
                "量刑中的 `imprisonment` 单位为\"月\"；例如\"3年\"须输出36（个月）。"
                "实际的刑期取值要结合所选法条量刑区间与事实情节（如是否情节严重、是否造成严重后果）选择合理值。"
            )
            decision_input_parts.append("## Sentencing Guidance\n" + guidance_text + "\n")
            try:
                if i > 0:
                    prev_blocks = []
                    if prev_round_judgment_raw and prev_round_judgment_raw.strip():
                        prev_blocks.append("## Previous Round - Judgment\n" + prev_round_judgment_raw.strip() + "\n")
                    if prev_round_verification_raw and prev_round_verification_raw.strip():
                        prev_blocks.append("## Previous Round - Verification\n" + prev_round_verification_raw.strip() + "\n")
                    if prev_blocks:
                        decision_input_parts.append("# Outputs from previous round agents:\n" + "".join(prev_blocks))
            except Exception:
                pass
            # 若判别已给出明确的法条编号，向决策输入补充该法条原文；若已在筛选法条上下文中出现，则避免重复注入
            try:
                import re
                j_out = role_outputs.get("judgment", "") or role_raw_outputs.get("judgment", "")
                m = re.search(r"predicted_article\s*:\s*(\d+)", j_out)
                if m:
                    art_id = int(m.group(1))
                    if getattr(self, '_law_retriever', None) is not None:
                        already_in_filtered = bool(filtered_law_context) and (art_id in retrieved_ids_set)
                        if not already_in_filtered:
                            arts_full = self._law_retriever.get_articles_by_ids([art_id])
                            law_text = self._law_retriever.format_context(arts_full, max_chars=2000)
                            if law_text:
                                decision_input_parts.append("## Selected Law Article (Full Text)\n" + law_text + "\n")
            except Exception:
                pass
            # 决策输入日志：展示注入的辅助块，便于确认“先例/洞见/法条全文”是否进入上下文
            dec_aux: list[str] = []
            if filtered_law_context:
                dec_aux.append("Law articles")
            if successful_shots:
                dec_aux.append("Precedents")
            if raw_rules:
                dec_aux.append("Insights")
            try:
                import re
                j_out_chk = role_outputs.get("judgment", "") or role_raw_outputs.get("judgment", "")
                if re.search(r"predicted_article\s*:\s*\d+", j_out_chk):
                    dec_aux.append("法条全文")
            except Exception:
                pass

            # 分块打印决策输入内容（标签精简，英文化 Precedents/Insights；不展示检索 Thought）
            try:
                printed_precedents = False
                for part in decision_input_parts:
                    if part.startswith("## Facts"):
                        self.notify_observers("【FACT】\n" + part)
                    elif part.startswith("## Relevant Law Articles (Filtered by Retrieval)"):
                        self.notify_observers("【筛选法条上下文】\n" + part)
                    elif part.startswith("## Retrieved Precedents (Examples)"):
                        if not printed_precedents:
                            self.notify_observers("【Precedents】\n" + part)
                            printed_precedents = True
                    elif part.startswith("## Insights"):
                        self.notify_observers("【Insights】\n" + part)
                    elif part.startswith("## Selected Law Article (Full Text)"):
                        self.notify_observers("【法条全文】\n" + part)
            except Exception:
                pass

            try:
                upstream_info = self._decision_node.get_spatial_upstream_info()
                if upstream_info:
                    upstream_roles = sorted({info['role'] for info in upstream_info.values()})
                    self.notify_observers(f"附加上游来源：{', '.join(upstream_roles)}")
                    role_label = {
                        'event': '【上游-事件抽取】',
                        'retrieval': '【上游-检索】',
                        'judgment': '【上游-法条推荐】',
                        'verification': '【上游-验证】'
                    }
                    for _, info in upstream_info.items():
                        r = info.get('role', '')
                        out = info.get('output', '')
                        label = role_label.get(r, '【上游】')
                        self.notify_observers(label + "\n" + out)
            except Exception:
                pass

            decision_schema_requirements = (
                "根据法律事实，综合分析各智能体输出，相关法条选择上综合分析判别智能体和验证智能体的意见，并适当参考先例Precedents和经验Insights，给出本案最终裁定结果"
                "## Output Requirements\n"
                "- 所有答案不能为空值，必须给出一个你认为最合适的结果\n"
                "- 法条和罪名单标签约束：`relevant_articles` 只输出一个最相关的法条编号；`accusation` 只保留一个最合适的罪名，要参考相关法条给出的完整罪名术语如“走私、贩卖、运输、制造毒品罪”，不能写成“贩卖毒品罪”这种不完整形式。\n"
                "- 量刑：`death_penalty`/`life_imprisonment` 为bool值，分别对应死刑和无期,`imprisonment` 为int数，单位为‘月’，如3年则取值36，实际量刑结合相关法条的规定的量刑标准和案件情节合理推断。\n"
                "- 保持 JSON 完整闭合、无注释、无尾逗号；最终必须输出 Finish[JSON]，不允许仅 Thought。\n"
                "- 输出格式：Finish[{\"relevant_articles\": [int], \"accusation\": [str], \"term_of_imprisonment\": {\"death_penalty\": bool, \"life_imprisonment\": bool, \"imprisonment\": int}}]\n"
                "- 输出示例：Finish[{\"relevant_articles\": [133], \"accusation\": [\"交通肇事罪\"], \"term_of_imprisonment\": {\"death_penalty\": false, \"life_imprisonment\": false, \"imprisonment\": 30}}]\n"
            )
            decision_input_parts.append(decision_schema_requirements)
            decision_user_message = Message('user', "".join(decision_input_parts))
            try:
                if dec_aux:
                    self.notify_observers(
                        f"附加辅助内容：{', '.join(dec_aux)}"
                    )
                included_blocks = ["Facts", "Precedents", "Insights", "SentencingGuidance"]
                if filtered_law_context:
                    included_blocks.append("FilteredLaw")
                try:
                    import re
                    j_out_chk = role_outputs.get("judgment", "") or role_raw_outputs.get("judgment", "")
                    if re.search(r"predicted_article\s*:\s*\d+", j_out_chk):
                        included_blocks.append("SelectedLawFullText")
                except Exception:
                    pass
                try:
                    if i > 0:
                        if prev_round_judgment_raw and prev_round_judgment_raw.strip():
                            included_blocks.append("PrevRoundJudgment")
                        if prev_round_verification_raw and prev_round_verification_raw.strip():
                            included_blocks.append("PrevRoundVerification")
                    
                except Exception:
                    pass
                self.notify_observers("【决策输入】：" + ", ".join(included_blocks))
            except Exception:
                pass
            raw_decision = self._decision_node.execute(decision_user_message, use_critic=False)
            self._disconnect_decision_node()
            prev_round_judgment_raw = ""
            prev_round_verification_raw = ""

            decision_action = env.process_action(raw_decision)
            try:
                import re, json
                txt = (decision_action or '').strip().lower()
                if not txt.startswith('finish'):
                    fallback_needed = True
                    # 简化重试：当首次决策未给出Finish时，进行一次轻量重试，强制仅输出Finish JSON（不允许Thought），并注入上一次原始输出供参考
                    try:
                        minimal_parts = []
                        minimal_parts.append(user_prompt)
                        if raw_decision:
                            minimal_parts.append("## Previous Decision Raw Output\n" + str(raw_decision) + "\n")
                        # 将判别输出的法条编号作为强约束提示
                        try:
                            m2 = re.search(r"predicted_article\s*:\s*(\d+)", role_outputs.get("judgment", "") or role_raw_outputs.get("judgment", ""))
                            if m2:
                                minimal_parts.append("## Judgment Selected Law\n" + m2.group(1) + "\n")
                        except Exception:
                            pass
                        # 明确提示仅输出Finish JSON，不要Thought
                        minimal_parts.append("## Output Constraint\n从上次不完整的输出中提取出完整答案，只输出Finish[JSON]，不要Thought，不要解释。\n")
                        retry_msg = Message('user', "".join(minimal_parts))
                        raw_decision_retry = self._decision_node.execute(retry_msg, use_critic=False)
                        decision_action_retry = env.process_action(raw_decision_retry)
                        if (decision_action_retry or '').strip().lower().startswith('finish'):
                            decision_action = decision_action_retry
                            fallback_needed = False
                            self.notify_observers("【决策重试】首次未给出Finish，已用简化输入重试成功")
                        else:
                            self.notify_observers("【决策重试】简化输入仍未生成Finish，继续兜底")
                    except Exception:
                        pass
                    if fallback_needed:
                        import re, json
                        art_id = None
                        j_out = role_outputs.get("judgment", "") or role_raw_outputs.get("judgment", "")
                        m = re.search(r"predicted_article\s*:\s*(\d+)", j_out)
                        if m:
                            try:
                                art_id = int(m.group(1))
                            except Exception:
                                art_id = None
                        if art_id is None:
                            r_out = role_outputs.get("retrieval", "") or role_raw_outputs.get("retrieval", "")
                            ids = re.findall(r"\d+", r_out)
                            if ids:
                                try:
                                    art_id = int(ids[0])
                                except Exception:
                                    art_id = None
                        acc_text = ""
                        ver_text = role_outputs.get("verification", "") or role_raw_outputs.get("verification", "")
                        try:
                            m2 = re.search(r"([\u4e00-\u9fff]+罪)", (ver_text or "") + " " + (j_out or ""))
                            if m2:
                                acc_text = m2.group(1)
                        except Exception:
                            acc_text = ""
                        if art_id is not None:
                            payload = {
                                "relevant_articles": [art_id],
                                "accusation": ([acc_text] if acc_text else []),
                                "term_of_imprisonment": {"death_penalty": False, "life_imprisonment": False, "imprisonment": 0}
                            }
                            decision_action = f"Finish[{json.dumps(payload, ensure_ascii=False)}]"
                            try:
                                self.notify_observers("【兜底决策】检测到缺失Finish，已构造最小合法JSON")
                            except Exception:
                                pass
                else:
                    # 优先保留原始正确的 Finish，如果 env.process_action 误识别为 Thought，则直接从 raw_decision 中恢复 Finish
                    if not re.search(r"(?i)finish\s*\[", decision_action or "") and re.search(r"(?i)finish\s*\[", raw_decision or ""):
                        try:
                            # 直接使用原始文本的 Finish 片段
                            decision_action = env.process_action(raw_decision)
                            self.notify_observers("【修复】从原始输出恢复Finish，避免被Thought覆盖")
                        except Exception:
                            pass
            except Exception:
                pass
            try:
                self.notify_observers(
                    f"输出（原始）：{raw_decision}"
                )
                self.notify_observers(
                    f"输出（处理）：{decision_action}"
                )
                self.notify_observers(
                    f"===== 最终决策 End ====="
                )
            except Exception:
                pass

            observation, reward, done = env.step(decision_action)
            step_message: str = f'【环境反馈】reward={reward}, step_done={done}; {observation}'
            self.notify_observers(step_message)
            self.meta_memory.move_memory_state(decision_action, observation, reward=reward)
            if done:
                try:
                    self.notify_observers(f"—— 回合结束：{i+1}（done=True）——")
                except Exception:
                    pass
                break

            # Round end marker when not done
            try:
                self.notify_observers(f"—— 回合结束：{i+1} ——")
            except Exception:
                pass

            # 若进入下一回合，缓存本回合的判别/验证原始输出，供下一回合决策注入
            if not done and (i + 1) < max_rounds:
                try:
                    prev_round_judgment_raw = role_raw_outputs.get("judgment", "") or prev_round_judgment_raw
                except Exception:
                    pass
                try:
                    prev_round_verification_raw = role_raw_outputs.get("verification", "") or prev_round_verification_raw
                except Exception:
                    pass

        final_reward, final_done, final_feedback = self.env.feedback()
        try:
            exp = getattr(self.env, 'config', {}).get('expected')
            pred = getattr(self.env, 'last_pred', None)
            if exp is not None and pred is not None and hasattr(self.env, '_compare_predictions'):
                try:
                    all_ok, _ = self.env._compare_predictions(pred, exp)
                    final_done = bool(all_ok)
                except Exception:
                    pass
        except Exception:
            pass
        self.notify_observers(f"【最终评估】done={final_done}; {final_feedback}")
        if self._run_mode == 'train':
            self.meta_memory.save_task_context(label=final_done, feedback=final_feedback)
            self.meta_memory.backward(final_done)
        return final_reward, final_done

    def add_observer(self, observer):
        self.observers.append(observer)

    def notify_observers(self, message: str):
        for observer in self.observers:
            observer.log(message)

    def _update_memory(self) -> None:
        for node in self._agent_nodes.values():
            node.update_memory()

    def _find_agent_node_by_index(self, id: int) -> Node:
        return self._agent_nodes[id]

    def _find_agent_node_by_uuid(self, uuid: str) -> Node:
        for node in self._agent_nodes.values():
            if node.id == uuid:
                return node
        return None

    def _init_nodes(self, reasoning_module: ReasoningBase) -> tuple[dict[int, Node], Node]:
        system_node: dict[int, Node] = {}

        # 自定义四角色执行顺序：事件抽取 → 检索 → 判别 → 验证
        role_prompts = [
            ('event', solver_event_extraction),
            ('retrieval', solver_retrieval),
            ('judgment', solver_judgment),
            ('verification', solver_verification_train_system if getattr(self, '_training_mode_verification', False) else solver_verification)
        ]

        # 统一使用全局 LLM 配置中的默认生成长度
        try:
            from mas.llm import MAX_TOKEN as GLOBAL_MAX_TOKEN
        except Exception:
            GLOBAL_MAX_TOKEN = 512

        for index in range(min(self._size, len(role_prompts))):
            role_name, role_system_prompt = role_prompts[index]
            agent: Agent = Agent(
                name=f'{role_name}_{index}',
                role=role_name,
                system_instruction=role_system_prompt,
                reasoning_module=reasoning_module,
                memory_module=None
            )
            node: Node = Node(agent)
            # 统一各角色生成长度到全局默认（不再强制不同上限）
            node.reasoning_config = ReasoningConfig(temperature=0, max_tokens=GLOBAL_MAX_TOKEN, stop_strs=None)
            system_node[index] = node

        # 如果节点数超过4，则补充为“判决”角色的副本
        for index in range(len(system_node), self._size):
            agent: Agent = Agent(
                name=f'judgment_{index}',
                role='judgment',
                system_instruction=solver_judgment,
                reasoning_module=reasoning_module,
                memory_module=None
            )
            node: Node = Node(agent)
            # 额外判别节点也统一为全局默认生成长度
            node.reasoning_config = ReasoningConfig(temperature=0, max_tokens=GLOBAL_MAX_TOKEN, stop_strs=None)
            system_node[index] = node

        # 使用通用决策系统提示（包含量刑单位与取值规范）
        decision_prompt = decision_system_prompt

        decision_agent: Agent = Agent(
            name='final_decision',
            role='decision',
            system_instruction=decision_prompt,
            reasoning_module=reasoning_module
        )
        decision_node = Node(decision_agent)
        decision_node.reasoning_config = ReasoningConfig(temperature=0, max_tokens=1024, stop_strs=None)
        return system_node, decision_node
        
    def _clear_spatial_connection(self) -> None:
        for node in self._agent_nodes.values():
            node.clear_spatial_connections()
        self._decision_node.clear_spatial_connections()

    def _clear_temporal_connection(self) -> None:
        for node in self._agent_nodes.values():
            node.clear_temporal_connections()
        self._decision_node.clear_temporal_connections()

    def _connect_decision_node(self) -> None:
        for node in self._agent_nodes.values():
            Node.add_spatial_edge(node, self._decision_node)

    def _disconnect_decision_node(self) -> None:
        for node in self._agent_nodes.values():
            Node.remove_spatial_edge(node, self._decision_node)

    def _disconnect_dicision_node(self) -> None:
        self._disconnect_decision_node()

    def _construct_spatial_connection(self) -> np.ndarray[bool]:
        self._clear_spatial_connection()
        spatial_matrix = np.zeros((self._size, self._size), dtype=bool)
        for out_node_index, in_nodes in enumerate(self.compute_graph.fixed_spatial_masks):
            for in_node_index, connection in enumerate(in_nodes):
                if connection == 0:
                    continue
                src_node: Node = self._find_agent_node_by_index(out_node_index)
                tar_node: Node = self._find_agent_node_by_index(in_node_index)
                Node.add_spatial_edge(src_node, tar_node)
                if self._check_system_cycle():
                    Node.remove_spatial_edge(src_node, tar_node)
                else:
                    spatial_matrix[out_node_index][in_node_index] = 1
        return spatial_matrix

    def _construct_temporal_connection(self) -> np.ndarray[bool]:
        self._clear_temporal_connection()
        temporal_matrix = np.zeros((self._size, self._size), dtype=bool)
        for out_node_index, in_nodes in enumerate(self.compute_graph.fixed_temporal_masks):
            for in_node_index, connection in enumerate(in_nodes):
                if connection == 0:
                    continue
                src_node: Node = self._find_agent_node_by_index(out_node_index)
                tar_node: Node = self._find_agent_node_by_index(in_node_index)
                temporal_matrix[out_node_index][in_node_index] = 1
                Node.add_temporal_edge(src_node, tar_node)
        return temporal_matrix

    def _check_system_cycle(self) -> bool:
        frontier = deque()
        visited = set()
        in_degrees = {}
        for node in self._agent_nodes.values():
            in_degrees[node.id] = len(node.spatial_predecessors)
        for node_id, in_degree in in_degrees.items():
            if in_degree == 0:
                frontier.append(node_id)
                visited.add(node_id)
        while frontier:
            node_id = frontier.popleft()
            node = self._find_agent_node_by_uuid(node_id)
            for succ in node.spatial_successors:
                in_degrees[succ.id] -= 1
                if in_degrees[succ.id] == 0 and succ.id not in visited:
                    frontier.append(succ.id)
                    visited.add(succ.id)
        return len(visited) != self._size

    def _project_insights(self, insights: list[str]) -> dict[str, list[str]]:
        roles_rules: dict[str, list[str]] = {}
        roles = set([agent.profile for agent in self.agents_team.values()])
        if not self._use_projector or not isinstance(self.meta_memory, CaseMemory):
            for role in roles:
                roles_rules[role] = insights
        else:
            for role in roles:
                roles_rules[role] = self.meta_memory.project_insights(insights, role)
        for role, ins in roles_rules.items():
            roles_rules[role] = ins[:self._insights_topk]
        return roles_rules
