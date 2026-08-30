from dataclasses import dataclass, replace
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from chromadb.config import Settings as ChromaSettings
import os
import copy
import re
from typing import Iterable
import random
from collections import defaultdict
import networkx as nx
import numpy as np
from finch import FINCH
import pickle
import networkx as nx
import logging

from .memory_base import MASMemoryBase
from ..common import MASMessage, StateChain
from ..utils import cosine_similarity, export_graph_png
from .prompt import CaseMemoryPrompts
from mas.utils import load_json, write_json, random_divide_list, InterProcessFileLock
from mas.llm import LLMCallable, Message

@dataclass
class CaseMemory(MASMemoryBase):
    """
    CaseMemory: Tracing Hierarchical Memory for Multi-Agent Systems
    A three-tier hierarchical graph structure compo sed of the Insight Graph, Query Graph, and Interaction Graph.

    1. Interaction Graph - Trajectory Condensation: During the task-solving process, the multi-agent system (MAS) generates a chain of states, where each state represents a step in the process of arriving at the final answer. Behind each state is a corresponding message graph.
       Each task corresponds to a chain of states, which connects the middle and bottom layers of the multi-layer graph.
    2. Query Graph - Based on the current task, the system retrieves previously successful records. A k-hop approach is used to expand the search scope within the query graph.
    3. Insight Graph - Insights Retrieval: Relevant insights are retrieved based on the current task to assist in decision-making.
    """
    def __post_init__(self):
        super().__post_init__()
        
        if Chroma is None:
            raise RuntimeError("Chroma is not available. Please install langchain_chroma.")
        client_settings = None
        try:
            if ChromaSettings is not None:
                try:
                    client_settings = ChromaSettings(anonymized_telemetry=False)
                except Exception:
                    pass
        except Exception:
            client_settings = None
        
        os.environ['ANONYMIZED_TELEMETRY'] = 'False'
        os.environ['CHROMA_SERVER_NO_ANALYTICS'] = 'True'
        os.environ['CHROMA_PRODUCT_TELEMETRY_IMPL'] = 'disabled'
        os.environ['POSTHOG_DISABLED'] = 'true'
        try:
            import chromadb.telemetry.product.posthog as _ch_posthog
            def _noop_capture(*args, **kwargs):
                return None
            _ch_posthog.capture = _noop_capture
        except Exception:
            pass
        try:
            import logging as _lg
            _lg.getLogger('chromadb.telemetry.product.posthog').setLevel(_lg.CRITICAL)
        except Exception:
            pass
        
        try:
            if client_settings is not None:
                self.main_memory = Chroma(
                    embedding_function=self.embedding_func,
                    persist_directory=self.persist_dir,
                    client_settings=client_settings
                )
            else:
                self.main_memory = Chroma(
                    embedding_function=self.embedding_func,
                    persist_directory=self.persist_dir
                )
        except TypeError:
            self.main_memory = Chroma(
                embedding_function=self.embedding_func,
                persist_directory=self.persist_dir
            )
        # Shared lock for Chroma persistence across processes
        self._chroma_lock_path: str = os.path.join(self.persist_dir, 'chroma.lock')

        self._hop: int = self.global_config.get('hop', 1)
        self._start_insights_threshold: int = self.global_config.get('start_insights_threshold', 10)
        self._rounds_per_insights: int = self.global_config.get('rounds_per_insights', 5) 
        self._insights_point_num: int = self.global_config.get('insights_point_num', 2)
        try:
            self._insights_sem_batch: int = int(self.global_config.get('insights_sem_batch', 16))
        except Exception:
            self._insights_sem_batch = 16
        try:
            self._embed_text_clip: int = int(self.global_config.get('embed_text_clip', 4096))
        except Exception:
            self._embed_text_clip = 4096

        self.task_layer = TaskLayer(
            working_dir=self.persist_dir,
            namespace='task_layer', 
            task_storage=self.main_memory
        )
        
        self.insights_cache: list[str] = []
        # Cache for insight embeddings to avoid re-vectorization
        self._insight_embedding_cache: dict[str, list[float]] = {}

        self.insights_layer = InsightsManager(
            working_dir=self.persist_dir, 
            namespace='insights', 
            llm_model=self.llm_model, 
            task_storage=self.main_memory,
            task_layer=self.task_layer,
            insights_sem_batch=self._insights_sem_batch,
            embed_text_clip=self._embed_text_clip,
            embedding_cache=self._insight_embedding_cache
        )

        try:
            _ = self.embedding_func.embed_query("warmup")
        except Exception:
            pass

        print(self._get_hyperparams_dict())
    
    def _get_hyperparams_dict(self) -> dict:
        return {
            'hop': self._hop,
            'start_insights_threshold': self._start_insights_threshold,
            'rounds_per_insights': self._rounds_per_insights,
            'insights_point_num': self._insights_point_num,
            'working_dir': self.persist_dir,
            'insights_sem_batch': self._insights_sem_batch,
            'embed_text_clip': self._embed_text_clip
        }


    def add_memory(self, mas_message: MASMessage) -> None:
        """
        Add the mas_message corresponding to a completed task into memory:
        Step 1: Sparsification - remove incorrect steps
        Step 2: Add the sparsified trajectories to memory
        Step 3: If the number of steps in memory reaches a certain threshold, perform fine-tuning on the insights in memory

        Args:
            mas_message (MASMessage): The MAS message corresponding to a completed task

        Raises:
            ValueError: mas_message must have label!
        """
        # sparsification
        mas_message = self._extract_mas_message(mas_message=mas_message)  
        
        # add into memory
        self.task_layer.add_task_node(mas_message.task_main)

        meta_data: dict = MASMessage.to_dict(mas_message)
        memory_doc = Document(
            page_content=mas_message.task_main,   
            metadata=meta_data
        )
        def _is_strict_success(msg: MASMessage) -> bool:
            try:
                exp = msg.get_extra_field('expected') or {}
                res = msg.get_extra_field('result_json') or {}
                acc_exp = exp.get('accusation') or []
                acc_res = res.get('accusation') or []
                arts_exp = exp.get('relevant_articles') or []
                arts_res = res.get('relevant_articles') or []
                acc_ok = bool(acc_exp) and bool(acc_res) and (set(acc_exp) == set(acc_res))
                arts_ok = bool(arts_exp) and bool(arts_res) and (set(int(x) for x in arts_exp) == set(int(y) for y in arts_res))
                return acc_ok and arts_ok
            except Exception:
                return False
        def _is_complete_success(msg: MASMessage) -> bool:
            try:
                if not _is_strict_success(msg):
                    return False
                exp = msg.get_extra_field('expected') or {}
                res = msg.get_extra_field('result_json') or {}
                def _months_to_bucket(months: int) -> int:
                    try:
                        m = int(months)
                    except Exception:
                        m = 0
                    if m <= 0:
                        return 10
                    if m <= 6:
                        return 9
                    if m <= 9:
                        return 8
                    if m <= 12:
                        return 7
                    if m <= 24:
                        return 6
                    if m <= 36:
                        return 5
                    if m <= 60:
                        return 4
                    if m <= 84:
                        return 3
                    if m <= 120:
                        return 2
                    return 1
                if 'term' in exp:
                    be = int(exp.get('term'))
                    gb = be
                else:
                    te = exp.get('term_of_imprisonment') or {}
                    if bool(te.get('death_penalty')) or bool(te.get('life_imprisonment')):
                        gb = 0
                    else:
                        gb = _months_to_bucket(te.get('imprisonment'))
                if 'term' in res:
                    br = int(res.get('term'))
                    rb = br
                else:
                    tr = res.get('term_of_imprisonment') or {}
                    if bool(tr.get('death_penalty')) or bool(tr.get('life_imprisonment')):
                        rb = 0
                    else:
                        rb = _months_to_bucket(tr.get('imprisonment'))
                return gb is not None and rb is not None and int(gb) == int(rb)
            except Exception:
                return False
        try:
            mas_message.add_extra_field('strict_success', _is_strict_success(mas_message))
            mas_message.add_extra_field('complete_success', _is_complete_success(mas_message))
        except Exception:
            pass

        if mas_message.label == True or mas_message.label == False:
            # Single-writer lock for Chroma updates to avoid DB-level contention
            try:
                with InterProcessFileLock(self._chroma_lock_path, timeout=180):
                    self.main_memory.add_documents([memory_doc])
            except TimeoutError as e:
                print(f"CaseMemory: lock timeout when adding to Chroma -> {e}")
        else:
            raise ValueError('The mas_message must have label!')
        
        # finetune and merge insights
        if self.memory_size >= self._start_insights_threshold and self.memory_size % self._rounds_per_insights == 0:
            self.insights_layer.finetune_insights(self._insights_point_num)
        if self.memory_size % 50 == 0: 
            self.insights_layer.merge_insights() 

        self._index_done()

    def _retrieve_memory_raw(
        self, 
        query_task: str,   
        successful_topk: int = 1, 
        failed_topk: int = 1, 
        insight_windows: int = 10,
        threshold: float = 0.3
    ) -> tuple[list, list, list]:

        def sort_and_filter_by_similarity(docs_with_score: list[tuple[Document, float]], threshold: float = 0.3) -> list[tuple[Document, float]]:
            if Document is None:
                raise RuntimeError("LangChain Document is unavailable.")
            result = []
            for doc, dist in docs_with_score:
                try:
                    sim = 1.0 - float(dist)
                except Exception:
                    sim = 0.0
                if sim >= threshold:
                    result.append((doc, sim))
            result.sort(key=lambda x: x[1], reverse=True)
            return result

        true_tasks_doc: list[tuple[Document, float]] = []
        false_tasks_doc: list[tuple[Document, float]] = []
        
        # find related tasks in task layer（统一用清洗后的 FACT 文本进行相似度检索）
        query_clean: str = (query_task or '').strip()
        try:
            if query_clean.lower().startswith('fact:'):
                query_clean = query_clean[len('fact:'):].strip()
        except Exception:
            pass
        related_point_num: int = max((successful_topk + failed_topk) // 2, 1)
        task_mains: list[str] = []
        try:
            if int(self._hop) > 0 and related_point_num > 0:
                task_mains = self.task_layer.retrieve_related_task(query_task=query_clean, node_num=related_point_num, hop=self._hop)
        except Exception:
            task_mains = []
        # 预抓取：一次性按 query_clean 获取更宽的候选集合，按元数据标签划分，减少多次向量化/IO
        ids_list = self.main_memory.get().get('ids') or []
        prefetch_k = min(max((successful_topk + failed_topk) * 2, 4), max(1, len(ids_list)))
        try:
            prefetch = self.main_memory.similarity_search_with_score(query=query_clean, k=prefetch_k)
        except Exception:
            prefetch = []
        for doc, dist in prefetch:
            lbl = doc.metadata.get('label')
            if lbl is True:
                true_tasks_doc.append((doc, dist))
            elif lbl is False:
                false_tasks_doc.append((doc, dist))
            else:
                # 兼容旧库：未标注 label 的样本默认视为成功
                true_tasks_doc.append((doc, dist))
        # 若仍不足，再对邻居集合做有限补充（去重并按需要截断）
        seen_pages = set([d.page_content for d, _ in true_tasks_doc + false_tasks_doc])
        uniq_task_mains = []
        for tm in task_mains:
            if isinstance(tm, str) and tm not in seen_pages:
                uniq_task_mains.append(tm)
        for task_main in uniq_task_mains[:related_point_num]:
            docs = self.main_memory.similarity_search_with_score(query=task_main, k=1)
            if not docs:
                continue
            doc, dist = docs[0]
            lbl = doc.metadata.get('label')
            if lbl is True:
                true_tasks_doc.append((doc, dist))
            elif lbl is False:
                false_tasks_doc.append((doc, dist))
            else:
                true_tasks_doc.append((doc, dist))
        
        # If the specified number is not met, fill in the rest using similarity-based augmentation.
        if len(true_tasks_doc) < successful_topk:
            k_adj = min(max(successful_topk, 1), max(1, len(ids_list)))
            fetched = self.main_memory.similarity_search_with_score(
                query=query_clean, k=k_adj, filter={'label': True}
            )
            for doc, dist in fetched:
                true_tasks_doc.append((doc, dist))
                if len(true_tasks_doc) >= successful_topk:
                    break
        
        if len(false_tasks_doc) < failed_topk:
            k_adj = min(max(failed_topk, 1), max(1, len(ids_list)))
            fetched_false = self.main_memory.similarity_search_with_score(
                query=query_clean, k=k_adj, filter={'label': False}
            )
            for doc, dist in fetched_false:
                false_tasks_doc.append((doc, dist))

        # order by similarity using returned distances from vector store
        true_tasks_doc_with_score = sort_and_filter_by_similarity(true_tasks_doc, threshold)[:successful_topk]
        false_tasks_doc_with_score = sort_and_filter_by_similarity(false_tasks_doc, threshold)[:failed_topk]

        true_task_messages: list[MASMessage] = []
        false_task_messages: list[MASMessage] = []
        for doc, _ in true_tasks_doc_with_score:
            meta_data: dict = doc.metadata
            mas_message: MASMessage = MASMessage.from_dict(meta_data)
            true_task_messages.append(mas_message)
        
        for doc, _ in false_tasks_doc_with_score:
            meta_data: dict = doc.metadata
            mas_message: MASMessage = MASMessage.from_dict(meta_data)
            false_task_messages.append(mas_message)
        
        # get insights and order by relelvance
        insights_with_score = self.insights_layer.query_insights_with_score(query_clean, top_k=insight_windows)
        insights = [insight for insight, _ in insights_with_score][:insight_windows]

        return true_task_messages, false_task_messages, insights

    def retrieve_memory(
        self, 
        query_task: str,         
        successful_topk: int = 2, 
        failed_topk: int = 1,
        insight_topk: int = 10,
        threshold: float = 0.3,
        importance: bool = True,
        **args
    ) -> tuple[list, list, list]: 
        """Access the memory and return the results.

        Args:
            query_task (str): The task to query.
            successful_topk (int, optional): Number of successful cases to retrieve. Defaults to 2.
            failed_topk (int, optional): Number of failed cases to retrieve. Defaults to 1.
            insight_topk (int, optional): Number of insights to retrieve. Defaults to 10.
            threshold (float, optional): Similarity threshold for retrieving cases. Defaults to 0.3.

        Returns:
            tuple[list, list, list]: A tuple containing successful cases, failed cases, and insights.
        """
        
        # retrieve raw tasks
        successful_task_trajectories: list[MASMessage]
        failed_task_trajectories: list[MASMessage]
        insights: list[str]
        # 统一用清洗后的 FACT 文本作为检索查询
        query_clean: str = (query_task or '').strip()
        try:
            if query_clean.lower().startswith('fact:'):
                query_clean = query_clean[len('fact:'):].strip()
        except Exception:
            pass
        successful_task_trajectories, failed_task_trajectories, insights = self._retrieve_memory_raw(
            query_clean, successful_topk, failed_topk, insight_topk, threshold)
        
        # retrieve tasks based on task relevance
        if importance:
            importance_score: list[float] = []
            for success_task in successful_task_trajectories:
                prompt: str = CaseMemoryPrompts.generative_task_user_prompt.format(
                    trajectory=success_task.task_description + '\n' + success_task.task_trajectory,
                    query_scenario=query_clean
                )
                response: str = self.llm_model(messages=[Message('system', CaseMemoryPrompts.generative_task_system_prompt), 
                                                         Message('user', prompt)])
                import re as _re
                m = _re.search(r"\d+", response or "")
                score = int(m.group()) if m else 0
                importance_score.append(score)
            sorted_success_tasks = [task for _, task in sorted(zip(importance_score, successful_task_trajectories), 
                                                               key=lambda x: x[0], reverse=True)]
            top_success_task_trajectories = sorted_success_tasks[:successful_topk]
        else:
            top_success_task_trajectories = successful_task_trajectories[:successful_topk]
        
        # directly get failed tasks
        top_fail_task_trajectories = failed_task_trajectories[:failed_topk]
        
        # directlt get insights
        top_k_insights = insights[:insight_topk]
        self.insights_cache = top_k_insights

        return top_success_task_trajectories, top_fail_task_trajectories, top_k_insights


    def _extract_mas_message(self, mas_message: MASMessage) -> MASMessage:

        mas_message_copy: MASMessage = copy.deepcopy(mas_message)
        state_chain: StateChain = mas_message_copy.chain_of_states
        
        for state_id in reversed(range(len(state_chain))):
            if state_chain.get_state(state_id).graph.get('reward', 0) < 0:
                state_chain.pop_state(state_id)
        
        trajectory = ''
        for state in state_chain:
            try:
                act = str(state.graph.get('action', ''))
            except Exception:
                act = ''
            try:
                obs = str(state.graph.get('observation', ''))
            except Exception:
                obs = ''
            trajectory += f'> {act}\n{obs}\n'
        
        if mas_message_copy.label == True:
            mas_message_copy.task_trajectory = trajectory

        
        import re as _re
        trajectory = _re.sub(r'\d+', '', trajectory)
        mas_message_copy.add_extra_field('clean_traj', trajectory)


        system_prompt = CaseMemoryPrompts.extract_true_traj_system_prompt
        prompt_template = CaseMemoryPrompts.extract_true_traj_user_prompt

        prompt: str = prompt_template.format(
            task=mas_message_copy.task_description,
            trajectory=mas_message_copy.get_extra_field('clean_traj')
        )
        messages: list[Message] = [Message('system', system_prompt), Message('user', prompt)]
        response: str = self.llm_model(messages, temperature=0.1)
        mas_message_copy.add_extra_field('key_steps', response)

        # 事实摘要：依据 task_main（优先）或 task_description 生成 concise 摘要
        try:
            facts_text: str = (mas_message_copy.task_main or mas_message_copy.task_description or '').strip()
            if facts_text:
                s_msgs = [
                    Message('system', CaseMemoryPrompts.summarize_facts_system_prompt),
                    Message('user', CaseMemoryPrompts.summarize_facts_user_prompt.format(facts=facts_text))
                ]
                facts_summary: str = self.llm_model(s_msgs, temperature=0.0)
                mas_message_copy.add_extra_field('facts_summary', facts_summary.strip())
        except Exception:
            pass

        # 提取各角色关键协作步骤：遍历状态链中的每个图节点，聚合每个智能体的输出
        try:
            from collections import defaultdict
            agent_steps_map: dict[str, list[str]] = defaultdict(list)
            for state in state_chain:
                try:
                    for _, attrs in state.nodes(data=True):
                        agent_name = attrs.get('agent_name')
                        message = attrs.get('message')
                        if agent_name and isinstance(message, str) and message.strip():
                            agent_steps_map[agent_name].append(message.strip())
                except Exception:
                    # 节点解析失败时跳过该状态
                    pass
            # 将每个角色的步骤精简为按时间顺序的摘要（保留最多2条）
            agent_steps_compact: dict[str, str] = {}
            for name, msgs in agent_steps_map.items():
                if not msgs:
                    continue
                # 取末尾两条代表性输出，避免过长
                trimmed = msgs[-2:] if len(msgs) >= 2 else msgs
                agent_steps_compact[name] = " -> ".join(trimmed)
            mas_message_copy.add_extra_field('agent_steps', agent_steps_compact)
        except Exception:
            # 保持向后兼容：若解析失败不影响主流程
            pass

        # 结果提取：解析最终动作中的 Finish[JSON]，保留原始 Finish[...] 文本
        try:
            import re, json
            jstr_candidate: str | None = None
            # 遍历所有动作，取最后一个包含 Finish[{...}] 的 JSON 片段
            for state in state_chain:
                try:
                    act = state.graph.get('action')
                except Exception:
                    act = None
                if isinstance(act, str) and act:
                    # 标准匹配
                    ms = re.findall(r"(?is)finish\s*\[\s*(\{.*?\})\s*\]", act)
                    if ms:
                        jstr_candidate = ms[-1]
                    else:
                        # 兼容少数模型输出的嵌套括号或额外空格
                        ms2 = re.findall(r"(?is)finish\s*\[\s*\{([\s\S]*?)\}\s*\]", act)
                        if ms2:
                            jstr_candidate = '{' + ms2[-1] + '}'
            if jstr_candidate:
                raw = jstr_candidate
                # 尝试解析；失败则降级修复双花括号
                def _try_load(txt: str):
                    try:
                        return json.loads(txt)
                    except Exception:
                        return None
                jobj = _try_load(raw)
                if jobj is None:
                    try:
                        fixed = raw.replace('{{', '{').replace('}}', '}')
                        jobj = _try_load(fixed)
                        if jobj is not None:
                            raw = fixed
                    except Exception:
                        pass
                if jobj is not None:
                    mas_message_copy.add_extra_field('result_json', jobj)
                    mas_message_copy.add_extra_field('result_finish', f"Finish[{raw}]")
        except Exception:
            # 不阻断流程
            pass


        if mas_message_copy.label == False:
            reason: str = ''
            exp = mas_message_copy.get_extra_field('expected') or {}
            res = mas_message_copy.get_extra_field('result_json') or {}
            if not (isinstance(res, dict) and res):
                raw_finish = mas_message_copy.get_extra_field('result_finish') or ''
                parsed = self._parse_result_loose(raw_finish)
                if isinstance(parsed, dict) and parsed:
                    res = parsed
            if isinstance(exp, dict) and exp:
                try:
                    reason = self._legal_failure_reason(exp, res if isinstance(res, dict) else {})
                except Exception:
                    reason = ''
            if not reason:
                reason = '罪名或法条或量刑不一致。'
            mas_message_copy.add_extra_field('fail_reason', reason)
            # 仅用于审计：追加原始 Finish 文本，不参与后续规则检索或生成
            try:
                raw_finish = mas_message_copy.get_extra_field('result_finish') or ''
                mas_message_copy.add_extra_field('fail_finish_raw', raw_finish)
            except Exception:
                pass
        
        return mas_message_copy
    
    
    def _detect_mistakes(self, mas_message: MASMessage) -> str:
        user_prompt: str = CaseMemoryPrompts.detect_mistakes_user_prompt.format(task=mas_message.task_description, trajectory=mas_message.get_extra_field('clean_traj'))
        messages: list[Message] = [Message('system', CaseMemoryPrompts.detect_mistakes_system_prompt), 
                                   Message('user', user_prompt)]
        response: str = self.llm_model(messages)

        return response

    def _parse_result_loose(self, finish_text: str) -> dict:
        import re
        s = finish_text or ''
        if not isinstance(s, str) or not s:
            return {}
        acc = []
        m = re.search(r'"accusation"\s*:\s*\[([^\]]+)\]', s, flags=re.I)
        if m:
            acc = [x.strip() for x in re.findall(r'"([^"]+)"', m.group(1)) if x.strip()]
        else:
            m2 = re.search(r'"accusation"\s*:\s*"([^"]+)"', s, flags=re.I)
            if m2:
                t = m2.group(1).strip()
                if t:
                    acc = [t]
        arts = []
        m = re.search(r'"relevant_articles"\s*:\s*\[([^\]]*)\]', s, flags=re.I)
        if m:
            arts = [int(x) for x in re.findall(r'\d+', m.group(1))]
        term = {}
        mdp = re.search(r'"death_penalty"\s*:\s*(true|false)', s, flags=re.I)
        if mdp:
            term['death_penalty'] = True if mdp.group(1).lower() == 'true' else False
        mlife = re.search(r'"life_imprisonment"\s*:\s*(true|false)', s, flags=re.I)
        if mlife:
            term['life_imprisonment'] = True if mlife.group(1).lower() == 'true' else False
        mimp = re.search(r'"imprisonment"\s*:\s*(\d+)', s, flags=re.I)
        if mimp:
            term['imprisonment'] = int(mimp.group(1))
        return {
            'accusation': acc,
            'relevant_articles': arts,
            'term_of_imprisonment': term
        }

    def _legal_failure_reason(self, expected: dict, result: dict) -> str:
        def _to_list(x):
            if x is None:
                return []
            if isinstance(x, list):
                return x
            return [x]
        acc_e = [str(a).strip() for a in _to_list(expected.get('accusation')) if str(a).strip()]
        acc_r = [str(a).strip() for a in _to_list(result.get('accusation')) if str(a).strip()]
        try:
            arts_e = [int(x) for x in _to_list(expected.get('relevant_articles'))]
        except Exception:
            arts_e = []
        try:
            arts_r = [int(x) for x in _to_list(result.get('relevant_articles'))]
        except Exception:
            arts_r = []
        te = (expected.get('term_of_imprisonment') or {}).get('imprisonment')
        tr = (result.get('term_of_imprisonment') or {}).get('imprisonment')
        de = (expected.get('term_of_imprisonment') or {}).get('death_penalty')
        dr = (result.get('term_of_imprisonment') or {}).get('death_penalty')
        le = (expected.get('term_of_imprisonment') or {}).get('life_imprisonment')
        lr = (result.get('term_of_imprisonment') or {}).get('life_imprisonment')
        rs = []
        if set(acc_e) != set(acc_r):
            if not acc_r:
                rs.append('未给出罪名预测')
            else:
                rs.append(f"罪名错误：预测{acc_r}，应为{acc_e}")
        if set(arts_e) != set(arts_r):
            if not arts_r:
                rs.append('未给出法条预测')
            else:
                rs.append(f"法条错误：预测{arts_r}，应为{arts_e}")
        try:
            def _months_to_bucket(months: int) -> int:
                try:
                    m = int(months)
                except Exception:
                    m = 0
                if m <= 0:
                    return 10
                if m <= 6:
                    return 9
                if m <= 9:
                    return 8
                if m <= 12:
                    return 7
                if m <= 24:
                    return 6
                if m <= 36:
                    return 5
                if m <= 60:
                    return 4
                if m <= 84:
                    return 3
                if m <= 120:
                    return 2
                return 1
            if 'term' in expected:
                gb = int(expected.get('term'))
            else:
                gt = expected.get('term_of_imprisonment') or {}
                if bool(gt.get('death_penalty')) or bool(gt.get('life_imprisonment')):
                    gb = 0
                else:
                    gb = _months_to_bucket(gt.get('imprisonment'))
            if 'term' in result:
                rb = int(result.get('term'))
            else:
                rt = result.get('term_of_imprisonment') or {}
                if bool(rt.get('death_penalty')) or bool(rt.get('life_imprisonment')):
                    rb = 0
                else:
                    rb = _months_to_bucket(rt.get('imprisonment'))
            if gb is not None and rb is not None and int(gb) != int(rb):
                rs.append("量刑分档不一致")
        except Exception:
            pass
        if isinstance(de, bool) and isinstance(dr, bool) and de != dr:
            rs.append("死刑判定错误")
        if isinstance(le, bool) and isinstance(lr, bool) and le != lr:
            rs.append("无期徒刑判定错误")
        if not rs:
            return "输出与期望不一致或格式问题"
        return '；'.join(rs) + '。'

    def backward(self, reward: bool):

        for insight in self.insights_cache:
            self.insights_layer.backward(insight, reward=-2 if reward == False else 1)

        self.insights_cache = []
    
    @property
    def memory_size(self):
        num_records = self.main_memory.get()["ids"]
        return len(num_records)
    
    def project_insights(self, raw_insights: list[str], role: str = None, task_traj: str = None) -> list[str]:
        """
        Projects raw insights into role-specific insights based on the given role and optionally a task trajectory.

        Args:
            raw_insights (list[str]): A list of raw insight strings.
            role (str, optional): The role to tailor the insights for. Defaults to None.
            task_traj (str, optional): A string representing the task trajectory. Defaults to None.

        Returns:
            list[str]: A list of processed insights tailored to the specified role.
        """
        def parse_numbered_list(text: str) -> list[str]:
            pattern = r'(?:^|\n)\s*\d+[\.|、]\s+(.*?)(?=\n\s*\d+[\.|、]|\Z)'
            items = re.findall(pattern, text.strip(), flags=re.DOTALL)
            return [item.strip() for item in items]
        
        # If no role is provided, return the raw insights as they are.
        if not role:
            return raw_insights
        
        # Determine which system and user prompts to use based on whether a task trajectory is provided
        raw_insights_str = '\n'.join(raw_insights)
        if not task_traj:
            system_prompt = CaseMemoryPrompts.project_insights_system_prompt
            user_prompt: str = CaseMemoryPrompts.project_insights_user_prompt.format(
                role=role,
                insights=raw_insights_str
            )
        else:
            system_prompt = CaseMemoryPrompts.project_insights_with_traj_system_prompt
            user_prompt: str = CaseMemoryPrompts.project_insights_with_traj_user_prompt.format(
                role=role,
                insights=raw_insights_str,
                trajectory=task_traj
            )
        messages = [Message('system', system_prompt),
                    Message('user', user_prompt)]
        
        # Use the language model to generate role-specific insights
        role_insights = self.llm_model(messages)

        try: 
            role_insights = parse_numbered_list(role_insights)
            return role_insights
        except:
            return raw_insights

@dataclass
class TaskLayer:
    
    working_dir: str
    namespace: str
    task_storage: Chroma
    
    def __post_init__(self):
        self.similarity_threshold = 0.5
        self.k_neighbors = 5
        try:
            v = os.environ.get('TASK_LAYER_SIMILARITY')
            if v:
                self.similarity_threshold = float(v)
        except Exception:
            pass
        try:
            k = os.environ.get('TASK_LAYER_K')
            if k:
                self.k_neighbors = int(k)
        except Exception:
            pass

        self._graph_pic_save_path: str = os.path.join(self.working_dir, 'graph.png')
        self._node_match_save_path: str = os.path.join(self.working_dir, 'match_nodes.txt')
        self._graph_save_path: str = os.path.join(self.working_dir, f'{self.namespace}_graph.pkl')
        self._lock_path: str = os.path.join(self.working_dir, f'{self.namespace}_graph.lock')

        if os.path.exists(self._graph_save_path):
            try:
                with InterProcessFileLock(self._lock_path, timeout=120):
                    with open(self._graph_save_path, 'rb') as f:
                        self.graph = pickle.load(f)
                print(f"Graph loaded from {self._graph_save_path}")
            except Exception:
                # Fallback: start fresh if loading fails under contention
                self.graph = nx.Graph()
                print("New empty graph created (fallback)")
        else:
            self.graph = nx.Graph()
            print("New empty graph created")

    def add_task_node(self, task_main: str) -> None:
        """Add a task node to the task graph.

        Args:
            task_main (str): task name
        """
        if task_main in self.graph:
            return  

        self.graph.add_node(task_main)

        ids_list = self.task_storage.get().get('ids') or []
        k_adj = min(self.k_neighbors, max(1, len(ids_list)))
        results: list[tuple[Document, float]] = self.task_storage.similarity_search_with_score(
            query=task_main,
            k=k_adj
        )

        self_msg = None
        try:
            self_res = self.task_storage.similarity_search_with_score(query=task_main, k=1)
            if self_res:
                self_msg = MASMessage.from_dict(self_res[0][0].metadata)
        except Exception:
            self_msg = None
        def _extract_labels(msg: MASMessage):
            accs = set(); arts = set()
            try:
                exp = msg.get_extra_field('expected') or {}
                res = msg.get_extra_field('result_json') or {}
                for a in (exp.get('accusation') or res.get('accusation') or []):
                    s = str(a).strip()
                    if s:
                        accs.add(s)
                for idv in (exp.get('relevant_articles') or res.get('relevant_articles') or []):
                    try:
                        arts.add(int(idv))
                    except Exception:
                        pass
            except Exception:
                pass
            return accs, arts
        acc_self, arts_self = _extract_labels(self_msg) if self_msg else (set(), set())
        for doc, distance in results:
            similarity = 1 - distance
            if similarity < self.similarity_threshold:
                continue
            neighbor = doc.page_content
            same = False
            try:
                nmsg = MASMessage.from_dict(doc.metadata)
                acc_n, arts_n = _extract_labels(nmsg)
                same = bool(acc_self) and bool(arts_self) and (acc_self == acc_n) and (arts_self == arts_n)
            except Exception:
                same = False
            if not same:
                continue
            if neighbor not in self.graph:
                self.graph.add_node(neighbor)
            try:
                # 将解析到的标签写入图节点属性，便于后续快速读取
                self.graph.nodes[neighbor]['labels_accusations'] = list(acc_n)
                self.graph.nodes[neighbor]['labels_articles'] = list(arts_n)
            except Exception:
                pass
            self.graph.add_edge(task_main, neighbor, weight=similarity)
        self._index_done()
 
    def retrieve_related_task(self, query_task: str, node_num: int, hop: int = 1) -> list[str]:
        """
        Retrieve related tasks from the graph based on similarity and local neighborhood expansion.

        Args:
            query_task (str): The task used as the query input.
            node_num (int): The number of top similar tasks to retrieve based on similarity scores.
            hop (int, optional): The number of hops used to expand the neighborhood in the graph. Defaults to 1.

        Returns:
            list[str]: A list of related task nodes, including top similar tasks and their neighbors within the given hop.
        """
        ids_list = self.task_storage.get().get('ids') or []
        k_base = max(node_num, self.k_neighbors)
        k_adj = min(k_base, max(1, len(ids_list)))
        tasks: list[tuple[Document, float]] = self.task_storage.similarity_search_with_score(query=query_task, k=k_adj)
        top_nodes = []
        for doc, dist in tasks:
            sim = 1 - float(dist)
            if sim >= self.similarity_threshold:
                top_nodes.append(doc.page_content)
                try:
                    # 将检索到的自身标签存入节点，减少后续读取开销
                    msg = MASMessage.from_dict(doc.metadata)
                    accs, arts = set(), set()
                    exp = msg.get_extra_field('expected') or {}
                    res = msg.get_extra_field('result_json') or {}
                    for a in (exp.get('accusation') or res.get('accusation') or []):
                        s = str(a).strip()
                        if s:
                            accs.add(s)
                    for idv in (exp.get('relevant_articles') or res.get('relevant_articles') or []):
                        try:
                            arts.add(int(idv))
                        except Exception:
                            pass
                    if doc.page_content not in self.graph:
                        self.graph.add_node(doc.page_content)
                    self.graph.nodes[doc.page_content]['labels_accusations'] = list(accs)
                    self.graph.nodes[doc.page_content]['labels_articles'] = list(arts)
                except Exception:
                    pass

        related_nodes = set(top_nodes)
        for node in top_nodes:
            neighbours = nx.single_source_shortest_path_length(self.graph, node, cutoff=hop).keys()
            related_nodes.update(neighbours)
        return list(related_nodes)
    
    def cluster_tasks(self) -> None:
        """
        Perform clustering on tasks in the graph using their embeddings and assign cluster IDs.

        This method extracts all nodes from the graph, computes embeddings for each node using the
        task storage's embedding function, and applies the FINCH clustering algorithm with cosine similarity.
        """
        nodes = list(self.graph.nodes)

        embeddings = []
        valid_nodes = []

        for node in nodes:
            embedding = self.task_storage._embedding_function.embed_query(node)  
            if embedding is not None:
                embeddings.append(embedding)
                valid_nodes.append(node)

        # Build embedding matrix and normalize rows to unit norm to stabilize cosine distance
        if len(embeddings) == 0:
            # No valid embeddings; assign default cluster 0
            for node in valid_nodes:
                self.graph.nodes[node]['cluster_id'] = 0
            self._index_done()
            return

        X = np.vstack(embeddings).astype(np.float32)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X = X / norms

        # FINCH in some versions exposes a functional API: FINCH(X, distance='cosine')
        # It returns labels across cuts; we pick the last cut as final labels.
        labels = None
        try:
            res = FINCH(X, distance='cosine')
        except TypeError:
            # Fallback: older versions may not support 'distance' kwarg
            res = FINCH(X)
        except Exception as e:
            print(f"FINCH clustering failed: {e}")
            res = None

        try:
            if res is None:
                labels = np.zeros(len(valid_nodes), dtype=int)
            else:
                c = res[0] if isinstance(res, tuple) else res
                labels = c[:, -1] if hasattr(c, 'shape') and len(c.shape) == 2 else np.asarray(c)
        except Exception as e:
            print(f"FINCH label extraction failed: {e}")
            labels = np.zeros(len(valid_nodes), dtype=int)

        for node, label in zip(valid_nodes, labels):
            self.graph.nodes[node]['cluster_id'] = int(label)
        self._index_done()

    def _index_done(self) -> None:
        # Persist graph with inter-process lock to avoid concurrent writes
        try:
            with InterProcessFileLock(self._lock_path, timeout=120):
                with open(self._graph_save_path, "wb") as f:
                    pickle.dump(self.graph, f)
                # auto export PNG snapshot alongside pickle under same lock
                try:
                    export_graph_png(self._graph_save_path, self._graph_pic_save_path)
                except Exception as e:
                    print(f"TaskLayer: export graph PNG failed -> {e}")
        except TimeoutError as e:
            print(f"TaskLayer: lock timeout when persisting graph -> {e}")

    def __iter__(self) -> Iterable[tuple[str, int]]: 
        return ((node, self.graph.nodes[node]['cluster_id']) for node in self.graph.nodes)

    


@dataclass
class InsightsManager:

    working_dir: str
    namespace: str
    llm_model: LLMCallable
    task_storage: Chroma
    task_layer: TaskLayer
    insights_sem_batch: int = 16
    embed_text_clip: int = 4096
    embedding_cache: dict = None

    def __post_init__(self):
        if self.embedding_cache is None:
            self.embedding_cache = {}
        self.persist_file: str = os.path.join(self.working_dir,f'{self.namespace}.json')
        self._lock_path: str = os.path.join(self.working_dir, f'{self.namespace}.lock')
        # Load insights under a short lock to avoid reading partial writes
        try:
            with InterProcessFileLock(self._lock_path, timeout=60):
                self.insights_memory: list[dict] = load_json(self.persist_file) or []
        except Exception:
            self.insights_memory = load_json(self.persist_file) or []
       
        log_path = os.path.join(self.working_dir, 'insights.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path, encoding='utf-8')
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Cache persistence
        self._cache_path = os.path.join(self.working_dir, f'{self.namespace}_embeddings_cache.pkl')
        if os.path.exists(self._cache_path):
            try:
                with open(self._cache_path, 'rb') as f:
                    loaded_cache = pickle.load(f)
                    if isinstance(loaded_cache, dict):
                        self.embedding_cache.update(loaded_cache)
            except Exception as e:
                print(f"InsightsManager: failed to load embedding cache -> {e}")

        # Pre-compute embeddings for existing insights
        self._pre_compute_embeddings()
        self._insights_inverted_index = {}
        self._build_inverted_index()
        self._task_labels_cache = {}
        self._rule_labels_cache = {}
        try:
            self._rule_index = {}
            for ins in self.insights_memory:
                rt = ins.get('rule')
                if rt:
                    self._rule_index[rt] = ins
        except Exception:
            self._rule_index = {}
        
    def _pre_compute_embeddings(self):
        if not self.insights_memory:
            return
        
        missing_rules = []
        for ins in self.insights_memory:
            rule = ins.get('rule')
            if rule and rule not in self.embedding_cache:
                missing_rules.append(rule)
        
        if missing_rules:
            self.logger.info(f"InsightsManager: Pre-computing embeddings for {len(missing_rules)} rules...")
            try:
                # Batch embed
                batch_size = 16
                for i in range(0, len(missing_rules), batch_size):
                    batch = missing_rules[i:i+batch_size]
                    try:
                        embeddings = self.task_storage._embedding_function.embed_documents(batch)
                        for rule, emb in zip(batch, embeddings):
                            self.embedding_cache[rule] = emb
                    except Exception as e:
                        self.logger.error(f"Batch embedding failed: {e}")
                        # Fallback to single
                        for rule in batch:
                            try:
                                emb = self.task_storage._embedding_function.embed_query(rule)
                                self.embedding_cache[rule] = emb
                            except Exception:
                                pass
                
                # Save after update
                self._save_cache()
            except Exception as e:
                self.logger.error(f"InsightsManager: Pre-computation failed -> {e}")

    def _save_cache(self):
        try:
            with open(self._cache_path, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
        except Exception as e:
            self.logger.error(f"InsightsManager: failed to save embedding cache -> {e}")
        
    def _build_inverted_index(self):
        idx = {}
        try:
            for ins in self.insights_memory:
                rt = ins.get('rule')
                tasks = ins.get('positive_correlation_tasks') or []
                for t in tasks:
                    s = (t or '').strip()
                    if not s:
                        continue
                    if s not in idx:
                        idx[s] = set()
                    idx[s].add(rt)
                    try:
                        if not s.lower().startswith('fact:'):
                            fs = f"Fact: {s}"
                            if fs not in idx:
                                idx[fs] = set()
                            idx[fs].add(rt)
                    except Exception:
                        pass
            self._insights_inverted_index = {k: list(v) for k, v in idx.items()}
        except Exception:
            self._insights_inverted_index = {}

    def get_labels_for_task(self, nm: str) -> tuple[set[str], set[int]]:
        key = (nm or '').strip()
        if not key:
            return set(), set()
        cached = self._task_labels_cache.get(key)
        if cached is not None:
            return cached[0], cached[1]
        accs: set[str] = set()
        arts: set[int] = set()
        try:
            docs = self.task_storage.similarity_search_with_score(query=key, k=1)
            if docs:
                md = docs[0][0].metadata
                msg = MASMessage.from_dict(md)
                exp = msg.get_extra_field('expected') or {}
                res = msg.get_extra_field('result_json') or {}
                for a in (exp.get('accusation') or res.get('accusation') or []):
                    s = str(a).strip()
                    if s:
                        accs.add(s)
                for idv in (exp.get('relevant_articles') or res.get('relevant_articles') or []):
                    try:
                        arts.add(int(idv))
                    except Exception:
                        pass
        except Exception:
            pass
        self._task_labels_cache[key] = (accs, arts)
        return accs, arts

    def articles_from_tasks(self, task_names: list[str]) -> set[int]:
        arts: set[int] = set()
        for nm in (task_names or []):
            _, a = self.get_labels_for_task(nm)
            arts |= a
        return arts

    def accs_from_tasks(self, task_names: list[str]) -> set[str]:
        accs: set[str] = set()
        for nm in (task_names or []):
            a, _ = self.get_labels_for_task(nm)
            accs |= a
        return accs

    def get_rule_source_labels(self, rule_text: str) -> tuple[set[str], set[int]]:
        cached = self._rule_labels_cache.get(rule_text)
        if cached is not None:
            return cached[0], cached[1]
        accs = set()
        arts = set()
        try:
            ins = None
            try:
                ins = self._rule_index.get(rule_text)
            except Exception:
                ins = None
            if ins is None:
                for it in self.insights_memory:
                    if it.get('rule') == rule_text:
                        ins = it
                        break
            if ins:
                src_tasks = ins.get('positive_correlation_tasks') or []
                accs = self.accs_from_tasks(src_tasks)
                arts = self.articles_from_tasks(src_tasks)
        except Exception:
            pass
        self._rule_labels_cache[rule_text] = (accs, arts)
        return accs, arts

    def _get_rule_counts_for_tasks(self, task_mains: list[str]) -> dict:
        from collections import defaultdict as _dd
        counts = _dd(int)
        idx = getattr(self, '_insights_inverted_index', {})
        for t in (task_mains or []):
            rules = idx.get(t) or []
            for rt in rules:
                if rt:
                    counts[rt] += 1
        return counts

    def query_insights_with_score(self, task_query: str, top_k: int = None) -> list[tuple[str, float]]:
        SUCC_NUM, FAIL_NUM = 4, 2
        related_successful_tasks, related_failed_tasks = self._retrieve_memory(task_query, successful_topk=SUCC_NUM, failed_topk=FAIL_NUM)
        neighbors = []
        try:
            if int(getattr(self, '_hop', 0)) > 0:
                neighbors = self.task_layer.retrieve_related_task(task_query, node_num=SUCC_NUM + FAIL_NUM, hop=self._hop)
        except Exception:
            neighbors = []
        task_mains = [task.task_main for task in related_successful_tasks + related_failed_tasks]
        task_mains += [nm for nm in neighbors if isinstance(nm, str)]
        task_mains.append(task_query)
        seen = set()
        task_mains = [t for t in task_mains if not (t in seen or seen.add(t))]

        # 读取当前查询的标签与法条编号（来自 expected/result 或候选集合）
        query_accs: set[str] = set()
        query_arts: set[int] = set()
        # 清洗查询文本（移除可能的 "Fact:" 前缀）
        query_clean: str = (task_query or '').strip()
        try:
            if query_clean.lower().startswith('fact:'):
                query_clean = query_clean[len('fact:'):].strip()
        except Exception:
            pass
        try:
            # 搜集与 query_task 相同的已存任务的标签，作为弱监督来源
            # 优化：减少候选集合大小，从 Top-8 降为 Top-3，提高相关性
            for doc, _ in self.task_storage.similarity_search_with_score(query_clean, k=3):
                md = doc.metadata
                msg = MASMessage.from_dict(md)
                exp = msg.get_extra_field('expected') or {}
                res = msg.get_extra_field('result_json') or {}
                for a in (exp.get('accusation') or res.get('accusation') or []):
                    s = str(a).strip()
                    if s:
                        query_accs.add(s)
                for idv in (exp.get('relevant_articles') or res.get('relevant_articles') or []):
                    try:
                        query_arts.add(int(idv))
                    except Exception:
                        pass
        except Exception:
            pass
        
        # 新增：将检索代理（Retrieval Agent）推荐的法条集合也加入到候选标签中
        # 这是对当前案件最直接的预估，通常比基于语义检索的相似案例更准确
        try:
            retrieved_ids = self.current_task_context.get_extra_field('retrieved_law_ids') or []
            for rid in retrieved_ids:
                try:
                    query_arts.add(int(rid))
                except Exception:
                    pass
        except Exception:
            pass
        # 任务键扩展（兼容是否带 "Fact:" 前缀的存储差异）
        augmented_task_mains = []
        for t in task_mains:
            s = (t or '').strip()
            augmented_task_mains.append(s)
            try:
                if not s.lower().startswith('fact:'):
                    augmented_task_mains.append(f"Fact: {s}")
            except Exception:
                pass
        seen2 = set()
        augmented_task_mains = [x for x in augmented_task_mains if not (x in seen2 or seen2.add(x))]

        corr_scores = defaultdict(float)
        counts_map = self._get_rule_counts_for_tasks(augmented_task_mains)
        for rt, cnt in counts_map.items():
            try:
                corr_scores[rt] += float(cnt)
            except Exception:
                pass

        ins_index = {}
        try:
            for ins in self.insights_memory:
                rt = ins.get('rule')
                if rt:
                    ins_index[rt] = ins
        except Exception:
            ins_index = {}

        def _get_labels_for_task(nm: str) -> tuple[set[str], set[int]]:
            key = (nm or '').strip()
            if not key:
                return set(), set()
            cached = self._task_labels_cache.get(key)
            if cached is not None:
                return cached[0], cached[1]
            accs: set[str] = set()
            arts: set[int] = set()
            try:
                docs = self.task_storage.similarity_search_with_score(query=key, k=1)
                if docs:
                    md = docs[0][0].metadata
                    msg = MASMessage.from_dict(md)
                    exp = msg.get_extra_field('expected') or {}
                    res = msg.get_extra_field('result_json') or {}
                    for a in (exp.get('accusation') or res.get('accusation') or []):
                        s = str(a).strip()
                        if s:
                            accs.add(s)
                    for idv in (exp.get('relevant_articles') or res.get('relevant_articles') or []):
                        try:
                            arts.add(int(idv))
                        except Exception:
                            pass
            except Exception:
                pass
            self._task_labels_cache[key] = (accs, arts)
            return accs, arts

        def _articles_from_tasks(task_names: list[str]) -> set[int]:
            arts: set[int] = set()
            for nm in (task_names or []):
                _, a = _get_labels_for_task(nm)
                arts |= a
            return arts

        def _accs_from_tasks(task_names: list[str]) -> set[str]:
            accs: set[str] = set()
            for nm in (task_names or []):
                a, _ = _get_labels_for_task(nm)
                accs |= a
            return accs

        def _get_rule_src_labels(rule_text: str) -> tuple[set[str], set[int]]:
            if rule_text in self._rule_labels_cache:
                return self._rule_labels_cache.get(rule_text)
            ins = ins_index.get(rule_text)
            accs = set()
            arts = set()
            if ins:
                src_tasks = ins.get('positive_correlation_tasks') or []
                accs = _accs_from_tasks(src_tasks)
                arts = _articles_from_tasks(src_tasks)
            self._rule_labels_cache[rule_text] = (accs, arts)
            return accs, arts

        def passes_hard_gate(rule_text: str) -> bool:
            if not rule_text:
                return False
            try:
                if (not query_accs) and (not query_arts):
                    return True
                
                src_accs, src_arts = self.get_rule_source_labels(rule_text)
                if src_arts & query_arts:
                    return True
                if src_accs & query_accs:
                    return True
                
                # 2. 检查洞见文本本身是否包含相关标签
                import re
                acc_hits = set(re.findall(r'([\u4e00-\u9fff]+罪)', rule_text)) & query_accs
                # 修复正则：匹配"第264条"或纯数字形式的法条，不再依赖单词边界\b，因为中文语境下\b可能不适用
                art_hits = set(int(x) for x in re.findall(r"(?:第)?(\d{2,3})(?:条)?", rule_text)) & query_arts
                
                # 3. 如果没有显式重叠，则检查是否有强冲突（即洞见包含明确的“其他”罪名/法条，且不包含当前需要的）
                # 如果洞见完全没提任何罪名/法条，则视为通用经验，允许通过
                rule_accs = set(re.findall(r'([\u4e00-\u9fff]+罪)', rule_text))
                rule_arts = set(int(x) for x in re.findall(r"(?:第)?(\d{2,3})(?:条)?", rule_text))
                
                if (not rule_accs) and (not rule_arts):
                    return True
                
                # 如果洞见提到了某些标签，必须至少命中一个
                return bool(acc_hits) or bool(art_hits)
            except Exception:
                return False

        use_sem = True
        try:
            env_flag = os.environ.get('INSIGHTS_SEMANTIC')
            if env_flag is not None and str(env_flag).strip().lower() in ('0', 'false'):
                use_sem = False
        except Exception:
            use_sem = True
        query_vec = None
        if use_sem:
            try:
                query_vec = self.task_storage._embedding_function.embed_query(query_clean[:self.embed_text_clip])
            except Exception:
                query_vec = None
        else:
            query_vec = None
        w_intersect, w_cooccur, w_sem = 0.5, 0.3, (0.2 if use_sem else 0.0)
        combined = []
        consider_rules = [rt for rt, cs in corr_scores.items() if passes_hard_gate(rt)]
        B = max(1, int(self.insights_sem_batch))
        updated_cache = False
        for i in range(0, len(consider_rules), B):
            chunk = consider_rules[i:i + B]
            ts_list = [0.0] * len(chunk)
            if query_vec is not None and chunk:
                # 检查缓存命中情况
                chunk_to_embed = []
                chunk_indices = []
                cached_vectors = []
                
                for idx, rt in enumerate(chunk):
                    if rt in self.embedding_cache:
                        cached_vectors.append((idx, self.embedding_cache[rt]))
                    else:
                        chunk_to_embed.append(rt)
                        chunk_indices.append(idx)
                
                # 对未缓存的进行向量化
                if chunk_to_embed:
                    try:
                        new_rvecs = self.task_storage._embedding_function.embed_documents(chunk_to_embed)
                        for i, rvec in enumerate(new_rvecs):
                            original_idx = chunk_indices[i]
                            rt = chunk_to_embed[i]
                            self.embedding_cache[rt] = rvec
                            updated_cache = True
                            try:
                                ts_list[original_idx] = cosine_similarity(query_vec, rvec)
                            except Exception:
                                ts_list[original_idx] = 0.0
                    except Exception:
                        # Fallback to individual embedding if batch fails
                        for i, rt in enumerate(chunk_to_embed):
                            original_idx = chunk_indices[i]
                            try:
                                rvec = self.task_storage._embedding_function.embed_query(rt)
                                self.embedding_cache[rt] = rvec
                                updated_cache = True
                                ts_list[original_idx] = cosine_similarity(query_vec, rvec)
                            except Exception:
                                ts_list[original_idx] = 0.0
                
                # 使用缓存的向量计算相似度
                for idx, rvec in cached_vectors:
                    try:
                        ts_list[idx] = cosine_similarity(query_vec, rvec)
                    except Exception:
                        ts_list[idx] = 0.0

            for idx, rule_text in enumerate(chunk):
                cscore = float(corr_scores.get(rule_text, 0.0))
                try:
                    import re
                    art_nums = [int(x) for x in re.findall(r"(?:第)?(\d{2,3})(?:条)?", rule_text)]
                    inter = len(set(art_nums) & query_arts)
                except Exception:
                    inter = 0
                ts = float(ts_list[idx] if idx < len(ts_list) else 0.0)
                score = w_intersect * float(inter) + w_cooccur * cscore + w_sem * ts
                combined.append((rule_text, score))

        if not combined:
            try:
                all_rules = [ins.get('rule') for ins in self.insights_memory]
                fallback_rules = [rt for rt in all_rules if passes_hard_gate(rt)]
                B = max(1, int(self.insights_sem_batch))
                for i in range(0, len(fallback_rules), B):
                    chunk = fallback_rules[i:i + B]
                    ts_list = [0.0] * len(chunk)
                    if query_vec is not None and chunk:
                        # 检查缓存命中情况
                        chunk_to_embed = []
                        chunk_indices = []
                        cached_vectors = []
                        
                        for idx, rt in enumerate(chunk):
                            if rt in self.embedding_cache:
                                cached_vectors.append((idx, self.embedding_cache[rt]))
                            else:
                                chunk_to_embed.append(rt)
                                chunk_indices.append(idx)
                        
                        # 对未缓存的进行向量化
                        if chunk_to_embed:
                            try:
                                new_rvecs = self.task_storage._embedding_function.embed_documents(chunk_to_embed)
                                for i, rvec in enumerate(new_rvecs):
                                    original_idx = chunk_indices[i]
                                    rt = chunk_to_embed[i]
                                    self.embedding_cache[rt] = rvec
                                    updated_cache = True
                                    try:
                                        ts_list[original_idx] = cosine_similarity(query_vec, rvec)
                                    except Exception:
                                        ts_list[original_idx] = 0.0
                            except Exception:
                                # Fallback to individual embedding if batch fails
                                for i, rt in enumerate(chunk_to_embed):
                                    original_idx = chunk_indices[i]
                                    try:
                                        rvec = self.task_storage._embedding_function.embed_query(rt)
                                        self.embedding_cache[rt] = rvec
                                        updated_cache = True
                                        ts_list[original_idx] = cosine_similarity(query_vec, rvec)
                                    except Exception:
                                        ts_list[original_idx] = 0.0
                        
                        # 使用缓存的向量计算相似度
                        for idx, rvec in cached_vectors:
                            try:
                                ts_list[idx] = cosine_similarity(query_vec, rvec)
                            except Exception:
                                ts_list[idx] = 0.0
                    for idx, rt in enumerate(chunk):
                        ts = float(ts_list[idx] if idx < len(ts_list) else 0.0)
                        combined.append((rt, w_sem * ts))
            except Exception:
                combined = []
        # 若当前查询包含明确法条编号集合（来自 LawArticleRetriever 初选或检索角色输出），过滤掉与编号无交集的洞见，避免跨案件类型
        try:
            retrieved_ids = set(int(x) for x in (self.current_task_context.get_extra_field('retrieved_law_ids') or []))
            if retrieved_ids:
                combined = [pair for pair in combined if any(int(x) in retrieved_ids for x in re.findall(r"(?:第)?(\d{2,3})(?:条)?", pair[0]))]
        except Exception:
            pass

        combined.sort(key=lambda x: x[1], reverse=True)

        
        try:
            if updated_cache:
                self._save_cache()
        except Exception:
            pass
        final_insights = combined[:max(3, top_k or 3)] if combined else []

        
        return final_insights
    
    def merge_insights(self) -> None:

        self.task_layer.cluster_tasks()
        
        label_tasks: dict[int, list[str]] = {}
        for task_main, label_id in self.task_layer:
            if label_id is None:
                raise RuntimeError('Label id should not be none.')
            if label_id not in label_tasks.keys():
                label_tasks[label_id] = [task_main]
            else:
                label_tasks[label_id].append(task_main)
        
        # 优化：在合并洞见时，除了依赖 TaskLayer 的无监督聚类，进一步引入“法条/罪名”标签约束
        # 确保合并后的规则只服务于特定的法律领域
        def refine_rules_with_labels(rules: list[str], tasks: list[str]) -> list[str]:
            # 提取这些任务的共有标签（罪名/法条），作为生成的约束上下文
            labels_summary = ""
            try:
                # 采样部分任务来获取标签分布
                sample_tasks = tasks[:10]
                accs_set = set()
                arts_set = set()
                for t in sample_tasks:
                    try:
                        # 尝试检索任务元数据
                        docs = self.task_storage.similarity_search(query=t, k=1)
                        if docs:
                            msg = MASMessage.from_dict(docs[0].metadata)
                            exp = msg.get_extra_field('expected') or {}
                            res = msg.get_extra_field('result_json') or {}
                            
                            for a in (exp.get('accusation') or res.get('accusation') or []):
                                s = str(a).strip()
                                if s: accs_set.add(s)
                            for idv in (exp.get('relevant_articles') or res.get('relevant_articles') or []):
                                try: arts_set.add(int(idv))
                                except: pass
                    except:
                        pass
                
                parts = []
                if accs_set:
                    parts.append(f"涉及罪名: {', '.join(accs_set)}")
                if arts_set:
                    parts.append(f"涉及法条: {', '.join(map(str, arts_set))}")
                if parts:
                    labels_summary = " | ".join(parts)
            except Exception:
                labels_summary = ""

            return self._merge_rules(rules, tasks, labels_hint=labels_summary)

        
        refined_label_tasks: dict[str, list[str]] = {} 
        def compute_jaccard(set_a: set, set_b: set) -> float:
            if not set_a and not set_b: return 1.0
            if not set_a or not set_b: return 0.0
            return len(set_a & set_b) / len(set_a | set_b)

        # 辅助函数：提取任务的标签集合（法条 + 罪名）
        # 修改：标签集 = GroundTruth (expected) U ModelPrediction (result)
        # 这样可以完整捕捉“该任务涉及的真实标签”以及“该任务容易被混淆成的错误标签”
        def get_task_labels(task_main: str) -> set[str]:
            lbls = set()
            try:
                docs = self.task_storage.similarity_search(query=task_main, k=1)
                if docs:
                    msg = MASMessage.from_dict(docs[0].metadata)
                    exp = msg.get_extra_field('expected') or {}
                    res = msg.get_extra_field('result_json') or {}
                    
                    # 收集所有相关法条（包括正确答案和模型误判）
                    t_arts = set()
                    for src in [exp, res]:
                        for idv in (src.get('relevant_articles') or []):
                            try: t_arts.add(f"ART_{idv}")
                            except: pass
                    if t_arts:
                        lbls.update(t_arts)
                    
                    # 收集所有相关罪名（包括正确答案和模型误判）
                    t_accs = set()
                    for src in [exp, res]:
                        for a in (src.get('accusation') or []):
                            s = str(a).strip()
                            if s: t_accs.add(s)
                    if t_accs:
                        lbls.update(t_accs)
            except:
                pass
            return lbls

        # 对每个 Finch 聚类结果进行二次检查
        for cluster_id, tasks in label_tasks.items():
            # 严格聚类：
            # 1. 取出一个任务作为新簇的种子
            # 2. 找到所有与该种子标签集完全一致（或高度重叠）的任务
            # 3. 用户的要求是：法条标签集或者罪名标签集完全一样 或者两个标签集都完全一样 才能合并
            #    鉴于罪名和法条通常是绑定的，我们这里严格要求整个标签集合（法条U罪名）完全一致
            #    或者在极少数情况下允许非常高的重叠（如 Jaccard > 0.9）以容错
            
            pending_tasks = list(tasks)
            sub_cluster_idx = 0
            
            while pending_tasks:
                seed_task = pending_tasks.pop(0)
                seed_labels = get_task_labels(seed_task)
                
                current_group = [seed_task]
                
                # 寻找同类任务
                unmatched = []
                for t in pending_tasks:
                    t_labels = get_task_labels(t)
                    
                    # 严格匹配：标签集合必须完全一致
                    # 例如：A={124, 288}, B={124, 288} -> 合并
                    #      A={124, 288}, C={124}      -> 不合并 (C属于纯124类，A属于124/288混淆类)
                    if seed_labels == t_labels:
                        current_group.append(t)
                    else:
                        unmatched.append(t)
                
                pending_tasks = unmatched
                
                # 保存该组
                if seed_labels:
                    # 签名直接反映了该组的特征，例如 "ART_124_ART_288"
                    sig = "_".join(sorted(list(seed_labels)))
                else:
                    sig = "UNK"
                
                new_key = f"{cluster_id}::{sub_cluster_idx}::{sig}"
                refined_label_tasks[new_key] = current_group
                sub_cluster_idx += 1

        merged_label_rules: dict[str, list[str]] = {}
        
        # 对细分后的组进行合并
        for group_key, related_task_mains in refined_label_tasks.items():
            # 如果组内任务太少（例如 < 2），可能不适合强行合并，容易产生过拟合经验
            # 但为了不丢弃，还是允许合并，只是 LLM 可能会保留原样
            
            related_ids, related_insights = self._find_related_insights(task_mains=related_task_mains)
            related_rules: list[str] = [insight['rule'] for insight in related_insights]
            
            if not related_rules:
                continue

            # 使用优化后的合并逻辑
            merged_rules: list[str] = refine_rules_with_labels(related_rules, related_task_mains)
            
            # Fallback: 若合并后规则为空（可能是 LLM 拒答或解析失败），则保留原规则，避免经验丢失
            if not merged_rules and related_rules:
                self.logger.warning(f"Merge returned empty for group {group_key}, fallback to keep original {len(related_rules)} rules.")
                merged_rules = related_rules

            merged_label_rules[group_key] = merged_rules

            self.logger.info('------- Merge Insights (Refined) -------')
            self.logger.info(f'Group Key: {group_key}')
            self.logger.info(f"Origin rules ({len(related_rules)}): \n{'\n'.join(related_rules)}")
            self.logger.info(f"Merged rules ({len(merged_rules)}): \n{'\n'.join(merged_rules)}")
            
        has_merged = any(len(rules) > 0 for rules in merged_label_rules.values())
        # 清除缓存，因为洞见已经改变
        if has_merged:
            self.embedding_cache.clear()

        self.insights_memory.clear()

        for group_key, related_rules in merged_label_rules.items():
            related_task_mains = refined_label_tasks.get(group_key)
            
            for rule in related_rules:
                insight: dict = {
                    'rule': rule,
                    'score': 2,          
                    'positive_correlation_tasks': list(related_task_mains),
                    'negative_correlation_tasks': list()
                }
                self.insights_memory.append(insight)
        
        self._index_done()

    def _merge_rules(self, rules: list[str], task_mains: list[str] = None, labels_hint: str = "") -> list[str]:
        def parse_numbered_list(text: str) -> list[str]:
            lines = [ln.rstrip() for ln in text.strip().splitlines()]
            def is_num_line(ln: str) -> bool:
                return bool(re.match(r'^\s*(?:\d+|[一二三四五六七八九十百]+|（\d+）|（[一二三四五六七八九十百]+）|[①-⑳])(?:[\.、\)]|）)\s+', ln))
            starts = [i for i, ln in enumerate(lines) if is_num_line(ln)]
            if not starts:
                t = text.strip()
                return [t] if t else []
            items = []
            for idx, s in enumerate(starts):
                e = starts[idx+1] if idx+1 < len(starts) else len(lines)
                item = "\n".join(lines[s:e]).strip()
                if item:
                    item = re.sub(r'^\s*(?:\d+|[一二三四五六七八九十百]+|（\d+）|（[一二三四五六七八九十百]+）|[①-⑳])(?:[\.、\)]|）)\s+', '', item)
                    items.append(item)
            return items
        
        merged_rules = []
        batch_size = 10

        for i in range(0, len(rules), batch_size):
            batch = rules[i:i + batch_size]
            if len(batch) == 0:
                continue
            limit_num: int = max(1, len(batch)*0.6)
            context_text = ""
            if task_mains:
                try:
                    samples = task_mains[:5]
                    samples_fmt = []
                    for idx, t in enumerate(samples, 1):
                        s = str(t or "")
                        if len(s) > 300:
                            s = s[:300]
                        samples_fmt.append(f"{idx}) {s}")
                    context_text = "\n".join(samples_fmt)
                except Exception:
                    context_text = ""
            
            # 如果有标签提示，则注入到 Context 中，强化模型对特定罪名/法条的关注
            if labels_hint:
                context_text = f"【案件类型约束：{labels_hint}】\n" + context_text

            user_prompt = CaseMemoryPrompts.merge_rules_user_prompt.format(
                current_rules='\n'.join(batch),
                limited_number=limit_num,
                task_context=context_text
            )
            messages = [Message('system', CaseMemoryPrompts.merge_rules_system_prompt),
                        Message('user', user_prompt)]
            raw_merged_rules = self.llm_model(messages)
            merged_rules.extend(parse_numbered_list(raw_merged_rules))

        return merged_rules

    def backward(self, insight: str, reward: float):
        
        for inner_insight in self.insights_memory:
            try:
                if insight in inner_insight['rule']:
                    inner_insight['score'] += reward
            except Exception:
                pass

        # 建立洞见与交互图的关联
        # 为每个生成的洞见，尝试找到它在交互图中对应的任务节点
        # 这里的关联主要是逻辑上的：insights.json 中存储了 positive_correlation_tasks
        # 而这些 task_main 是 TaskLayer 图中的节点 Key
        # 因此，只要保持 positive_correlation_tasks 准确，就能通过 TaskLayer.graph 追溯到交互轨迹
        
        self.clear_insights()
        self._index_done()

    def clear_insights(self):
        self.insights_memory = [self.insights_memory[i] for i in range(len(self.insights_memory)) 
                        if self.insights_memory[i]['score'] > 0] 

    def _retrieve_memory(
        self,
        query_task: str,   
        successful_topk: int = 1, 
        failed_topk: int = 1
    ) -> tuple[list[MASMessage], list[MASMessage]]:

        true_tasks_doc: list[tuple[Document, float]] = []
        false_tasks_doc: list[tuple[Document, float]] = []

        if successful_topk != 0:
            true_tasks_doc = self.task_storage.similarity_search_with_score(
                query=query_task, k=successful_topk, filter={'label': True}
            )
        if failed_topk != 0:
            false_tasks_doc = self.task_storage.similarity_search_with_score(
                query=query_task, k=failed_topk, filter={'label': False}
            )
        sorted(true_tasks_doc, key=lambda x: x[1]) 
        sorted(false_tasks_doc, key=lambda x: x[1]) 

        true_task_messages: list[MASMessage] = []
        false_task_messages: list[MASMessage] = []
        for doc in true_tasks_doc:
            meta_data: dict = doc[0].metadata
            mas_message: MASMessage = MASMessage.from_dict(meta_data)
            true_task_messages.append(mas_message)
        
        for doc in false_tasks_doc:
            meta_data: dict = doc[0].metadata
            mas_message: MASMessage = MASMessage.from_dict(meta_data)
            false_task_messages.append(mas_message)

        return true_task_messages, false_task_messages
    
    @property
    def task_size(self):
        num_records = self.task_storage.get()["ids"]
        return len(num_records)
    
    def _find_related_insights(
        self,
        task_mains: list[str],
        threshold: float = 1
    ) -> tuple[list[int], list[dict]]:

        rule_set: list[tuple[dict, int, int]] = []  # (rule, score, index)

        for idx, rule in enumerate(self.insights_memory):
            score: int = sum(task in rule.get('positive_correlation_tasks', []) for task in task_mains)
            if score >= threshold:
                rule_set.append((rule, score, idx))

        rule_set.sort(key=lambda x: x[1], reverse=True)

        rule_indices = [item[2] for item in rule_set]
        sorted_rules = [item[0] for item in rule_set]

        return rule_indices, sorted_rules
    def finetune_insights(self, num_points: int):
        SUCCESS_TASK_NUM, FAIL_TASK_NUM = 3, 1
        all_ids = self.task_storage.get()['ids']
        for _ in range(num_points):  
            random_id = random.choice(all_ids)
            random_entry = self.task_storage.get(ids=[random_id])
            if 'metadatas' in random_entry and random_entry['metadatas']:
                random_metadata = random_entry['metadatas'][0]  
            else:
                raise RuntimeError('Incomplete data.')
            mas_message: MASMessage = MASMessage.from_dict(random_metadata)
            true_trajs, false_trajs = self._retrieve_memory(
                query_task=mas_message.task_main, successful_topk=SUCCESS_TASK_NUM, failed_topk=FAIL_TASK_NUM
            )
            # 新增：强制过滤检索回来的案例，必须与 random_metadata 共享至少一个标签（罪名或法条）
            # 避免检索到“语义相似但罪名完全不同”的案例参与混淆
            def has_common_labels(msg_a: MASMessage, msg_b: MASMessage) -> bool:
                def get_l_a(m):
                    e = m.get_extra_field('expected') or {}
                    r = m.get_extra_field('result_json') or {}
                    l = set(e.get('accusation') or r.get('accusation') or [])
                    a = set()
                    for x in (e.get('relevant_articles') or r.get('relevant_articles') or []):
                        try: a.add(int(x))
                        except: pass
                    return l, a
                l1, a1 = get_l_a(msg_a)
                l2, a2 = get_l_a(msg_b)
                return bool((l1 & l2) or (a1 & a2))

            true_trajs = [t for t in true_trajs if has_common_labels(t, mas_message)]
            false_trajs = [t for t in false_trajs if has_common_labels(t, mas_message)]
            
            if mas_message.label == True:
                true_trajs.append(mas_message)
            else:
                false_trajs.append(mas_message)
            all_task_mains: list[str] = [traj.task_main for traj in true_trajs + false_trajs]
            try:
                neigh = self.task_layer.retrieve_related_task(mas_message.task_main, node_num=SUCCESS_TASK_NUM + FAIL_TASK_NUM, hop=1)
                all_task_mains += [x for x in neigh if isinstance(x, str)]
            except Exception:
                pass

            related_insight_ids, _ = self._find_related_insights(all_task_mains, len(all_task_mains) / 2)
            self._finetune_insights(true_trajs, false_trajs, related_insight_ids)
        
        self.clear_insights()
        self._index_done()
    def _finetune_insights(
        self,
        successful_task_trajectories: list[MASMessage],
        failed_task_trajectories: list[MASMessage],
        insight_ids: list[int]
    ) -> None:

        def map_operations(origin_operations: list[tuple]) -> list[tuple]:
            processed_operations: list[tuple] = []
            for (operation, text) in origin_operations:
                res: list = operation.split(' ')

                if len(res) == 2:
                    if len(insight_ids) == 0:    
                        continue
                    insight_id: int = int(res[1]) - 1
                    if insight_id >= len(insight_ids) or insight_id < 0:
                        continue
                    
                    res[1] = str(insight_ids[insight_id] + 1)   
                    operation: str = ' '.join(res)
                processed_operations.append((operation, text))
            
            return processed_operations

        rule_list: list[dict] = [self.insights_memory[i] for i in insight_ids]

        compare_pairs: list[tuple[MASMessage, MASMessage]] = []
        def extract_labels(msg: MASMessage) -> tuple[set[str], set[int]]:
            labels: set[str] = set()
            arts: set[int] = set()
            try:
                exp = msg.get_extra_field('expected') or {}
                res = msg.get_extra_field('result_json') or {}
                accs = exp.get('accusation') or res.get('accusation') or []
                arts_list = exp.get('relevant_articles') or res.get('relevant_articles') or []
                for a in accs:
                    s = str(a).strip()
                    if s:
                        labels.add(s)
                for idv in arts_list:
                    try:
                        arts.add(int(idv))
                    except Exception:
                        pass
            except Exception:
                pass
            # 移除：不再从 query (Fact) 中尝试提取关键词罪名
            # 原因：与检索阶段保持一致，避免引入不准确的“猜测标签”导致跨罪名合并
            # if not labels: ...
            
            return labels, arts

        from collections import defaultdict
        succ_map: dict[str, list[MASMessage]] = defaultdict(list)
        fail_map: dict[str, list[MASMessage]] = defaultdict(list)
        for m in successful_task_trajectories:
            lbls, arts = extract_labels(m)
            for l in (lbls or {'__UNK__'}):
                succ_map[l].append(m)
            for a in arts:
                succ_map[f'ART_{a}'].append(m)
        for m in failed_task_trajectories:
            lbls, arts = extract_labels(m)
            for l in (lbls or {'__UNK__'}):
                fail_map[l].append(m)
            for a in arts:
                fail_map[f'ART_{a}'].append(m)
        used_succ: set[int] = set()
        used_fail: set[int] = set()
        def add_pairs_for_keys(keys: list[str]):
            nonlocal compare_pairs, used_succ, used_fail
            for k in keys:
                s_list = succ_map.get(k, [])
                f_list = fail_map.get(k, [])
                si = 0
                fi = 0
                while si < len(s_list) and fi < len(f_list):
                    s_msg = s_list[si]; f_msg = f_list[fi]
                    si += 1; fi += 1
                    try:
                        s_idx = successful_task_trajectories.index(s_msg)
                        f_idx = failed_task_trajectories.index(f_msg)
                    except Exception:
                        continue
                    if (s_idx in used_succ) or (f_idx in used_fail):
                        continue
                    compare_pairs.append((s_msg, f_msg))
                    used_succ.add(s_idx); used_fail.add(f_idx)
        keys = list(set(succ_map.keys()) & set(fail_map.keys()))
        # 过滤掉 UNK 和 纯数字键（如果有的话，这里主要是为了确保键是有意义的标签或ART_前缀）
        keys = [k for k in keys if k != '__UNK__']
        add_pairs_for_keys(keys)
        for id, fail_task in enumerate(failed_task_trajectories):
            if id >= len(successful_task_trajectories):
                break
            if (id in used_fail) or (id in used_succ):
                continue
            success_task = successful_task_trajectories[id]
            
            # 只有当成功案例和失败案例标签集完全一致时，才进行对比
            # 例如：成功案例是 {124, 288} (破坏交通工具 + 破坏交通设施混淆)
            #       失败案例也必须是 {124, 288} 才能对比
            #       如果失败案例只是 {124}，则不能对比，因为经验适用范围不同
            
            s_lbls, s_arts = extract_labels(success_task)
            f_lbls, f_arts = extract_labels(fail_task)
            
            # 构建统一的标签集合 (GroundTruth U ModelPrediction)
            # 注意：extract_labels 内部已经处理了 exp 和 res 的并集
            s_set = set(s_lbls) | set(f"ART_{a}" for a in s_arts)
            f_set = set(f_lbls) | set(f"ART_{a}" for a in f_arts)
            
            # 严格校验：标签集合必须完全一致
            if s_set != f_set:
                continue

            compare_pairs.append((success_task, fail_task))
        
        # 修改：按法条/罪名分组成功案例，确保同一组内的案例具有相似的法律特征
        # 避免将风马牛不相及的案例（如强奸 vs 诈骗）混在一起提取经验
        def group_tasks_by_labels(tasks: list[MASMessage]) -> list[list[MASMessage]]:
            # 基于严格标签匹配的贪婪分组
            
            def get_msg_labels(m: MASMessage) -> set[str]:
                l, a = extract_labels(m)
                return set(l) | set(f"ART_{x}" for x in a)

            remaining = list(tasks)
            chunks = []
            
            while remaining:
                seed = remaining.pop(0)
                seed_lbls = get_msg_labels(seed)
                current_chunk = [seed]
                
                unmatched = []
                for t in remaining:
                    t_lbls = get_msg_labels(t)
                    # 严格匹配：标签集必须完全一致
                    if seed_lbls == t_lbls:
                        current_chunk.append(t)
                    else:
                        unmatched.append(t)
                
                remaining = unmatched
                
                # 切分过大的组
                random.shuffle(current_chunk)
                for i in range(0, len(current_chunk), 5):
                    chunks.append(current_chunk[i:i+5])
            
            return chunks

        successful_task_chunks = group_tasks_by_labels(successful_task_trajectories)
        
        MAX_RULE_THRESHOLD: int = 10
        suffix: str = CaseMemoryPrompts.finetune_insights_suffix['full'] if len(self.insights_memory) > MAX_RULE_THRESHOLD \
                      else CaseMemoryPrompts.finetune_insights_suffix['not_full']


        self.logger.info('--------------- Finetune Insights ---------------')
        for pair in compare_pairs:
            compare_prompts: list[Message] = self._build_comparative_prompts(pair[0], pair[1], rule_list)
            compare_prompts[0] = replace(compare_prompts[0], content=compare_prompts[0].content + suffix)
            response: str = self.llm_model(compare_prompts)
            parsed_operations = self._parse_rules(response)
            processed_operations = map_operations(parsed_operations)
            self._update_rules(
                [pair[0].task_main, pair[1].task_main], 
                processed_operations, 
                MAX_RULE_THRESHOLD
            )
            self.logger.info(compare_prompts[0].role + compare_prompts[0].content + '\n\n' + compare_prompts[1].role + compare_prompts[1].content)
            self.logger.info(response)
            self.logger.info('\n---------------\n')

        for chunk in successful_task_chunks:
            success_prompts: list[Message] = self._build_success_prompts(chunk, rule_list) 
            success_prompts[0] = replace(success_prompts[0], content=success_prompts[0].content + suffix)
            response: str = self.llm_model(success_prompts)
            parsed_operations = self._parse_rules(response)
            processed_operations = map_operations(parsed_operations)
            task_mains: list[str] = [traj.task_main for traj in chunk]
            self._update_rules(
                task_mains, 
                processed_operations, 
                MAX_RULE_THRESHOLD
            )
            self.logger.info(success_prompts[0].role + success_prompts[0].content + '\n\n' + success_prompts[1].role + success_prompts[1].content)
            self.logger.info(response)
            self.logger.info('\n---------------\n')
        
        self.clear_insights()
        self._index_done()

    def _index_done(self):
        # Persist insights.json with inter-process lock
        try:
            with InterProcessFileLock(self._lock_path, timeout=120):
                write_json(self.insights_memory, self.persist_file)
        except TimeoutError as e:
            print(f"InsightsManager: lock timeout when writing insights -> {e}")

    def _build_comparative_prompts(self, true_traj: MASMessage, false_traj: MASMessage, insights: list[dict]) -> list[Message]:
        existing_rules: list[str] = [insight['rule'] for insight in insights]
        if len(existing_rules) == 0:
            existing_rules.append('')
        rule_text: str = '\n'.join([f'{i}. {r}' for i, r in enumerate(existing_rules, 1)])
        rule_text = rule_text.replace('{', '{{').replace('}', '}}')

        t1 = (true_traj.task_description or '').replace('{', '{{').replace('}', '}}')
        t1traj = (true_traj.task_trajectory or '').replace('{', '{{').replace('}', '}}')
        t2 = (false_traj.task_description or '').replace('{', '{{').replace('}', '}}')
        t2traj = (false_traj.task_trajectory or '').replace('{', '{{').replace('}', '}}')
        freason = (false_traj.get_extra_field('fail_reason') or '').replace('{', '{{').replace('}', '}}')

        sug_text = ''
        try:
            sugs = true_traj.get_extra_field('verify_suggestions') or []
            if isinstance(sugs, list) and sugs:
                recent = sugs[-3:]
                sug_text = '\n'.join(recent)
        except Exception:
            sug_text = ''
        prompt = CaseMemoryPrompts.critique_compare_rules_user_prompt.format(   
            task1=t1,
            task1_trajectory=t1traj,   
            task2=t2,
            task2_trajectory=t2traj,
            fail_reason=(freason + ("\n\n# 同案验证建议摘要\n" + sug_text if sug_text else "")),
            existing_rules=rule_text
        )
        try:
            _ = prompt
        except Exception:
            prompt = (
                "## Trial Task 1 (success):\n" + t1 + "\n" + t1traj + "\n\n" +
                "## Trial Task 2 (fail):\n" + freason + "\n" + t2 + "\n" + t2traj + "\n\n" +
                "## Here are the EXISTING RULES:\n" + rule_text
            )

        return [Message(role='system', content= CaseMemoryPrompts.critique_compare_rules_system_prompt), Message(role='user', content=prompt)] 
    
    def _build_success_prompts(
        self,
        success_trajectories: Iterable[MASMessage],
        insights: list[dict],
    ) -> list[Message]:

        existing_rules: list[str] = [insight['rule'] for insight in insights]
        if len(existing_rules) == 0:
            existing_rules.append('')
        rule_text: str = '\n'.join([f'{i}. {r}' for i, r in enumerate(existing_rules, 1)])

        history: list[str] = []
        for i, task in enumerate(success_trajectories):
            # 优先使用原始 FACT（去除前缀“Fact:”），无则回退到事实摘要
            raw_facts = (task.task_main or '')
            try:
                if raw_facts.lower().startswith('fact:'):
                    raw_facts = raw_facts[len('fact:'):].strip()
            except Exception:
                pass
            facts_src = raw_facts if raw_facts.strip() else (task.get_extra_field('facts_summary') or '')
            facts = (facts_src or '').replace('{', '{{').replace('}', '}}')
            agent_map = task.get_extra_field('agent_steps') or {}
            result_finish = (task.get_extra_field('result_finish') or '').replace('{', '{{').replace('}', '}}')
            try:
                from legal_task.mas_workflow.format import format_task_context
                block = format_task_context(
                    facts=facts,
                    task_description=task.task_description,
                    agent_steps=agent_map if isinstance(agent_map, dict) else None,
                    result_finish=result_finish
                )
            except Exception:
                key_steps = (task.get_extra_field('key_steps') or '').replace('{', '{{').replace('}', '}}')
                agent_steps = ''
                try:
                    if isinstance(agent_map, dict) and agent_map:
                        agent_steps = '\n'.join([f"- {k}: {v}" for k, v in agent_map.items()])
                except Exception:
                    agent_steps = ''
                block = (
                    "### 成功先例：事实摘要\n" + facts + "\n\n" +
                    ("### 成功先例：协作推理摘要\n" + agent_steps + "\n" if agent_steps else "") +
                    ("### 成功先例：判决结果（原始）\n" + result_finish + "\n" if result_finish else "")
                )
            history.append(f"Task {i+1}:\n" + block)
        history_text = '\n'.join(history)
        try:
            sug_blocks = []
            for task in success_trajectories:
                sugs = task.get_extra_field('verify_suggestions') or []
                if isinstance(sugs, list) and sugs:
                    recent = sugs[-3:]
                    sug_blocks.append('\n'.join(recent))
            if sug_blocks:
                history_text = history_text + "\n\n# 同案验证建议摘要\n" + ('\n---\n'.join(sug_blocks))
        except Exception:
            pass
        try:
            prompt = CaseMemoryPrompts.critique_success_rules_user_prompt.format(
                success_history=history_text,
                existing_rules=rule_text
            )
        except Exception:
            prompt = (
                "## Here are the trials:\n" + history_text + "\n\n" +
                "## Here are the EXISTING RULES:\n" + rule_text
            )

        return [Message(role='system', content=CaseMemoryPrompts.critique_success_rules_system_prompt), Message(role='user', content=prompt)]
    
    def _parse_rules(self, llm_text):
        pattern = r'((?:REMOVE|EDIT|ADD|AGREE)(?: \d+|)): (?:[a-zA-Z\s\d]+: |)(.*)'
        matches = re.findall(pattern, llm_text)

        res = []
        banned_words = ['ADD', 'AGREE', 'EDIT']
        for operation, text in matches:
            text = text.strip()
            if text != '' and not any([w in text for w in banned_words]) and (text.endswith('.') or text.endswith('。')):

                if 'ADD' in operation:
                    res.append(('ADD', text))
                else:
                    res.append((operation.strip(), text))
        return(res)
    
    def _update_rules(
        self,
        relative_tasks: list[str],
        operations: list[tuple[str, str]], 
        max_rules_num: int = 10
    ) -> None:

        delete_indices = []
        for i in range(len(operations)):
            operation, operation_rule_text = operations[i]
            operation_type = operation.split(' ')[0]
            rule_num = int(operation.split(' ')[1]) if ' ' in operation else None

            if operation_type == 'ADD':    
                if self._is_existing_rule(operation_rule_text): 
                    delete_indices.append(i)
                    
            elif operation_type == 'EDIT':   
                if self._is_existing_rule(operation_rule_text): 
                    rule_num: int = self._retrieve_rule_index(operation_rule_text)
                    operations[i] = (f'AGREE {rule_num + 1}', operation_rule_text)   

                elif (rule_num is None) or (rule_num > len(self.insights_memory)) or (rule_num <= 0):   
                    delete_indices.append(i)
                        
            elif operation_type == 'REMOVE' or operation_type == 'AGREE':  
                if (rule_num is None) or (rule_num > len(self.insights_memory)) or (rule_num <= 0):   
                    delete_indices.append(i)
            
            else: 
                delete_indices.append(i)

        operations = [operations[i] for i in range(len(operations)) if i not in delete_indices] 
        

        list_full: bool = len(self.insights_memory) >= max_rules_num  
        for op in ['REMOVE', 'AGREE', 'EDIT', 'ADD']: 
            for i in range(len(operations)):
                operation, operation_rule_text = operations[i]
                operation_type = operation.split(' ')[0]
                if operation_type != op:
                    continue

                if operation_type == 'REMOVE': 
                    rule_index = int(operation.split(' ')[1]) - 1
                    rule_data: dict = self.insights_memory[rule_index]
                    remove_strength = 3 if list_full else 1
                    rule_data['score'] -= remove_strength
                    rule_data['negative_correlation_tasks'] = list(set(rule_data['negative_correlation_tasks'] + relative_tasks))  

                elif operation_type == 'AGREE':
                    rule_index: int = self._retrieve_rule_index(operation_rule_text) 
                    rule_data: dict = self.insights_memory[rule_index]
                    rule_data['score'] += 1
                    rule_data['positive_correlation_tasks'] = list(set(rule_data['positive_correlation_tasks'] + relative_tasks))

                elif operation_type == 'EDIT': 
                    rule_index = int(operation.split(' ')[1]) - 1
                    rule_data: dict = self.insights_memory[rule_index]
                    rule_data['rule'] = operation_rule_text
                    rule_data['score'] += 1
                    rule_data['positive_correlation_tasks'] = list(set(rule_data['positive_correlation_tasks'] + relative_tasks))

                elif operation_type == 'ADD': 
                    meta_data: dict = {
                        'rule': operation_rule_text,
                        'score': 2,         
                        'positive_correlation_tasks': list(relative_tasks),
                        'negative_correlation_tasks': list()
                    }
                    self.insights_memory.append(meta_data)

    def _is_existing_rule(self, operation_rule_text: str) -> bool:

        for insight in self.insights_memory:
            if insight['rule'] in operation_rule_text:
                return True
        return False
    
    def _retrieve_rule_index(self, operation_rule_text: str) -> int:

        for idx, insight in enumerate(self.insights_memory):
            if insight['rule'] in operation_rule_text:
                return idx
        return -1
