from __future__ import annotations
from typing import List, Tuple, Dict

from mas.agents import Agent
from mas.llm import Message
from mas.reasoning import ReasoningConfig


class Node:

    def __init__(self, agent: Agent):
        assert agent is not None, "Node's agent cannot be None."
        self._id: str = agent.name
        self._agent: Agent = agent
        self._spatial_predecessors: List[Node] = []
        self._spatial_successors: List[Node] = []
        self._temporal_predecessors: List[Node] = []
        self._temporal_successors: List[Node] = []
        self._input: list[str] = []
        self._output: list[str] = []
        self._memory: Dict[str, List[str]] = {'inputs': [], 'outputs': []}
        self.reasoning_config = ReasoningConfig(temperature=0, max_tokens=128, stop_strs=None)

    @property
    def id(self) -> str:
        return self._id

    @property
    def role(self) -> str:
        return self._agent.profile

    @property
    def spatial_successors(self) -> Tuple[Node]:
        return tuple(self._spatial_successors)

    @property
    def spatial_predecessors(self) -> Tuple[Node]:
        return tuple(self._spatial_predecessors)

    @property
    def temporal_successors(self) -> Tuple[Node]:
        return tuple(self._temporal_successors)

    @property
    def temporal_predecessors(self) -> Tuple[Node]:
        return tuple(self._temporal_predecessors)

    @property
    def current_output(self) -> Tuple[str]:
        return tuple(self._output)

    @property
    def current_input(self) -> Tuple[str]:
        return tuple(self._input)

    @property
    def memory(self) -> Tuple[list, list]:
        return (self._memory['inputs'].copy(), self._memory['outputs'].copy())

    @staticmethod
    def add_spatial_edge(source_node: Node, target_node: Node) -> None:
        source_node._spatial_successors.append(target_node)
        target_node._spatial_predecessors.append(source_node)

    @staticmethod
    def add_temporal_edge(source_node: Node, target_node: Node) -> None:
        source_node._temporal_successors.append(target_node)
        target_node._temporal_predecessors.append(source_node)

    @staticmethod
    def remove_spatial_edge(source_node: Node, target_node: Node) -> None:
        if target_node in source_node._spatial_successors:
            source_node._spatial_successors.remove(target_node)
        if source_node in target_node._spatial_predecessors:
            target_node._spatial_predecessors.remove(source_node)

    @staticmethod
    def remove_temporal_edge(source_node: Node, target_node: Node) -> None:
        if target_node in source_node._temporal_successors:
            source_node._temporal_successors.remove(target_node)
        if source_node in target_node._temporal_predecessors:
            target_node._temporal_predecessors.remove(source_node)

    def clear_spatial_connections(self) -> None:
        for successor in list(self._spatial_successors):
            Node.remove_spatial_edge(self, successor)
        for predecessor in list(self._spatial_predecessors):
            Node.remove_spatial_edge(predecessor, self)

    def clear_temporal_connections(self) -> None:
        for successor in list(self._temporal_successors):
            Node.remove_temporal_edge(self, successor)
        for predecessor in list(self._temporal_predecessors):
            Node.remove_temporal_edge(predecessor, self)

    def update_memory(self):
        if self._input:
            self._memory['inputs'].append(self._input[0])
        if self._output:
            self._memory['outputs'].append(self._output[0])

    def clear_state(self):
        self._memory['inputs'] = []
        self._memory['outputs'] = []
        self._output = []
        self._input = []

    def get_spatial_upstream_info(self) -> Dict[str, Dict[str, str]]:
        predecessors: Tuple[Node] = self.spatial_predecessors
        upstream_info = {}
        for predecessor in predecessors:
            if not predecessor._output:
                continue
            predecessor_output = predecessor._output[0]
            upstream_info[predecessor._id] = {"role": predecessor.role, "output": predecessor_output}
        return upstream_info

    def get_temporal_upstream_info(self) -> Dict[str, Dict[str, str]]:
        predecessors: Tuple[Node] = self.temporal_predecessors
        upstream_info = {}
        for predecessor in predecessors:
            history_outputs = predecessor._memory['outputs']
            if not history_outputs:
                continue
            filtered_outputs = history_outputs[:-1] if len(history_outputs) > 1 else []
            if not filtered_outputs:
                continue
            combined = ""
            for idx, out in enumerate(filtered_outputs):
                combined += f"## Round {idx+1} Output\n\n{out}\n\n"
            upstream_info[predecessor._id] = {"role": predecessor.role, "output": combined.strip()}
        return upstream_info

    def execute(self, user_message: Message, use_critic: bool) -> str:
        self._output, self._input = [], []
        spatial_info: Dict[str, Dict] = self.get_spatial_upstream_info()
        temporal_info: Dict[str, Dict] = self.get_temporal_upstream_info()
        user_prompt: str = self._process_inputs(user_message, spatial_info, temporal_info, use_critic)
        answer: str = self._agent.response(user_prompt, self.reasoning_config)
        self._input = [user_prompt]
        self._output = [answer]
        return answer

    def _process_inputs(self, user_message: Message, spatial_info: dict, temporal_info: dict, use_critic: bool) -> str:
        user_prompt = user_message.content
        role = self._agent.profile
        allowed_spatial_roles = {
            'event': set(),
            'retrieval': {'event'},
            'judgment': set(),
            'verification': {'judgment'},
            'decision': {'event', 'retrieval', 'judgment', 'verification'}
        }.get(role, set())
        allowed_temporal_roles = {
            'event': set(),
            'retrieval': set(),
            'judgment': set(),
            'verification': set(),
            'decision': {'judgment', 'verification'}
        }.get(role, set())

        spatial_upstream_message: str = ""
        temporal_upstream_message: str = ""
        for uuid, info in spatial_info.items():
            if info['role'] not in allowed_spatial_roles:
                continue
            spatial_upstream_message += f"## Agent {uuid}, role is {info['role']}, output is:\n\n {info['output']}\n"
            if use_critic and role != 'judgment':
                critic_message: str = self._critic_upstream_agent(user_prompt, info['output'])
                spatial_upstream_message += f"## Critic's Suggestions for Improvement:{critic_message}\n\n"
        for uuid, info in temporal_info.items():
            if info['role'] not in allowed_temporal_roles:
                continue
            temporal_upstream_message += f"## Agent {uuid}, role is {info['role']}, output is:\n\n {info['output']}\n\n"
            if use_critic and role != 'judgment':
                critic_message: str = self._critic_upstream_agent(user_prompt, info['output'])
                temporal_upstream_message += f"## Critic's Suggestions for Improvement:{critic_message}\n\n"
        final_prompt = ''
        if spatial_upstream_message != "":
            final_prompt += f"\n# Outputs from other agents in the current round:\n{spatial_upstream_message}\n"
            final_prompt += "-" * 20 + "\n"
        if temporal_upstream_message != "":
            final_prompt += f"# Outputs from previous round agents:\n{temporal_upstream_message}\n"
            final_prompt += "-" * 20 + "\n"
        return final_prompt + user_prompt

    def _critic_upstream_agent(self, task: str, agent_response: str) -> str:
        from .graph_prompt import critic_system_prompt, critic_user_prompt
        user_prompt: str = critic_user_prompt.format(task=task, agent_answer=agent_response)
        messages: list[Message] = [Message('system', critic_system_prompt), Message('user', user_prompt)]
        return self._agent.reasoning(messages, self.reasoning_config)
