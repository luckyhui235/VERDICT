import json
import re
import ast
import os
import fcntl
import time
from typing import Any, Literal

from .base_env import BaseEnv, BaseRecorder
from .utils import normalize_answer


_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


class CAILEnv(BaseEnv):
    """
    CAIL2018 环境：输入为事实文本 fact，输出需为包含三个子任务的 JSON：
    {
      "relevant_articles": [int, ...],
      "accusation": [str, ...],
      "term_of_imprisonment": {"death_penalty": bool, "life_imprisonment": bool, "imprisonment": int}
    }
    评估：全部字段完全匹配则判定正确；同时在观察信息中输出各子任务的正确性细节。
    """

    def __init__(self, env_config: dict[str, Any], max_trials: int = 12) -> None:
        self.env_config = env_config or {}
        self.max_trials: int = max_trials
        self.success_threshold: float = float(self.env_config.get('success_threshold', 1.0))
        self._term_bucket_map = None
        mapping_path = os.environ.get(
            'TERM_BUCKET_MAPPING_PATH',
            str(self.env_config.get('term_bucket_mapping_path') or ''),
        )
        self._term_bucket_mapping_path = (
            mapping_path
            if os.path.isabs(mapping_path)
            else os.path.join(_PROJECT_ROOT, mapping_path)
        )
        self.reset()

    def set_env(self, configs: dict) -> tuple[str, str]:
        if configs.get('task') is None:
            raise ValueError('The configs dict should have the `task` attribute (fact text).')
        if configs.get('expected') is None:
            raise ValueError('Please provide the `expected` meta for CAIL evaluation.')
        self.config = configs

        task_main: str = f"Fact: {self.config.get('task')}"
        instruction = (
            "请根据上述法律事实，输出一个 JSON 对象，包含三部分内容的预测：\n"
            "- relevant_articles: 法条编号数组，如 [266]\n"
            "- accusation: 罪名数组（中文短语），如 [\"诈骗\"]\n"
            "- term_of_imprisonment: {death_penalty: bool, life_imprisonment: bool, imprisonment: 月数int}\n"
            "格式要求：必须以 Finish[<JSON>] 输出，其中 <JSON> 为无注释、可解析的标准 JSON。"
        )
        return task_main, instruction

    def reset(self) -> None:
        self.current_task: str = None
        self.reward: float = 0
        self.last_pred: dict | None = None

    @staticmethod
    def _parse_action_type(action: str) -> Literal['action', 'thought']:
        if 'thought' in action.lower():
            return 'thought'
        else:
            return 'action'

    @staticmethod
    def process_action(action: str) -> str:
        import re
        text = (action or "").strip()
        text = text.replace('<', '').replace('>', '')
        text = text.replace('【', '[').replace('】', ']').replace('（', '(').replace('）', ')')

        # Prefer extracting an explicit Finish[...] (or variants) anywhere in the text
        # Use balanced bracket/paren scanning to avoid truncation on inner arrays
        def _extract_balanced(src: str, open_ch: str, close_ch: str, start_pos: int) -> str | None:
            in_str = False
            str_ch = ''
            depth = 0
            i = start_pos + 1  # start scanning AFTER the opening char
            while i < len(src):
                ch = src[i]
                if not in_str and (ch == '"' or ch == "'"):
                    in_str = True
                    str_ch = ch
                elif in_str and ch == str_ch:
                    # toggle only if not escaped
                    if i == 0 or src[i-1] != '\\':
                        in_str = False
                        str_ch = ''
                if not in_str:
                    if ch == open_ch:
                        depth += 1
                    elif ch == close_ch:
                        if depth == 0:
                            # Found the outer closing char corresponding to the opener at start_pos
                            return src[start_pos+1:i]
                        else:
                            depth -= 1
                i += 1
            return None

        # Find a Finish[...] occurrence and extract its full, balanced argument
        finish_idx = re.search(r'\bfinish\b', text, flags=re.IGNORECASE | re.DOTALL)
        if finish_idx:
            # Move to the next non-space char after the keyword
            j = finish_idx.end()
            while j < len(text) and text[j].isspace():
                j += 1
            if j < len(text):
                if text[j] == '[':
                    arg = _extract_balanced(text, '[', ']', j)
                    if arg is not None:
                        return f"Finish[{arg}]"
                elif text[j] == '(':
                    arg = _extract_balanced(text, '(', ')', j)
                    if arg is not None:
                        return f"Finish({arg})"
                elif text[j] == ':':
                    # Colon form: take the remainder after ':' as the argument
                    return text[j+1:].strip()

        # Fallback to first non-empty line
        first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), '')

        # Thought lines: return as-is
        if 'thought' in first_line.lower():
            return first_line

        # If there's a colon and the prefix is not Finish/Thought, drop the prefix
        # BUT avoid dropping when the colon belongs to JSON (inside braces/brackets/quotes)
        if ':' in first_line:
            pre, post = first_line.split(':', 1)
            pre_str = pre.strip()
            pre_lower = pre_str.lower()
            # Only treat as a line prefix if it's a simple token (no brackets/braces/quotes)
            is_simple_prefix = re.match(r'^[A-Za-z_][A-Za-z0-9_ ]*$', pre_str) is not None
            if is_simple_prefix and pre_lower not in ('finish', 'thought'):
                first_line = post.strip()

        # Clean trailing OK tokens
        first_line = first_line.replace('OK.', '').replace('OK', '').strip()

        # Final fallback: if not explicitly a Finish/Thought line, coerce to Finish[...] to avoid invalid actions
        lower = first_line.lower()
        if not (lower.startswith('finish') or lower.startswith('thought')) and first_line:
            return f"Finish[{first_line}]"

        return first_line

    # --- Robust JSON extraction & cleaning helpers ---
    @staticmethod
    def _extract_json_substring(s: str) -> str | None:
        """
        Try to locate the most likely JSON object substring by matching braces.
        Supports multi-line inputs and ignores braces inside quoted strings.
        Returns the substring or None if not found.
        """
        if not s:
            return None
        s = s.strip()
        # Normalize common non-ASCII quotes/brackets
        s = s.replace('【', '[').replace('】', ']').replace('（', '(').replace('）', ')')
        s = s.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")

        start_idx = None
        stack = []
        in_str = False
        str_char = ''
        i = 0
        while i < len(s):
            ch = s[i]
            # toggle string context
            if not in_str and (ch == '"' or ch == "'"):
                in_str = True
                str_char = ch
            elif in_str and ch == str_char:
                # only toggle if not escaped
                if i == 0 or s[i-1] != '\\':
                    in_str = False
                    str_char = ''

            if not in_str:
                if ch == '{':
                    if start_idx is None:
                        start_idx = i
                    stack.append('{')
                elif ch == '}':
                    if stack and stack[-1] == '{':
                        stack.pop()
                        if not stack and start_idx is not None:
                            return s[start_idx:i+1]
            i += 1
        # No balanced JSON object found
        return None

    @staticmethod
    def _clean_json_like_string(s: str) -> str:
        """
        Clean common near-JSON issues to increase parsing success:
        - Convert Python booleans/None to JSON.
        - Remove trailing commas before } or ].
        - Strip // and /* */ comments.
        - Normalize non-ASCII quotes.
        - Convert simple 'key': 'value' patterns to JSON-compliant double quotes.
        """
        if not s:
            return s
        # Remove comments
        s = re.sub(r'//.*', '', s)
        s = re.sub(r'/\*.*?\*/', '', s, flags=re.S)
        # Normalize quotes
        s = s.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
        # Python -> JSON literals
        s = re.sub(r'\bTrue\b', 'true', s)
        s = re.sub(r'\bFalse\b', 'false', s)
        s = re.sub(r'\bNone\b', 'null', s)
        # Remove trailing commas
        s = re.sub(r',\s*([}\]])', r'\1', s)
        # Convert 'key': to "key":
        s = re.sub(r"'([^'\\]*)'\s*:", r'"\1":', s)
        # Convert: : 'value' to : "value"
        s = re.sub(r":\s*'([^'\\]*)'", r': "\1"', s)
        # Ensure standard brackets
        s = s.replace('【', '[').replace('】', ']').replace('（', '(').replace('）', ')')
        return s

    @staticmethod
    def _parse_action(string: str) -> tuple[str, str]:
        s = (string or '').strip()
        s = s.replace('【', '[').replace('】', ']').replace('（', '(').replace('）', ')')

        # Balanced extraction for Finish[...] to avoid premature stop on inner arrays
        def _extract_balanced(src: str, open_ch: str, close_ch: str, start_pos: int) -> str | None:
            in_str = False
            str_ch = ''
            depth = 0
            i = start_pos + 1
            while i < len(src):
                ch = src[i]
                if not in_str and (ch == '"' or ch == "'"):
                    in_str = True
                    str_ch = ch
                elif in_str and ch == str_ch:
                    if i == 0 or src[i-1] != '\\':
                        in_str = False
                        str_ch = ''
                if not in_str:
                    if ch == open_ch:
                        depth += 1
                    elif ch == close_ch:
                        if depth == 0:
                            return src[start_pos+1:i]
                        else:
                            depth -= 1
                i += 1
            return None

        # Try to parse Finish[...] first
        m = re.match(r'^\s*finish\b', s, flags=re.IGNORECASE | re.DOTALL)
        if m:
            j = m.end()
            while j < len(s) and s[j].isspace():
                j += 1
            if j < len(s):
                if s[j] == '[':
                    arg = _extract_balanced(s, '[', ']', j)
                    if arg is not None:
                        return 'Finish', arg
                    # Fallback: if 'Finish[' is present but closing ']' is missing,
                    # try to locate a JSON object substring and use it as argument.
                    candidate = CAILEnv._extract_json_substring(s[j+1:])
                    if candidate:
                        return 'Finish', candidate
                elif s[j] == '(':
                    arg = _extract_balanced(s, '(', ')', j)
                    if arg is not None:
                        return 'Finish', arg
                    # Fallback for missing ')' similar to square bracket case
                    candidate = CAILEnv._extract_json_substring(s[j+1:])
                    if candidate:
                        return 'Finish', candidate
                elif s[j] == ':':
                    return 'Finish', s[j+1:].strip()

        # Thought variants using simpler patterns (lower risk of nesting issues)
        m = re.match(r'^\s*thought\s*\[\s*(.+?)\s*\]\s*$', s, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return 'Thought', m.group(1)
        m = re.match(r'^\s*thought\s*:\s*(.+?)\s*$', s, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return 'Thought', m.group(1)
        m = re.match(r'^\s*thought\s*\(\s*(.+?)\s*\)\s*$', s, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return 'Thought', m.group(1)

        # Plain "Finish answer" (space-separated)
        m = re.match(r'^\s*finish\s+(.+)$', s, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return 'Finish', m.group(1).strip()
        return None, None

    def _normalize_accusations(self, items: list[str]) -> list[str]:
        # 归一化：移除括号与中英文标点、统一顿号与逗号、去掉尾缀"罪"
        return [normalize_answer(str(x or "")) for x in items]

    def _months_to_bucket(self, months: int) -> int:
        try:
            if self._term_bucket_map is None:
                if os.path.exists(self._term_bucket_mapping_path):
                    with open(self._term_bucket_mapping_path, 'r', encoding='utf-8') as f:
                        mp = json.load(f)
                    try:
                        self._term_bucket_map = {int(k): int(v) for k, v in mp.items()}
                    except Exception:
                        self._term_bucket_map = {}
                else:
                    self._term_bucket_map = {}
        except Exception:
            self._term_bucket_map = {}
        if isinstance(months, int) and self._term_bucket_map and months in self._term_bucket_map:
            return int(self._term_bucket_map.get(months))
        if months <= 0:
            return 10
        if months <= 6:
            return 9
        if months <= 9:
            return 8
        if months <= 12:
            return 7
        if months <= 24:
            return 6
        if months <= 36:
            return 5
        if months <= 60:
            return 4
        if months <= 84:
            return 3
        if months <= 120:
            return 2
        return 1

    def _compare_predictions(self, pred: dict, gold: dict) -> tuple[bool, dict[str, bool]]:
        # relevant_articles: compare as sorted list of ints
        pred_articles = sorted([int(x) for x in pred.get('relevant_articles', [])])
        gold_articles = sorted([int(x) for x in gold.get('relevant_articles', [])])
        ok_articles = pred_articles == gold_articles

        # accusation: compare normalized set of strings
        pred_acc = set(self._normalize_accusations(pred.get('accusation', [])))
        gold_acc = set(self._normalize_accusations(gold.get('accusation', [])))
        ok_accusation = pred_acc == gold_acc

        pred_term = pred.get('term_of_imprisonment', {})
        gold_term_bucket = None
        try:
            if 'term' in gold:
                gold_term_bucket = int(gold.get('term'))
            else:
                gt = gold.get('term_of_imprisonment', {})
                if bool(gt.get('death_penalty', False)) or bool(gt.get('life_imprisonment', False)):
                    gold_term_bucket = 0
                else:
                    try:
                        gm = int(gt.get('imprisonment', 0))
                    except Exception:
                        gm = 0
                    gold_term_bucket = self._months_to_bucket(gm)
        except Exception:
            gold_term_bucket = None
        try:
            if 'term' in pred:
                pred_term_bucket = int(pred.get('term'))
            else:
                if bool(pred_term.get('death_penalty', False)) or bool(pred_term.get('life_imprisonment', False)):
                    pred_term_bucket = 0
                else:
                    try:
                        pm = int(pred_term.get('imprisonment', 0))
                    except Exception:
                        pm = 0
                    pred_term_bucket = self._months_to_bucket(pm)
        except Exception:
            pred_term_bucket = None
        ok_term = (pred_term_bucket is not None and gold_term_bucket is not None and pred_term_bucket == gold_term_bucket)

        all_ok = ok_articles and ok_accusation and ok_term
        details = {
            'articles': ok_articles,
            'accusation': ok_accusation,
            'term': ok_term
        }
        return all_ok, details

    @staticmethod
    def _validate_structure(pred: dict) -> tuple[bool, str | None]:
        """
        Validate CAIL JSON structure and enforce single-label articles.
        Returns (ok, message). If not ok, message describes the issue.
        """
        if not isinstance(pred, dict):
            return False, '输出内容必须为 JSON 对象（字典）'
        # relevant_articles: list of ints, length == 1
        arts = pred.get('relevant_articles')
        if not isinstance(arts, list):
            return False, 'relevant_articles 必须为整数数组，如 [351]'
        try:
            arts_int = [int(x) for x in arts]
        except Exception:
            return False, 'relevant_articles 中的元素必须为整数'
        if len(arts_int) != 1:
            return False, '本任务为单标签，请仅输出一个法条编号，例如 [351]'

        # accusation: list of strings
        acc = pred.get('accusation')
        if not isinstance(acc, list) or not all(isinstance(x, str) for x in acc):
            return False, 'accusation 必须为中文罪名字符串数组，例如 ["诈骗"]'

        # term_of_imprisonment: object with three fields
        term = pred.get('term_of_imprisonment')
        if not isinstance(term, dict):
            return False, 'term_of_imprisonment 必须为对象且包含三个字段'
        for k in ('death_penalty', 'life_imprisonment', 'imprisonment'):
            if k not in term:
                return False, 'term_of_imprisonment 必须包含 death_penalty、life_imprisonment、imprisonment'
        if not isinstance(term.get('death_penalty'), (bool, int)) or not isinstance(term.get('life_imprisonment'), (bool, int)):
            return False, 'death_penalty 与 life_imprisonment 必须为布尔值'
        try:
            int(term.get('imprisonment'))
        except Exception:
            return False, 'imprisonment 必须为整数（单位：月）'
        return True, None

    def step(self, action: str) -> tuple[str, float, bool]:
        action: str = self.process_action(action)

        if self._parse_action_type(action) == 'thought':
            return 'OK.', 0, False

        action_type, argument = self._parse_action(action)

        if action_type == 'Finish':
            errors: list[str] = []
            pred = None

            # Attempt 1: direct JSON
            try:
                pred = json.loads(argument)
            except Exception as e:
                errors.append(f'direct: {e}')

            # Attempt 2: extract JSON substring and clean
            if pred is None:
                candidate = self._extract_json_substring(argument)
                if candidate:
                    cleaned = self._clean_json_like_string(candidate)
                    try:
                        pred = json.loads(cleaned)
                    except Exception as e:
                        errors.append(f'cleaned: {e}')

            # Attempt 3: lenient Python literal eval as fallback
            if pred is None:
                try:
                    lit = ast.literal_eval(self._clean_json_like_string(argument))
                    # Only accept dicts
                    if isinstance(lit, dict):
                        pred = lit
                except Exception as e:
                    errors.append(f'literal_eval: {e}')

            if pred is None:
                # Compose informative error message with head/tail snippet and bracket balance
                arg_text = (argument or '').strip().replace('\n', ' ')
                if len(arg_text) <= 400:
                    snippet = arg_text
                else:
                    snippet = arg_text[:200] + ' ... ' + arg_text[-200:]
                balance_info = (
                    f"braces={{open:{arg_text.count('{')}, close:{arg_text.count('}')}}}, "
                    f"brackets=[open:{arg_text.count('[')}, close:{arg_text.count(']')}]"
                )
                detail = '; '.join(errors)
                observation = (
                    f'Output is not valid JSON. Parsing attempts failed: {detail}.\n'
                    f'Got: {snippet}\n'
                    f'Balance: {balance_info}\n'
                    f"Expected Finish[{{\"relevant_articles\":[int,...],\"accusation\":[str,...],\"term_of_imprisonment\":{{\"death_penalty\":bool,\"life_imprisonment\":bool,\"imprisonment\":int}}}}]"
                )
                return observation, -1, False

            # Preprocess: coerce accusation to array if model outputs a single string
            if isinstance(pred.get('accusation'), str):
                pred['accusation'] = [pred['accusation']]

            # Structure validation (enforce single-label and types)
            ok_struct, msg = self._validate_structure(pred)
            if not ok_struct:
                observation = (
                    f"结构不合规：{msg}\n"
                    f"请严格输出 Finish[{{\"relevant_articles\":[单个整数],\"accusation\":[中文罪名],\"term_of_imprisonment\":{{\"death_penalty\":bool,\"life_imprisonment\":bool,\"imprisonment\":int}}}}]"
                )
                return observation, -1, False

            self.last_pred = pred
            all_ok, details = self._compare_predictions(pred, self.config.get('expected'))
            # Partial credit: each subtask contributes 1/3 to total reward
            correct_count = int(bool(details.get('articles'))) + int(bool(details.get('accusation'))) + int(bool(details.get('term')))
            score = correct_count / 3.0
            obs_detail = (
                f"Articles: {'CORRECT' if details.get('articles') else 'INCORRECT'}; "
                f"Accusation: {'CORRECT' if details.get('accusation') else 'INCORRECT'}; "
                f"Term: {'CORRECT' if details.get('term') else 'INCORRECT'}"
            )
            summary = 'CORRECT' if score == 1.0 else ('PARTIAL' if score > 0 else 'INCORRECT')
            observation = f"{obs_detail}\nAnswer is {summary} (score={score:.2f})"
            self.reward = score
            return observation, self.reward, True
        else:
            observation = 'Invalid Action. Valid Actions are Finish[<JSON>] or Thought.'
            return observation, -1, False

    def feedback(self) -> tuple[float, bool, str]:
        done = False
        try:
            if self.last_pred is not None and self.config.get('expected') is not None:
                _, details = self._compare_predictions(self.last_pred, self.config.get('expected'))
                done = bool(details.get('articles')) and bool(details.get('accusation'))
        except Exception:
            done = False
        if done:
            feedback = 'You successfully finished this task.'
        elif self.reward > 0:
            feedback = 'You partially finished this task.'
        else:
            feedback = 'You failed the task.'
        return self.reward, done, feedback


class CAILRecorder(BaseRecorder):
    def __post_init__(self):
        super().__post_init__()
        self.task = 'cail2018'
        self.counts = 0
        self.dones = 0
        self.rewards = 0
        self._eval_base_dir = os.environ.get(
            'EVAL_OUTPUT_DIR', os.path.join(self.working_dir, 'eval')
        )
        os.makedirs(self._eval_base_dir, exist_ok=True)
        self._eval_run_dir = None
        self._eval_file = None
        self._eval_lock = None

    def task_begin(self, task_id: int, task_config: dict):
        super().task_begin(task_id, task_config)
        # Clear and explicit task boundary with core input
        self.log(f"---------- Task: {task_id} ----------")
        fact = task_config.get('task')
        if fact:
            self.log(f"Input Fact: {fact}")

    def task_end(self, reward: float, done: bool):
        self.rewards += reward
        self.dones += done
        self.counts += 1
        message = (
            f"reward: {reward}, ave reward: {self.rewards / max(self.counts, 1):.4f}.\n"
            f"done: {done}, ave done: {self.dones / max(self.counts, 1):.4f}"
        )
        self.log(message)

    def dataset_begin(self) -> None:
        super().dataset_begin()
        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(self.start_time)) if hasattr(self, 'start_time') else time.strftime("%Y%m%d_%H%M%S")
        self._eval_run_dir = os.path.join(self._eval_base_dir, ts)
        os.makedirs(self._eval_run_dir, exist_ok=True)
        self._eval_file = os.path.join(self._eval_run_dir, f"{self.task}_predictions.jsonl")
        self._eval_lock = os.path.join(self._eval_run_dir, f"{self.task}_predictions.jsonl.lock")

    def dataset_end(self) -> None:
        super().dataset_end()
        try:
            metrics = self._compute_metrics()
            out_path = os.path.join(self._eval_run_dir or self._eval_base_dir, f"{self.task}_metrics.json")
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(json.dumps(metrics, ensure_ascii=False, indent=2))
            self.log(f"Metrics saved: {out_path}")
        except Exception as e:
            try:
                self.log(f"Metrics compute failed: {e}")
            except Exception:
                pass

    def _compute_metrics(self) -> dict:
        y_true_articles: list[int] = []
        y_pred_articles: list[int] = []
        y_true_acc: list[str] = []
        y_pred_acc: list[str] = []
        y_true_term: list[int] = []
        y_pred_term: list[int] = []
        try:
            if self._eval_file and os.path.exists(self._eval_file):
                with open(self._eval_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            row = json.loads(line)
                        except Exception:
                            continue
                        gold = (row.get('expected') or {})
                        pred = (row.get('pred') or {})
                        # 统一 articles 取首元素进行单标签评估
                        try:
                            ga = gold.get('relevant_articles') or []
                            pa = pred.get('relevant_articles') or []
                            if isinstance(ga, list) and len(ga) >= 1 and isinstance(int(ga[0]), int):
                                y_true_articles.append(int(ga[0]))
                            else:
                                continue
                            if isinstance(pa, list) and len(pa) >= 1:
                                y_pred_articles.append(int(pa[0]))
                            else:
                                y_pred_articles.append(None)
                        except Exception:
                            y_pred_articles.append(None)
                        # accusation 统一使用 normalize_answer（支持去括号与中英文标点归一）
                        try:
                            ga2 = gold.get('accusation') or []
                            pa2 = pred.get('accusation') or []
                            if isinstance(ga2, list) and len(ga2) >= 1 and isinstance(ga2[0], str):
                                y_true_acc.append(normalize_answer(ga2[0]))
                            else:
                                continue
                            if isinstance(pa2, list) and len(pa2) >= 1 and isinstance(pa2[0], str):
                                y_pred_acc.append(normalize_answer(pa2[0]))
                            else:
                                y_pred_acc.append(None)
                        except Exception:
                            y_pred_acc.append(None)

                        # term: 使用分档 bucket 比较（gold 可为 term 或 term_of_imprisonment）
                        try:
                            # gold term bucket
                            if 'term' in gold:
                                y_true_term.append(int(gold.get('term')))
                            else:
                                gt = gold.get('term_of_imprisonment') or {}
                                if bool(gt.get('death_penalty', False)) or bool(gt.get('life_imprisonment', False)):
                                    y_true_term.append(0)
                                else:
                                    gm = int(gt.get('imprisonment', 0))
                                    y_true_term.append(self._months_to_bucket(gm))
                            # pred term bucket
                            if 'term' in pred:
                                y_pred_term.append(int(pred.get('term')))
                            else:
                                pt = pred.get('term_of_imprisonment') or {}
                                if bool(pt.get('death_penalty', False)) or bool(pt.get('life_imprisonment', False)):
                                    y_pred_term.append(0)
                                else:
                                    pm = int(pt.get('imprisonment', 0))
                                    y_pred_term.append(self._months_to_bucket(pm))
                        except Exception:
                            y_pred_term.append(None)
        except Exception:
            pass

        art_metrics = self._metrics_for_labels(y_true_articles, y_pred_articles)
        acc_metrics = self._metrics_for_labels(y_true_acc, y_pred_acc)
        term_metrics = self._metrics_for_labels(y_true_term, y_pred_term)
        return {
            'samples': len(y_true_articles),
            'relevant_articles': art_metrics,
            'accusation': acc_metrics,
            'term': term_metrics
        }

    @staticmethod
    def _metrics_for_labels(y_true: list, y_pred: list) -> dict:
        n = len(y_true)
        correct = 0
        labels = set(x for x in y_true if x is not None)
        for i in range(n):
            if i < len(y_pred) and y_pred[i] is not None and y_pred[i] == y_true[i]:
                correct += 1
        accuracy = (correct / n) if n > 0 else 0.0
        precisions = []
        recalls = []
        f1s = []
        for c in labels:
            tp = sum(1 for i in range(n) if y_true[i] == c and i < len(y_pred) and y_pred[i] == c)
            fp = sum(1 for i in range(n) if y_true[i] != c and i < len(y_pred) and y_pred[i] == c)
            fn = sum(1 for i in range(n) if y_true[i] == c and (i >= len(y_pred) or y_pred[i] != c))
            p = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            r = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
            precisions.append(p)
            recalls.append(r)
            f1s.append(f1)
        macro_precision = (sum(precisions) / len(precisions)) if precisions else 0.0
        macro_recall = (sum(recalls) / len(recalls)) if recalls else 0.0
        macro_f1 = (sum(f1s) / len(f1s)) if f1s else 0.0
        return {
            'accuracy': accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1
        }

    def save_eval(self, env: CAILEnv):
        try:
            pred = getattr(env, 'last_pred', None)
        except Exception:
            pred = None
        fact = None
        expected = None
        try:
            fact = self.current_task_config.get('task') if self.current_task_config else None
        except Exception:
            fact = None
        try:
            expected = self.current_task_config.get('expected') if self.current_task_config else None
        except Exception:
            expected = None
        entry = {
            'task_id': self.current_task_id,
            'fact': fact,
            'pred': pred,
            'expected': expected
        }
        try:
            with open(self._eval_lock or os.path.join(self._eval_base_dir, f"{self.task}_predictions.jsonl.lock"), 'a+') as lock_fd:
                fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
                target_file = self._eval_file or os.path.join(self._eval_base_dir, f"{self.task}_predictions.jsonl")
                os.makedirs(os.path.dirname(target_file), exist_ok=True)
                with open(target_file, 'a+', encoding='utf-8') as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
        except Exception:
            try:
                target_file = self._eval_file or os.path.join(self._eval_base_dir, f"{self.task}_predictions.jsonl")
                with open(target_file, 'a+', encoding='utf-8') as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            except Exception:
                pass
