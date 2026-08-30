import string
import re


def normalize_answer(s: str):

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        # 扩展：移除常见中文标点与中括号等分组符号
        zh_punc = set("，。、；：！？（）【】《》“”‘’—…··、")
        return "".join(ch for ch in text if ch not in exclude and ch not in zh_punc)

    def lower(text):
        return text.lower()

    # Basic normalization: lowercase, remove English articles/punctuations, fix whitespace
    norm = white_space_fix(remove_articles(remove_punc(lower(s))))
    # 额外：移除中括号分组，如 "[走私、贩卖、运输、制造]毒品" -> "走私、贩卖、运输、制造毒品"
    norm = re.sub(r"\[(.*?)\]", r"\1", norm)
    # 额外：统一全角逗号/顿号为逗号分隔再回并，避免标点差异导致不等价
    norm = norm.replace("、", "，")
    # 额外：合并重复逗号与空格
    norm = re.sub(r"[，,]+", "，", norm)
    norm = re.sub(r"\s+", " ", norm).strip()
    # Extra: strip common trailing Chinese legal suffixes to align labels
    # e.g., "非法种植毒品原植物罪" -> "非法种植毒品原植物"
    norm = re.sub(r"罪$", "", norm)
    return norm


def match_exactly(answer, key) -> bool:
    n_answer = normalize_answer(answer)
    n_key = normalize_answer(key)
    return n_answer == n_key