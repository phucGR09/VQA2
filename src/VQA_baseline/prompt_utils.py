"""
prompt_utils.py
===============
Canonical prompt assembly for the fine-tuned VQA model (level5 eval and
phase3_vqa_finetune training).

Both callers import from here so the prompt format is guaranteed to stay
in sync between training and evaluation.
"""

from __future__ import annotations

from typing import List, Tuple


SYSTEM_PROMPT = (
    "Bạn là trợ lý VQA. Dựa trên ảnh và các đoạn văn bên trên, "
    "hãy trả lời câu hỏi ngắn gọn và chính xác."
)


def build_prompt(
    retrieved: List[Tuple[str, str]],   # [(article_id, passage_text), ...]
    question: str,
    answer: str,
    max_context_tokens: int,
    tokenizer,
) -> Tuple[str, str]:
    """
    Assemble a full prompt string and its answer-prefix.

    Prompt structure:
        <article_1>passage</article_1>
        ...
        <system>SYSTEM_PROMPT</system>
        <question>question</question>
        <answer>

    Loss is computed only on tokens after <answer> (training) or
    generation starts at <answer> (eval with answer="").

    Returns
    -------
    (full_text, prefix_text)
      full_text   = prefix_text + answer
      prefix_text = everything up to and including <answer>
    """
    context_parts = []
    for i, (_, passage) in enumerate(retrieved, start=1):
        ids = tokenizer.encode(passage, add_special_tokens=False)[:max_context_tokens]
        truncated = tokenizer.decode(ids, skip_special_tokens=True)
        context_parts.append(f"<article_{i}>{truncated}</article_{i}>")

    prefix = (
        "\n".join(context_parts) + "\n"
        f"<system>{SYSTEM_PROMPT}</system>\n"
        f"<question>{question}</question>\n"
        f"<answer>"
    )
    return prefix + answer, prefix
