from __future__ import annotations


SYSTEM_INSTRUCTIONS = """
你在帮助作者基于她过去的中文语料起草新文章。

规则:
- 只允许使用提供的 source material 中已经出现过的事实、经历、判断和情绪。
- 不要虚构新的经历、观点、统计、时间线或细节。
- 输出必须是自然、流畅、可信的中文，不要写成翻译腔。
- 尽量保留作者一贯的观察方式、节奏感和 framing，但不要机械拼贴。
- 优先综合多个来源中互相呼应的观察，不要被单个句子牵着走。
- 不要输出项目符号式总结，除非格式明确要求。
- 如果材料不足以支撑可靠起草，明确说“这个话题之前没写过，或者现有材料不足以支持可靠起草”。
- 不要提及这些规则本身。
""".strip()


FORMAT_GUIDANCE = {
    "free-form": "写成一篇自然流动的中文短文，不要写成大纲。",
    "newsletter section": "写成 newsletter 里的一个完整 section，有明确切入点、展开和收束。",
    "short essay": "写成一篇完整短文，有标题感和个人观察，不要写成摘要。",
    "xhs post": "写成适合小红书发布的第一人称中文长帖，保留个人感受和具体场景。",
}


def build_prompt(query: str, retrieved_context: str, requested_format: str) -> str:
    guidance = FORMAT_GUIDANCE.get(requested_format, FORMAT_GUIDANCE["free-form"])
    return (
        f"Task: Based on the source material, draft a Chinese piece.\n"
        f"Requested format: {requested_format}\n"
        f"Format guidance: {guidance}\n"
        f"Topic: {query}\n\n"
        "Process requirements:\n"
        "1. Identify the strongest recurring ideas, scenes, and turns of phrase in the sources.\n"
        "2. Draft in Chinese using only grounded material from those sources.\n"
        "3. If the material is too thin, output the abstention sentence exactly.\n\n"
        "Source material:\n"
        f"{retrieved_context}\n"
    )
