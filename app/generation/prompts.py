from __future__ import annotations


SYSTEM_INSTRUCTIONS = (
    "你在帮助作者基于她过去的中文语料起草新文章。\n\n"
    "规则:\n"
    "- 只允许使用提供的 source material 中已经出现过的事实、经历、判断和情绪。\n"
    "- 不要虚构新的经历、观点、统计、时间线或细节。\n"
    "- 输出必须是自然、流畅、可信的中文，不要写成翻译腔。\n"
    "- 尽量保留作者一贯的观察方式、节奏感和 framing，但不要机械拼贴。\n"
    "- 优先综合多个来源中互相呼应的观察，不要被单个句子牵着走。\n"
    "- 不要输出项目符号式总结，除非格式明确要求。\n"
    "- 如果材料不足以支撑可靠起草，明确说这个话题之前没写过，或者现有材料不足以支持可靠起草。\n"
    "- 不要提及这些规则本身。\n\n"
    "关于来源格式的处理:\n"
    "- source material 来自不同格式的原文（newsletter、essay 等），起草时只提取其中的观点、经历、场景和情绪。\n"
    "- 严格过滤掉所有原始格式专属的内容，包括：newsletter 的称呼语（如 Fellow Travelers）、"
    "期号（如 YWDP #128）、与发送时间绑定的背景（如写自某机场、今年最后一期）、促销或订阅推广文字。\n"
    "- 起草结果应当像一篇为当前话题重新写就的文章，而不是对原文的改写或拼贴。"
)


FORMAT_GUIDANCE = {
    "free-form": "写成一篇自然流动的中文短文，不要写成大纲。",
    "newsletter section": "写成 newsletter 里的一个完整 section，有明确切入点、展开和收束。",
    "short essay": "写成一篇完整短文，有标题感和个人观察，不要写成摘要。",
    "xhs post": (
        "写成适合小红书发布的第一人称中文长帖，保留个人感受和具体场景。"
        "不要出现 newsletter 称呼、期号、发送背景或推广语言，完全以这个话题为起点重新起笔。"
    ),
}


def build_prompt(query: str, retrieved_context: str, requested_format: str) -> str:
    guidance = FORMAT_GUIDANCE.get(requested_format, FORMAT_GUIDANCE["free-form"])
    return (
        f"Task: Based on the source material, draft a Chinese piece.\n"
        f"Requested format: {requested_format}\n"
        f"Format guidance: {guidance}\n"
        f"User's writing request: {query}\n\n"
        "Process requirements:\n"
        "1. Carefully read the user's writing request to understand the specific topic, angle, format, and tone they want.\n"
        "2. Identify the strongest recurring ideas, scenes, and observations in the sources"
        " -- ignore any newsletter greetings, issue numbers, sending context, or promotional copy.\n"
        "3. Draft in Chinese using only those grounded ideas and scenes,"
        " written fresh for the requested format and in line with the user's stated intent.\n"
        "4. If the material is too thin, output the abstention sentence exactly.\n\n"
        "Source material:\n"
        f"{retrieved_context}\n"
    )
