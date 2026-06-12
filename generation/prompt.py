from typing import List

from schemas import RerankedResult

def build_context(rerank_results: List[RerankedResult], max_chars: int = 6000) -> str:
    blocks = []
    used_chars = 0
    
    for index, item in enumerate(rerank_results, start=1):
        source = item.get("source", "")
        page_start = item.get("page_start", "")
        page_end = item.get("page_end", "")
        heading = item.get("heading", "")
        text = item.get("text", "").strip()
        
        if not text:
            continue

        block = (
            f"[{index}]\n"
            f"Source: {source}\n"
            f"Pages: {page_start}-{page_end}\n"
            f"heading: {heading}\n"
            f"text: {text}\n"
        )
        next_size = len(block) + 2
        if used_chars + next_size > max_chars:
            break

        blocks.append(block)
        used_chars += next_size
        
    return "\n\n".join(blocks)


def build_prompt(context: str, query: str) -> str:
    # 1. 系统/角色说明
    instruction = """
你是一个论文回答助手。
你只能基于给定 Context 回答问题。
如果 Context 中没有足够证据，就明确说“根据当前论文内容无法回答”。
回答时必须使用[1]、[2]这样的引用编号。
不要编造论文中没有出现的信息。
""".strip()
    
    # 2. 用户问题
    question_block = f"""
Question:
{query}
""".strip()
    
    # 3. 检索上下文
    context_block = f"""
Context:
{context}
""".strip()
    
    # 4. 输出要求
    output_format = """
Answer:
请给出简洁、准确的回答，并在关键结论后附上引用编号。
""".strip()
    
    # 5. 拼成最终 prompt
    prompt = "\n\n".join([
        instruction,
        question_block,
        context_block,
        output_format,
    ])

    return prompt