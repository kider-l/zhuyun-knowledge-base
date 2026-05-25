import httpx

from app.config import get_settings
from app.schemas import AnswerHistoryTurn, SearchResult


class AnswerService:
    def __init__(self) -> None:
        self.settings = get_settings()

    def answer(self, query: str, results: list[SearchResult], history: list[AnswerHistoryTurn] | None = None) -> str:
        if not results:
            return "没有在已入库资料中找到足够相关的内容。请尝试换一个关键词，或确认相关 PDF 已在后台审核入库。"
        usable_results = self._usable_results(results)
        if not usable_results:
            return "当前命中的资料页还没有可读文字，无法给出准确答案。请先在后台完成 OCR 解析和重新入库后再查询。"
        recent_history = [item for item in (history or []) if item.query.strip() and item.answer.strip()][-6:]
        try:
            if self.settings.model_provider == "openai_compatible":
                return self._answer_with_openai_compatible(query, usable_results, recent_history)
            if self.settings.model_provider == "none":
                return self._extractive_answer(query, usable_results, recent_history)
            return self._answer_with_ollama(query, usable_results, recent_history)
        except Exception:
            return self._extractive_answer(query, usable_results, recent_history)

    def _usable_results(self, results: list[SearchResult]) -> list[SearchResult]:
        unusable_markers = ["等待 OCR", "暂未识别", "可能需要 OCR", "OCR 未完成", "OCR 已跳过"]
        usable = [
            item
            for item in results
            if not (item.metadata.get("needs_ocr") and any(marker in item.snippet for marker in unusable_markers))
        ]
        return usable or [item for item in results if not any(marker in item.snippet for marker in unusable_markers)]

    def _history_section(self, history: list[AnswerHistoryTurn]) -> str:
        if not history:
            return "最近对话：无"
        lines = ["最近对话摘要："]
        for index, item in enumerate(history, start=1):
            lines.append(f"{index}. 用户问题：{item.query}")
            lines.append(f"   助手回答：{item.answer}")
        return "\n".join(lines)

    def _build_prompt(self, query: str, results: list[SearchResult], history: list[AnswerHistoryTurn]) -> str:
        contexts: list[str] = []
        for index, item in enumerate(results[:8], start=1):
            contexts.append(
                f"[{index}] 文件：{item.document_name}；页码：第 {item.page_number} 页；类型：{item.kind}\n{item.snippet}"
            )
        image_evidence = any(item.asset_url for item in results[:8])
        return f"""你是 GraphSearch 的工程资料检索问答助手。请只依据下面的检索证据回答当前问题。
要求：
1. 用中文回答。禁止输出任何英文、日文或韩文内容。你的所有输出必须使用简体中文。
2. 不要编造资料中没有的信息。
3. 每一个关键结论后必须使用 [1]、[2] 这种编号引用证据；同一句可引用多个证据，例如 [1][3]。
4. 仔细判断每条检索证据是否与当前问题真正相关。对于完全无关或仅有非常微弱关联的证据，不要引用、不要提及、不要在回答中使用。宁可只引用 1-2 条高度相关的证据并明确说"相关资料有限"，也不要强行凑内容。
5. 如果证据不足，要明确说明不足。
6. 不要输出"参考资料如下"这种单独列表，引用编号直接放在回答正文里。
7. 内容必须结构化分层呈现，禁止使用 ##、**、-、---、|、> 等 Markdown 符号。请严格按照以下格式组织：
   ## 一级标题用"一、""二、""三、" 等中文序号（如"一、设备参数"），单独一行
   ## 二级标题用"1.""2.""3." 序号（如"1.技术指标"），前面无缩进
   ## 三级子条目用"（1）""（2）""（3）" 或"A.""B.""C." 序号
   ## 列表内容用"· " 或自然段落中的逗号分隔
   ## 表格型数据直接用自然分段呈现，每行一个项目，项目名和值之间用中文冒号
   ## 核心结论与辅助说明用空行或"说明："前缀明确区分
   - 用"首先/其次/最后"、"一方面/另一方面"等衔接词组织逻辑关系
   - 不同主题之间用空行隔开，保持清晰阅读节奏
8. 回答要紧凑，段落之间用一个空行分隔，不要多余空行。
9. 历史上下文只用于理解代词、省略和追问，不能覆盖当前检索证据；如果历史说法与本轮证据不一致，以本轮证据为准。
10. 如果用户想看图纸、路线、位置、存放点、平面图、示意图，且证据中有页面截图或图纸证据，必须直接说明"已找到对应图纸/页面"，写明文件名、页码和图名/附件名，并写"下方已直接展示了该页图纸"。不要只写引用编号（如[1]）让用户自行查看，图纸会直接显示在回答下方，请在文字末尾明确指出图纸已展示。
11. 只有在证据中没有任何可展示页面或图纸时，才说明资料不足。当前证据是否包含可展示页面/图纸：{"是" if image_evidence else "否"}。
12. 这是最重要的要求：你的语言必须严格强制为简体中文。如果用户用英文提问，你依然必须用中文回答。回答中不得出现任何非中文内容。
13. 禁止输出你的思考过程或推理过程。直接给出最终答案。禁止以"思考："或"让我思考"等开头。禁止回答中出现英文词汇"think"、"thinking"、"thought"、"reasoning"。

{self._history_section(history)}

当前问题：{query}

本轮检索证据：
{chr(10).join(contexts)}
"""

    def _answer_with_ollama(self, query: str, results: list[SearchResult], history: list[AnswerHistoryTurn]) -> str:
        response = httpx.post(
            f"{self.settings.ollama_base_url.rstrip('/')}/api/generate",
            json={"model": self.settings.chat_model, "prompt": self._build_prompt(query, results, history), "stream": False},
            timeout=90,
        )
        response.raise_for_status()
        answer = response.json().get("response", "").strip()
        if not answer:
            raise RuntimeError("empty answer")
        return self._clean_answer(answer)

    def _answer_with_openai_compatible(
        self, query: str, results: list[SearchResult], history: list[AnswerHistoryTurn]
    ) -> str:
        if not self.settings.cloud_api_base_url or not self.settings.cloud_api_key:
            raise RuntimeError("cloud chat API is not configured")
        response = httpx.post(
            f"{self.settings.cloud_api_base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {self.settings.cloud_api_key}"},
            json={
                "model": self.settings.cloud_chat_model,
                "messages": [
                    {'role': 'system', 'content': '你是严谨的中文资料检索问答助手。必须用简体中文回答，禁止输出任何英文。严禁输出思考过程。回答必须结构化分层：一、二、三 一级标题；1. 2. 3. 二级标题；（1）（2）（3）子条目；数据用"项目名：值"分段排列；核心结论与辅助说明用"--------"分隔。禁止使用##、**、-等Markdown符号。'},
                    {"role": "user", "content": self._build_prompt(query, results, history)},
                ],
                "temperature": 0.1,
                "max_completion_tokens": 2048,
            },
            timeout=self.settings.cloud_timeout_seconds,
        )
        response.raise_for_status()
        choices = response.json().get("choices") or []
        answer = choices[0].get("message", {}).get("content", "").strip() if choices else ""
        if not answer:
            raise RuntimeError("empty answer")
        return self._format_answer(self._clean_answer(answer))

    @staticmethod
    def _format_answer(answer: str) -> str:
        import re
        # 去除残留的 Markdown 符号
        answer = re.sub(r"^##\s+", "", answer, flags=re.MULTILINE)
        answer = re.sub(r"\*\*(.*?)\*\*", r"\1", answer)
        # 确保分隔线前后有空行
        answer = re.sub(r"(?<!\n)\n-{5,}\n(?!\n)", "\n\n--------\n\n", answer)
        answer = re.sub(r"\n{3,}", "\n\n", answer)
        return answer.strip()

    def _extractive_answer(self, query: str, results: list[SearchResult], history: list[AnswerHistoryTurn]) -> str:
        lines = []
        if history:
            lines.append("已结合最近对话理解当前追问，并优先依据本轮检索结果回答。")
        lines.append(f"根据已入库资料，和「{query}」最相关的内容如下：")
        for index, item in enumerate(results[:5], start=1):
            lines.append(f"[{index}] {item.document_name} 第 {item.page_number} 页：{item.snippet}")
        return "\n\n".join(lines)

    @staticmethod
    def _clean_answer(answer: str) -> str:
        import re
        # 去除思考标记包裹的内容：如 思考：xxx、think: xxx、<think>xxx</think> 等
        patterns = [
            r"思考[：:].*?(?=\n\s*\n|\Z)",
            r"(?:让我来思考|让我想想|好的?，?让我).*?(?=\n\s*\n|\Z)",
            r"<think>.*?</think>",
            r"<reasoning>.*?</reasoning>",
        ]
        for pattern in patterns:
            answer = re.sub(pattern, "", answer, flags=re.DOTALL)
        # 去除包含 "think" 的整行（不区分大小写）
        lines = [line for line in answer.split("\n") if "think" not in line.lower()]
        answer = "\n".join(lines).strip()
        return answer
