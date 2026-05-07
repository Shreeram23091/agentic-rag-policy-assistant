"""
app/agent/agent.py — Core AI Agent

Architecture:
  1. Receives a user query + session history
  2. Asks the LLM (with tool definitions) whether to answer directly or call the
     `search_documents` tool (RAG retrieval).
  3. If tool is called → retrieves relevant chunks → constructs augmented prompt
  4. Generates the final structured answer
  5. Returns answer + list of source documents cited

Tool calling uses OpenAI's function-calling API; works identically on Azure OpenAI.
"""
from __future__ import annotations

import json
from typing import Any

import httpx
import structlog

from app.config import get_settings
from app.llm_client import get_chat_client
from app.memory.session_memory import MessageDict

logger = structlog.get_logger(__name__)

# ── Tool definitions (OpenAI function-calling schema) ────────────────────────

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Search the internal company knowledge base (policy documents, FAQs, "
                "technical guides) to find relevant information. Use this tool whenever "
                "the user's question is about company policies, products, procedures, or "
                "any internal knowledge. Do NOT use it for general world knowledge questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The refined search query to look up in the document store.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "Search the internet for real-time information, news, or general world knowledge "
                "that is not likely to be in the internal knowledge base. Use this for questions "
                "about current events, public figures, or technical details not covered in internal docs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up on the web.",
                    }
                },
                "required": ["query"],
            },
        },
    }
]

# ── System prompts ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are STMicroelectronics Assistant, a professional and helpful technical expert from STMicroelectronics.

Your goal is to provide clear, direct, and natural answers to questions about STM32 products, manufacturing, and company technical guidelines.

## Conversational Guidelines
- Answer naturally, as a human expert would. 
- **CRITICAL**: Do NOT say things like "I searched the documents" or "According to the stm32_overview.txt file". Just state the facts.
- Use your internal technical knowledge and the provided context to answer.
- If you don't know the answer, politely explain that you don't have that specific information yet.
- Be concise but thorough. Use formatting (like bolding) to make key points stand out.
"""

AUGMENTED_PROMPT_TEMPLATE = """Based on the following retrieved document excerpts, answer the user's question.

## Retrieved Context
{context}

## Instructions
- Answer using the context above as your primary source.
- If the context doesn't fully answer the question, say what you found and what's missing.
- Cite the specific document name(s) in your answer.
- Do not fabricate information not present in the context.
"""


class AgentResponse:
    """Structured response from the agent."""
    def __init__(self, answer: str, sources: list[str], used_rag: bool, raw_chunks: list[dict] | None = None):
        self.answer = answer
        self.sources = sources
        self.used_rag = used_rag
        self.raw_chunks = raw_chunks or []


class Agent:
    def __init__(self, retrieve_fn) -> None:
        self._retrieve = retrieve_fn
        self._client = get_chat_client()
        self._settings = get_settings()

    async def run(
        self,
        user_query: str,
        history: list[MessageDict] | None = None,
    ) -> AgentResponse:
        history = history or []
        
        logger.info("Agent invoked", query=user_query[:80])

        used_rag = False
        retrieved_chunks: list[dict] = []
        web_results: list[str] = []
        answer = ""

        try:
            # ── Step 1: Pre-retrieve local documents (Always first) ──────────
            retrieved_chunks = await self._retrieve(user_query)
            used_rag = len(retrieved_chunks) > 0
            context = self._format_context(retrieved_chunks)

            # ── Step 2: Single LLM call with context + web tool as backup ──
            messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
            messages.extend(history)
            
            prompt = (
                f"User Question: {user_query}\n\n"
                f"INTERNAL CONTEXT FROM STMICRO DOCUMENTS:\n{context}\n\n"
                "INSTRUCTION: Use the internal context above to answer the question naturally. "
                "If the context does not contain the answer, use the 'search_web' tool to find the information on the internet."
            )
            messages.append({"role": "user", "content": prompt})

            response = await self._client.chat.completions.create(
                model=self._settings.chat_model,
                messages=messages,
                tools=TOOLS, # Include search_web tool
                tool_choice="auto",
                temperature=0.2,
            )
            choice = response.choices[0]

            # ── Step 3: Handle Web Search Fallback (only if LLM chooses) ─────
            if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
                tool_call = choice.message.tool_calls[0]
                if tool_call.function.name == "search_web":
                    args = json.loads(tool_call.function.arguments)
                    web_query = args.get("query", user_query)
                    
                    logger.info("Local docs insufficient, calling search_web", query=web_query)
                    web_results = await self._search_web(web_query)
                    
                    # Final generation with web results
                    web_context = "\n".join([f"- {r}" for r in web_results])
                    messages.append(choice.message)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": "search_web",
                        "content": f"Web search results:\n{web_context}"
                    })
                    
                    final_response = await self._client.chat.completions.create(
                        model=self._settings.chat_model,
                        messages=messages,
                        temperature=0.1,
                    )
                    answer = final_response.choices[0].message.content or ""
                else:
                    # Should not happen as we only gave search_web, but for safety:
                    answer = choice.message.content or ""
            else:
                answer = choice.message.content or ""

        except Exception as e:
            logger.exception("Agent execution failed")
            answer = "I'm sorry, I'm having a little trouble summarizing that information right now. Could you please try asking in a slightly different way?"

        except Exception as e:
            logger.exception("Agent execution failed")
            answer = "I'm sorry, I'm having a little trouble summarizing that information right now. Could you please try asking in a slightly different way?"

        sources = list({chunk["source"] for chunk in retrieved_chunks})
        if web_results:
            sources.append("Web Search")

        return AgentResponse(
            answer=answer,
            sources=sources,
            used_rag=used_rag,
            raw_chunks=retrieved_chunks,
        )

    async def _search_web(self, query: str) -> list[str]:
        """Call Tavily API to get search results."""
        if not self._settings.tavily_api_key:
            logger.warning("Tavily API key not set, skipping web search")
            return ["Web search skipped: No API key provided."]

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": self._settings.tavily_api_key,
                        "query": query,
                        "search_depth": "basic",
                        "max_results": 3,
                    },
                    timeout=10.0
                )
                response.raise_for_status()
                data = response.json()
                results = [f"{r['title']}: {r['content']} (Link: {r['url']})" for r in data.get("results", [])]
                return results
        except Exception as e:
            logger.error("Web search failed", error=str(e))
            return [f"Web search failed: {str(e)}"]

    def _format_context(self, chunks: list[dict]) -> str:
        if not chunks: return "No relevant documents found."
        parts = [f"[{i+1}] Source: {c['source']}\n{c['text']}" for i, c in enumerate(chunks)]
        return "\n\n---\n\n".join(parts)