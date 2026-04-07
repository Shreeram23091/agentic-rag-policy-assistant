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
                    },
                    "reason": {
                        "type": "string",
                        "description": "Brief explanation of why you're searching the documents.",
                    },
                },
                "required": ["query"],
            },
        },
    }
]

# ── System prompts ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are ACME Assistant, an intelligent internal support agent for ACME Corporation.

Your role is to help employees and customers by answering their questions accurately and concisely.

## Decision Framework
- For questions about ACME's **internal policies, products, procedures, FAQs, or technical documentation**:
  → ALWAYS use the `search_documents` tool to look up current information first.
- For **general knowledge questions** (e.g., definitions, calculations, general advice) not specific to ACME:
  → Answer directly from your training knowledge.
- When **uncertain** whether the information exists in documents:
  → Use the tool — it's better to search and find nothing than to hallucinate.

## Response Guidelines
- Be **concise and direct** — answer the question, then offer context.
- **Cite your sources** when answering from documents (e.g., "According to the Leave Policy…").
- If the documents don't contain the answer, say so clearly and suggest who to contact.
- Keep a professional but friendly tone.
- If the question is ambiguous, ask for clarification before using the tool.

## Formatting
- Use bullet points or numbered lists for multi-part answers.
- Highlight key numbers, dates, and thresholds (e.g., **18 days**, **90 days**).
- Keep responses under 400 words unless detail is explicitly requested.
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


# ... (Keep all your imports, TOOLS, and PROMPTS exactly as they are) ...

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
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_query})

        logger.info("Agent invoked", query=user_query[:80], history_len=len(history))

        used_rag = False
        retrieved_chunks: list[dict] = []
        answer = ""

        try:
            # ── Step 1: Attempt First LLM call (Tool Use) ────────────────────
            response = await self._client.chat.completions.create(
                model=self._settings.chat_model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.2,
                max_tokens=1024,
            )
            choice = response.choices[0]

            # ── Step 2: Handle Tool Call ─────────────────────────────────────
            if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
                tool_call = choice.message.tool_calls[0]
                function_args = json.loads(tool_call.function.arguments)
                search_query = function_args.get("query", user_query)

                retrieved_chunks = await self._retrieve(search_query)
                used_rag = True

                context_str = self._format_context(retrieved_chunks)
                tool_result_content = AUGMENTED_PROMPT_TEMPLATE.format(context=context_str)

                # ── Step 3: Attempt Final Answer call ────────────────────────
                messages_with_tool = messages + [
                    choice.message,
                    {"role": "tool", "tool_call_id": tool_call.id, "content": tool_result_content},
                ]

                final_response = await self._client.chat.completions.create(
                    model=self._settings.chat_model,
                    messages=messages_with_tool,
                    temperature=0.2,
                    max_tokens=1024,
                )
                answer = final_response.choices[0].message.content or ""
            else:
                answer = choice.message.content or ""

        except Exception as e:
            # ── THE DEADLINE SURVIVAL FALLBACK ──
            logger.warning("Azure Chat API failed. Switching to Mock Response.", error=str(e))
            
            # Fallback: Force a RAG retrieval manually if the LLM couldn't decide
            if not retrieved_chunks:
                retrieved_chunks = await self._retrieve(user_query)
                used_rag = True
            
            if retrieved_chunks:
                top_context = retrieved_chunks[0]['text'][:300]
                source_file = retrieved_chunks[0]['source']
                answer = (
                    f"**[MOCK RESPONSE - Azure Policy Restricted]**\n\n"
                    f"I found relevant information in **{source_file}**: \n\n"
                    f"> \"...{top_context}...\"\n\n"
                    "The RAG pipeline successfully retrieved this context, but the final LLM "
                    "generation is currently simulated due to Azure deployment restrictions."
                )
            else:
                answer = "I'm sorry, I couldn't access the AI model or find relevant documents to answer your question."

        sources = list({chunk["source"] for chunk in retrieved_chunks})
        return AgentResponse(answer=answer, sources=sources, used_rag=used_rag, raw_chunks=retrieved_chunks)

    def _format_context(self, chunks: list[dict]) -> str:
        if not chunks: return "No relevant documents found."
        parts = [f"[{i+1}] Source: {c['source']}\n{c['text']}" for i, c in enumerate(chunks)]
        return "\n\n---\n\n".join(parts)