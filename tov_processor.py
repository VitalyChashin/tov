"""
Customer Support Tone-of-Voice Processor

Takes an incoming customer support message, rewrites it according to configurable
tone-of-voice rules, and returns a structured response.

Stack:
- LangChain (orchestration, structured outputs)
- LiteLLM via OpenAI-compatible API (LLM provider)
- Langfuse (tracing/observability)
- Pydantic (output schema)
"""

import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langfuse.langchain import CallbackHandler
from pydantic import BaseModel, Field

load_dotenv()


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------
class SupportReply(BaseModel):
    """Structured reply from the support agent."""

    reply: str = Field(
        ...,
        description="The final customer-facing reply rewritten in the required tone of voice.",
    )
    detected_intent: Literal[
        "complaint", "question", "refund_request", "feedback", "other"
    ] = Field(
        ...,
        description="The classified intent of the incoming customer message.",
    )
    sentiment: Literal["positive", "neutral", "negative"] = Field(
        ...,
        description="Sentiment of the incoming customer message.",
    )
    requires_human_handoff: bool = Field(
        ...,
        description="Whether this case should be escalated to a human agent.",
    )


# ---------------------------------------------------------------------------
# Tone-of-voice rules (configurable)
# ---------------------------------------------------------------------------
DEFAULT_TOV_RULES = """\
1. Address the customer politely on a first-name basis when known; otherwise use neutral polite forms.
2. Keep sentences short and clear — no more than 20 words per sentence.
3. Always acknowledge the customer's feeling/problem in the first sentence.
4. Avoid corporate jargon, passive voice, and filler phrases ("we strive to", "kindly note").
5. Never blame the customer. Use "we" / "our team" for any responsibility statements.
6. End with a concrete next step or a clear question — never with empty pleasantries.
7. No exclamation marks except for a single greeting if appropriate.
8. Match the customer's language (reply in the language the customer wrote in).
"""


SYSTEM_PROMPT = """\
You are a customer support assistant. Your job is to rewrite/answer customer
messages strictly following the brand tone-of-voice rules below.

# Tone-of-voice rules
{tov_rules}

# Task
1. Read the incoming customer message.
2. Classify intent and sentiment.
3. Decide whether the case needs human escalation (e.g. legal threats,
   safety issues, repeated unresolved complaints).
4. Produce a final reply that fully complies with the tone-of-voice rules.

Return the result strictly in the requested structured format.
"""


# ---------------------------------------------------------------------------
# Chain factory
# ---------------------------------------------------------------------------
def build_chain(tov_rules: str = DEFAULT_TOV_RULES):
    """Build a LangChain runnable that produces a SupportReply."""

    # LiteLLM exposes an OpenAI-compatible /v1/chat/completions endpoint,
    # so we use ChatOpenAI and just point it at the LiteLLM proxy.
    llm = ChatOpenAI(
        model=os.getenv("LITELLM_MODEL", "gpt-4o-mini"),
        base_url=os.getenv("LITELLM_BASE_URL", "http://localhost:4000"),
        api_key=os.getenv("LITELLM_API_KEY", "sk-anything"),
        temperature=0.2,
    )

    # Structured outputs — LangChain handles JSON-schema/function-calling under the hood.
    structured_llm = llm.with_structured_output(SupportReply)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "Customer message:\n\n{customer_message}"),
        ]
    ).partial(tov_rules=tov_rules)

    return prompt | structured_llm


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def process_message(
    customer_message: str,
    tov_rules: str = DEFAULT_TOV_RULES,
    session_id: str | None = None,
    user_id: str | None = None,
) -> SupportReply:
    """Process one customer message end-to-end with Langfuse tracing."""

    chain = build_chain(tov_rules)

    # Langfuse callback — auto-traces every LLM call, prompt, latency, tokens, cost.
    langfuse_handler = CallbackHandler()

    config = {
        "callbacks": [langfuse_handler],
        "metadata": {
            "langfuse_session_id": session_id or "default-session",
            "langfuse_user_id": user_id or "anonymous",
            "langfuse_tags": ["customer-support", "tov-processor"],
        },
        "run_name": "tov_support_reply",
    }

    return chain.invoke({"customer_message": customer_message}, config=config)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample = (
        "Hi, I ordered a chair from you 3 weeks ago and it's still not here. "
        "Your support didn't reply to my last 2 emails. I want a refund NOW or "
        "I'm going to leave a bad review everywhere."
    )

    result = process_message(
        customer_message=sample,
        session_id="demo-session-001",
        user_id="customer-42",
    )

    print("=== Structured result ===")
    print(result.model_dump_json(indent=2))
    print("\n=== Reply to customer ===")
    print(result.reply)
