"""
ChatPromptTemplates for retrieval-augmented generation.

Templates declare `{context}` and `{question}` variables consumed by LCEL
RunnableParallel / RunnablePassthrough.assign blocks upstream.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def build_rag_prompt(*, include_history: bool = False) -> ChatPromptTemplate:
    """
    Construct the system + human messages for grounded Q&A.

    Args:
        include_history: When True, prepends a chat history placeholder for
                         multi-turn extensions without changing the core RAG path.

    Returns:
        A ChatPromptTemplate expecting `context` and `question` input keys.
    """
    system_message = (
        "You are a precise technical assistant. Answer the user's question using ONLY "
        "the provided context. If the context is insufficient, say you do not have "
        "enough information — do not invent facts.\n\n"
        "Context:\n{context}"
    )

    messages: list[tuple[str, str] | MessagesPlaceholder] = [
        ("system", system_message),
    ]

    if include_history:
        messages.append(MessagesPlaceholder(variable_name="chat_history", optional=True))

    messages.append(("human", "{question}"))

    return ChatPromptTemplate.from_messages(messages)
