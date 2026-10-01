"""
Prompt templates for CoT-Evo initialization module.

This module contains system prompts and templates for multi-thinker CoT generation.
"""

# ============================================================================
# System Prompts for Different Datasets
# ============================================================================

SYSTEM_PROMPTS = {
    "ChemCoTDataset": """You are a helpful assistant for answering chemistry-related questions. Given a chemical query, your task is to solve it thoroughly and explicitly provide the final result. For tasks such as Molecule Editing, Molecule Optimization, or Reaction Prediction, you should output the final molecule in SMILES format using a JSON structure in the last line of your response. Unless the user specifies a different output format, the default JSON format should be used: {"result": "final SMILES or answer here"}""",

    "ChemCoTBench": """You are a helpful assistant for answering chemistry-related questions. Given a chemical query, your task is to solve it thoroughly and explicitly provide the final result. For tasks such as Molecule Editing, Molecule Optimization, or Reaction Prediction, you should output the final molecule in SMILES format using a JSON structure in the last line of your response. Unless the user specifies a different output format, the default JSON format should be used: {"result": "final SMILES or answer here"}""",

    "BioProBench": """You are a helpful assistant for answering biological protocol-related questions. Given a biological query, your task is to solve it thoroughly and explicitly provide the final result. You should output the final answer in the last line of your response. Unless the user specifies a different output format, the default format should be used: [ANSWER_START]final answer here[ANSWER_END]""",

    "SciKnowEval": """You are a helpful assistant for answering science-related questions. Given a scientific query, your task is to solve it thoroughly and explicitly provide the final result. You should output the final answer using a JSON structure in the last line of your response. Unless the user specifies a different output format, the default JSON format should be used: {"result": "final answer here"}""",

    "DiscourseMT": """You are an expert Japanese-to-English translator and a linguist specialising in discourse. Given a Japanese passage and its preceding context, your task is to resolve the context-dependent part of the translation by reasoning explicitly about the discourse, and then state the final result. Ground your reasoning in linguistic evidence: what the Japanese source actually says, what is underdetermined in it (a dropped argument, a polysemous word, or a noun with several possible renderings), what the context sentence establishes, and which English form that forces. Do not decide on the basis of which English wording sounds more natural in isolation. You should output the final answer in the last line of your response. Unless the user specifies a different output format, the default format should be used: [ANSWER_START]final answer here[ANSWER_END]""",

    # Context-blind control. Identical to DiscourseMT except that every
    # reference to a preceding context is dropped, because there is none: an
    # ablation cannot keep instructions that point at the thing it removed, and
    # telling the model to consult an absent context would depress the score
    # below the genuine context-blind prior the control is meant to measure.
    "DiscourseMT_NoContext": """You are an expert Japanese-to-English translator and a linguist specialising in discourse. Given a Japanese sentence, your task is to resolve the translation by reasoning explicitly about the source, and then state the final result. Ground your reasoning in linguistic evidence: what the Japanese actually says, what is underdetermined in it (a dropped argument, a polysemous word, or a noun with several possible renderings), and which English form the source supports. Do not decide on the basis of which English wording sounds more natural in isolation. You should output the final answer in the last line of your response. Unless the user specifies a different output format, the default format should be used: [ANSWER_START]final answer here[ANSWER_END]"""
}

# ============================================================================
# Prompt family resolution
# ============================================================================

# Derived datasets that should reuse another dataset's prompts verbatim.
#
# Use this for a variant that changes the data but not what the model is asked
# to do. A variant that changes the task itself needs its own entry in
# SYSTEM_PROMPTS and its own template instead - DiscourseMT_NoContext is that
# case, since its scaffold cannot tell the model to consult a context sentence
# the ablation removed.
#
# Either way, register every new dataset somewhere. Forgetting is silent: the
# lookups fall back to the generic science prompt, which is wrong for a
# translation task but raises no error. (That is exactly how a run of
# DiscourseMT_NoContext once went out asking for JSON answers to a science
# question; the warning below now catches it.)
PROMPT_FAMILY_ALIASES = {
    # (none at present - DiscourseMT_NoContext has its own family, because its
    # reasoning scaffold cannot refer to a context sentence that was removed.)
}


def resolve_prompt_family(dataset_name: str) -> str:
    """
    Map a dataset name onto the dataset whose prompts it should use.

    Args:
        dataset_name: Name as registered in config/datasets.yaml

    Returns:
        The dataset name to look up in SYSTEM_PROMPTS / STOP_SEQUENCES and to
        branch on when building the user prompt.
    """
    return PROMPT_FAMILY_ALIASES.get(dataset_name, dataset_name)


# ============================================================================
# Knowledge Generation Prompts
# ============================================================================

KNOWLEDGE_GENERATION_PROMPT = """You are a scientific reasoning expert. Your task is to identify and extract the necessary knowledge required to solve a given scientific problem.

[Problem]
{query}

[Correct Answer]
{ground_truth}

### Instructions

Analyze the correct answer and identify the key domain knowledge, concepts, formulas, or facts that are necessary to solve this problem. Extract knowledge that:

1. Is essential for reaching the correct answer
2. Might not be commonly known
3. Can be stated as general, context-independent principles

Format your response as clear, concise knowledge snippets that:
- Are accurate and verifiable
- General enough to be useful for similar problems
- Specific enough to be actionable

Output the knowledge in the following format:

[KNOWLEDGE_START]
Your extracted knowledge here
[KNOWLEDGE_END]
"""

# Discourse-translation variant. The scientific prompt above asks for formulas
# and domain facts, which do not exist for a translation item; here the useful
# knowledge is the linguistic cue that licenses the correct rendering.
DISCOURSE_KNOWLEDGE_GENERATION_PROMPT = """You are an expert in translation and discourse linguistics. Your task is to identify and state the linguistic knowledge required to resolve a context-dependent translation choice.

[Problem]
{query}

[Correct Answer]
{ground_truth}

### Instructions

Using the correct answer, work out *why* it is correct and extract the knowledge needed to reach it. Focus on:

1. The specific cue in the context that determines the answer (an antecedent, a disambiguating collocation, an earlier lexical choice, a topic marker, an elided argument).
2. The property of the Japanese source that makes the sentence ambiguous on its own (e.g. a dropped subject or object, a polysemous word, a repeated noun).
3. The general translation principle this item instantiates, stated so it would apply to other passages as well.

Constraints:
- State the knowledge as claims about language, not as a restatement of the answer.
- Do not describe your own reasoning process; state the knowledge directly.
- Be concrete about which word or phrase in the source and context carries the cue.
- Keep it to a few sentences.

Output the knowledge in the following format:

[KNOWLEDGE_START]
Your extracted knowledge here
[KNOWLEDGE_END]
"""

# ============================================================================
# Knowledge-Augmented Generation Template
# ============================================================================

KNOWLEDGE_AUGMENTED_TEMPLATE = """You are a helpful assistant for solving scientific problems. You will be provided with additional knowledge to help you reason through the problem.

[Problem]
{query}

[Additional Knowledge]
{knowledge}

### Instructions

Use the provided knowledge to solve the problem. Make sure to:
1. Explicitly reference the knowledge when relevant
2. Show your reasoning clearly
3. Provide the final answer in the specified format

Your response:
"""

# ============================================================================
# Vanilla Generation Template (without knowledge)
# ============================================================================

VANILLA_TEMPLATE = """You are a helpful assistant for solving scientific problems.

[Problem]
{query}

### Instructions

Solve the problem step by step, showing your reasoning clearly. Provide the final answer in the specified format.

Your response:
"""

# ============================================================================
# DiscourseMT Generation Templates (Japanese-to-English discourse translation)
# ============================================================================

# The query built by scripts/prepare_discourse_mt.py already carries the context
# sentence, the sentence to translate and the required answer content, so these
# templates only add the <|think|>/<|answer|> scaffolding plus guidance on what
# discourse reasoning should look like.

DISCOURSE_MT_VANILLA_TEMPLATE = """Please resolve the following Japanese-to-English translation problem by reasoning about the discourse step by step.

Your response must follow this exact format:
<|think|>
1. Literal gloss: ...
2. What the context establishes: ...
3. Link: ...
4. Check: ...
<|answer|>
[ANSWER_START]answer here[ANSWER_END]

Problem: {query}

In your reasoning, work through these steps explicitly:
1. Give a literal gloss of the Japanese sentence to translate, quoting the Japanese, and name what in it is underdetermined: a dropped argument (subject, object, possessor), a word with more than one sense, or a noun that could be rendered by several English words. Identify which of these the sentence actually turns on - do not assume it is a dropped pronoun.
2. State what the context sentence establishes: which entities it introduces, how it renders key words, and which reading of the ambiguous material it supports.
3. Link the two: identify the exact word or phrase in the context that determines the context-dependent part of the translation, and say why the alternative reading is ruled out.
4. Check your conclusion against the Japanese source once more, then commit to it.

Base your decision on linguistic evidence from the source and the context, not on which English wording reads more naturally in isolation.

How to write it:
- Your reply must begin with the <|think|> marker and go straight into step 1.
- Write finished reasoning only: no thinking aloud, no false starts, no "hmm" or "wait", no weighing an option you then abandon. Work the problem out first, then write the four steps as a clean explanation.
- Keep the four steps to roughly 150-200 words in total.
- Put only the answer itself between [ANSWER_START] and [ANSWER_END]: no explanation inside the markers, and nothing at all after the closing marker."""

DISCOURSE_MT_KNOWLEDGE_TEMPLATE = """Please resolve the following Japanese-to-English translation problem by reasoning about the discourse step by step. Use the provided knowledge to help you.

Relevant knowledge:
{knowledge}

Your response must follow this exact format:
<|think|>
1. Literal gloss: ...
2. What the context establishes: ...
3. Link: ...
4. Check: ...
<|answer|>
[ANSWER_START]answer here[ANSWER_END]

Problem: {query}

In your reasoning, work through these steps explicitly:
1. Give a literal gloss of the Japanese sentence to translate, quoting the Japanese, and name what in it is underdetermined: a dropped argument (subject, object, possessor), a word with more than one sense, or a noun that could be rendered by several English words. Identify which of these the sentence actually turns on - do not assume it is a dropped pronoun.
2. State what the context sentence establishes, and connect it to the provided knowledge.
3. Apply the knowledge to identify the exact word or phrase in the context that determines the context-dependent part of the translation, and say why the alternative reading is ruled out.
4. Check your conclusion against the Japanese source once more, then commit to it.

Base your decision on linguistic evidence from the source and the context, not on which English wording reads more naturally in isolation.

How to write it:
- Your reply must begin with the <|think|> marker and go straight into step 1.
- Write finished reasoning only: no thinking aloud, no false starts, no "hmm" or "wait", no weighing an option you then abandon. Work the problem out first, then write the four steps as a clean explanation.
- Keep the four steps to roughly 150-200 words in total.
- Put only the answer itself between [ANSWER_START] and [ANSWER_END]: no explanation inside the markers, and nothing at all after the closing marker."""


# Context-blind control templates. Kept word-for-word identical to the two
# above apart from the steps that refer to the context sentence, which has been
# removed from the item. Steps 2-3 collapse into a single "what the source alone
# determines" step, so the model is never told to consult something absent.
# If you edit the templates above, edit these to match.

DISCOURSE_MT_NOCTX_VANILLA_TEMPLATE = """Please resolve the following Japanese-to-English translation problem by reasoning about the source sentence step by step.

Your response must follow this exact format:
<|think|>
1. Literal gloss: ...
2. What the source determines: ...
3. Check: ...
<|answer|>
[ANSWER_START]answer here[ANSWER_END]

Problem: {query}

In your reasoning, work through these steps explicitly:
1. Give a literal gloss of the Japanese sentence, quoting the Japanese, and name what in it is underdetermined: a dropped argument (subject, object, possessor), a word with more than one sense, or a noun that could be rendered by several English words. Identify which of these the sentence actually turns on - do not assume it is a dropped pronoun.
2. State what the sentence alone determines: which reading the wording, collocations and idioms support, and which alternative is ruled out. Where the sentence genuinely underdetermines the choice, say so plainly and give the reading the Japanese most supports.
3. Check your conclusion against the Japanese source once more, then commit to it.

Base your decision on linguistic evidence from the source, not on which English wording reads more naturally in isolation.

How to write it:
- Your reply must begin with the <|think|> marker and go straight into step 1.
- Write finished reasoning only: no thinking aloud, no false starts, no "hmm" or "wait", no weighing an option you then abandon. Work the problem out first, then write the steps as a clean explanation.
- Keep the steps to roughly 120-180 words in total.
- Put only the answer itself between [ANSWER_START] and [ANSWER_END]: no explanation inside the markers, and nothing at all after the closing marker."""

DISCOURSE_MT_NOCTX_KNOWLEDGE_TEMPLATE = """Please resolve the following Japanese-to-English translation problem by reasoning about the source sentence step by step. Use the provided knowledge to help you.

Relevant knowledge:
{knowledge}

Your response must follow this exact format:
<|think|>
1. Literal gloss: ...
2. What the source determines: ...
3. Check: ...
<|answer|>
[ANSWER_START]answer here[ANSWER_END]

Problem: {query}

In your reasoning, work through these steps explicitly:
1. Give a literal gloss of the Japanese sentence, quoting the Japanese, and name what in it is underdetermined: a dropped argument (subject, object, possessor), a word with more than one sense, or a noun that could be rendered by several English words. Identify which of these the sentence actually turns on - do not assume it is a dropped pronoun.
2. Apply the provided knowledge to state what the sentence determines and which alternative is ruled out.
3. Check your conclusion against the Japanese source once more, then commit to it.

Base your decision on linguistic evidence from the source, not on which English wording reads more naturally in isolation.

How to write it:
- Your reply must begin with the <|think|> marker and go straight into step 1.
- Write finished reasoning only: no thinking aloud, no false starts, no "hmm" or "wait", no weighing an option you then abandon. Work the problem out first, then write the steps as a clean explanation.
- Keep the steps to roughly 120-180 words in total.
- Put only the answer itself between [ANSWER_START] and [ANSWER_END]: no explanation inside the markers, and nothing at all after the closing marker."""


# ============================================================================
# No-CoT baseline (the missing cell)
# ============================================================================

# Every template above scaffolds explicit reasoning, so comparing two of them
# measures the contribution of whatever else changed - the context, the model -
# and never the contribution of reasoning itself. Answering "does reasoning
# about discourse help?" needs an arm with no reasoning at all.
#
# Use with the provider's thinking mode OFF. A reasoning endpoint still thinks
# internally when thinking mode is on, so a no-CoT prompt over a thinking model
# is not a no-CoT condition; it just hides the trace.

DISCOURSE_MT_DIRECT_SYSTEM = """You are an expert Japanese-to-English translator. Answer the question directly and immediately. Do not explain, do not justify, do not show any reasoning."""

DISCOURSE_MT_DIRECT_TEMPLATE = """{query}

Answer immediately. Do not explain your choice, do not reason step by step, and do not write anything except the answer.

Your entire response must be exactly:
[ANSWER_START]answer here[ANSWER_END]"""


# ============================================================================
# Stop Sequences for Different Tasks
# ============================================================================

STOP_SEQUENCES = {
    "BioProBench": ["[ANSWER_END]", "\n\n"],
    "ChemCoTDataset": [],
    "ChemCoTBench": [],
    "SciKnowEval": [],
    # No "\n\n" here: these CoTs are multi-paragraph and would be truncated
    # mid-reasoning before the answer marker is ever emitted.
    "DiscourseMT": ["[ANSWER_END]"],
    "DiscourseMT_NoContext": ["[ANSWER_END]"]
}
