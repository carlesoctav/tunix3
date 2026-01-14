from jinja2 import Template
import difflib


def compare_templates():
    # 1. Define the PROMPT_TEMPLATE from the user
    PROMPT_TEMPLATE = """<start_of_turn>user
Think through your approach in <reasoning></reasoning> tags, then provide your complete response in <answer></answer> tags.
For creative tasks, briefly plan in reasoning, then write the full creative output in answer.


{question}
<end_of_turn>

<start_of_turn>model

"""

    # 2. Load the Jinja template content
    with open("gemma_think_new.jinja", "r") as f:
        jinja_content = f.read()

    # MOCK CUSTOM TAGS: Remove {%- generation -%} and {%- endgeneration -%}
    # These are HF specific and don't affect text output for this comparison
    jinja_content = jinja_content.replace("{%- generation -%}", "").replace(
        "{%- endgeneration -%}", ""
    )

    # 3. Render the Jinja template
    # We need to simulate the environment variables
    t = Template(jinja_content)

    question = "What is the capital of France?"

    # Render with standard single-turn parameters
    # Note: transformers usually passes 'bos_token' and 'messages'
    # We assume bos_token is empty string for fair string comparison, or we strip it.
    jinja_output = t.render(
        bos_token="",
        messages=[{"role": "user", "content": question}],
        add_generation_prompt=True,
        raise_exception=lambda x: f"EXCEPTION: {x}",
    )

    # Render the PROMPT_TEMPLATE
    string_output = PROMPT_TEMPLATE.format(question=question)

    print("--- String Template Output ---")
    print(repr(string_output))
    print("\n--- Jinja Template Output ---")
    print(repr(jinja_output))

    print("\n--- Comparison ---")
    if string_output == jinja_output:
        print("MATCH")
    else:
        print("MISMATCH")

    print("\n--- Detailed Diff ---")
    for line in difflib.unified_diff(
        string_output.splitlines(keepends=True),
        jinja_output.splitlines(keepends=True),
        fromfile="String Template",
        tofile="Jinja Template",
    ):
        print(line, end="")


if __name__ == "__main__":
    compare_templates()
