def render_alpaca_template(instruction: str, response: str = "") -> str:
    return _ALPACA_TEMPLATE.format(instruction=instruction, response=response)


_ALPACA_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{response}"""
