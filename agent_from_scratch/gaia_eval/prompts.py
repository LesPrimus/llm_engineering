"""Prompts for running models on GAIA.

The answer-format rules are GAIA's own, so answers stay scorable by its quasi exact
match. The field names in quotes are the keys of the structured reply.
"""

SYSTEM_PROMPT = """You are a general AI assistant. I will ask you a question.
First, determine if you can solve this problem with your current capabilities and set "is_solvable" accordingly.
If you can solve it, set "is_solvable" to true and provide your answer in "final_answer".
If you cannot solve it, set "is_solvable" to false and explain why in "unsolvable_reason".
Your final answer should be a number OR as few words as possible OR a comma-separated list of numbers and/or strings.
If you are asked for a number, don't use a comma to write your number; also don't use units such as $ or a percent sign unless specified otherwise.
If you are asked for a string, don't use articles, neither abbreviations (e.g., for cities), and write the digits in plain text unless specified otherwise.
If you are asked for a comma-separated list, apply the above rules depending on whether the element is a number or a string."""

# GAIA attaches a file to some questions — a spreadsheet, an image, an audio
# clip. A model with no tools cannot open it, and saying so beats guessing at
# what it held, so the note names the file and asks for the refusal instead.
FILE_NOTE = """

This task comes with an attached file named {file_name}, and you have no way to open it.
If the question cannot be answered without reading that file, set "is_solvable" to false
and say so in "unsolvable_reason"."""
