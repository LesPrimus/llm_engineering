"""The structured reply a model gives to a GAIA question."""

from pydantic import BaseModel, ConfigDict


# The docstring below is sent to the model as the schema's description, so notes
# for us live here instead.
#
# The field names are the keys ``prompts.SYSTEM_PROMPT`` asks for: rename them in
# both places or not at all. Both text fields are nullable rather than defaulted,
# because strict structured outputs require every key to be present.
class GaiaReply(BaseModel):
    """Either an answer, or the reason there isn't one."""

    model_config = ConfigDict(frozen=True)

    is_solvable: bool
    final_answer: str | None
    unsolvable_reason: str | None
