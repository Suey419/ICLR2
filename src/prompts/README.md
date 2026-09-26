# Prompt Isolation

This directory stores all prompt templates used by the LLM. Application code must
not embed prompt strings; providers should load templates from this directory and
fill in their variables.

The templates use different context boundaries: the diagnosis module sees only
visible history, the planning modules see only the visible state, and the
execution-simulation module sees only de-identified hidden context. Caches for
these calls must also remain independent.
