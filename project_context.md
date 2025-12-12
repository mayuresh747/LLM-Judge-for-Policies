# Project Context

## Current Phase
Phase 4: Verify & Refine

## Next Steps
Phase 5: Pre-commit & Submit

## Key Decisions
- **Experiment Logic:** Implemented in `src/utils/experiment.py` and tested with mocks.
- **Dependency Issues:** Downgraded `langchain` and related packages to be compatible with `<1.0.0` or `<0.4` to avoid import errors with `langchain.chains`. Ended up upgrading to `0.3.x` ecosystem which works.
- **Verification:** Unit tests passed.

## File Structure
- `app.py`
- `src/utils/experiment.py`
