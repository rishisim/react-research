---
description: How to update project documentation after making changes
---

# Update Project State Workflow

When you make significant changes to this repository, you MUST update the `PROJECT_STATE.md` file.

## What Counts as "Significant Changes"

- Adding, removing, or modifying agent implementations
- Adding or removing datasets
- Changing project structure (moving files/folders)
- Updating dependencies
- Fixing bugs that affect how agents work
- Adding new features or experiments

## Steps to Update

1. Open `PROJECT_STATE.md` in the project root

2. Update the **"Last Updated"** timestamp at the top:
   ```markdown
   > **Last Updated**: YYYY-MM-DD by [Your Name/Agent]
   ```

3. If you changed the project structure, update the **"Project Structure"** section

4. If you added/removed datasets, update the **"Datasets"** table

5. Add an entry to the **"Recent Changes"** section with today's date:
   ```markdown
   ### YYYY-MM-DD
   - Brief description of what you changed
   - Another change
   ```

6. If you discovered any issues, add them to **"Known Issues"**

7. If you fixed a known issue, remove it from the list

## Example Update

If you added a new agent for the MuSiQue dataset:

```markdown
### 2025-12-30
- Added MuSiQue agent implementation in `src/agents/musique/`
- Created prompts for MuSiQue fact verification
- Updated `AGENTS.md` with MuSiQue agent documentation
```

## Note

You do NOT need to update for:
- Minor typo fixes
- Code formatting changes
- Comments-only changes
- Test runs that don't change code
