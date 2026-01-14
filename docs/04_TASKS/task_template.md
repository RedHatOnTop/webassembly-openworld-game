# Task Template

Use this template when creating development tasks for Aether Engine.

---

## Task Header

| Field          | Value                                |
|----------------|--------------------------------------|
| **Task ID**    | TASK-XXXX                            |
| **Title**      | [Brief descriptive title]            |
| **Status**     | [ ] Not Started / [/] In Progress / [x] Complete |
| **Priority**   | Critical / High / Medium / Low       |
| **Estimate**   | [Hours]                              |

---

## Goal

[One or two sentences describing the concrete objective of this task.]

---

## Relevant Documentation

| Document | Purpose |
|----------|---------|
| [coordinate_systems.md](../01_STANDARDS/coordinate_systems.md) | [Why referenced] |
| [threading_model.md](../02_ARCHITECTURE/threading_model.md) | [Why referenced] |

---

## Implementation Steps

### Phase 1: [Phase Name]

- [ ] Step 1 description
- [ ] Step 2 description
  - [ ] Sub-step if needed

### Phase 2: [Phase Name]

- [ ] Step 3 description
- [ ] Step 4 description

---

## Future Impact Analysis

Before marking complete, verify this task does not block future features:

| Future Feature | Compatibility Check | Status |
|----------------|---------------------|--------|
| Shadow Mapping | Does Device support required bind groups? | [ ] Verified |
| Depth Reconstruction | Is depth texture samplable (TEXTURE_BINDING)? | [ ] Verified |
| Deferred Rendering | Are G-Buffer formats supported? | [ ] Verified |
| Compute Shaders | Are storage buffers/textures enabled? | [ ] Verified |

### Checklist

- [ ] Render pass structure allows future CSM insertion
- [ ] Buffer/texture usage flags include future requirements
- [ ] Device limits accommodate planned feature set
- [ ] Code structure supports multi-threading migration

---

## Verification

### Success Criteria

- [ ] Criterion 1
- [ ] Criterion 2

### Test Commands

```bash
cargo run
cargo test
```

### Expected Result

[Describe what success looks like - window behavior, console output, etc.]

---

## Notes

[Implementation notes, decisions made, blockers encountered]

---

## Changelog

| Date       | Author | Change |
|------------|--------|--------|
| YYYY-MM-DD | [Name] | Created |
