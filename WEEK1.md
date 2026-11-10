# Week 1 — Python + ML Foundations

**Goal:** Build your first working ML model from scratch and understand every line.

**Time budget:** ~20 hours (5 hrs/day × 4 days)

---

## Day 1 — Python Basics + Environment

**Tasks:**
- [ ] Install Python + VS Code (or use Google Colab)
- [ ] Create a folder `projects/iris/`
- [ ] Write a Python script that:
  - Loads the iris dataset from sklearn
  - Prints the first 5 rows
  - Prints the number of samples and features

**Acceptance criteria:**
- Script runs without error
- Output shows 150 samples, 4 features

---

## Day 2 — Train/Test Split + ModelWrapper

**Tasks:**
- [ ] Split data: 80% train, 20% test (use `train_test_split`)
- [ ] Create `ModelWrapper` class with:
  - `__init__()`: sets up `Pipeline([StandardScaler, LogisticRegression])`
  - `train(X, y)`: calls `.fit()`
  - `predict(X)`: calls `.predict()`
  - `evaluate(X, y)`: returns accuracy + F1 score
- [ ] Run training on training data
- [ ] Run evaluation on test data

**Acceptance criteria:**
- Class has all 4 methods
- Prints accuracy and F1 score
- Accuracy is between 0.8 and 1.0

---

## Day 3 — Understand the Metrics

**Tasks:**
- [ ] Run your evaluator on iris
- [ ] Write answers in `LEARNING_LOG.md`:
  1. What does accuracy mean here?
  2. What would happen if you tested on training data instead of test data?
  3. Why do we need StandardScaler before LogisticRegression?

**Acceptance criteria:**
- 3 written answers in learning log
- Can explain in your own words (not copied)

---

## Day 4 — Git + Commit + Push

**Tasks:**
- [ ] Initialize git repo in `projects/iris/` (or root)
- [ ] Create `.gitignore` with `__pycache__/`, `.venv/`, `.ipynb_checkpoints`
- [ ] Commit `model_wrapper.py` and `test_iris.py`
- [ ] Push to GitHub (create repo `ai-mentor` if not exists)

**Acceptance criteria:**
- GitHub repo exists
- Contains your Week 1 code
- At least 2 commits

---

## Week 1 Gate (Do Not Skip)

Before moving to Week 2, you must:
- [ ] `ModelWrapper` class exists with train/predict/evaluate
- [ ] Runs on iris dataset, prints accuracy + F1
- [ ] Code committed and pushed to GitHub
- `LEARNING_LOG.md` has Week 1 entry with 3 takeaways

**If any box is unchecked → repeat Week 1. Do not proceed to Week 2.**

---

## Questions for Your Mentor

When you meet your mentor (AI agent), ask:
1. "Is my Week 1 gate correct?"
2. "Can you review my `ModelWrapper` code?"
3. "What should my Week 2 focus be?"
