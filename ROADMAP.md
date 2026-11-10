# AI Engineer Mastery Roadmap — Beginner Level

**Student:** Maddela Anvesh  
**Goal:** AI Engineer / LLM Engineer Internship (Spring/Summer 2025)  
**Time available:** ~35 hrs/week (5 hrs/day)  
**Timeline:** 12 weeks (~3 months)  
**Starting point:** True beginner — learning from scratch  

---

## Philosophy

> **Evidence over tutorials.**  
> You do not learn by watching videos. You learn by building, breaking, and fixing.  
> Every week ends with a **shipped artifact** you can show in an interview.

---

## What You Will Actually Build (4 Projects)

| Project | What It Does | Skills You Prove |
|---------|-------------|------------------|
| **Iris Classifier** (Week 1) | Predicts flower species from measurements | Python, sklearn, train/test split, eval metrics |
| **DocuChat RAG** (Weeks 2–3) | Answers questions from uploaded PDFs | LangChain, vector DB, hybrid search, eval harness |
| **Nexus Agent** (Weeks 4–6) | Multi-agent system with tools + memory | LangGraph, FastAPI, WebSocket, observability |
| **Signature Project** (Weeks 7–12) | Your own AI idea from idea to deployed demo | Full stack AI: idea → data → model → eval → deploy |

---

## 12-Week Schedule

### Phase 1: Foundations (Weeks 1–3)

| Week | Focus | Deliverable | Gate (Must Pass) |
|------|-------|-------------|------------------|
| 1 | Python + ML basics | Iris classifier with `ModelWrapper` class | `train()`/`predict()`/`evaluate()` work, accuracy printed |
| 2 | RAG basics | Simple PDF QA with LangChain + FAISS | Answers questions from a PDF, shows source |
| 3 | Advanced RAG + Eval | DocuChat v2: hybrid search + LLM-as-judge eval | Eval report shows faithfulness score, CI fails on bad answers |

### Phase 2: Serving + Agents (Weeks 4–6)

| Week | Focus | Deliverable | Gate |
|------|-------|-------------|------|
| 4 | API + Deployment | FastAPI serving your RAG model, deployed on Render/Vercel | Public URL answers questions |
| 5 | Observability | Dashboard showing latency + cost per request | Numbers visible, not guesses |
| 6 | Agents | LangGraph agent with tool use + memory | Agent can call a tool and remember context |

### Phase 3: Open Source + Signature (Weeks 7–9)

| Week | Focus | Deliverable | Gate |
|------|-------|-------------|------|
| 7 | OSS contribution #1 | Find issue, write fix, open PR | PR is open (not necessarily merged) |
| 8 | OSS contribution #2 | Second PR, docs or bug fix | PR under review or merged |
| 9 | OSS contribution #3 | Third contribution | 3 total contributions tracked |

### Phase 4: Ship + Apply (Weeks 10–12)

| Week | Focus | Deliverable | Gate |
|------|-------|-------------|------|
| 10 | Signature Project A | Live demo + eval report + architecture diagram | Recruiter can click and use it |
| 11 | Signature Project B | Second live demo + cost breakdown | You can explain $/query |
| 12 | Job Ready | Portfolio site + resume + 10 applications sent | Applications submitted |

---

## Hard Rules

1. **No skipping gates.** If Week 1 gate fails, repeat Week 1. Do not move forward.
2. **One thing at a time.** Finish the current week's deliverable before starting next week.
3. **Push to GitHub every Sunday.** Even if it's broken, push it.
4. **Write down what you learned.** Every week, add to `LEARNING_LOG.md`.

---

## Files in This Repo

```
ai-mentor/
├── ROADMAP.md        ← This file (the plan)
├── PROGRESS.md       ← Your daily log (update every coding day)
├── LEARNING_LOG.md   ← Weekly deep-dive notes
├── EVAL_REPORTS/     ← JSON files from every evaluation
├── projects/
│   ├── iris/         ← Week 1
│   ├── docuchat/     ← Weeks 2–3
│   ├── nexus/        ← Weeks 4–6
│   └── signature/    ← Weeks 7–12
└── WEEK1.md          ← Day-by-day tasks for Week 1
```

---

## Your First Action

**This Sunday:** Run the 60-minute ritual (see `WEEK1.md`). Then start **Week 1, Day 1**.

**You do not need to know ML yet.** You will learn by doing Week 1 tasks.
