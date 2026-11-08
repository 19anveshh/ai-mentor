# 24-Week AI Engineering Mentorship Roadmap

## Structure
6 phases × 4 weeks. Each phase ends in a hard gate: working system + eval report + cost/latency numbers + runbook before advancing.

| Phase | Weeks | Focus | Gate Deliverable |
|-------|-------|-------|------------------|
| 1 — Foundation | 1–4 | Python, ML fundamentals, Git/GitHub, Linux | Working local ML pipeline + eval baseline |
| 2 — Serving | 5–8 | FastAPI, model serving, Docker, basic infra | Deployed model API with latency/cost log |
| 3 — Fine-tuning | 9–12 | PEFT/LoRA, QLoRA, dataset curation, eval | Fine-tuned model + eval report vs baseline |
| 4 — Infra | 13–16 | K8s, observability, chaos testing, cost tracking | K8s deployment + chaos test runbook |
| 5 — Signature Projects | 17–20 | 2 end-to-end AI systems (RAG + agent) | 2 shipped projects with eval reports |
| 6 — Job Ready | 21–24 | 5 OSS PRs, resume, interview prep, portfolio | Portfolio live + PRs merged |

## Weekly Ritual (Sunday, 60 min)
Review → Plan → Learn → Commit, with calendar blocking.

## Tracking Artifacts
- `ROADMAP.md` — this plan
- `PROGRESS.md` — current phase/week, status, blockers, next actions
- `LEARNING_LOG.md` — weekly entries (what worked, what didn't, judgments made)
- `EVAL_REPORTS/` — one eval report per phase gate
- `ARCHITECTURE.md` — system design decisions
- `COST_LOG.md` — cost + latency numbers per deployment

## Differentiation Checklist ("Beast Mode")
- [ ] Quantized model deployment (INT4/INT8)
- [ ] Fine-tuned model with eval vs baseline
- [ ] K8s deployment with observability
- [ ] Chaos test runbook executed
- [ ] 2 signature projects shipped
- [ ] 5 OSS PRs merged
- [ ] Cost/latency numbers for every deployment

## Mentor Protocol
Coached on judgment, not syntax. Questions I'll ask: "Show me the eval report," "Where's the bottleneck?", "What's the cost per inference?"

## Escalation Rules
- Stuck 2 weeks → scope change
- Burnout → mandatory reset
- Scope creep → backlog freeze
