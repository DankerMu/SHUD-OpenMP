# qinyijiang/nanlin.rivqdown.dat 回归 audit（PR #196 关闭后调查）

**日期**：2026-06-21
**触发**：PR #196（#178 S5d.1）关闭理由"qinyijiang bitwise gate 失败"
**结论**：无真实回归。SHA 波动源于 orchestrator 侧并发 `shud` 进程对 `output/nanlin.out/` 的写入污染，不是源码回归。

---

## 1. 背景

PR #196（#178 S5d.1）在 review/verify 阶段观察到 `qinyijiang/nanlin.rivqdown.dat` 的 SHA-256 在多次 run 之间波动（fce9675b / a4226f3 / 953cf9d1 / 937a931e / a3383c19…），与 B1a-tag golden `48036c5e57680f970c3de53e2bea97cfe4572d7e92d6ef5c828c116a86dfbc57` 不一致。当时定性为"pre-existing baseline regression"，并依此关闭 PR #196。

User 指令："暂停 #178，先 audit pre-existing qinyijiang regression"。

## 2. Audit 方法

- 严格单进程：`for i in 1 2 3; do rm -rf output && ../../shud nanlin; SHA=$(shasum -a 256 output/nanlin.out/nanlin.rivqdown.dat); done`
- 同一 case：qinyijiang nanlin（3155 element，90 天截断）
- 三个 SHUD HEAD 顺次测：
  - `0b3998d`（B1a-tag SHUD，frozen baseline 参照）
  - `d82d36e`（baseline/B1b 当前 SHUD HEAD，post-#175 S5c-C polish）
  - `d21ee34`（PR #196 S5d.1，关闭时所测 HEAD）
- 每个 HEAD：`make clean && make shud` 重建，再 3 次连续单进程跑

## 3. 结果

B1a-tag golden（`git show B1a-tag:benchmarks/qinyijiang/B0_output/nanlin.rivqdown.dat | shasum -a 256`）：

```
48036c5e57680f970c3de53e2bea97cfe4572d7e92d6ef5c828c116a86dfbc57
```

| SHUD HEAD | Run 1 | Run 2 | Run 3 | vs golden |
|---|---|---|---|---|
| `0b3998d` (B1a-tag) | `48036c5e…` | `48036c5e…` | `48036c5e…` | ✅ |
| `d82d36e` (baseline/B1b) | `48036c5e…` | `48036c5e…` | `48036c5e…` | ✅ |
| `d21ee34` (PR #196 S5d.1) | `48036c5e…` | `48036c5e…` | `48036c5e…` | ✅ |

三个 SHUD HEAD 单进程跑都收敛到 `48036c5e57680f970c3de53e2bea97cfe4572d7e92d6ef5c828c116a86dfbc57`，与 B1a-tag golden 严格 bitwise 一致。**S5d.1 (d21ee34) 在 ElementHotData SoA 引入后仍保持 bitwise neutrality**。

## 4. 根因

之前 session 在 PR #196 review/verify 时为提速调用了多个 `run_in_background: true` 的 `shud nanlin` 进程，它们**对同一 `output/nanlin.out/` 目录并发写**。SHA 取自最后一个进程 partial 写完的状态，多次 run 自然 SHA 不同——这是 orchestrator-side process contamination，**不是源码非确定性**。

证据：
- 严格单进程 3-run，三个 SHUD HEAD 全部 PASS
- B1a-tag SHUD `0b3998d` 长期 frozen 且 master_plan §A2 已证 bitwise + 3-run；这次复测再次确认
- `output/nanlin.out/` 路径是 case-relative，并发覆盖必然撞车

## 5. 影响

- **baseline/B1b 健康**：post-#175 S5c-C 没有引入 qinyijiang 回归。后续 PR base 不受影响。
- **PR #196 关闭性质**：理由（"pre-existing baseline regression"）**不成立**。`d21ee34` 3-run 全部 bitwise 匹配 golden，S5d.1 SoA 改造未破 neutrality。**关闭决定错误，将 reopen**。

## 6. Orchestration 教训

在 `tools/` 或 `.github/workflows/` 加序列化护栏：
- 同一 case 的 `output/` 目录：禁止并发 shud 进程写入
- subagent-workflow review/verify 阶段对 bitwise 测试**强制串行**，不允许 `run_in_background: true` shud
- 任何 bitwise 失败先做 process-leak check（`pgrep -f "shud "`）再启动 audit

待补：把这条约定写进 `.claude/skills/subagent-workflow/SKILL.md`（或对应 phase 4.5 verifier 提示）。

## 7. 下一步

- `gh pr reopen 196`，并 comment 链接本 audit
- 继续 #178 Phase 7+8 closeout（Gap Sweep + merge）
- 在 `.claude/skills/subagent-workflow/SKILL.md` 增补 bitwise 测试前 `pgrep -f shud` 护栏
- 后续 review/verify 阶段 bitwise 测试**严格串行**，禁用并发后台 shud
