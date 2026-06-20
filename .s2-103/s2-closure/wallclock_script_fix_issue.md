## 背景

#127 P7-Gates Wallclock 实测 qinyijiang 跑出 `0.01s` INVALID — 实际不是真跑, 是脚本 case-name vs mesh-prefix 解耦缺失:

- `tools/server_validation/run_p7_wallclock.sbatch` (服务器部署版 `.p7-wallclock-validation/run_p7_wallclock_deployed.sbatch`) 用 case-name (`qinyijiang`) 直接当 mesh-prefix 调 `./shud_omp qinyijiang`
- 但 qinyijiang case 实际 mesh 文件命名是 `nanlin.cfg.para` (在 `Basins/qinyijiang/input/nanlin/`); `input/qinyijiang/qinyijiang.cfg.para` **不存在**
- shud 找不到 cfg.para 立即报 `Fatal Error: input/qinyijiang/qinyijiang.cfg.para is in use or does not exist! EXIT 12 (FILEIO)`, 但 `time -p` 报 0.01s real
- ⇒ 8453 wallclock_qinyijiang.txt 是无效数据

详见 [#127 评论](https://github.com/DankerMu/SHUD-OpenMP/issues/127#issuecomment-4757292279) Profile 章节 + `.s2-103/p7-acceptance/wallclock_qinyijiang.txt` 0.01s 异常数据。

## 范围

### 主要修改

`tools/server_validation/run_p7_wallclock.sbatch` 加 case → mesh 映射表:

```bash
# 当前 (隐式 case == mesh):
for case in qinyijiang heihe_x4 heihe; do
    cd "$PROJECT_ROOT/SHUD/Basins/$case"
    ./shud_omp "$case"   # ← qinyijiang 时找不到 input/qinyijiang/qinyijiang.cfg.para
done

# 改后 (显式 case → mesh map):
declare -A CASE_TO_MESH=(
    [qinyijiang]=nanlin
    [heihe]=heihe
    [heihe_x4]=heihe_x4
)
for case in "${!CASE_TO_MESH[@]}"; do
    mesh="${CASE_TO_MESH[$case]}"
    cd "$PROJECT_ROOT/SHUD/Basins/$case"
    ./shud_omp "$mesh"   # ← qinyijiang 时调 ./shud_omp nanlin
done
```

### 复测

- 修后重跑 qinyijiang wallclock 3-rep serial + 3-rep omp8 = 6 runs, 写入 wallclock_qinyijiang.txt
- 比较 vs profile 数据 (qinyijiang NumEle=3155 显著小于 heihe_x4 40046)
- 验证 qinyijiang 速度比与 heihe_x4 是否一致 (如 ~1.0x 同样无 scaling, 印证 memory-bandwidth bound 不只是 heihe_x4 独有)

### 配套校验

- A3b 脚本 (`.p7-a3b-validation/run_a3b_ulp.sbatch`) 用了正确 mesh prefix (e.g., `[qinyijiang]="nanlin:3155"` map), 与本修复无冲突
- 验证其它 case 仍跑正确 (heihe 和 heihe_x4 由于 case-name == mesh-name 不受影响)

## 验收 gate

- qinyijiang wallclock 数据有效 (serial median in 100s-800s range, 不是 0.01s)
- qinyijiang ≥ 1.5x 是否达到 (如果未达, fold 入 #133 SoA 优化 scope)
- heihe + heihe_x4 wallclock 结果不变 (回归测试)

## 不在范围

- qinyijiang ≥ 1.5x speedup 本身 — 取决于 NumEle=3155 的 RHS scaling, 若同 heihe_x4 一样 bandwidth bound 则需要 #133 的 SoA 优化才能达到
- 其它 case 加入 wallclock 范围 (keliya / xinanjiang_upstream / qhh 不在 §1.1.1 量化目标内)

## 影响 / 阻塞 / 依赖

- 阻塞: 补 qinyijiang wallclock 有效数据 + 印证 / 反驳 "heihe_x4 bandwidth bound 是个例" 假设
- 不阻塞 #133 实施 (heihe_x4 优化独立), 但 #133 验收时一并跑 qinyijiang 比 #133 单独跑更高效

## 工作量

- 脚本改动: ~30 min
- 服务器重跑 qinyijiang: ~30-40 min (3-rep serial + 3-rep omp8, qinyijiang 单 run ~8 min based on 8452 数据)
- 总计 ~1-1.5 hour wall-clock + ~1h coder time

## Labels

- `s2-strict`
- `acceptance-gates`
- `priority:p1`
- `runs-on:server`
