# SHUD 求解器加速实施路线（决策文档）

> 围绕“计算速度”目标的实施路线、依据、适配性与风险评估

- 版本：v1.0（先行路线稿，不含代码级改造清单）
- 编写日期：2026-04-23

## 说明

说明：本文有意停在“路线决策”层，不下沉到文件、函数或 patch 级实施清单；其目的是先把方向、优先级和风险边界说清楚，再进入后续的改造设计。

## 1. 文档目的与边界

本文只讨论 SHUD 求解器加速 的路线决策，不展开到文件级或函数级改造清单。它回答五个问题：先做什么、为什么这么排、理论和工程依据是什么、是否适合 SHUD、以及这样做会带来什么新的风险。文中的判断基于当前公开 SHUD 仓库、当前源码实现和官方 SUNDIALS 文档，而不是基于一个假设中的未来版本。[R1][R3][R4][R5][R6][R11]

本文把“加速”界定为：在不破坏 SHUD 水量守恒、过程一致性和主观可解释性的前提下，降低单位模拟时段的 wall-clock time，并且让不同网格规模下的性能行为更可预测。换句话说，目标不是单纯追求某一次 benchmark 的高倍数加速，而是把 SHUD 从“能跑”提升到“可度量、可比较、可持续优化”的求解器后端。

## 2. 当前 SHUD 求解器基线：为什么它既有潜力，也有隐患

SHUD 当前的总体架构非常适合做后端加速。它已经是一个 fully coupled、基于有限体积法的综合水文模型，核心求解由 CVODE 驱动，并支持 OpenMP；这意味着它天然具备向更高质量数值求解与线性代数后端升级的接口基础。[R1][R3]

但从当前源码看，SHUD 在“先做什么”这个问题上不能直接跳到更激进的并行或更换算法，因为现在还存在一些会污染性能判断的实现差异。最典型的是：串行与 OpenMP 路径并不完全等价。串行路径包含 lake 分支和 ET 通量计算，而当前 OpenMP 通量路径中没有对应的 lake 通量分支，也看不到与串行路径一致的 ET 通量处理；此外，river 状态导数在串行与 OpenMP 路径中的公式也不同。这意味着如果不先做一致性归并，任何加速结果都可能混入“方程没完全一样”的误差。[R4][R5]

第二个隐患是 forcing 读取与状态更新开销。当前 TimeSeriesData::read_csv() 在每次补充队列时都会重新打开 forcing 文件，并从头跳过前面已经读取过的行；同时 getX() 直接返回当前记录值，没有时间插值。再加上 updateforcing() 里已经对元素做了一次 updateElement，而 f_loop() 里又在入渗/补给前再调用一次 updateElement，热点路径里存在重复更新和 I/O 型低效。[R6][R4]

第三个隐患是求解控制粒度仍偏粗。当前 Control_Data 暴露的是标量 abstol/reltol，而 SHUD 的状态向量同时含 surface、unsaturated、groundwater、river 乃至 lake，不同状态的数量级、噪声水平和阈值敏感性并不相同。SUNDIALS 官方文档明确指出，当不同分量尺度不同，vector absolute tolerances 更合适，否则会造成过度求解或错误的误差分配。[R7][R13]

## 3. 总体路线：先“修基线”，再“抠热点”，最后“动线性代数”

我对 SHUD 求解器加速的推荐路线是三层推进。第一层是正确性与可观测性层：先把串行/OpenMP 行为统一、把基准测试和守恒检查搭起来；第二层是低风险热点优化层：解决 forcing/I/O、重复状态更新、输出写盘和内存访问问题；第三层是高收益数值后端层：针对 CVODE 的线性求解、预条件和稀疏结构做系统优化。[R4][R5][R6][R11][R12]

这样排序的原因不是保守，而是为了防止你在一个不干净的基线上做错误优化。对 SHUD 这种 fully coupled 模型来说，最常见的失败模式并不是“没有更先进的算法”，而是“在没有统一方程路径、没有记录线性求解统计、没有控制数值误差分配之前就开始堆并行和换求解器”，结果最终既解释不了速度变化，也解释不了精度漂移。

## 4. 第一阶段：建立可加速的基线，而不是先追求倍数

这一阶段的目标不是提速本身，而是确保之后所有加速都站在同一个物理问题和同一个数值问题上。核心任务包括：统一串行/OpenMP 的物理路径；建立一组固定 benchmark 流域；建立水量守恒、状态非负、线程数一致性和结果容差对比的自动检查；输出 CVODE 的步数、RHS 调用次数、线性求解次数、收敛失败次数等统计信息。[R4][R5][R11]

为什么这一步一定要放在第一位？因为从当前源码可见，串行和 OpenMP 路径在 lake/ET/river DY 上已有差异。如果此时直接去比较“开线程”和“不开线程”的 wall-clock time，得到的并不是同一数值问题上的加速比，而更像是两个不同实现的混合比较。[R4][R5]

这一步的方法依据来自两个层面。第一，工程依据来自当前 SHUD 源码本身；第二，数值依据来自 CVODE 的官方用法，即求解器允许并且鼓励用户通过可选输出理解内部步进和线性求解行为，而不是只盯着最终运行时间。[R4][R5][R11]

它适合 SHUD 的原因在于：SHUD 已经采用了统一的 global ODE + CVODE 驱动框架，添加运行统计和路径统一不会改变模型结构，只是把现有结构变得“可审计”。这类工作短期收益看起来不如换线性求解器那么显眼，但它实际上决定了后面所有速度结论能不能成立。

主要新风险是：你会在这一步投入时间，却暂时看不到显著加速；另外，在统一路径时可能暴露出先前被线程差异掩盖的老问题，例如 lake 质量守恒、负水位或 event timing 的变化。这个风险不是坏事，反而说明这一步在清理真实技术债。

## 5. 第二阶段：低风险热点优化——优先处理 forcing、重复更新、输出与内存访问

在 SHUD 当前实现里，最值得先动的热点不是“更复杂的数学”，而是 forcing 数据访问、重复状态更新、输出 I/O 和热数组布局。这是因为这些部分位于每一步时间推进都会经过的路径上，而且它们的改造对物理方程几乎是透明的。[R6][R4]

第一条具体路线是 forcing 读取改造。当前 read_csv() 每次补队列时重新打开文件、跳过历史行，再把新数据塞入 ring buffer；这会让长时序、细时间步模拟在 I/O 与字符串解析上反复付费。[R6] 因此更合理的方向是把 forcing 访问改成持久句柄或一次性块读取缓存；如果内存允许，可对高频 forcing 做分块常驻；如果不允许，则至少改成顺序流式读取，避免每次从文件头重扫。

第二条具体路线是去除热点路径里的重复状态更新。当前 updateforcing() 已经对每个 element 执行 updateElement，然后 f_loop() 在入渗/补给前又再次 updateElement。[R4] 这说明 SHUD 至少存在一次“为了保险而重复刷新派生参数”的设计。对 fully coupled 模型来说，这类冗余很常见，但它也意味着可以通过重构派生量的更新时间点，减少热路径中的重复计算。

第三条具体路线是输出与 flush 行为控制。SHUD 本身就带有较丰富的输出项和不同 dt 的 print control。[R7] 对求解器加速而言，输出频率、二进制/ASCII 选择、校准模式下的输出裁剪，都会直接影响 wall-clock time。这里的原则应当是：把“求解速度”与“诊断输出完整性”解耦。也就是说，默认 benchmark 运行只保留必要摘要输出，完整诊断则用于少量对照实验。

第四条具体路线是内存访问与数组布局梳理。SHUD 的元素、河道、湖泊以及大量 q/y 数组混合存在，当前热点循环以 element 为中心展开。[R4][R5] 这非常适合进一步做 cache-aware 重排：把真正频繁访问的状态和通量做连续布局，减少对象内部跳转；把线程写入冲突高的数组改为 thread-local accumulation 再归并；把循环顺序与数据访问顺序对齐。这里不需要一开始就做大规模面向对象重构，先把热路径中最常访问的数组整理好，通常收益就很稳定。

这一阶段为什么适合 SHUD？因为 SHUD 的主要耦合仍是局部邻接关系，element/river 计算呈明显 stencil-like 特征，forcing 与状态更新又高度重复。与其一开始就换整个数值方案，不如先把这些“对数学零侵入、对速度高影响”的部分清干净。

主要新风险有三个：一是 forcing 缓存会增加内存占用；**二是去除重复更新时可能改变某些派生参数的刷新时序，进而影响边界情况；三是 thread-local accumulation 虽然更快，但会带来并行归并顺序差异，从而使逐位复现实验更困难**。（影响精度）对策是把这些优化始终绑在 benchmark + mass balance + tolerance regression 的框架里，而不是单独推进。

## 6. 第三阶段：数值后端优化——把 SHUD 从“调用 CVODE”升级到“用好 CVODE”

当第一、二阶段完成后，SHUD 才进入真正意义上的数值后端优化阶段。这里的重点不是把 CVODE 换掉，而是让 SHUD 根据问题规模和耦合特征，更合理地使用 CVODE 的线性求解与误差控制能力。[R11][R12][R13]

第一条路线是 线性求解器分层选择。SUNDIALS 官方文档明确指出：对 very large stiff ODE systems，Krylov 方法通常优于直接法，且预条件通常对效率至关重要；在 Krylov 系列中，官方总体推荐 GMRES 作为最好的一般选择。[R11] 同时，KLU 适用于稀疏线性系统，并且能在稀疏模式不变的情况下复用 symbolic factorization 和 refactor 逻辑。[R12] 这与 SHUD 的局部邻接稀疏结构高度契合。因此，合理的策略不是“只押一个后端”，而是：中等规模网格优先评估 KLU；大规模网格优先评估 GMRES/FGMRES + 预条件器。

第二条路线是 按 SHUD 物理结构做预条件器，而不是只用黑箱默认值。SHUD 的状态天然分为 surface、unsaturated、groundwater、river、lake 几个物理块。[R10] 这意味着最有希望的预条件思路不是纯数学块，而是“按物理块近似耦合强、块间保留主导项”的 block preconditioner。这样做的原因很简单：如果你完全忽略 surface–subsurface–river–lake 间的结构，就很难把预条件器做到既便宜又有效；如果你直接做全 Jacobian 精确因子分解，成本又会迅速上升。

第三条路线是 误差控制精细化。虽然本文主轴是速度，但在 fully coupled 水文模型里，误差控制从来是速度和精度的共同杠杆。当前 SHUD 暴露的是标量 abstol/reltol。[R7] SUNDIALS 官方文档明确建议，当不同分量的量级和噪声水平不同，绝对容差应采用向量形式。[R13] 对 SHUD 来说，surface ponding、soil moisture、groundwater head、river stage、lake stage 的有效噪声水平显然不同。如果还用一个标量绝对容差，常见结果要么是过度约束某些小量分量，导致步长过小；要么是把某些关键阈值过程算得太粗。

第四条路线是 事件定位而不是阈值碰撞。CVODE 提供 rootfinding 接口，可在积分过程中定位用户定义函数的根。[R14] 对 SHUD 这种充满阈值切换的模型，像 surface-to-groundwater 切换、frozen/unfrozen regime 切换、bank overflow 或 lake outlet 条件切换，如果继续完全依赖固定时间步或靠误差控制碰运气，既会拖慢速度，也容易在事件附近放大局部误差。因此，事件根检测不是锦上添花，而是让“复杂过程 + 自适应时间步”更加稳健的工具。

为什么这些方法适合 SHUD？因为 SHUD 不是一个简单经验模型，而是一个全耦合、局部稀疏、状态分组明显的 ODE 系统。SUNDIALS 已经是它的既有求解框架，所以最自然的路径是深挖 CVODE 能力，而不是贸然迁移到完全不同的积分器生态。

这一阶段的新风险主要是三类。第一，KLU 或稀疏 Jacobian 路线会要求更明确地管理稀疏模式与 Jacobian 更新时机，工程复杂度明显上升。第二，预条件器如果做得过重，setup time 反而会吃掉 Krylov 迭代节省的时间。第三，向量容差与事件根检测会改变时间步分配方式，因此既可能加速，也可能在某些极端情形下降速。也正因为如此，这一阶段必须依赖第一阶段建立起来的基准和统计，而不能凭主观感觉判断。

## 7. 为什么我不建议把下列方向放在最前面

不建议把 GPU 放在第一优先级。 SHUD 当前公开实现仍以 CPU + OpenMP + CVODE 为主。[R1][R3] 在串行/OpenMP 行为尚未归一、稀疏线性代数路径尚未理顺之前，GPU 化会把数据搬运、稀疏格式管理和可重复性问题一起放大。它并不是永远不值得做，而是不适合现在做。

不建议把“重新设计积分器”放在第一优先级。 例如换成 IMEX、分裂积分甚至完全自研时间推进。这类路线理论上可能有收益，但工程代价和验证代价都很高。对于已经用上 CVODE 的 SHUD，更合理的顺序是先把现有求解框架的潜力吃透，再判断是否有必要更换框架。[R3][R11]

不建议一开始就做大规模对象架构重写。 SHUD 当然存在数据布局和模块边界需要整理的地方，但如果一上来就大改类结构，几乎一定会同时引入性能、正确性和版本管理三重风险。更稳妥的方式是：先围绕热路径做局部结构收缩，等 benchmark 和 regression 都成熟以后，再考虑更大规模的架构改造。

## 8. 风险总表：这条加速路线会带来什么新的问题

第一类风险是 “快了，但不是同一个模型了”。这主要来自串行/OpenMP 路径不一致、去除冗余时改变刷新时序、以及事件处理方式变化。它的控制方法不是口头解释，而是对固定流域做基准回归：水量平衡、关键状态轨迹、流量过程线、线程数敏感性一起看。[R4][R5][R6]

第二类风险是 “数值更快，但更脆弱”。这主要来自预条件器、KLU 稀疏模式管理、向量容差和事件根检测。解决思路是让每一类设置都能被 benchmark 驱动选择，而不是硬编码为唯一默认。

第三类风险是 “局部热点变快，但整体收益不大”。这在科学计算中很常见。比如 forcing I/O 很慢，但某个 benchmark 的瓶颈可能其实在 nonlinear solve；或者某个网格 KLU 很强，另一个网格则 setup time 太重。所以这条路线必须允许多后端并存，而不是追求一个对所有问题都完美的单解。

第四类风险是 “可重复性变差”。线程归并顺序、稀疏因子分解顺序和事件定位都会使位级结果更难重现。这不必然等于错误，但你需要事先把“逐位一致”与“工程容差内一致”区分开，否则团队后面会在验收口径上不断争论。

## 9. 里程碑与决策门槛

如果只从路线设计角度给出 go/no-go 门槛，我建议这样定义。第一，只有在串行/OpenMP benchmark 的物理结果已达到既定容差，并且 CVODE 统计与 wall-clock time 已被统一采集后，才进入更深的 solver-layer 优化。第二，只有在 forcing/I/O/重复更新这些低风险热点都处理过后，才值得把大量精力投入稀疏 Jacobian 与预条件器。第三，只有在 CPU 侧后端已形成稳定收益曲线后，才考虑更重型的 GPU 或积分器迁移路线。

这个顺序并不炫技，但它更适合 SHUD 现在的现实状态：既有成熟的物理框架和 CVODE 基础，也有明显的工程型技术债。如果路线排序错了，后面极容易出现“做了很多高难度工作，却解释不了速度为何变好或变坏”的局面。

## 10. 结论：SHUD 求解器加速的主攻点应该是什么

如果只用一句话概括，我会把 SHUD 求解器加速的主攻点定义为：先把 SHUD 变成一个可审计、可基准、可比较的 CVODE 后端，再围绕 forcing/I/O、重复状态更新、稀疏线性代数和物理块预条件器做分层优化。

这条路线之所以最适合 SHUD，是因为它既承认 SHUD 已经有一个值得深挖的 fully coupled + CVODE 基础，也正视当前公开实现里仍然存在的路径差异和工程低效。它的目标不是追求最激进的技术标签，而是先把速度工作建立在可靠、可复验、不会反噬物理一致性的地基上。[R1][R3][R4][R5][R6][R11][R12]

## 参考依据

文中使用 [R1]—[Rn] 的方式标识依据。以下列出本文件涉及的主要公开来源。

- [R1] SHUD README（GitHub 仓库说明）：当前 SHUD 将自身定义为 fully coupled、基于有限体积法的多过程多尺度水文模型，语言为 C/C++，依赖 SUNDIALS/CVODE 6.0+，支持 OpenMP。来源：https://raw.githubusercontent.com/SHUD-system/shud/master/README.md

- [R3] SHUD 当前主求解入口 shud.cpp：默认 global_implicit_mode=1，主循环为 updateforcing → ET → CVode；同时保留 uncoupled 路径。来源：https://raw.githubusercontent.com/SHUD-system/shud/master/src/Model/shud.cpp

- [R4] SHUD 当前串行通量/状态实现 MD_f.cpp：串行路径包含 lake 分支、ET 通量和当前 river DY 公式。来源：https://raw.githubusercontent.com/SHUD-system/shud/master/src/ModelData/MD_f.cpp

- [R5] SHUD 当前 OpenMP 通量/状态实现 MD_f_omp.cpp：OpenMP 路径与串行路径在 lake/ET 处理和 river DY 公式上存在显式差异。来源：https://raw.githubusercontent.com/SHUD-system/shud/master/src/ModelData/MD_f_omp.cpp

- [R6] SHUD 当前 forcing 读取实现 TimeSeriesData.cpp：read_csv() 每次补队列都会重新打开文件并跳过前面行；getX() 直接返回当前队列值，未做时间插值。来源：https://raw.githubusercontent.com/SHUD-system/shud/master/src/classes/TimeSeriesData.cpp

- [R7] SHUD 当前求解控制参数 Model_Control.hpp：当前接口暴露的是标量 abstol/reltol，同时包含 SolverStep、ETStep、cryosphere 等控制项。来源：https://raw.githubusercontent.com/SHUD-system/shud/master/src/classes/Model_Control.hpp

- [R11] SUNDIALS CVODE 官方文档：对 very large stiff ODE systems，Krylov 方法通常优于直接法；在 Krylov 方法中，官方总体推荐 GMRES，并强调预条件对于效率至关重要。来源：https://sundials.readthedocs.io/en/v7.1.1/cvode/Introduction_link.html

- [R12] SUNDIALS KLU 官方文档：SUNLinSol_KLU 针对稀疏线性系统，支持 symbolic factorization、numeric factorization 与 refactor，并可在相同稀疏模式下复用信息以减少开销。来源：https://sundials.readthedocs.io/en/latest/sunlinsol/SUNLinSol_links.html

- [R13] SUNDIALS CVODE 使用文档：当状态分量尺度和噪声水平不同，vector absolute tolerances 比单一标量绝对容差更合适；同时该文档专门讨论了非物理负值和误差控制。来源：https://sundials.readthedocs.io/en/v6.1.0/cvode/Usage/

- [R14] SUNDIALS CVODE 使用文档：CVodeRootInit 支持在积分过程中对用户定义的事件根进行检测，可用于阈值类物理过程的事件定位。来源：https://sundials.readthedocs.io/en/v6.1.0/cvode/Usage/

- [R17] Bui et al., 2020, Geosciences：综述指出，针对大尺度流域，简化的一维热传导或 Stefan 类冻融方案有规模优势；但若优先追求冻土区 subsurface hydrology 精度，则需要更强的热–水–相变表述，因为缺少热容量、土层分层和三相变化会影响 frost depth、storage 与 routing 的准确性。来源：https://www.mdpi.com/2076-3263/10/10/401

## 附注：如何使用这份文档

这份文档适合在正式动手改 SHUD 求解器前，用作团队内部对齐材料。建议先基于它确认三件事：一是是否接受“先修基线，再追求倍数”的排序；二是是否接受 CVODE 线性求解器与预条件器作为中期主战场；三是是否接受 GPU/重积分器迁移不作为当前第一优先级。
