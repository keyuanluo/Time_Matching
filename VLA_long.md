# 长程 VLA 模型的记忆机制：架构对比与跨 episode 记忆深度综述

## TL;DR
- **当前主流长程 VLA 仍停留在「episode 内记忆」阶段**：Agentic Robot 完全无记忆、依靠硬编码 SAP 协议；π0.6-MEM 用 video token + 文本 notes 实现 15 分钟级 in-episode 隐式记忆；ReAcTree、STRAP/MemER 引入显式检索但都在 episode 内或仅做训练期增强；π0.7 用 BAGEL-初始化的 14B 世界模型生成式产出 subgoal image 作为「生成式子目标条件」；AtomicVLA 用 SG-MoE 原子技能库支持持续学习但无 RAG。
- **真正具备「跨 episode、跨部署」记忆的开源 VLA 几乎不存在**：Voyager / LOTUS / BOSS 等持续学习工作的 skill library 思想没有迁移到端到端 VLA 中。主要技术阻碍是 catastrophic forgetting、embedding drift、检索分布偏移与 covariate shift 下的 brittleness——MemER (Sridhar, Pan et al., arXiv:2510.20328, Stanford 2025) 通过 ablation 实证发现"纯视觉 keyframe memory 比文本 memory 更鲁棒"，因为文本记忆在 covariate shift 下过拟合 canonical subtask ordering。
- **对于目标 2027 的「结构化 skill library + 语义 RAG + 全跨 episode + 全开源」方案**，建议站位是：(a) 采用 MemoryVLA / ReMem-VLA 风格的 dual-level（working + episodic）token 表示作为底层；(b) 借鉴 Voyager 的 code-as-skill 持久化思想但用 VLA 子策略替代代码；(c) 用 STRAP 的 sub-trajectory + DTW 检索结合 ReAcTree 的 Sentence-BERT 子目标级语义检索作为读机制；(d) 用 AtomicVLA 的 MoE 模块化扩展机制对抗灾难性遗忘。

## Key Findings

1. **「记忆」一词在 VLA 文献中至少存在五种不同含义**，必须区分：(a) 单步观测 vs 短窗口 history（per Octo Model Team, arXiv:2405.12213, 2024 官方 GitHub 文档明确："Octo was trained with a history window size of 2, meaning the model can predict an action using both the current observation and the previous observation."；OpenVLA-OFT 支持 4 帧）；(b) episode 内隐式上下文（π0.6-MEM 的 video encoder + 文本笔记）；(c) episode 内显式检索（ReAcTree, MemER）；(d) 训练期跨 episode 检索（STRAP）；(e) 部署期跨 episode 持久化技能/知识（Voyager, LOTUS, AtomicVLA 局部）。当前没有任何一个端到端 VLA 同时具备 (b)+(c)+(e)。

2. **Physical Intelligence 的 π0.5 → π0.6 → π0.7 揭示了一条清晰的演化路径**：π0.5 引入 hierarchical chain-of-thought 子任务推理；π0.6 在其上加 MEM（多尺度具身记忆，short-term video + long-term 文本笔记）；π0.7 引入 BAGEL-初始化的 14B mixture-of-transformers 世界模型，每 4 秒或 subtask 变更时生成一张 subgoal image 作为「视觉子目标」插入 prompt。三者都是 in-episode 隐式记忆，跨 episode 知识仍只通过参数权重承载。

3. **ReAcTree 是当前最干净的「episodic 记忆 + 子目标级语义 RAG」参考实现**：它把每个 LLM agent node 的成功轨迹按 `(text trajectory, Sentence-BERT 子目标 embedding, 终止状态)` 三元组存入 episodic memory，并用余弦相似度 + 5K token 预算检索；working memory 是一个共享的 Python dict，通过 `recall location of <object>` 工具调用读取。但其检索语料是 intra-benchmark 自己产生的成功轨迹，部署时不在不同 episode 间继续积累。

4. **STRAP / MemER 走 retrieval 路径，但属于两种不同范式**：STRAP 是**训练/微调时**用 sub-trajectory + DINO 嵌入 + 子序列 DTW 从大规模 prior dataset 中检索，给 few-shot policy 当训练数据增量；MemER 是**推理时**用一个 finetuned Qwen2.5-VL（3B/7B）做高层 keyframe selection + 子任务下发，由 π0.5 做低层执行。两者都属 "trajectory-level retrieval" 但作用阶段不同。

5. **AtomicVLA 是少数显式拥抱「原子技能库 + 持续学习」的端到端 VLA**：它把库实现为 Skill-Guided Mixture-of-Experts（SG-MoE），每个专家对应一个固定语义 embedding $Z_\sigma$；新增技能时冻结老专家、仅训练新专家与扩展的 router，避免灾难性遗忘。per arXiv:2603.07648，真实 Franka 机械臂实验中 long-horizon 任务提升 18.3%、continual learning 场景下提升 21%；其抗遗忘可量化为：在持续学习中 AtomicVLA 老技能成功率仅下降 1.3%，而 π0.5 baseline 在同一场景下下降 15%。但它不做 RAG / 在线检索——技能选择完全靠 VLM 在 prompt 内生成 atomic action abstraction，再由 router 直接路由到 expert。

6. **真正的跨 episode、跨部署记忆仍是开放问题**：MemER 论文 (Sridhar, Pan et al., arXiv:2510.20328, Stanford 2025) 已实证发现"纯视觉 keyframe 比文本 memory 更鲁棒"——文本记忆易"过拟合 canonical subtask ordering"并在 covariate shift 下脆裂；这印证了 embedding drift / 分布偏移是阻碍跨 episode 记忆落地的核心技术问题。

## Details

### 1. Agentic Robot (Yang et al., arXiv:2505.23450, May 2025)

- **记忆形式**：**无可学习记忆**。仅有一个 `temporal frame buffer`（sliding window）供 VLM verifier 周期性检视；没有跨步骤的可写入记忆模块。
- **写入机制**：N/A——frame buffer 是只读环形缓冲。
- **读取机制**：verifier 每隔 k 步把窗口内 third-person + wrist 帧喂给 VLM，输出 `continue / retry / recover` 三态决策。
- **与策略的接口**：完全外挂的 agentic loop——LRM planner 把高层指令切成 subgoal 序列；OpenVLA-类策略接收当前 subgoal 字符串 + 当前 RGB 输出 7-DoF 动作；verifier 在每个 subgoal 完成后由 SAP（Standardized Action Procedure）协议决定是否进入下一 subgoal 或触发 hard-coded 恢复动作。
- **跨 episode**：完全无。planner 的"skill library"实际上是**人工写死的 SAP 模板集**（pick up [object], place [object] in/on [location], turn on/off [device]……），并非可学习/可增长的库。
- **性能**：per arXiv:2505.23450，在 LIBERO 长程任务上达 79.6% 平均成功率，比 SpatialVLA 高 6.1%、比 OpenVLA 高 7.4%。
- **开源**：是。https://github.com/Agentic-Robot/agentic-robot

**架构口诀**：*"hard-coded skill list + closed-loop SAP verifier + memoryless policy"。*

---

### 2. π0.6 + MEM (Physical Intelligence, March 2026)

- **记忆形式（dual-scale, multimodal）**：
  - **短期视觉记忆**：交错时空 attention 的 ViT 视频编码器，把过去若干秒帧压缩成定长 token；高层使用更稀疏。
  - **长期语言记忆**：模型自己生成的 textual notes 链——例如 *"I placed the lid on the countertop. I placed the pot in the sink. I picked up the white bottle of milk……"*。可被模型主动 summarize（如 "I picked up the plates" 概括前若干步）。
- **写入机制**：piggyback 在 π0.5 的 chain-of-thought 子任务推理过程上——每选一次新子任务，同一推理步也"额外吐出"新文本 memory；视频 token 由滑窗自动写入。
- **读取机制**：下一步推理时，textual memory 作为 prompt 拼接，video token 作为额外 vision token 输入 VLM；模型可"动态选择"哪些事件保留、哪些总结合并，使总上下文长度可控。
- **与策略接口**：MEM 完全融合在 π0.6 的 VLM 主干内，subtask 选择频率较低，action expert 在高频率上条件于最新 subtask + memory。
- **任务时长**：最长 15 分钟（如"做一份烤奶酪三明治"、"从零清理厨房"）。
- **跨 episode**：**仍是 in-episode**。新 episode 开始时 textual memory 清空；唯一的跨 episode 知识在 π0.6 权重中（通过 RECAP 离线 RL 慢速更新——见 π*0.6 论文 arXiv:2511.14759）。
- **开源**：否。π0/π0.5 之前部分开源（openpi），但 π0.6 + MEM 权重未释出。

---

### 3. ReAcTree (Choi et al., arXiv:2511.02424, 2025)

- **记忆形式**：**纯文本，结构化双记忆**。
  - **Episodic memory $M_{ep}$**：以三元组 $(t^e, v^e, s^e)$ 存储——$t^e$ 是该 agent node 的完整文本轨迹 $(g^e, o^e_1, a^e_1, \ldots)$；$v^e = f_{sen}(g^e)$ 是子目标的 Sentence-BERT 句向量；$s^e \in \{\text{success, failure, expand}\}$ 是终止状态。
  - **Working memory**：一个 Python dict，键为对象类别、值为已观测到的实例与位置（房间、receptacle、ID）。
- **写入机制**：
  - Episodic：只保留任务**最终成功**的 run 中所有 agent node 的轨迹；可由几条手工轨迹 bootstrap，后续在训练集上自动增广。WAH-NL 每类 1 条手工轨迹；ALFRED 每类 3 条、上限 100 条。
  - Working：agent 每次观察到可移动物体时自动 update（"opened fridge, found juice → 写入 dict"）。
- **读取机制**：
  - Episodic：每个 agent node 在开始决策前，把当前子目标 $g^n$ 编码成 $v^n$，对 $M_{ep}$ 做 cosine similarity，按 top-k（受 5K token 预算约束）拼入 in-context examples；当多条相似度相同，会**在 success/failure/expand 三态间均匀采样**以提升多样性。
  - Working：通过把 `recall location of <object>` 当作一个特殊动作注入 agent 的 action space 中，agent 在需要时"调用工具"读取。
- **与策略接口**：每个 LLM agent node 的 prompt 形如 $P^n = (P_\text{sys}, P_\text{ic}^n)$，$P_{ic}^n$ 即检索得到的子目标级示范；动作采样 $a_t^n \sim p_{LLM}(\cdot | P^n, g^n, c^n_t)$。
- **跨 episode**：**部分**——episodic memory 在训练集上累积、可跨任务复用，但**仅在同一 benchmark 内**；部署到一个新房子/新 deployment 时不会继续往里写。
- **粒度**：子目标级（如 "find and pick up pudding"），明显短于 ReAct 的整任务级轨迹。
- **开源**：是。https://github.com/Choi-JaeWoo/ReAcTree。在 WAH-NL 上用 Qwen2.5 72B 达 61% GSR，几乎是 ReAct 31% 的两倍。

**架构口诀**：*"per-agent-node subgoal-level RAG (Sentence-BERT + cosine + 5K tokens) + global Python-dict working memory as tool"。*

---

### 4. STRAP & MemER (sub-trajectory / experience retrieval, 2024-2025)

#### 4a. STRAP (Memmel et al., arXiv:2412.15182, ICLR 2025)

- **记忆形式**：**离线 prior dataset** 中的 sub-trajectory 集合（自动按 chunking heuristic 切分）；每条 sub-trajectory 用预训练 vision foundation model（如 DINO）编码成 embedding 序列。
- **写入机制**：一次性，对整个 prior corpus（如 DROID 5k demos）做编码 + 切分。
- **读取机制**：在 test time 给少量 in-domain demonstration $\mathcal{D}_\text{target}$，用 **subsequence Dynamic Time Warping** 在 prior 中检索相似 sub-trajectory（容许不同长度，是其与 BR/FR/FlowRetrieval 等 baseline 的关键差异）。
- **与策略接口**：检索到的 sub-trajectory 作为**训练时增量数据**喂给 transformer-based imitation policy（如 BC-Transformer），而不是 inference time 注入。
- **跨 episode**：是 **跨任务/跨实验室级**（DROID 数据来自其他实验室），但仅作用于训练阶段；部署后不再检索。
- **开源**：是。https://weirdlabuw.github.io/strap/

#### 4b. MemER (Sridhar, Pan et al., arXiv:2510.20328, Stanford 2025)

- **记忆形式**：**视觉 keyframe 集合**——hierarchical 框架，high-level policy 主动从"经验"中选哪些过去帧作为 visual memory。
- **写入机制**：finetune Qwen2.5-VL (3B/7B 版本) 学习一个"keyframe nomination"输出；在线 streaming 时持续 nominate 关键帧到内部缓冲。
- **读取机制**：high-level policy 用所选 keyframes + 最近帧生成自然语言 subtask 字符串。
- **与策略接口**：low-level 是一个 finetuned π0.5，接收 subtask 字符串 + 当前观测产出动作；高层与低层异步、可独立训练。
- **任务时长**：分钟级 long-horizon（如"dust shelves and replace items"），用 50 条 teleop demonstrations 微调即可。
- **重要发现（论文 ablation）**：纯视觉 keyframe memory 比文本 memory 或多模态混合**更鲁棒**——文本 memory 容易"过拟合 canonical subtask ordering"并在 covariate shift 下脆裂。这是值得 user 在设计 RAG VLA 时认真考虑的反例数据点。
- **跨 episode**：**否**——keyframe 缓冲随 episode 终结清空。
- **开源**：是。https://jen-pan.github.io/memer/

---

### 5. AtomicVLA (arXiv:2603.07648, 2026)

- **记忆形式**：**Skill-Guided Mixture-of-Experts (SG-MoE) 原子技能库**。每个 expert ↔ 一个原子动作（pick / place / turn / open / ……），由固定的高维 embedding $Z_\sigma$ 索引。共享 expert（继承自预训练 VLA backbone）保留通用能力。
- **写入机制**：训练时——VLM 从机器人轨迹自动产出 "atomic skill abstraction" 标注；每条原子技能数据训练对应 expert。新技能：**只训练新 expert + 扩展 router 的新分支**（用小随机值初始化），老 expert 冻结。
- **读取机制**：**无 RAG / 无相似度检索**。当前 step 由 VLM 在 prompt 内自回归生成 "task plan → atomic skill abstraction → action" 链；router 把 abstraction 的 embedding 直接选 top-1 或 top-k expert。
- **与策略接口**：最终 action = 共享 expert 输出 + selected skill expert 输出的加权和（diffusion / flow-matching head 后融合）。
- **跨 episode / 持续学习**：是其核心卖点——per arXiv:2603.07648 实验在真实 Franka 机械臂上进行，long-horizon 任务提升 18.3%、continual learning 提升 21%；抗遗忘量化为 AtomicVLA 老技能仅下降 1.3%（vs. π0.5 baseline 同一持续学习场景下下降 15%）。LIBERO-LONG 比 π0 高 10%。
- **跨 episode 内存（部署期，长期）**：本质仍受限——library 在训练期增长，**部署期不自动增长新 expert**。
- **开源**：项目页声称将开源，但截至 2026 年 5 月权重/代码尚未公开释出（仅 arXiv preprint + project page）。

---

### 6. π0.7 (Physical Intelligence, April 16 2026, arXiv:2604.15483)

- **总参数**：约 5B = 4B Gemma3 VLM 主干（含 400M ViT vision encoder）+ MEM-style 视频历史 encoder + 860M flow-matching action expert。
- **世界模型（"generated retrieval" 的本质）**：
  - 14B mixture-of-transformers，从 **BAGEL**（ByteDance Seed 开发的开源图像生成/编辑模型，HuggingFace: `ByteDance-Seed/BAGEL-7B-MoT`；BAGEL 自身为 "7B activated parameters / 14B total parameters" 的 MoT 架构）初始化（继承 web-scale 视觉-语言先验）。原文："We initialize from BAGEL [105], a 14B mixture-of-transformers model capable of image understanding, editing, and generation."
  - 训练目标：flow-matching CFM loss，给定 $(o_t, \hat{\ell}_t, m)$ 生成未来 subgoal 图像 $g^*_t$，ground truth 为 segment 结束时的真实图像 $g^*_t = o_{t_\text{end}}$。
  - **是纯生成、不是检索**——没有外部 corpus 索引，没有 k-NN；用户原本表述中的"generated retrieval"是一个比喻，实际架构是 conditional generative world model。
  - **触发刷新**：subtask 语义变化或距上次生成 Δ=4s（取先发生者）；异步推理（visual subgoal 与 subtask 文本分别由独立线程产出，VLA 总用最新可用版本）。
  - **延迟**：根据 CTOL Digital Solutions 二手综述，subgoal 图像生成在 4×H100 上约 1.25 秒/张；论文 Appendix A-D 应有对应数字但本次未完全验证，正式引用时建议标 "as reported by CTOL Digital Solutions" 或留待 Appendix 确认。
- **subgoal 图像如何条件入 policy**：每次最多 3 张 subgoal 图（front + 两路 wrist，不含 rear），都过同一个 vision encoder 与观测图像共享 tokenizer。采用 **block-causal mask**——观测 token 与 subgoal token 内部双向注意，subgoal token 还可单向 attend 观测 token，文本 token 因果注意。**所以是统一 token-in-VLM 而非外挂 cross-attention**。
- **训练 trick**（与 user 写 paper 高度相关）：
  - 仅 25% 的训练样本带 subgoal 图像（让模型也能脱离 subgoal 工作）。
  - 给定 batch 中带 subgoal 的样本：30% 概率 drop 文本 subtask（强迫 visual subgoal 单独承载意图）；metadata 整体 15% drop，各分量再 5% drop。
  - 真实 subgoal 时间戳采样：**0.25 概率从 segment 末帧（与世界模型 ground truth 一致），0.75 概率从未来 0-4 秒均匀采**。
  - 同时用真实未来帧 + 世界模型生成帧两类样本训练 VLA，缓解 train-test 视觉分布 mismatch。
- **Episode metadata** prompt 内容：`Speed: 8000. Quality: 5. Mistake: false. Control Mode: joint.`——speed = episode 长度（500 步一档），quality = 1-5 人工分，mistake = 该 segment 是否含错误，control mode ∈ {joint, ee}。**这是 PI 拿来融合 sub-optimal 自主数据与失败 episode 的关键 conditioning 机制**——把"数据质量"做成模型可见的离散标签，避免 naive 训练把好数据稀释成噪声。
- **跨 episode**：**仍是 in-episode**——世界模型与 VLA 的状态在 episode 间清空；跨 episode 知识仍只由权重承载，但通过更广的 prompt 与 metadata 条件能更稳健地利用过去 episode 的非完美数据。
- **开源**：否。论文未宣布权重/代码释出；blog 与 arXiv 都无开源链接，与 π0/π0.5 OpenPI 部分释出的传统不同。

---

### 7. 其它代表性工作快速一览

| 工作 | 年份 | 记忆形式 | 写 | 读 | 与策略接口 | 跨 episode |
|---|---|---|---|---|---|---|
| **RT-1 / RT-2** | 2022/23 | 6 帧 history 拼成 token 序列 | 滑窗 | 内置 transformer 注意 | tokens → action head | 否 |
| **Octo** | 2024 | history_window=2（per arXiv:2405.12213 官方文档） | 滑窗 | transformer 注意 | diffusion action head | 否 |
| **OpenVLA / OpenVLA-OFT** | 2024-25 | 单帧（OFT 支持 4 帧 history） | 滑窗 | 注意 | LLaMA-2 backbone | 否 |
| **RoboFlamingo** | 2024 | flamingo-style cross-attn 在帧序列上 | 滑窗 | cross-attn | LSTM head | 否 |
| **VIMA** | 2023 | multimodal prompt（图+文 token） | prompt 写入 | 注意 prompt | T5 backbone | prompt 跨 episode 但需手工 |
| **GR-1 / GR-2** | 2024 | autoregressive video prediction（生成式 world model） | pretraining | 隐式 latent | 同一 transformer 输出 action | 否 |
| **Helix (Figure AI)** | 2025-26 | S2 → S1 共享的 latent intent vector（per-step）| 实时 | latent 注入 S1 200Hz 控制 | 双系统：7B VLM (7-9Hz) → 80M ViT-policy | 否（无显式跨 episode）|
| **MemoryVLA** | 2025 | Perceptual-Cognitive Memory Bank：感知 token + 认知 token 双流 | working memory 触发写入 PCMB | working memory 用时间位置编码查 PCMB，gating fusion；token-merge consolidation | 7B Prismatic VLM + diffusion action expert | 否（episode 内）|
| **ReMem-VLA** | 2026 | 两组 learnable recurrent queries（frame-level + chunk-level）| recurrent BPTT | 隐式（端到端）| backbone 内 | 否 |
| **EchoVLA** | 2025 | scene memory（spatial-semantic map）+ episodic memory（multimodal episode 特征）| 双库分别独立写 | 双库分别检索，coarse + fine attention fusion | base-arm diffusion | **部分**——scene map 可跨 episode |
| **Embodied-SlotSSM** | 2025 | slot-based state-space model，object-centric persistent slots | recurrent | slot attention | LIBERO-Mem 上 SOTA | episode 内为主 |
| **Voyager** | 2023 | JavaScript 可执行 code skill library，GPT-3.5 embedding 索引 | 任务成功时 commit | 子任务时 top-5 语义相似检索 | 注入 GPT-4 prompt | **完全跨 episode、跨世界**（最强标杆）|
| **JARVIS-1** | 2023 | 多模态 memory + skill library 扩展 Voyager | 同 Voyager + 视觉 | 多模态检索 | LLM prompt | 跨 episode |
| **LOTUS** | 2023-24 | 无监督发现的 sensorimotor skill 库 + meta-controller | continual skill discovery | meta-controller 在 library 上 compose | hierarchical IL | **跨 task 终身学习**；per arXiv:2311.02058 (Wang et al., ICRA 2024)："LOTUS outperforms state-of-the-art baselines by over 11% in success rate" |
| **BOSS** | CoRL 2023 | 离线 RL skill library + LLM 引导 chain | "skill bootstrapping" 自我练习 | LLM 选下一 skill | conditional policy | 跨 task |
| **RoboCat / Gato 后裔** | DeepMind 2023 | 自生成 trajectory 进训练集做下一轮 fine-tune | self-improvement loop | embodiment-specialized spin-off | visual goal-conditioned decision transformer | 跨 task；通过权重更新而非显式检索 |
| **SayCan** | 2022 | 静态 skill set + affordance value | 训练期固定 | LLM 给每个 skill 打分，affordance 加权 | 选择 skill 触发底层 policy | 否（skill 集合静态）|
| **Code-as-Policies / ProgPrompt** | 2022 | LLM 生成 Python 调度代码 | 推理时即兴 | 推理时即兴；few-shot example pool | 解释执行 | 否 |
| **DreamerV3 / DayDreamer** | 2023 | RSSM latent world model | online RL update | imagined rollout | actor-critic | 通过权重；非显式 RAG |
| **Genie / GAIA** | 2024 | latent action 世界模型 | pretraining | 生成式 | 与 VLA 联合训练（实验室阶段）| 否（pretraining 内）|

---

### 8. 完整架构对比表（扩展 user 已有表格）

| 方法 | 记忆形式（form） | 写（write） | 读/检索（retrieve） | 与策略接口（integrate） | 跨 episode | 开源 |
|---|---|---|---|---|---|---|
| Agentic Robot | 仅 sliding frame buffer + 硬编码 SAP 模板集 | N/A | verifier VLM 每 k 步看窗 | 外挂 planner-executor-verifier 三模块 | 否 | 是 |
| π0.6 + MEM | short: 视频 token；long: 模型自写文本 notes | piggyback 在 high-level subtask 推理上 | next-step prompt 拼接 | VLM 主干内统一 token，hierarchical 子任务 | 否（episode 内 15 min）| 否 |
| ReAcTree | 文本 episodic（subgoal trajectory + Sentence-BERT 向量 + 状态）+ 共享 Python dict working memory | 仅成功任务 + 自动 observe | cosine sim + 5K token 预算 top-k；working 用 `recall location` 工具调用 | 注入每个 LLM agent node 的 prompt | 同一 benchmark 内累积，部署期不增长 | 是 |
| STRAP | DINO embedding 的 sub-trajectory 集 | 一次性 prior dataset 编码 | subsequence DTW | 增训阶段提供数据 | 训练期跨实验室 | 是 |
| MemER | 视觉 keyframes（最强 ablation：纯视觉 > 文本）| Qwen2.5-VL nominate | high-level policy 直接选 | high-level 出文本 subtask → π0.5 低层 | 否 | 是 |
| AtomicVLA | SG-MoE 原子技能库（每 expert 一个固定 $Z_\sigma$）| 训练新 expert，冻结老 expert | router 直接路由（无 RAG）| 共享 expert + 选中 expert 加权融合 | 训练期可扩，部署期不增 | 部分（论文已出，权重未释）|
| π0.7 | history video tokens + BAGEL-init 14B 生成式 world model 的 subgoal images | world model 异步生成 + 4s 刷新 | 不检索；纯生成 | subgoal images 与观测共用 vision encoder → block-causal 注意 | 否 | 否 |
| MemoryVLA | Perceptual-Cognitive Memory Bank | working memory consolidate | 时间位置编码查询 + gating fusion + token merge | 7B VLM + diffusion expert | 否 | 是 |
| ReMem-VLA | 双层 learnable recurrent queries | 端到端 BPTT | 隐式 | backbone 内 | 否 | 是 |
| Voyager | JS code skill library + 文本描述 embedding | 任务成功 commit | top-5 语义检索 | GPT-4 prompt 内 | **是（强参照）** | 是 |
| LOTUS | 无监督发现的 sensorimotor skill | continual skill discovery | meta-controller compose | hierarchical IL | **跨 task lifelong** | 是 |
| BOSS | 离线 RL skill 库 | skill bootstrapping 自练 | LLM 选下一个 skill | conditional policy | 跨 task | 是 |
| RoboCat | 模型自身做 demonstration generator | self-improvement 加回训练集 | N/A | retrain | 通过权重 | 部分 |

---

### 9. 跨 episode 记忆的重要性 —— 三个维度的论证

#### 9a. 任务能力维度：哪些任务**必须**跨 episode 记忆？

并非所有长程任务都需要跨 episode 记忆，但下列类别**在原则上无法**仅靠 in-episode 记忆解决：

1. **多日 / 多 session 任务**：用户对 π0.6-MEM blog 中明确提到的"未来 prompt"——*"我 6 点回家，请把晚饭准备好，并且周三打扫家里"*——必须跨日记忆"上次擦了哪几个房间"、"冰箱里上周买的牛奶过期了吗"。π0.6 的 textual memory 在 episode 终结即清空，无法支撑此类 chore schedule。
2. **移动操作 + 家居地图学习**：mobile manipulator 在新家部署，第一周应建立空间-语义地图，第二周开始就该"知道剪刀在第三个抽屉"。EchoVLA 是为数不多明确把 scene memory（spatial-semantic map）与 episodic memory（multimodal episode 特征）分离、并允许 scene memory 跨 deployment 持久化的工作，但 episodic memory 部分仍以单 episode 为单元。
3. **个性化 / 用户偏好沉淀**：*"主人喜欢咖啡少糖"、"周六晨跑后桌上要放香蕉"*——这些是经多次交互慢慢累积的 semantic memory，与 Tulving 的 personal semantic memory 同构。
4. **失败模式记忆**：跨 episode 失败记忆能让机器人记住"上次开冰箱门 hinge 在左侧"——π0.6 in-episode adaptation 只能在一个 session 内学，重启后失忆。
5. **稀有事件 / 安全规约**：曾经把杯子摔碎过的桌角、易滑的瓷砖——这类 episodic memory 必须长期保留。

#### 9b. 持续学习 / lifelong learning 维度

跨 episode 记忆是连接"基础模型 frozen 权重"和"开放世界部署不断遇到新情况"的桥梁。两条主流路径：

- **参数路径**（catastrophic forgetting 风险）：RoboCat self-improvement loop、π*0.6 RECAP 离线 RL 都通过把新数据加回训练集再微调来更新权重——慢、贵、容易遗忘老技能。
- **非参数路径**（user 的目标方向）：Voyager skill library / LOTUS skill discovery / AtomicVLA SG-MoE 把"持续学到的东西"放在外部记忆/模块里，避免动权重。AtomicVLA 的实验证明这条路在端到端 VLA 上技术上可行（连续学习场景下比 baseline 高 21%、老技能仅退化 1.3% 对比 π0.5 的 15%）。

将"显式 RAG over skill library"与"模块化 expert 增量"结合是 user 目标方向的合理技术押注：RAG 给灵活性、MoE 给容量隔离。

#### 9c. 认知科学 / 类人记忆维度（Tulving 体系）

Tulving 经典三分：
- **Procedural memory（程序性）**：技能、自动化运动模式。↔ VLA 权重 + AtomicVLA expert library。
- **Semantic memory（语义性）**：世界知识、概念。↔ π0.6 textual memory、ReAcTree working memory 中的 object-location dict、未来跨 episode 的 scene-semantic map。
- **Episodic memory（情节性）**：可重体验的 first-person 时空事件。↔ MemER keyframes、ReAcTree episodic trajectory、MemoryVLA PCMB 的 low-level perceptual detail。

另外两个生物机制对设计具有启发价值：
- **Hippocampal replay & consolidation**：海马在睡眠/休息期把短期 episode 重放到 neocortex，固化为长期 semantic 知识——对应工程上的"夜间从 episodic buffer 蒸馏到权重 / skill library 的离线管线"（RoboCat 的 self-improvement loop 就是粗暴近似）。
- **Complementary Learning Systems（McClelland 1995）**：海马快速学新 episode + 皮层慢速整合到 schema，避免灾难性遗忘——这正是 VLA 跨 episode 记忆最直接的生物学 blueprint，建议在 paper 中显式引用。MemoryVLA 论文本身就明确提到了 hippocampal system 启发。

#### 9d. 为什么当前 VLA 几乎都缺跨 episode 记忆？技术挑战

1. **Catastrophic forgetting**：直接把新 episode 数据微调进权重，老技能掉链。AtomicVLA 的 MoE 隔离与 Voyager 的非参数路径是两类规避方案。
2. **检索规模 (scaling retrieval)**：episode 库到几千、几万条后，简单 dense embedding 检索的 recall 下降、计算与延迟上升。MemER 论文用 finetuned VLM 学一个 keyframe-selection policy 而非纯 ANN，正是为了规避这点。
3. **Embedding drift / 分布偏移**：episodic memory 中存的 embedding 来自旧 vision encoder，新 episode 用更新过的 encoder 时空间不对齐——除非冻结 encoder 或定期 re-embed。
4. **Covariate shift 下的 brittleness**：MemER (Sridhar, Pan et al., arXiv:2510.20328, Stanford 2025) ablation 显示，文本 memory 在新场景下让策略"过拟合 canonical subtask ordering"，反而比无记忆更脆。这是 user 设计 semantic RAG 时必须正面回应的反例。
5. **Causal confusion**（π0.6 blog 中明确点名 Haan & Levine 2019 arXiv:1905.11979 的现象）：history 中夹带"我自己的动作"会让 imitation 学到 spurious correlation——"我刚才抬了手→现在该放下手"。任何把过去动作放回 context 的设计都需要回应这一问题。
6. **数据稀缺与归因**：哪些 episode 应该被记住？人类有强归因/情绪信号，机器人没有显式 reward signal 时无法判断"这条 episode 值不值得入库"。Voyager 的 self-verifier、ReAcTree 的 "只保留成功 run" 是两种简化策略，都不完美。
7. **隐私与生命周期**：长期保留视觉 episode 涉及部署环境的隐私（家庭场景尤甚），需考虑可删除/可遗忘的设计。

## Recommendations

针对 user 提到的 2027 目标 "Ours (Target) = 结构化 skill library + 语义 RAG + 全跨 episode + 全开源"，按落地难度分阶段：

**Stage 1（基础，6 个月内可立 baseline）**：
- **Backbone**：在 π0 / π0.5 OpenPI 之上加 MemoryVLA-style 双流 token（perceptual + cognitive）作为 in-episode working memory——这是 user 设计中 in-episode 记忆部分的 fastest path 且已有开源参考。
- **Episodic RAG**：照搬 ReAcTree 三元组结构 `(subgoal text, Sentence-BERT embedding, success/failure state)` + 5K token budget + 余弦相似度 top-k。**改进点**：把 success/failure 标签改成 reward-weighted（借鉴 RECAP 的 advantage conditioning），让检索器优先返回"高优势" trajectory。
- **基准**：LIBERO-Long + WAH-NL + LIBERO-Mem 同时报，证明 in-episode 和跨任务两种记忆都增益。

**Stage 2（跨 episode skill library，6-12 个月）**：
- **Skill 表示**：放弃 Voyager 的"code as skill"（VLA 不需要符号代码），改用"VLA sub-policy + Sentence-BERT 语义描述"双视图——execution view = 参数化 sub-policy，retrieval view = 文本 + DINO embedding。
- **Skill 写入策略**：成功 trajectory 自动 → ReAcTree 风格三元组；同时离线用 LOTUS-style 无监督 segmentation 在原始 demo 数据上挖再用 STRAP-DTW 去重。
- **Skill 增长 / 抗遗忘**：用 AtomicVLA 的 SG-MoE 给每个 skill 一个轻量 LoRA expert（不是整个 expert）——平衡参数膨胀与隔离。
- **检索接口**：分级——先用文本 query 召回 candidate skill 名字，再用当前观测图像与 skill 的 prototypical observation 做视觉相似度 re-rank。

**Stage 3（真跨 deployment，12-18 个月）**：
- **Scene memory 持久化**：借鉴 EchoVLA 把 spatial-semantic map 与 episodic memory 分库；map 用 ROS-style persistent storage 跨 deployment 保存。
- **Consolidation 管线**：夜间离线把 high-confidence episode 蒸馏成 (a) skill library 新增条目 或 (b) LoRA 权重更新，明确对应 hippocampal-to-neocortical consolidation。

**Benchmarks / 阈值（什么发生时调整方向）**：
- 若在 LIBERO-Mem 上 ReMem-VLA 已显著 outperform 你的 RAG 路径 → 说明对 episode 内的 implicit recurrent memory 不能完全用显式 RAG 替代，需保留 dual-system。
- 若你的 skill library 规模到 10K+ 时 retrieval 精度掉 >15% → 需引入 hierarchical 索引（先按 task type 分桶，再 dense）。
- 若 catastrophic forgetting 老技能成功率掉 >10% → 切换 LoRA-per-skill 方案，停止全参数 finetune。

**开源策略**：把你的 skill library schema、retrieval 索引、consolidation 管线和 sub-policy LoRA 都 release，与 ReAcTree / MemER / AtomicVLA 形成"四件套" lifelong VLA 工具链；这是论文的 narrative 制高点（"the first fully open-source cross-episode VLA stack"）。

## Caveats

1. **π0.7 的 14B world model 不是 "retrieval"**：尽管 user 描述为 "generated retrieval"，论文的世界模型是 conditional generative，无 corpus 索引；"generated retrieval" 是高度比喻性的表述，写论文时建议改述为 "generative subgoal conditioning" 以免被审稿人质疑。
2. **π0.7 subgoal image 1.25s/4×H100 的数字来自 CTOL Digital Solutions 二手综述**，论文 Appendix 我未完全验证，引用时需标"as reported by CTOL Digital Solutions"或 Appendix-pending。
3. **AtomicVLA 与 π0.7 的 arXiv ID（2603.07648, 2604.15483）的"26"前缀**反映 2026 年提交，但 AtomicVLA 项目页声称将开源、截至 5/23/2026 未公开权重——time-stamp 写作时建议保留模糊"2026"。
4. **ReMem-VLA, MemoryVLA, EchoVLA 都是近 6 个月的 preprint**，部分结果尚未经过会议同行评议，引用时建议留余地。
5. **Helix (Figure AI) 的细节多数为公司 blog + 二手新闻**，没有同行评议论文；论文里只能作为 industry context 而非可严格对比的 baseline。
6. **跨 episode 记忆的最强经验证据仍来自非 VLA 领域**（Voyager 在 Minecraft、LOTUS 在窄域 IL），把它们的优越性外推到通用 VLA 是研究假设而非已证事实；user 的 paper 应明确把"跨 episode 记忆是否真给端到端 VLA 带来稳定增益"作为 empirical question 来回答，而非默认结论。
7. **causal confusion** 与 covariate shift 是任何 explicit memory 设计的潜在杀手——必须有针对性的 ablation（如 MemER 已做的 visual vs text memory 对比）。
8. **没有任何当前方法同时拿下"in-episode dense memory + cross-episode skill library + open source"三项**——user 的 target slot 在文献空白中确实存在，但意味着没有现成 baseline，建议为每个组件分别选最强参照对手（in-episode: MemoryVLA / ReMem-VLA；跨 episode：Voyager-style + AtomicVLA；开源：Octo + OpenVLA + ReAcTree）。
