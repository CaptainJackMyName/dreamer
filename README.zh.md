设计一个支持多种开源视频生成大模型的统一推理引擎，从架构层面需要解决几个核心问题：多模型架构的统一抽象、以及离线推理与在线API服务的双模式支持。

---

## 一、核心设计挑战

在动手设计架构之前，首先需要认清视频生成推理与LLM文本推理之间的关键差异：

1. **非自回归生成范式**：LLM采用逐token自回归（AR）生成，而视频生成主流采用扩散Transformer（DiT）架构，通过多步去噪完成并行生成。这决定了调度器和内存管理策略与LLM截然不同。

2. **异构模型流水线**：单个视频生成请求往往涉及多个异构模型组件的协作——文本编码器（T5）、图像编码器（CLIP/ViT）、VAE编解码器、以及核心DiT去噪模型。推理引擎需要像乐队指挥一样，精准编排这些组件的执行顺序和数据流转。

3. **显存压力巨大**：视频数据的时空维度（B×T×H×W×C）使显存需求远超文本，单卡往往难以完整加载所有组件。

4. **并行策略多样**：视频生成天然适合序列并行（Sequence Parallel）、张量并行（TP）、数据并行（DP）等多种并行模式的组合，如何在它们之间灵活切换是个难题。


## 二、总体架构设计

建议将引擎划分为**七层架构**，每层承担独立职责并通过标准化接口交互：

```
┌─────────────────────────────────────────────────────────────┐
│                    用户接口层 (EntryPoints)                    │
│   ┌─────────────────────┐  ┌──────────────────────────────┐  │
│   │  LLM 类 (离线推理)    │  │  OpenAI兼容API Server (在线)  │  │
│   └─────────┬───────────┘  └──────────────┬───────────────┘  │
├─────────────┼──────────────────────────────┼─────────────────┤
│             ▼                              ▼                  │
│                    引擎调度层 (Orchestrator)                   │
│         请求路由 · 流水线编排 · 多阶段协同 · 负载均衡           │
├─────────────────────────────────────────────────────────────┤
│                    模型抽象层 (Model Abstraction)              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐ │
│  │TextEncoder│  │ImageEnc. │  │ DiT Core │  │VAE Decoder   │ │
│  │Interface  │  │Interface │  │Interface │  │Interface     │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                    组件执行层 (Pipeline Executor)              │
│       PipelineUnit链式执行 · 条件分支 · 数据上下文传递         │
├─────────────────────────────────────────────────────────────┤
│                    智能调度层 (Scheduler)                     │
│     动态批处理 · 去噪步调度 · Cache-DiT管理 · 显存分页         │
├─────────────────────────────────────────────────────────────┤
│                    分布式执行层 (Distributed Executor)         │
│   TP/SP/DP策略 · 序列分片 · 并行折叠 · 节点间通信(ZMQ/NCCL)   │
├─────────────────────────────────────────────────────────────┤
│                    硬件加速层 (Acceleration Backend)           │
│  CUDA Kernel · 量化推理(FP16/INT8) · 算子融合 · FlashAttention│
└─────────────────────────────────────────────────────────────┘
```

### 各层职责详解

**用户接口层**：提供统一的Python API（`VideoGenLLM`类）用于离线推理，以及基于FastAPI的OpenAI兼容HTTP服务器。离线模式下直接返回生成的视频tensor或保存到文件；在线模式下通过REST API接收参数、流式返回生成进度，最终返回视频URL或Base64数据。

**引擎调度层（Orchestrator）** ：系统的“大脑”，负责将一个视频生成请求拆解为多个阶段任务：文本编码 → 噪声初始化 → 多步DiT去噪 → VAE解码。借鉴vLLM-Omni的OmniStage抽象思想，将编码器、AR推理、DiT生成定义为可解耦的阶段，由Orchestrator统一调度。

**模型抽象层**：定义统一的模型接口，屏蔽不同开源模型（Open-Sora、WanVideo、ModelScope等）的实现差异。每个模型组件（TextEncoder、ImageEncoder、DiT、VAE）都通过抽象基类定义标准接口（`encode()`、`decode()`、`denoise_step()`等），新增模型只需实现这些接口即可接入引擎。

**组件执行层（Pipeline Executor）** ：借鉴DiffSynth-Studio的PipelineUnit设计，将视频生成流程分解为一系列可组合的处理单元，按序执行并通过共享上下文字典传递中间数据。

**智能调度层**：针对扩散模型的特殊调度需求——支持动态批处理（不同请求的去噪步数可能不同）、Cache-DiT缓存加速、以及类PagedAttention的显存分页管理，避免显存碎片化。

**分布式执行层**：支持多种并行策略的灵活组合。例如，对文本编码器使用模型并行，对DiT使用序列并行（SP），对不同请求使用数据并行。借鉴SGLang-Diffusion的Token级分片和并行折叠技术，将文本编码器与DiT的并行组解耦，避免显存浪费。

**硬件加速层**：集成CUDA Kernel融合、FP16/BF16混合精度推理、FlashAttention等底层优化，确保单卡推理效率最大化。


## 三、离线推理模式设计

离线推理模式面向开发调试和批量生成场景，核心接口设计如下：

```python
from videogen_engine import VideoGenLLM, GenerationConfig

# 初始化引擎（支持同时加载多个模型）
engine = VideoGenLLM(
    model="open-sora/open-sora-2.0",  # 主模型标识
    components={
        "text_encoder": "t5-xxl",
        "vae": "stabilityai/sd-vae",
    },
    precision="fp16",
    device_map="auto",               # 自动分配组件到多卡
    gpu_memory_utilization=0.9,
)

# 单条生成
config = GenerationConfig(
    prompt="A cat walking on the beach at sunset",
    num_frames=81,
    width=1280,
    height=720,
    num_inference_steps=50,
    seed=42,
)

video_tensor = engine.generate(config)
engine.save_video(video_tensor, "output.mp4")

# 批量生成
configs = [GenerationConfig(prompt=p) for p in prompts]
videos = engine.generate_batch(configs, batch_size=4)
```

离线模式的关键设计要点：
- `VideoGenLLM`类内部持有`AsyncEngineCore`实例，与在线模式共享同一套推理核心；
- 通过`device_map="auto"`实现多组件在多GPU上的自动分布；
- `generate_batch`内部自动进行动态批处理，合并相同去噪步数的请求。


## 四、在线API服务设计

在线模式采用FastAPI构建OpenAI兼容的REST API服务器，同时扩展视频生成特有的端点：

```python
# 服务启动命令
# videogen serve open-sora/open-sora-2.0 --host 0.0.0.0 --port 8000

# --- API 端点设计 ---

# 1. 视频生成（核心端点）
POST /v1/generation/text2video
{
    "model": "open-sora-2.0",
    "prompt": "A cat walking on the beach",
    "num_frames": 81,
    "width": 1280,
    "height": 720,
    "num_inference_steps": 50,
    "seed": 42,
    "response_format": "url"  // 或 "base64"
}

# 2. 图生视频
POST /v1/generation/image2video
{
    "model": "open-sora-2.0",
    "image": "<base64_or_url>",
    "prompt": "Make this image come alive",
    "num_frames": 81,
    ...
}

# 3. 查询任务状态（异步模式）
GET /v1/query/video/{task_id}

# 4. 模型列表
GET /v1/models

# 5. 健康检查
GET /health
```

### 服务端架构

借鉴vLLM V1的多进程架构，API Server与Engine Core通过ZMQ以多对多拓扑通信，使任何API Server都能将请求路由到任何Engine Core：

```
                    ┌──────────────────┐
    HTTP Request ──▶│  FastAPI Server 1 │──┐
                    └──────────────────┘  │
                    ┌──────────────────┐  │    ZMQ (多对多)
    HTTP Request ──▶│  FastAPI Server 2 │──┼──────────────────┐
                    └──────────────────┘  │                  │
                                         │    ┌─────────────▼──────────────┐
                                         └───▶│     Engine Core Process     │
                                              │  ┌───────────────────────┐ │
                                              │  │    Orchestrator       │ │
                                              │  │  ┌─────┐ ┌────┐ ┌──┐ │ │
                                              │  │  │T5   │→│DiT │→│VAE│ │ │
                                              │  │  └─────┘ └────┘ └──┘ │ │
                                              │  └───────────────────────┘ │
                                              └────────────────────────────┘
```

关键实现细节：
- API Server使用多CPU线程进行媒体加载（参考vLLM的`VLLM_MEDIA_LOADING_THREAD_COUNT`）；
- 支持流式响应：通过Server-Sent Events（SSE）向客户端推送去噪进度（当前步/总步数）；
- 异步任务模式：对于长时间视频生成，先返回`task_id`，客户端轮询获取结果。


## 五、统一引擎核心（Engine Core）设计

Engine Core是离线与在线模式共享的推理核心，其内部通过`Orchestrator`协调多阶段执行：

```python
class VideoGenEngineCore:
    """视频生成引擎核心，离线/在线共用"""
    
    def __init__(self, model_config: ModelConfig):
        # 按模型抽象层加载各组件
        self.text_encoder = TextEncoderRegistry.load(model_config.text_encoder)
        self.image_encoder = ImageEncoderRegistry.load(model_config.image_encoder)
        self.dit_model = DiTRegistry.load(model_config.dit_model)
        self.vae_decoder = VAERegistry.load(model_config.vae)
        
        # 分布式配置
        self.parallel_config = ParallelConfig(model_config)
        
        # 调度器
        self.scheduler = DiffusionScheduler(
            cache_config=CacheDiTConfig(),
            memory_config=PagedMemoryConfig(),
        )
    
    def generate(self, request: GenerationRequest) -> torch.Tensor:
        """同步生成（离线模式使用）"""
        ...
    
    async def generate_async(self, request: GenerationRequest) -> AsyncResult:
        """异步生成（在线模式使用）"""
        ...
    
    def _execute_pipeline(self, context: PipelineContext):
        """执行标准视频生成流水线"""
        # Stage 1: 文本编码
        text_embeds = self.text_encoder.encode(context.prompt)
        
        # Stage 2: 噪声初始化
        latents = self._init_noise(context.config)
        
        # Stage 3: 多步DiT去噪（调度器管理）
        for step in self.scheduler.timesteps:
            noise_pred = self.dit_model.denoise_step(
                latents, text_embeds, step
            )
            latents = self.scheduler.step(noise_pred, step, latents)
        
        # Stage 4: VAE解码
        video_frames = self.vae_decoder.decode(latents)
        return video_frames
```


## 六、多模型支持的扩展机制

要实现“启动各种开源视频生成大模型”的目标，关键在于设计一套灵活的**模型注册与适配机制**：

```python
# 模型注册示例
@register_model("open-sora-2.0")
class OpenSoraModelAdapter(BaseModelAdapter):
    components = {
        "text_encoder": "t5-xxl",
        "dit": "STDiT-XL/2",
        "vae": "video-vae-3d",
    }
    
    def build_pipeline(self) -> List[PipelineUnit]:
        return [
            PromptEmbedUnit(),
            NoiseInitUnit(),
            STDiTDenoiseUnit(),
            VAEDecodeUnit(),
        ]

@register_model("wan-1.3b")
class WanModelAdapter(BaseModelAdapter):
    components = {
        "text_encoder": "umt5-xxl",
        "image_encoder": "xlm-roberta-clip",
        "dit": "WanModel-1.3B",
        "vae": "WanVideoVAE",
    }
    
    def build_pipeline(self) -> List[PipelineUnit]:
        # WanVideo支持文本/图像/语音多输入，流水线更复杂
        ...
```

通过`BaseModelAdapter`抽象，新增模型只需：
1. 声明组件映射关系；
2. 构建对应的PipelineUnit执行链；
3. 注册到全局模型表。


## 七、分布式与性能优化

### 并行策略自动选择

引擎应根据模型规模和GPU拓扑自动选择最优并行策略：

| 模型规模 | 单卡场景 | 多卡场景 | 推荐策略 |
|---------|---------|---------|---------|
| <3B参数 | 单卡加载全部 | 数据并行 | DP + 各卡独立生成 |
| 3B-10B参数 | 需要显存优化 | 模型并行 | T5用TP，DiT用SP |
| >10B参数 | 难以单卡运行 | 序列+模型并行 | Token级SP分片 + 流水线 |

### 视频生成特化优化

| 优化技术 | 说明 | 预期收益 |
|---------|------|---------|
| Cache-DiT | 缓存去噪中间结果，跳过已收敛步 | 推理速度提升2-5倍 |
| Token级序列分片 | 将T×H×W展平为统一序列维度再分片，避免帧级padding开销 | 通信量降低12.5% |
| 并行折叠 | 文本编码器复用DiT的SP组作为TP组，避免显存浪费 | 显存节省30%+ |
| FP16/BF16量化 | 混合精度推理 | 显存占用降低35% |

其中Token级序列分片是将视频帧的T×H×W维度展平后再分片，相比传统的帧级分片（沿T维度），可以显著减少padding开销和全对全通信量。并行折叠则是让文本编码器借用DiT序列并行组作为其张量并行组，避免每个GPU都保留一份完整的文本编码器副本。


## 八、核心实现参考与设计权衡

### 关键开源参考项目

- **vLLM-Omni**：首个正式支持全模态（含视频生成）的开源推理框架，采用解耦流水线架构和OmniStage抽象，是最接近你需求的生产级参考实现。其设计理念和代码结构可作为核心参考。
- **SGLang-Diffusion**：专注扩散模型的推理优化，在Token级序列分片、Cache-DiT集成、分布式VAE等视频生成特化优化方面有深入实践。其在并行策略上的工程优化值得借鉴。
- **DiffSynth-Studio**：通过PipelineUnit系统实现灵活的模型流水线编排，WanVideoPipeline管理9个模型组件和22个预处理单元，是多组件协作的典型范例。

### 架构设计中的关键权衡

| 设计维度 | 方案A（轻量优先） | 方案B（完备优先） |
|---------|-----------------|-----------------|
| 多模型支持 | 插件化Adapter，按需加载 | 统一模型抽象层，深度适配 |
| 并行策略 | 手动指定，支持DP/TP | 自动策略搜索，支持SP/PP混合 |
| API兼容 | 自定义REST接口 | OpenAI兼容 + 视频扩展端点 |
| 调度策略 | FIFO队列 | 优先级队列 + 动态批处理 |

### 演进路线建议

1. **Phase 1（MVP）** ：选定1-2个代表性模型（如Wan2.2），实现基础流水线 + FastAPI服务 + 单卡推理；
2. **Phase 2（扩展）** ：引入模型注册机制、PipelineUnit抽象、多模型适配器；
3. **Phase 3（生产级）** ：分布式多卡支持、Token级序列并行、Cache-DiT加速、OpenAI兼容API、流式响应。

这样的分层演进既能快速验证架构可行性，又能为后续的性能和功能扩展留有充足的架构空间。