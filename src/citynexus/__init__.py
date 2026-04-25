"""CITYNEXUS — self-evolving multi-agent urban simulation."""

# Engine
from citynexus.city.grid import Cell, Grid
from citynexus.city.zones import Zone
from citynexus.env.core import CityNexusEnv, EnvConfig, StepInfo
from citynexus.env.events import Accident, Roadblock, Severity, Weather
from citynexus.env.world_state import WorldState

# Entities
from citynexus.entities.delivery import Delivery, DeliveryStatus
from citynexus.entities.incident import Incident, IncidentKind
from citynexus.entities.unit import ResponderUnit, UnitKind, UnitStatus

# Agents
from citynexus.agents import (
    AgentContext,
    AgentRole,
    BaseAgent,
    CoordinatorConfig,
    DeliveryAgent,
    EmergencyAgent,
    MultiAgentCoordinator,
    PlannerAgent,
    PoliceAgent,
    TrafficAgent,
)

# Training pipeline + LLM-planner glue
from citynexus.training import (
    MODES,
    CallablePolicy,
    ComparisonResult,
    EvalConfig,
    EvalResult,
    Evaluator,
    HeuristicPolicy,
    LLMPlannerPolicy,
    MetricsLogger,
    Policy,
    PolicyBundle,
    PromptSample,
    Trajectory,
    TrainingConfig,
    TrainingPipeline,
    TrainingSummary,
    Transition,
    build_dataset,
    expert_distribution,
    expert_mode,
    grpo_reward,
    obs_to_prompt,
    run_llm_episode,
)

# Persistent memory
from citynexus.memory import (
    HighRiskZone,
    MemoryKind,
    MemoryRecord,
    MemoryStore,
    MemoryWriter,
    PastFailure,
    SuccessfulStrategy,
    WriterConfig,
)

# Multi-agent reward system (rich: per-agent + global city score + process-aware)
from citynexus.rewards import (
    CityScore,
    GatingMode,
    MultiAgentRewardSystem,
    PerAgentReward,
    RewardBreakdown,
    RewardSystemConfig,
)

# Verification (rules) + simple gated reward
from citynexus.verify import (
    Check,
    CheckResult,
    CheckStatus,
    GatedReward,
    GatedRewardResult,
    RewardCalculator,
    RewardComponents,
    VerificationContext,
    VerificationReport,
    Verifier,
    default_programmatic_checks,
    default_semantic_checks,
    default_system_state_checks,
)

# Scenarios
from citynexus.scenarios import (
    AdversarialGenerator,
    BlockedRoutes,
    Curriculum,
    EmergencyCluster,
    EpisodeMetrics,
    EpisodeRunner,
    EpisodeRunnerConfig,
    FailureMode,
    Scenario,
    Shock,
    TrafficSpike,
    WeatherStorm,
    apply_shock,
)

__all__ = [
    # Engine
    "CityNexusEnv", "EnvConfig", "StepInfo", "WorldState",
    "Weather", "Accident", "Roadblock", "Severity",
    "Zone", "Grid", "Cell",
    # Entities
    "Delivery", "DeliveryStatus",
    "ResponderUnit", "UnitKind", "UnitStatus",
    "Incident", "IncidentKind",
    # Agents
    "AgentRole", "AgentContext", "BaseAgent",
    "MultiAgentCoordinator", "CoordinatorConfig",
    "DeliveryAgent", "TrafficAgent", "EmergencyAgent", "PoliceAgent", "PlannerAgent",
    # Scenarios
    "Scenario", "Shock", "TrafficSpike", "EmergencyCluster", "BlockedRoutes", "WeatherStorm",
    "AdversarialGenerator", "Curriculum",
    "EpisodeRunner", "EpisodeRunnerConfig",
    "EpisodeMetrics", "FailureMode",
    "apply_shock",
    # Verification + rewards
    "Check", "CheckResult", "CheckStatus",
    "VerificationContext", "VerificationReport", "Verifier",
    "default_programmatic_checks", "default_system_state_checks", "default_semantic_checks",
    "RewardComponents", "RewardCalculator", "GatedReward", "GatedRewardResult",
    # Multi-agent reward system
    "MultiAgentRewardSystem", "RewardSystemConfig",
    "PerAgentReward", "CityScore", "RewardBreakdown", "GatingMode",
    # Persistent memory
    "MemoryStore", "MemoryWriter", "WriterConfig",
    "MemoryRecord", "MemoryKind",
    "PastFailure", "SuccessfulStrategy", "HighRiskZone",
    # Training pipeline
    "TrainingConfig", "EvalConfig",
    "TrainingPipeline", "TrainingSummary",
    "Evaluator", "EvalResult", "ComparisonResult",
    "Policy", "HeuristicPolicy", "CallablePolicy", "PolicyBundle",
    "Transition", "Trajectory",
    "MetricsLogger",
    # LLM planner (GRPO target + verifiable reward + inference)
    "MODES", "expert_mode", "obs_to_prompt", "grpo_reward",
    "PromptSample", "build_dataset", "expert_distribution",
    "LLMPlannerPolicy", "run_llm_episode",
]
