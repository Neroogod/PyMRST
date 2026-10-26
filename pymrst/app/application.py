"""Composition root for the PyMRST application backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .database.models import AppDatabase
from .services.notification_service import NotificationService
from .services.project_service import ProjectService
from .services.task_service import TaskService
from .services.state_service import ApplicationStateService
from .services.model_service import ModelInspectionService
from .services.patch_service import InputPatchService
from .services.grid_service import GridInspectionService
from .services.run_service import RunService
from .workers.process import IsolatedRunManager
from .services.result_service import ResultService
from .services.construction_service import ModelConstructionService
from .services.experiment_service import ExperimentService
from .services.agent_service import AgentInvestigationService
from .services.provider_service import ReasoningProviderService
from .services.backup_service import WorkspaceBackupService
from .services.security_service import SecurityService
from .services.installation_service import EnvironmentQualificationService
from .services.workload_service import WorkloadQualificationService
from .services.example_service import ExampleCatalogService
from .workers.local import BackgroundTaskManager


@dataclass(slots=True)
class ApplicationContext:
    root: Path
    database: AppDatabase
    projects: ProjectService
    tasks: TaskService
    notifications: NotificationService
    background: BackgroundTaskManager
    state: ApplicationStateService
    models: ModelInspectionService
    patches: InputPatchService
    grids: GridInspectionService
    runs: RunService
    isolated: IsolatedRunManager
    results: ResultService
    construction: ModelConstructionService
    experiments: ExperimentService
    providers: ReasoningProviderService
    investigations: AgentInvestigationService
    backups: WorkspaceBackupService
    security: SecurityService
    environment: EnvironmentQualificationService
    workloads: WorkloadQualificationService
    examples: ExampleCatalogService

    def close(self) -> None:
        self.workloads.shutdown(wait=True)
        self.backups.shutdown(wait=True)
        self.isolated.shutdown(wait=True)
        self.background.shutdown(wait=True)
        self.database.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def create_application(
    root: str | Path, *, background_workers: int = 2,
) -> ApplicationContext:
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    database = AppDatabase(root / ".pymrst-app" / "metadata.sqlite")
    notifications = NotificationService(database)
    projects = ProjectService(database, root)
    tasks = TaskService(database, notifications)
    state = ApplicationStateService(database)
    models = ModelInspectionService(database, projects)
    patches = InputPatchService(database, projects, models)
    grids = GridInspectionService(database, projects)
    runs = RunService(database, projects, tasks)
    results = ResultService(database, projects, runs)
    construction = ModelConstructionService(database, projects, grids)
    experiments = ExperimentService(database, projects, runs)
    providers = ReasoningProviderService(database)
    investigations = AgentInvestigationService(database, projects, results, providers)
    backups = WorkspaceBackupService(database, projects, root)
    security = SecurityService(database)
    environment = EnvironmentQualificationService(database, root)
    workloads = WorkloadQualificationService(database, root)
    examples = ExampleCatalogService()
    backups.recover_interrupted()
    backups.start_scheduler()
    runs.recover_interrupted()
    tasks.recover_interrupted()
    background = BackgroundTaskManager(tasks, max_workers=background_workers)
    isolated = IsolatedRunManager(runs, tasks)
    isolated.register(
        "protocol-qualification",
        "pymrst.app.workers.protocol_jobs:qualification_probe",
    )
    isolated.register(
        "pymrst-spe1",
        "pymrst.app.workers.simulation_jobs:run_spe1",
        requires_model=True,
    )
    isolated.register(
        "pymrst-ensemble",
        "pymrst.app.workers.ensemble_jobs:run_experiment_ensemble",
    )
    isolated.register(
        "pymrst-branch",
        "pymrst.app.workers.ensemble_jobs:run_experiment_branch",
    )
    isolated.register(
        "pymrst-example",
        "pymrst.app.workers.example_entry:run_example",
    )
    isolated.register(
        "pymrst-construction-run",
        "pymrst.app.workers.construction_jobs:run_constructed_model",
    )
    return ApplicationContext(
        root=root, database=database, projects=projects, tasks=tasks,
        notifications=notifications, background=background, state=state,
        models=models, patches=patches, grids=grids,
        runs=runs, isolated=isolated,
        results=results,
        construction=construction,
        experiments=experiments,
        providers=providers,
        investigations=investigations,
        backups=backups,
        security=security,
        environment=environment,
        workloads=workloads,
        examples=examples,
    )


__all__ = ["ApplicationContext", "create_application"]
