from app.components.base import ComponentDefinition, Port

_DEFINITIONS = [
    ComponentDefinition('dataset.generate', '1.0.0', 'Generate Dataset', 'Generate a reproducible synthetic classification dataset.', outputs=(Port('dataset','TabularDataset'),), status='implemented'),
    ComponentDefinition('data.validate', '1.0.0', 'Validate Data', 'Validate schema, target and missing values.', inputs=(Port('dataset','TabularDataset'),), outputs=(Port('validated_dataset','TabularDataset'),), status='implemented'),
    ComponentDefinition('data.split', '1.0.0', 'Split Dataset', 'Create stratified training and held-out test partitions.', inputs=(Port('dataset','TabularDataset'),), outputs=(Port('train','DatasetPartition'),Port('test','DatasetPartition')), status='implemented'),
    ComponentDefinition('model.train', '1.0.0', 'Train Baseline Model', 'Fit a scikit-learn random forest classifier.', inputs=(Port('train','DatasetPartition'),), outputs=(Port('model','TrainedModel'),), status='implemented'),
    ComponentDefinition('model.tune', '1.0.0', 'Tune Hyperparameters', 'Run cross-validated grid search on training data only.', inputs=(Port('train','DatasetPartition'),), outputs=(Port('best_model','TrainedModel'),), status='implemented'),
    ComponentDefinition('model.evaluate', '1.0.0', 'Evaluate Model', 'Evaluate on the held-out test partition.', inputs=(Port('model','TrainedModel'),Port('test','DatasetPartition')), outputs=(Port('metrics','MetricCollection'),), status='implemented'),
    ComponentDefinition('model.explain', '1.0.0', 'Explain Model', 'Produce model feature importance.', inputs=(Port('model','TrainedModel'),), outputs=(Port('explanation','ExplanationReport'),), status='planned'),
    ComponentDefinition('llm.interpret', '1.0.0', 'LLM Insight Report', 'Turn measured results into grounded natural-language interpretation.', inputs=(Port('evidence','EvaluationReport'),), outputs=(Port('report','TextReport'),), status='planned'),
    ComponentDefinition('quality.gate', '1.0.0', 'Quality Gate', 'Compare a measured metric against a threshold.', inputs=(Port('metrics','MetricCollection'),), outputs=(Port('decision','QualityDecision'),), status='implemented'),
]

def list_components():
    return [{
        'type': c.type, 'version': c.version, 'name': c.name, 'description': c.description,
        'inputs': [p.__dict__ for p in c.inputs], 'outputs': [p.__dict__ for p in c.outputs],
        'config_schema': c.config_schema, 'status': c.status
    } for c in _DEFINITIONS]
