from sklearn.pipeline import make_pipeline, Pipeline

from src.evaluation import run_test
from src.features import baseline_pipeline_steps

run_test(Pipeline(baseline_pipeline_steps),"Whole root classifier")
