.PHONY: all a_import b_prepare c_fit d_evaluate e_urn f_plot g_nonergodic h_gate_ablation i_retention

all: f_plot

a_import:
	cd a_import/src && jupyter nbconvert --execute --to notebook a_import.ipynb --output a_import_executed.ipynb --output-dir ../output --ExecutePreprocessor.timeout=-1

b_prepare: a_import
	cd b_prepare/src && jupyter nbconvert --execute --to notebook b_prepare.ipynb --output b_prepare_executed.ipynb --output-dir ../output --ExecutePreprocessor.timeout=-1

c_fit: b_prepare
	cd c_fit/src && jupyter nbconvert --execute --to notebook c_fit.ipynb --output c_fit_executed.ipynb --output-dir ../output --ExecutePreprocessor.timeout=-1

d_evaluate: c_fit
	cd d_evaluate/src && jupyter nbconvert --execute --to notebook d_evaluate.ipynb --output d_evaluate_executed.ipynb --output-dir ../output --ExecutePreprocessor.timeout=-1

e_urn: d_evaluate
	cd e_urn/src && jupyter nbconvert --execute --to notebook e_urn.ipynb --output e_urn_executed.ipynb --output-dir ../output --ExecutePreprocessor.timeout=-1

f_plot: e_urn
	cd f_plot/src && jupyter nbconvert --execute --to notebook f_plot.ipynb --output f_plot_executed.ipynb --output-dir ../output --ExecutePreprocessor.timeout=-1

# Independent synthetic experiment; does not rerun the E001 pipeline.
g_nonergodic:
	cd g_nonergodic/src && jupyter nbconvert --execute --to notebook g_nonergodic.ipynb --output g_nonergodic_executed.ipynb --output-dir ../output --ExecutePreprocessor.timeout=-1

# Reads the saved g_nonergodic checkpoints; no fitting or E001 task is run.
h_gate_ablation:
	cd h_gate_ablation/src && jupyter nbconvert --execute --to notebook h_gate_ablation.ipynb --output h_gate_ablation_executed.ipynb --output-dir ../output --ExecutePreprocessor.timeout=-1

# Controlled retention study; prior task checkpoints are not retrained.
i_retention:
	cd i_retention/src && jupyter nbconvert --execute --to notebook i_retention.ipynb --output i_retention_executed.ipynb --output-dir ../output --ExecutePreprocessor.timeout=-1
