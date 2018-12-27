This project shows how to use models to recommend optimal decisions, rather than predict outcomes.

The workflow is
1) Collect a static dataset from the data generating environment
2) Write a rich (potentially multi-equation model) dynamic model of the business problem, informed by domain expertise. This model typically looks more like a probabilistic programming or structural equation model, and less like a conventional Machine Learning model.
3) Identify components of the model that can be approximated with machine learning. Estimate the machine learning model.
4) Plug the machine learning (aka predictive) model into the structural model. Treat this as a simulation environment (as done in [Recurrent World Models for Policy Evolution](https://worldmodels.github.io/)). Optimize a policy in this virtual environment
5) Apply the optimized policy in the real data generating environment.

The notebook in this directory gives a more complete explanation and applies this workflow in an example.
