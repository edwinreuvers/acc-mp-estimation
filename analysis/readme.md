---
title: "Analysis"
---

On these pages, you can find the complete analysis accompanying our manuscript. The analysis is divided into three parts, corresponding to the three main sections in the Results of our manuscript.

## 1. Parameter accuracy

*Section: 'Evaluating parameter value estimation accuracy – simulated data'*

[This page](param-acc.ipynb) contains the analyses used to assess how accurately muscle–tendon complex parameters can be retrieved from simulated experiments that mimic quick-release, step-ramp, and isometric experiments performed on real muscle.

Using a Hill-type muscle–tendon complex model with known parameter values, we evaluated how well both the traditional estimation method (which does not account for muscle fibre shortening during quick-releases) and the improved estimation method (which accounts for muscle fibre shortening during quick-releases) recover the underlying muscle property values.

## 2. Sensitivity analysis

*Section: 'Sensitivity analysis'*

[This page](sens-analysis.ipynb) contains the Monte Carlo simulation to examine the robustness of the estimated muscle properties and the analysis used to examine the interdependency of the estimated muscle properties. .

Using Monte Carlo simulations with systematic perturbations of the experimental data, we quantified how variations in the data influence the estimated parameters. In the second analysis, we systematically varied individual muscle property values to assess the sensitivity of the estimation procedure and to quantify dependencies among the estimated muscle properties.

## 3. Model predictions

*Section: 'Evaluating model predictions – in situ data'*

[This page](model-pred.ipynb) contains the analyses used to estimate muscle properties from *in situ* data of rat *m. gastrocnemius medialis*.

Using the estimated parameters, we predicted series elastic element (SEE) force. Model predictions were evaluated against quick-release, step-ramp, and isometric experiments, as well as independently measured stretch–shortening cycles.